"""Driver-level metrics and cross-track profiling.

Feeds the Driver Coach and Track Historian agents. Season-scale batch
workload: on AWS this module runs under Bedrock Batch orchestration with
one invocation per (driver, season) and per (circuit, era).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stints import fit_stints
from .strategy import pit_loss
from ..data.loader import F1Data, clean_lap_mask


def driver_race_metrics(data: F1Data, race_id: int) -> pd.DataFrame:
    """Per-driver, per-race: clean pace, consistency, deg discipline, teammate delta."""
    laps = data.race_laps(race_id)
    pits = data.race_pit_stops(race_id)
    fits, _ = fit_stints(laps, pits)

    rows = []
    for code, grp in laps.groupby("code"):
        clean = grp[clean_lap_mask(grp.lap_s)]
        if len(clean) < 10:
            continue
        d_fits = [f for f in fits if f.code == code]
        rows.append({
            "code": code,
            "driver": grp.driver.iloc[0],
            "team": grp.team.iloc[0],
            "median_clean_lap_s": float(clean.lap_s.median()),
            "consistency_iqr_s": float(stats.iqr(clean.lap_s)),
            "mean_deg_slope": float(np.mean([f.deg_slope_s_per_lap for f in d_fits])) if d_fits else np.nan,
            "n_stints_fitted": len(d_fits),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # teammate delta on median clean pace
    df["teammate_delta_s"] = df.groupby("team").median_clean_lap_s.transform(
        lambda s: s - s.min() if len(s) > 1 else np.nan
    )
    return df.sort_values("median_clean_lap_s").reset_index(drop=True)


def season_driver_table(data: F1Data, year: int) -> pd.DataFrame:
    races = data.seasons([year])
    frames = []
    for _, race in races.iterrows():
        try:
            m = driver_race_metrics(data, int(race.raceId))
        except Exception:
            continue
        if m.empty:
            continue
        m["race"] = race["name"]
        m["round"] = race["round"]
        # normalize pace to field median so circuits are comparable
        field_med = m.median_clean_lap_s.median()
        m["pace_vs_field_pct"] = (m.median_clean_lap_s / field_med - 1) * 100
        frames.append(m)
    season = pd.concat(frames, ignore_index=True)
    agg = season.groupby(["code", "driver"]).agg(
        races=("race", "nunique"),
        pace_vs_field_pct=("pace_vs_field_pct", "median"),
        consistency_iqr_s=("consistency_iqr_s", "median"),
        mean_deg_slope=("mean_deg_slope", "median"),
        teammate_delta_s=("teammate_delta_s", "median"),
    ).reset_index()
    return agg[agg.races >= 8].sort_values("pace_vs_field_pct").reset_index(drop=True)


def track_profiles(data: F1Data, years: list[int]) -> pd.DataFrame:
    """Per-circuit archetype features across an era.

    Features: median deg slope, pit loss, neutralisation rate (share of
    laps > 1.3x field median: SC/VSC/rain proxy), position volatility
    (mean |grid - finish| among classified finishers).
    """
    races = data.seasons(years)
    rows = []
    for _, race in races.iterrows():
        rid = int(race.raceId)
        try:
            laps = data.race_laps(rid)
            pits = data.race_pit_stops(rid)
        except Exception:
            continue
        if laps.empty:
            continue
        fits, _ = fit_stints(laps, pits)
        pl = pit_loss(laps, pits)
        field_med = laps.lap_s.median()
        neutral = float((laps.lap_s > 1.3 * field_med).mean())
        res = data.results[(data.results.raceId == rid) & data.results.position.notna()]
        volatility = float((res.grid - res.positionOrder).abs().mean()) if len(res) else np.nan
        rows.append({
            "circuit_id": int(race.circuitId),
            "race": race["name"], "year": int(race.year),
            "median_deg_slope": float(np.median([f.deg_slope_s_per_lap for f in fits])) if fits else np.nan,
            "pit_loss_s": pl["pit_loss_s"],
            "neutralisation_rate": round(neutral, 4),
            "position_volatility": round(volatility, 2) if not np.isnan(volatility) else None,
            "stops_per_driver": round(len(pits) / max(laps.driverId.nunique(), 1), 2),
        })
    df = pd.DataFrame(rows)
    circ = data.circuits[["circuitId", "name", "country"]].rename(
        columns={"circuitId": "circuit_id", "name": "circuit"})
    prof = df.groupby("circuit_id").agg(
        races=("year", "nunique"),
        median_deg_slope=("median_deg_slope", "median"),
        pit_loss_s=("pit_loss_s", "median"),
        neutralisation_rate=("neutralisation_rate", "median"),
        position_volatility=("position_volatility", "median"),
        stops_per_driver=("stops_per_driver", "median"),
    ).reset_index().merge(circ, on="circuit_id")
    return prof.sort_values("median_deg_slope", ascending=False).reset_index(drop=True)
