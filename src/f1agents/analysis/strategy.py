"""Pit loss, undercut measurement, and counterfactual pit-window simulation.

Everything here is deterministic and unit-testable. The Strategy Analyst
and Rival Watcher agents receive these outputs as structured JSON; they do
not compute anything themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stints import StintFit, build_stints, fuel_slope
from ..data.loader import clean_lap_mask


def pit_loss(race_laps: pd.DataFrame, race_pits: pd.DataFrame) -> dict:
    """Estimate total pit loss (s) at this circuit for this race.

    Pit loss = (in-lap + out-lap excess over the driver's clean median),
    aggregated over all stops with usable neighbours. Robust to SC-window
    stops via median aggregation.
    """
    losses = []
    for _, stop in race_pits.iterrows():
        d = race_laps[race_laps.driverId == stop.driverId]
        med = d[clean_lap_mask(d.lap_s)].lap_s.median()
        in_lap = d[d.lap == stop.lap].lap_s
        out_lap = d[d.lap == stop.lap + 1].lap_s
        if len(in_lap) and len(out_lap) and not np.isnan(med):
            excess = float(in_lap.iloc[0] + out_lap.iloc[0] - 2 * med)
            if 5 < excess < 60:  # discard SC-distorted stops
                losses.append(excess)
    if not losses:
        return {"pit_loss_s": None, "n_stops_used": 0}
    return {
        "pit_loss_s": round(float(np.median(losses)), 2),
        "pit_loss_iqr_s": round(float(np.subtract(*np.percentile(losses, [75, 25]))), 2),
        "n_stops_used": len(losses),
        "n_stops_total": len(race_pits),
    }


def cumulative_time(race_laps: pd.DataFrame) -> pd.DataFrame:
    df = race_laps.sort_values(["driverId", "lap"]).copy()
    df["cum_s"] = df.groupby("driverId").lap_s.cumsum()
    return df


def undercut_events(race_laps: pd.DataFrame, race_pits: pd.DataFrame,
                    window_s: float = 4.0) -> list[dict]:
    """Measure realized undercut/overcut outcomes between rival pairs.

    For each pit stop, find rivals who were within `window_s` on track at
    the in-lap and pitted on a different lap. Report the gap swing across
    the pit cycle (both drivers back on settled tyres).
    """
    cum = cumulative_time(race_laps)
    events = []
    pits_by_driver = {d: sorted(g.lap.tolist()) for d, g in race_pits.groupby("driverId")}

    for _, stop in race_pits.iterrows():
        a = stop.driverId
        lap0 = int(stop.lap)
        a_row = cum[(cum.driverId == a) & (cum.lap == lap0)]
        if a_row.empty:
            continue
        a_cum0 = float(a_row.cum_s.iloc[0])

        same_lap = cum[cum.lap == lap0]
        rivals = same_lap[(same_lap.driverId != a) & (abs(same_lap.cum_s - a_cum0) <= window_s)]
        for _, rv in rivals.iterrows():
            b = rv.driverId
            b_stops = [l for l in pits_by_driver.get(b, []) if lap0 < l <= lap0 + 5]
            if not b_stops:
                continue  # rival did not respond within 5 laps -> not a pit battle
            lap_settle = b_stops[0] + 2
            a_s = cum[(cum.driverId == a) & (cum.lap == lap_settle)]
            b_s = cum[(cum.driverId == b) & (cum.lap == lap_settle)]
            if a_s.empty or b_s.empty:
                continue
            gap_before = float(rv.cum_s - a_cum0)               # +ve: rival behind
            gap_after = float(b_s.cum_s.iloc[0] - a_s.cum_s.iloc[0])
            events.append({
                "attacker": race_laps[race_laps.driverId == a].driver.iloc[0],
                "defender": race_laps[race_laps.driverId == b].driver.iloc[0],
                "attacker_pit_lap": lap0,
                "defender_pit_lap": b_stops[0],
                "gap_before_s": round(gap_before, 2),
                "gap_after_s": round(gap_after, 2),
                "swing_s": round(gap_after - gap_before, 2),
                "worked": bool(gap_after > 0 >= gap_before or (gap_after - gap_before) > 1.0),
            })
    # deduplicate mirrored pairs, keep earliest pit
    seen, out = set(), []
    for e in sorted(events, key=lambda x: x["attacker_pit_lap"]):
        key = frozenset([e["attacker"], e["defender"]])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def counterfactual_pit(race_laps: pd.DataFrame, race_pits: pd.DataFrame,
                       fits: list[StintFit], driver_code: str,
                       shift_range: range = range(-6, 7)) -> dict:
    """Simulate total race time if one driver had pitted N laps earlier/later.

    Model: within each stint, lap time = base_pace + deg_slope * tyre_age
    + fuel_slope * race_lap. Shifting the pit boundary reassigns laps
    between the two adjacent stint models and re-prices them. Pit loss is
    held constant. Interaction effects (traffic, SC) are out of scope and
    stated as such in the agent brief.
    """
    d_fits = [f for f in fits if f.code == driver_code]
    if len(d_fits) < 2:
        return {"driver": driver_code, "viable": False, "reason": "needs >= 2 fitted stints"}

    fslope = fuel_slope(race_laps)
    d_fits = sorted(d_fits, key=lambda f: f.stint)
    s1, s2 = d_fits[0], d_fits[1]
    actual_pit = s1.end_lap

    def stint_time(fit: StintFit, laps: np.ndarray) -> float:
        ages = np.arange(len(laps))
        return float(np.sum(fit.base_pace_s + fit.deg_slope_s_per_lap * ages + fslope * laps))

    results = []
    span_end = s2.end_lap
    for shift in shift_range:
        pit_lap = actual_pit + shift
        if pit_lap <= s1.start_lap + 3 or pit_lap >= span_end - 3:
            continue
        laps1 = np.arange(s1.start_lap, pit_lap + 1)
        laps2 = np.arange(pit_lap + 1, span_end + 1)
        total = stint_time(s1, laps1) + stint_time(s2, laps2)
        results.append({"pit_lap": int(pit_lap), "shift": int(shift),
                        "modeled_time_s": round(total, 2)})

    base = next(r for r in results if r["shift"] == 0)["modeled_time_s"]
    for r in results:
        r["delta_vs_actual_s"] = round(r["modeled_time_s"] - base, 2)
    best = min(results, key=lambda r: r["modeled_time_s"])
    return {
        "driver": driver_code, "viable": True,
        "actual_pit_lap": int(actual_pit),
        "optimal_pit_lap": best["pit_lap"],
        "time_left_on_table_s": round(base - best["modeled_time_s"], 2),
        "curve": results,
        "assumptions": "fixed pit loss; no traffic/SC interaction; linear deg per stint",
    }
