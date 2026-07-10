"""Stint segmentation and fuel-corrected degradation modeling.

The deterministic layer. Agents never fit these models; they consume the
fitted parameters and residuals. Every number an agent reasons about is
reproducible from this module.

Method
------
1. Stints are the lap intervals between pit stops (Ergast pit_stops table).
2. A global fuel-burn slope is estimated per race from the field-median
   lap time trend across the race distance, using only clean laps. This
   absorbs the ~0.03-0.06 s/lap gain from fuel burn-off.
3. Per stint, a robust linear fit of fuel-corrected lap time vs. tyre age
   gives: base pace (intercept) and degradation slope (s/lap). Theil-Sen
   estimation keeps traffic outliers from bending the slope.

Without compound labels this is a pace-decay proxy, not true compound
degradation. The FastF1 adapter in a live deployment attaches compounds
and the identical fit becomes compound-conditional.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from ..data.loader import clean_lap_mask


@dataclass
class StintFit:
    driver: str
    code: str
    team: str
    stint: int
    start_lap: int
    end_lap: int
    n_laps: int
    n_clean: int
    base_pace_s: float          # fuel-corrected pace at tyre age 0
    deg_slope_s_per_lap: float  # fuel-corrected pace decay
    slope_ci_low: float
    slope_ci_high: float
    median_lap_s: float
    residual_iqr_s: float       # consistency within the stint

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def fuel_slope(race_laps: pd.DataFrame) -> float:
    """Field-level fuel-burn slope (s/lap), estimated from clean-lap medians."""
    df = race_laps.copy()
    df["clean"] = df.groupby("driverId").lap_s.transform(clean_lap_mask)
    med = df[df.clean].groupby("lap").lap_s.median()
    if len(med) < 10:
        return 0.0
    slope, _, _, _, _ = stats.linregress(med.index.values, med.values)
    # Fuel effect is a monotonic gain; clamp to a physically plausible band.
    return float(np.clip(slope, -0.12, 0.0))


def build_stints(race_laps: pd.DataFrame, race_pits: pd.DataFrame) -> pd.DataFrame:
    """Assign a stint number to every lap of every driver."""
    df = race_laps.copy()
    df["stint"] = 1
    for driver_id, grp in race_pits.groupby("driverId"):
        for _, stop in grp.iterrows():
            df.loc[(df.driverId == driver_id) & (df.lap > stop.lap), "stint"] += 0
        # laps after pit on lap L belong to the next stint
        stops = sorted(grp.lap.tolist())
        for i, stop_lap in enumerate(stops, start=1):
            df.loc[(df.driverId == driver_id) & (df.lap > stop_lap), "stint"] = i + 1
    return df


def fit_stints(race_laps: pd.DataFrame, race_pits: pd.DataFrame,
               min_clean_laps: int = 6) -> tuple[list[StintFit], float]:
    """Fit degradation per stint for every driver. Returns fits + fuel slope."""
    fslope = fuel_slope(race_laps)
    df = build_stints(race_laps, race_pits)
    fits: list[StintFit] = []

    for (driver_id, stint), grp in df.groupby(["driverId", "stint"]):
        grp = grp.sort_values("lap")
        lap_s = grp.lap_s.values
        clean = clean_lap_mask(grp.lap_s).values.copy()
        # drop first lap of stint (out lap or race start) and last (often in-lap)
        if len(grp) > 2:
            clean[0] = False
            clean[-1] = False
        if clean.sum() < min_clean_laps:
            continue

        tyre_age = np.arange(len(grp))[clean]
        corrected = lap_s[clean] - fslope * grp.lap.values[clean]

        ts = stats.theilslopes(corrected, tyre_age)
        resid = corrected - (ts.intercept + ts.slope * tyre_age)

        fits.append(StintFit(
            driver=grp.driver.iloc[0], code=str(grp.code.iloc[0]),
            team=str(grp.team.iloc[0]), stint=int(stint),
            start_lap=int(grp.lap.min()), end_lap=int(grp.lap.max()),
            n_laps=len(grp), n_clean=int(clean.sum()),
            base_pace_s=float(ts.intercept),
            deg_slope_s_per_lap=float(ts.slope),
            slope_ci_low=float(ts.low_slope), slope_ci_high=float(ts.high_slope),
            median_lap_s=float(np.median(lap_s[clean])),
            residual_iqr_s=float(stats.iqr(resid)),
        ))
    return fits, fslope


def deg_anomalies(fits: list[StintFit], z_threshold: float = 1.5) -> list[dict]:
    """Stints whose degradation slope deviates from the field distribution.

    These records are the trigger payload for the Deg Explainer agent: the
    deterministic layer flags, the agent explains.
    """
    slopes = np.array([f.deg_slope_s_per_lap for f in fits])
    if len(slopes) < 5:
        return []
    med, mad = np.median(slopes), stats.median_abs_deviation(slopes)
    if mad == 0:
        return []
    out = []
    for f in fits:
        z = (f.deg_slope_s_per_lap - med) / (1.4826 * mad)
        if abs(z) >= z_threshold:
            rec = f.to_dict()
            rec["field_median_slope"] = round(float(med), 4)
            rec["robust_z"] = round(float(z), 2)
            rec["direction"] = "high_deg" if z > 0 else "low_deg"
            out.append(rec)
    return sorted(out, key=lambda r: -abs(r["robust_z"]))
