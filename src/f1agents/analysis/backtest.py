"""Frozen-clock backtest: predictions from laps 1..N scored on laps N+1..end.

The protocol answers the question a strategist would ask before trusting
the system: when the models could only see the race up to lap N, how good
were their forward statements? Three prediction families are scored, and
each one is the exact computation an agent brief would carry during the
slow loop.

1. Degradation continuation. For every stint that spans the freeze lap
   with enough clean laps on both sides, a Theil-Sen fit on pre-freeze
   laps predicts the fuel-corrected pace of every post-freeze clean lap
   in that stint. Scored as MAE and signed bias in seconds.

2. Undercut swing. For pit battles where the attacker stopped at least
   two laps before the freeze and the defender had not yet responded,
   the predicted swing is built only from pre-freeze information: the
   defender's fitted old-tyre pace extrapolated over the exposure laps
   against the attacker's observed fresh-tyre pace. Scored against the
   realized swing measured on the full race (sign hit rate and absolute
   error in seconds).

3. Pit window. For drivers still on their opening stint at the freeze,
   the predicted optimal remaining stop lap uses the driver's pre-freeze
   stint-1 fit plus a second-stint pace model estimated only from rivals
   who had already pitted. Scored against the post-hoc optimum from the
   full-race counterfactual model, in laps and in priced seconds.

No lookahead anywhere: fuel slope, clean-lap masks, and every fit used
for prediction are computed on the pre-freeze frame alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stints import build_stints, clean_lap_mask, fit_stints


def _prefreeze(race_laps: pd.DataFrame, race_pits: pd.DataFrame, freeze: int):
    laps = race_laps[race_laps.lap <= freeze].copy()
    pits = race_pits[race_pits.lap <= freeze].copy()
    return laps, pits


def deg_continuation(race_laps, race_pits, freeze: int,
                     min_pre: int = 5, min_post: int = 3) -> list[dict]:
    pre_laps, pre_pits = _prefreeze(race_laps, race_pits, freeze)
    pre_fits, fslope = fit_stints(pre_laps, pre_pits, min_clean_laps=min_pre)

    full = build_stints(race_laps, race_pits)
    results = []
    for f in pre_fits:
        driver_full = full[(full.code == f.code) & (full.stint == f.stint)]
        post = driver_full[driver_full.lap > freeze].sort_values("lap")
        if post.empty:
            continue  # stint ended before the freeze; nothing to predict
        clean = clean_lap_mask(post.lap_s)
        post = post[clean.values]
        if len(post) > 1:
            post = post.iloc[:-1]  # drop probable in-lap
        if len(post) < min_post:
            continue

        stint_start = int(driver_full.lap.min())
        tyre_age = post.lap.values - stint_start
        corrected = post.lap_s.values - fslope * post.lap.values
        predicted = f.base_pace_s + f.deg_slope_s_per_lap * tyre_age
        err = predicted - corrected
        results.append({
            "code": f.code, "stint": f.stint,
            "fit_window": [stint_start, freeze],
            "scored_laps": [int(post.lap.min()), int(post.lap.max())],
            "n_scored": len(post),
            "n_fit_clean_laps": f.n_clean,
            "pred_slope_s_per_lap": round(f.deg_slope_s_per_lap, 4),
            "mae_s": round(float(np.mean(np.abs(err))), 3),
            "bias_s": round(float(np.mean(err)), 3),
        })
    return results


def undercut_calls(race_laps, race_pits, freeze: int, pit_loss_s: float,
                   window_s: float = 4.0, min_fresh_laps: int = 2) -> list[dict]:
    """Score conditional undercut forecasts against realized full-cycle swings.

    The realized measure (strategy.undercut_events) is the gap change from
    the attacker's in-lap to two laps after the defender's response stop.
    The forecast reproduces that same quantity using only data available
    at the freeze: the observed gap change from the attacker's in-lap to
    the freeze (which already contains the attacker's pit loss), plus a
    modeled remainder in which the defender runs its fitted old-tyre pace
    until its stop, pays the pre-freeze pit loss estimate, and both cars
    run the attacker's observed fresh pace to the settle lap. The
    defender's actual stop lap conditions the forecast; the live brief
    states the per-lap growth rate instead.
    """
    from .strategy import cumulative_time, undercut_events

    pre_laps, pre_pits = _prefreeze(race_laps, race_pits, freeze)
    pre_fits, fslope = fit_stints(pre_laps, pre_pits, min_clean_laps=4)
    fits_by = {(f.code, f.stint): f for f in pre_fits}
    pre_cum = cumulative_time(pre_laps)

    realized = undercut_events(race_laps, race_pits, window_s=window_s)
    pre_stints = build_stints(pre_laps, pre_pits)
    calls = []
    for ev in realized:
        a_pit, b_pit = ev["attacker_pit_lap"], ev["defender_pit_lap"]
        # open at the freeze: attacker committed, defender had not responded
        if not (a_pit <= freeze - min_fresh_laps and b_pit > freeze):
            continue
        a_code = race_laps[race_laps.driver == ev["attacker"]].code.iloc[0]
        b_code = race_laps[race_laps.driver == ev["defender"]].code.iloc[0]

        # observed component (all <= freeze): gap change since the in-lap
        def gap_at(lap):
            a = pre_cum[(pre_cum.code == a_code) & (pre_cum.lap == lap)]
            b = pre_cum[(pre_cum.code == b_code) & (pre_cum.lap == lap)]
            if a.empty or b.empty:
                return None
            return float(b.cum_s.iloc[0] - a.cum_s.iloc[0])

        gap0, gap_frozen = gap_at(a_pit), gap_at(freeze)
        if gap0 is None or gap_frozen is None:
            continue

        # attacker fresh pace: observed post-pit laps up to the freeze
        a_fresh = pre_stints[(pre_stints.code == a_code)
                             & (pre_stints.lap > a_pit + 1)
                             & (pre_stints.lap <= freeze)]
        if len(a_fresh) < min_fresh_laps:
            continue
        a_pace = float(np.median(a_fresh.lap_s - fslope * a_fresh.lap))

        # defender old-tyre model from the pre-freeze fit of the current stint
        b_stint = int(pre_stints[pre_stints.code == b_code].stint.max())
        bf = fits_by.get((b_code, b_stint))
        if bf is None:
            continue
        b_start = int(pre_stints[(pre_stints.code == b_code)
                                 & (pre_stints.stint == b_stint)].lap.min())

        settle = b_pit + 2
        # modeled remainder, freeze+1 .. settle (fuel term cancels pairwise)
        old_laps = np.arange(freeze + 1, b_pit + 1)
        b_old = np.sum(bf.base_pace_s + bf.deg_slope_s_per_lap * (old_laps - b_start))
        b_future = float(b_old) + pit_loss_s + a_pace * (settle - b_pit)
        a_future = a_pace * (settle - freeze)

        predicted_swing = (gap_frozen + (b_future - a_future)) - gap0
        calls.append({
            "attacker": ev["attacker"], "defender": ev["defender"],
            "attacker_pit_lap": a_pit, "defender_pit_lap": b_pit,
            "observed_by_freeze_s": round(gap_frozen - gap0, 2),
            "modeled_remainder_s": round(b_future - a_future, 2),
            "predicted_swing_s": round(predicted_swing, 2),
            "realized_swing_s": ev["swing_s"],
            "sign_correct": bool(np.sign(predicted_swing) == np.sign(ev["swing_s"])),
            "abs_error_s": round(abs(predicted_swing - ev["swing_s"]), 2),
        })
    return calls


def pit_window(race_laps, race_pits, freeze: int,
               pit_loss_s: float, horizon: int = 14) -> list[dict]:
    from .strategy import counterfactual_pit

    pre_laps, pre_pits = _prefreeze(race_laps, race_pits, freeze)
    pre_fits, fslope = fit_stints(pre_laps, pre_pits, min_clean_laps=5)
    full_fits, _ = fit_stints(race_laps, race_pits)
    race_len = int(race_laps.lap.max())

    # second-stint pace prior: pooled Theil-Sen over rivals already on
    # stint 2, first three laps of each stint dropped (tyre warmup reads
    # as false degradation in young fits), driver offsets removed.
    pre_stints = build_stints(pre_laps, pre_pits)
    s2_laps = pre_stints[pre_stints.stint == 2].copy()
    s2_laps = s2_laps[clean_lap_mask(s2_laps.lap_s).values]
    rows = []
    for code, g in s2_laps.groupby("code"):
        g = g.sort_values("lap")
        age = g.lap.values - g.lap.min()
        keep = age >= 3
        if keep.sum() < 3:
            continue
        corrected = g.lap_s.values[keep] - fslope * g.lap.values[keep]
        rows.append(pd.DataFrame({"age": age[keep],
                                  "y": corrected - np.median(corrected)}))
    if not rows:
        return []
    pooled = pd.concat(rows)
    s2_slope = float(stats.theilslopes(pooled.y.values, pooled.age.values).slope)
    s2_offsets = [f.base_pace_s for f in pre_fits if f.stint == 2]
    s1_bases = [f.base_pace_s for f in pre_fits if f.stint == 1]
    s2_offset = (float(np.median(s2_offsets)) if s2_offsets
                 else float(np.median(s1_bases)))

    out = []
    for f in (x for x in pre_fits if x.stint == 1):
        actual_stops = race_pits[race_pits.driverId.isin(
            race_laps[race_laps.code == f.code].driverId.unique())]
        first_stop = actual_stops.lap.min() if not actual_stops.empty else None
        if first_stop is None or first_stop <= freeze:
            continue  # already pitted or never pitted: no open decision

        # price each candidate stop lap with pre-freeze info only
        cands = np.arange(freeze + 1, min(freeze + horizon, race_len - 5))
        driver_s2_base = f.base_pace_s + (s2_offset - float(np.median(s1_bases)))
        costs = []
        for L in cands:
            a1 = np.arange(1, L + 1)
            a2 = np.arange(1, race_len - L + 1)
            t = (np.sum(f.base_pace_s + f.deg_slope_s_per_lap * a1)
                 + pit_loss_s
                 + np.sum(driver_s2_base + s2_slope * a2))
            costs.append(t)
        pred_opt = int(cands[int(np.argmin(costs))])
        at_boundary = pred_opt in (int(cands[0]), int(cands[-1]))

        cf = counterfactual_pit(race_laps, race_pits, full_fits, f.code)
        if not cf.get("viable"):
            continue
        post_opt = cf["optimal_pit_lap"]
        out.append({
            "code": f.code,
            "predicted_optimal_lap": pred_opt,
            "actual_pit_lap": int(first_stop),
            "posthoc_optimal_lap": post_opt,
            "prediction_at_horizon_boundary": at_boundary,
            "lap_error_vs_posthoc": abs(pred_opt - post_opt),
            "actual_left_on_table_s": cf["time_left_on_table_s"],
        })
    return out


def run_backtest(race_laps, race_pits, freeze: int, pit_loss_s: float) -> dict:
    deg = deg_continuation(race_laps, race_pits, freeze)
    uc = undercut_calls(race_laps, race_pits, freeze, pit_loss_s)
    pw = pit_window(race_laps, race_pits, freeze, pit_loss_s)

    report = {
        "protocol": "frozen-clock: fits use laps 1..N only; scoring uses laps N+1..end",
        "freeze_lap": freeze,
        "deg_continuation": deg,
        "deg_summary": {
            "n_stints": len(deg),
            "mae_s": round(float(np.mean([d["mae_s"] for d in deg])), 3) if deg else None,
            "bias_s": round(float(np.mean([d["bias_s"] for d in deg])), 3) if deg else None,
            "mature_fits_mae_s": (lambda m: round(float(np.mean(m)), 3) if m else None)(
                [d["mae_s"] for d in deg if d["n_fit_clean_laps"] >= 10]),
            "mature_fits_n": len([d for d in deg if d["n_fit_clean_laps"] >= 10]),
        },
        "undercut_calls": uc,
        "undercut_summary": {
            "n_calls": len(uc),
            "sign_hit_rate": round(float(np.mean([c["sign_correct"] for c in uc])), 2) if uc else None,
            "mean_abs_error_s": round(float(np.mean([c["abs_error_s"] for c in uc])), 2) if uc else None,
        },
        "pit_window": pw,
    }
    return report
