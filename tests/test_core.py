"""Tests for the deterministic layer. Run: pytest tests/ -q

The agent layer is intentionally untested here: agents are prompts over
payloads, evaluated separately (see blog: groundedness eval). What must
never regress silently is the math the agents reason from.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1agents.data.loader import F1Data, clean_lap_mask
from f1agents.analysis.stints import fit_stints, fuel_slope, deg_anomalies
from f1agents.analysis.strategy import pit_loss, undercut_events, counterfactual_pit
from f1agents.compliance.rules import validate


@pytest.fixture(scope="module")
def spanish_2024():
    data = F1Data()
    rid = data.race_id(2024, "Spanish")
    return data.race_laps(rid), data.race_pit_stops(rid)


def test_clean_lap_mask_excludes_outliers():
    s = pd.Series([80.0] * 20 + [120.0, 95.0])
    m = clean_lap_mask(s)
    assert m.iloc[:20].all() and not m.iloc[20] and not m.iloc[21]


def test_fuel_slope_is_negative_and_bounded(spanish_2024):
    laps, _ = spanish_2024
    fs = fuel_slope(laps)
    assert -0.12 <= fs <= 0.0


def test_stint_fits_cover_field(spanish_2024):
    laps, pits = spanish_2024
    fits, _ = fit_stints(laps, pits)
    assert len(fits) >= 40
    slopes = [f.deg_slope_s_per_lap for f in fits]
    assert -0.1 < np.median(slopes) < 0.3  # physically plausible decay


def test_anomaly_z_scores_sorted(spanish_2024):
    laps, pits = spanish_2024
    fits, _ = fit_stints(laps, pits)
    anoms = deg_anomalies(fits)
    zs = [abs(a["robust_z"]) for a in anoms]
    assert zs == sorted(zs, reverse=True)
    assert all(z >= 1.5 for z in zs)


def test_pit_loss_plausible(spanish_2024):
    laps, pits = spanish_2024
    pl = pit_loss(laps, pits)
    assert 15 < pl["pit_loss_s"] < 35


def test_undercut_events_symmetric_dedup(spanish_2024):
    laps, pits = spanish_2024
    events = undercut_events(laps, pits)
    pairs = [frozenset([e["attacker"], e["defender"]]) for e in events]
    assert len(pairs) == len(set(pairs))


def test_counterfactual_optimum_not_worse_than_actual(spanish_2024):
    laps, pits = spanish_2024
    fits, _ = fit_stints(laps, pits)
    cf = counterfactual_pit(laps, pits, fits, "NOR")
    assert cf["viable"]
    assert cf["time_left_on_table_s"] >= 0


def test_compliance_gate_fails_closed():
    bad = {"recommended_pit_lap": 20, "race_distance_laps": 66,
           "compound_plan": ["medium", "medium"],
           "stint_lengths": [20, 46], "auto_execute": True}
    rec = validate("strategy_analyst", bad)
    failed = {v.rule_id for v in rec.verdicts if not v.passed}
    assert "SR-30.5m-two-compounds" in failed
    assert "GOV-advisory-only" in failed
    assert not rec.passed


def test_compliance_gate_passes_valid_plan():
    ok = {"recommended_pit_lap": 17, "race_distance_laps": 66,
          "compound_plan": ["soft", "medium", "soft"],
          "stint_lengths": [17, 24, 25], "auto_execute": False}
    assert validate("strategy_analyst", ok).passed


def test_audit_record_carries_hash():
    ok = {"recommended_pit_lap": 17, "race_distance_laps": 66,
          "compound_plan": ["soft", "medium"], "stint_lengths": [17, 30],
          "auto_execute": False}
    import json
    d = json.loads(validate("strategy_analyst", ok).to_json())
    assert len(d["record_hash"]) == 16 and d["ruleset_version"]


# --- OpenF1 adapter ---

def test_openf1_normalize_and_stops():
    from f1agents.data.openf1 import normalize_laps, stops_from_laps
    import csv
    fixture = Path(__file__).parent / "fixtures" / "openf1_laps_9839_44.csv"
    with open(fixture) as f:
        raw = [{"driver_number": int(r["driver_number"]),
                "lap_number": int(r["lap_number"]),
                "lap_duration": float(r["lap_duration"]),
                "is_pit_out_lap": r["is_pit_out_lap"] == "True"}
               for r in csv.DictReader(f)]
    laps = normalize_laps(raw)
    assert list(laps.columns[:3]) == ["driverId", "lap", "lap_s"]
    assert len(laps) == 58 and laps.lap_s.between(85, 115).all()
    stops = stops_from_laps(laps)
    assert sorted(stops.lap.tolist()) == [8, 31]  # real 2025 Abu Dhabi stops


# --- frozen-clock backtest ---

def _synthetic_two_stint_race(n_drivers=6, race_len=40, pit_lap=18,
                              slope1=0.08, slope2=0.05, base=90.0):
    rows, pit_rows = [], []
    rng = np.random.default_rng(7)
    for d in range(1, n_drivers + 1):
        offset = 0.1 * d
        for lap in range(1, race_len + 1):
            stint_start = 1 if lap <= pit_lap else pit_lap + 1
            slope = slope1 if lap <= pit_lap else slope2
            t = base + offset + slope * (lap - stint_start) + rng.normal(0, 0.05)
            if lap == pit_lap:
                t += 21.0
            rows.append({"driverId": d, "lap": lap, "lap_s": round(t, 3),
                         "driver": f"Driver {d}", "code": f"D{d:02d}",
                         "team": "T"})
        pit_rows.append({"driverId": d, "lap": pit_lap})
    return pd.DataFrame(rows), pd.DataFrame(pit_rows)


def test_backtest_deg_recovers_synthetic_slope():
    from f1agents.analysis.backtest import deg_continuation
    laps, pits = _synthetic_two_stint_race()
    res = deg_continuation(laps, pits, freeze=10)
    assert res, "expected scored stints"
    for r in res:
        assert abs(r["pred_slope_s_per_lap"] - 0.08) < 0.03
        assert r["mae_s"] < 0.25


def test_backtest_no_lookahead_in_fit_window():
    from f1agents.analysis.backtest import deg_continuation
    laps, pits = _synthetic_two_stint_race()
    res = deg_continuation(laps, pits, freeze=10)
    for r in res:
        assert r["fit_window"][1] <= 10
        assert r["scored_laps"][0] > 10


def test_backtest_report_shape():
    from f1agents.analysis.backtest import run_backtest
    laps, pits = _synthetic_two_stint_race()
    rep = run_backtest(laps, pits, freeze=10, pit_loss_s=21.0)
    assert rep["freeze_lap"] == 10
    assert rep["deg_summary"]["n_stints"] >= 1
    assert "undercut_summary" in rep and "pit_window" in rep


# --- Strands roster + AgentCore app (construction only, no AWS calls) ---

def test_strands_roster_builds_all_agents():
    strands = pytest.importorskip("strands")
    from f1agents.agents.strands_roster import TIERS, build_agent, load_prompt
    assert len(TIERS) == 7
    for name in TIERS:
        assert len(load_prompt(name)) > 100
        agent = build_agent(name, region="us-east-1")
        assert agent.name == name
        assert agent.tool_names == []  # payload contract: no tools


def test_build_message_carries_previous_brief():
    from f1agents.agents.strands_roster import build_message
    import json as _json
    msg = _json.loads(build_message({"a": 1}, previous_brief="prior text"))
    assert msg["payload"] == {"a": 1}
    assert msg["previous_brief"] == "prior text"
    assert "changed" in msg["instruction"]
    bare = _json.loads(build_message({"a": 1}))
    assert "previous_brief" not in bare
