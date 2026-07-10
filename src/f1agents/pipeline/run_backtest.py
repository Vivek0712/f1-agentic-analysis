"""Frozen-clock backtest CLI.

Historical (Ergast schema):
    python -m f1agents.pipeline.run_backtest --year 2024 --race Spanish --freeze 20

Live/recent (OpenF1, after saving a session):
    python -m f1agents.data.openf1 --year 2026 --country "Great Britain"
    python -m f1agents.pipeline.run_backtest --openf1 data/openf1/<session_key> --freeze 20

Writes outputs/<race>/backtest_<freeze>.json.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ..analysis.backtest import run_backtest
from ..analysis.strategy import pit_loss
from ..data.loader import F1Data
from ..data.openf1 import stops_from_laps


def load_ergast(year: int, race: str):
    data = F1Data()
    races = data.races
    row = races[(races.year == year) & races.name.str.contains(race, case=False)]
    if row.empty:
        raise ValueError(f"race not found: {year} {race}")
    race_id = int(row.raceId.iloc[0])
    return data.race_laps(race_id), data.race_pit_stops(race_id), str(row["name"].iloc[0])


def load_openf1(session_dir: str):
    d = Path(session_dir)
    laps = pd.read_csv(d / "laps.csv")
    pits = pd.read_csv(d / "pits.csv")
    if pits.empty:
        pits = stops_from_laps(laps)
    return laps, pits, d.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int)
    p.add_argument("--race")
    p.add_argument("--openf1", help="directory written by f1agents.data.openf1")
    p.add_argument("--freeze", type=int, required=True)
    a = p.parse_args()

    if a.openf1:
        laps, pits, label = load_openf1(a.openf1)
    else:
        laps, pits, label = load_ergast(a.year, a.race)

    ploss = pit_loss(laps, pits)["pit_loss_s"]
    report = run_backtest(laps, pits, a.freeze, ploss)
    report["session"] = str(label)
    report["pit_loss_s"] = ploss

    out_dir = Path("../outputs") if Path.cwd().name == "src" else Path("outputs")
    slug = str(label).lower().replace(" ", "_").replace("grand_prix", "").strip("_")
    dest = out_dir / (f"{a.year}_{slug}" if a.year else slug)
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / f"backtest_{a.freeze}.json"
    out_file.write_text(json.dumps(report, indent=2))

    ds, us = report["deg_summary"], report["undercut_summary"]
    print(f"wrote {out_file}")
    print(f"deg continuation: {ds['n_stints']} stints, MAE {ds['mae_s']}s, bias {ds['bias_s']}s")
    print(f"undercut calls: {us['n_calls']}, sign hit rate {us['sign_hit_rate']}, "
          f"mean abs err {us['mean_abs_error_s']}s")
    print(f"pit windows scored: {len(report['pit_window'])}")


if __name__ == "__main__":
    main()
