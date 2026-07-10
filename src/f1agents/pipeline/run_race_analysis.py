"""End-to-end race analysis pipeline.

Offline:  python -m f1agents.pipeline.run_race_analysis --year 2024 --race Spanish
Produces: outputs/<race>/results.json (deterministic layer, fully reproducible)
          outputs/<race>/batch_input.jsonl (Bedrock Batch manifest for the agent fleet)
Live:     add --invoke to call Bedrock Converse per agent (requires AWS credentials).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.loader import F1Data
from ..analysis.stints import fit_stints, deg_anomalies
from ..analysis.strategy import pit_loss, undercut_events, counterfactual_pit
from ..analysis.profiles import driver_race_metrics, track_profiles, season_driver_table
from ..compliance.rules import validate
from ..agents.base import AGENTS, BedrockAgent

OUT = Path(__file__).resolve().parents[3] / "outputs"


def analyse_race(data: F1Data, year: int, race_name: str,
                 counterfactual_codes: list[str]) -> dict:
    rid = data.race_id(year, race_name)
    race_row = data.races[data.races.raceId == rid].iloc[0]
    laps = data.race_laps(rid)
    pits = data.race_pit_stops(rid)

    fits, fslope = fit_stints(laps, pits)
    result = {
        "race": {"raceId": rid, "name": race_row["name"], "year": year,
                 "laps": int(laps.lap.max()), "drivers": int(laps.driverId.nunique())},
        "fuel_slope_s_per_lap": round(fslope, 4),
        "pit_loss": pit_loss(laps, pits),
        "stint_fits": [f.to_dict() for f in fits],
        "deg_anomalies": deg_anomalies(fits),
        "undercut_events": undercut_events(laps, pits),
        "counterfactuals": [counterfactual_pit(laps, pits, fits, c)
                            for c in counterfactual_codes],
        "driver_metrics": driver_race_metrics(data, rid).round(3).to_dict("records"),
        "lap_traces": _lap_traces(laps, top_n=6),
    }
    return result


def _lap_traces(laps, top_n: int) -> dict:
    """Gap-to-leader traces for the dashboard, top N finishers."""
    top = laps[laps.positionOrder <= top_n]
    cum = top.sort_values(["driverId", "lap"]).copy()
    cum["cum_s"] = cum.groupby("driverId").lap_s.cumsum()
    leader = cum.groupby("lap").cum_s.min()
    cum["gap_s"] = cum.apply(lambda r: r.cum_s - leader.get(r.lap), axis=1)
    traces = {}
    for code, grp in cum.groupby("code"):
        traces[str(code)] = {
            "driver": grp.driver.iloc[0], "team": grp.team.iloc[0],
            "laps": grp.lap.tolist(),
            "gap_s": [round(g, 2) for g in grp.gap_s.tolist()],
            "lap_s": [round(l, 3) for l in grp.lap_s.tolist()],
        }
    return traces


def build_batch_manifest(results: dict, season_table, tracks) -> list[dict]:
    """One JSONL record per agent invocation for Bedrock Batch."""
    records = []
    rname = results["race"]["name"].replace(" ", "_")

    records.append(BedrockAgent(AGENTS["stint_analyst"]).batch_record(
        f"{rname}-stint-analyst",
        {"stint_fits": results["stint_fits"][:40],
         "fuel_slope": results["fuel_slope_s_per_lap"],
         "pit_loss": results["pit_loss"]}))

    records.append(BedrockAgent(AGENTS["rival_watcher"]).batch_record(
        f"{rname}-rival-watcher",
        {"undercut_events": results["undercut_events"],
         "pit_loss": results["pit_loss"]}))

    for i, anom in enumerate(results["deg_anomalies"][:5]):
        records.append(BedrockAgent(AGENTS["deg_explainer"]).batch_record(
            f"{rname}-deg-explainer-{i}",
            {"anomaly": anom,
             "context_stints": [f for f in results["stint_fits"]
                                if f["team"] == anom["team"]]}))

    records.append(BedrockAgent(AGENTS["race_reporter"]).batch_record(
        f"{rname}-race-reporter", {k: results[k] for k in
                                   ["race", "pit_loss", "undercut_events",
                                    "counterfactuals", "deg_anomalies"]}))

    records.append(BedrockAgent(AGENTS["track_historian"]).batch_record(
        f"{rname}-track-historian",
        {"circuit_profiles": tracks.to_dict("records"),
         "focus": results["race"]["name"]}))

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--race", type=str, default="Spanish")
    ap.add_argument("--counterfactual", nargs="+", default=["NOR", "VER"])
    ap.add_argument("--profile-years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--invoke", action="store_true",
                    help="call Bedrock Converse live (requires AWS credentials)")
    args = ap.parse_args()

    data = F1Data()
    results = analyse_race(data, args.year, args.race, args.counterfactual)

    # compliance demo: validate a synthetic recommendation derived from the
    # counterfactual optimum, then a deliberately non-compliant one
    cf = next((c for c in results["counterfactuals"] if c.get("viable")), None)
    audits = []
    if cf:
        d_stints = sorted([f for f in results["stint_fits"] if f["code"] == cf["driver"]],
                          key=lambda f: f["stint"])
        bounds = [cf["optimal_pit_lap"]] + [f["end_lap"] for f in d_stints[1:]]
        lengths, prev = [], 0
        for b in bounds:
            lengths.append(b - prev)
            prev = b
        lengths.append(results["race"]["laps"] - prev)
        rec_ok = {"recommended_pit_lap": cf["optimal_pit_lap"],
                  "race_distance_laps": results["race"]["laps"],
                  "compound_plan": ["soft", "medium", "soft"][:len(lengths)],
                  "stint_lengths": [l for l in lengths if l > 0],
                  "auto_execute": False}
        rec_bad = dict(rec_ok, compound_plan=["medium", "medium"], auto_execute=True)
        audits = [json.loads(validate("strategy_analyst", rec_ok).to_json()),
                  json.loads(validate("strategy_analyst", rec_bad).to_json())]
    results["compliance_audits"] = audits

    season = season_driver_table(data, args.year)
    tracks = track_profiles(data, args.profile_years)
    results["season_driver_table"] = season.round(3).to_dict("records")
    results["track_profiles"] = tracks.round(4).to_dict("records")

    out_dir = OUT / f"{args.year}_{args.race.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))

    manifest = build_batch_manifest(results, season, tracks)
    with open(out_dir / "batch_input.jsonl", "w") as f:
        for rec in manifest:
            f.write(json.dumps(rec, default=str) + "\n")

    if args.invoke:
        briefs = {}
        for rec in manifest:
            key = rec["recordId"].split("-", 1)[1]
            agent_key = next(k for k in AGENTS if k.replace("_", "-") in rec["recordId"])
            agent = BedrockAgent(AGENTS[agent_key])
            briefs[rec["recordId"]] = agent.run(
                json.loads(rec["modelInput"]["messages"][0]["content"][0]["text"]))
        (out_dir / "agent_briefs.json").write_text(json.dumps(briefs, indent=1))

    print(f"wrote {out_dir}/results.json "
          f"({len(results['stint_fits'])} stint fits, "
          f"{len(results['undercut_events'])} pit battles, "
          f"{len(manifest)} batch records)")


if __name__ == "__main__":
    main()
