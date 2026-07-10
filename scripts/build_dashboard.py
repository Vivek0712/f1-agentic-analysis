"""Build the standalone dashboard: inject trimmed results + briefs into the template.

Usage: python scripts/build_dashboard.py outputs/2024_spanish dashboard/index.html
"""

import json
import sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/2024_spanish")
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dashboard/index.html")

AGENT_SLUGS = ["stint-analyst", "rival-watcher", "deg-explainer", "driver-coach",
               "race-reporter", "track-historian", "compliance-guardian"]


def normalize_briefs(raw: dict) -> dict:
    """Record-id keyed briefs (Spanish_Grand_Prix-stint-analyst, ...) -> agent-keyed."""
    out = {}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, str):
            continue
        tail = k.split("-", 1)[1] if "-" in k else k
        for slug in AGENT_SLUGS:
            if tail.startswith(slug):
                out.setdefault(slug.replace("-", "_"), v)
                break
        else:
            out.setdefault(k, v)
    return out


results = json.loads((src / "results.json").read_text())
briefs_file = src / ("agent_briefs.json" if (src / "agent_briefs.json").exists()
                     else "sample_briefs.json")
briefs = normalize_briefs(json.loads(briefs_file.read_text()))
backtest_files = sorted(src.glob("backtest_*.json"))
if not backtest_files:
    raise FileNotFoundError(f"no backtest_*.json in {src}; run f1agents.pipeline.run_backtest first")
backtests = sorted(
    (json.loads(f.read_text()) for f in backtest_files),
    key=lambda b: b["freeze_lap"])

# trim payload to what the page renders
trim = {
    "race": results["race"],
    "fuel_slope_s_per_lap": results["fuel_slope_s_per_lap"],
    "pit_loss": results["pit_loss"],
    "stint_fits": [{k: f[k] for k in ["code", "stint", "start_lap", "end_lap"]}
                   for f in results["stint_fits"]],
    "undercut_events": results["undercut_events"],
    "counterfactuals": results["counterfactuals"],
    "deg_anomalies": results["deg_anomalies"][:6],
    "driver_metrics": results["driver_metrics"],
    "lap_traces": results["lap_traces"],
    "track_profiles": results["track_profiles"],
    "compliance_audits": results["compliance_audits"],
}
trim["stint_fits_full_count"] = len(results["stint_fits"])

tpl = Path("dashboard/template.html").read_text()
html = tpl.replace("/*__RESULTS__*/", json.dumps(trim)) \
          .replace("/*__BRIEFS__*/", json.dumps(briefs)) \
          .replace("/*__BACKTESTS__*/", json.dumps(backtests))
dst.write_text(html)
print(f"wrote {dst} ({len(html)//1024} KB)")
