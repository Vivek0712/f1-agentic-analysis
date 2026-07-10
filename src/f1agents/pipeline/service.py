"""Race catalog, cached generation, and S3-backed output store.

Backs scripts/serve_dashboard.py. Three responsibilities:

1. Catalog: Ergast seasons (2011+, pit stop data era) plus OpenF1 seasons
   for years past the Ergast dump, as one season -> races tree.
2. Generation: source-agnostic deterministic run (results + backtests) for
   any race in the catalog, plus fresh agent briefs via Bedrock Converse
   when credentials allow. Every failure in the agent step degrades to a
   bundle without briefs; the deterministic outputs always land.
3. Cache: outputs/<key>/ locally, mirrored to s3://$F1_OUTPUTS_BUCKET/
   outputs/<key>/. A bundle read tries local, then S3, then reports absent
   so the caller can trigger generation.

Published reference outputs (outputs/2024_spanish by default) are treated
as read-only: regeneration is refused so the frozen numbers cannot drift.
"""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import F1Data, clean_lap_mask
from ..data.openf1 import save_race, stops_from_laps
from ..analysis.stints import fit_stints, deg_anomalies
from ..analysis.strategy import pit_loss, undercut_events, counterfactual_pit
from ..analysis.backtest import run_backtest
from ..compliance.rules import validate
from ..agents.base import AGENTS, BedrockAgent

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "outputs"

S3_BUCKET = os.environ.get("F1_OUTPUTS_BUCKET", "f1-agentic-analysis-outputs")
S3_PREFIX = "outputs"
AWS_PROFILE = os.environ.get("F1_AWS_PROFILE") or os.environ.get("AWS_PROFILE")
BEDROCK_MODEL = os.environ.get("F1_BEDROCK_MODEL_ID", "us.amazon.nova-pro-v1:0")
PROTECTED_KEYS = set(os.environ.get("F1_PROTECT_KEYS", "2024_spanish").split(","))

ERGAST_FIRST_YEAR = 2011  # pit_stops table starts here

AGENT_KEYS = ["stint_analyst", "rival_watcher", "deg_explainer", "driver_coach",
              "race_reporter", "track_historian", "compliance_guardian"]

_catalog_cache: dict = {"at": 0.0, "value": None}
_catalog_lock = threading.Lock()


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("grand_prix", "").strip("_")


# ------------------------------------------------------------------ catalog

def list_races(ttl_s: int = 3600) -> dict:
    with _catalog_lock:
        if _catalog_cache["value"] and time.time() - _catalog_cache["at"] < ttl_s:
            return _catalog_cache["value"]
        cat = _build_catalog()
        _catalog_cache.update(at=time.time(), value=cat)
        return cat


def _build_catalog() -> dict:
    seasons = []
    data = F1Data()
    races = data.races
    ergast_years = sorted(y for y in races.year.unique() if y >= ERGAST_FIRST_YEAR)
    for year in ergast_years:
        rows = races[races.year == year].sort_values("round")
        seasons.append({
            "year": int(year), "source": "ergast",
            "races": [{"key": f"{year}_{_slug(r['name'])}", "source": "ergast",
                       "year": int(year), "raceId": int(r.raceId),
                       "name": r["name"], "round": int(r["round"])}
                      for _, r in rows.iterrows()],
        })
    max_ergast = max(ergast_years) if ergast_years else ERGAST_FIRST_YEAR
    this_year = datetime.date.today().year
    for year in range(max_ergast + 1, this_year + 1):
        rows = _openf1_races(year)
        if rows:
            seasons.append({"year": year, "source": "openf1", "races": rows})
    return {"seasons": seasons, "default_year": 2024}


def _openf1_races(year: int) -> list[dict]:
    url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            sessions = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    now = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for s in sessions:
        try:
            start = datetime.datetime.fromisoformat(s["date_start"].replace("Z", "+00:00"))
        except Exception:
            continue
        if start > now:
            continue  # scheduled, not yet raced
        name = f"{s.get('country_name', s.get('location', '?'))} Grand Prix"
        out.append({"key": f"openf1_{s['session_key']}", "source": "openf1",
                    "year": year, "session_key": int(s["session_key"]),
                    "name": name, "circuit": s.get("circuit_short_name")})
    return out


def _find_ref(key: str) -> dict | None:
    for season in list_races()["seasons"]:
        for race in season["races"]:
            if race["key"] == key:
                return race
    return None


# ------------------------------------------------------------- data loading

def _load_frames(ref: dict):
    """Return (laps, pits, meta) for either source, in the pipeline schema."""
    if ref["source"] == "ergast":
        data = F1Data()
        rid = ref["raceId"]
        laps = data.race_laps(rid)
        pits = data.race_pit_stops(rid)
        race_row = data.races[data.races.raceId == rid].iloc[0]
        circ = data.circuits[data.circuits.circuitId == race_row.circuitId]
        circuit = str(circ.name.iloc[0]) if len(circ) else None
        meta = {"raceId": rid, "name": ref["name"], "year": ref["year"],
                "circuit": circuit, "source": "ergast"}
    else:
        session_dir = ROOT / "data" / "openf1" / str(ref["session_key"])
        if not (session_dir / "laps.csv").exists():
            save_race(ref["session_key"], out_dir=str(ROOT / "data" / "openf1"))
        laps = pd.read_csv(session_dir / "laps.csv")
        pits = pd.read_csv(session_dir / "pits.csv")
        if pits.empty:
            pits = stops_from_laps(laps)
        meta = {"session_key": ref["session_key"], "name": ref["name"],
                "year": ref["year"], "circuit": ref.get("circuit"),
                "source": "openf1"}
    if "positionOrder" not in laps.columns:
        laps = _with_position_order(laps)
    return laps, pits, meta


def _with_position_order(laps: pd.DataFrame) -> pd.DataFrame:
    """Finishing order from laps completed, then cumulative time."""
    df = laps.sort_values(["driverId", "lap"]).copy()
    df["cum_s"] = df.groupby("driverId").lap_s.cumsum()
    final = df.groupby("driverId").agg(n=("lap", "max"), t=("cum_s", "max"))
    final = final.sort_values(["n", "t"], ascending=[False, True])
    order = {d: i + 1 for i, d in enumerate(final.index)}
    df["positionOrder"] = df.driverId.map(order)
    return df.drop(columns=["cum_s"])


# ---------------------------------------------------------------- analysis

def _lap_traces(laps: pd.DataFrame, top_n: int = 6) -> dict:
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


def _driver_metrics(laps: pd.DataFrame, fits: list) -> list[dict]:
    rows = []
    for code, grp in laps.groupby("code"):
        clean = grp[clean_lap_mask(grp.lap_s)]
        if len(clean) < 10:
            continue
        d_fits = [f for f in fits if f.code == code]
        q75, q25 = np.percentile(clean.lap_s, [75, 25])
        rows.append({
            "code": str(code), "driver": grp.driver.iloc[0], "team": grp.team.iloc[0],
            "median_clean_lap_s": round(float(clean.lap_s.median()), 3),
            "consistency_iqr_s": round(float(q75 - q25), 3),
            "mean_deg_slope": round(float(np.mean([f.deg_slope_s_per_lap for f in d_fits])), 3) if d_fits else None,
            "n_stints_fitted": len(d_fits),
        })
    return sorted(rows, key=lambda r: r["median_clean_lap_s"])


def _compliance_demo(results: dict) -> list[dict]:
    cf = next((c for c in results["counterfactuals"] if c.get("viable")), None)
    if not cf:
        return []
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
    return [json.loads(validate("strategy_analyst", rec_ok).to_json()),
            json.loads(validate("strategy_analyst", rec_bad).to_json())]


def _shared_track_profiles() -> list[dict]:
    """Cross-season circuit features are race-independent; reuse the published set."""
    ref = OUT_DIR / "2024_spanish" / "results.json"
    if ref.exists():
        return json.loads(ref.read_text()).get("track_profiles", [])
    return []


def analyse(ref: dict) -> dict:
    laps, pits, meta = _load_frames(ref)
    fits, fslope = fit_stints(laps, pits)
    finishing = laps.sort_values("positionOrder").drop_duplicates("driverId")
    cf_codes = [str(c) for c in finishing.code.head(2).tolist()]
    results = {
        "race": {"name": meta["name"], "year": meta["year"],
                 "circuit": meta.get("circuit"), "source": meta["source"],
                 "laps": int(laps.lap.max()), "drivers": int(laps.driverId.nunique())},
        "fuel_slope_s_per_lap": round(fslope, 4),
        "pit_loss": pit_loss(laps, pits),
        "stint_fits": [f.to_dict() for f in fits],
        "deg_anomalies": deg_anomalies(fits),
        "undercut_events": undercut_events(laps, pits),
        "counterfactuals": [counterfactual_pit(laps, pits, fits, c) for c in cf_codes],
        "driver_metrics": _driver_metrics(laps, fits),
        "lap_traces": _lap_traces(laps),
        "track_profiles": _shared_track_profiles(),
    }
    results["compliance_audits"] = _compliance_demo(results)
    return results, laps, pits


def _freezes(race_len: int) -> list[int]:
    return sorted({max(6, int(race_len * f)) for f in (0.3, 0.45, 0.6)})


# ------------------------------------------------------------ agent briefs

def _agent_payloads(results: dict) -> dict:
    p = {
        "stint_analyst": {"stint_fits": results["stint_fits"][:40],
                          "fuel_slope": results["fuel_slope_s_per_lap"],
                          "pit_loss": results["pit_loss"]},
        "rival_watcher": {"undercut_events": results["undercut_events"],
                          "pit_loss": results["pit_loss"]},
        "race_reporter": {k: results[k] for k in
                          ["race", "pit_loss", "undercut_events",
                           "counterfactuals", "deg_anomalies"]},
    }
    if results["deg_anomalies"]:
        anom = results["deg_anomalies"][0]
        p["deg_explainer"] = {"anomaly": anom,
                              "context_stints": [f for f in results["stint_fits"]
                                                 if f["team"] == anom["team"]]}
    cf = next((c for c in results["counterfactuals"] if c.get("viable")), None)
    if cf and results["driver_metrics"]:
        p["driver_coach"] = {"focus_driver": cf["driver"],
                             "driver_metrics": results["driver_metrics"],
                             "counterfactual": cf}
    if results["track_profiles"]:
        p["track_historian"] = {"circuit_profiles": results["track_profiles"],
                                "focus": results["race"]["name"]}
    if len(results["compliance_audits"]) > 1:
        p["compliance_guardian"] = {"audit_record": results["compliance_audits"][1]}
    return p


def run_agents(results: dict, progress=lambda s: None) -> tuple[dict, str | None]:
    """Invoke each agent once via Bedrock Converse. Returns (briefs, error_note)."""
    briefs, first_error = {}, None
    for name, payload in _agent_payloads(results).items():
        progress(f"agent brief: {name}")
        try:
            agent = BedrockAgent(AGENTS[name], model_id=BEDROCK_MODEL)
            briefs[name] = agent.run(payload)
        except Exception as e:  # degrade: deterministic outputs stand without briefs
            first_error = first_error or f"{name}: {e}"
    return briefs, first_error


# ------------------------------------------------------------------- cache

def _s3():
    import boto3
    session = boto3.Session(profile_name=AWS_PROFILE) if AWS_PROFILE else boto3.Session()
    return session.client("s3")


def _ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=S3_BUCKET)


def push_outputs(key: str, progress=lambda s: None) -> str | None:
    """Upload outputs/<key>/ to S3. Returns an error note or None."""
    local = OUT_DIR / key
    try:
        s3 = _s3()
        _ensure_bucket(s3)
        for f in sorted(local.glob("*.json")):
            progress(f"s3 upload: {f.name}")
            s3.upload_file(str(f), S3_BUCKET, f"{S3_PREFIX}/{key}/{f.name}")
        return None
    except Exception as e:
        return f"s3 push failed: {e}"


def pull_outputs(key: str) -> bool:
    """Download outputs/<key>/ from S3 if present. True when files landed."""
    try:
        s3 = _s3()
        listing = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/{key}/")
        objs = listing.get("Contents", [])
        if not objs:
            return False
        dest = OUT_DIR / key
        dest.mkdir(parents=True, exist_ok=True)
        for o in objs:
            fname = o["Key"].rsplit("/", 1)[-1]
            if fname:
                s3.download_file(S3_BUCKET, o["Key"], str(dest / fname))
        return True
    except Exception:
        return False


def _normalize_briefs(raw: dict) -> dict:
    slugs = [k.replace("_", "-") for k in AGENT_KEYS]
    out = {}
    for k, v in raw.items():
        if k.startswith("_") or not isinstance(v, str):
            continue
        if k in AGENT_KEYS:
            out.setdefault(k, v)
            continue
        tail = k.split("-", 1)[1] if "-" in k else k
        for slug in slugs:
            if tail.startswith(slug):
                out.setdefault(slug.replace("-", "_"), v)
                break
    return out


def bundle(key: str, allow_s3: bool = True) -> dict | None:
    local = OUT_DIR / key
    if not (local / "results.json").exists() and allow_s3:
        pull_outputs(key)
    if not (local / "results.json").exists():
        return None
    results = json.loads((local / "results.json").read_text())
    backtests = sorted((json.loads(f.read_text()) for f in local.glob("backtest_*.json")),
                       key=lambda b: b["freeze_lap"])
    briefs = {}
    for fname in ("agent_briefs.json", "sample_briefs.json"):
        if (local / fname).exists():
            briefs = _normalize_briefs(json.loads((local / fname).read_text()))
            break
    if "stint_fits_full_count" not in results:
        results["stint_fits_full_count"] = len(results.get("stint_fits", []))
    return {"key": key, "results": results, "briefs": briefs,
            "backtests": backtests, "from": "local"}


# --------------------------------------------------------------- generation

def generate(key: str, force: bool = False, invoke_agents: bool = True,
             progress=lambda s: None) -> dict:
    """Deterministic run + backtests + agent briefs for one race key."""
    if key in PROTECTED_KEYS and force:
        raise PermissionError(
            f"{key} holds published reference outputs and is read-only; "
            "its numbers are frozen")
    ref = _find_ref(key)
    if ref is None:
        raise KeyError(f"unknown race key: {key}")
    dest = OUT_DIR / key
    dest.mkdir(parents=True, exist_ok=True)

    progress("deterministic analysis")
    results, laps, pits = analyse(ref)
    (dest / "results.json").write_text(json.dumps(results, indent=1, default=str))

    ploss = results["pit_loss"]["pit_loss_s"]
    for fz in _freezes(results["race"]["laps"]):
        progress(f"backtest at freeze L{fz}")
        report = run_backtest(laps, pits, fz, ploss)
        report["session"] = results["race"]["name"]
        report["pit_loss_s"] = ploss
        (dest / f"backtest_{fz}.json").write_text(json.dumps(report, indent=2))

    note = None
    if invoke_agents:
        need = force or not (dest / "agent_briefs.json").exists()
        if need:
            briefs, err = run_agents(results, progress)
            if briefs:
                (dest / "agent_briefs.json").write_text(json.dumps(briefs, indent=1))
            if err:
                note = f"agent briefs degraded ({err})"
        else:
            note = "reused cached agent briefs"

    s3_err = push_outputs(key, progress)
    if s3_err:
        note = f"{note}; {s3_err}" if note else s3_err
    return {"key": key, "note": note}
