"""Session, driver, season, and championship views over both data sources.

Everything here is deterministic; the championship strategist agent only
narrates numbers computed in this module. Slow lookups (OpenF1 season
sweeps, session pace) cache under outputs/ and mirror to S3 through
service.push_outputs, same as race bundles.

Views:
- meeting_sessions:  FP / Sprint / Qualifying / Race sessions of one GP
- session_pace:      any-session inputs (best lap, clean median, long-run fits)
- season_overview:   standings + per-round points progression
- driver_season:     one driver across every GP of a season
- predictor:         bootstrap projection of the ongoing season + per-driver
                     path-to-title scenarios
- predictor_brief:   agent narrative over the predictor payload
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import F1Data, clean_lap_mask
from ..data.openf1 import save_race
from ..analysis.stints import fit_stints
from ..agents.base import AGENTS, BedrockAgent
from . import service

ROOT = service.ROOT
OUT_DIR = service.OUT_DIR

_mem: dict = {}
_mem_lock = threading.Lock()

MAX_POINTS_PER_ROUND = 25  # sprint points excluded from remaining-max; stated in payloads
N_SIMS = 4000


def _cached(key: str, ttl_s: int, fn):
    with _mem_lock:
        hit = _mem.get(key)
        if hit and time.time() - hit[0] < ttl_s:
            return hit[1]
    value = fn()
    with _mem_lock:
        _mem[key] = (time.time(), value)
    return value


_last_call = [0.0]
_MIN_INTERVAL_S = 0.45  # free tier: 3 req/s, 30 req/min


def _get(url: str):
    for attempt in range(5):
        wait = _MIN_INTERVAL_S - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []  # OpenF1 signals an empty result set with 404
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"rate-limited after retries: {url}")


def _openf1(endpoint: str, **filters):
    q = urllib.parse.urlencode(filters)
    return _get(f"https://api.openf1.org/v1/{endpoint}?{q}")


# --------------------------------------------------------------- sessions

def meeting_sessions(race_key: str) -> list[dict]:
    """All sessions (FP, Sprint, Qualifying, Race) of one OpenF1 grand prix."""
    ref = service._find_ref(race_key)
    if not ref or ref["source"] != "openf1":
        return []  # Ergast carries race laps + qualifying classification only

    def fetch():
        me = _openf1("sessions", session_key=ref["session_key"])
        if not me:
            return []
        meeting_key = me[0]["meeting_key"]
        sessions = _openf1("sessions", meeting_key=meeting_key)
        return [{"session_key": int(s["session_key"]), "name": s["session_name"],
                 "type": s.get("session_type")} for s in sessions]
    return _cached(f"sessions:{race_key}", 6 * 3600, fetch)


def session_pace(session_key: int) -> dict:
    """Deterministic inputs for any session: best lap, clean median, long runs."""
    key = f"session_{session_key}"
    dest = OUT_DIR / key
    cached = dest / "pace.json"
    if cached.exists() or (service.pull_outputs(key) and cached.exists()):
        return json.loads(cached.read_text())

    session_dir = ROOT / "data" / "openf1" / str(session_key)
    if not (session_dir / "laps.csv").exists():
        save_race(session_key, out_dir=str(ROOT / "data" / "openf1"))
    laps = pd.read_csv(session_dir / "laps.csv")
    if laps.empty:
        return {"session_key": session_key, "rows": [], "fits": []}
    from ..data.openf1 import stops_from_laps
    pits = stops_from_laps(laps)
    fits, fslope = fit_stints(laps, pits, min_clean_laps=5)

    meta = _openf1("sessions", session_key=session_key)
    sess = meta[0] if meta else {}
    rows = []
    for code, grp in laps.groupby("code"):
        clean = grp[clean_lap_mask(grp.lap_s)]
        d_fits = [f for f in fits if f.code == code]
        rows.append({
            "code": str(code), "driver": grp.driver.iloc[0], "team": grp.team.iloc[0],
            "best_s": round(float(grp.lap_s.min()), 3),
            "median_clean_s": round(float(clean.lap_s.median()), 3) if len(clean) else None,
            "n_laps": int(len(grp)),
            "long_run_slope": round(float(np.median([f.deg_slope_s_per_lap for f in d_fits])), 4) if d_fits else None,
            "n_long_runs": len(d_fits),
        })
    rows.sort(key=lambda r: r["best_s"])
    best = rows[0]["best_s"] if rows else None
    for r in rows:
        r["gap_to_best_s"] = round(r["best_s"] - best, 3) if best is not None else None
    out = {"session_key": session_key,
           "session_name": sess.get("session_name"),
           "year": sess.get("year"),
           "location": sess.get("location"),
           "fuel_slope_s_per_lap": round(fslope, 4),
           "rows": rows,
           "fits": [f.to_dict() for f in fits]}
    dest.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out, indent=1))
    service.push_outputs(key)
    return out


def ergast_qualifying(race_key: str) -> list[dict]:
    """Qualifying classification for an Ergast race (Q1/Q2/Q3 strings)."""
    ref = service._find_ref(race_key)
    if not ref or ref["source"] != "ergast":
        return []
    data = F1Data()
    q = data._read("qualifying")
    q = q[q.raceId == ref["raceId"]].merge(
        data.drivers[["driverId", "driver", "code"]], on="driverId")
    q = q.sort_values("position")
    return [{"position": int(r.position), "code": str(r.code), "driver": r.driver,
             "q1": None if pd.isna(r.q1) else str(r.q1),
             "q2": None if pd.isna(r.q2) else str(r.q2),
             "q3": None if pd.isna(r.q3) else str(r.q3)}
            for _, r in q.iterrows()]


# ----------------------------------------------------------------- season

def season_overview(year: int) -> dict:
    """Standings and per-round cumulative points, either source."""
    key = f"season_{year}"
    dest = OUT_DIR / key / "season.json"
    if year <= 2024:
        return _cached(f"season:{year}", 24 * 3600, lambda: _ergast_season(year))
    if dest.exists() or (service.pull_outputs(key) and dest.exists()):
        cached = json.loads(dest.read_text())
        # a season in progress grows; refresh when a newer race has completed
        done_now = len([s for s in _openf1_completed(year)
                        if s["session_name"] == "Race" and s["_done"]])
        if len(cached.get("rounds", [])) >= done_now:
            return cached
    out = _openf1_season(year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    service.push_outputs(key)
    return out


def _ergast_season(year: int) -> dict:
    data = F1Data()
    races = data.races[data.races.year == year].sort_values("round")
    res = data.results.merge(races[["raceId", "round", "name"]], on="raceId")
    res = res.merge(data.drivers[["driverId", "driver", "code"]], on="driverId")
    res = res.merge(data.constructors[["constructorId", "name"]]
                    .rename(columns={"name": "team"}), on="constructorId", how="left")
    rounds = races["name"].str.replace(" Grand Prix", "", regex=False).tolist()
    prog, standings = {}, []
    for code, grp in res.groupby("code"):
        by_round = grp.set_index("round").points
        cum, total = [], 0.0
        for rnd in races["round"]:
            total += float(by_round.get(rnd, 0.0))
            cum.append(round(total, 1))
        prog[str(code)] = cum
        standings.append({"code": str(code), "driver": grp.driver.iloc[0],
                          "team": grp.team.iloc[-1], "points": round(total, 1),
                          "wins": int((grp.positionOrder == 1).sum()),
                          "per_round": [float(by_round.get(r, 0.0)) for r in races["round"]]})
    standings.sort(key=lambda s: -s["points"])
    return {"year": year, "source": "ergast", "rounds": rounds,
            "rounds_done": len(rounds), "rounds_total": len(rounds),
            "standings": standings, "progression": prog}


def _openf1_completed(year: int) -> list[dict]:
    """Completed race sessions of a year, one API call, sorted by date."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    done = []
    for s in _openf1("sessions", year=year):
        if s.get("session_name") not in ("Race", "Sprint"):
            continue
        try:
            start = datetime.datetime.fromisoformat(s["date_start"].replace("Z", "+00:00"))
        except Exception:
            continue
        s["_done"] = start < now
        done.append(s)
    return sorted(done, key=lambda s: s["date_start"])


def _openf1_season(year: int) -> dict:
    sessions = _openf1_completed(year)
    races_all = [s for s in sessions if s["session_name"] == "Race"]
    races = [s for s in races_all if s["_done"]]
    sprints = {s["meeting_key"]: s for s in sessions
               if s["session_name"] == "Sprint" and s["_done"]}
    total_scheduled = len(races_all)

    drivers_by_num: dict = {}
    if races:  # one drivers call covers the season; fill gaps per race below
        for d in _openf1("drivers", session_key=races[-1]["session_key"]):
            drivers_by_num.setdefault(d["driver_number"], d)

    rows = []
    for race in races:
        session_keys = [race["session_key"]]
        if race["meeting_key"] in sprints:
            session_keys.append(sprints[race["meeting_key"]]["session_key"])
        race_pts = {}
        for s in session_keys:
            for r in _openf1("session_result", session_key=s):
                num = r["driver_number"]
                race_pts[num] = race_pts.get(num, 0.0) + float(r.get("points") or 0.0)
                if s == race["session_key"] and r.get("position") == 1:
                    race_pts[f"win_{num}"] = 1
        unknown = [n for n in race_pts if isinstance(n, int) and n not in drivers_by_num]
        if unknown:  # mid-season substitute: fetch that race's driver list once
            for d in _openf1("drivers", session_key=race["session_key"]):
                drivers_by_num.setdefault(d["driver_number"], d)
        name = f"{race.get('country_name', race.get('location', '?'))}"
        rows.append({"name": name, "points": race_pts})

    prog, standings = {}, []
    for num, info in drivers_by_num.items():
        code = info.get("name_acronym", f"#{num}")
        cum, total, wins, per_round = [], 0.0, 0, []
        for row in rows:
            pts = float(row["points"].get(num, 0.0))
            total += pts
            wins += int(row["points"].get(f"win_{num}", 0))
            per_round.append(pts)
            cum.append(round(total, 1))
        prog[code] = cum
        standings.append({"code": code, "driver": info.get("full_name", code),
                          "team": info.get("team_name"), "points": round(total, 1),
                          "wins": wins, "per_round": per_round})
    standings.sort(key=lambda s: -s["points"])
    return {"year": year, "source": "openf1",
            "rounds": [r["name"] for r in rows],
            "rounds_done": len(rows), "rounds_total": total_scheduled,
            "standings": standings, "progression": prog}


def driver_season(year: int, code: str) -> dict:
    """One driver across every GP of a season."""
    if year <= 2024:
        data = F1Data()
        races = data.races[data.races.year == year].sort_values("round")
        drv = data.drivers[data.drivers.code == code]
        if drv.empty:
            return {"year": year, "code": code, "races": []}
        # codes are not unique across eras (VER: Vergne and Verstappen);
        # keep the entry that actually raced this season
        res = data.results[data.results.driverId.isin(drv.driverId)
                           & data.results.raceId.isin(races.raceId)]
        if res.empty:
            return {"year": year, "code": code, "driver": drv.driver.iloc[0], "races": []}
        did = int(res.driverId.iloc[0])
        drv = drv[drv.driverId == did]
        res = res[res.driverId == did]
        res = res.merge(races[["raceId", "round", "name"]], on="raceId").sort_values("round")
        rows, cum = [], 0.0
        for _, r in res.iterrows():
            cum += float(r.points)
            rows.append({"round": int(r["round"]),
                         "race": r["name"].replace(" Grand Prix", ""),
                         "grid": int(r.grid), "finish": int(r.positionOrder),
                         "gained": int(r.grid - r.positionOrder) if r.grid > 0 else None,
                         "points": float(r.points), "cum_points": round(cum, 1)})
        return {"year": year, "code": code, "driver": drv.driver.iloc[0], "races": rows}
    season = season_overview(year)
    entry = next((s for s in season["standings"] if s["code"] == code), None)
    if not entry:
        return {"year": year, "code": code, "races": []}
    rows, cum = [], 0.0
    for i, name in enumerate(season["rounds"]):
        pts = entry["per_round"][i]
        cum += pts
        rows.append({"round": i + 1, "race": name, "grid": None, "finish": None,
                     "gained": None, "points": pts, "cum_points": round(cum, 1)})
    return {"year": year, "code": code, "driver": entry["driver"], "races": rows}


# ------------------------------------------------------------- predictor

def predictor(year: int = 2026) -> dict:
    """Bootstrap projection of the ongoing season + path-to-title scenarios."""
    season = season_overview(year)
    done, total = season["rounds_done"], season["rounds_total"]
    left = max(0, total - done)
    top = season["standings"][:10]
    if not top or done == 0:
        return {"year": year, "error": "no completed rounds yet"}

    rng = np.random.default_rng(year)
    sims = {}
    for s in top:
        hist = np.array(s["per_round"], dtype=float)
        draws = rng.choice(hist, size=(N_SIMS, left)).sum(axis=1) if left else np.zeros(N_SIMS)
        sims[s["code"]] = s["points"] + draws
    matrix = np.vstack([sims[s["code"]] for s in top])
    champions = np.argmax(matrix, axis=0)
    leader = top[0]

    projection = []
    for i, s in enumerate(top):
        projection.append({
            **{k: s[k] for k in ("code", "driver", "team", "points", "wins")},
            "avg_points_per_round": round(s["points"] / done, 2),
            "projected_points": round(float(matrix[i].mean()), 1),
            "title_probability": round(float((champions == i).mean()), 3),
            "scenario": _title_scenario(s, leader, left, done),
        })
    return {"year": year, "rounds_done": done, "rounds_total": total,
            "rounds_left": left,
            "method": (f"bootstrap of each driver's own {done} per-round scores over the "
                       f"remaining {left} rounds, {N_SIMS} simulations; sprint points count "
                       "in the standings, remaining-max assumes 25 per round (sprints excluded)"),
            "projection": projection}


def _title_scenario(s: dict, leader: dict, left: int, done: int) -> dict:
    max_remaining = MAX_POINTS_PER_ROUND * left
    gap = round(leader["points"] - s["points"], 1)
    alive = s["points"] + max_remaining > leader["points"]
    leader_avg = leader["points"] / done
    # beat a leader who keeps scoring at their season average
    needed_total = leader["points"] + leader_avg * left + 1
    required_avg = (needed_total - s["points"]) / left if left else None
    wins_needed = None
    if left and required_avg is not None:
        need = needed_total - s["points"]
        for w in range(left + 1):
            if w * 25 + (left - w) * 18 >= need:  # wins + P2s everywhere else
                wins_needed = w
                break
    return {
        "gap_to_leader": gap, "max_remaining_points": max_remaining,
        "mathematically_alive": bool(alive),
        "required_avg_vs_leader_form": round(required_avg, 2) if required_avg is not None else None,
        "wins_needed_rest_p2": wins_needed,
        "note": ("needs more than a win per round; only alive if the leader underscores"
                 if required_avg is not None and required_avg > 25 else None),
    }


def _run_brief(agent_key: str, cache_key: str, fname: str, build_payload,
               force: bool = False, cached_only: bool = False) -> dict:
    """Generic cached agent brief: local -> S3 -> generate via Bedrock."""
    dest = OUT_DIR / cache_key
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / fname
    if not force and not cached.exists():
        service.pull_outputs(cache_key)
    if not force and cached.exists():
        return json.loads(cached.read_text())
    if cached_only:
        return {"cached": False}
    payload = build_payload()
    if "error" in payload:
        return payload
    try:
        agent = BedrockAgent(AGENTS[agent_key], model_id=service.BEDROCK_MODEL)
        out = {"agent": agent_key, "brief": agent.run(payload),
               "model": service.BEDROCK_MODEL}
    except Exception as e:
        return {"agent": agent_key, "error": str(e)}
    cached.write_text(json.dumps(out, indent=1))
    service.push_outputs(cache_key)
    return out


def insight_brief(kind: str, session_key: int | None = None, year: int | None = None,
                  code: str | None = None, force: bool = False,
                  cached_only: bool = False) -> dict:
    """Agent analysis for the drill-down views: session pace, driver season, season."""
    if kind == "session":
        def build():
            d = session_pace(int(session_key))
            if not d.get("rows"):
                return {"error": "no laps in this session"}
            return {"session": {k: d.get(k) for k in ("session_name", "location", "year")},
                    "fuel_slope": d.get("fuel_slope_s_per_lap"),
                    "pace_rows": d["rows"][:20],
                    "stint_fits": d.get("fits", [])[:30],
                    "pit_loss": None}
        return _run_brief("stint_analyst", f"session_{session_key}", "brief.json",
                          build, force, cached_only)
    if kind == "driver":
        def build():
            d = driver_season(int(year), code)
            if not d.get("races"):
                return {"error": f"no {year} races for {code}"}
            season = season_overview(int(year))
            codes = [s["code"] for s in season["standings"]]
            pos = codes.index(code) + 1 if code in codes else None
            neighbours = []
            if pos:
                lo, hi = max(0, pos - 2), min(len(codes), pos + 1)
                neighbours = [{k: s[k] for k in ("code", "driver", "team", "points", "wins")}
                              for s in season["standings"][lo:hi] if s["code"] != code]
            me = next((s for s in season["standings"] if s["code"] == code), {})
            return {"focus": {"code": code, "driver": d.get("driver"),
                              "championship_position": pos,
                              "points": me.get("points"), "wins": me.get("wins")},
                    "races": d["races"], "standings_neighbours": neighbours}
        return _run_brief("driver_review", f"season_{year}", f"brief_driver_{code}.json",
                          build, force, cached_only)
    if kind == "season":
        def build():
            season = season_overview(int(year))
            if not season["standings"]:
                return {"error": f"no standings for {year}"}
            top = season["standings"][:10]
            leader_margin = round(top[0]["points"] - top[1]["points"], 1) if len(top) > 1 else None
            return {"year": season["year"],
                    "rounds_done": season["rounds_done"],
                    "rounds_total": season["rounds_total"],
                    "leader_margin": leader_margin,
                    "standings": [{**{k: s[k] for k in ("code", "driver", "team", "points", "wins")},
                                   "avg_points_per_round": round(s["points"] / max(1, season["rounds_done"]), 2),
                                   "last_three_rounds": s["per_round"][-3:]}
                                  for s in top]}
        return _run_brief("season_analyst", f"season_{year}", "brief_season.json",
                          build, force, cached_only)
    return {"error": f"unknown insight kind: {kind}"}


def predictor_brief(year: int, focus: str, force: bool = False,
                    cached_only: bool = False) -> dict:
    """Championship strategist agent over the predictor payload. Cached per driver."""
    key = f"predictor_{year}"
    dest = OUT_DIR / key
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / f"brief_{focus}.json"
    if not force and not cached.exists():
        service.pull_outputs(key)
    if not force and cached.exists():
        return json.loads(cached.read_text())
    if cached_only:
        return {"cached": False}

    pred = predictor(year)
    if "error" in pred:
        return {"error": pred["error"]}
    entry = next((p for p in pred["projection"] if p["code"] == focus), None)
    if entry is None:
        return {"error": f"driver {focus} not in the projection top 10"}
    payload = {"season": year, "rounds_done": pred["rounds_done"],
               "rounds_left": pred["rounds_left"], "method": pred["method"],
               "standings_projection": pred["projection"][:6],
               "focus_driver": entry}
    try:
        agent = BedrockAgent(AGENTS["championship_strategist"],
                             model_id=service.BEDROCK_MODEL)
        text = agent.run(payload)
        out = {"focus": focus, "brief": text, "model": service.BEDROCK_MODEL}
    except Exception as e:
        out = {"focus": focus, "error": str(e)}
    if "brief" in out:
        cached.write_text(json.dumps(out, indent=1))
        service.push_outputs(key)
    return out
