"""OpenF1 REST adapter: live and historical sessions into the pipeline schema.

OpenF1 (api.openf1.org) serves F1 timing data from 2023 onward. Historical
data is free without authentication; true real-time REST/MQTT access is a
paid tier. Data lags the track by roughly 3 seconds, which is far inside
the slow-loop tier budget (30-120s), so the same adapter serves both the
in-session cadence and post-session pulls.

Endpoints used:
  /v1/sessions   resolve year+country to a session_key (or pass 'latest')
  /v1/laps       lap_number, lap_duration, is_pit_out_lap per driver
  /v1/pit        pit stop lap numbers and lane durations
  /v1/stints     tyre compound per stint (upgrade over the Ergast schema:
                 degradation fits become compound-conditional)
  /v1/drivers    names, acronyms, teams

Output frames match data/loader.py exactly (driverId, lap, lap_s, driver,
code, team / driverId, lap), so fit_stints, undercut_events,
counterfactual_pit, and the backtest run unchanged on live data.

Rate limits (free tier): 3 req/s, 30 req/min. The client sleeps between
calls and backs off on HTTP 429.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://api.openf1.org/v1"
_MIN_INTERVAL_S = 0.4


class OpenF1Client:
    def __init__(self, base: str = BASE):
        self.base = base
        self._last_call = 0.0

    def get(self, endpoint: str, **filters) -> list[dict]:
        query = urllib.parse.urlencode(filters)
        url = f"{self.base}/{endpoint}?{query}" if query else f"{self.base}/{endpoint}"
        for attempt in range(5):
            wait = _MIN_INTERVAL_S - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.time()
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"rate-limited after retries: {url}")

    def resolve_race(self, year: int, country: str) -> dict:
        sessions = self.get("sessions", year=year, country_name=country,
                            session_name="Race")
        if not sessions:
            raise ValueError(f"no race session for {year} {country}")
        return sessions[-1]

    def race_frames(self, session_key: int | str
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (race_laps, race_pits, stints) in the pipeline schema."""
        drivers = {d["driver_number"]: d
                   for d in self.get("drivers", session_key=session_key)}
        laps_raw = self.get("laps", session_key=session_key)
        pits_raw = self.get("pit", session_key=session_key)
        stints_raw = self.get("stints", session_key=session_key)

        laps = normalize_laps(laps_raw, drivers)
        pits = pd.DataFrame(
            [{"driverId": p["driver_number"], "lap": int(p["lap_number"]),
              "lane_duration_s": p.get("lane_duration")}
             for p in pits_raw if p.get("lap_number")])
        stints = pd.DataFrame(
            [{"driverId": s["driver_number"], "stint": s["stint_number"],
              "compound": s.get("compound"),
              "lap_start": s.get("lap_start"), "lap_end": s.get("lap_end"),
              "tyre_age_at_start": s.get("tyre_age_at_start")}
             for s in stints_raw])
        return laps, pits, stints


def normalize_laps(laps_raw: list[dict], drivers: dict | None = None
                   ) -> pd.DataFrame:
    """OpenF1 lap records into the pipeline lap frame.

    Laps with null lap_duration (first lap behind the wall, red flags) are
    dropped. Timed pit in/out laps stay in; the stint fitter's clean-lap
    mask and stint-edge trimming handle them, same as the Ergast path.
    """
    drivers = drivers or {}
    rows = []
    for lap in laps_raw:
        if lap.get("lap_duration") is None:
            continue
        n = lap["driver_number"]
        info = drivers.get(n, {})
        rows.append({
            "driverId": n,
            "lap": int(lap["lap_number"]),
            "lap_s": float(lap["lap_duration"]),
            "is_pit_out_lap": bool(lap.get("is_pit_out_lap")),
            "driver": info.get("full_name", f"#{n}"),
            "code": info.get("name_acronym", f"#{n}"),
            "team": info.get("team_name", "unknown"),
        })
    return pd.DataFrame(rows).sort_values(["driverId", "lap"]).reset_index(drop=True)


def stops_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Fallback pit table when /v1/pit is empty: stop lap = out-lap - 1."""
    out = laps[laps.get("is_pit_out_lap", False) & (laps.lap > 1)]
    return pd.DataFrame({"driverId": out.driverId.values,
                         "lap": out.lap.values - 1})


def save_race(session_key: int | str, out_dir: str = "data/openf1") -> Path:
    """Pull a session and persist it for the offline pipeline."""
    client = OpenF1Client()
    laps, pits, stints = client.race_frames(session_key)
    if pits.empty:
        pits = stops_from_laps(laps)
    out = Path(out_dir) / str(session_key)
    out.mkdir(parents=True, exist_ok=True)
    laps.to_csv(out / "laps.csv", index=False)
    pits.to_csv(out / "pits.csv", index=False)
    stints.to_csv(out / "stints.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session-key", default="latest",
                   help="OpenF1 session_key, or 'latest'")
    p.add_argument("--year", type=int)
    p.add_argument("--country")
    a = p.parse_args()
    key = a.session_key
    if a.year and a.country:
        key = OpenF1Client().resolve_race(a.year, a.country)["session_key"]
    path = save_race(key)
    print(f"saved session {key} -> {path}")
