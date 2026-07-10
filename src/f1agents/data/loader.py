"""Ergast-schema data loader.

Loads the historical F1 dataset (Ergast schema, CSV export) and exposes
merged, analysis-ready frames. In a live AWS deployment the same interface
is backed by FastF1/live-timing ingestion through Kinesis; this loader is
the offline/batch path and the one used for all published results.

Data limitations (documented deliberately):
- Lap times are total lap times in milliseconds. No sector times.
- No tyre compound labels (Ergast never carried them). Degradation is
  therefore computed as a fuel-corrected pace-decay proxy per stint.
- Pit stop durations are pit-lane transit times (2011+).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

TABLES = [
    "races", "results", "lap_times", "pit_stops",
    "drivers", "constructors", "circuits", "qualifying", "status",
]


class F1Data:
    """Lazy container for the Ergast-schema tables plus merged views."""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)

    @functools.cached_property
    def races(self) -> pd.DataFrame:
        df = self._read("races")
        df["date"] = pd.to_datetime(df["date"])
        return df

    @functools.cached_property
    def results(self) -> pd.DataFrame:
        return self._read("results")

    @functools.cached_property
    def lap_times(self) -> pd.DataFrame:
        return self._read("lap_times")

    @functools.cached_property
    def pit_stops(self) -> pd.DataFrame:
        return self._read("pit_stops")

    @functools.cached_property
    def drivers(self) -> pd.DataFrame:
        df = self._read("drivers")
        df["driver"] = df["forename"].str.strip() + " " + df["surname"].str.strip()
        return df

    @functools.cached_property
    def constructors(self) -> pd.DataFrame:
        return self._read("constructors")

    @functools.cached_property
    def circuits(self) -> pd.DataFrame:
        return self._read("circuits")

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / f"{name}.csv", na_values=["\\N"])

    # ---------------------------------------------------------------- views

    def race_id(self, year: int, name_contains: str) -> int:
        r = self.races[
            (self.races.year == year)
            & (self.races.name.str.contains(name_contains, case=False))
        ]
        if len(r) != 1:
            raise ValueError(f"ambiguous or missing race: {year} {name_contains!r} -> {len(r)} rows")
        return int(r.raceId.iloc[0])

    def race_laps(self, race_id: int) -> pd.DataFrame:
        """Lap times for one race, joined with driver + constructor identity."""
        lt = self.lap_times[self.lap_times.raceId == race_id].copy()
        res = self.results[self.results.raceId == race_id][
            ["driverId", "constructorId", "grid", "positionOrder", "statusId"]
        ]
        lt = lt.merge(res, on="driverId", how="left")
        lt = lt.merge(self.drivers[["driverId", "driver", "code"]], on="driverId")
        lt = lt.merge(
            self.constructors[["constructorId", "name"]].rename(columns={"name": "team"}),
            on="constructorId", how="left",
        )
        lt["lap_s"] = lt.milliseconds / 1000.0
        return lt.sort_values(["driverId", "lap"]).reset_index(drop=True)

    def race_pit_stops(self, race_id: int) -> pd.DataFrame:
        ps = self.pit_stops[self.pit_stops.raceId == race_id].copy()
        ps = ps.merge(self.drivers[["driverId", "driver", "code"]], on="driverId")
        ps["pit_s"] = ps.milliseconds / 1000.0
        return ps.sort_values(["driverId", "stop"]).reset_index(drop=True)

    def seasons(self, years: list[int]) -> pd.DataFrame:
        return self.races[self.races.year.isin(years)].sort_values(["year", "round"])


def clean_lap_mask(lap_s: pd.Series, tolerance: float = 1.07) -> pd.Series:
    """True for representative racing laps.

    Excludes in/out laps and neutralised laps by thresholding against the
    driver's own median lap: anything slower than tolerance * median is
    treated as traffic, SC/VSC, or a pit-affected lap.
    """
    med = np.nanmedian(lap_s)
    return lap_s < med * tolerance
