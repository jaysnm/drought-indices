"""Configuration loading."""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"
# Overrides DEFAULT_CONFIG when --config is not given (used by the Docker image)
CONFIG_ENV = "DROUGHT_INDICES_CONFIG"


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        west, south, east, north = (float(v) for v in values)
        if not (west < east and south < north):
            raise ValueError(f"Invalid bbox {values}; expected [west, south, east, north]")
        return cls(west, south, east, north)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)


@dataclass(frozen=True)
class Calibration:
    start_year: int
    end_year: int


class Config:
    def __init__(self, raw: dict[str, Any], root: Path):
        self.raw = raw
        self.root = root
        self.bbox = BBox.from_list(raw["bbox"])
        self.data_dir = self._path(raw.get("data_dir", "data"))
        self.calibration = Calibration(**raw["calibration"])

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def source(self, name: str) -> dict[str, Any]:
        return self.raw["sources"][name]

    def enabled_sources(self) -> list[str]:
        return [name for name, cfg in self.raw["sources"].items() if cfg.get("enabled", True)]

    def enabled_engines(self) -> list[str]:
        return [name for name, cfg in self.raw["engines"].items() if cfg.get("enabled", True)]

    def calibration_for(self, source: str) -> Calibration:
        override = self.source(source).get("calibration")
        return Calibration(**override) if override else self.calibration

    def start_month(self, source: str) -> dt.date:
        configured = parse_month(self.raw["period"]["start"])
        first = parse_month(self.source(source).get("first_month", "1900-01"))
        return max(configured, first)

    def end_month(self) -> dt.date | None:
        """Configured last month, or None for 'latest available'."""
        end = self.raw["period"].get("end", "latest")
        return None if end in (None, "latest") else parse_month(end)

    # Standard locations ---------------------------------------------------------------

    def raw_dir(self, source: str) -> Path:
        return self.data_dir / "raw" / source

    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def precip_path(self, source: str) -> Path:
        return self.processed_dir() / f"{source}_precip_monthly.nc"

    def aux_dir(self) -> Path:
        return self.data_dir / "aux"

    def indices_dir(self, source: str, engine: str) -> Path:
        return self.data_dir / "indices" / source / engine


def parse_month(value: str | dt.date) -> dt.date:
    if isinstance(value, dt.date):
        return value.replace(day=1)
    year, month = str(value).split("-")[:2]
    return dt.date(int(year), int(month), 1)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    with open(path) as stream:
        try:
            raw = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from None
    root = path.resolve().parent.parent if path.resolve().parent.name == "config" else Path.cwd()
    return Config(raw, root)
