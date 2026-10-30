"""Monthly rainfall sources, clipped to the bounding box while they are read.

Every source writes one clipped GeoTIFF per month to ``data/raw/<source>/YYYY/`` and the
months are then stacked into ``data/processed/<source>_precip_monthly.nc``. Months already
on disk are skipped, so an interrupted run resumes where it stopped and a scheduled run
only fetches newly published months.
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import BBox, Config
from ..raster import Clipped, RemoteNotFound, read_clipped, vsi_url, write_geotiff

log = logging.getLogger(__name__)

# Months this close to today that are not published yet are expected, not errors
PUBLICATION_LAG_MONTHS = 3


class PrecipSource:
    """A source that publishes one file per month (CHIRPS)."""

    def __init__(self, name: str, cfg: dict, bbox: BBox, raw_dir: Path):
        self.name = name
        self.cfg = cfg
        self.bbox = bbox
        self.raw_dir = raw_dir

    def month_path(self, month: dt.date) -> Path:
        return self.raw_dir / f"{month:%Y}" / f"{self.name}_{month:%Y%m}.tif"

    def candidate_urls(self, month: dt.date) -> list[str]:
        # Several URL templates are tried in order: CHC sometimes publishes only a gzipped
        # or only a plain GeoTIFF for a month
        return [url.format(year=month.year, month=month.month) for url in self.cfg["urls"]]

    def read_month(self, month: dt.date) -> tuple[Clipped, dict[str, str]]:
        for url in self.candidate_urls(month):
            try:
                return read_clipped(vsi_url(url), self.bbox), {"source_url": url}
            except RemoteNotFound:
                continue
        raise RemoteNotFound(f"{self.name} {month:%Y-%m}")

    def fetch_month(self, month: dt.date, overwrite: bool = False) -> Path | None:
        out = self.month_path(month)
        if out.exists() and not overwrite:
            return out
        try:
            clipped, tags = self.read_month(month)
        except RemoteNotFound:
            return None
        write_geotiff(clipped, out, month=f"{month:%Y-%m}", units="mm/month", **tags)
        log.info("%s %s -> %s", self.name, f"{month:%Y-%m}", out)
        return out


class DailyPrecipSource(PrecipSource):
    """A source that publishes daily files (CPC RFE2, ARC2); summed to monthly totals.

    A month is only written once it is complete. Months with more than
    ``max_missing_days`` missing days are written as all-NaN so they are not retried
    forever, and the number of missing days is stored in the file's tags.
    """

    def _read_day(self, date: dt.date) -> Clipped | None:
        url = self.cfg["url"].format(date=date)
        # Some archives name the file inside differently (ARC2 July 2024 to February 2025
        # holds daily_clim.YYYYMMDD.tif), so the configured alternatives are tried in turn
        members = [None] + [m.format(date=date) for m in self.cfg.get("zip_members", [])]
        for member in members:
            try:
                return read_clipped(vsi_url(url, member), self.bbox)
            except RemoteNotFound:
                continue
        return None

    def read_month(self, month: dt.date) -> tuple[Clipped, dict[str, str]]:
        days = calendar.monthrange(month.year, month.month)[1]
        dates = [month.replace(day=day) for day in range(1, days + 1)]
        # The days of a month are read concurrently, so a single-month update is not
        # limited by the latency of 31 sequential requests
        with ThreadPoolExecutor(max_workers=int(self.cfg.get("day_workers", 4))) as pool:
            results = list(pool.map(self._read_day, dates))

        total: np.ndarray | None = None
        first: Clipped | None = None
        missing: list[str] = []
        for date, clipped in zip(dates, results):
            if clipped is None:
                missing.append(f"{date:%d}")
                continue
            if first is None:
                first, total = clipped, clipped.data.copy()
            else:
                # NaN in any day makes the monthly total NaN for that pixel
                total += clipped.data

        if first is None:
            raise RemoteNotFound(f"{self.name} {month:%Y-%m}")
        if self._is_incomplete_recent(month, missing):
            raise RemoteNotFound(f"{self.name} {month:%Y-%m} is not complete yet")
        if len(missing) > int(self.cfg.get("max_missing_days", 0)):
            log.warning("%s %s: %d missing days, writing NaN", self.name, f"{month:%Y-%m}", len(missing))
            total[:] = np.nan
        tags = {"missing_days": ",".join(missing), "days_used": str(days - len(missing))}
        return Clipped(total, first.transform, first.crs), tags

    @staticmethod
    def _is_incomplete_recent(month: dt.date, missing: list[str]) -> bool:
        if not missing:
            return False
        months_ago = (dt.date.today().year - month.year) * 12 + dt.date.today().month - month.month
        return months_ago <= PUBLICATION_LAG_MONTHS


def make_source(name: str, config: Config) -> PrecipSource:
    cfg = config.source(name)
    cls = DailyPrecipSource if "url" in cfg and "{date" in cfg["url"] else PrecipSource
    return cls(name, cfg, config.bbox, config.raw_dir(name))


def month_range(start: dt.date, end: dt.date) -> list[dt.date]:
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return months


def last_complete_month(today: dt.date | None = None) -> dt.date:
    today = today or dt.date.today()
    return (today.replace(day=1) - dt.timedelta(days=1)).replace(day=1)


def download(name: str, config: Config, refresh_last: int = 0) -> list[Path]:
    """Fetch every missing month for ``name``; re-fetch the last ``refresh_last`` months.

    ``refresh_last`` replaces months that a provider may have revised since they were
    first downloaded (the mukau-spi-chirps SYNC_CURRENT_MONTH behaviour).
    """
    source = make_source(name, config)
    end = config.end_month() or last_complete_month()
    months = month_range(config.start_month(name), end)
    refresh = set(months[-refresh_last:]) if refresh_last else set()

    workers = int(config["download"].get("workers", 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda m: source.fetch_month(m, overwrite=m in refresh), months))

    fetched = [path for path in results if path is not None]
    absent = [m for m, path in zip(months, results) if path is None]
    for month in absent:
        lag = (end.year - month.year) * 12 + end.month - month.month
        if lag >= PUBLICATION_LAG_MONTHS:
            log.error("%s %s is missing on the server", name, f"{month:%Y-%m}")
        else:
            log.info("%s %s not published yet", name, f"{month:%Y-%m}")
    return fetched
