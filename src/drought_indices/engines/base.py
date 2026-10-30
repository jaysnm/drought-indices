"""Common engine interface.

Every engine works in two stages, following the operational mukau-spi-chirps design:

  fit    distribution parameters per calendar month and pixel over the calibration
         period, stored in ``data/params/<source>/<engine>/``
  apply  the stored parameters to the whole record (or to newly arrived months)

so a monthly update never refits, and the fitted parameters of different engines can
be compared directly as well as their indices.
"""

from __future__ import annotations

import abc
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr

from ..config import Calibration

# Constant added to the monthly water balance (P - PET) before SPEI fitting so that it
# is positive; the same offset climate_indices and R's SPEI package use.
SPEI_OFFSET_MM = 1000.0


@dataclass(frozen=True)
class Job:
    index: str  # "spi" or "spei"
    dist: str  # "gamma", "pearson3" or "loglogistic"
    scale: int

    @property
    def stem(self) -> str:
        return f"{self.index}_{self.dist}_{self.scale:02d}"


@dataclass
class Inputs:
    """Monthly model inputs on one grid, starting in January."""

    precip: xr.DataArray  # mm/month (time, lat, lon)
    pet: xr.DataArray | None = None  # mm/month, same grid; required for SPEI and Palmer
    awc: xr.DataArray | None = None  # mm (lat, lon); required for Palmer

    def __post_init__(self) -> None:
        if int(self.precip.time.dt.month[0]) != 1:
            raise ValueError("Inputs must start in January")

    @property
    def start_year(self) -> int:
        return int(self.precip.time.dt.year[0])

    def water_balance(self) -> xr.DataArray:
        """P - PET + offset (mm/month), NaN outside the common record."""
        wb = self.precip.clip(min=0) - self.pet + SPEI_OFFSET_MM
        wb.attrs = {"units": "mm/month", "long_name": f"P - PET + {SPEI_OFFSET_MM:g} mm"}
        return wb


class Engine(abc.ABC):
    name: str
    # index -> distributions this engine can fit
    supported: dict[str, tuple[str, ...]]

    def __init__(self, cfg: dict, compute_cfg: dict):
        self.cfg = cfg
        self.chunk = int(compute_cfg.get("chunk", 100))
        workers = int(cfg.get("workers", 0) or 0)
        self.workers = workers or os.cpu_count() or 1

    def supports(self, job: Job) -> bool:
        return job.dist in self.supported.get(job.index, ())

    @property
    @abc.abstractmethod
    def version(self) -> str: ...

    @abc.abstractmethod
    def describe(self, job: Job) -> dict[str, str]:
        """Exact distribution and fitting method, written to the output attributes."""

    @abc.abstractmethod
    def fit(self, job: Job, inputs: Inputs, calibration: Calibration) -> xr.Dataset: ...

    @abc.abstractmethod
    def apply(self, job: Job, inputs: Inputs, params: xr.Dataset, calibration: Calibration) -> xr.DataArray: ...

    has_extras = False

    def extras(
        self, spi_inputs: Inputs, wb_inputs: Inputs | None, calibration: Calibration, scales: list[int], palmer: bool
    ) -> dict[str, xr.DataArray]:
        """Indices that are not distribution fits (Palmer, percent of normal)."""
        return {}


# Spatial tiling ---------------------------------------------------------------------------


def tiles(ny: int, nx: int, size: int) -> list[tuple[slice, slice]]:
    return [
        (slice(r, min(r + size, ny)), slice(c, min(c + size, nx)))
        for r in range(0, ny, size)
        for c in range(0, nx, size)
    ]


def map_tiles(
    func: Callable[..., dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
    size: int,
    workers: int,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Run ``func`` on spatial tiles of ``arrays`` in parallel and reassemble the outputs.

    Every array has its last two axes as (lat, lon). ``func`` receives the tiled arrays
    as keyword arguments plus ``kwargs`` and returns a dict of arrays whose last two axes
    are the tile's (lat, lon). Tiles with no valid data in the first array are skipped
    and stay NaN in the output.
    """
    first = next(iter(arrays.values()))
    ny, nx = first.shape[-2:]
    jobs = []
    for rows, cols in tiles(ny, nx, size):
        tile = {k: v[..., rows, cols] for k, v in arrays.items()}
        if not np.isfinite(next(iter(tile.values()))).any():
            continue
        jobs.append(((rows, cols), tile))

    outputs: dict[str, np.ndarray] = {}

    def place(rows: slice, cols: slice, result: dict[str, np.ndarray]) -> None:
        for key, value in result.items():
            if key not in outputs:
                outputs[key] = np.full(value.shape[:-2] + (ny, nx), np.nan, dtype="float32")
            outputs[key][..., rows, cols] = value

    if workers <= 1 or len(jobs) <= 1:
        for (rows, cols), tile in jobs:
            place(rows, cols, func(**tile, **kwargs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(func, **tile, **kwargs): key for key, tile in jobs}
            for future, (rows, cols) in futures.items():
                place(rows, cols, future.result())
    return outputs


def clip_index(da: xr.DataArray, limit: float | None) -> xr.DataArray:
    if not limit:
        return da
    return da.clip(min=-limit, max=limit).where(da.notnull())
