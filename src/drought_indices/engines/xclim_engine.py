"""xclim (Ouranos) standardized indices, with the APP or PWM fitting method.

APP  closed-form approximate fit of a 2-parameter distribution with loc fixed at 0
     (``fitkwargs={"floc": 0}``); with gamma this is the same Thom estimator that
     climate_indices uses.
       spi:  gamma
       spei: gamma, loglogistic (scipy "fisk", 2-parameter)
PWM  probability-weighted moments (L-moments) with lmoments3 distributions
       spi:  gamma (lmoments3 "gam"), pearson3 ("pe3")
       spei: gamma, pearson3, loglogistic (generalized logistic "glo", as in R's SPEI)

SPI uses xclim's zero-inflated fit; the probability given to a zero is the
"upper"/"ecdf" plotting position, i.e. the fraction of zeros in the calibration
period, which is how climate_indices places zeros too.

To reuse stored parameters the accumulation is done here (a rolling mean, as xclim
does internally) and xclim's ``standardized_index`` is called with ``params``.
"""

from __future__ import annotations

import json
import logging
import warnings

import scipy.stats
import xarray as xr

from ..config import Calibration
from .base import Engine, Inputs, Job

log = logging.getLogger(__name__)

SUPPORTED = {
    "APP": {"spi": ("gamma",), "spei": ("gamma", "loglogistic")},
    "PWM": {"spi": ("gamma", "pearson3"), "spei": ("gamma", "pearson3", "loglogistic")},
}


def _lmoments_dist(dist: str):
    from lmoments3 import distr

    lmom = {"gamma": distr.gam, "pearson3": distr.pe3, "loglogistic": distr.glo}[dist]
    # xclim looks a stored distribution up by name in scipy.stats when parameters are
    # reused; lmoments3 names such as "glo" are not there, so register them
    if not hasattr(scipy.stats, lmom.name):
        setattr(scipy.stats, lmom.name, lmom)
    return lmom


class XclimEngine(Engine):
    def __init__(self, name: str, cfg: dict, compute_cfg: dict):
        super().__init__(cfg, compute_cfg)
        self.name = name
        self.method = cfg["method"].upper()
        if self.method not in SUPPORTED:
            raise ValueError(f"{name}: unsupported xclim method {self.method}")
        self.supported = SUPPORTED[self.method]
        self.scheduler = cfg.get("scheduler", "threads")

    @property
    def version(self) -> str:
        import xclim

        return xclim.__version__

    def _dist(self, job: Job):
        if self.method == "APP":
            return {"gamma": "gamma", "loglogistic": "fisk"}[job.dist], {"floc": 0}
        return _lmoments_dist(job.dist), {}

    def describe(self, job: Job) -> dict[str, str]:
        dist, _ = self._dist(job)
        name = dist if isinstance(dist, str) else f"lmoments3.{type(dist).__name__} ({dist.name})"
        method = (
            "approximate (APP), loc fixed at 0" if self.method == "APP" else "probability weighted moments (PWM)"
        )
        zero = " with probability of zero" if job.index == "spi" else ""
        return {"distribution": f"{name}{zero}", "fit_method": method}

    def _series(self, job: Job, inputs: Inputs) -> xr.DataArray:
        da = inputs.precip if job.index == "spi" else inputs.water_balance()
        da = da.chunk({"time": -1, "lat": self.chunk, "lon": self.chunk})
        # Monthly totals averaged over the window; standardization does not depend on
        # whether the window is a sum or a mean
        return da.rolling(time=job.scale).mean(skipna=False) if job.scale > 1 else da

    def _compute(self, obj):
        import dask

        with dask.config.set(scheduler=self.scheduler, num_workers=self.workers), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return obj.compute()

    def fit(self, job: Job, inputs: Inputs, calibration: Calibration) -> xr.Dataset:
        from xclim.indices.stats import standardized_index_fit_params

        dist, fitkwargs = self._dist(job)
        rolled = self._series(job, inputs)
        calib = rolled.sel(time=slice(f"{calibration.start_year}-01-01", f"{calibration.end_year}-12-31"))
        params = standardized_index_fit_params(
            calib,
            freq=None,
            window=1,
            dist=dist,
            method=self.method,
            zero_inflated=job.index == "spi",
            fitkwargs=fitkwargs,
        )
        params = self._compute(params)
        # NetCDF attributes cannot hold tuples; xclim reads these back as strings anyway
        attrs = {k: (json.dumps(v) if isinstance(v, (tuple, list)) else v) for k, v in params.attrs.items()}
        params.attrs = attrs
        return params.to_dataset(name="params")

    def apply(self, job: Job, inputs: Inputs, params: xr.Dataset, calibration: Calibration) -> xr.DataArray:
        from xclim.indices.stats import standardized_index

        if self.method == "PWM":
            _lmoments_dist(job.dist)  # make the stored distribution name resolvable
        p = params["params"]
        p.attrs = dict(params["params"].attrs)
        p.attrs["freq"], p.attrs["window"] = "", 1
        rolled = self._series(job, inputs)
        si = standardized_index(
            rolled,
            freq=None,
            window=None,
            dist=None,
            method=None,
            zero_inflated=None,
            fitkwargs=None,
            cal_start=None,
            cal_end=None,
            params=p.chunk({"lat": self.chunk, "lon": self.chunk}),
        )
        si = self._compute(si)
        si = si.drop_vars([c for c in si.coords if c not in ("time", "lat", "lon")])
        return si.transpose("time", "lat", "lon").astype("float32").rename(job.index)

