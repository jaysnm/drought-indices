"""climate_indices (NOAA NCEI reference implementation, v3 with native kernels).

Distributions:
  gamma        2-parameter gamma, Thom (1958) approximate maximum likelihood,
               with the probability of zero as a mixed distribution (NOAA/NCEI SPI)
  pearson3     Pearson Type III by L-moments, with probability of zero
  loglogistic  generalized logistic by unbiased PWM / L-moments: the "log-Logistic"
               of R's SPEI package and SPEIbase (SPEI only)

Also the Palmer family (PDSI, PHDI, PMDI, Z-Index, scPDSI) and percent of normal.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import xarray as xr

from ..config import Calibration
from .base import SPEI_OFFSET_MM, Engine, Inputs, Job, map_tiles

log = logging.getLogger(__name__)

MM_PER_INCH = 25.4

PARAM_NAMES = {
    "gamma": ("alpha", "beta", "prob_zero"),
    "pearson3": ("prob_zero", "loc", "scale", "skew"),
    "loglogistic": ("loc", "scale", "shape"),
}

_DISTRIBUTIONS = {"gamma": "gamma", "pearson3": "pearson", "loglogistic": "loglogistic"}


def _quiet() -> None:
    """climate_indices logs every fit through structlog; keep worker output readable."""
    logging.getLogger("climate_indices").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Goodness-of-fit warnings repeat for every tile of a large grid
    warnings.filterwarnings("ignore", message=".*poor goodness-of-fit.*")


def _series(precip: np.ndarray, pet: np.ndarray | None, index: str) -> tuple[np.ndarray, bool]:
    """Values to scale, and whether negative values must be clipped first."""
    precip = precip.astype("float64")
    if index == "spei":
        return np.clip(precip, 0, None) - pet.astype("float64") + SPEI_OFFSET_MM, False
    return precip, True


def _fit_tile(
    precip: np.ndarray,
    pet: np.ndarray | None = None,
    *,
    index: str,
    dist: str,
    scale: int,
    start_year: int,
    cal: tuple[int, int],
) -> dict[str, np.ndarray]:
    from climate_indices import compute

    _quiet()
    monthly = compute.Periodicity.monthly
    values, clip_negatives = _series(precip, pet, index)
    folded = compute.prepare_scaled(
        values, scale, monthly, clip_negatives=clip_negatives, spatial_time_major=True
    )
    if folded.ndim != 4:  # all missing
        return {}
    args = (start_year, cal[0], cal[1], monthly)
    if dist == "gamma":
        alpha, beta = compute.gamma_parameters(folded, *args)
        rows = slice(cal[0] - start_year, cal[1] - start_year + 1)
        calib = folded[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_zero = (calib == 0).sum(axis=0) / np.isfinite(calib).sum(axis=0)
        return {"alpha": alpha, "beta": beta, "prob_zero": prob_zero}
    if dist == "pearson3":
        return dict(zip(PARAM_NAMES[dist], compute.pearson_parameters(folded, *args)))
    if dist == "loglogistic":
        return dict(zip(PARAM_NAMES[dist], compute.loglogistic_parameters(folded, *args)))
    raise ValueError(dist)


def _apply_tile(
    precip: np.ndarray,
    pet: np.ndarray | None = None,
    *,
    index: str,
    dist: str,
    scale: int,
    start_year: int,
    cal: tuple[int, int],
    **params: np.ndarray,
) -> dict[str, np.ndarray]:
    from climate_indices import compute, indices

    _quiet()
    distribution = indices.Distribution(_DISTRIBUTIONS[dist])
    # parameter arrays arrive prefixed so that Pearson's "scale" cannot clash with the
    # accumulation scale argument
    fitting_params = {k.removeprefix("param_"): v.astype("float64") for k, v in params.items()}
    common = dict(
        scale=scale,
        distribution=distribution,
        data_start_year=start_year,
        calibration_year_initial=cal[0],
        calibration_year_final=cal[1],
        periodicity=compute.Periodicity.monthly,
        fitting_params=fitting_params,
        spatial_time_major=True,
    )
    if index == "spi":
        out = indices.spi(precip.astype("float64"), **common)
    else:
        out = indices.spei(precip.astype("float64"), pet.astype("float64"), **common)
    return {"index": out.astype("float32")}


def _palmer_tile(
    precip: np.ndarray, pet: np.ndarray, awc: np.ndarray, *, start_year: int, cal: tuple[int, int], scpdsi: bool
) -> dict[str, np.ndarray]:
    from climate_indices import palmer

    _quiet()
    p_in = precip.astype("float64") / MM_PER_INCH
    e_in = pet.astype("float64") / MM_PER_INCH
    a_in = awc.astype("float64") / MM_PER_INCH
    nt, ny, nx = p_in.shape
    out = {k: np.full((nt, ny, nx), np.nan, dtype="float32") for k in ("pdsi", "phdi", "pmdi", "zindex")}
    if scpdsi:
        out["scpdsi"] = np.full((nt, ny, nx), np.nan, dtype="float32")

    keys = ("pdsi", "phdi", "pmdi", "zindex")
    valid = np.isfinite(p_in).any(axis=0) & np.isfinite(e_in).any(axis=0)
    # Cells without data are skipped by climate_indices, but still need a finite AWC
    a_block = np.where(np.isfinite(a_in), a_in, np.nanmedian(a_in) if np.isfinite(a_in).any() else 4.0)

    # PDSI over the whole tile at once uses climate_indices' native kernels and is ~30x
    # faster than looping over pixels (identical results). If one pixel's record makes
    # the block fail, fall back to pixel by pixel so it cannot fail its neighbours.
    try:
        block = palmer.pdsi(p_in, e_in, a_block, start_year, cal[0], cal[1], spatial_time_major=True)
        for key, value in zip(keys, block[:4]):
            out[key][:] = np.where(valid, value, np.nan)
    except Exception as exc:  # noqa: BLE001
        log.debug("Block PDSI failed (%s); computing pixel by pixel", exc)
        for j, i in zip(*np.nonzero(valid)):
            try:
                result = palmer.pdsi(p_in[:, j, i], e_in[:, j, i], float(a_block[j, i]), start_year, cal[0], cal[1])
                for key, value in zip(keys, result[:4]):
                    out[key][:, j, i] = value
            except Exception as pixel_exc:  # noqa: BLE001
                log.debug("PDSI failed at tile pixel (%d, %d): %s", j, i, pixel_exc)

    # scPDSI has no block form: one recursion per pixel
    if scpdsi:
        for j, i in zip(*np.nonzero(valid)):
            try:
                out["scpdsi"][:, j, i] = palmer.scpdsi(
                    p_in[:, j, i], e_in[:, j, i], float(a_block[j, i]), start_year, cal[0], cal[1]
                )[0]
            except Exception as exc:  # noqa: BLE001
                log.debug("scPDSI failed at tile pixel (%d, %d): %s", j, i, exc)
    return out


def _pnp_tile(precip: np.ndarray, *, scale: int, start_year: int, cal: tuple[int, int]) -> dict[str, np.ndarray]:
    from climate_indices import compute, indices

    _quiet()
    out = indices.percentage_of_normal(
        precip.astype("float64"), scale, start_year, cal[0], cal[1], compute.Periodicity.monthly,
        spatial_time_major=True,
    )
    return {"pnp": np.asarray(out, dtype="float32")}


class ClimateIndicesEngine(Engine):
    name = "climate_indices"
    supported = {
        "spi": ("gamma", "pearson3"),
        "spei": ("gamma", "pearson3", "loglogistic"),
    }

    @property
    def version(self) -> str:
        import climate_indices

        return climate_indices.__version__

    def describe(self, job: Job) -> dict[str, str]:
        methods = {
            "gamma": ("gamma (2-parameter) with probability of zero", "Thom approximate maximum likelihood"),
            "pearson3": ("Pearson Type III with probability of zero", "L-moments"),
            "loglogistic": ("generalized logistic (R SPEI 'log-Logistic')", "unbiased PWM / L-moments"),
        }
        dist, method = methods[job.dist]
        return {"distribution": dist, "fit_method": method}

    def _arrays(self, job: Job, inputs: Inputs) -> dict[str, np.ndarray]:
        arrays = {"precip": inputs.precip.values}
        if job.index == "spei":
            arrays["pet"] = inputs.pet.values
        return arrays

    def fit(self, job: Job, inputs: Inputs, calibration: Calibration) -> xr.Dataset:
        out = map_tiles(
            _fit_tile,
            self._arrays(job, inputs),
            self.chunk,
            self.workers,
            index=job.index,
            dist=job.dist,
            scale=job.scale,
            start_year=inputs.start_year,
            cal=(calibration.start_year, calibration.end_year),
        )
        coords = {"month": np.arange(1, 13), "lat": inputs.precip.lat, "lon": inputs.precip.lon}
        return xr.Dataset({k: (("month", "lat", "lon"), out[k]) for k in PARAM_NAMES[job.dist]}, coords=coords)

    def apply(self, job: Job, inputs: Inputs, params: xr.Dataset, calibration: Calibration) -> xr.DataArray:
        arrays = self._arrays(job, inputs)
        arrays.update({f"param_{k}": params[k].values for k in PARAM_NAMES[job.dist]})
        out = map_tiles(
            _apply_tile,
            arrays,
            self.chunk,
            self.workers,
            index=job.index,
            dist=job.dist,
            scale=job.scale,
            start_year=inputs.start_year,
            cal=(calibration.start_year, calibration.end_year),
        )
        return xr.DataArray(out["index"], coords=inputs.precip.coords, dims=inputs.precip.dims, name=job.index)

    has_extras = True

    def extras(
        self, spi_inputs: Inputs, wb_inputs: Inputs | None, calibration: Calibration, scales: list[int], palmer: bool
    ) -> dict[str, xr.DataArray]:
        cal = (calibration.start_year, calibration.end_year)
        results: dict[str, xr.DataArray] = {}

        precip = spi_inputs.precip
        for scale in scales:
            out = map_tiles(
                _pnp_tile, {"precip": precip.values}, self.chunk, self.workers,
                scale=scale, start_year=spi_inputs.start_year, cal=cal,
            )
            results[f"pnp_{scale:02d}"] = xr.DataArray(out["pnp"], coords=precip.coords, dims=precip.dims, name="pnp")

        if palmer:
            inputs = wb_inputs
            if inputs is None or inputs.pet is None or inputs.awc is None:
                raise ValueError("Palmer indices need PET and AWC")
            coords, dims = inputs.precip.coords, inputs.precip.dims
            out = map_tiles(
                _palmer_tile,
                {"precip": inputs.precip.values, "pet": inputs.pet.values, "awc": inputs.awc.values},
                # Palmer is per pixel, so small tiles balance the work across processes
                max(10, self.chunk // 4),
                self.workers,
                start_year=inputs.start_year,
                cal=cal,
                scpdsi=bool(self.cfg.get("scpdsi", True)),
            )
            for key, value in out.items():
                results[key] = xr.DataArray(value, coords=coords, dims=dims, name=key)
            results["palmer_qc"] = palmer_qc(results["pdsi"], results["zindex"])
        return results


# PDSI is designed to stay within about +/-10. Outside that, Palmer's CAFEC coefficients
# have broken down: in strongly seasonal climates the mean potential runoff (or recharge,
# or loss) of a calendar month can be almost zero over the calibration period, so the
# coefficient (e.g. gamma = mean RO / mean PRO) explodes and the Z-index of any month with
# real runoff becomes absurd. The values are left as climate_indices returns them, and
# the affected pixels are flagged.
PDSI_QC_LIMIT = 15.0


def palmer_qc(pdsi: xr.DataArray, zindex: xr.DataArray) -> xr.DataArray:
    suspect = ((abs(pdsi) > PDSI_QC_LIMIT) | (abs(zindex) > 3 * PDSI_QC_LIMIT)).any("time")
    suspect = suspect.where(pdsi.notnull().any("time")).astype("float32")
    count = int((suspect == 1).sum())
    if count:
        log.warning(
            "Palmer: %d of %d pixels exceed |PDSI| > %g (degenerate CAFEC coefficients); "
            "flagged in palmer_qc", count, int(suspect.notnull().sum()), PDSI_QC_LIMIT,
        )
    suspect.attrs = {
        "long_name": "Palmer indices unreliable at this pixel",
        "flag_values": "0, 1",
        "flag_meanings": "ok suspect",
        "comment": f"1 where |PDSI| > {PDSI_QC_LIMIT:g} or |Z-index| > {3 * PDSI_QC_LIMIT:g} in any month, "
        "the signature of a degenerate CAFEC coefficient (near-zero calibration mean of a "
        "potential water-balance term)",
    }
    return suspect.rename("palmer_qc")
