import numpy as np
import pytest
import xarray as xr

from drought_indices.config import Calibration
from drought_indices.engines import Job
from drought_indices.engines.base import clip_index
from drought_indices.engines.climate_indices_engine import ClimateIndicesEngine
from drought_indices.engines.xclim_engine import XclimEngine

CAL = Calibration(1991, 2020)
COMPUTE = {"chunk": 3}


def _ci():
    return ClimateIndicesEngine({"workers": 1}, COMPUTE)


def _xclim(method):
    return XclimEngine(f"xclim_{method.lower()}", {"method": method, "scheduler": "synchronous"}, COMPUTE)


@pytest.mark.parametrize("index,dist", [("spi", "gamma"), ("spi", "pearson3"), ("spei", "gamma"),
                                        ("spei", "pearson3"), ("spei", "loglogistic")])
def test_climate_indices_stored_params_match_one_shot(inputs, index, dist):
    """Fitting once and applying stored parameters must equal climate_indices' own fit."""
    from climate_indices import compute, indices

    engine, job = _ci(), Job(index, dist, 3)
    out = engine.apply(job, inputs, engine.fit(job, inputs, CAL), CAL)

    distribution = indices.Distribution({"gamma": "gamma", "pearson3": "pearson", "loglogistic": "loglogistic"}[dist])
    kwargs = dict(scale=3, distribution=distribution, data_start_year=1981, calibration_year_initial=1991,
                  calibration_year_final=2020, periodicity=compute.Periodicity.monthly)
    p = inputs.precip.values[:, 2, 3].astype("float64")
    if index == "spi":
        expected = indices.spi(p, **kwargs)
    else:
        expected = indices.spei(p, inputs.pet.values[:, 2, 3].astype("float64"), **kwargs)
    np.testing.assert_allclose(out.values[:, 2, 3], expected, atol=1e-3, equal_nan=True)


def test_ocean_pixel_stays_missing(inputs):
    engine, job = _ci(), Job("spi", "gamma", 1)
    out = engine.apply(job, inputs, engine.fit(job, inputs, CAL), CAL)
    assert out.isel(lat=0, lon=0).isnull().all()
    assert out.isel(lat=1, lon=1).notnull().sum() > 400


def test_xclim_app_stored_params_match_one_shot(inputs):
    """Our split fit/apply must equal xclim's own SPI with the same calibration period."""
    from xclim.indices.stats import standardized_index

    engine, job = _xclim("APP"), Job("spi", "gamma", 3)
    out = engine.apply(job, inputs, engine.fit(job, inputs, CAL), CAL)
    precip = inputs.precip.copy()
    precip.attrs["units"] = "mm/month"
    expected = standardized_index(
        precip, freq=None, window=3, dist="gamma", method="APP", zero_inflated=True, fitkwargs={"floc": 0},
        cal_start="1991-01-01", cal_end="2020-12-31",
    )
    xr.testing.assert_allclose(out, expected.transpose(*out.dims).astype("float32").rename("spi")
                               .drop_vars([c for c in expected.coords if c not in out.coords]), atol=1e-4)


@pytest.mark.parametrize("scale", [1, 6])
def test_engines_agree_on_gamma_spi(inputs, scale):
    """climate_indices gamma and xclim APP gamma use the same estimator, so they must agree
    closely; the remaining differences come from details such as float precision."""
    job = Job("spi", "gamma", scale)
    a = clip_index(_ci().apply(job, inputs, _ci().fit(job, inputs, CAL), CAL), 3.09)
    xe = _xclim("APP")
    b = clip_index(xe.apply(job, inputs, xe.fit(job, inputs, CAL), CAL), 3.09)
    valid = a.notnull() & b.notnull()
    assert valid.sum() > 1000
    assert float(np.abs(a - b).where(valid).max()) < 0.05


@pytest.mark.parametrize("index,dist", [("spi", "gamma"), ("spi", "pearson3"), ("spei", "pearson3"),
                                        ("spei", "loglogistic")])
def test_xclim_pwm_runs_and_correlates(inputs, index, dist):
    job = Job(index, dist, 3)
    xe = _xclim("PWM")
    out = clip_index(xe.apply(job, inputs, xe.fit(job, inputs, CAL), CAL), 3.09)
    ref = clip_index(_ci().apply(job, inputs, _ci().fit(job, inputs, CAL), CAL), 3.09)
    valid = out.notnull() & ref.notnull()
    assert valid.sum() > 1000
    assert float(xr.corr(out.where(valid), ref.where(valid))) > 0.97


def test_unsupported_combinations_are_reported():
    assert not _xclim("APP").supports(Job("spi", "pearson3", 1))
    assert not _ci().supports(Job("spi", "loglogistic", 1))
    assert _xclim("PWM").supports(Job("spei", "loglogistic", 1))


def test_palmer_and_pnp(inputs):
    engine = _ci()
    out = engine.extras(inputs, inputs, CAL, [1, 3], palmer=True)
    assert {"pnp_01", "pnp_03", "pdsi", "phdi", "pmdi", "zindex", "scpdsi", "palmer_qc"} <= set(out)
    assert set(np.unique(out["palmer_qc"].values[np.isfinite(out["palmer_qc"].values)])) <= {0.0, 1.0}
    pdsi = out["pdsi"].isel(lat=2, lon=2)
    assert pdsi.notnull().sum() > 400
    assert -10 < float(pdsi.min()) and float(pdsi.max()) < 10
    assert out["pdsi"].isel(lat=0, lon=0).isnull().all()


def test_block_pdsi_matches_single_pixel(inputs):
    from climate_indices import palmer

    out = _ci().extras(inputs, inputs, CAL, [1], palmer=True)
    p = inputs.precip.values[:, 3, 2].astype("float64") / 25.4
    e = inputs.pet.values[:, 3, 2].astype("float64") / 25.4
    expected = palmer.pdsi(p, e, 120.0 / 25.4, 1981, 1991, 2020)[0]
    np.testing.assert_allclose(out["pdsi"].values[:, 3, 2], expected, atol=1e-4, equal_nan=True)
