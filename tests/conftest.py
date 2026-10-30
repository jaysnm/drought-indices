import numpy as np
import pandas as pd
import pytest
import xarray as xr

from drought_indices.engines import Inputs


def _synthetic(years: int = 40, ny: int = 6, nx: int = 5, seed: int = 0) -> Inputs:
    """Seasonal gamma-distributed rainfall with dry-season zeros, PET and AWC."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("1981-01-01", periods=years * 12, freq="MS")
    month = time.month.values
    seasonal_mean = 20 + 120 * np.clip(np.sin((month - 2) / 12 * 2 * np.pi), 0, None)
    shape = (len(time), ny, nx)
    precip = rng.gamma(2.0, seasonal_mean[:, None, None] / 2.0, size=shape)
    dry = (month[:, None, None] >= 6) & (month[:, None, None] <= 8)
    precip = np.where(dry & (rng.random(shape) < 0.4), 0.0, precip)
    precip[:, 0, 0] = np.nan  # an ocean pixel
    pet = 120 + 30 * np.cos((month - 3) / 12 * 2 * np.pi)[:, None, None] + rng.normal(0, 5, shape)
    coords = {"time": time, "lat": np.linspace(-1, 1, ny), "lon": np.linspace(35, 37, nx)}
    p = xr.DataArray(precip.astype("float32"), coords=coords, dims=("time", "lat", "lon"), name="precip")
    e = xr.DataArray(pet.astype("float32"), coords=coords, dims=("time", "lat", "lon"), name="pet")
    e = e.where(p.notnull().any("time"))
    awc = xr.full_like(p.isel(time=0, drop=True), 120.0)
    return Inputs(p, e, awc)


@pytest.fixture(scope="session")
def inputs() -> Inputs:
    return _synthetic()
