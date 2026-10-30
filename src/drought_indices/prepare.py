"""Model-ready inputs: monthly precipitation cubes plus PET and AWC on the same grid.

PET is computed once per source grid and shared by every engine, so differences between
engine outputs come only from the index computation itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine

from .config import Config
from .sources import aux

log = logging.getLogger(__name__)


# Precipitation -------------------------------------------------------------------------


def _same_grid(a: Affine, b: Affine) -> bool:
    """Equal up to float32 noise in the stored geotransform (0.1% of a pixel).

    CPC re-published ARC2 and RFE2 for 2020-2023 with an origin of 18.95 where the other
    years have 18.950001, which is the same grid.
    """
    return a.almost_equals(b, precision=1e-3 * min(abs(b.a), abs(b.e)))


def build_precip_cube(name: str, config: Config) -> Path:
    """Stack the clipped monthly GeoTIFFs of ``name`` into one NetCDF time series."""
    files = sorted(config.raw_dir(name).glob("*/*.tif"))
    if not files:
        raise FileNotFoundError(f"No downloaded months for {name}; run the download step first")

    arrays, times, transform, shape = [], [], None, None
    for path in files:
        with rasterio.open(path) as src:
            if transform is None:
                transform, shape = src.transform, (src.height, src.width)
            elif not _same_grid(src.transform, transform) or (src.height, src.width) != shape:
                raise ValueError(f"{path} is on a different grid from {files[0]}")
            arrays.append(src.read(1))
            times.append(pd.Timestamp(src.tags()["month"] + "-01"))

    height, width = shape
    lon = transform.c + transform.a * (np.arange(width) + 0.5)
    lat = transform.f + transform.e * (np.arange(height) + 0.5)
    da = xr.DataArray(
        np.stack(arrays),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat.round(6), "lon": lon.round(6)},
        name="precip",
        attrs={"units": "mm/month", "long_name": "monthly precipitation total", "source": name},
    )
    # Fill gaps in the monthly sequence with NaN so that rolling sums never span them
    full = pd.date_range(da.time.values[0], da.time.values[-1], freq="MS")
    da = da.sortby("time").reindex(time=full).sortby("lat")

    out = config.precip_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".nc.part")
    da.to_dataset().to_netcdf(
        tmp,
        encoding={"precip": {"zlib": True, "complevel": 4, "chunksizes": (1, height, width)}},
    )
    tmp.replace(out)
    log.info("%s: %d months, %dx%d pixels -> %s", name, da.sizes["time"], height, width, out)
    return out


def load_precip(name: str, config: Config) -> xr.DataArray:
    with xr.open_dataset(config.precip_path(name)) as ds:
        da = ds["precip"].load()
    factor = int(config["compute"].get("coarsen", 1))
    if factor > 1:
        # Block mean keeps mm/month units; a block with any missing pixel stays missing
        da = da.coarsen(lat=factor, lon=factor, boundary="trim").mean(skipna=False)
    return da


# PET -------------------------------------------------------------------------------------


def _thornthwaite(tmp: xr.DataArray) -> xr.DataArray:
    """Thornthwaite PET (mm/month) with the climate_indices implementation."""
    from climate_indices import eto

    start_year = int(tmp.time.dt.year[0])

    def per_pixel(temps: np.ndarray, lat: float) -> np.ndarray:
        if np.all(np.isnan(temps)):
            return temps
        return eto.eto_thornthwaite(temps, float(lat), start_year)

    lat = xr.broadcast(tmp.lat, tmp.isel(time=0))[0]
    return xr.apply_ufunc(
        per_pixel,
        tmp,
        lat,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        vectorize=True,
    ).transpose("time", "lat", "lon")


def _hargreaves(tmp: xr.DataArray, tmn: xr.DataArray, tmx: xr.DataArray) -> xr.DataArray:
    """FAO-56 Hargreaves PET (mm/month) from monthly mean, min and max temperature.

    Extraterrestrial radiation is evaluated on the 15th of each month (FAO-56 eqs. 21-25).
    """
    doy = xr.DataArray(
        (tmp.time + pd.Timedelta(days=14)).dt.dayofyear.values, dims="time", coords={"time": tmp.time}
    )
    phi = np.deg2rad(tmp.lat)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos((-np.tan(phi) * np.tan(delta)).clip(-1, 1))
    ra_mj = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    ra_mm = 0.408 * ra_mj
    daily = 0.0023 * ra_mm * (tmp + 17.8) * np.sqrt((tmx - tmn).clip(min=0))
    return (daily * tmp.time.dt.days_in_month).transpose("time", "lat", "lon")


def _open_aux(path: Path, variable: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        return ds[variable].load()


def pet_coarse(config: Config) -> xr.DataArray:
    """PET (mm/month) on the native grid of its inputs, cached in ``data/aux``."""
    method = config["pet"]["method"]
    temp_source = config["temperature"]["source"]
    tag = method if method != "thornthwaite" else f"thornthwaite_{temp_source}"
    out = config.aux_dir() / f"pet_{tag}.nc"
    if out.exists():
        return _open_aux(out, "pet")

    start_year = int(config["period"]["start"][:4])
    if method == "cru_penman_monteith":
        pet_daily = _open_aux(aux.fetch_cru("pet", config, start_year), "pet")
        pet = pet_daily * pet_daily.time.dt.days_in_month
        source = "CRU TS FAO-56 Penman-Monteith PET"
    elif method == "thornthwaite":
        if temp_source == "cru":
            tmp = _open_aux(aux.fetch_cru("tmp", config, start_year), "tmp")
        elif temp_source == "era5_land":
            tmp = _open_aux(aux.fetch_era5_land_t2m(config, start_year), "tmp")
        else:
            raise ValueError(f"Unknown temperature.source {temp_source!r}")
        pet = _thornthwaite(tmp)
        source = f"Thornthwaite (climate_indices) from {temp_source} temperature"
    elif method == "hargreaves":
        tmp, tmn, tmx = (
            _open_aux(aux.fetch_cru(v, config, start_year), v) for v in ("tmp", "tmn", "tmx")
        )
        pet = _hargreaves(tmp, tmn, tmx)
        source = "FAO-56 Hargreaves from CRU TS tmp/tmn/tmx"
    else:
        raise ValueError(f"Unknown pet.method {method!r}")

    pet = pet.astype("float32").rename("pet")
    pet.attrs = {"units": "mm/month", "long_name": "potential evapotranspiration", "source": source}
    pet.to_dataset().to_netcdf(out)
    return pet


def pet_on_grid(config: Config, precip: xr.DataArray) -> xr.DataArray:
    pet = aux.regrid(pet_coarse(config), precip)
    # Only keep PET where precipitation exists, so both inputs share one land mask
    pet = pet.where(precip.notnull().any("time"))
    pet.attrs = {"units": "mm/month", "long_name": "potential evapotranspiration"}
    return pet.rename("pet")


def common_period(*arrays: xr.DataArray) -> list[xr.DataArray]:
    """Trim arrays to the months they all cover (PET usually ends before precipitation)."""
    start = max(a.time.values[0] for a in arrays)
    end = min(a.time.values[-1] for a in arrays)
    return [a.sel(time=slice(start, end)) for a in arrays]
