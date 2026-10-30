"""Auxiliary inputs for SPEI and the Palmer indices: temperature, PET and soil AWC.

Sources:
  * CRU TS 4.x (University of East Anglia): monthly 0.5 deg pet (FAO-56 Penman-Monteith,
    mm/day), tmp, tmn, tmx (degC). Open access; the reference input of SPEIbase.
  * ERA5-Land monthly means (Copernicus CDS): 2 m temperature at 0.1 deg with ~1 month
    latency, for Thornthwaite PET when CRU's ~6-month release lag is too long. Optional,
    needs ``cdsapi`` and a CDS account.
  * SoilGrids 2.0 (ISRIC): water content at 33 kPa and 1500 kPa, for profile AWC.

Each input is cached under ``data/aux/`` already clipped to the bounding box.
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import requests
import xarray as xr

from ..config import BBox, Config

log = logging.getLogger(__name__)

T = TypeVar("T")


# Pause before retry n is RETRY_DELAY * n seconds; servers such as CRU's occasionally
# reset connections, and an immediate retry tends to hit the same condition
RETRY_DELAY = 10


def _with_retries(url: str, config: Config, fetch: Callable[[int], T]) -> T:
    """Call ``fetch(timeout)``, retrying on request errors with a growing delay."""
    dl = config["download"]
    retries, timeout = int(dl.get("retries", 4)), int(dl.get("timeout", 120))
    for attempt in range(1, retries + 1):
        try:
            return fetch(timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("GET %s failed (%s), retry %d/%d", url, exc, attempt, retries)
            time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError("unreachable")


def http_get_text(url: str, config: Config) -> str:
    """GET ``url`` as text with retries."""

    def fetch(timeout: int) -> str:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text

    return _with_retries(url, config, fetch)


def http_get(url: str, dest: Path, config: Config) -> Path:
    """Stream ``url`` to ``dest`` with retries; written atomically."""
    tmp = dest.with_suffix(dest.suffix + ".part")

    def fetch(timeout: int) -> Path:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as out:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        tmp.replace(dest)
        return dest

    return _with_retries(url, config, fetch)


def subset_bbox(da: xr.DataArray, bbox: BBox, margin: float = 0.0) -> xr.DataArray:
    """Select the bbox on a regular lat/lon grid of either latitude orientation."""
    lat_slice = (
        slice(bbox.south - margin, bbox.north + margin)
        if da.lat[0] < da.lat[-1]
        else slice(bbox.north + margin, bbox.south - margin)
    )
    return da.sel(lon=slice(bbox.west - margin, bbox.east + margin), lat=lat_slice)


def to_month_start(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    times = da[time_dim].values.astype("datetime64[M]").astype("datetime64[ns]")
    return da.assign_coords({time_dim: times})


# CRU TS ------------------------------------------------------------------------------


def cru_file_urls(variable: str, config: Config, start_year: int) -> list[str]:
    """Decadal CRU files that cover ``start_year`` onwards (smaller than the full record)."""
    cru = config["cru"]
    listing_url = f"{cru['base_url']}/{variable}/"
    html = http_get_text(listing_url, config)
    pattern = rf"cru_ts{re.escape(cru['version'])}\.(\d{{4}})\.(\d{{4}})\.{variable}\.dat\.nc\.gz"
    files = sorted(set(re.findall(pattern, html)))
    decadal = [(int(a), int(b)) for a, b in files if int(b) - int(a) < 50]
    chosen = [
        f"{listing_url}cru_ts{cru['version']}.{a}.{b}.{variable}.dat.nc.gz"
        for a, b in decadal
        if b >= start_year
    ]
    if not chosen:
        raise RuntimeError(f"No CRU TS {cru['version']} {variable} files found at {listing_url}")
    return chosen


def fetch_cru(variable: str, config: Config, start_year: int = 1981) -> Path:
    out = config.aux_dir() / f"cru_ts{config['cru']['version']}_{variable}.nc"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    pieces = []
    with tempfile.TemporaryDirectory(dir=out.parent) as tmpdir:
        for url in cru_file_urls(variable, config, start_year):
            log.info("CRU TS %s: %s", variable, url)
            gz = http_get(url, Path(tmpdir) / Path(url).name, config)
            nc = gz.with_suffix("")
            with gzip.open(gz, "rb") as src, open(nc, "wb") as dst:
                shutil.copyfileobj(src, dst)
            gz.unlink()
            with xr.open_dataset(nc) as ds:
                # A 1-cell margin lets the 0.5 deg grid be interpolated right up to the bbox edges
                pieces.append(subset_bbox(ds[variable], config.bbox, margin=0.5).load())
            nc.unlink()

    da = xr.concat(pieces, dim="time").sortby("time")
    da = da.sel(time=~da.get_index("time").duplicated())
    da = to_month_start(da)
    da = da.sel(time=slice(f"{start_year}-01-01", None))
    da.attrs["source"] = f"CRU TS {config['cru']['version']} ({config['cru']['base_url']})"
    da.to_dataset(name=variable).to_netcdf(out, encoding={variable: {"zlib": True, "complevel": 4}})
    return out


# ERA5-Land -----------------------------------------------------------------------------


def fetch_era5_land_t2m(config: Config, start_year: int = 1981, end_year: int | None = None) -> Path:
    out = config.aux_dir() / "era5_land_t2m.nc"
    if out.exists():
        return out
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "temperature.source=era5_land needs the 'era5' extra (uv sync --extra era5) "
            "and CDS API credentials (~/.cdsapirc, or CDSAPI_URL and CDSAPI_KEY)"
        ) from exc

    import datetime as dt

    end_year = end_year or dt.date.today().year
    b = config.bbox
    out.parent.mkdir(parents=True, exist_ok=True)
    yearly = []
    client = cdsapi.Client()
    for year in range(start_year, end_year + 1):
        part = out.parent / f"era5_land_t2m_{year}.nc"
        if not part.exists():
            client.retrieve(
                "reanalysis-era5-land-monthly-means",
                {
                    "product_type": ["monthly_averaged_reanalysis"],
                    "variable": ["2m_temperature"],
                    "year": [str(year)],
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "time": ["00:00"],
                    "area": [b.north + 0.5, b.west - 0.5, b.south - 0.5, b.east + 0.5],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                str(part),
            )
        yearly.append(part)

    das = []
    for path in yearly:
        with xr.open_dataset(path) as ds:
            da = ds["t2m"].load()
        rename = {"latitude": "lat", "longitude": "lon", "valid_time": "time"}
        da = da.rename({k: v for k, v in rename.items() if k in da.dims or k in da.coords})
        da = da.drop_vars([c for c in da.coords if c not in ("time", "lat", "lon")])
        das.append(da)
    t2m = to_month_start(xr.concat(das, dim="time").sortby("time")) - 273.15
    t2m.attrs = {"units": "degC", "source": "ERA5-Land monthly means (Copernicus CDS)"}
    t2m.to_dataset(name="tmp").to_netcdf(out)
    return out


# SoilGrids AWC -----------------------------------------------------------------------


def _wcs_layer(prop: str, layer: str, config: Config, dest: Path) -> xr.DataArray:
    soil, b = config["soil"], config.bbox
    res = float(soil["resolution"])
    params = {
        "SERVICE": "WCS",
        "VERSION": "2.0.1",
        "REQUEST": "GetCoverage",
        "COVERAGEID": f"{prop}_{layer}_mean",
        "FORMAT": "image/tiff",
        "SUBSETTINGCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "OUTPUTCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    query += (
        f"&SUBSET=long({b.west},{b.east})&SUBSET=lat({b.south},{b.north})"
        f"&RESOLUTION=long({res})&RESOLUTION=lat({res})"
    )
    url = soil["wcs_url"].format(property=prop) + "&" + query
    if not dest.exists():
        log.info("SoilGrids %s %s", prop, layer)
        http_get(url, dest, config)

    import rioxarray  # noqa: F401  (registers the .rio accessor)

    da = xr.open_dataarray(dest, engine="rasterio").squeeze("band", drop=True).load()
    nodata = da.rio.nodata
    da = da.where(da != nodata) if nodata is not None else da
    return da.rename({"x": "lon", "y": "lat"}).astype("float32")


def fetch_awc(config: Config) -> Path:
    """Profile AWC (mm) over the configured SoilGrids layers."""
    out = config.aux_dir() / "soilgrids_awc_mm.nc"
    if out.exists():
        return out
    tiles = config.aux_dir() / "soilgrids"
    tiles.mkdir(parents=True, exist_ok=True)

    total = None
    for layer in config["soil"]["layers"]:
        top, bottom = (int(v) for v in layer.removesuffix("cm").split("-"))
        thickness_mm = (bottom - top) * 10
        fc = _wcs_layer("wv0033", layer, config, tiles / f"wv0033_{layer}.tif")
        wp = _wcs_layer("wv1500", layer, config, tiles / f"wv1500_{layer}.tif")
        # SoilGrids water content is stored as 10^-3 cm3/cm3
        layer_mm = ((fc - wp).clip(min=0) / 1000.0) * thickness_mm
        total = layer_mm if total is None else total + layer_mm

    factor = int(config["soil"].get("aggregate", 1))
    if factor > 1:
        total = total.coarsen(lat=factor, lon=factor, boundary="trim").mean()
    total = total.drop_vars([c for c in total.coords if c not in ("lat", "lon")])
    total.attrs = {
        "units": "mm",
        "long_name": "profile available water capacity",
        "source": "SoilGrids 2.0 (ISRIC) wv0033 - wv1500, layers " + ",".join(config["soil"]["layers"]),
    }
    total.to_dataset(name="awc").to_netcdf(out)
    return out


def awc_on_grid(config: Config, target: xr.DataArray) -> xr.DataArray:
    """AWC (mm) interpolated to ``target``'s lat/lon, with the fallback where missing."""
    with xr.open_dataset(fetch_awc(config)) as ds:
        awc = ds["awc"].load()
    on_grid = awc.interp(lat=target.lat, lon=target.lon, method="linear")
    nearest = awc.interp(lat=target.lat, lon=target.lon, method="nearest")
    on_grid = on_grid.fillna(nearest).fillna(float(config["soil"]["fallback_mm"]))
    # Zero AWC (e.g. rock) breaks the Palmer water balance
    return on_grid.where(on_grid > 1.0, float(config["soil"]["fallback_mm"]))


def regrid(da: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation of a coarse lat/lon field to ``target``'s grid.

    Pixels that fall between coarse land and ocean cells are filled from the nearest
    coarse cell so the coastline does not lose pixels.
    """
    linear = da.interp(lat=target.lat, lon=target.lon, method="linear")
    nearest = da.interp(lat=target.lat, lon=target.lon, method="nearest")
    return linear.fillna(nearest)

