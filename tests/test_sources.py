import datetime as dt

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from affine import Affine

from drought_indices.config import BBox
from drought_indices.prepare import _hargreaves
from drought_indices.raster import Clipped, RemoteNotFound, read_clipped, write_geotiff
from drought_indices.sources import precip as precip_mod
from drought_indices.sources.precip import DailyPrecipSource, month_range

ICPAC = BBox(19.0, -14.0, 54.0, 26.0)


@pytest.fixture
def africa_tif(tmp_path):
    """A CHIRPS-like 0.05 deg Africa raster (-20..55E, -40..40N) with a -9999 fill."""
    width, height = 1500, 1600
    transform = Affine(0.05, 0, -20.0, 0, -0.05, 40.0)
    data = np.full((height, width), 7.5, dtype="float32")
    data[:10, :10] = -9999
    path = tmp_path / "africa.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1, dtype="float32",
                       crs="EPSG:4326", transform=transform, nodata=-9999) as dst:
        dst.write(data, 1)
    return path


def test_clip_snaps_to_bbox(africa_tif):
    clipped = read_clipped(str(africa_tif), ICPAC)
    assert clipped.data.shape == (800, 700)
    assert clipped.transform.c == pytest.approx(19.0)
    assert clipped.transform.f == pytest.approx(26.0)
    assert np.all(clipped.data == 7.5)


def test_clip_marks_fill_values_missing(africa_tif):
    clipped = read_clipped(str(africa_tif), BBox(-20.0, 39.0, -19.0, 40.0))
    assert np.isnan(clipped.data[:10, :10]).all()


def test_missing_remote_file_is_reported(tmp_path):
    with pytest.raises(RemoteNotFound):
        read_clipped(str(tmp_path / "nope.tif"), ICPAC)


def test_transient_remote_error_is_retried(monkeypatch, africa_tif):
    import drought_indices.raster as raster_mod

    real_open = rasterio.open
    calls = []

    def flaky_open(path, *args, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise rasterio.errors.RasterioIOError("TIFFReadDirectory:Failed to read directory")
        return real_open(africa_tif, *args, **kwargs)

    monkeypatch.setattr(raster_mod.rasterio, "open", flaky_open)
    monkeypatch.setattr(raster_mod, "READ_RETRY_DELAY", 0)
    clipped = read_clipped("/vsizip//vsicurl/https://x/a.tif.zip/a.tif", ICPAC)
    assert len(calls) == 2
    assert clipped.data.shape == (800, 700)


def test_curl_prefix_drops_archive_member():
    from drought_indices.raster import _curl_prefix

    assert _curl_prefix("/vsizip//vsicurl/https://x/a.tif.zip/a.tif") == "/vsicurl/https://x/a.tif.zip"
    assert _curl_prefix("/vsicurl/https://x/b.tif") == "/vsicurl/https://x/b.tif"


def test_write_geotiff_roundtrip(tmp_path, africa_tif):
    clipped = read_clipped(str(africa_tif), ICPAC)
    out = write_geotiff(clipped, tmp_path / "x" / "m.tif", month="2020-01")
    with rasterio.open(out) as src:
        assert src.tags()["month"] == "2020-01"
        assert src.bounds.left == pytest.approx(19.0)


def _fake_daily(missing_days=()):
    def fake_read(path, bbox):
        day = int(path.split("africa_rfe.")[1][6:8])
        if day in missing_days:
            raise RemoteNotFound(path)
        return Clipped(np.full((2, 2), float(day), dtype="float32"), Affine(0.1, 0, 19, 0, -0.1, 26), None)

    return fake_read


def _rfe(tmp_path, **cfg):
    url = "https://example.org/africa_rfe.{date:%Y%m%d}.tif.zip"
    return DailyPrecipSource("rfe2", {"url": url, **cfg}, ICPAC, tmp_path)


def test_daily_source_tries_alternative_zip_members(tmp_path, monkeypatch):
    real = _fake_daily()

    def fake_read(path, bbox):
        if not path.endswith("/clim.tif"):
            raise RemoteNotFound(path)
        return real(path.split(".tif.zip")[0] + ".tif.zip", bbox)

    monkeypatch.setattr(precip_mod, "read_clipped", fake_read)
    source = _rfe(tmp_path, zip_members=["clim.tif"])
    clipped, tags = source.read_month(dt.date(2020, 2, 1))
    assert tags["missing_days"] == ""
    assert clipped.data[0, 0] == sum(range(1, 30))


def test_daily_source_sums_month(tmp_path, monkeypatch):
    monkeypatch.setattr(precip_mod, "read_clipped", _fake_daily())
    clipped, tags = _rfe(tmp_path).read_month(dt.date(2020, 2, 1))
    assert clipped.data[0, 0] == sum(range(1, 30))  # leap February
    assert tags["days_used"] == "29"


def test_daily_source_incomplete_old_month_is_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(precip_mod, "read_clipped", _fake_daily(missing_days={5}))
    clipped, tags = _rfe(tmp_path, max_missing_days=0).read_month(dt.date(2010, 1, 1))
    assert np.isnan(clipped.data).all()
    assert tags["missing_days"] == "05"


def test_daily_source_tolerates_configured_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(precip_mod, "read_clipped", _fake_daily(missing_days={5}))
    clipped, _ = _rfe(tmp_path, max_missing_days=1).read_month(dt.date(2010, 1, 1))
    assert clipped.data[0, 0] == sum(range(1, 32)) - 5


def test_month_range():
    months = month_range(dt.date(2019, 11, 1), dt.date(2020, 2, 1))
    assert [m.isoformat() for m in months] == ["2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01"]


def test_hargreaves_plausible_for_nairobi():
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    coords = {"time": time, "lat": [-1.3], "lon": [36.8]}

    def field(value):
        return xr.DataArray(np.full((12, 1, 1), value), coords=coords, dims=("time", "lat", "lon"))

    pet = _hargreaves(field(19.0), field(12.0), field(26.0))
    # FAO-56 Hargreaves gives roughly 4-5.5 mm/day near the equator at these temperatures
    daily = pet / pet.time.dt.days_in_month
    assert 3.5 < float(daily.min()) and float(daily.max()) < 6.0
