"""Reading remote rasters clipped to a bounding box, and writing clipped GeoTIFFs.

Remote files are opened through GDAL's virtual file systems (/vsicurl/, /vsigzip/,
/vsizip/), so only the bytes needed for the bounding box are fetched whenever the file
format allows HTTP range requests (plain or zipped GeoTIFFs). Gzipped files have to be
streamed, but they are decompressed on the fly and never written to disk in full.
"""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from .config import BBox

log = logging.getLogger(__name__)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "GDAL_HTTP_TIMEOUT": "120",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.gz,.zip",
    "VSI_CACHE": "TRUE",
    # RFE2/ARC2 declare a spherical datum (EPSG:4035) that disagrees with its registry
    # entry; take the registry definition rather than warning on every file
    "GTIFF_SRS_SOURCE": "EPSG",
    # ...and keep that code as-is: EPSG:4035 is deprecated, and swapping it for EPSG:4047
    # also warns on every file. The source CRS is replaced by OUTPUT_CRS anyway
    "OSR_USE_NON_DEPRECATED": "NO",
}

# Every product is a regular lat/lon grid; the datum differences (WGS84 vs a sphere) are
# far below the pixel size, so all outputs are labelled WGS84 for consistency
OUTPUT_CRS = rasterio.crs.CRS.from_epsg(4326)

# Substrings in GDAL error messages that mean "this file does not exist (yet)"
_NOT_FOUND_MARKERS = ("404", "does not exist", "No such file", "not recognized as")

# Other read errors (truncated or corrupt responses, e.g. "TIFFReadDirectory: Failed to
# read directory") are usually transient, so the read is retried before giving up
READ_RETRIES = 4
READ_RETRY_DELAY = 10


class RemoteNotFound(Exception):
    """The remote file is not published (HTTP 404 or equivalent)."""


@dataclass
class Clipped:
    data: np.ndarray  # float32, NaN where missing
    transform: Affine
    crs: rasterio.crs.CRS


def snapped_window(src: rasterio.io.DatasetReader, bbox: BBox) -> Window:
    """Window covering ``bbox``, expanded outwards to whole pixels and limited to the raster."""
    t = src.transform
    res_x, res_y = t.a, -t.e
    # Resolutions are often stored as float32 (CHIRPS: 0.0500000007), which puts a bbox
    # edge a hair off a pixel edge; a tolerance of 0.1% of a pixel absorbs that noise
    eps = 1e-3
    col0 = math.floor((bbox.west - t.c) / res_x + eps)
    col1 = math.ceil((bbox.east - t.c) / res_x - eps)
    row0 = math.floor((t.f - bbox.north) / res_y + eps)
    row1 = math.ceil((t.f - bbox.south) / res_y - eps)
    col0, row0 = max(col0, 0), max(row0, 0)
    col1, row1 = min(col1, src.width), min(row1, src.height)
    if col1 <= col0 or row1 <= row0:
        raise ValueError(f"bbox {bbox} does not intersect raster bounds {src.bounds}")
    return Window(col0, row0, col1 - col0, row1 - row0)


def read_clipped(path: str, bbox: BBox, valid_min: float | None = 0.0) -> Clipped:
    """Read band 1 of ``path`` inside ``bbox`` as float32 with NaN for nodata.

    Values below ``valid_min`` are treated as missing as well, because some precipitation
    rasters mark ocean or missing cells with negative fill values that are not declared
    as nodata.
    """
    for attempt in range(1, READ_RETRIES + 1):
        env = dict(GDAL_ENV)
        if attempt > 1:
            # Fetch the bytes again instead of reusing what GDAL cached from the bad read
            env["CPL_VSIL_CURL_NON_CACHED"] = _curl_prefix(path)
        try:
            with rasterio.Env(**env), rasterio.open(path) as src:
                window = snapped_window(src, bbox)
                band = src.read(1, window=window, masked=True)
                transform = src.window_transform(window)
            break
        except RasterioIOError as exc:
            if any(marker in str(exc) for marker in _NOT_FOUND_MARKERS):
                raise RemoteNotFound(path) from exc
            if attempt == READ_RETRIES or "/vsicurl/" not in path:
                raise
            log.warning("read %s failed (%s), retry %d/%d", path, exc, attempt, READ_RETRIES)
            time.sleep(READ_RETRY_DELAY * attempt)

    data = band.astype("float32").filled(np.nan)
    if valid_min is not None:
        data[data < valid_min] = np.nan
    return Clipped(data, transform, OUTPUT_CRS)


def write_geotiff(clipped: Clipped, out_path: Path, **tags: str) -> Path:
    """Write atomically, so an interrupted run never leaves a half-written file behind."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    height, width = clipped.data.shape
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": height,
        "width": width,
        "crs": clipped.crs,
        "transform": clipped.transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(clipped.data, 1)
        if tags:
            dst.update_tags(**tags)
    os.replace(tmp, out_path)
    return out_path


def vsi_url(url: str, member: str | None = None) -> str:
    """GDAL path for a remote URL, unwrapping .gz and .zip archives.

    The file read from a zip is ``member``, by default the archive name without ``.zip``.
    """
    if url.endswith(".gz"):
        return f"/vsigzip//vsicurl/{url}"
    if url.endswith(".zip"):
        inner = member or Path(url).name[: -len(".zip")]
        return f"/vsizip//vsicurl/{url}/{inner}"
    return f"/vsicurl/{url}"


def _curl_prefix(path: str) -> str:
    """The ``/vsicurl/<url>`` part of a GDAL path, without any archive member after it."""
    match = re.search(r"/vsicurl/.+?\.(?:zip|gz)(?=/|$)", path) or re.search(r"/vsicurl/.+", path)
    return match.group(0) if match else path
