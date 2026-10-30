"""Index computation: fit (or reuse) parameters, apply them, write outputs and metrics."""

from __future__ import annotations

import datetime as dt
import json
import logging
import resource
import time
from pathlib import Path

import xarray as xr

from . import prepare
from .config import Calibration, Config
from .engines import Engine, Inputs, Job, make_engine
from .engines.base import clip_index

log = logging.getLogger(__name__)

WMO_MIN_CALIBRATION_YEARS = 30


def jobs(config: Config) -> list[Job]:
    idx = config["indices"]
    return [
        Job(index, dist, scale)
        for index, dists in idx["distributions"].items()
        for dist in dists
        for scale in idx["scales"]
    ]


def build_inputs(source: str, config: Config, need_pet: bool, need_awc: bool) -> tuple[Inputs, Inputs | None]:
    """SPI inputs over the whole precipitation record, and SPEI/Palmer inputs limited to
    the months where PET is available too."""
    precip = prepare.load_precip(source, config)
    first_january = precip.time.where(precip.time.dt.month == 1, drop=True).values[0]
    precip = precip.sel(time=slice(first_january, None))
    spi_inputs = Inputs(precip)
    if not need_pet:
        return spi_inputs, None

    pet = prepare.pet_on_grid(config, precip)
    p, e = prepare.common_period(precip, pet)
    first_january = p.time.where(p.time.dt.month == 1, drop=True).values[0]
    p, e = p.sel(time=slice(first_january, None)), e.sel(time=slice(first_january, None))
    awc = None
    if need_awc:
        from .sources import aux

        awc = aux.awc_on_grid(config, precip).where(precip.notnull().any("time"))
    return spi_inputs, Inputs(p, e, awc)


def _output_attrs(config: Config, source: str, engine: Engine, calibration: Calibration) -> dict:
    years = calibration.end_year - calibration.start_year + 1
    return {
        "source": source,
        "engine": engine.name,
        "engine_version": engine.version,
        "calibration_period": f"{calibration.start_year}-{calibration.end_year}",
        "calibration_years": years,
        "wmo_compliant_calibration": int(years >= WMO_MIN_CALIBRATION_YEARS),
        "clip": config["indices"].get("clip") or "none",
        "pet_method": config["pet"]["method"],
        "bbox": json.dumps(list(config.bbox.as_tuple())),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _write(da: xr.DataArray, path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    da = da.copy()
    da.attrs = {**{k: v for k, v in da.attrs.items() if isinstance(v, (str, int, float))}, **attrs}
    tmp = path.with_suffix(".nc.part")
    da.to_dataset(name=da.name or "value").to_netcdf(tmp, encoding={da.name or "value": {"zlib": True, "complevel": 4}})
    tmp.replace(path)


def _params_valid(path: Path, calibration: Calibration) -> bool:
    if not path.exists():
        return False
    with xr.open_dataset(path) as ds:
        return ds.attrs.get("calibration_period") == f"{calibration.start_year}-{calibration.end_year}"


class Metrics:
    """Wall time, CPU time and peak memory of every step, for the compute benchmark."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    @staticmethod
    def _cpu() -> float:
        own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime

    def timed(self, engine: str, step: str, target: str, func, *args):
        cpu0, t0 = self._cpu(), time.perf_counter()
        result = func(*args)
        cpu1, t1 = self._cpu(), time.perf_counter()
        # ru_maxrss is a high-water mark in KiB on Linux: of this process, and of the
        # largest child process (the climate_indices tile workers)
        own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        self.rows.append(
            {
                "engine": engine,
                "step": step,
                "target": target,
                "wall_s": round(t1 - t0, 3),
                "cpu_s": round(cpu1 - cpu0, 3),
                "peak_rss_mb": round(own.ru_maxrss / 1024, 1),
                "peak_child_rss_mb": round(children.ru_maxrss / 1024, 1),
            }
        )
        log.info("%s %s %s: %.1fs", engine, step, target, t1 - t0)
        return result


def compute_indices(source: str, config: Config, engines: list[str] | None = None, refit: bool = False) -> Path:
    calibration = config.calibration_for(source)
    if calibration.end_year - calibration.start_year + 1 < WMO_MIN_CALIBRATION_YEARS:
        log.warning(
            "%s calibration %d-%d is shorter than the WMO-recommended %d years",
            source, calibration.start_year, calibration.end_year, WMO_MIN_CALIBRATION_YEARS,
        )

    all_jobs = jobs(config)
    palmer = bool(config["indices"].get("palmer"))
    engine_names = engines or config.enabled_engines()
    need_pet = palmer or any(j.index == "spei" for j in all_jobs)
    spi_inputs, wb_inputs = build_inputs(source, config, need_pet, need_awc=palmer and "climate_indices" in engine_names)
    clip = config["indices"].get("clip")
    metrics = Metrics()

    for name in engine_names:
        engine = make_engine(name, config)
        attrs = _output_attrs(config, source, engine, calibration)
        out_dir = config.indices_dir(source, name)
        params_dir = config.data_dir / "params" / source / name

        for job in all_jobs:
            if not engine.supports(job):
                log.info("%s does not support %s/%s; skipped", name, job.index, job.dist)
                continue
            inputs = spi_inputs if job.index == "spi" else wb_inputs
            params_path = params_dir / f"{job.stem}.nc"
            if refit or not _params_valid(params_path, calibration):
                params = metrics.timed(name, "fit", job.stem, engine.fit, job, inputs, calibration)
                params.attrs.update({**attrs, **engine.describe(job), "index": job.index, "scale": job.scale})
                params_path.parent.mkdir(parents=True, exist_ok=True)
                params.to_netcdf(params_path)
            else:
                with xr.open_dataset(params_path) as ds:
                    params = ds.load()

            index = metrics.timed(name, "apply", job.stem, engine.apply, job, inputs, params, calibration)
            index = clip_index(index, clip)
            index.attrs = {"long_name": f"{job.index.upper()}-{job.scale}", "units": "1"}
            _write(index, out_dir / f"{job.stem}.nc", {**attrs, **engine.describe(job), "scale": job.scale})

        if engine.has_extras:
            extras = metrics.timed(
                name, "apply", "palmer+pnp", engine.extras,
                spi_inputs, wb_inputs, calibration, config["indices"]["scales"], palmer,
            )
            for key, da in extras.items():
                _write(da, out_dir / f"{key}.nc", attrs)

        metrics_path = out_dir / "run_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r for r in metrics.rows if r["engine"] == name]
        grid = {"lat": spi_inputs.precip.sizes["lat"], "lon": spi_inputs.precip.sizes["lon"],
                "months": spi_inputs.precip.sizes["time"]}
        metrics_path.write_text(json.dumps({"source": source, "grid": grid, "steps": rows}, indent=2))
    return config.data_dir / "indices" / source
