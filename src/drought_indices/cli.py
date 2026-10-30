"""Command line interface.

    drought-indices download [--sources chirps_v3 arc2] [--refresh-last 2]
    drought-indices prepare  [--sources ...]
    drought-indices compute  [--sources ...] [--engines climate_indices xclim_app] [--refit]
    drought-indices run      # download + prepare + compute
    drought-indices update   # monthly: fetch new months, re-fetch the last two, apply stored parameters
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from . import pipeline, prepare
from .config import load_config, parse_month
from .sources import precip

log = logging.getLogger("drought_indices")


def _download(config, sources, refresh_last):
    for name in sources:
        log.info("Downloading %s", name)
        precip.download(name, config, refresh_last=refresh_last)


def _prepare(config, sources):
    for name in sources:
        prepare.build_precip_cube(name, config)


def _compute(config, sources, engines, refit):
    for name in sources:
        log.info("Computing indices for %s", name)
        pipeline.compute_indices(name, config, engines=engines, refit=refit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drought-indices", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="path to config.json (default: $DROUGHT_INDICES_CONFIG or config/config.json)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, engines=False):
        p.add_argument("--sources", nargs="+", help="default: every enabled source")
        p.add_argument("--start", help="first month, YYYY-MM (overrides period.start)")
        p.add_argument("--end", help="last month, YYYY-MM (overrides period.end)")
        if engines:
            p.add_argument("--engines", nargs="+", help="default: every enabled engine")
            p.add_argument("--refit", action="store_true", help="refit parameters even if stored ones match")

    p = sub.add_parser("download", help="fetch and clip monthly rainfall")
    common(p)
    p.add_argument("--refresh-last", type=int, default=0, help="re-fetch the last N months (provider revisions)")
    common(sub.add_parser("prepare", help="stack downloaded months into NetCDF cubes"))
    common(sub.add_parser("compute", help="fit/apply drought indices"), engines=True)
    p = sub.add_parser("run", help="download + prepare + compute")
    common(p, engines=True)
    p.add_argument("--refresh-last", type=int, default=0)
    p = sub.add_parser("update", help="monthly update reusing stored parameters")
    common(p, engines=True)
    p.add_argument("--refresh-last", type=int, default=2)

    args = parser.parse_args(argv)
    # climate_indices replaces the root logging handlers when it is imported, so import
    # it first and then take the root configuration back
    import climate_indices  # noqa: F401

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    for noisy in ("rasterio", "urllib3", "fiona", "distributed", "climate_indices"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # xclim redefines some pint units on import
    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="Import\\(s\\) unavailable to set up matplotlib")

    config = load_config(args.config)
    if args.start:
        config.raw["period"]["start"] = str(parse_month(args.start))[:7]
    if args.end:
        config.raw["period"]["end"] = str(parse_month(args.end))[:7]
    sources = args.sources or config.enabled_sources()
    unknown = set(sources) - set(config["sources"])
    if unknown:
        parser.error(f"unknown sources: {sorted(unknown)}")

    engines = getattr(args, "engines", None)
    refit = getattr(args, "refit", False)
    if args.command == "download":
        _download(config, sources, args.refresh_last)
    elif args.command == "prepare":
        _prepare(config, sources)
    elif args.command == "compute":
        _compute(config, sources, engines, refit)
    elif args.command in ("run", "update"):
        _download(config, sources, args.refresh_last)
        _prepare(config, sources)
        _compute(config, sources, engines, refit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
