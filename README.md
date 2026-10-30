# drought-indices

Monthly rainfall for the 11 ICPAC member states from **CHIRPS v3**, **CHIRPS v2**, **ARC2**
and **RFE2**, clipped to the region as it is read. From that rainfall, SPI, SPEI, the Palmer
indices and percent of normal are computed with three engines:

| engine            | library                     | fitting                                                      |
|-------------------|-----------------------------|--------------------------------------------------------------|
| `climate_indices` | climate-indices 3.x (NOAA)  | gamma: Thom approx. ML · Pearson III / log-logistic: L-moments |
| `xclim_app`       | xclim, `method="APP"`       | closed-form approximate fit, `loc = 0`                         |
| `xclim_pwm`       | xclim, `method="PWM"`       | probability-weighted moments with lmoments3 distributions      |

The outputs share the same inputs, calibration period and clipping, so they can be
benchmarked against each other and against WMO and R CDT outputs.

## Quick start

```sh
uv sync                                   # or: pip install -e .
uv run drought-indices run                # download + prepare + compute, all sources
uv run drought-indices run --sources chirps_v3 --engines climate_indices xclim_app
uv run drought-indices update             # monthly: new months only, stored parameters
uv run pytest
```

`run` and `update` are resumable: months on disk are skipped, and stored parameters are
reused while their calibration period matches the config. `--refit` forces refitting.

## Docker

```sh
cp .env.example .env                                   # optional: user, paths, interval
docker compose build
docker compose run --rm pipeline                       # full run (download + prepare + compute)
docker compose run --rm pipeline update                # one monthly update
docker compose run --rm pipeline compute --sources arc2 --engines xclim_pwm
docker compose --profile scheduler up -d scheduler     # update every UPDATE_INTERVAL_HOURS (24)
docker compose logs -f scheduler
```

| service     | runs                                   | lifetime |
|-------------|----------------------------------------|----------|
| `pipeline`  | `drought-indices run`, or the command given after the service name | one-off |
| `scheduler` | `drought-indices update` at start and then every `UPDATE_INTERVAL_HOURS`; a failed update is retried at the next interval | `restart: unless-stopped`, profile `scheduler` |

Both services use the same image and mounts: `DATA_DIR` (default `./data`) at `/app/data`
and `CONFIG_FILE` (default `./config/config.json`) read-only at
`/app/config/config.json`. They run as `CONTAINER_USER` (default `1000:1000`), so with
rootful Docker the outputs belong to that host user. With rootless Docker or Podman, set
`CONTAINER_USER=0:0`: container root is the host user there, and any other uid cannot
write to the mount. Sources run one after another inside a container. Running
several containers at once would download the shared CRU and SoilGrids files twice and
compete for the same CPU cores. ERA5-Land needs the image built with
`UV_EXTRAS=--extra era5` and `CDSAPI_URL`/`CDSAPI_KEY` in `.env`.

## Configuration

`config/config.json`, or the file given by `--config` or `$DROUGHT_INDICES_CONFIG`.
Relative paths are resolved from the project root (the parent of `config/`).

| key | meaning |
|-----|---------|
| `bbox` | `[west, south, east, north]` in degrees (see below) |
| `data_dir` | where everything is written (default `data`) |
| `period.start`, `period.end` | first and last month, `YYYY-MM`. `end: "latest"` means the most recent month the source has published. `--start` and `--end` override them |
| `calibration.start_year`, `.end_year` | reference period for every fit (1991–2020) |
| `download.workers`, `.retries`, `.timeout` | parallel month downloads, retries per file, HTTP timeout (s) |
| `sources.<name>.enabled` | include the source by default |
| `sources.<name>.urls` / `.url` | URL templates, tried in order. Monthly sources use `{year}` and `{month:02d}`; daily sources use `{date:%Y%m%d}` |
| `sources.<name>.first_month` | first month the product exists |
| `sources.<name>.max_missing_days` | daily sources: a month missing more days than this is written as NaN instead of summed short |
| `sources.<name>.zip_members` | daily sources: other names to try for the file inside a zip, e.g. `daily_clim.{date:%Y%m%d}.tif` (ARC2 Jul 2024–Feb 2025) |
| `sources.<name>.day_workers` | daily sources: parallel day downloads per month (default 4) |
| `sources.<name>.calibration` | per-source override of `calibration` (RFE2: 2001–2020) |
| `pet.method` | `cru_penman_monteith` (CRU TS FAO-56 PM, as SPEIbase), `thornthwaite` or `hargreaves` |
| `temperature.source` | for Thornthwaite: `cru` (0.5°, ~6-month lag) or `era5_land` (0.1°, ~1-month lag, needs CDS credentials) |
| `cru.version`, `cru.base_url` | CRU TS release and its directory |
| `soil.wcs_url`, `.layers` | SoilGrids WCS and the 0–100 cm layers summed into AWC |
| `soil.resolution`, `.aggregate` | server sampling step (0.025°) and local block-mean factor (2 → 0.05°). The WCS samples nearest neighbour, so sampling finer and averaging is more representative |
| `soil.fallback_mm` | AWC where SoilGrids has no data (open water, rock, urban) |
| `indices.scales` | accumulation periods in months |
| `indices.distributions.spi`, `.spei` | distributions to compute; each engine computes those it supports |
| `indices.palmer` | compute PDSI, PHDI, PMDI, Z-Index and scPDSI (climate_indices) |
| `indices.clip` | every standardized index is clipped to ± this value (3.09) |
| `engines.<name>.enabled` | include the engine by default |
| `engines.climate_indices.scpdsi` | scPDSI is pixel by pixel and the slowest step (see below) |
| `engines.xclim_*.method` | `APP` or `PWM` |
| `engines.xclim_*.scheduler` | dask scheduler, `threads` (default), `processes` or `synchronous` |
| `engines.<name>.workers` | parallel workers (processes for climate_indices, dask workers for xclim); 0 = all cores |
| `compute.chunk` | tile size in pixels per side for parallel and dask work |
| `compute.coarsen` | block-mean the rainfall grid before computing (1 = native, 2 = 0.05° → 0.1°) |

## Data sources

| source      | product                                     | grid    | record        | access |
|-------------|---------------------------------------------|---------|---------------|--------|
| `chirps_v3` | CHC CHIRPS v3.0 monthly, Africa             | 0.05°   | 1981–present  | HTTP range reads of the GeoTIFF |
| `chirps_v2` | CHC CHIRPS v2.0 monthly, Africa             | 0.05°   | 1981–present  | streamed `.tif.gz` (plain `.tif` fallback) |
| `arc2`      | NOAA CPC ARC2 daily → monthly               | 0.1°    | 1983–present  | range reads inside the zip |
| `rfe2`      | NOAA CPC / FEWS NET RFE2.0 daily → monthly  | 0.1°    | 2001–present  | range reads inside the zip |

Each month is clipped to the bounding box while it is read, through GDAL's `/vsicurl/`,
`/vsizip/` and `/vsigzip/`, and the clipped window is all that is written to disk. Range
reads fetch only the bytes inside the box. The `.gz` files have to be streamed, but they
are decompressed on the fly. Daily sources are summed to monthly totals. A month with
missing days is written as NaN (`max_missing_days`) rather than summed short.

**Bounding box** `[19, -14, 54, 26]` (W, S, E, N). The member states span about
`[21.8, -11.8, 51.5, 23.2]`. The margin keeps border pixels clear of the edges when the
auxiliary data is regridded, and the edges are multiples of 0.5° so that the 0.05°, 0.1°
and 0.5° grids line up.

### Auxiliary data (SPEI and Palmer)

| input | default source | why |
|-------|----------------|-----|
| PET   | **CRU TS 4.10** FAO-56 Penman-Monteith PET, 0.5° | Open access, and the PET used by SPEIbase. Alternatives: `thornthwaite` (CRU or ERA5-Land temperature, through climate_indices) or `hargreaves` (CRU tmin/tmax). |
| temperature | CRU TS 4.10 `tmp` | Used only by Thornthwaite. `era5_land` (0.1°, ~1-month lag) needs `uv sync --extra era5` and a CDS key in `~/.cdsapirc`. |
| AWC   | **SoilGrids 2.0** water content at 33 kPa minus 1500 kPa, 0–100 cm | Read through ISRIC's WCS, which resamples on the server, so only a few MB are downloaded. Sampled at 0.025° and block-averaged to 0.05°. |

PET is computed once on its native grid, interpolated bilinearly to each rainfall grid,
and shared by every engine. CRU TS has a release lag of about 6 months (v4.10 ends
2025-12), so **SPEI and Palmer end where PET ends**. SPI and percent of normal run to the
latest rainfall month. Switch to `thornthwaite` with `era5_land` if SPEI must be current.

## Calibration

The reference period is **1991–2020** for every source except RFE2. RFE2 starts in 2001,
so it is calibrated on 2001–2020. Its outputs carry `wmo_compliant_calibration = 0`.
**ARC2**, the long-record companion of RFE2, is calibrated on 1991–2020 and is the
RFE-family series to benchmark.

## Indices and support matrix

`indices.distributions` in the config lists the distributions to compute. Each engine
computes the ones it supports and logs the rest.

| index | dist          | climate_indices | xclim_app       | xclim_pwm      |
|-------|---------------|-----------------|-----------------|----------------|
| SPI   | gamma         | ✓               | ✓               | ✓ (`gam`)      |
| SPI   | pearson3      | ✓               | –               | ✓ (`pe3`)      |
| SPEI  | gamma         | ✓               | ✓               | ✓              |
| SPEI  | pearson3      | ✓               | –               | ✓              |
| SPEI  | loglogistic   | ✓ (GLO, as R SPEI) | ✓ (2-param `fisk`) | ✓ (GLO, as R SPEI) |
| PDSI, PHDI, PMDI, Z-Index, scPDSI | – | ✓ | – | – |
| Percent of normal | – | ✓ | – | – |

These choices are fixed so that the outputs can be compared like for like:

* **Zero rainfall.** SPI uses the mixed distribution P(0) + (1 − P(0))·G(x). A zero total
  scores Φ⁻¹(P(0)), as in NOAA/NCEI and SPEIbase. xclim is run with the equivalent
  `upper`/`ecdf` zero placement.
* **SPEI water balance.** P − PET + 1000 mm, as in climate_indices and R `SPEI`.
* **Clipping.** Every standardized index is clipped to ±3.09 (`indices.clip`).
  climate_indices always does this; xclim's own bound is ±8.21.
* **Accumulation.** The window is summed (xclim averages, which standardizes the same).
  It is computed over the whole record before the calibration years are cut out, so
  January of the first calibration year includes the preceding months.

The Palmer indices take AWC per pixel, with `soil.fallback_mm` where SoilGrids has no
value. PDSI, PHDI, PMDI and Z-Index run on whole tiles with climate_indices' native
kernels, which is about 30× faster than pixel by pixel and gives identical results. scPDSI
has no block form and costs about 0.04 CPU-seconds per pixel, roughly 6 CPU-hours for the
full 0.05° grid. Set `"scpdsi": false` under `engines.climate_indices` to skip it, or
`compute.coarsen` to work on a coarser grid.

**Palmer QC.** In strongly seasonal climates a calendar month's calibration-period mean
potential runoff (or recharge, or loss) can be almost zero. The CAFEC coefficient (e.g.
γ = mean RO / mean PRO) then explodes, and the Z-index and PDSI of any month with real
runoff become absurd (e.g. γ = 64,597 and PDSI = −292 at a central-Kenya pixel). The values
are left as climate_indices returns them. `palmer_qc.nc` flags the pixels (1 = suspect)
where |PDSI| > 15 or |Z| > 45. The problem is sensitive to the calibration period: the same
pixel stays within ±4.4 with 1981–2010.

## Outputs

```
data/
  raw/<source>/YYYY/<source>_YYYYMM.tif          clipped monthly rainfall (mm/month)
  processed/<source>_precip_monthly.nc            stacked cube
  aux/                                            CRU TS, SoilGrids, PET caches
  params/<source>/<engine>/<index>_<dist>_<SS>.nc fitted parameters (month, lat, lon)
  indices/<source>/<engine>/<index>_<dist>_<SS>.nc index time series
  indices/<source>/climate_indices/{pdsi,phdi,pmdi,zindex,scpdsi,palmer_qc,pnp_SS}.nc
  indices/<source>/<engine>/run_metrics.json      wall/CPU time and peak memory per step
```

Every output records the engine and its version, the exact distribution and fitting
method, the calibration period, the clip limit, the PET method and the bounding box in
its attributes. `run_metrics.json` times `fit` and `apply` separately, so the cost of
the one-off calibration and of a monthly update can be compared across engines.

## Design notes (from mukau-spi-chirps)

* **Fit once, apply many times.** The distribution parameters are fitted on the
  calibration period and stored. Monthly updates only apply them, which is the cheapest
  path, and the stored parameters can be compared across engines.
* **Idempotent incremental sync** with `--refresh-last N`. This re-fetches the latest
  months, which providers sometimes revise (the old `SYNC_CURRENT_MONTH`).

mukau-spi-chirps itself has issues this project avoids:
* Its SPI ignores zero rainfall (`prob_cero_check=False`), so totals of zero become NaN.
* It loads the 1-month probability of zero for every scale.
* It maps ±∞ to ±2 instead of clipping to ±3.09.
* It calibrates on 1981–2010.

## Benchmarking notes

First comparison on real data (CHIRPS v3, central Kenya 36–38°E 2°S–0°, 1981–2026,
1991–2020 calibration). Each row compares climate_indices with the named xclim engine:

| index / dist             | vs xclim_app            | vs xclim_pwm              |
|--------------------------|-------------------------|---------------------------|
| SPI gamma                | identical               | RMSE 0.04–0.11, 92–96 % same class |
| SPI Pearson III          | –                       | identical at 12 mo, 99.5 % at 1 mo (zero handling) |
| SPEI gamma               | identical               | RMSE 0.04–0.06, 97 % same class |
| SPEI Pearson III         | –                       | identical |
| SPEI log-logistic        | RMSE 0.20–0.24, **87–89 %** same class | identical |

xclim APP's log-logistic is a 2-parameter `fisk` with loc = 0. It is not the 3-parameter
GLO of R `SPEI`/SPEIbase, so use `climate_indices` or `xclim_pwm` for
`spei_loglogistic_*` when comparing with R CDT.

* Compare on the same calibration period (1991–2020), distribution, fitting method and
  clip limit. Most disagreement between tools comes from these choices, not from bugs.
* R `SPEI::spei()` defaults to the log-logistic (GLO) with unbiased PWM fitting. The
  matching outputs here are `spei_loglogistic_*` from `climate_indices` or `xclim_pwm`.
* `peak_rss_mb` in `run_metrics.json` is the process high-water mark, so it only isolates
  an engine's memory when that engine runs alone (`compute --engines <name>`).
* ARC2 and RFE2 have values over the ocean, unlike CHIRPS. SPEI and Palmer are land-only
  anyway, because CRU PET is.
* `compute.coarsen` aggregates CHIRPS to a coarser grid (e.g. `2` → 0.1°), which is
  useful for quick comparisons with ARC2 and RFE2 on the same resolution.
