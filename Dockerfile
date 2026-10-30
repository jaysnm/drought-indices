# drought-indices image: the CLI with its locked dependencies.
#
#   docker build -t drought-indices .
#   docker build --build-arg UV_EXTRAS="--extra era5" -t drought-indices .   # with ERA5-Land support

FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
ARG UV_EXTRAS=""
WORKDIR /src
# Dependencies first, so that code changes do not reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${UV_EXTRAS}

FROM python:3.12-slim
# libexpat1: needed by the GDAL bundled in the rasterio wheel, no longer in the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY docker/scheduler.sh /usr/local/bin/drought-indices-scheduler
WORKDIR /app
COPY config ./config
# data/ is normally a bind mount; it is world-writable so that any --user can run the image
RUN mkdir -p data && chmod 0777 data
ENV PATH=/opt/venv/bin:$PATH \
    DROUGHT_INDICES_CONFIG=/app/config/config.json \
    HOME=/tmp \
    PYTHONUNBUFFERED=1
ENTRYPOINT ["drought-indices"]
CMD ["--help"]
