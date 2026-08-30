# Multi-stage build for the two CPU services. The GPU engine is never built here:
# vLLM / Triton / Ray / KServe each ship their own image and run outside compose,
# one at a time, on the single 3060.
#
#   docker build -f deploy/docker/Dockerfile --target gateway -t ocr-gateway .
#   docker build -f deploy/docker/Dockerfile --target worker  -t ocr-worker  .

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# libglib2.0-0 is required by opencv-python-headless; curl backs the healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Install from pyproject rather than a hand-copied dependency list: one source of
# truth, no drift. A src-layout build needs the package present, so src/ is copied
# before the install — code changes do re-resolve wheels, which is the price of not
# maintaining the dependency set twice.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[worker,db]" && rm -rf /root/.cache

COPY benchmarks ./benchmarks
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 ocr \
    && mkdir -p /data/storage \
    && chown -R ocr:ocr /app /data
USER ocr
ENV OCR_STORAGE_DIR=/data/storage
STOPSIGNAL SIGTERM


# --------------------------------------------------------------------- gateway
FROM base AS gateway
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/healthz || exit 1
# One worker process: state lives in Redis, so scale with replicas, not threads.
CMD ["uvicorn", "ocr_serving.gateway.main:app", "--host", "0.0.0.0", "--port", "8080", \
     "--no-access-log", "--timeout-graceful-shutdown", "30"]


# ---------------------------------------------------------------------- worker
FROM base AS worker
EXPOSE 9101
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:9101/metrics >/dev/null || exit 1
CMD ["python", "-m", "ocr_serving.workers.cpu_worker"]


# ------------------------------------------------------------------ mock engine
# CPU stand-in for the serving stack: lets the whole pipeline and the dashboards
# run on a laptop or in CI. Never used for benchmark numbers.
FROM base AS mock-engine
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD curl -fsS http://localhost:8001/v1/models >/dev/null || exit 1
CMD ["uvicorn", "ocr_serving.serving.mock.server:app", "--host", "0.0.0.0", "--port", "8001"]
