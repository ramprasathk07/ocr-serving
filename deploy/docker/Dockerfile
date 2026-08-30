# Multi-stage build for the two CPU services. The GPU engine is never built here:
# vLLM / Triton / Ray / KServe each ship their own image and run outside compose,
# one at a time, on the single 3060.
#
#   docker build -f infra/docker/Dockerfile --target gateway -t ocr-gateway .
#   docker build -f infra/docker/Dockerfile --target worker  -t ocr-worker  .

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

# Dependency layer first: source changes must not re-resolve the wheel set.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \n      fastapi "uvicorn[standard]" redis pydantic pydantic-settings python-multipart \n      httpx orjson prometheus-client pymupdf opencv-python-headless numpy pillow \n      onnxruntime asyncpg \n    && rm -rf /root/.cache

COPY src ./src
COPY benchmarks ./benchmarks
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e . && rm -rf /root/.cache

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
