"""Prometheus metrics shared by gateway and worker.

The gateway exposes them on ``GET /metrics``; the worker starts its own tiny
HTTP server (``OCR_WORKER_METRICS_PORT``) because it has no HTTP surface of its
own. Names follow the ``ocr_`` prefix and Prometheus base-unit convention so the
Grafana dashboard in ``monitoring/`` works against either process.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

# ------------------------------------------------------------------ pipeline
JOBS_TOTAL = Counter("ocr_jobs_total", "OCR jobs by terminal status", ["status"])
JOBS_INFLIGHT = Gauge("ocr_jobs_inflight", "Jobs currently being processed by this worker")
JOB_SECONDS = Histogram(
    "ocr_job_seconds", "End-to-end job processing time",
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600),
)
QUEUE_WAIT_SECONDS = Histogram(
    "ocr_queue_wait_seconds", "Time from enqueue to worker pickup",
    buckets=(0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300),
)
PAGES_TOTAL = Counter("ocr_pages_total", "Pages processed", ["source"])
PAGE_SECONDS = Histogram(
    "ocr_page_seconds", "End-to-end OCR time per page",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
)
TTFT_SECONDS = Histogram(
    "ocr_ttft_seconds", "Time to first token per engine request",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
RENDER_SECONDS = Histogram(
    "ocr_render_seconds", "Page render + preprocess time",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
JOB_RETRIES = Counter("ocr_job_retries_total", "Job redeliveries after a worker failure")
DEAD_LETTERED = Counter("ocr_jobs_dead_lettered_total", "Jobs moved to the dead-letter stream")

# -------------------------------------------------------------------- engine
ENGINE_REQUESTS = Counter(
    "ocr_engine_requests_total", "Requests to the serving stack", ["outcome"]
)
ENGINE_RETRIES = Counter("ocr_engine_retries_total", "Engine request retries")
ENGINE_TOKENS = Counter("ocr_engine_tokens_total", "Completion tokens returned by the engine")
ENGINE_INFLIGHT = Gauge("ocr_engine_inflight", "In-flight requests to the serving stack")
ENGINE_SECONDS = Histogram(
    "ocr_engine_seconds", "Engine request wall time",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

# --------------------------------------------------------------------- queue
QUEUE_DEPTH = Gauge("ocr_queue_depth", "Jobs waiting in the queue (undelivered)")
QUEUE_PENDING = Gauge("ocr_queue_pending", "Jobs delivered but not yet acknowledged")
QUEUE_DEAD = Gauge("ocr_queue_dead", "Jobs in the dead-letter stream")

# ------------------------------------------------------------------- storage
RETENTION_REMOVED = Counter(
    "ocr_retention_removed_total", "Blobs deleted by the retention sweep", ["kind"]
)
STORAGE_BYTES = Gauge("ocr_storage_bytes", "Bytes held in the blob store after the last sweep")

# ------------------------------------------------------------------- gateway
HTTP_REQUESTS = Counter(
    "ocr_http_requests_total", "Gateway HTTP requests", ["method", "path", "status"]
)
HTTP_SECONDS = Histogram(
    "ocr_http_request_seconds", "Gateway HTTP latency", ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
UPLOAD_BYTES = Histogram(
    "ocr_upload_bytes", "Uploaded document size",
    buckets=(1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8),
)
STREAM_CLIENTS = Gauge("ocr_stream_clients", "Connected SSE/WebSocket clients", ["transport"])
RATE_LIMITED = Counter("ocr_rate_limited_total", "Rejected requests", ["reason"])

BUILD_INFO = Info("ocr_build", "Service build information")


def serve_worker_metrics(port: int) -> None:
    """Expose /metrics from a non-HTTP process (the worker)."""
    start_http_server(port)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
