"""Cross-cutting HTTP concerns: request ids, access logs, latency metrics.

Kept out of ``main.py`` so route handlers stay about OCR. The request id is
generated (or taken from an upstream ``X-Request-ID``) and pushed into a
contextvar, so every log line emitted while handling the request — including
ones from deep inside the worker client — carries it, and the same id comes back
in the response header and in error bodies.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ocr_serving.common.logging import get_logger, request_id_var
from ocr_serving.common.metrics import HTTP_REQUESTS, HTTP_SECONDS

log = get_logger("gateway.access")

#: Paths whose access logs are noise (scrapers and liveness probes).
QUIET_PATHS = {"/metrics", "/healthz", "/readyz"}


def route_template(request: Request) -> str:
    """Use the route pattern, not the raw path, so job ids do not explode label cardinality."""
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            path = route_template(request)
            HTTP_REQUESTS.labels(request.method, path, "500").inc()
            HTTP_SECONDS.labels(request.method, path).observe(elapsed)
            log.exception("unhandled error", extra={"path": request.url.path,
                                                    "duration_ms": round(elapsed * 1000, 1)})
            request_id_var.reset(token)
            return JSONResponse(
                status_code=500,
                content={"error": "internal_error", "detail": "unexpected server error",
                         "request_id": request_id},
                headers={"x-request-id": request_id},
            )

        elapsed = time.perf_counter() - start
        path = route_template(request)
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_SECONDS.labels(request.method, path).observe(elapsed)
        response.headers["x-request-id"] = request_id
        if request.url.path not in QUIET_PATHS:
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                    "client": request.client.host if request.client else "",
                },
            )
        request_id_var.reset(token)
        return response
