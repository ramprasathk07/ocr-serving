"""FastAPI gateway — the client-facing slice of the pipeline.

    POST   /v1/ocr                    upload -> blob store -> job state -> enqueue -> 202
    GET    /v1/ocr/{job_id}           status while running, full result when done
    GET    /v1/ocr/{job_id}/stream    SSE token stream, resumable via Last-Event-ID
    WS     /v1/ocr/{job_id}/ws        the same events over a WebSocket
    GET    /v1/ocr/{job_id}/text|markdown|pdf   artifacts
    DELETE /v1/ocr/{job_id}           cooperative cancel
    GET    /v1/jobs                   recent jobs for the caller's tenant
    GET    /healthz /readyz /metrics  liveness, readiness (deps), Prometheus

The gateway never touches the GPU and never blocks: uploads stream to disk in
1 MiB chunks, and streaming endpoints tail a Redis Stream so a reconnecting
client resumes exactly where it left off rather than losing tokens.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import StreamingResponse

from ocr_serving.common import events
from ocr_serving.common.config import get_settings
from ocr_serving.common.db import Database
from ocr_serving.common.engine import client_from_settings
from ocr_serving.common.filetype import describe, matches_extension
from ocr_serving.common.logging import get_logger, setup_logging
from ocr_serving.common.metrics import (
    BUILD_INFO,
    QUEUE_DEAD,
    QUEUE_DEPTH,
    QUEUE_PENDING,
    RATE_LIMITED,
    STREAM_CLIENTS,
    UPLOAD_BYTES,
    metrics_payload,
)
from ocr_serving.common.queue import JobQueue
from ocr_serving.common.ratelimit import PageQuota, RateLimiter
from ocr_serving.common.redis_client import close_redis, get_redis
from ocr_serving.common.redis_client import ping as redis_ping
from ocr_serving.common.schemas import (
    ErrorResponse,
    HealthResponse,
    JobAccepted,
    JobStatus,
    StreamEvent,
)
from ocr_serving.common.storage import LocalBlobStore, UploadTooLarge
from ocr_serving.gateway.auth import Principal, rate_limited, require_api_key
from ocr_serving.gateway.middleware import RequestContextMiddleware

log = get_logger("gateway")
settings = get_settings()

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"}
#: Bound concurrent long-lived streams so they cannot exhaust the Redis pool.
MAX_STREAM_CLIENTS = 64
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging("gateway", settings.log_level, settings.log_format)
    BUILD_INFO.info({"version": settings.api_version, "model": settings.model_id,
                     "engine": settings.engine_base_url, "env": settings.env})

    app.state.redis = get_redis()
    app.state.queue = JobQueue(app.state.redis, consumer="gateway")
    app.state.blobs = LocalBlobStore(settings.uploads_dir)
    app.state.limiter = RateLimiter(app.state.redis, settings.rate_limit_rps,
                                    settings.rate_limit_burst)
    app.state.quota = PageQuota(app.state.redis, settings.quota_pages_per_day)
    app.state.db = Database(settings.postgres_dsn, settings.postgres_enabled)
    app.state.engine = client_from_settings(settings, max_retries=0)
    app.state.stream_slots = asyncio.Semaphore(MAX_STREAM_CLIENTS)

    try:
        await app.state.queue.ensure_group()
    except Exception as exc:
        log.warning("could not ensure queue group", extra={"queue_error": str(exc)})
    await app.state.db.connect()
    log.info("gateway ready", extra={"env": settings.env, "model": settings.model_id})
    try:
        yield
    finally:
        await app.state.engine.aclose()
        await app.state.db.close()
        await close_redis()
        log.info("gateway stopped")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Streaming OCR over one VLM, served by whichever stack is active "
        "(vLLM / Triton / Ray Serve / KServe)."
    ),
    lifespan=lifespan,
    responses={401: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["x-request-id", "retry-after"],
)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    body = ErrorResponse(
        error=exc.__class__.__name__ if exc.status_code >= 500 else _slug(exc.status_code),
        detail=str(exc.detail),
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(body.model_dump(), status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(
            error="validation_error",
            detail=json.dumps(exc.errors(), default=str)[:1000],
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(),
        status_code=422,
    )


def _slug(code: int) -> str:
    return {400: "bad_request", 401: "unauthorized", 404: "not_found", 409: "conflict",
            413: "payload_too_large", 415: "unsupported_media_type",
            429: "too_many_requests", 503: "unavailable"}.get(code, "error")


# --------------------------------------------------------------------- submit
async def _chunks(upload: UploadFile, size: int = 1 << 20) -> AsyncIterator[bytes]:
    while chunk := await upload.read(size):
        yield chunk


@app.post(
    "/v1/ocr",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a document for OCR",
    tags=["ocr"],
)
async def submit(
    request: Request,
    file: UploadFile,
    principal: Principal = Depends(require_api_key),
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> JobAccepted:
    await rate_limited(request, principal)
    r = request.app.state.redis

    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported type {suffix or '(none)'}; allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    # Idempotency: the same key from the same tenant returns the original job.
    idem_key = f"idem:{principal.tenant}:{idempotency_key}" if idempotency_key else ""
    if idem_key and (existing := await r.get(idem_key)):
        return _accepted(existing)

    job_id = uuid.uuid4().hex[:12]
    try:
        blob = await request.app.state.blobs.put_stream(
            f"{job_id}{suffix}", _chunks(file), max_bytes=settings.max_upload_bytes
        )
    except UploadTooLarge:
        RATE_LIMITED.labels(reason="too_large").inc()
        # 413 by number: starlette renamed the constant, and the number never moves.
        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB") from None
    if blob.size == 0:
        request.app.state.blobs.delete(blob.key)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty upload")

    # Trusting the extension would hand a mislabelled file straight to pymupdf or
    # OpenCV; check the magic bytes at the door instead.
    if not matches_extension(suffix, blob.head):
        request.app.state.blobs.delete(blob.key)
        RATE_LIMITED.labels(reason="content_mismatch").inc()
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"content is {describe(blob.head)}, which does not match the {suffix} extension",
        )
    UPLOAD_BYTES.observe(blob.size)

    # Quota is charged in pages, so count them before accepting the job.
    from ocr_serving.workers.documents import probe_page_count

    pages = await asyncio.to_thread(probe_page_count, blob.path, settings.max_pages)
    if pages == 0:
        request.app.state.blobs.delete(blob.key)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unreadable or encrypted document")
    decision = await request.app.state.quota.check(principal.tenant, pages)
    if not decision.allowed:
        request.app.state.blobs.delete(blob.key)
        RATE_LIMITED.labels(reason="quota").inc()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"daily page quota exhausted ({settings.quota_pages_per_day} pages/day)",
        )

    await events.set_status(
        r, job_id, JobStatus.QUEUED,
        filename=file.filename or blob.key,
        path=str(blob.path),
        sha256=blob.sha256,
        bytes=blob.size,
        tenant=principal.tenant,
        page_count=pages,
        pages_done=0,
        attempts=0,
        created_at=datetime.now(UTC).isoformat(),
    )
    await request.app.state.queue.enqueue(job_id, tenant=principal.tenant)
    if idem_key:
        await r.setex(idem_key, 86_400, job_id)

    log.info(
        "job accepted",
        extra={"job_id": job_id, "pages": pages, "size_bytes": blob.size,
               "tenant": principal.tenant, "document": file.filename},
    )
    return _accepted(job_id)


def _accepted(job_id: str) -> JobAccepted:
    return JobAccepted(
        job_id=job_id,
        stream_url=f"/v1/ocr/{job_id}/stream",
        ws_url=f"/v1/ocr/{job_id}/ws",
        result_url=f"/v1/ocr/{job_id}",
    )


# --------------------------------------------------------------------- result
async def _load_job(request: Request, job_id: str, principal: Principal) -> dict[str, str]:
    job = await events.get_job(request.app.state.redis, job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
    if job.get("tenant", "default") != principal.tenant:
        # Do not leak existence across tenants.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown job")
    return job


@app.get("/v1/ocr/{job_id}", summary="Job status or final result", tags=["ocr"])
async def result(
    request: Request, job_id: str, principal: Principal = Depends(require_api_key)
) -> JSONResponse:
    try:
        job = await _load_job(request, job_id, principal)
    except HTTPException:
        # Redis state expires after result_ttl_days; PostgreSQL is the long-term record.
        stored = await request.app.state.db.get_result(job_id)
        if stored is None or stored.tenant != principal.tenant:
            raise
        return JSONResponse(stored.model_dump(mode="json"))

    if job.get("status") == JobStatus.COMPLETED.value and job.get("result_path"):
        path = Path(job["result_path"])
        if path.exists():
            return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    state = events.parse_state(job_id, job)
    return JSONResponse(
        {**state.model_dump(mode="json"), "progress": state.progress},
        status_code=200,
    )


@app.delete("/v1/ocr/{job_id}", summary="Cancel a running job", tags=["ocr"])
async def cancel(
    request: Request, job_id: str, principal: Principal = Depends(require_api_key)
) -> JSONResponse:
    job = await _load_job(request, job_id, principal)
    state = events.parse_state(job_id, job)
    if state.status.terminal:
        return JSONResponse({"job_id": job_id, "status": state.status.value, "cancelled": False})
    await events.request_cancel(request.app.state.redis, job_id)
    log.info("cancel requested", extra={"job_id": job_id})
    return JSONResponse({"job_id": job_id, "status": "cancelling", "cancelled": True})


@app.get("/v1/jobs", summary="Recent jobs for this tenant", tags=["ocr"])
async def recent_jobs(
    request: Request,
    principal: Principal = Depends(require_api_key),
    limit: int = Query(default=25, ge=1, le=200),
) -> JSONResponse:
    rows = await request.app.state.db.recent(principal.tenant, limit)
    return JSONResponse({"jobs": rows}, headers={"x-source": "postgres"})


# ------------------------------------------------------------------ artifacts
async def _artifact(request: Request, job_id: str, principal: Principal, field: str) -> Path:
    job = await _load_job(request, job_id, principal)
    if job.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"job is {job.get('status')}, artifacts not ready")
    raw = job.get(field)
    if not raw or not Path(raw).exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{field} not available for this job")
    return Path(raw)


@app.get("/v1/ocr/{job_id}/text", response_class=PlainTextResponse, tags=["artifacts"])
async def text_artifact(
    request: Request, job_id: str, principal: Principal = Depends(require_api_key)
) -> PlainTextResponse:
    path = await _artifact(request, job_id, principal, "result_path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PlainTextResponse(payload.get("full_text", ""))


@app.get("/v1/ocr/{job_id}/markdown", response_class=PlainTextResponse, tags=["artifacts"])
async def markdown_artifact(
    request: Request, job_id: str, principal: Principal = Depends(require_api_key)
) -> PlainTextResponse:
    path = await _artifact(request, job_id, principal, "markdown_path")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")


@app.get("/v1/ocr/{job_id}/pdf", tags=["artifacts"])
async def pdf_artifact(
    request: Request, job_id: str, principal: Principal = Depends(require_api_key)
) -> FileResponse:
    path = await _artifact(request, job_id, principal, "pdf_path")
    return FileResponse(path, media_type="application/pdf", filename=f"{job_id}.pdf")


# ------------------------------------------------------------------ streaming
def _sse(event: StreamEvent) -> str:
    payload = json.dumps(
        {"type": event.type.value, "job_id": event.job_id, "page": event.page,
         "data": event.data, "seq": event.seq},
        ensure_ascii=False,
    )
    return f"id: {event.id}\nevent: {event.type.value}\ndata: {payload}\n\n"


async def _event_source(
    request: Request, job_id: str, last_id: str
) -> AsyncIterator[StreamEvent | None]:
    """Yield events after ``last_id``; ``None`` marks a keepalive tick."""
    r = request.app.state.redis
    cursor = last_id
    block_ms = int(settings.sse_keepalive_s * 1000)
    while True:
        if await request.is_disconnected():
            return
        batch = await events.read_events(r, job_id, cursor, block_ms=block_ms)
        if not batch:
            job = await events.get_job(r, job_id)
            if not job or JobStatus(job.get("status", "queued")).terminal:
                # Terminal state reached with nothing left on the stream.
                return
            yield None
            continue
        for event in batch:
            cursor = event.id or cursor
            yield event
            if event.type.terminal:
                return


@app.get("/v1/ocr/{job_id}/stream", summary="SSE token stream", tags=["stream"])
async def stream(
    request: Request,
    job_id: str,
    principal: Principal = Depends(require_api_key),
    last_event_id: str = Header(default="", alias="Last-Event-ID"),
    replay: bool = Query(default=True, description="Replay events already produced"),
) -> StreamingResponse:
    """Server-Sent Events.

    Reconnecting clients send ``Last-Event-ID`` (browsers do it automatically)
    and resume from the next event — no tokens lost, none repeated.
    """
    job = await _load_job(request, job_id, principal)
    start_id = last_event_id or (events.FROM_START if replay else events.FROM_NOW)
    slots = request.app.state.stream_slots
    if slots.locked():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "too many open streams")

    async def generator() -> AsyncIterator[str]:
        await slots.acquire()
        STREAM_CLIENTS.labels(transport="sse").inc()
        try:
            state = JobStatus(job.get("status", "queued"))
            yield (
                "event: status\n"
                f"data: {json.dumps({'job_id': job_id, 'status': state.value})}\n\n"
            )
            async for event in _event_source(request, job_id, start_id):
                yield _sse(event) if event is not None else ": keepalive\n\n"
            final = await events.get_job(request.app.state.redis, job_id)
            summary = {"job_id": job_id, "status": final.get("status", "unknown")}
            yield f"event: end\ndata: {json.dumps(summary)}\n\n"
        finally:
            STREAM_CLIENTS.labels(transport="sse").dec()
            slots.release()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no",
                 "connection": "keep-alive"},
    )


@app.websocket("/v1/ocr/{job_id}/ws")
async def websocket_stream(websocket: WebSocket, job_id: str) -> None:
    """Same event stream over a WebSocket.

    Browsers cannot set headers on a WebSocket handshake, so the key arrives as
    ``?api_key=`` (or the ``X-API-Key`` header for non-browser clients).
    """
    presented = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key", "")
    tenant = get_settings().api_key_map.get(presented)
    if tenant is None:
        await websocket.close(code=4401, reason="invalid api key")
        return

    r = get_redis()
    job = await events.get_job(r, job_id)
    if not job or job.get("tenant", "default") != tenant:
        await websocket.close(code=4404, reason="unknown job")
        return

    await websocket.accept()
    STREAM_CLIENTS.labels(transport="ws").inc()
    cursor = websocket.query_params.get("last_event_id") or events.FROM_START
    try:
        await websocket.send_json({"type": "status", "job_id": job_id,
                                   "status": job.get("status", "queued")})
        while True:
            batch = await events.read_events(r, job_id, cursor, block_ms=10_000)
            if not batch:
                state = await events.get_job(r, job_id)
                if not state or JobStatus(state.get("status", "queued")).terminal:
                    break
                await websocket.send_json({"type": "keepalive", "job_id": job_id})
                continue
            terminal = False
            for event in batch:
                cursor = event.id or cursor
                await websocket.send_json(
                    {"type": event.type.value, "job_id": job_id, "page": event.page,
                     "data": event.data, "seq": event.seq, "id": event.id}
                )
                terminal = terminal or event.type.terminal
            if terminal:
                break
    except WebSocketDisconnect:
        pass  # client went away mid-stream; nothing to clean up beyond the finally block
    except Exception as exc:
        log.warning("websocket error", extra={"job_id": job_id, "ws_error": str(exc)})
    finally:
        STREAM_CLIENTS.labels(transport="ws").dec()
        with contextlib.suppress(RuntimeError):
            await websocket.close()


# --------------------------------------------------------------------- ops
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness: the process is up. Never touches dependencies."""
    return HealthResponse(status="ok", version=settings.api_version)


@app.get("/readyz", response_model=HealthResponse, tags=["ops"])
async def readyz(request: Request, response: Response) -> HealthResponse:
    """Readiness: Redis is mandatory, engine and Postgres are reported but not fatal."""
    redis_ok = await redis_ping()
    engine_ok = await request.app.state.engine.health()
    db_ok = await request.app.state.db.healthy()
    pg_state = "ok" if db_ok else ("down" if settings.postgres_enabled else "disabled")
    ok = redis_ok
    response.status_code = 200 if ok else 503
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=settings.api_version,
        checks={
            "redis": "ok" if redis_ok else "down",
            "engine": "ok" if engine_ok else "down",
            "postgres": pg_state,
        },
    )


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    try:
        stats = await request.app.state.queue.stats()
        QUEUE_DEPTH.set(stats["depth"])
        QUEUE_PENDING.set(stats["pending"])
        QUEUE_DEAD.set(stats["dead"])
    except Exception:
        pass
    payload, content_type = metrics_payload()
    return Response(payload, media_type=content_type)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def demo() -> HTMLResponse:
    """Minimal streaming demo page (also what the README GIF records)."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>XFinite-OCR gateway</h1><p>See <a href='/docs'>/docs</a>.</p>")
    return HTMLResponse(index.read_text(encoding="utf-8"))
