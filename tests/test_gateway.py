"""Gateway API tests: auth, validation, submission, streaming, cancel, ops endpoints."""
from __future__ import annotations

import json

import httpx
import pytest

from ocr_serving.common import events
from ocr_serving.common.schemas import EventType, JobStatus, StreamEvent

AUTH = {"X-API-Key": "test-key"}


@pytest.fixture
async def api(redis_client, engine_transport):
    """Gateway app with lifespan run, Redis faked and the engine wired in-process."""
    from ocr_serving.gateway.main import app

    async with app.router.lifespan_context(app):
        app.state.engine._client = httpx.AsyncClient(
            transport=engine_transport, base_url="http://engine.test/v1"
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        ) as client:
            client.app = app
            yield client


def _upload(name: str = "doc.pdf", data: bytes = b"%PDF-1.4 fake") -> dict:
    return {"file": (name, data, "application/pdf")}


async def submit_pdf(api, text_pdf, **kwargs) -> dict:
    files = {"file": (text_pdf.name, text_pdf.read_bytes(), "application/pdf")}
    response = await api.post("/v1/ocr", files=files, **kwargs)
    assert response.status_code == 202, response.text
    return response.json()


# ----------------------------------------------------------------------- auth
async def test_missing_key_is_rejected(api, text_pdf):
    response = await api.post("/v1/ocr", files=_upload())
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
    assert response.headers["x-request-id"]


async def test_bearer_token_is_accepted(api, text_pdf):
    files = {"file": (text_pdf.name, text_pdf.read_bytes(), "application/pdf")}
    response = await api.post("/v1/ocr", files=files, headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 202


async def test_unknown_key_is_rejected(api):
    response = await api.post("/v1/ocr", files=_upload(), headers={"X-API-Key": "nope"})
    assert response.status_code == 401


# ----------------------------------------------------------------- validation
async def test_unsupported_extension(api):
    response = await api.post(
        "/v1/ocr", files={"file": ("notes.docx", b"x", "application/octet-stream")}, headers=AUTH
    )
    assert response.status_code == 415


async def test_corrupt_pdf_is_rejected_before_queueing(api, redis_client):
    response = await api.post("/v1/ocr", files=_upload(data=b"not really a pdf"), headers=AUTH)
    assert response.status_code == 400
    assert await redis_client.xlen("ocr:jobs") == 0


async def test_oversized_upload(api, text_pdf, monkeypatch):
    from ocr_serving.gateway import main as gateway_main

    monkeypatch.setattr(gateway_main.settings, "max_upload_mb", 0)
    files = {"file": (text_pdf.name, text_pdf.read_bytes(), "application/pdf")}
    response = await api.post("/v1/ocr", files=files, headers=AUTH)
    assert response.status_code == 413


# --------------------------------------------------------------------- submit
async def test_submit_creates_job_state_and_queue_entry(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)

    assert job["status"] == "queued"
    assert job["stream_url"].endswith("/stream") and job["ws_url"].endswith("/ws")

    state = await events.get_job(redis_client, job["job_id"])
    assert state["tenant"] == "acme"
    assert state["filename"] == text_pdf.name
    assert int(state["page_count"]) == 2
    assert len(state["sha256"]) == 64

    entries = await redis_client.xrange("ocr:jobs")
    assert [e[1]["job_id"] for e in entries] == [job["job_id"]]


async def test_idempotency_key_returns_the_same_job(api, text_pdf, redis_client):
    headers = {**AUTH, "Idempotency-Key": "abc-123"}
    first = await submit_pdf(api, text_pdf, headers=headers)
    second = await submit_pdf(api, text_pdf, headers=headers)

    assert first["job_id"] == second["job_id"]
    assert await redis_client.xlen("ocr:jobs") == 1


async def test_quota_rejects_when_exhausted(api, text_pdf, monkeypatch):
    api.app.state.quota.limit = 1  # the 2-page PDF no longer fits
    files = {"file": (text_pdf.name, text_pdf.read_bytes(), "application/pdf")}
    response = await api.post("/v1/ocr", files=files, headers=AUTH)
    assert response.status_code == 429
    assert "quota" in response.json()["detail"]


async def test_rate_limit_returns_429_with_retry_after(api, text_pdf, redis_client):
    from ocr_serving.common.ratelimit import RateLimiter

    api.app.state.limiter = RateLimiter(redis_client, rate=0.1, burst=1)
    files = {"file": (text_pdf.name, text_pdf.read_bytes(), "application/pdf")}

    assert (await api.post("/v1/ocr", files=files, headers=AUTH)).status_code == 202
    second = await api.post("/v1/ocr", files=files, headers=AUTH)

    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1


# --------------------------------------------------------------------- result
async def test_status_before_completion_reports_progress(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    await events.set_status(redis_client, job["job_id"], JobStatus.PROCESSING, page_count=4)
    await events.incr_pages_done(redis_client, job["job_id"])

    body = (await api.get(f"/v1/ocr/{job['job_id']}", headers=AUTH)).json()
    assert body["status"] == "processing"
    assert body["progress"] == 0.25


async def test_unknown_job_is_404(api):
    assert (await api.get("/v1/ocr/doesnotexist", headers=AUTH)).status_code == 404


async def test_jobs_are_isolated_between_tenants(api, text_pdf):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    other = await api.get(f"/v1/ocr/{job['job_id']}", headers={"X-API-Key": "other-key"})
    assert other.status_code == 404


async def test_artifacts_conflict_while_running(api, text_pdf):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    response = await api.get(f"/v1/ocr/{job['job_id']}/markdown", headers=AUTH)
    assert response.status_code == 409


# --------------------------------------------------------------------- cancel
async def test_cancel_sets_the_flag(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)

    response = await api.delete(f"/v1/ocr/{job['job_id']}", headers=AUTH)

    assert response.json() == {"job_id": job["job_id"], "status": "cancelling", "cancelled": True}
    assert await events.is_cancelled(redis_client, job["job_id"])


async def test_cancel_is_a_noop_for_finished_jobs(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    await events.set_status(redis_client, job["job_id"], JobStatus.COMPLETED)

    body = (await api.delete(f"/v1/ocr/{job['job_id']}", headers=AUTH)).json()
    assert body["cancelled"] is False


# ------------------------------------------------------------------ streaming
def parse_sse(body: str) -> list[dict]:
    frames = []
    for block in body.split("\n\n"):
        event = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event["event"] = line[6:].strip()
            elif line.startswith("data:"):
                event["data"] = line[5:].strip()
            elif line.startswith("id:"):
                event["id"] = line[3:].strip()
        if event.get("event"):
            frames.append(event)
    return frames


async def _publish_run(redis_client, job_id: str) -> None:
    await events.publish(
        redis_client, StreamEvent(type=EventType.PAGE_STARTED, job_id=job_id, page=0)
    )
    for i, token in enumerate(["Hello ", "world"]):
        await events.publish(
            redis_client,
            StreamEvent(type=EventType.TOKEN, job_id=job_id, page=0, data=token, seq=i),
        )
    await events.publish(redis_client, StreamEvent(type=EventType.JOB_COMPLETE, job_id=job_id))
    await events.set_status(redis_client, job_id, JobStatus.COMPLETED)


async def test_sse_replays_the_whole_run(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    await _publish_run(redis_client, job["job_id"])

    response = await api.get(f"/v1/ocr/{job['job_id']}/stream", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = parse_sse(response.text)
    kinds = [f["event"] for f in frames]
    assert kinds == ["status", "page_started", "token", "token", "job_complete", "end"]
    tokens = [json.loads(f["data"])["data"] for f in frames if f["event"] == "token"]
    assert "".join(tokens) == "Hello world"
    assert all("id" in f for f in frames if f["event"] == "token")


async def test_sse_resumes_from_last_event_id(api, text_pdf, redis_client):
    job = await submit_pdf(api, text_pdf, headers=AUTH)
    await _publish_run(redis_client, job["job_id"])

    full = parse_sse((await api.get(f"/v1/ocr/{job['job_id']}/stream", headers=AUTH)).text)
    first_token = next(f for f in full if f["event"] == "token")

    resumed = await api.get(
        f"/v1/ocr/{job['job_id']}/stream",
        headers={**AUTH, "Last-Event-ID": first_token["id"]},
    )

    kinds = [f["event"] for f in parse_sse(resumed.text)]
    assert kinds == ["status", "token", "job_complete", "end"]  # the first token is not repeated


async def test_stream_requires_auth_and_a_known_job(api):
    assert (await api.get("/v1/ocr/nope/stream", headers=AUTH)).status_code == 404
    assert (await api.get("/v1/ocr/nope/stream")).status_code == 401


# ------------------------------------------------------------------------ ops
async def test_healthz_is_dependency_free(api):
    body = (await api.get("/healthz")).json()
    assert body["status"] == "ok"


async def test_readyz_reports_dependencies(api):
    response = await api.get("/readyz")
    body = response.json()
    assert response.status_code == 200
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["engine"] == "ok"
    assert body["checks"]["postgres"] == "disabled"


async def test_metrics_exposes_pipeline_series(api, text_pdf):
    await submit_pdf(api, text_pdf, headers=AUTH)
    body = (await api.get("/metrics")).text
    assert "ocr_http_requests_total" in body
    assert "ocr_queue_depth" in body
    assert "ocr_build_info" in body


async def test_demo_page_is_served(api):
    response = await api.get("/")
    assert response.status_code == 200
    assert "XFinite-OCR" in response.text


async def test_openapi_documents_the_api(api):
    spec = (await api.get("/openapi.json")).json()
    assert "/v1/ocr" in spec["paths"]
    assert "/v1/ocr/{job_id}/stream" in spec["paths"]
