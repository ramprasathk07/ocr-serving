"""End-to-end worker tests: the full page pipeline against the in-process mock engine.

No Redis server, no GPU, no model — but every stage runs for real: intake,
native-text fast path, preprocessing, layout, streaming, stitching, artifacts,
job state, events, retry and dead-letter handling.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ocr_serving.common import events
from ocr_serving.common.engine import OCRClient
from ocr_serving.common.queue import QueuedJob
from ocr_serving.common.schemas import EventType, JobStatus, PageSource
from ocr_serving.workers.cpu_worker import JobCancelled, TokenRelay, Worker


@pytest.fixture
async def worker(settings, redis_client, ocr_client):
    w = Worker(settings)
    await w.client.aclose()
    w.client = ocr_client              # in-process mock engine
    await w.queue.ensure_group()
    yield w
    await w.client.aclose()


async def enqueue(worker: Worker, source: Path, tenant: str = "acme") -> QueuedJob:
    job_id = uuid.uuid4().hex[:12]
    await events.set_status(
        worker.redis, job_id, JobStatus.QUEUED,
        filename=source.name, path=str(source), tenant=tenant,
        created_at=datetime.now(UTC).isoformat(), attempts=0, pages_done=0,
    )
    entry_id = await worker.queue.enqueue(job_id)
    return QueuedJob(entry_id=entry_id, job_id=job_id)


async def event_types(redis_client, job_id: str) -> list[str]:
    stream = await events.read_events(redis_client, job_id, events.FROM_START, block_ms=10)
    return [e.type.value for e in stream]


async def load_result(worker: Worker, job_id: str) -> dict:
    state = await events.get_job(worker.redis, job_id)
    return json.loads(Path(state["result_path"]).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ happy path
async def test_scanned_pdf_runs_the_full_ocr_path(worker, scanned_pdf, redis_client):
    job = await enqueue(worker, scanned_pdf)

    await worker._process(job)

    state = await events.get_job(redis_client, job.job_id)
    assert state["status"] == JobStatus.COMPLETED.value
    result = await load_result(worker, job.job_id)
    assert result["page_count"] == 1
    page = result["pages"][0]
    assert page["source"] == PageSource.OCR.value
    assert page["chars"] > 0 and page["tokens"] > 0
    assert page["ttft_s"] is not None
    assert result["full_text"].strip()
    assert result["model"] == "mock-ocr-vl"

    kinds = await event_types(redis_client, job.job_id)
    assert kinds[0] == EventType.JOB_STARTED.value
    assert EventType.TOKEN.value in kinds
    assert kinds[-1] == EventType.JOB_COMPLETE.value


async def test_native_text_pages_skip_the_engine(worker, text_pdf, redis_client):
    job = await enqueue(worker, text_pdf)

    await worker._process(job)

    result = await load_result(worker, job.job_id)
    sources = {p["index"]: p["source"] for p in result["pages"]}
    assert sources[0] == PageSource.NATIVE.value      # text layer read directly
    assert sources[1] == PageSource.BLANK.value       # empty second page
    assert "quick brown fox" in result["full_text"]
    # A native page costs no engine tokens at all.
    assert result["pages"][0]["tokens"] == 0
    assert await event_types(redis_client, job.job_id) != []


async def test_image_input_produces_one_page(worker, tmp_path, sample_png):
    path = tmp_path / "page.png"
    path.write_bytes(sample_png)
    job = await enqueue(worker, path)

    await worker._process(job)

    result = await load_result(worker, job.job_id)
    assert result["page_count"] == 1
    assert result["pages"][0]["source"] == PageSource.OCR.value


async def test_artifacts_are_written(worker, scanned_pdf):
    job = await enqueue(worker, scanned_pdf)
    await worker._process(job)

    result = await load_result(worker, job.job_id)
    artifacts = result["artifacts"]
    assert Path(artifacts["json_path"]).exists()
    markdown = Path(artifacts["markdown_path"]).read_text(encoding="utf-8")
    assert markdown.startswith("# scan.pdf")
    pdf_bytes = Path(artifacts["pdf_path"]).read_bytes()
    assert pdf_bytes.startswith(b"%PDF")


async def test_searchable_pdf_contains_the_ocr_text(worker, scanned_pdf):
    import pymupdf

    job = await enqueue(worker, scanned_pdf)
    await worker._process(job)
    result = await load_result(worker, job.job_id)

    with pymupdf.open(result["artifacts"]["pdf_path"]) as doc:
        embedded = doc.load_page(0).get_text("text")
    first_word = result["full_text"].split()[0]
    assert first_word in embedded


async def test_duplicate_pages_are_detected(worker, tmp_path, sample_png):
    import pymupdf

    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(0, 0, 420, 600), stream=sample_png)
    path = tmp_path / "dupes.pdf"
    doc.save(str(path))
    doc.close()

    job = await enqueue(worker, path)
    await worker._process(job)

    result = await load_result(worker, job.job_id)
    pages = sorted(result["pages"], key=lambda p: p["index"])
    assert sorted(p["source"] for p in pages) == [
        PageSource.DUPLICATE.value, PageSource.OCR.value
    ]
    # The duplicate spends no engine tokens but still carries the original's text,
    # so the assembled document is complete regardless of page ordering.
    duplicate = next(p for p in pages if p["source"] == PageSource.DUPLICATE.value)
    original = next(p for p in pages if p["source"] == PageSource.OCR.value)
    assert duplicate["tokens"] == 0
    assert duplicate["text"] == original["text"] != ""
    assert duplicate["duplicate_of"] == original["index"]


async def test_progress_events_track_pages(worker, text_pdf, redis_client):
    job = await enqueue(worker, text_pdf)
    await worker._process(job)

    stream = await events.read_events(redis_client, job.job_id, events.FROM_START, block_ms=10)
    progress = [e.data for e in stream if e.type is EventType.PROGRESS]
    assert progress[-1] == "2/2"
    assert int((await events.get_job(redis_client, job.job_id))["pages_done"]) == 2


# --------------------------------------------------------------------- failure
async def test_missing_upload_fails_permanently(worker, tmp_path, redis_client):
    from ocr_serving.workers.documents import UnreadableDocument

    job = await enqueue(worker, tmp_path / "gone.pdf")

    with pytest.raises(UnreadableDocument):
        await worker._process(job)

    await worker._fail(job, "upload missing", permanent=True)
    state = await events.get_job(redis_client, job.job_id)
    assert state["status"] == JobStatus.FAILED.value
    assert (await worker.queue.stats())["pending"] == 0      # acked, not redelivered
    assert (await worker.queue.stats())["dead"] == 0         # and not dead-lettered


async def test_transient_failure_is_left_for_redelivery(worker, scanned_pdf, redis_client):
    job = await enqueue(worker, scanned_pdf)
    await worker.queue.reserve(block_ms=50)

    await worker._fail(job, "engine timeout", permanent=False)

    state = await events.get_job(redis_client, job.job_id)
    assert state["status"] == JobStatus.QUEUED.value        # untouched, will be reclaimed
    assert (await worker.queue.stats())["pending"] == 1


async def test_exhausted_attempts_go_to_the_dead_letter_stream(worker, scanned_pdf, redis_client):
    job = await enqueue(worker, scanned_pdf)
    reserved = (await worker.queue.reserve(block_ms=50))[0]
    await events.update_job(redis_client, job.job_id, attempts=worker.s.max_attempts)

    await worker._fail(reserved, "engine unreachable", permanent=False)

    assert (await worker.queue.stats())["dead"] == 1
    state = await events.get_job(redis_client, job.job_id)
    assert state["status"] == JobStatus.FAILED.value
    assert EventType.JOB_FAILED.value in await event_types(redis_client, job.job_id)


async def test_engine_errors_degrade_to_a_partial_page(worker, scanned_pdf, redis_client):
    """A page whose engine call fails is recorded with an error; the job still completes."""
    failing = OCRClient(base_url="http://engine.test/v1", model="m", max_retries=0)
    failing._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500, json={"e": 1})),
        base_url="http://engine.test/v1",
    )
    worker.client = failing

    job = await enqueue(worker, scanned_pdf)
    await worker._process(job)

    result = await load_result(worker, job.job_id)
    assert result["status"] == JobStatus.COMPLETED.value
    assert result["pages"][0]["error"]
    assert result["pages"][0]["text"] == ""
    await failing.aclose()


async def test_cancellation_stops_the_job(worker, text_pdf, redis_client):
    job = await enqueue(worker, text_pdf)
    await events.request_cancel(redis_client, job.job_id)

    with pytest.raises(JobCancelled):
        await worker._process(job)

    state = await events.get_job(redis_client, job.job_id)
    assert state["status"] == JobStatus.CANCELLED.value


async def test_run_job_acks_on_success(worker, scanned_pdf):
    await enqueue(worker, scanned_pdf)
    reserved = (await worker.queue.reserve(block_ms=50))[0]

    await worker._run_job(reserved)

    stats = await worker.queue.stats()
    assert stats["pending"] == 0 and stats["depth"] == 0


# ----------------------------------------------------------------- token relay
async def test_token_relay_batches_deltas(redis_client):
    relay = TokenRelay(redis_client, "job-relay", page=0, flush_chars=10, flush_interval_s=99)
    for _ in range(4):
        await relay("abc")          # 12 chars total -> one flush at the fourth delta
    await relay.flush()

    stream = await events.read_events(redis_client, "job-relay", events.FROM_START, block_ms=10)
    assert len(stream) < 4
    assert "".join(e.data for e in stream) == "abc" * 4
    assert all(e.type is EventType.TOKEN for e in stream)
