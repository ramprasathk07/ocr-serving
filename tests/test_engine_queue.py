"""Engine client (streaming, retries, accounting), reliable queue, and event bus."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ocr_serving.common import events
from ocr_serving.common.engine import OCRClient, png_to_data_uri
from ocr_serving.common.queue import JobQueue
from ocr_serving.common.schemas import EventType, JobStatus, StreamEvent


# --------------------------------------------------------------------- engine
async def test_streaming_collects_text_ttft_and_usage(ocr_client, sample_png):
    deltas: list[str] = []

    async def collect(delta: str) -> None:
        deltas.append(delta)

    stats = await ocr_client.ocr_page_stream(sample_png, on_delta=collect)

    assert stats.ok, stats.error
    assert stats.text.strip()
    assert "".join(deltas) == stats.text
    assert stats.ttft_s > 0
    assert stats.e2e_s >= stats.ttft_s
    # NB: httpx's ASGI transport buffers, so in-process TTFT is not a latency number.
    # The mock reports usage, so the count comes from the engine, not chunk counting.
    assert stats.completion_tokens > 0
    assert stats.prompt_tokens == 512
    assert stats.tokens_per_s > 0


async def test_same_image_gives_same_text(ocr_client, sample_png):
    first = await ocr_client.ocr_page_stream(sample_png)
    second = await ocr_client.ocr_page_stream(sample_png)
    assert first.text == second.text


async def test_health_probe(ocr_client):
    assert await ocr_client.health() is True


def _sse_body(chunks: list[str]) -> bytes:
    frames = [
        json.dumps({"choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}]})
        for c in chunks
    ]
    return ("".join(f"data: {f}\n\n" for f in frames) + "data: [DONE]\n\n").encode()


def _no_backoff(monkeypatch) -> None:
    """Skip retry backoff sleeps without recursing into the patched function."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *_a, **_k: real_sleep(0))


def _flaky_client(failures: int, status_code: int = 503) -> tuple[OCRClient, dict]:
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] <= failures:
            return httpx.Response(status_code, json={"error": "overloaded"})
        return httpx.Response(200, content=_sse_body(["ok ", "text"]))

    client = OCRClient(base_url="http://engine.test/v1", model="m", max_retries=2)
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://engine.test/v1"
    )
    return client, state


async def test_retries_transient_engine_errors(sample_png, monkeypatch):
    _no_backoff(monkeypatch)
    client, state = _flaky_client(failures=2)

    stats = await client.ocr_page_stream(sample_png)

    assert stats.ok, stats.error
    assert stats.text == "ok text"
    assert state["calls"] == 3
    assert stats.attempts == 3


async def test_gives_up_after_max_retries(sample_png, monkeypatch):
    _no_backoff(monkeypatch)
    client, state = _flaky_client(failures=99)

    stats = await client.ocr_page_stream(sample_png)

    assert not stats.ok
    assert "503" in stats.error
    assert state["calls"] == 3  # initial + max_retries


async def test_client_errors_are_not_retried(sample_png):
    client, state = _flaky_client(failures=99, status_code=400)
    stats = await client.ocr_page_stream(sample_png)
    assert not stats.ok
    assert state["calls"] == 1


def test_data_uri_encoding(sample_png):
    uri = png_to_data_uri(sample_png)
    assert uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------- queue
@pytest.fixture
def queue(redis_client):
    return JobQueue(redis_client, consumer="test-worker")


async def test_enqueue_reserve_ack_roundtrip(queue):
    await queue.ensure_group()
    await queue.enqueue("job-1", tenant="acme")

    assert (await queue.stats())["depth"] == 1

    reserved = await queue.reserve(block_ms=50)
    assert [j.job_id for j in reserved] == ["job-1"]
    assert reserved[0].fields["tenant"] == "acme"
    assert (await queue.stats())["pending"] == 1

    await queue.ack(reserved[0].entry_id)
    stats = await queue.stats()
    assert stats["pending"] == 0 and stats["depth"] == 0


async def test_reserve_returns_empty_when_idle(queue):
    await queue.ensure_group()
    assert await queue.reserve(block_ms=10) == []


async def test_unacked_job_is_reclaimed_by_a_peer(redis_client, monkeypatch):
    from ocr_serving.common import queue as queue_module

    monkeypatch.setattr(queue_module.settings, "visibility_timeout_s", 0)
    dead = JobQueue(redis_client, consumer="worker-that-dies")
    alive = JobQueue(redis_client, consumer="worker-that-lives")
    await dead.ensure_group()
    await dead.enqueue("job-2")

    reserved = await dead.reserve(block_ms=50)
    assert reserved  # delivered but never acked: the worker "crashes" here

    reclaimed = await alive.reclaim()
    assert [j.job_id for j in reclaimed] == ["job-2"]
    assert reclaimed[0].deliveries == 2


async def test_dead_letter_removes_the_job_and_records_it(queue, redis_client):
    await queue.ensure_group()
    await queue.enqueue("job-3")
    job = (await queue.reserve(block_ms=50))[0]

    await queue.dead_letter(job, "engine unreachable")

    stats = await queue.stats()
    assert stats["pending"] == 0
    assert stats["dead"] == 1
    entries = await redis_client.xrange("ocr:jobs:dead")
    assert entries[0][1]["job_id"] == "job-3"


async def test_ensure_group_is_idempotent(queue):
    await queue.ensure_group()
    await queue.ensure_group()


# --------------------------------------------------------------------- events
async def test_events_replay_and_resume(redis_client):
    for i in range(3):
        await events.publish(
            redis_client,
            StreamEvent(type=EventType.TOKEN, job_id="j1", page=0, data=f"tok{i}", seq=i),
        )

    replay = await events.read_events(redis_client, "j1", events.FROM_START, block_ms=10)
    assert [e.data for e in replay] == ["tok0", "tok1", "tok2"]

    # Resuming from the second id yields only what came after it.
    resumed = await events.read_events(redis_client, "j1", replay[1].id, block_ms=10)
    assert [e.data for e in resumed] == ["tok2"]
    assert resumed[0].page == 0 and resumed[0].type is EventType.TOKEN


async def test_read_events_times_out_cleanly(redis_client):
    assert await events.read_events(redis_client, "nothing", block_ms=10) == []


async def test_job_state_roundtrip(redis_client):
    await events.set_status(
        redis_client, "j2", JobStatus.PROCESSING, filename="a.pdf", page_count=4, tenant="acme"
    )
    await events.incr_pages_done(redis_client, "j2")
    await events.incr_pages_done(redis_client, "j2")

    state = events.parse_state("j2", await events.get_job(redis_client, "j2"))
    assert state.status is JobStatus.PROCESSING
    assert state.page_count == 4 and state.pages_done == 2
    assert state.progress == 0.5
    assert state.tenant == "acme"


async def test_cancel_flag(redis_client):
    assert not await events.is_cancelled(redis_client, "j3")
    await events.request_cancel(redis_client, "j3")
    assert await events.is_cancelled(redis_client, "j3")


def test_terminal_status_and_event_flags():
    assert JobStatus.COMPLETED.terminal and JobStatus.FAILED.terminal
    assert not JobStatus.PROCESSING.terminal
    assert EventType.JOB_COMPLETE.terminal and not EventType.TOKEN.terminal
