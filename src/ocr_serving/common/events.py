"""Job state (Redis hash) + replayable event stream (Redis Stream).

Design note — why streams and not pub/sub: with ``PUBLISH``/``SUBSCRIBE`` a
client that connects a few milliseconds late, or reconnects after a dropped
connection, silently loses tokens. Each job instead gets a capped Redis Stream
``job:{id}:events``; the stream id doubles as the SSE ``id:`` field, so a
browser reconnect sends ``Last-Event-ID`` and resumes exactly where it stopped.
"""
from __future__ import annotations

from datetime import datetime

import redis.asyncio as aioredis

from ocr_serving.common.config import get_settings
from ocr_serving.common.logging import get_logger
from ocr_serving.common.schemas import EventType, JobState, JobStatus, StreamEvent

log = get_logger(__name__)
settings = get_settings()

#: Read from the beginning of the stream (replay everything).
FROM_START = "0-0"
#: Read only events produced after subscribing.
FROM_NOW = "$"


# --------------------------------------------------------------------- events
async def publish(r: aioredis.Redis, event: StreamEvent) -> str:
    """Append an event; returns the stream id used as the SSE event id."""
    return await r.xadd(
        settings.events_stream(event.job_id),
        {
            "type": event.type.value,
            "page": "" if event.page is None else str(event.page),
            "data": event.data,
            "seq": str(event.seq),
            "ts": event.ts.isoformat(),
        },
        maxlen=settings.event_stream_maxlen,
        approximate=True,
    )


def _parse(job_id: str, entry_id: str, fields: dict[str, str]) -> StreamEvent:
    page = fields.get("page") or ""
    return StreamEvent(
        type=EventType(fields["type"]),
        job_id=job_id,
        page=int(page) if page else None,
        data=fields.get("data", ""),
        seq=int(fields.get("seq") or 0),
        ts=(
            datetime.fromisoformat(fields["ts"])
            if fields.get("ts")
            else datetime.now().astimezone()
        ),
        id=entry_id,
    )


async def read_events(
    r: aioredis.Redis,
    job_id: str,
    last_id: str = FROM_START,
    block_ms: int = 15_000,
    count: int = 500,
) -> list[StreamEvent]:
    """Blocking read of events newer than ``last_id`` (``[]`` on timeout)."""
    resp = await r.xread({settings.events_stream(job_id): last_id}, count=count, block=block_ms)
    if not resp:
        return []
    _, entries = resp[0]
    return [_parse(job_id, entry_id, fields) for entry_id, fields in entries]


async def expire_events(r: aioredis.Redis, job_id: str) -> None:
    """Retain a completed job's events briefly so late subscribers can replay."""
    await r.expire(settings.events_stream(job_id), settings.event_ttl_s)


# ------------------------------------------------------------------ job state
async def set_status(r: aioredis.Redis, job_id: str, status: JobStatus, **fields: object) -> None:
    mapping = {"status": status.value, "updated_at": datetime.now().astimezone().isoformat()}
    mapping.update({k: str(v) for k, v in fields.items() if v is not None})
    key = settings.job_key(job_id)
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, settings.result_ttl_days * 86_400)
        await pipe.execute()


async def update_job(r: aioredis.Redis, job_id: str, **fields: object) -> None:
    payload = {k: str(v) for k, v in fields.items() if v is not None}
    if payload:
        await r.hset(settings.job_key(job_id), mapping=payload)


async def incr_pages_done(r: aioredis.Redis, job_id: str, by: int = 1) -> int:
    return int(await r.hincrby(settings.job_key(job_id), "pages_done", by))


async def get_job(r: aioredis.Redis, job_id: str) -> dict[str, str]:
    return await r.hgetall(settings.job_key(job_id))


def parse_state(job_id: str, raw: dict[str, str]) -> JobState:
    def dt(key: str) -> datetime | None:
        value = raw.get(key)
        return datetime.fromisoformat(value) if value else None

    return JobState(
        job_id=job_id,
        status=JobStatus(raw.get("status", JobStatus.QUEUED.value)),
        filename=raw.get("filename", ""),
        tenant=raw.get("tenant", "default"),
        attempts=int(raw.get("attempts") or 0),
        pages_done=int(raw.get("pages_done") or 0),
        page_count=int(raw.get("page_count") or 0),
        error=raw.get("error"),
        created_at=dt("created_at"),
        updated_at=dt("updated_at"),
    )


# ----------------------------------------------------------------- cancelation
async def request_cancel(r: aioredis.Redis, job_id: str) -> None:
    """Cooperative cancel flag; the worker checks it between pages."""
    await r.setex(settings.cancel_key(job_id), settings.event_ttl_s, "1")


async def is_cancelled(r: aioredis.Redis, job_id: str) -> bool:
    return bool(await r.exists(settings.cancel_key(job_id)))
