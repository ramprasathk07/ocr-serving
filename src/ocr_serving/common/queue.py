"""Reliable job queue on a Redis Stream consumer group.

``LPUSH``/``BRPOP`` loses a job if the worker dies between the pop and the
result write. A consumer group keeps every delivered id in the Pending Entries
List until it is acknowledged, so:

* a crashed worker's jobs are reclaimed by a peer (``XAUTOCLAIM`` after
  ``visibility_timeout_s``),
* redeliveries are counted, and a job that keeps failing lands in a dead-letter
  stream instead of spinning forever,
* queue depth and pending counts are observable (``XINFO GROUPS``).
"""
from __future__ import annotations

from dataclasses import dataclass

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from ocr_serving.common.config import get_settings
from ocr_serving.common.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


@dataclass(slots=True)
class QueuedJob:
    entry_id: str          # stream id, needed to ack
    job_id: str
    deliveries: int = 1    # 1 on first delivery, >1 after a reclaim
    fields: dict[str, str] | None = None


class JobQueue:
    """Producer (gateway) and consumer (worker) sides of ``ocr:jobs``."""

    def __init__(
        self,
        redis: aioredis.Redis,
        consumer: str = "worker-1",
        stream: str | None = None,
        group: str | None = None,
    ) -> None:
        self.r = redis
        self.stream = stream or settings.queue_stream
        self.group = group or settings.queue_group
        self.consumer = consumer

    # --------------------------------------------------------------- producer
    async def enqueue(self, job_id: str, **fields: str) -> str:
        return await self.r.xadd(self.stream, {"job_id": job_id, **fields})

    # --------------------------------------------------------------- consumer
    async def ensure_group(self) -> None:
        try:
            await self.r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            log.info("created consumer group", extra={"stream": self.stream, "group": self.group})
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def reserve(self, count: int = 1, block_ms: int = 5_000) -> list[QueuedJob]:
        """Claim up to ``count`` new jobs, blocking until one arrives or timeout."""
        resp = await self.r.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=count, block=block_ms
        )
        if not resp:
            return []
        _, entries = resp[0]
        return [
            QueuedJob(entry_id=eid, job_id=f["job_id"], fields=f)
            for eid, f in entries
            if f and "job_id" in f
        ]

    async def reclaim(self, count: int = 10) -> list[QueuedJob]:
        """Take over jobs whose owner stopped heartbeating (crash recovery)."""
        min_idle = settings.visibility_timeout_s * 1000
        try:
            _, entries, *_ = await self.r.xautoclaim(
                self.stream, self.group, self.consumer, min_idle_time=min_idle, count=count
            )
        except ResponseError as exc:  # NOGROUP on a freshly flushed redis
            log.warning("reclaim failed", extra={"error": str(exc)})
            return []
        jobs: list[QueuedJob] = []
        for eid, f in entries:
            if not f or "job_id" not in f:          # tombstone from a trimmed stream
                await self.ack(eid)
                continue
            jobs.append(QueuedJob(entry_id=eid, job_id=f["job_id"], deliveries=2, fields=f))
        if jobs:
            log.warning("reclaimed stalled jobs", extra={"count": len(jobs)})
        return jobs

    async def ack(self, entry_id: str) -> None:
        await self.r.xack(self.stream, self.group, entry_id)
        await self.r.xdel(self.stream, entry_id)

    async def dead_letter(self, job: QueuedJob, reason: str) -> None:
        await self.r.xadd(
            settings.dead_letter_stream,
            {"job_id": job.job_id, "reason": reason[:500], "entry_id": job.entry_id},
            maxlen=1000,
            approximate=True,
        )
        await self.ack(job.entry_id)
        log.error("job dead-lettered", extra={"job_id": job.job_id, "reason": reason[:200]})

    # -------------------------------------------------------------- telemetry
    async def stats(self) -> dict[str, int]:
        """``{'depth': undelivered, 'pending': in-flight, 'dead': dead-lettered}``."""
        depth = pending = 0
        try:
            length = int(await self.r.xlen(self.stream))
            for info in await self.r.xinfo_groups(self.stream):
                if info.get("name") == self.group:
                    pending = int(info.get("pending") or 0)
            # Acknowledged entries are XDELed, so what remains is delivered-but-unacked
            # plus never-delivered. XINFO's own `lag` is unreliable across Redis forks.
            depth = max(length - pending, 0)
        except ResponseError:
            depth = int(await self.r.xlen(self.stream))
        try:
            dead = int(await self.r.xlen(settings.dead_letter_stream))
        except ResponseError:
            dead = 0
        return {"depth": depth, "pending": pending, "dead": dead}
