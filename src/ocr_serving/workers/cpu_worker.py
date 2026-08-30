"""CPU worker — the pull side of the pipeline.

    reserve(job) -> fetch blob -> classify + render pages incrementally ->
    preprocess -> layout -> tile -> stream regions through the serving stack ->
    publish events -> stitch -> persist (JSON / Markdown / searchable PDF /
    PostgreSQL) -> ack

Operational properties, all of which the previous single-threaded loop lacked:

* **at-least-once delivery.** Jobs come from a Redis Stream consumer group and
  are acknowledged only after the result is durable. A worker killed mid-job
  leaves the entry in the PEL; a peer reclaims it after
  ``OCR_VISIBILITY_TIMEOUT_S``. Repeated failures go to a dead-letter stream.
* **heartbeats.** In-flight entries are re-claimed by their owner periodically so
  a long job is never mistaken for a dead worker.
* **bounded concurrency at three levels** — jobs, pages within a job, regions
  within a page — so a 200-page PDF cannot monopolise the engine.
* **graceful shutdown.** SIGTERM stops new reservations and waits for in-flight
  jobs, so a ``docker compose down`` does not orphan work.
* **partial results.** One page failing does not fail the document; the page
  carries its error and the job completes with the rest of the text.

Run: ``make worker`` (needs Redis and an engine endpoint reachable).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ocr_serving.common import events
from ocr_serving.common.config import Settings, get_settings
from ocr_serving.common.db import Database
from ocr_serving.common.engine import OCRClient, client_from_settings
from ocr_serving.common.logging import get_logger, job_id_var, setup_logging
from ocr_serving.common.metrics import (
    DEAD_LETTERED,
    JOB_RETRIES,
    JOB_SECONDS,
    JOBS_INFLIGHT,
    JOBS_TOTAL,
    PAGE_SECONDS,
    PAGES_TOTAL,
    QUEUE_DEAD,
    QUEUE_DEPTH,
    QUEUE_PENDING,
    QUEUE_WAIT_SECONDS,
    RENDER_SECONDS,
    serve_worker_metrics,
)
from ocr_serving.common.queue import JobQueue, QueuedJob
from ocr_serving.common.ratelimit import PageQuota
from ocr_serving.common.redis_client import close_redis, get_redis
from ocr_serving.common.schemas import (
    Artifacts,
    EventType,
    JobStatus,
    JobTimings,
    OCRResult,
    PageResult,
    PageSource,
    Region,
    StreamEvent,
)
from ocr_serving.common.storage import LocalBlobStore
from ocr_serving.workers import postprocess
from ocr_serving.workers.documents import Document, RenderedPage, UnreadableDocument
from ocr_serving.workers.layout import LayoutDetector, crop, tile
from ocr_serving.workers.pdf_writer import build_searchable_pdf
from ocr_serving.workers.preprocess import is_blank, page_hash, prepare, to_bgr, to_png

log = get_logger("worker")


class JobCancelled(Exception):
    """Raised when the cancel flag is observed between pages."""


class TokenRelay:
    """Coalesces token deltas into ~50 ms batches before hitting Redis.

    One XADD per token would put 500 writes per page on Redis and swamp SSE
    clients with 6-byte frames. Batching keeps the stream visually continuous
    while cutting event volume by 10-30x.
    """

    def __init__(self, redis, job_id: str, page: int, flush_chars: int = 24,
                 flush_interval_s: float = 0.05) -> None:
        self.r, self.job_id, self.page = redis, job_id, page
        self.flush_chars, self.flush_interval_s = flush_chars, flush_interval_s
        self._buf: list[str] = []
        self._size = 0
        self._last = time.perf_counter()
        self._seq = 0

    async def __call__(self, delta: str) -> None:
        self._buf.append(delta)
        self._size += len(delta)
        now = time.perf_counter()
        if self._size >= self.flush_chars or now - self._last >= self.flush_interval_s:
            await self.flush()

    async def flush(self) -> None:
        if not self._buf:
            return
        payload, self._buf, self._size = "".join(self._buf), [], 0
        self._last = time.perf_counter()
        self._seq += 1
        await events.publish(
            self.r,
            StreamEvent(type=EventType.TOKEN, job_id=self.job_id, page=self.page,
                        data=payload, seq=self._seq),
        )


async def aiter_pages(document: Document, prefetch: int) -> AsyncIterator[RenderedPage]:
    """Run the blocking renderer on a thread, yielding pages as they are produced."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=max(prefetch, 1))
    done = object()

    def produce() -> None:
        try:
            for page in document.iter_pages():
                asyncio.run_coroutine_threadsafe(queue.put(page), loop).result()
        except Exception as exc:  # surfaced on the consumer side
            asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(done), loop).result()

    producer = loop.run_in_executor(None, produce)
    try:
        while True:
            item = await queue.get()
            if item is done:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        await producer


class Worker:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"
        self.redis = get_redis()
        self.queue = JobQueue(self.redis, consumer=self.consumer)
        self.blobs = LocalBlobStore(settings.uploads_dir)
        self.db = Database(settings.postgres_dsn, settings.postgres_enabled)
        self.quota = PageQuota(self.redis, settings.quota_pages_per_day)
        self.detector = LayoutDetector(
            settings.layout_model_path, settings.layout_score_threshold,
            settings.layout_iou_threshold, enabled=settings.layout_enabled,
        )
        self.client: OCRClient = client_from_settings(settings)
        self._stopping = asyncio.Event()
        self._inflight: dict[str, str] = {}   # entry_id -> job_id, for heartbeats
        self._sem = asyncio.Semaphore(settings.worker_concurrency)
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        await self.queue.ensure_group()
        await self.db.connect()
        healthy = await self.client.health()
        log.info(
            "worker started",
            extra={
                "consumer": self.consumer,
                "engine": self.s.engine_base_url,
                "engine_healthy": healthy,
                "model": self.s.model_id,
                "layout": self.detector.active,
                "concurrency": self.s.worker_concurrency,
            },
        )

    def request_stop(self) -> None:
        if not self._stopping.is_set():
            log.info("shutdown requested, draining in-flight jobs")
            self._stopping.set()

    async def run(self) -> None:
        await self.start()
        housekeeping = [
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            asyncio.create_task(self._reclaim_loop(), name="reclaim"),
            asyncio.create_task(self._metrics_loop(), name="queue-metrics"),
        ]
        try:
            while not self._stopping.is_set():
                await self._sem.acquire()
                if self._stopping.is_set():
                    self._sem.release()
                    break
                jobs = await self.queue.reserve(count=1, block_ms=2_000)
                if not jobs:
                    self._sem.release()
                    continue
                task = asyncio.create_task(self._run_job(jobs[0]))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                task.add_done_callback(lambda _t: self._sem.release())
        finally:
            for task in housekeeping:
                task.cancel()
            if self._tasks:
                log.info("waiting for in-flight jobs", extra={"count": len(self._tasks)})
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            await asyncio.gather(*housekeeping, return_exceptions=True)
            await self.client.aclose()
            await self.db.close()
            await close_redis()
            log.info("worker stopped")

    # ----------------------------------------------------------- housekeeping
    async def _heartbeat_loop(self) -> None:
        """Reset the idle timer on our own PEL entries so peers do not steal them."""
        interval = max(self.s.visibility_timeout_s / 3, 5)
        while True:
            await asyncio.sleep(interval)
            entry_ids = list(self._inflight)
            if not entry_ids:
                continue
            try:
                await self.redis.xclaim(
                    self.s.queue_stream, self.s.queue_group, self.consumer,
                    min_idle_time=0, message_ids=entry_ids, justid=True,
                )
            except Exception as exc:
                log.warning("heartbeat failed", extra={"hb_error": str(exc)})

    async def _reclaim_loop(self) -> None:
        while True:
            await asyncio.sleep(max(self.s.visibility_timeout_s / 2, 10))
            if self._stopping.is_set():
                return
            try:
                for job in await self.queue.reclaim():
                    JOB_RETRIES.inc()
                    await self._sem.acquire()
                    task = asyncio.create_task(self._run_job(job))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                    task.add_done_callback(lambda _t: self._sem.release())
            except Exception as exc:
                log.warning("reclaim loop error", extra={"reclaim_error": str(exc)})

    async def _metrics_loop(self) -> None:
        while True:
            try:
                stats = await self.queue.stats()
                QUEUE_DEPTH.set(stats["depth"])
                QUEUE_PENDING.set(stats["pending"])
                QUEUE_DEAD.set(stats["dead"])
            except Exception:
                pass
            await asyncio.sleep(5)

    # ------------------------------------------------------------------- job
    async def _run_job(self, job: QueuedJob) -> None:
        job_id_var.set(job.job_id)
        self._inflight[job.entry_id] = job.job_id
        JOBS_INFLIGHT.inc()
        started = time.perf_counter()
        try:
            await self._process(job)
            await self.queue.ack(job.entry_id)
        except UnreadableDocument as exc:
            await self._fail(job, str(exc), permanent=True)
        except JobCancelled:
            await self.queue.ack(job.entry_id)
            log.info("job cancelled")
        except Exception as exc:
            log.exception("job failed")
            await self._fail(job, f"{type(exc).__name__}: {exc}", permanent=False)
        finally:
            JOBS_INFLIGHT.dec()
            JOB_SECONDS.observe(time.perf_counter() - started)
            self._inflight.pop(job.entry_id, None)
            job_id_var.set(None)

    async def _mark_cancelled(self, job_id: str) -> None:
        """Idempotent terminal transition for a cancelled job."""
        await events.set_status(self.redis, job_id, JobStatus.CANCELLED)
        await events.publish(
            self.redis, StreamEvent(type=EventType.JOB_CANCELLED, job_id=job_id)
        )
        await events.expire_events(self.redis, job_id)
        JOBS_TOTAL.labels(status="cancelled").inc()

    async def _fail(self, job: QueuedJob, error: str, *, permanent: bool) -> None:
        """Permanent errors fail immediately; transient ones get redelivered."""
        attempts = int(await self.redis.hincrby(self.s.job_key(job.job_id), "attempts", 0) or 1)
        exhausted = permanent or attempts >= self.s.max_attempts
        if not exhausted:
            # Leave the entry unacknowledged: the reclaim loop redelivers it.
            log.warning("job will be retried", extra={"attempts": attempts, "job_error": error})
            await events.publish(
                self.redis,
                StreamEvent(type=EventType.PROGRESS, job_id=job.job_id,
                            data=f"retrying after error: {error}"),
            )
            return
        await events.set_status(self.redis, job.job_id, JobStatus.FAILED, error=error)
        await events.publish(
            self.redis,
            StreamEvent(type=EventType.JOB_FAILED, job_id=job.job_id, data=error),
        )
        await events.expire_events(self.redis, job.job_id)
        JOBS_TOTAL.labels(status="failed").inc()
        if permanent:
            await self.queue.ack(job.entry_id)
        else:
            DEAD_LETTERED.inc()
            await self.queue.dead_letter(job, error)

    async def _process(self, job: QueuedJob) -> None:
        meta = await events.get_job(self.redis, job.job_id)
        if not meta:
            raise UnreadableDocument("job state missing (expired?)")
        if meta.get("status") in {JobStatus.COMPLETED.value, JobStatus.CANCELLED.value}:
            log.info("skipping already-finished job", extra={"status": meta.get("status")})
            return
        if await events.is_cancelled(self.redis, job.job_id):
            await self._mark_cancelled(job.job_id)
            raise JobCancelled

        attempts = int(await self.redis.hincrby(self.s.job_key(job.job_id), "attempts", 1))
        started_at = datetime.now(UTC)
        queue_wait_s: float | None = None
        if enqueued := meta.get("created_at"):
            with contextlib.suppress(ValueError):
                queue_wait_s = (started_at - datetime.fromisoformat(enqueued)).total_seconds()
                QUEUE_WAIT_SECONDS.observe(queue_wait_s)

        await events.set_status(
            self.redis, job.job_id, JobStatus.PROCESSING,
            started_at=started_at.isoformat(), attempts=attempts, pages_done=0,
        )
        await events.publish(
            self.redis, StreamEvent(type=EventType.JOB_STARTED, job_id=job.job_id)
        )

        source = Path(meta["path"])
        if not source.exists():
            raise UnreadableDocument(f"upload missing at {source}")

        document = await asyncio.to_thread(
            Document, source,
            render_dpi=self.s.render_dpi, adaptive_dpi=self.s.adaptive_dpi,
            min_dpi=self.s.min_dpi, max_dpi=self.s.max_dpi, max_pages=self.s.max_pages,
            native_text_enabled=self.s.native_text_extraction,
            native_text_min_chars=self.s.native_text_min_chars,
        )
        await events.update_job(self.redis, job.job_id, page_count=document.page_count)
        log.info("processing document", extra={
            "pages": document.page_count, "kind": document.meta.kind, "attempt": attempts,
        })

        t0 = time.perf_counter()
        pages = await self._process_pages(job.job_id, document)
        document.close()
        process_s = time.perf_counter() - t0

        pages.sort(key=lambda p: p.index)
        page_texts = [p.text for p in pages]
        result = OCRResult(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            filename=meta.get("filename", ""),
            tenant=meta.get("tenant", "default"),
            page_count=len(pages),
            pages=pages,
            full_text=postprocess.merge_pages(page_texts),
            model=self.s.model_id,
            engine=self.s.engine_base_url,
            attempts=attempts,
            timings=JobTimings(
                queued_at=(
                    datetime.fromisoformat(meta["created_at"])
                    if meta.get("created_at")
                    else None
                ),
                started_at=started_at,
                completed_at=datetime.now(UTC),
                queue_wait_s=round(queue_wait_s, 3) if queue_wait_s is not None else None,
                process_s=round(process_s, 3),
            ),
        )
        result.artifacts = await self._write_artifacts(result, source, page_texts)

        await self.db.save_result(result)
        await self.quota.consume(result.tenant, result.ocr_pages)
        await events.set_status(
            self.redis, job.job_id, JobStatus.COMPLETED,
            result_path=result.artifacts.json_path,
            pdf_path=result.artifacts.pdf_path,
            markdown_path=result.artifacts.markdown_path,
            page_count=result.page_count,
            completed_at=result.timings.completed_at.isoformat(),
        )
        await events.publish(
            self.redis,
            StreamEvent(type=EventType.JOB_COMPLETE, job_id=job.job_id,
                        data=str(result.page_count)),
        )
        await events.expire_events(self.redis, job.job_id)
        JOBS_TOTAL.labels(status="completed").inc()
        log.info("job complete", extra={
            "pages": result.page_count, "ocr_pages": result.ocr_pages,
            "chars": len(result.full_text), "process_s": round(process_s, 2),
        })

    async def _process_pages(self, job_id: str, document: Document) -> list[PageResult]:
        """Stream pages through the pipeline with bounded page concurrency."""
        # hash -> future holding the text of the first page with that hash. Pages run
        # concurrently, so a duplicate must *wait* for the original rather than assume
        # it already finished; it then copies the text without spending a GPU call.
        dedupe: dict[str, asyncio.Future[str]] = {}
        pages: list[PageResult] = []
        page_sem = asyncio.Semaphore(self.s.page_concurrency)
        tasks: set[asyncio.Task] = set()

        async def handle(rendered: RenderedPage) -> None:
            async with page_sem:
                page = await self._process_page(job_id, rendered, dedupe)
            pages.append(page)
            done = await events.incr_pages_done(self.redis, job_id)
            await events.publish(
                self.redis,
                StreamEvent(type=EventType.PAGE_COMPLETE, job_id=job_id,
                            page=page.index, data=page.text, seq=done),
            )
            await events.publish(
                self.redis,
                StreamEvent(type=EventType.PROGRESS, job_id=job_id,
                            data=f"{done}/{document.page_count}", seq=done),
            )

        try:
            async for rendered in aiter_pages(document, self.s.worker_prefetch):
                if await events.is_cancelled(self.redis, job_id):
                    raise JobCancelled
                task = asyncio.create_task(handle(rendered))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                # Keep at most page_concurrency+prefetch pages resident in memory.
                while len(tasks) > self.s.page_concurrency + self.s.worker_prefetch:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if tasks:
                results = await asyncio.gather(*list(tasks), return_exceptions=True)
                for item in results:
                    if isinstance(item, BaseException) and not isinstance(item, JobCancelled):
                        raise item
        except JobCancelled:
            for task in tasks:
                task.cancel()
            await self._mark_cancelled(job_id)
            raise
        return pages

    async def _process_page(
        self, job_id: str, rendered: RenderedPage, dedupe: dict[str, asyncio.Future[str]]
    ) -> PageResult:
        index = rendered.index
        start = time.perf_counter()

        # Fast path 1: the PDF already had a text layer for this page.
        if rendered.native_text is not None:
            PAGES_TOTAL.labels(source=PageSource.NATIVE.value).inc()
            text = postprocess.normalize_text(rendered.native_text)
            await events.publish(
                self.redis,
                StreamEvent(type=EventType.PAGE_STARTED, job_id=job_id, page=index, data="native"),
            )
            return PageResult(
                index=index, text=text, source=PageSource.NATIVE, regions=0,
                chars=len(text), duration_ms=(time.perf_counter() - start) * 1000,
                width=rendered.width, height=rendered.height,
            )

        assert rendered.png is not None
        img, skew, regions = await asyncio.to_thread(self._prepare_page, rendered.png)
        RENDER_SECONDS.observe(time.perf_counter() - start)
        height, width = img.shape[:2]

        # Fast path 2: blank page or a duplicate of one already OCR'd.
        if is_blank(img, self.s.blank_std_threshold):
            PAGES_TOTAL.labels(source=PageSource.BLANK.value).inc()
            return PageResult(index=index, source=PageSource.BLANK, regions=0, dpi=rendered.dpi,
                              width=width, height=height, skew_deg=skew,
                              duration_ms=(time.perf_counter() - start) * 1000)
        digest = page_hash(img)
        original = dedupe.get(digest) if self.s.deduplicate_pages else None
        if original is not None:
            text = await original
            PAGES_TOTAL.labels(source=PageSource.DUPLICATE.value).inc()
            return PageResult(
                index=index, text=text, source=PageSource.DUPLICATE, regions=0,
                chars=len(text), dpi=rendered.dpi, width=width, height=height, skew_deg=skew,
                duplicate_of=getattr(original, "page_index", None),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        shared: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        shared.page_index = index  # type: ignore[attr-defined]
        if self.s.deduplicate_pages:
            dedupe[digest] = shared

        await events.publish(
            self.redis,
            StreamEvent(type=EventType.PAGE_STARTED, job_id=job_id, page=index,
                        data=str(len(regions))),
        )

        relay = TokenRelay(self.redis, job_id, index)
        region_sem = asyncio.Semaphore(self.s.region_concurrency)
        texts: list[str] = [""] * len(regions)
        ttfts: list[float] = []
        tokens = 0
        errors: list[str] = []

        async def run_region(position: int, region: Region) -> None:
            nonlocal tokens
            png = await asyncio.to_thread(lambda: to_png(crop(img, region)))
            async with region_sem:
                stats = await self.client.ocr_page_stream(png, on_delta=relay)
            await relay.flush()
            if stats.error:
                errors.append(stats.error)
                return
            texts[position] = stats.text
            tokens += stats.completion_tokens
            if stats.ttft_s:
                ttfts.append(stats.ttft_s)

        try:
            await asyncio.gather(*(run_region(i, r) for i, r in enumerate(regions)))
        finally:
            await relay.flush()
            if not shared.done():   # never leave a duplicate waiting on a failed page
                shared.set_result(postprocess.stitch_regions(texts))

        text = postprocess.stitch_regions(texts)
        text, flagged = postprocess.filter_degenerate(text)
        if flagged:
            log.warning("degenerate generation truncated", extra={"page": index})

        duration_ms = (time.perf_counter() - start) * 1000
        PAGE_SECONDS.observe(duration_ms / 1000)
        PAGES_TOTAL.labels(source=PageSource.OCR.value).inc()
        return PageResult(
            index=index, text=text, source=PageSource.OCR, regions=len(regions),
            chars=len(text), duration_ms=duration_ms,
            ttft_s=min(ttfts) if ttfts else None, tokens=tokens, dpi=rendered.dpi,
            width=width, height=height, skew_deg=skew,
            error="; ".join(errors)[:500] or None,
        )

    def _prepare_page(self, png: bytes) -> tuple[np.ndarray, float, list[Region]]:
        """Blocking CPU stage: decode, preprocess, detect layout, tile."""
        img = to_bgr(png)
        img, skew = prepare(
            img,
            max_px=self.s.max_image_px,
            do_deskew=self.s.deskew,
            deskew_max_angle=self.s.deskew_max_angle,
            do_clahe=self.s.clahe,
            do_denoise=self.s.denoise,
        )
        regions = tile(self.detector.detect(img), self.s.tile_max_height, self.s.tile_overlap)
        return img, skew, regions

    # ------------------------------------------------------------- artifacts
    async def _write_artifacts(
        self, result: OCRResult, source: Path, page_texts: list[str]
    ) -> Artifacts:
        json_path = self.s.results_dir / f"{result.job_id}.json"
        artifacts = Artifacts(json_path=str(json_path))

        md_path = self.s.artifacts_dir / f"{result.job_id}.md"
        markdown = postprocess.to_markdown(result.job_id, result.filename, page_texts)
        await asyncio.to_thread(md_path.write_text, markdown, "utf-8")
        artifacts.markdown_path = str(md_path)

        if self.s.searchable_pdf and result.full_text.strip():
            pdf_path = self.s.artifacts_dir / f"{result.job_id}.pdf"
            try:
                await asyncio.to_thread(build_searchable_pdf, source, page_texts, pdf_path)
                artifacts.pdf_path = str(pdf_path)
            except Exception as exc:  # a missing PDF must not fail the job
                log.warning("searchable pdf failed", extra={"pdf_error": str(exc)})

        # Written last so the persisted JSON lists every artifact, itself included.
        result.artifacts = artifacts
        await asyncio.to_thread(json_path.write_text, result.model_dump_json(indent=2), "utf-8")
        return artifacts


async def main_async() -> None:
    settings = get_settings()
    setup_logging("worker", settings.log_level, settings.log_format)
    serve_worker_metrics(settings.worker_metrics_port)
    worker = Worker(settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, worker.request_stop)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: worker.request_stop())
    await worker.run()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main_async())


if __name__ == "__main__":
    main()
