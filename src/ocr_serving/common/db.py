"""PostgreSQL result store (the "PostgreSQL Results" box).

Redis holds *live* job state and events with a TTL; PostgreSQL is the durable
record used for reprocessing, audit and CER/WER evaluation later. The JSON on
disk stays the fast path for ``GET /v1/ocr/{id}`` — Postgres is the system of
record behind it.

The layer degrades on purpose: if ``OCR_POSTGRES_ENABLED=false`` or the server
is unreachable at startup, the service runs filesystem-only and says so in
``/readyz`` instead of refusing to boot.
"""
from __future__ import annotations

import json
from typing import Any

from ocr_serving.common.logging import get_logger
from ocr_serving.common.schemas import Artifacts, JobStatus, JobTimings, OCRResult, PageResult

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    tenant       TEXT NOT NULL DEFAULT 'default',
    filename     TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL,
    page_count   INT  NOT NULL DEFAULT 0,
    ocr_pages    INT  NOT NULL DEFAULT 0,
    chars        INT  NOT NULL DEFAULT 0,
    model        TEXT NOT NULL DEFAULT '',
    engine       TEXT NOT NULL DEFAULT '',
    attempts     INT  NOT NULL DEFAULT 1,
    error        TEXT,
    full_text    TEXT NOT NULL DEFAULT '',
    artifacts    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ,
    started_at   TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    process_s    DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS pages (
    job_id      TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    idx         INT  NOT NULL,
    source      TEXT NOT NULL DEFAULT 'ocr',
    regions     INT  NOT NULL DEFAULT 1,
    chars       INT  NOT NULL DEFAULT 0,
    tokens      INT  NOT NULL DEFAULT 0,
    duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    ttft_s      DOUBLE PRECISION,
    text        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (job_id, idx)
);

CREATE INDEX IF NOT EXISTS jobs_tenant_created_idx ON jobs (tenant, created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status);
"""


class Database:
    """Thin asyncpg wrapper. All methods are no-ops when disabled."""

    def __init__(self, dsn: str, enabled: bool = True) -> None:
        self.dsn = dsn
        self.enabled = enabled
        self.pool: Any = None

    async def connect(self, min_size: int = 1, max_size: int = 5) -> bool:
        if not self.enabled:
            return False
        try:
            import asyncpg
        except ImportError:
            log.warning("asyncpg not installed; running without PostgreSQL")
            self.enabled = False
            return False
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn, min_size=min_size, max_size=max_size, command_timeout=30
            )
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA)
            log.info("postgres connected")
            return True
        except Exception as exc:
            log.warning("postgres unavailable, continuing without it", extra={"db_error": str(exc)})
            self.enabled = False
            self.pool = None
            return False

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def healthy(self) -> bool:
        if not self.enabled or self.pool is None:
            return False
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:
            return False

    # ------------------------------------------------------------------ write
    async def save_result(self, result: OCRResult) -> None:
        if not self.enabled or self.pool is None:
            return
        try:
            async with self.pool.acquire() as conn, conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO jobs (job_id, tenant, filename, status, page_count, ocr_pages,
                                      chars, model, engine, attempts, error, full_text, artifacts,
                                      created_at, started_at, completed_at, process_s)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14,$15,$16,$17)
                    ON CONFLICT (job_id) DO UPDATE SET
                        status=EXCLUDED.status, page_count=EXCLUDED.page_count,
                        ocr_pages=EXCLUDED.ocr_pages, chars=EXCLUDED.chars,
                        attempts=EXCLUDED.attempts, error=EXCLUDED.error,
                        full_text=EXCLUDED.full_text, artifacts=EXCLUDED.artifacts,
                        completed_at=EXCLUDED.completed_at, process_s=EXCLUDED.process_s
                    """,
                    result.job_id, result.tenant, result.filename, result.status.value,
                    result.page_count, result.ocr_pages, len(result.full_text), result.model,
                    result.engine, result.attempts, result.error, result.full_text,
                    json.dumps(result.artifacts.model_dump()), result.timings.queued_at,
                    result.timings.started_at, result.timings.completed_at,
                    result.timings.process_s,
                )
                if result.pages:
                    await conn.execute("DELETE FROM pages WHERE job_id = $1", result.job_id)
                    await conn.executemany(
                        """
                        INSERT INTO pages (job_id, idx, source, regions, chars, tokens,
                                           duration_ms, ttft_s, text)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        """,
                        [
                            (result.job_id, p.index, p.source.value, p.regions, p.chars,
                             p.tokens, p.duration_ms, p.ttft_s, p.text)
                            for p in result.pages
                        ],
                    )
        except Exception as exc:  # persistence must not fail the job
            log.warning(
                "postgres write failed", extra={"job_id": result.job_id, "db_error": str(exc)}
            )

    # ------------------------------------------------------------------- read
    async def get_result(self, job_id: str) -> OCRResult | None:
        if not self.enabled or self.pool is None:
            return None
        try:
            async with self.pool.acquire() as conn:
                job = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
                if job is None:
                    return None
                rows = await conn.fetch(
                    "SELECT * FROM pages WHERE job_id = $1 ORDER BY idx", job_id
                )
        except Exception as exc:
            log.warning("postgres read failed", extra={"job_id": job_id, "db_error": str(exc)})
            return None
        return OCRResult(
            job_id=job["job_id"],
            status=JobStatus(job["status"]),
            filename=job["filename"],
            tenant=job["tenant"],
            page_count=job["page_count"],
            model=job["model"],
            engine=job["engine"],
            attempts=job["attempts"],
            error=job["error"],
            full_text=job["full_text"],
            artifacts=Artifacts(**json.loads(job["artifacts"] or "{}")),
            timings=JobTimings(
                queued_at=job["created_at"],
                started_at=job["started_at"],
                completed_at=job["completed_at"],
                process_s=job["process_s"],
            ),
            pages=[
                PageResult(
                    index=r["idx"], text=r["text"], source=r["source"], regions=r["regions"],
                    chars=r["chars"], tokens=r["tokens"], duration_ms=r["duration_ms"],
                    ttft_s=r["ttft_s"],
                )
                for r in rows
            ],
        )

    async def recent(self, tenant: str | None = None, limit: int = 50) -> list[dict]:
        if not self.enabled or self.pool is None:
            return []
        query = (
            "SELECT job_id, tenant, filename, status, page_count, chars, completed_at, process_s "
            "FROM jobs {where} ORDER BY created_at DESC NULLS LAST LIMIT $1"
        )
        try:
            async with self.pool.acquire() as conn:
                if tenant:
                    rows = await conn.fetch(query.format(where="WHERE tenant = $2"), limit, tenant)
                else:
                    rows = await conn.fetch(query.format(where=""), limit)
            return [dict(r) for r in rows]
        except Exception:
            return []
