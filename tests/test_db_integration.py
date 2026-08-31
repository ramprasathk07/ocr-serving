"""PostgreSQL result store, against a real server.

Every other test runs with PostgreSQL disabled, which means the SQL in
``common/db.py`` had never once executed: a typo in the schema or an INSERT
would only show up in production as a "postgres write failed" warning, with the
job completing and the durable record silently missing.

Skipped unless ``OCR_TEST_POSTGRES_DSN`` points at a throwaway database; CI sets
it against a service container.

    docker run --rm -e POSTGRES_PASSWORD=ocr -e POSTGRES_USER=ocr -e POSTGRES_DB=ocr \\
        -p 5432:5432 postgres:16-alpine
    OCR_TEST_POSTGRES_DSN=postgresql://ocr:ocr@localhost:5432/ocr \
        pytest tests/test_db_integration.py
"""
from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

from ocr_serving.common.db import Database
from ocr_serving.common.schemas import (
    Artifacts,
    JobStatus,
    JobTimings,
    OCRResult,
    PageResult,
    PageSource,
)

DSN = os.environ.get("OCR_TEST_POSTGRES_DSN", "")
pytestmark = pytest.mark.skipif(not DSN, reason="OCR_TEST_POSTGRES_DSN is not set")


@pytest.fixture
async def db():
    database = Database(DSN, enabled=True)
    connected = await database.connect()
    assert connected, "could not reach the test database"
    async with database.pool.acquire() as conn:
        await conn.execute("TRUNCATE pages, jobs RESTART IDENTITY CASCADE")
    try:
        yield database
    finally:
        await database.close()


def make_result(job_id: str = "job-1", **overrides) -> OCRResult:
    now = datetime.now(UTC)
    defaults = {
        "job_id": job_id,
        "status": JobStatus.COMPLETED,
        "filename": "contract.pdf",
        "tenant": "acme",
        "page_count": 3,
        "full_text": "page one\n\npage two",
        "model": "PaddlePaddle/PaddleOCR-VL",
        "engine": "http://localhost:8001/v1",
        "attempts": 1,
        "pages": [
            PageResult(index=0, text="page one", source=PageSource.NATIVE, chars=8, regions=0),
            PageResult(
                index=1, text="page two", source=PageSource.OCR, chars=8, regions=2,
                tokens=42, duration_ms=1234.5, ttft_s=0.42,
            ),
            PageResult(index=2, text="", source=PageSource.BLANK, regions=0),
        ],
        "timings": JobTimings(
            queued_at=now, started_at=now, completed_at=now, queue_wait_s=0.1, process_s=12.5
        ),
        "artifacts": Artifacts(
            json_path="/data/a.json", markdown_path="/data/a.md", pdf_path="/data/a.pdf"
        ),
    }
    return OCRResult(**{**defaults, **overrides})


# ------------------------------------------------------------------- schema
async def test_connect_creates_the_schema(db):
    async with db.pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename IN ('jobs', 'pages')"
        )

    names = {row["table_name"] for row in tables}
    assert {"jobs", "pages"} <= names
    assert "jobs_tenant_created_idx" in {row["indexname"] for row in indexes}


async def test_connect_is_idempotent(db):
    assert await db.connect() is True    # CREATE TABLE IF NOT EXISTS, twice


# ---------------------------------------------------------------- roundtrip
async def test_save_and_read_back_a_result(db):
    original = make_result()

    await db.save_result(original)
    stored = await db.get_result("job-1")

    assert stored is not None
    assert stored.job_id == original.job_id
    assert stored.status is JobStatus.COMPLETED
    assert stored.filename == "contract.pdf"
    assert stored.tenant == "acme"
    assert stored.page_count == 3
    assert stored.full_text == original.full_text
    assert stored.model == original.model
    assert stored.artifacts.pdf_path == "/data/a.pdf"
    assert stored.timings.process_s == pytest.approx(12.5)

    assert [p.index for p in stored.pages] == [0, 1, 2]
    assert [p.source for p in stored.pages] == [
        PageSource.NATIVE, PageSource.OCR, PageSource.BLANK
    ]
    ocr_page = stored.pages[1]
    assert ocr_page.tokens == 42
    assert ocr_page.ttft_s == pytest.approx(0.42)
    assert ocr_page.duration_ms == pytest.approx(1234.5)
    assert ocr_page.text == "page two"


async def test_unknown_job_reads_as_none(db):
    assert await db.get_result("never-existed") is None


async def test_unicode_and_null_ttft_survive(db):
    result = make_result(
        job_id="job-unicode",
        full_text="Ünïcödé — ligature ﬁ, CJK 文字, emoji ✅",
        pages=[PageResult(index=0, text="文字 ﬁ ✅", source=PageSource.OCR, ttft_s=None)],
        page_count=1,
    )

    await db.save_result(result)
    stored = await db.get_result("job-unicode")

    assert stored.full_text == result.full_text
    assert stored.pages[0].text == "文字 ﬁ ✅"
    assert stored.pages[0].ttft_s is None


# -------------------------------------------------------------------- upsert
async def test_saving_twice_updates_rather_than_duplicates(db):
    await db.save_result(make_result())
    await db.save_result(
        make_result(status=JobStatus.FAILED, full_text="partial", error="engine died",
                    attempts=3, pages=[PageResult(index=0, text="partial")], page_count=1)
    )

    async with db.pool.acquire() as conn:
        job_rows = await conn.fetchval("SELECT count(*) FROM jobs WHERE job_id = 'job-1'")
        page_rows = await conn.fetchval("SELECT count(*) FROM pages WHERE job_id = 'job-1'")

    assert job_rows == 1
    assert page_rows == 1, "pages are replaced, not appended, on re-save"

    stored = await db.get_result("job-1")
    assert stored.status is JobStatus.FAILED
    assert stored.error == "engine died"
    assert stored.attempts == 3


async def test_chars_and_ocr_pages_are_recorded(db):
    await db.save_result(make_result())

    async with db.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT chars, ocr_pages FROM jobs WHERE job_id = 'job-1'")

    assert row["chars"] == len("page one\n\npage two")
    assert row["ocr_pages"] == 1, "only the page that actually hit the engine counts"


# -------------------------------------------------------------------- listing
async def test_recent_filters_by_tenant_and_respects_the_limit(db):
    for i in range(3):
        await db.save_result(make_result(job_id=f"acme-{i}", tenant="acme"))
    await db.save_result(make_result(job_id="globex-0", tenant="globex"))

    acme = await db.recent("acme", limit=2)
    globex = await db.recent("globex", limit=10)
    everyone = await db.recent(None, limit=10)

    assert len(acme) == 2
    assert {row["tenant"] for row in acme} == {"acme"}
    assert [row["job_id"] for row in globex] == ["globex-0"]
    assert len(everyone) == 4


async def test_deleting_a_job_takes_its_pages(db):
    await db.save_result(make_result())

    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM jobs WHERE job_id = 'job-1'")
        orphans = await conn.fetchval("SELECT count(*) FROM pages WHERE job_id = 'job-1'")

    assert orphans == 0, "ON DELETE CASCADE keeps the pages table from leaking rows"


# ----------------------------------------------------------------- health
async def test_healthy_reports_a_live_connection(db):
    assert await db.healthy() is True


async def test_an_unreachable_server_degrades_instead_of_raising():
    database = Database("postgresql://ocr:ocr@127.0.0.1:1/nope", enabled=True)

    assert await database.connect() is False
    assert database.enabled is False
    assert await database.healthy() is False
    # And the write path stays a no-op rather than exploding mid-job.
    await database.save_result(make_result())
    assert await database.get_result("job-1") is None
    assert await database.recent() == []
