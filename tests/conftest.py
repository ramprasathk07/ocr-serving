"""Shared test fixtures.

The whole pipeline is exercised without Redis, PostgreSQL, a GPU or a model:
``fakeredis`` backs the queue and event streams, and ``serving.mock.server``
provides an OpenAI-compatible engine over an in-process ASGI transport. That is
what makes ``pytest`` on a laptop meaningful rather than a smoke test of imports.
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _configure_env(tmp_path: Path) -> None:
    os.environ.update(
        OCR_STORAGE_DIR=str(tmp_path / "storage"),
        OCR_API_KEYS="test-key:acme,other-key:globex",
        OCR_POSTGRES_ENABLED="false",
        OCR_LOG_FORMAT="text",
        OCR_LOG_LEVEL="WARNING",
        OCR_SEARCHABLE_PDF="true",
        OCR_RATE_LIMIT_RPS="100",
        OCR_RATE_LIMIT_BURST="200",
        OCR_QUOTA_PAGES_PER_DAY="1000",
        OCR_ENGINE_BASE_URL="http://engine.test/v1",
        OCR_MODEL_ID="mock-ocr-vl",
        OCR_DESKEW="false",          # keep unit tests fast and deterministic
        OCR_WORKER_METRICS_PORT="0",
    )


@pytest.fixture(scope="session", autouse=True)
def _session_env(tmp_path_factory) -> None:
    _configure_env(tmp_path_factory.mktemp("ocr"))


@pytest.fixture
def settings(_session_env):
    from ocr_serving.common.config import get_settings

    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
async def redis_client(settings):
    """Fresh in-memory Redis for each test, installed as the process-wide client."""
    import fakeredis

    from ocr_serving.common import redis_client as rc

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    rc.set_redis(client)
    try:
        yield client
    finally:
        # The gateway lifespan may already have closed it; teardown must not care.
        with contextlib.suppress(Exception):
            await client.flushall()
            await client.aclose()
        rc.set_redis(None)


@pytest.fixture
def engine_transport():
    """ASGI transport wired to the mock engine app."""
    import httpx

    from ocr_serving.serving.mock.server import app as mock_app

    return httpx.ASGITransport(app=mock_app)


@pytest.fixture
def ocr_client(engine_transport):
    """OCRClient whose HTTP layer talks to the in-process mock engine."""
    import httpx

    from ocr_serving.common.engine import OCRClient

    client = OCRClient(base_url="http://engine.test/v1", model="mock-ocr-vl", max_retries=1)
    client._client = httpx.AsyncClient(transport=engine_transport, base_url="http://engine.test/v1")
    return client


@pytest.fixture
def sample_png() -> bytes:
    """A small synthetic 'document' page with dark text-like bars on white."""
    import cv2
    import numpy as np

    img = np.full((600, 420, 3), 255, dtype=np.uint8)
    for row in range(60, 540, 40):
        cv2.rectangle(img, (40, row), (380, row + 14), (30, 30, 30), -1)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture
def blank_png() -> bytes:
    import cv2
    import numpy as np

    return cv2.imencode(".png", np.full((300, 200, 3), 255, dtype=np.uint8))[1].tobytes()


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Two-page PDF: page 1 has a real text layer, page 2 is effectively blank."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    body = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 700), body, fontsize=11, fontname="helv")
    doc.new_page()
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path, sample_png: bytes) -> Path:
    """Single-page PDF with no text layer — forces the OCR path."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 420, 600), stream=sample_png)
    path = tmp_path / "scan.pdf"
    doc.save(str(path))
    doc.close()
    return path
