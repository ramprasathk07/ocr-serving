"""Storage retention sweep.

Without this the blob store grows until the volume fills and the worker starts
failing on writes — a slow failure that only shows up in production.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ocr_serving.common.retention import SweepResult, sweep

DAY = 86_400


def age(path: Path, days: float) -> None:
    """Backdate a file so the sweep sees it as old."""
    when = time.time() - days * DAY
    os.utime(path, (when, when))


@pytest.fixture
def storage(settings, tmp_path, monkeypatch):
    """Point the settings at a scratch storage tree with the standard subdirs."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    for directory in (settings.uploads_dir, settings.results_dir, settings.artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


def test_old_blobs_go_and_recent_ones_stay(storage):
    old_upload = storage.uploads_dir / "old.pdf"
    old_upload.write_bytes(b"x" * 2048)
    age(old_upload, 30)

    fresh_upload = storage.uploads_dir / "fresh.pdf"
    fresh_upload.write_bytes(b"y" * 512)

    old_result = storage.results_dir / "old.json"
    old_result.write_text("{}")
    age(old_result, 30)

    old_artifact = storage.artifacts_dir / "old.pdf"
    old_artifact.write_bytes(b"z" * 1024)
    age(old_artifact, 30)

    result = sweep(storage, days=7)

    assert result.removed == {"uploads": 1, "results": 1, "artifacts": 1}
    assert result.total_removed == 3
    assert result.freed_bytes >= 2048 + 1024
    assert not old_upload.exists() and not old_result.exists() and not old_artifact.exists()
    assert fresh_upload.exists(), "a blob inside the window must survive"
    assert result.remaining_bytes == 512


def test_a_blob_just_inside_the_window_survives(storage):
    borderline = storage.uploads_dir / "borderline.pdf"
    borderline.write_bytes(b"x")
    age(borderline, 6.9)          # window is 7 days

    result = sweep(storage, days=7)

    assert result.total_removed == 0
    assert borderline.exists()


def test_zero_days_disables_the_sweep(storage):
    ancient = storage.uploads_dir / "ancient.pdf"
    ancient.write_bytes(b"x")
    age(ancient, 900)

    result = sweep(storage, days=0)

    assert result.total_removed == 0
    assert ancient.exists(), "days=0 means retention is off, not delete everything"


def test_sweep_on_an_empty_store_is_a_no_op(storage):
    result = sweep(storage, days=7)
    assert result.total_removed == 0
    assert result.freed_bytes == 0


async def test_worker_registers_the_retention_loop(settings, redis_client, monkeypatch):
    """The loop has to actually be scheduled — that was the gap this closes."""
    import asyncio

    from ocr_serving.workers.cpu_worker import Worker

    monkeypatch.setattr(settings, "retention_sweep_hours", 0.0001)   # ~0.36 s
    worker = Worker(settings)
    await worker.client.aclose()

    calls: list[object] = []

    def fake_sweep(config) -> SweepResult:
        calls.append(config)
        return SweepResult(removed={"uploads": 2}, freed_bytes=10, remaining_bytes=5)

    monkeypatch.setattr("ocr_serving.workers.cpu_worker.retention_sweep", fake_sweep)

    task = asyncio.create_task(worker._retention_loop())
    await asyncio.sleep(0.8)
    task.cancel()

    assert calls, "the retention loop never invoked the sweep"


async def test_retention_loop_exits_when_disabled(settings, redis_client, monkeypatch):
    import asyncio

    from ocr_serving.workers.cpu_worker import Worker

    monkeypatch.setattr(settings, "retention_sweep_hours", 0.0)
    worker = Worker(settings)
    await worker.client.aclose()

    await asyncio.wait_for(worker._retention_loop(), timeout=1)   # returns immediately
