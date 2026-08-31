"""Storage retention — the sweep that keeps the blob store from growing forever.

Uploads, result JSON and generated artifacts accumulate on every job. Redis job
state expires on its own TTL and PostgreSQL keeps the durable record, but the
files on disk have nothing expiring them, so a long-running deployment fills its
volume and the worker starts failing on writes.

The worker runs :func:`sweep` on a timer. It is deliberately dumb — mtime older
than the retention window, delete — because anything cleverer (asking Redis
whether a job is still live) would couple disk cleanup to a service that may be
down, and the window is days long: nothing in flight is that old.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ocr_serving.common.config import Settings
from ocr_serving.common.logging import get_logger
from ocr_serving.common.storage import LocalBlobStore

log = get_logger(__name__)


@dataclass
class SweepResult:
    removed: dict[str, int] = field(default_factory=dict)
    freed_bytes: int = 0
    remaining_bytes: int = 0

    @property
    def total_removed(self) -> int:
        return sum(self.removed.values())


def sweep(settings: Settings, days: int | None = None) -> SweepResult:
    """Delete blobs older than the retention window. Blocking; call from a thread."""
    days = settings.result_ttl_days if days is None else days
    result = SweepResult()
    if days <= 0:
        return result

    for kind, directory in (
        ("uploads", settings.uploads_dir),
        ("results", settings.results_dir),
        ("artifacts", settings.artifacts_dir),
    ):
        store = LocalBlobStore(directory)
        before = store.usage_bytes()
        try:
            removed = store.purge_older_than(days)
        except OSError as exc:  # a locked or vanished file must not stop the sweep
            log.warning("retention sweep failed", extra={"kind": kind, "sweep_error": str(exc)})
            continue
        after = store.usage_bytes()
        result.removed[kind] = removed
        result.freed_bytes += max(before - after, 0)
        result.remaining_bytes += after
    return result
