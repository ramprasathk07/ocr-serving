"""Blob storage — the "Object Storage" box in the architecture diagrams.

Local-filesystem implementation behind an S3-shaped interface (``put``/``open``/
``delete`` on an opaque key), so swapping in MinIO or S3 later touches this file
only. Uploads are streamed to disk in chunks and hashed as they land: a 400 MB
PDF never sits in the gateway heap, and the content hash gives free idempotency
for repeat submissions.
"""
from __future__ import annotations

import hashlib
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK = 1 << 20  # 1 MiB


class UploadTooLarge(Exception):
    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


@dataclass(slots=True)
class StoredBlob:
    key: str
    path: Path
    size: int
    sha256: str


class LocalBlobStore:
    """Content-addressable-ish local store rooted at ``base``."""

    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        # Keys are generated internally (job id + suffix); reject traversal anyway.
        if "\\" in key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"illegal blob key {key!r}")
        return self.base / key

    async def put_stream(
        self, key: str, chunks: AsyncIterator[bytes], max_bytes: int | None = None
    ) -> StoredBlob:
        """Stream an async byte iterator to disk, enforcing ``max_bytes``."""
        dest = self.path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        try:
            with dest.open("wb") as fh:
                async for chunk in chunks:
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise UploadTooLarge(max_bytes)
                    digest.update(chunk)
                    fh.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return StoredBlob(key=key, path=dest, size=size, sha256=digest.hexdigest())

    def put_bytes(self, key: str, data: bytes) -> StoredBlob:
        dest = self.path_for(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return StoredBlob(key, dest, len(data), hashlib.sha256(data).hexdigest())

    def open(self, key: str) -> BinaryIO:
        return self.path_for(key).open("rb")

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def purge_older_than(self, days: int) -> int:
        """Retention sweep; returns the number of blobs removed."""
        import time

        cutoff = time.time() - days * 86_400
        removed = 0
        for path in self.base.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    def usage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.base.rglob("*") if p.is_file())

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.base).free
