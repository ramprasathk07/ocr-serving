"""Content sniffing for uploads.

The gateway used to trust the filename: anything ending in ``.pdf`` was handed
to pymupdf and anything ending in ``.png`` to OpenCV. A mislabelled — or
deliberately misnamed — file then failed deep inside a parser instead of at the
door, which is both a worse error message and a wider attack surface than it
needs to be.

Magic numbers only, from the first few bytes. This is a gate, not a full
identification library: it answers "does the content match the extension the
caller claimed", and everything it does not recognise is rejected.
"""
from __future__ import annotations

#: Bytes needed to recognise every format below (WEBP needs 12).
HEAD_BYTES = 32

#: Canonical kind -> the extensions the API accepts for it.
KIND_EXTENSIONS: dict[str, set[str]] = {
    "pdf": {".pdf"},
    "png": {".png"},
    "jpeg": {".jpg", ".jpeg"},
    "tiff": {".tif", ".tiff"},
    "webp": {".webp"},
    "bmp": {".bmp"},
}
EXTENSION_KINDS: dict[str, str] = {
    ext: kind for kind, exts in KIND_EXTENSIONS.items() for ext in exts
}


def sniff(head: bytes) -> str | None:
    """Identify a document/image kind from its leading bytes, or ``None``."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith((b"II\x2a\x00", b"MM\x00\x2a")):
        return "tiff"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"BM"):
        return "bmp"
    return None


def matches_extension(suffix: str, head: bytes) -> bool:
    """True when the content is the kind the file extension claims."""
    expected = EXTENSION_KINDS.get(suffix.lower())
    return expected is not None and sniff(head) == expected


def describe(head: bytes) -> str:
    """Human-readable content kind for an error message."""
    return sniff(head) or "unrecognised"
