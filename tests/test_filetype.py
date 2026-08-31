"""Magic-byte sniffing for uploads."""
from __future__ import annotations

import pytest

from ocr_serving.common.filetype import (
    EXTENSION_KINDS,
    HEAD_BYTES,
    describe,
    matches_extension,
    sniff,
)

SAMPLES = {
    "pdf": b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj",
    "png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    "jpeg": b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01",
    "tiff": b"II\x2a\x00\x08\x00\x00\x00",
    "webp": b"RIFF\x24\x00\x00\x00WEBPVP8 ",
    "bmp": b"BM\x36\x00\x0c\x00\x00\x00\x00\x00",
}


@pytest.mark.parametrize(("kind", "head"), SAMPLES.items())
def test_every_accepted_kind_is_recognised(kind, head):
    assert sniff(head) == kind


def test_big_endian_tiff_is_recognised():
    assert sniff(b"MM\x00\x2a\x00\x00\x00\x08") == "tiff"


@pytest.mark.parametrize(
    "head",
    [
        b"",
        b"not a document at all",
        b"PK\x03\x04zip archive",
        b"RIFF\x24\x00\x00\x00AVI ",       # RIFF, but not WEBP
        b"%PD",                             # truncated signature
    ],
)
def test_unknown_content_is_rejected(head):
    assert sniff(head) is None
    assert describe(head) == "unrecognised"


def test_extension_must_agree_with_content():
    assert matches_extension(".pdf", SAMPLES["pdf"])
    assert matches_extension(".PDF", SAMPLES["pdf"]), "extension check is case-insensitive"
    assert matches_extension(".jpeg", SAMPLES["jpeg"])
    assert matches_extension(".jpg", SAMPLES["jpeg"]), "both spellings map to one kind"
    assert matches_extension(".tif", SAMPLES["tiff"])

    assert not matches_extension(".pdf", SAMPLES["png"])
    assert not matches_extension(".png", SAMPLES["pdf"])
    assert not matches_extension(".exe", SAMPLES["pdf"]), "unknown extension is never a match"
    assert not matches_extension(".pdf", b"")


def test_head_window_is_wide_enough_for_every_signature():
    for kind, head in SAMPLES.items():
        assert sniff(head[:HEAD_BYTES]) == kind


def test_extension_map_is_consistent():
    assert EXTENSION_KINDS[".jpg"] == EXTENSION_KINDS[".jpeg"] == "jpeg"
    assert set(EXTENSION_KINDS.values()) == set(SAMPLES)


async def test_blob_store_captures_the_head(tmp_path):
    from ocr_serving.common.storage import LocalBlobStore

    store = LocalBlobStore(tmp_path)

    async def chunks():
        yield b"%PD"          # signature split across chunk boundaries
        yield b"F-1.7 rest of the document"

    blob = await store.put_stream("doc.pdf", chunks())

    assert blob.head.startswith(b"%PDF-1.7")
    assert len(blob.head) == min(HEAD_BYTES, blob.size)   # short files keep what there is
    assert matches_extension(".pdf", blob.head)


def test_put_bytes_also_captures_the_head(tmp_path):
    from ocr_serving.common.storage import LocalBlobStore

    blob = LocalBlobStore(tmp_path).put_bytes("a.png", SAMPLES["png"] + b"\x00" * 100)
    assert sniff(blob.head) == "png"
