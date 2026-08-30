"""Unit tests for the CPU pipeline stages: preprocess, layout, postprocess, storage."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr_serving.common.schemas import Region
from ocr_serving.common.storage import LocalBlobStore, UploadTooLarge
from ocr_serving.workers import postprocess
from ocr_serving.workers.layout import LayoutDetector, assign_reading_order, crop, tile
from ocr_serving.workers.preprocess import (
    downscale_to_budget,
    estimate_skew,
    is_blank,
    page_hash,
    prepare,
    rotate,
    to_bgr,
    to_png,
)


def _page(width: int = 420, height: int = 600, angle: float = 0.0) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for row in range(60, height - 60, 40):
        cv2.rectangle(img, (40, row), (width - 40, row + 14), (20, 20, 20), -1)
    return rotate(img, angle) if angle else img


# ----------------------------------------------------------------- preprocess
def test_blank_detection_separates_blank_from_text():
    assert is_blank(np.full((200, 200, 3), 255, dtype=np.uint8))
    assert not is_blank(_page())


def test_page_hash_is_stable_and_scale_tolerant():
    img = _page()
    assert page_hash(img) == page_hash(img.copy())
    resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    assert page_hash(img) == page_hash(resized)
    assert page_hash(img) != page_hash(_page(width=300))


def test_downscale_respects_pixel_budget():
    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    out = downscale_to_budget(img, 1_000_000)
    assert out.shape[0] * out.shape[1] <= 1_000_000
    assert downscale_to_budget(img, 0).shape == img.shape


@pytest.mark.parametrize("angle", [-6.0, 4.0])
def test_skew_estimate_returns_the_correction_angle(angle):
    """estimate_skew reports the rotation needed to straighten, i.e. -applied."""
    tilted = _page(angle=angle)
    correction = estimate_skew(tilted)
    assert correction == pytest.approx(-angle, abs=1.5)
    assert abs(estimate_skew(rotate(tilted, correction))) < 1.0


def test_prepare_returns_image_and_skew():
    img, skew = prepare(_page(angle=3.0), do_deskew=True, do_clahe=True)
    assert img.ndim == 3
    assert abs(skew) > 0.3


def test_png_roundtrip():
    img = _page()
    assert to_bgr(to_png(img)).shape == img.shape
    with pytest.raises(ValueError):
        to_bgr(b"not a png")


# --------------------------------------------------------------------- layout
def test_detector_without_model_returns_whole_page():
    detector = LayoutDetector(model_path="does/not/exist.onnx")
    assert not detector.active
    regions = detector.detect(_page())
    assert len(regions) == 1
    assert regions[0].bbox == (0, 0, 420, 600)


def test_reading_order_handles_two_columns():
    regions = [
        Region(bbox=(320, 400, 580, 500)),   # right column, lower
        Region(bbox=(20, 400, 280, 500)),    # left column, lower
        Region(bbox=(320, 100, 580, 200)),   # right column, upper
        Region(bbox=(20, 100, 280, 200)),    # left column, upper
    ]
    ordered = assign_reading_order(regions, page_width=600)
    assert [r.bbox[0] for r in ordered] == [20, 20, 320, 320]
    assert [r.order for r in ordered] == [0, 1, 2, 3]


def test_full_width_region_stays_ahead_of_columns():
    regions = [
        Region(bbox=(20, 300, 280, 400)),
        Region(bbox=(320, 300, 580, 400)),
        Region(bbox=(20, 40, 580, 120)),     # spanning title
        Region(bbox=(20, 420, 280, 500)),
        Region(bbox=(320, 420, 580, 500)),
    ]
    ordered = assign_reading_order(regions, page_width=600)
    assert ordered[0].bbox == (20, 40, 580, 120)


def test_tiling_splits_tall_regions_with_overlap():
    tiles = tile([Region(bbox=(0, 0, 500, 5000))], max_height=1600, overlap=100)
    assert len(tiles) >= 3
    assert all(t.bbox[3] - t.bbox[1] <= 1600 for t in tiles)
    assert tiles[1].bbox[1] < tiles[0].bbox[3]          # overlapping
    assert tiles[-1].bbox[3] == 5000                    # covers the tail
    assert [t.order for t in tiles] == list(range(len(tiles)))


def test_tiling_leaves_small_regions_alone():
    regions = [Region(bbox=(0, 0, 500, 900))]
    assert tile(regions, max_height=1600)[0].bbox == (0, 0, 500, 900)


def test_crop_pads_and_clamps():
    img = _page()
    out = crop(img, Region(bbox=(0, 0, 100, 100)), pad=8)
    assert out.shape[0] <= img.shape[0] and out.shape[1] <= img.shape[1]


# ---------------------------------------------------------------- postprocess
def test_normalize_unwraps_fences_and_fixes_hyphenation():
    raw = "```markdown\nInter-\nnational trade  \n\n\n\nnext\n```"
    assert postprocess.normalize_text(raw) == "International trade\n\nnext"


def test_normalize_expands_ligatures():
    assert postprocess.normalize_text("oﬃce ﬂow") == "office flow"


def test_degenerate_generation_is_truncated():
    text = "real content line\n" + "repeated row | 1 | 2\n" * 40
    cleaned, flagged = postprocess.filter_degenerate(text)
    assert flagged
    assert cleaned.count("repeated row") == 1
    assert "real content line" in cleaned


def test_healthy_text_is_not_flagged():
    text = "\n".join(f"line number {i} with distinct content" for i in range(30))
    cleaned, flagged = postprocess.filter_degenerate(text)
    assert not flagged and cleaned == text


def test_stitch_drops_tile_overlap():
    shared = "the quick brown fox jumps over the lazy dog and keeps running"
    stitched = postprocess.stitch_regions([f"header\n{shared}", f"{shared}\nfooter"])
    assert stitched.count(shared) == 1
    assert stitched.startswith("header") and stitched.endswith("footer")


def test_merge_pages_repairs_split_words():
    assert postprocess.merge_pages(["ends with hyphen-", "ation continues"]) == (
        "ends with hyphenation continues"
    )


def test_markdown_artifact_has_page_anchors():
    md = postprocess.to_markdown("job1", "doc.pdf", ["one", "two"])
    assert md.startswith("# doc.pdf")
    assert md.count("<!-- page") == 2


# -------------------------------------------------------------------- storage
async def test_blob_store_streams_hashes_and_enforces_limits(tmp_path):
    store = LocalBlobStore(tmp_path)

    async def chunks():
        yield b"hello "
        yield b"world"

    blob = await store.put_stream("a.bin", chunks())
    assert blob.size == 11
    assert store.path_for("a.bin").read_bytes() == b"hello world"
    assert blob.sha256 == (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )

    async def big():
        yield b"x" * 100

    with pytest.raises(UploadTooLarge):
        await store.put_stream("b.bin", big(), max_bytes=10)
    assert not store.exists("b.bin")          # partial upload cleaned up


def test_blob_store_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        LocalBlobStore(tmp_path).path_for("../escape.txt")


# -------------------------------------------------------------------- logging
def test_extra_fields_never_crash_the_caller(caplog):
    """`extra={"filename": ...}` would raise KeyError with a stock logger."""
    from ocr_serving.common.logging import get_logger

    log = get_logger("test.safe")
    with caplog.at_level("INFO"):
        log.info("hello", extra={"filename": "doc.pdf", "job_id": "abc", "module": "x"})

    record = caplog.records[-1]
    assert record.filename_ == "doc.pdf"     # renamed, not dropped
    assert record.job_id == "abc"


def test_json_formatter_includes_context_and_extras():
    import json as _json

    from ocr_serving.common.logging import JsonFormatter, job_id_var

    token = job_id_var.set("job-42")
    try:
        import logging

        record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", (), None)
        record.pages = 3
        payload = _json.loads(JsonFormatter("worker").format(record))
    finally:
        job_id_var.reset(token)

    assert payload["service"] == "worker"
    assert payload["job_id"] == "job-42"
    assert payload["pages"] == 3
    assert payload["msg"] == "msg"
