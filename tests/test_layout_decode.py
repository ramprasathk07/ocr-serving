"""Layout detection: raw-tensor decode and the ONNX session wiring.

The decode is the highest-risk arithmetic in the repo — two export layouts,
letterbox padding to undo, a scale to divide out, NMS that must run for one
layout and not the other — and until now only the no-model fallback was
covered. These tests drive it with synthetic tensors, and drive the session
path with a stub session, so no ONNX file is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from ocr_serving.common.schemas import Region
from ocr_serving.workers.layout import (
    CLASS_NAMES,
    LayoutDetector,
    _letterbox,
    decode_detections,
)

PAGE_W, PAGE_H = 800, 1000
INPUT = 1024

TITLE = CLASS_NAMES.index("title")
TEXT = CLASS_NAMES.index("text")
FIGURE = CLASS_NAMES.index("figure")
TABLE = CLASS_NAMES.index("table")


def letterbox_params(width: int = PAGE_W, height: int = PAGE_H, size: int = INPUT):
    """The same scale/pad the detector computes, without building an image."""
    _, scale, pad_x, pad_y = _letterbox(np.zeros((height, width, 3), np.uint8), size)
    return scale, pad_x, pad_y


def to_letterbox(box, scale: float, pad_x: int, pad_y: int) -> list[float]:
    """Page-space box -> the coordinates a model would emit for it."""
    x0, y0, x1, y1 = box
    return [x0 * scale + pad_x, y0 * scale + pad_y, x1 * scale + pad_x, y1 * scale + pad_y]


def v10_tensor(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def v8_tensor(boxes_xyxy, classes, scores) -> np.ndarray:
    """(4 + C, N) with cx, cy, w, h and a per-class score column block."""
    count = len(boxes_xyxy)
    out = np.zeros((4 + len(CLASS_NAMES), count), dtype=np.float32)
    for i, ((x0, y0, x1, y1), cls, score) in enumerate(
        zip(boxes_xyxy, classes, scores, strict=True)
    ):
        out[0, i] = (x0 + x1) / 2
        out[1, i] = (y0 + y1) / 2
        out[2, i] = x1 - x0
        out[3, i] = y1 - y0
        out[4 + cls, i] = score
    return out


def decode(raw: np.ndarray, score_threshold: float = 0.3, iou_threshold: float = 0.45):
    scale, pad_x, pad_y = letterbox_params()
    return decode_detections(
        raw, scale=scale, pad_x=pad_x, pad_y=pad_y,
        width=PAGE_W, height=PAGE_H,
        score_threshold=score_threshold, iou_threshold=iou_threshold,
    )


# ------------------------------------------------------------------ geometry
def test_letterbox_padding_is_undone_exactly():
    scale, pad_x, pad_y = letterbox_params()
    assert pad_x > 0 or pad_y > 0, "a 800x1000 page must be padded into a square"

    original = (100, 200, 500, 600)
    raw = v10_tensor([to_letterbox(original, scale, pad_x, pad_y) + [0.9, TEXT]])

    (region,) = decode(raw)
    assert region.bbox == pytest.approx(original, abs=1)
    assert region.cls == "text"
    assert region.score == pytest.approx(0.9, abs=1e-6)


def test_boxes_are_clamped_to_the_page():
    scale, pad_x, pad_y = letterbox_params()
    # A box the model pushed past both edges of the real page.
    raw = v10_tensor([to_letterbox((-50, -80, PAGE_W + 120, PAGE_H + 90), scale, pad_x, pad_y)
                      + [0.8, TEXT]])

    (region,) = decode(raw)
    assert region.bbox == (0, 0, PAGE_W, PAGE_H)


def test_regions_smaller_than_the_minimum_are_dropped():
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([
        to_letterbox((100, 100, 108, 400), scale, pad_x, pad_y) + [0.9, TEXT],   # 8 px wide
        to_letterbox((100, 500, 400, 508), scale, pad_x, pad_y) + [0.9, TEXT],   # 8 px tall
        to_letterbox((100, 600, 400, 700), scale, pad_x, pad_y) + [0.9, TEXT],   # keep
    ])
    kept = decode(raw)
    assert len(kept) == 1
    # Floor/ceil rounding never shrinks a box below the model's own coordinates.
    x0, y0, x1, y1 = kept[0].bbox
    assert x0 <= 100 and y0 <= 600 and x1 >= 400 and y1 >= 700


# ------------------------------------------------------------------ filtering
def test_low_scores_are_filtered():
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([
        to_letterbox((10, 10, 300, 200), scale, pad_x, pad_y) + [0.10, TEXT],
        to_letterbox((10, 300, 300, 500), scale, pad_x, pad_y) + [0.55, TEXT],
    ])
    kept = decode(raw, score_threshold=0.3)
    assert len(kept) == 1 and kept[0].score == pytest.approx(0.55, abs=1e-6)


def test_figure_and_abandon_classes_carry_no_text():
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([
        to_letterbox((10, 10, 300, 200), scale, pad_x, pad_y) + [0.9, FIGURE],
        to_letterbox((10, 300, 300, 500), scale, pad_x, pad_y) + [0.9, TABLE],
    ])
    kept = decode(raw)
    assert [r.cls for r in kept] == ["table"]


def test_unknown_class_id_degrades_to_text():
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([to_letterbox((10, 10, 300, 200), scale, pad_x, pad_y) + [0.9, 99]])
    assert decode(raw)[0].cls == "text"


def test_empty_output_is_not_an_error():
    assert decode(np.zeros((0, 6), dtype=np.float32)) == []


def test_non_2d_output_is_rejected():
    with pytest.raises(ValueError, match="unexpected layout output shape"):
        decode(np.zeros((2, 3, 4), dtype=np.float32))


# ------------------------------------------------------- v8 layout (with NMS)
def test_v8_layout_is_transposed_and_decoded():
    scale, pad_x, pad_y = letterbox_params()
    original = (120, 240, 520, 640)
    raw = v8_tensor([to_letterbox(original, scale, pad_x, pad_y)], [TITLE], [0.77])

    (region,) = decode(raw)
    assert region.bbox == pytest.approx(original, abs=1)
    assert region.cls == "title"


def test_v8_duplicates_are_suppressed_but_distinct_boxes_survive():
    scale, pad_x, pad_y = letterbox_params()
    a = to_letterbox((100, 100, 400, 300), scale, pad_x, pad_y)
    a_dupe = to_letterbox((104, 104, 404, 304), scale, pad_x, pad_y)   # ~96% IoU
    b = to_letterbox((100, 600, 400, 800), scale, pad_x, pad_y)        # elsewhere

    raw = v8_tensor([a, a_dupe, b], [TEXT, TEXT, TEXT], [0.9, 0.85, 0.8])
    kept = decode(raw)

    assert len(kept) == 2
    tops = sorted(r.bbox[1] for r in kept)
    assert tops[0] == pytest.approx(100, abs=6) and tops[1] == pytest.approx(600, abs=6)


def test_v10_layout_is_not_run_through_nms():
    """Overlapping boxes from an NMS-free export must all survive."""
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([
        to_letterbox((100, 100, 400, 300), scale, pad_x, pad_y) + [0.9, TEXT],
        to_letterbox((104, 104, 404, 304), scale, pad_x, pad_y) + [0.85, TEXT],
    ])
    assert len(decode(raw)) == 2


# ------------------------------------------------------------ session wiring
class StubSession:
    """Stands in for onnxruntime.InferenceSession, recording what it was fed."""

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.blob: np.ndarray | None = None

    def run(self, _outputs, feeds):
        (self.blob,) = feeds.values()
        return [self.output[None]]          # models emit a leading batch axis


def _detector_with(output: np.ndarray) -> tuple[LayoutDetector, StubSession]:
    detector = LayoutDetector(model_path="does/not/exist.onnx")
    session = StubSession(output)
    detector.session = session
    detector.input_name = "images"
    detector.input_size = INPUT
    return detector, session


def test_session_receives_a_normalised_rgb_nchw_blob():
    page = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
    page[:, :, 0] = 10        # distinct blue channel, so RGB/BGR order is observable
    detector, session = _detector_with(np.zeros((0, 6), np.float32))

    detector.detect(page)

    blob = session.blob
    assert blob.shape == (1, 3, INPUT, INPUT)
    assert blob.dtype == np.float32
    assert 0.0 <= blob.min() and blob.max() <= 1.0
    # Channel 2 of the blob is the image's channel 0 (BGR -> RGB), i.e. the 10s.
    centre = blob[0, :, INPUT // 2, INPUT // 2]
    assert centre[2] == pytest.approx(10 / 255, abs=1e-3)
    assert centre[0] == pytest.approx(1.0, abs=1e-3)


def test_detect_returns_page_space_regions_in_reading_order():
    scale, pad_x, pad_y = letterbox_params()
    raw = v10_tensor([
        to_letterbox((420, 420, 780, 560), scale, pad_x, pad_y) + [0.9, TEXT],   # right, lower
        to_letterbox((20, 120, 380, 260), scale, pad_x, pad_y) + [0.9, TEXT],    # left, upper
        to_letterbox((420, 120, 780, 260), scale, pad_x, pad_y) + [0.9, TEXT],   # right, upper
        to_letterbox((20, 420, 380, 560), scale, pad_x, pad_y) + [0.9, TEXT],    # left, lower
    ])
    detector, _ = _detector_with(raw)

    regions = detector.detect(np.full((PAGE_H, PAGE_W, 3), 255, np.uint8))

    assert len(regions) == 4
    assert [r.order for r in regions] == [0, 1, 2, 3]
    # Column-aware: both left-column regions come before the right column.
    assert [r.bbox[0] < PAGE_W / 2 for r in regions] == [True, True, False, False]


def test_a_failing_session_falls_back_to_the_whole_page():
    class Boom:
        def run(self, *_a, **_k):
            raise RuntimeError("onnxruntime exploded")

    detector = LayoutDetector(model_path="does/not/exist.onnx")
    detector.session = Boom()
    detector.input_name = "images"

    regions = detector.detect(np.full((PAGE_H, PAGE_W, 3), 255, np.uint8))

    assert regions == [Region(bbox=(0, 0, PAGE_W, PAGE_H), cls="page", order=0)]


def test_empty_detections_fall_back_to_the_whole_page():
    detector, _ = _detector_with(np.zeros((0, 6), np.float32))
    regions = detector.detect(np.full((PAGE_H, PAGE_W, 3), 255, np.uint8))
    assert len(regions) == 1 and regions[0].cls == "page"
