"""Layout analysis — the "DocYOLO ONNX Layout Detection" column.

Region segmentation, table/text split, tiling of oversized regions and reading
order, all on CPU via onnxruntime so the GPU stays free for the VLM.

The detector is optional by design. With ``models/doclayout_yolo.onnx`` present
it segments the page; without it the page is emitted as a single region. That is
not a stub — for a page-level OCR VLM a whole page is a perfectly valid unit of
work, and it keeps the pipeline runnable on a machine that has no model file.
Tiling still applies in both modes, because an A0 poster or a stitched receipt
strip will otherwise blow past the vision encoder budget.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from ocr_serving.common.logging import get_logger
from ocr_serving.common.schemas import Region

log = get_logger(__name__)

#: DocLayout-YOLO (DocStructBench) class order.
CLASS_NAMES = [
    "title", "text", "abandon", "figure", "figure_caption",
    "table", "table_caption", "table_footnote", "formula", "formula_caption",
]
#: Regions of these classes carry no text worth an OCR call.
SKIP_CLASSES = {"abandon", "figure"}


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    idxs = cv2.dnn.NMSBoxes(
        [[float(x0), float(y0), float(x1 - x0), float(y1 - y0)] for x0, y0, x1, y1 in boxes],
        scores.astype(float).tolist(), 0.0, iou_threshold,
    )
    if len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


#: Smallest region worth an engine call, in page pixels.
MIN_REGION_PX = 16


def decode_detections(
    raw: np.ndarray,
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
    score_threshold: float,
    iou_threshold: float,
) -> list[Region]:
    """Turn a raw YOLO output tensor into page-space regions.

    Kept separate from the session call because this is where the mistakes live:
    two different export layouts, letterbox padding to undo, a scale to divide
    out, and NMS that must run for one layout and must not for the other.

    Handles both exports seen in the wild:

    * **v10-style** ``(N, 6)`` — ``x0, y0, x1, y1, score, class``, already
      NMS-free, boxes in letterbox pixel space;
    * **v8-style** ``(4 + C, N)`` — ``cx, cy, w, h`` plus per-class scores,
      needing a transpose, an argmax and NMS.
    """
    if raw.ndim != 2:
        raise ValueError(f"unexpected layout output shape {raw.shape}")

    feature_axis_len = 4 + len(CLASS_NAMES)
    if raw.shape[-1] == 6 and raw.shape[0] != feature_axis_len:
        boxes = raw[:, :4].astype(np.float32)
        scores = raw[:, 4].astype(np.float32)
        classes = raw[:, 5].astype(int)
        needs_nms = False
    else:
        # The feature axis is the one that matches 4 + C; fall back to the
        # shorter axis, which is what every real export has.
        pred = raw.T if raw.shape[0] == feature_axis_len or raw.shape[0] < raw.shape[1] else raw
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        cls_scores = pred[:, 4:]
        classes = cls_scores.argmax(axis=1)
        scores = cls_scores.max(axis=1).astype(np.float32)
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        needs_nms = True

    keep = scores >= score_threshold
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if len(boxes) == 0:
        return []
    if needs_nms:
        sel = _nms(boxes, scores, iou_threshold)
        boxes, scores, classes = boxes[sel], scores[sel], classes[sel]

    regions: list[Region] = []
    for (x0, y0, x1, y1), score, cls_id in zip(boxes, scores, classes, strict=True):
        name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "text"
        if name in SKIP_CLASSES:
            continue
        # Floor the near corner and ceil the far one: truncating both would bias
        # every box up and left by a pixel and clip ascenders off the top line.
        bbox = (
            max(int(math.floor((x0 - pad_x) / scale)), 0),
            max(int(math.floor((y0 - pad_y) / scale)), 0),
            min(int(math.ceil((x1 - pad_x) / scale)), width),
            min(int(math.ceil((y1 - pad_y) / scale)), height),
        )
        if bbox[2] - bbox[0] < MIN_REGION_PX or bbox[3] - bbox[1] < MIN_REGION_PX:
            continue
        regions.append(Region(bbox=bbox, cls=name, score=float(score)))
    return regions


class LayoutDetector:
    """DocLayout-YOLO ONNX wrapper with a whole-page fallback."""

    def __init__(
        self,
        model_path: Path | str = Path("models/doclayout_yolo.onnx"),
        score_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        input_size: int = 1024,
        enabled: bool = True,
    ) -> None:
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.session = None
        path = Path(model_path)
        if enabled and path.exists():
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                shape = self.session.get_inputs()[0].shape
                if isinstance(shape[-1], int):
                    self.input_size = shape[-1]
                log.info("layout model loaded", extra={"path": str(path), "size": self.input_size})
            except Exception as exc:
                log.warning("layout model failed to load", extra={"layout_error": str(exc)})
                self.session = None
        elif enabled:
            log.info("no layout model, using whole-page regions", extra={"path": str(path)})

    @property
    def active(self) -> bool:
        return self.session is not None

    # ------------------------------------------------------------------ infer
    def detect(self, img: np.ndarray) -> list[Region]:
        h, w = img.shape[:2]
        if self.session is None:
            return [Region(bbox=(0, 0, w, h), cls="page", order=0)]
        try:
            regions = self._detect_onnx(img)
        except Exception as exc:
            log.warning("layout inference failed, falling back", extra={"layout_error": str(exc)})
            regions = []
        if not regions:
            return [Region(bbox=(0, 0, w, h), cls="page", order=0)]
        return assign_reading_order(regions, w)

    def _detect_onnx(self, img: np.ndarray) -> list[Region]:
        canvas, scale, pad_x, pad_y = _letterbox(img, self.input_size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: blob})[0]
        return decode_detections(
            np.squeeze(raw),
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            width=img.shape[1],
            height=img.shape[0],
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        )


# ------------------------------------------------------------------- helpers
def _widest_gap(regions: list[Region], axis: int) -> tuple[float, float] | None:
    """Widest empty band across ``axis`` (0 = x, 1 = y) that no region crosses.

    Returns ``(gap_width, cut_position)``, or ``None`` when the projections
    cover the axis with no gap at all.
    """
    intervals = sorted((r.bbox[axis], r.bbox[axis + 2]) for r in regions)
    best: tuple[float, float] | None = None
    reach = intervals[0][1]
    for start, end in intervals[1:]:
        if start > reach:                      # nothing occupies [reach, start]
            width = start - reach
            if best is None or width > best[0]:
                best = (width, (reach + start) / 2)
        reach = max(reach, end)
    return best


def _xy_cut(regions: list[Region], min_gap: float, depth: int = 0) -> list[Region]:
    """Recursive XY-cut: split on the widest empty band, vertical or horizontal.

    Cutting on whichever gap is wider is what makes this work for both page
    shapes without a special case. A two-column article has a gutter running the
    full height, so the vertical gap wins and the left column is emitted before
    the right. A page broken up by a full-width table or title has no such
    gutter, so the horizontal gap wins and the page is read band by band.
    """
    if len(regions) <= 1 or depth >= 16:
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

    vertical = _widest_gap(regions, axis=0)
    horizontal = _widest_gap(regions, axis=1)
    candidates = [c for c in ((vertical, 0), (horizontal, 1)) if c[0] and c[0][0] >= min_gap]
    if not candidates:
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

    (_, cut), axis = max(candidates, key=lambda c: c[0][0])
    near = [r for r in regions if (r.bbox[axis] + r.bbox[axis + 2]) / 2 < cut]
    far = [r for r in regions if (r.bbox[axis] + r.bbox[axis + 2]) / 2 >= cut]
    if not near or not far:                    # degenerate split, stop recursing
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

    return _xy_cut(near, min_gap, depth + 1) + _xy_cut(far, min_gap, depth + 1)


def assign_reading_order(regions: list[Region], page_width: int) -> list[Region]:
    """Order regions the way a person reads the page.

    Sorting by ``y`` alone interleaves the two columns of an academic paper line
    by line and produces unreadable output. The previous fix — assign every
    region to a column by its centre relative to the page midline — broke on the
    first real page it saw: a centred title 54% of the page wide fell just under
    the "full width" threshold, was filed as a right-column element, and came
    out fifth instead of first.

    XY-cut has no such threshold. It splits the page on whichever empty band is
    widest and recurses, so bands and columns fall out of the geometry itself.
    """
    if not regions:
        return regions

    ordered = _xy_cut(regions, min_gap=max(page_width * 0.01, 8))
    for i, region in enumerate(ordered):
        region.order = i
    return ordered


def tile(regions: list[Region], max_height: int, overlap: int = 64) -> list[Region]:
    """Split regions taller than the encoder budget into overlapping tiles."""
    out: list[Region] = []
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        height = y1 - y0
        if max_height <= 0 or height <= max_height:
            out.append(region)
            continue
        step = max(max_height - overlap, max_height // 2)
        top = y0
        while top < y1:
            bottom = min(top + max_height, y1)
            out.append(Region(bbox=(x0, top, x1, bottom), cls=region.cls,
                              order=region.order, score=region.score))
            if bottom >= y1:
                break
            top += step
    for i, region in enumerate(out):
        region.order = i
    return out


def crop(img: np.ndarray, region: Region, pad: int = 4) -> np.ndarray:
    """Crop with a small margin — tight boxes clip ascenders and descenders."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region.bbox
    x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
    x1, y1 = min(x1 + pad, w), min(y1 + pad, h)
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]
