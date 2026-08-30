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
        h, w = img.shape[:2]
        canvas, scale, pad_x, pad_y = _letterbox(img, self.input_size)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: blob})[0]
        raw = np.squeeze(raw)

        # YOLOv10-style export: (N, 6) = x0,y0,x1,y1,score,cls — already NMS-free.
        # YOLOv8-style export: (4+C, N) — transpose, argmax over classes, then NMS.
        if raw.ndim == 2 and raw.shape[-1] == 6:
            boxes, scores, classes = raw[:, :4], raw[:, 4], raw[:, 5].astype(int)
        elif raw.ndim == 2:
            pred = raw.T if raw.shape[0] < raw.shape[1] else raw
            cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            cls_scores = pred[:, 4:]
            classes = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        else:
            raise ValueError(f"unexpected layout output shape {raw.shape}")

        keep = scores >= self.score_threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        if len(boxes) == 0:
            return []
        if raw.shape[-1] != 6:
            sel = _nms(boxes, scores, self.iou_threshold)
            boxes, scores, classes = boxes[sel], scores[sel], classes[sel]

        regions: list[Region] = []
        for (x0, y0, x1, y1), score, cls_id in zip(boxes, scores, classes, strict=False):
            name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "text"
            if name in SKIP_CLASSES:
                continue
            bbox = (
                int(max((x0 - pad_x) / scale, 0)), int(max((y0 - pad_y) / scale, 0)),
                int(min((x1 - pad_x) / scale, w)), int(min((y1 - pad_y) / scale, h)),
            )
            if bbox[2] - bbox[0] < 16 or bbox[3] - bbox[1] < 16:
                continue
            regions.append(Region(bbox=bbox, cls=name, score=float(score)))
        return regions


# ------------------------------------------------------------------- helpers
def assign_reading_order(regions: list[Region], page_width: int) -> list[Region]:
    """Column-aware ordering: split at the page midline, then top-to-bottom.

    A single ``sorted(by y)`` interleaves the two columns of an academic paper
    line by line and produces unreadable output; detecting the two-column case
    from the x-centre distribution fixes the common case cheaply.
    """
    if not regions:
        return regions
    mid = page_width / 2
    centres = [(r.bbox[0] + r.bbox[2]) / 2 for r in regions]
    spans = [(r.bbox[2] - r.bbox[0]) / page_width for r in regions]
    pairs = list(zip(centres, spans, strict=True))
    left = sum(1 for c, s in pairs if c < mid and s < 0.55)
    right = sum(1 for c, s in pairs if c >= mid and s < 0.55)
    two_column = left >= 2 and right >= 2

    def key(r: Region) -> tuple:
        x_centre = (r.bbox[0] + r.bbox[2]) / 2
        width_frac = (r.bbox[2] - r.bbox[0]) / page_width
        column = 0 if (not two_column or width_frac >= 0.55 or x_centre < mid) else 1
        return (column, r.bbox[1], r.bbox[0])

    ordered = sorted(regions, key=key)
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
