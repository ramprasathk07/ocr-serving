"""CPU preprocessing — the "Preprocessing Worker" column of planflow-2.

Order matters and is deliberate:

1. **downscale to the encoder budget** first, so every later filter is cheap;
2. **deskew** before layout detection, because a 3° tilt widens every box and
   merges adjacent columns;
3. **CLAHE** on luminance only (contrast, not colour) — helps scans, is a no-op
   on already-clean digital pages;
4. **denoise** last and off by default: ``fastNlMeansDenoising`` costs more than
   the OCR call itself on a 3060 and only pays off on genuinely noisy scans.

Blank detection and the perceptual duplicate hash run before any of it — they
exist to *avoid* GPU work, so they must be the cheapest thing in the path.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def to_bgr(png_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("undecodable image")
    return img


def to_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("png encode failed")
    return buf.tobytes()


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------- gate checks
def is_blank(img: np.ndarray, std_threshold: float = 4.0) -> bool:
    """A page with almost no luminance variance carries no text."""
    return float(to_gray(img).std()) < std_threshold


def page_hash(img: np.ndarray) -> str:
    """Perceptual hash of a 64x64 thumbnail — tolerant of rescaling and light noise."""
    thumb = cv2.resize(to_gray(img), (64, 64), interpolation=cv2.INTER_AREA)
    bits = (thumb > thumb.mean()).astype(np.uint8) * 255
    return hashlib.sha1(bits.tobytes()).hexdigest()


# ------------------------------------------------------------------ geometry
def downscale_to_budget(img: np.ndarray, max_px: int) -> np.ndarray:
    """Cap total pixels; vision encoders tile anything larger anyway."""
    h, w = img.shape[:2]
    if max_px <= 0 or h * w <= max_px:
        return img
    scale = (max_px / (h * w)) ** 0.5
    return cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                      interpolation=cv2.INTER_AREA)


def estimate_skew(img: np.ndarray, max_angle: float = 15.0) -> float:
    """Rotation in degrees needed to straighten the page (feed straight to :func:`rotate`).

    Works on a downscaled binary mask: dilate glyphs into line blobs, then take
    the median angle of the large blobs' minimum-area rectangles. Median beats
    the single global ``minAreaRect`` used by most snippets, which a stray
    border artifact can swing by 40°.
    """
    gray = to_gray(img)
    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue
        (_, _), (bw, bh), angle = cv2.minAreaRect(contour)
        if min(bw, bh) < 5 or max(bw, bh) < 40:
            continue
        if bw < bh:                      # normalise to the long side being horizontal
            angle += 90.0
        if abs(angle) <= max_angle:
            angles.append(angle)
    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas and padding with white."""
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(
        img, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def deskew(
    img: np.ndarray, max_angle: float = 15.0, min_angle: float = 0.3
) -> tuple[np.ndarray, float]:
    """Straighten the page; skips the warp when the tilt is not worth the cost."""
    angle = estimate_skew(img, max_angle)
    if abs(angle) < min_angle:
        return img, 0.0
    return rotate(img, angle), angle


# ------------------------------------------------------------------- filters
def clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Local contrast on the L channel — cheap, helps scans, harmless on digital pages."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l_chan)
    return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means. Expensive (~1 s per page) — gate behind OCR_DENOISE=true."""
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def is_scanned(img: np.ndarray) -> bool:
    """Heuristic: scans have a noisy, non-saturated background; digital renders do not."""
    gray = to_gray(img)
    noise = float(cv2.absdiff(gray, cv2.medianBlur(gray, 3)).mean())
    return noise > 1.5


# ------------------------------------------------------------------ pipeline
def prepare(
    img: np.ndarray,
    *,
    max_px: int = 4_000_000,
    do_deskew: bool = True,
    deskew_max_angle: float = 15.0,
    do_clahe: bool = True,
    do_denoise: bool = False,
) -> tuple[np.ndarray, float]:
    """Full page preprocess. Returns ``(image, applied_skew_degrees)``."""
    img = downscale_to_budget(img, max_px)
    skew = 0.0
    if do_deskew:
        img, skew = deskew(img, deskew_max_angle)
    if do_denoise:
        img = denoise(img)
    if do_clahe:
        img = clahe(img)
    return img, skew
