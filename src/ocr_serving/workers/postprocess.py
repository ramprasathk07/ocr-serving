"""Post-processing — text normalisation, confidence filtering, stitching.

The "Text Normalizer / Confidence Filter / Patch Stitcher / Page Stitcher /
Document Merger" tail of planflow-2. Everything here is deterministic string
work; it never calls the model.

The two non-obvious pieces:

* **degeneration filter.** Small OCR VLMs loop — a smudged table turns into the
  same row emitted 200 times until ``max_tokens`` runs out. Shipping that as
  "text" is worse than shipping nothing, so a repetition ratio check truncates
  the tail and flags the page.
* **overlap-aware tile stitching.** Tiles are cut with a pixel overlap, so
  consecutive tiles repeat a line or two. Joining them naively duplicates that
  text; the stitcher finds the longest suffix/prefix match and drops it.
"""
from __future__ import annotations

import re
import unicodedata

#: Fence the model sometimes wraps its answer in.
_FENCE = re.compile(r"^\s*```(?:markdown|md|text)?\s*\n(.*?)\n?```\s*$", re.S)
#: Line-end hyphenation: "inter-\nnational" -> "international".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)
_BLANK_LINES = re.compile(r"\n{3,}")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize_text(text: str) -> str:
    """NFKC, unwrap code fences, repair hyphenation, tidy whitespace."""
    if not text:
        return ""
    if (match := _FENCE.match(text)):
        text = match.group(1)
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _TRAILING_WS.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def repetition_ratio(text: str, window: int = 30) -> float:
    """Share of lines that are duplicates of an earlier line in the window."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    if len(lines) < 6:
        return 0.0
    seen: dict[str, int] = {}
    repeats = 0
    for i, line in enumerate(lines):
        prev = seen.get(line)
        if prev is not None and i - prev <= window:
            repeats += 1
        seen[line] = i
    return repeats / len(lines)


def filter_degenerate(text: str, threshold: float = 0.5) -> tuple[str, bool]:
    """Truncate a looping generation at its first repeated line.

    Returns ``(text, flagged)``. Flagged pages keep the good prefix so a partial
    result still reaches the user, with the loop cut off.
    """
    if repetition_ratio(text) < threshold:
        return text, False
    seen: set[str] = set()
    kept: list[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen and len(key) > 3:
            break
        if key:
            seen.add(key)
        kept.append(line)
    return "\n".join(kept).strip(), True


def _overlap_len(a: str, b: str, max_check: int = 400) -> int:
    """Longest suffix of ``a`` that is also a prefix of ``b``."""
    a_tail = a[-max_check:]
    limit = min(len(a_tail), len(b))
    for size in range(limit, 20, -1):
        if a_tail[-size:] == b[:size]:
            return size
    return 0


def stitch_regions(texts: list[str], dedupe_overlap: bool = True) -> str:
    """Join region/tile texts in reading order, dropping tile overlap duplicates."""
    parts = [normalize_text(t) for t in texts if t and t.strip()]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        overlap = _overlap_len(out, part) if dedupe_overlap else 0
        out = f"{out}\n\n{part[overlap:].lstrip()}" if overlap else f"{out}\n\n{part}"
    return out.strip()


def merge_pages(page_texts: list[str], separator: str = "\n\n") -> str:
    """Concatenate pages, repairing words hyphenated across the page break."""
    kept = [t.strip() for t in page_texts if t and t.strip()]
    if not kept:
        return ""
    merged = kept[0]
    for text in kept[1:]:
        if merged.endswith("-") and text[:1].isalpha():
            merged = merged[:-1] + text
        else:
            merged = f"{merged}{separator}{text}"
    return merged.strip()


def to_markdown(job_id: str, filename: str, page_texts: list[str]) -> str:
    """Human-readable artifact with page anchors, for the ``.md`` download."""
    header = f"# {filename or job_id}\n\n"
    body = "\n\n".join(
        f"<!-- page {i + 1} -->\n\n{text.strip()}"
        for i, text in enumerate(page_texts)
        if text and text.strip()
    )
    return header + body + "\n"
