"""Searchable-PDF generator (the "Searchable PDF Generator" box).

Takes the original document plus the per-page OCR text and produces a PDF whose
pages look identical but carry a text layer, so Ctrl-F and any downstream
indexer work.

Honest limitation, stated up front: a page-level OCR VLM returns *text*, not
per-word bounding boxes. The text layer is therefore inserted as an invisible
(``render_mode=3``) block covering the page rather than word-positioned under
each glyph. Search, copy-paste and indexing work; text selection will not
highlight the exact word. Word-accurate positioning needs a detector that emits
boxes, which is a different model than the one this repo benchmarks.
"""
from __future__ import annotations

from pathlib import Path

#: Start here and shrink until the page's text fits in the invisible layer.
_FONT_SIZES = (8.0, 6.0, 4.5, 3.0, 2.0, 1.5, 1.0)


def build_searchable_pdf(source: Path, page_texts: list[str], dest: Path) -> Path:
    """Write ``dest`` = source pages + invisible text layer. Returns ``dest``."""
    import pymupdf

    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".pdf":
        doc = pymupdf.open(source)
    else:
        doc = pymupdf.open()
        img = pymupdf.open(source)
        pdf_bytes = img.convert_to_pdf()
        img.close()
        doc.insert_pdf(pymupdf.open("pdf", pdf_bytes))

    try:
        for index, page in enumerate(doc):
            text = page_texts[index] if index < len(page_texts) else ""
            if not text.strip():
                continue
            _insert_invisible(page, text)
        doc.save(str(dest), garbage=3, deflate=True)
    finally:
        doc.close()
    return dest


def _insert_invisible(page, text: str) -> None:
    """Insert text with render_mode=3, shrinking the font until the page fits."""
    rect = page.rect + (18, 18, -18, -18)  # keep a margin so nothing is clipped
    for size in _FONT_SIZES:
        leftover = page.insert_textbox(
            rect, text, fontsize=size, fontname="helv", render_mode=3, align=0
        )
        if leftover >= 0:
            return
    # Still too long at 1 pt: keep what fits rather than dropping the page.
    page.insert_textbox(
        rect, text[:20_000], fontsize=1.0, fontname="helv", render_mode=3, align=0
    )
