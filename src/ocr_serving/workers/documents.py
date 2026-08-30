"""Document intake — PDF metadata, page classification, incremental rendering.

Covers the "PDF Intake / Metadata Extraction / Per Page Classifier / Native Text
Extractor / Incremental PDF Renderer" chain from planflow-2.

Two decisions here matter more than anything downstream:

* **native text extraction.** A digitally-generated PDF page already contains
  its text. Running a 1B VLM over it is slower *and* less accurate than reading
  the text layer. Those pages skip the GPU entirely, which is the single largest
  throughput win in the pipeline on real-world mixed corpora.
* **adaptive DPI.** Rendering a 600 dpi scan at 150 dpi throws away glyph
  detail; rendering a vector page at 300 dpi wastes encoder tokens. The renderer
  reads the embedded image resolution and picks a DPI inside a configured band.

Rendering is synchronous and CPU-bound; the worker drives it from a thread.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

PDF_SUFFIX = ".pdf"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp"}

#: Collapse the run-on whitespace pymupdf leaves between columns.
_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class UnreadableDocument(Exception):
    """Corrupt, encrypted, or unsupported input — a permanent (non-retryable) failure."""


@dataclass(slots=True)
class RenderedPage:
    index: int
    png: bytes | None          # None when the native text layer was used
    dpi: int
    width: int = 0
    height: int = 0
    native_text: str | None = None


@dataclass
class DocumentMeta:
    filename: str = ""
    pages: int = 0
    kind: str = "image"        # pdf | image
    title: str = ""
    producer: str = ""
    encrypted: bool = False
    extra: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    return _BLANK_LINES.sub("\n\n", _WS.sub(" ", text)).strip()


class Document:
    """Lazily opened document that yields pages one at a time.

    Streaming matters for the product story: the client sees page 1 while page
    40 is still being rendered, and peak memory is one page, not one document.
    """

    def __init__(
        self,
        path: Path,
        *,
        render_dpi: int = 150,
        adaptive_dpi: bool = True,
        min_dpi: int = 110,
        max_dpi: int = 250,
        max_pages: int = 200,
        native_text_enabled: bool = True,
        native_text_min_chars: int = 200,
    ) -> None:
        self.path = Path(path)
        self.render_dpi = render_dpi
        self.adaptive_dpi = adaptive_dpi
        self.min_dpi = min_dpi
        self.max_dpi = max_dpi
        self.max_pages = max_pages
        self.native_text_enabled = native_text_enabled
        self.native_text_min_chars = native_text_min_chars
        self._doc = None
        self.meta = DocumentMeta(filename=self.path.name)

        suffix = self.path.suffix.lower()
        if suffix == PDF_SUFFIX:
            self.meta.kind = "pdf"
            self._open_pdf()
        elif suffix in IMAGE_SUFFIXES:
            self.meta.kind = "image"
            self.meta.pages = 1
        else:
            raise UnreadableDocument(f"unsupported file type {suffix!r}")

    # ------------------------------------------------------------------- open
    def _open_pdf(self) -> None:
        import pymupdf

        try:
            self._doc = pymupdf.open(self.path)
        except Exception as exc:
            raise UnreadableDocument(f"corrupt PDF: {exc}") from exc
        if self._doc.needs_pass:
            self._doc.close()
            raise UnreadableDocument("encrypted PDF: password required")
        info = self._doc.metadata or {}
        self.meta.pages = min(self._doc.page_count, self.max_pages)
        self.meta.title = (info.get("title") or "")[:200]
        self.meta.producer = (info.get("producer") or "")[:200]
        self.meta.extra = {"total_pages": self._doc.page_count}

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    def __enter__(self) -> Document:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        return self.meta.pages

    # ------------------------------------------------------------------- dpi
    def _dpi_for(self, page) -> int:
        """Match render DPI to the page's own resolution, clamped to the band."""
        if not self.adaptive_dpi:
            return self.render_dpi
        try:
            width_pt = page.rect.width or 612.0
            best = 0
            for img in page.get_images(full=True):
                pixel_width = img[2]
                if pixel_width:
                    best = max(best, int(pixel_width / (width_pt / 72.0)))
            if best:
                return max(self.min_dpi, min(self.max_dpi, best))
        except Exception:
            pass
        return self.render_dpi

    # ----------------------------------------------------------------- pages
    def iter_pages(self) -> Iterator[RenderedPage]:
        if self.meta.kind == "image":
            yield from self._iter_image()
        else:
            yield from self._iter_pdf()

    def _iter_image(self) -> Iterator[RenderedPage]:
        yield RenderedPage(index=0, png=self.path.read_bytes(), dpi=self.render_dpi)

    def _iter_pdf(self) -> Iterator[RenderedPage]:
        assert self._doc is not None
        for index in range(self.page_count):
            page = self._doc.load_page(index)

            if self.native_text_enabled:
                text = clean_text(page.get_text("text") or "")
                if len(text) >= self.native_text_min_chars:
                    yield RenderedPage(
                        index=index, png=None, dpi=0, native_text=text,
                        width=int(page.rect.width), height=int(page.rect.height),
                    )
                    continue

            dpi = self._dpi_for(page)
            pixmap = page.get_pixmap(dpi=dpi)
            yield RenderedPage(
                index=index, png=pixmap.tobytes("png"), dpi=dpi,
                width=pixmap.width, height=pixmap.height,
            )


def probe_page_count(path: Path, max_pages: int = 200) -> int:
    """Cheap page count for quota checks, without rendering anything."""
    if Path(path).suffix.lower() != PDF_SUFFIX:
        return 1
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                return 0
            return min(doc.page_count, max_pages)
    except Exception:
        return 0
