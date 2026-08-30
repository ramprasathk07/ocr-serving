"""Build the benchmark corpus: PDFs -> page PNGs at a fixed DPI, plus a manifest.

    python benchmarks/build_corpus.py --pdf-dir ~/papers --max-pages 1000

With no ``--pdf-dir`` it downloads a handful of well-known arXiv papers (few
files, sequential, with a delay — be polite to arXiv). Text-heavy pages with
tables and figures are what an OCR VLM should be measured on.

Outputs:
    benchmarks/corpus/pages/doc000_p000.png    rendered pages
    benchmarks/corpus/manifest.json            page list + the fixed eval subset

The manifest pins the eval subset (seed 42) so every serving stack is measured
on byte-identical inputs even if the corpus is later extended.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import httpx
import pymupdf

OUT_DIR = Path("benchmarks/corpus/pages")
MANIFEST = Path("benchmarks/corpus/manifest.json")
EVAL_SEED = 42
EVAL_PAGES = 20

# Long, text+table+figure heavy papers — good OCR variety.
ARXIV_IDS = [
    "1706.03762", "2005.14165", "2302.13971", "2303.08774", "1810.04805",
    "2203.02155", "2106.09685", "2201.11903", "1512.03385", "2010.11929",
]


def download_arxiv(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for arxiv_id in ARXIV_IDS:
            path = dest / f"{arxiv_id}.pdf"
            if not path.exists():
                print(f"downloading arXiv:{arxiv_id}")
                path.write_bytes(client.get(f"https://arxiv.org/pdf/{arxiv_id}").content)
                time.sleep(3)  # politeness
            paths.append(path)
    return paths


def render(pdfs: list[Path], out_dir: Path, max_pages: int, dpi: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for doc_index, pdf in enumerate(pdfs):
        if len(names) >= max_pages:
            break
        with pymupdf.open(pdf) as doc:
            for page_index, page in enumerate(doc):
                if len(names) >= max_pages:
                    break
                out = out_dir / f"doc{doc_index:03d}_p{page_index:03d}.png"
                if not out.exists():
                    page.get_pixmap(dpi=dpi).save(out)
                names.append(out.name)
        print(f"{pdf.name}: {len(names)} pages so far")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", type=Path, default=None)
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--eval-pages", type=int, default=EVAL_PAGES)
    args = ap.parse_args()

    pdfs = (
        sorted(args.pdf_dir.glob("*.pdf"))
        if args.pdf_dir
        else download_arxiv(Path("benchmarks/corpus/pdfs"))
    )
    if not pdfs:
        raise SystemExit(f"no PDFs found in {args.pdf_dir}")

    names = render(pdfs, args.out_dir, args.max_pages, args.dpi)

    # Same selection rule as harness.pick_eval_set: sort, then shuffle with the fixed seed.
    shuffled = sorted(names)
    random.Random(EVAL_SEED).shuffle(shuffled)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "pages": len(names),
                "dpi": args.dpi,
                "sources": [p.name for p in pdfs],
                "eval_seed": EVAL_SEED,
                "eval_set": shuffled[: args.eval_pages],
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"corpus ready: {len(names)} pages in {args.out_dir}; manifest at {MANIFEST}")


if __name__ == "__main__":
    main()
