"""End-to-end smoke test / example client.

Submits a document to a running gateway, consumes the SSE stream, prints live
timings, then fetches the result and every artifact. Exits non-zero if anything
fails — safe to run in CI against a compose stack, or by hand after a deploy.

    python scripts/smoke.py --file benchmarks/corpus/pdfs/1706.03762.pdf
    python scripts/smoke.py --gateway http://localhost:8080 --api-key dev-key

With no ``--file`` it generates a small synthetic two-page PDF, so the smoke test
has no data dependency.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx


def synthetic_pdf(path: Path) -> Path:
    """A page with a text layer plus a rendered page — exercises both fast paths."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 700),
        "XFinite-OCR smoke test. " * 40,
        fontsize=11,
        fontname="helv",
    )
    page = doc.new_page()
    for y in range(80, 700, 34):
        page.draw_rect(pymupdf.Rect(60, y, 60 + (y % 300) + 200, y + 14), fill=(0.1, 0.1, 0.1))
    doc.save(str(path))
    doc.close()
    return path


async def stream_job(client: httpx.AsyncClient, job_id: str, headers: dict) -> dict:
    """Consume the SSE stream, reporting TTFT and per-page completion."""
    counts: dict[str, int] = {}
    start = time.perf_counter()
    first_token: float | None = None
    chars = 0

    async with client.stream(
        "GET", f"/v1/ocr/{job_id}/stream", headers=headers, timeout=600
    ) as response:
        response.raise_for_status()
        event = ""
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:") and event:
                counts[event] = counts.get(event, 0) + 1
                payload = json.loads(line[5:].strip() or "{}")
                if event == "token":
                    chars += len(payload.get("data", ""))
                    if first_token is None:
                        first_token = time.perf_counter() - start
                        print(f"  first token after {first_token:.2f}s")
                elif event == "page_complete":
                    print(f"  page {payload.get('page')} complete "
                          f"({len(payload.get('data', ''))} chars)")
                elif event == "progress":
                    print(f"  progress {payload.get('data')}")
                elif event in {"job_complete", "job_failed", "job_cancelled"}:
                    print(f"  {event}")
                event = ""
    return {"events": counts, "ttft_s": first_token, "stream_chars": chars,
            "wall_s": time.perf_counter() - start}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--api-key", default="dev-key")
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the generated PDF")
    args = ap.parse_args()

    headers = {"X-API-Key": args.api_key}
    source = args.file or synthetic_pdf(Path("smoke-sample.pdf"))
    print(f"gateway={args.gateway} file={source}")

    async with httpx.AsyncClient(base_url=args.gateway, timeout=60) as client:
        health = await client.get("/readyz")
        print(f"readyz: {health.status_code} {health.text.strip()}")

        files = {"file": (source.name, source.read_bytes(), "application/pdf")}
        submit = await client.post("/v1/ocr", files=files, headers=headers)
        if submit.status_code != 202:
            print(f"FAIL: submit returned {submit.status_code}: {submit.text}", file=sys.stderr)
            return 1
        job = submit.json()
        print(f"job {job['job_id']} accepted")

        stats = await stream_job(client, job["job_id"], headers)

        result = (await client.get(f"/v1/ocr/{job['job_id']}", headers=headers)).json()
        if result.get("status") != "completed":
            print(f"FAIL: job ended as {result.get('status')}: {result.get('error')}",
                  file=sys.stderr)
            return 1

        artifacts = {}
        for name, path in (("text", "text"), ("markdown", "markdown"), ("pdf", "pdf")):
            response = await client.get(f"/v1/ocr/{job['job_id']}/{path}", headers=headers)
            artifacts[name] = response.status_code
            if response.status_code != 200:
                print(f"FAIL: artifact {name} returned {response.status_code}", file=sys.stderr)
                return 1

    pages = result["pages"]
    print("\n--- summary ---")
    print(f"pages       : {result['page_count']} "
          f"({', '.join(sorted({p['source'] for p in pages}))})")
    print(f"characters  : {len(result['full_text'])}")
    print(f"ttft        : {stats['ttft_s']:.2f}s" if stats["ttft_s"] else "ttft        : n/a")
    print(f"process     : {result['timings']['process_s']}s "
          f"(queue wait {result['timings']['queue_wait_s']}s)")
    print(f"events      : {stats['events']}")
    print(f"artifacts   : {artifacts}")

    if not args.keep and args.file is None:
        source.unlink(missing_ok=True)
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
