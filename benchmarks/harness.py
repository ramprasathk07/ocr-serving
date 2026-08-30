"""Load harness — one harness for all serving stacks (fairness protocol, PLAN.md §5).

    python benchmarks/harness.py --label vllm --base-url http://localhost:8001/v1 \
        --concurrency 1 4 8 16 --requests 30

Same eval set (seed 42), same prompt, ``temperature=0``, same ``max_tokens`` for
every stack — so the numbers measure orchestration overhead, not a different
workload. Writes ``benchmarks/results/{label}.json``; ``report.py`` is the only
thing allowed to turn those files into tables.

What is measured per concurrency level:

* TTFT p50/p95/p99 — first *content* delta, not the first HTTP byte;
* end-to-end p50/p95/p99 per request;
* per-request output tokens/s (mean) and aggregate tokens/s over the window;
* pages/min and error rate, with errors classified rather than swallowed.

Warmup requests are excluded from every statistic.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import statistics as st
import subprocess
import time
from collections import Counter
from pathlib import Path

from ocr_serving.common.config import get_settings
from ocr_serving.common.engine import OCRClient, StreamStats

RESULTS_DIR = Path("benchmarks/results")
EVAL_SEED = 42
EVAL_PAGES = 20


def pick_eval_set(image_dir: Path, count: int = EVAL_PAGES, seed: int = EVAL_SEED) -> list[bytes]:
    """The fixed subset every stack is measured on. Sorted first, so the shuffle is stable."""
    pages = sorted(image_dir.glob("*.png"))
    if not pages:
        raise SystemExit(f"no pages in {image_dir} — run `make corpus` first")
    random.Random(seed).shuffle(pages)
    return [p.read_bytes() for p in pages[:count]]


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    ordered = sorted(values)

    def pct(q: float) -> float:
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return round(ordered[idx], 3)

    return {
        "p50": round(st.median(ordered), 3),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "mean": round(st.fmean(ordered), 3),
    }


def classify(error: str) -> str:
    lowered = error.lower()
    for needle, label in (
        ("timeout", "timeout"), ("connect", "connection"), ("429", "rate_limited"),
        ("503", "unavailable"), ("500", "server_error"), ("400", "bad_request"),
    ):
        if needle in lowered:
            return label
    return "other"


async def run_level(
    client: OCRClient, images: list[bytes], concurrency: int, total: int, warmup: int
) -> dict:
    """Fire ``total`` requests keeping ``concurrency`` of them in flight."""
    counter = {"next": 0}
    lock = asyncio.Lock()

    async def worker(collect: list[StreamStats], budget: int) -> None:
        while True:
            async with lock:
                index = counter["next"]
                if index >= budget:
                    return
                counter["next"] = index + 1
            collect.append(await client.ocr_page_stream(images[index % len(images)]))

    # Warmup — same code path, results discarded.
    counter["next"] = 0
    await asyncio.gather(*(worker([], warmup) for _ in range(min(concurrency, warmup or 1))))

    counter["next"] = 0
    results: list[StreamStats] = []
    start = time.perf_counter()
    await asyncio.gather(*(worker(results, total) for _ in range(concurrency)))
    wall = time.perf_counter() - start

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(failed),
        "error_kinds": dict(Counter(classify(r.error or "") for r in failed)),
        "ttft_s": percentiles([r.ttft_s for r in ok]),
        "e2e_s": percentiles([r.e2e_s for r in ok]),
        "req_tokens_per_s_mean": round(st.fmean([r.tokens_per_s for r in ok]), 1) if ok else None,
        "agg_tokens_per_s": round(sum(r.completion_tokens for r in ok) / wall, 1) if wall else 0,
        "output_tokens_mean": round(st.fmean([r.completion_tokens for r in ok]), 1) if ok else None,
        "pages_per_min": round(len(ok) / wall * 60, 1) if wall else 0,
        "wall_s": round(wall, 1),
    }


def environment() -> dict:
    """Recorded with every run so a number can be traced back to the box it came from."""
    info = {"host": platform.node(), "python": platform.python_version(), "os": platform.platform()}
    for key, cmd in (
        ("gpu", ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        ("git", ["git", "rev-parse", "--short", "HEAD"]),
    ):
        try:
            info[key] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip().splitlines()[0]
        except Exception:
            info[key] = "unknown"
    return info


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--label", required=True, help="vllm | triton | ray_serve | kserve | bentoml | mock"
    )
    ap.add_argument("--base-url", default=s.engine_base_url)
    ap.add_argument("--model", default=s.model_id)
    ap.add_argument("--image-dir", default="benchmarks/corpus/pages", type=Path)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--eval-pages", type=int, default=EVAL_PAGES)
    ap.add_argument("--max-tokens", type=int, default=s.engine_max_tokens)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    images = pick_eval_set(args.image_dir, args.eval_pages)
    print(f"[{args.label}] {len(images)} eval pages, model={args.model}, url={args.base_url}")

    levels = []
    for concurrency in args.concurrency:
        client = OCRClient(
            args.base_url, args.model, s.engine_api_key,
            max_tokens=args.max_tokens, max_retries=0, max_concurrency=concurrency,
        )
        if not await client.health():
            print(f"WARNING: {args.base_url}/models did not answer 200 — is the stack up?")
        print(f"[{args.label}] concurrency={concurrency} ...", flush=True)
        level = await run_level(client, images, concurrency, args.requests, args.warmup)
        await client.aclose()
        print(
            f"  ttft p50/p95={level['ttft_s']['p50']}/{level['ttft_s']['p95']}s "
            f"e2e p50={level['e2e_s']['p50']}s agg={level['agg_tokens_per_s']} tok/s "
            f"pages/min={level['pages_per_min']} errors={level['errors']}"
        )
        levels.append(level)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.label}.json"
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "base_url": args.base_url,
                "model": args.model,
                "eval_pages": len(images),
                "seed": EVAL_SEED,
                "max_tokens": args.max_tokens,
                "requests_per_level": args.requests,
                "warmup": args.warmup,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "environment": environment(),
                "levels": levels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
