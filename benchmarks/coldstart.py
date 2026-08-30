"""Cold-start measurement (protocol in PLAN.md §5).

Two numbers, reported separately because they answer different questions:

* ``ready_s`` — from the moment this script starts polling until ``/v1/models``
  answers 200. That covers container pull (if any), engine init, weight load and
  CUDA graph capture.
* ``first_token_after_ready_s`` — TTFT of the very first request, which is
  slower than steady-state TTFT because nothing is warm yet.

Usage — start the timer first, then launch the stack in another terminal:

    python benchmarks/coldstart.py --label vllm --base-url http://localhost:8001/v1

For Kubernetes, delete the pod first and pass ``--mode k8s``:

    kubectl delete pod -l serving.kserve.io/inferenceservice=ocr-vlm
    python benchmarks/coldstart.py --label kserve --mode k8s --base-url http://.../openai/v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from ocr_serving.common.config import get_settings
from ocr_serving.common.engine import OCRClient


async def wait_ready(base_url: str, timeout_s: float) -> float | None:
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=5) as probe:
        while time.perf_counter() - start < timeout_s:
            try:
                if (await probe.get(f"{base_url.rstrip('/')}/models")).status_code == 200:
                    return time.perf_counter() - start
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    return None


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--mode", choices=["process", "k8s"], default="process")
    ap.add_argument("--model", default=s.model_id)
    ap.add_argument("--image", default="benchmarks/corpus/pages")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    print(f"polling {args.base_url}/models — start the stack now")
    ready_s = await wait_ready(args.base_url, args.timeout)
    if ready_s is None:
        raise SystemExit(f"never became ready within {args.timeout}s")
    print(f"ready after {ready_s:.1f}s; sending the first request")

    image_path = Path(args.image)
    png = (
        sorted(image_path.glob("*.png"))[0].read_bytes()
        if image_path.is_dir()
        else image_path.read_bytes()
    )
    client = OCRClient(args.base_url, args.model, s.engine_api_key, max_retries=0)
    stats = await client.ocr_page_stream(png)
    await client.aclose()

    result = {
        "label": args.label,
        "mode": args.mode,
        "ready_s": round(ready_s, 1),
        "first_token_after_ready_s": round(stats.ttft_s, 2),
        "total_cold_to_first_token_s": round(ready_s + stats.ttft_s, 1),
        "first_request_e2e_s": round(stats.e2e_s, 2),
        "error": stats.error,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(f"benchmarks/results/coldstart_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
