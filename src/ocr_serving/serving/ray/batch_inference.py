"""Ray Data batch inference over the 1k-page corpus (Week 2 Day 3).

    python serving/ray/batch_inference.py --corpus benchmarks/corpus/pages --batch-size 16

Reads page PNGs, runs the vLLM engine offline (no HTTP), writes parquet, prints pages/min.
Also consider ray.data.llm (build_llm_processor) — see 'Working with LLMs' in Ray docs.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import ray


class OCRPredictor:
    """One actor holds the engine for its lifetime; Ray Data streams batches through it."""

    def __init__(self, model_id: str):
        from PIL import Image  # noqa: F401
        from vllm import LLM, SamplingParams  # imported inside the actor (GPU worker)

        self.llm = LLM(model=model_id, gpu_memory_utilization=0.85,
                       max_model_len=8192, trust_remote_code=True)
        self.params = SamplingParams(temperature=0, max_tokens=512)

    def __call__(self, batch: dict) -> dict:
        from PIL import Image

        from ocr_serving.common.engine import OCR_PROMPT

        # TODO(week2): confirm multimodal offline API for the pinned vLLM
        # (llm.chat with image content parts, or generate(multi_modal_data=...)).
        inputs = [
            {
                "prompt": f"<|user|>\n<image>\n{OCR_PROMPT}<|assistant|>\n",
                "multi_modal_data": {"image": Image.fromarray(arr.astype(np.uint8))},
            }
            for arr in batch["image"]
        ]
        outs = self.llm.generate(inputs, self.params)
        batch["text"] = np.array([o.outputs[0].text for o in outs], dtype=object)
        del batch["image"]
        return batch


class GpuSampler:
    """Poll nvidia-smi in the background so the batch job reports mean GPU utilisation.

    Batch throughput without a utilisation number is not interpretable: 900 pages/min
    at 40% util means the bottleneck is the data path, not the card.
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        while not self._stop.wait(self.interval_s):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                return  # no nvidia-smi (CPU box): stop sampling, report n/a

    def __enter__(self) -> "GpuSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def mean(self) -> float | None:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="benchmarks/corpus/pages")
    ap.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="benchmarks/results/ray_batch_output")
    args = ap.parse_args()

    ray.init()
    ds = ray.data.read_images(args.corpus, include_paths=True)
    n_pages = ds.count()

    t0 = time.perf_counter()
    with GpuSampler() as gpu:
        ds.map_batches(
            OCRPredictor,
            fn_constructor_kwargs={"model_id": args.model},
            batch_size=args.batch_size,
            num_gpus=1,        # whole GPU for batch throughput
            concurrency=1,     # 1 engine actor on the 3060
        ).write_parquet(args.out)
    wall = time.perf_counter() - t0

    stats = {
        "label": "ray_batch",
        "pages": n_pages,
        "wall_s": round(wall, 1),
        "pages_per_min": round(n_pages / wall * 60, 1),
        "gpu_util_mean": gpu.mean,
        "batch_size": args.batch_size,
        "model": args.model,
    }
    out = Path("benchmarks/results/ray_batch.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(stats)


if __name__ == "__main__":
    main()
