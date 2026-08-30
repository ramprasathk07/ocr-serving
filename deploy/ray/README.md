# Ray Serve + Ray Data

```bash
uv sync --all-extras                        # brings ray[serve,data]
make ray-serve                              # OpenAI-compatible app on :8000
open http://localhost:8265                  # dashboard — screenshot the replica scale-up
make bench LABEL=ray_serve BASE_URL=http://localhost:8000/v1
make corpus && make ray-batch               # 1k-page Ray Data job
```

`serve_app.py` holds two apps deliberately:

* **`app`** — the production path: `ray.serve.llm` builds an OpenAI-compatible deployment around
  vLLM, with autoscaling (`min/max_replicas`, `target_ongoing_requests`) and **fractional GPU**
  (`num_gpus: 0.5`, `gpu_memory_utilization: 0.42`) so two replicas of the 0.9B model share one
  3060. That scale-up under `--concurrency 16` is the headline demo.
* **`app_manual`** — the same idea without the sugar: a plain deployment with `@serve.batch`
  dynamic microbatching, kept because it is the fallback if `ray.serve.llm` APIs drift, and
  because it shows what the sugar is doing.

Pin the Ray version the day the app first works (PLAN.md §7): `ray.serve.llm` moved across 2.4x.

Start the cluster with `--metrics-export-port=8090` so the Prometheus config in `deploy/monitoring/`
picks up Serve and vLLM metrics.

Capture for the writeup: the autoscale event (dashboard), fractional-GPU VRAM split
(`nvidia-smi`), `results/ray_serve.json`, `results/ray_batch.json` (pages/min vs the online path),
and cold start from `serve run` to first token.
