# vLLM — the baseline stack

```bash
make vllm                                   # deploy/vllm/serve.sh, :8001
curl localhost:8001/v1/models               # ready check
OCR_ENGINE_BASE_URL=http://localhost:8001/v1 make worker
make bench LABEL=vllm BASE_URL=http://localhost:8001/v1
make coldstart LABEL=vllm BASE_URL=http://localhost:8001/v1
```

Smoke-test the model here **first** (PLAN.md Week 1 Day 1). If the model and the pinned vLLM
version fight, switch to `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` and pin both — then never change the
model again, or the four-stack comparison is meaningless.

What you get for free: OpenAI-compatible API, continuous batching, `/metrics` (already a
Prometheus scrape target). What you do not get: multiple replicas, autoscaling, a rollout story.
That absence is exactly what the other three stacks are being measured against.

Knobs (env, see `serve.sh`): `VLLM_GPU_FRACTION` (0.85 default; ~0.42 per replica when sharing
the card), `VLLM_MAX_MODEL_LEN`, `VLLM_PORT`.
