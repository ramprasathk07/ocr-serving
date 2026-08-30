# Blog 6 — "Same OCR model, four serving stacks: what 12GB of VRAM taught me"

1. **Hook** — everyone benchmarks engines; almost nobody benchmarks *orchestrators*.
   Held the engine constant (vLLM) and swapped the layer above: standalone, Triton,
   Ray Serve, KServe. One RTX 3060.
2. **Setup** — PaddleOCR-VL 0.9B, fixed 20-page eval set, one harness, fairness protocol
   (link PLAN.md §2/§5). Architecture diagram.
2b. **The half nobody benchmarks** — the pipeline around the engine: native-text pages and
   duplicates that never reach the GPU, a resumable Redis-Stream token feed instead of pub/sub,
   a consumer group that survives a worker dying mid-document. Numbers move more from these
   than from the choice of orchestrator — worth saying out loud before the table.
3. **vLLM standalone** — the baseline; what you get free (OpenAI API, metrics, batching)
   and what you don't (multi-replica, rollout story).
4. **Triton** — model repository model, decoupled streaming, when the extra ceremony pays.
5. **Ray Serve** — autoscaling 1→2 replicas *on one GPU* (fractional GPU), dashboard
   screenshot, @serve.batch; Ray Data 1k-page batch job (pages/min vs online path).
6. **KServe** — what "just deploy an InferenceService" actually costs on bare k3s:
   RawDeployment vs Serverless, HPA, the canary caveat, cold-start reality on a 12GB card.
7. **The table** — latency / throughput / cold start / ops ledger (from docs/comparison.md).
8. **Decision guide** — one paragraph each: when I'd pick which. Honest about what a
   single-GPU homelab can and can't prove (no multi-node claims).
9. **What broke** — WSL2+k3s GPU pain, model/vLLM version pinning, VRAM contention.
   (These sections get read the most.)
