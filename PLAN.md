# XFinite-OCR — Serving Breadth: 4-Week Execution Plan

**Repo purpose.** One OCR VLM, served four ways — **vLLM, Triton, Ray Serve, KServe** — behind the
same gateway/worker pipeline, measured by one benchmark harness. Output: honest numbers in the
README plus Blog 6 (*"Same model, four serving stacks: latency, throughput, cold start, ops overhead"*).

**Hardware.** RTX 3060 12 GB on Windows 10. Everything GPU-touching runs inside **WSL2 Ubuntu 24.04**
(vLLM, Ray, k3s). Windows-native vLLM/k3s is a dead end — do not fight it.

**Timeline.** 4 weeks. Week 1 is Phase-1 polish; if the main XFinite-OCR repo already has
queue + monitoring + streaming working, compress Week 1 to 2 days and this becomes a 3-week plan
(matches the v6 window M2 W4 → M3 W2).

---

## 1. The architecture (distilled from `docs/diagrams/*.svg`)

- **`other.svg`** — sequence diagram, the request lifecycle:
  `POST /v1/ocr` → auth/quota → save file to object storage → `SET job:{id}=queued` in Redis →
  enqueue → `202 + job_id` → client opens WS/SSE → CPU worker pulls job → classify pages →
  render incrementally → preprocess → DocYOLO layout → tile regions → GPU batch scheduler →
  encoder (TRT/PyTorch) + decoder (vLLM) → publish partial text per page → stitch patches →
  merge pages → persist → `job_complete` → `GET /v1/ocr/{job_id}` → JSON / searchable PDF.
- **`planflow-1.svg` / `planflow-2.svg`** — component graph: FastAPI gateway (auth, rate limit,
  tenant quotas), Redis (job state + pub/sub streams), Ray internal queue, CPU worker pool
  (PDF intake, metadata, page classifier, native-text extractor, incremental renderer,
  deskew/denoise/CLAHE, blank-page + duplicate-hash cache check), DocYOLO ONNX layout
  (region segmentation, table/paragraph split, tiling, reading order), GPU admission controller
  (leaky bucket, priority queue, model router), GPU runtime (TensorRT vision encoder,
  **PaddleOCR-VL-1B / Tencent HunyuanOCR-1B**, vLLM decoder, dynamic microbatcher, token streaming),
  post (normalize, confidence filter, patch/page stitch), PostgreSQL, searchable-PDF generator,
  observability (latency, queue depth, GPU util, model load time, CER/WER).
- **`planflow-3.svg`** — the target state for this phase: same pipeline with the GPU runtime
  replaced by a **Ray Serve OCR endpoint** (dynamic batching inside the deployment).

This repo implements a *minimal but real* slice of that pipeline (gateway → Redis → CPU worker →
serving endpoint → results) so every serving stack sees realistic traffic, not toy `curl` loops.

## 2. Ground rules (fairness protocol)

1. **Same model everywhere.** Default `PaddlePaddle/PaddleOCR-VL` (0.9B, vLLM-supported).
   Alt: `tencent/HunyuanOCR` (1B). Safety fallback if either fights vLLM on your pinned version:
   `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` (guaranteed vLLM multimodal path). Pick once in Week 1 Day 1,
   never change mid-benchmarks.
2. **Same engine where possible.** vLLM *is* the engine in all four stacks (standalone, Triton
   vLLM backend, `ray.serve.llm`, KServe huggingface runtime `--backend=vllm`). The comparison then
   honestly measures **orchestration overhead + ops story**, not engine differences.
3. **One harness** (`benchmarks/harness.py`), one eval set (fixed 20-page subset of the corpus),
   same prompt, `temperature=0`, same `max_model_len`, same concurrency sweep.
4. **One stack on the GPU at a time.** 12 GB. `make down-gpu` before switching.
5. Numbers only enter README/blog via `benchmarks/report.py` from `benchmarks/results/*.json` —
   no hand-typed numbers.

## 3. Substrate & tools to install

| Tool | Where | Why | Install |
|---|---|---|---|
| WSL2 Ubuntu 24.04 + systemd | Windows | run everything GPU | `wsl --install -d Ubuntu-24.04`; enable systemd in `/etc/wsl.conf` |
| NVIDIA driver (Windows side only) | Windows | CUDA in WSL needs *no* Linux driver | GeForce driver ≥ latest |
| `uv` | WSL | Python env/deps | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Engine (inside WSL, not Docker Desktop) | WSL | compose infra, Triton, dcgm-exporter | `apt` per docs; add nvidia-container-toolkit |
| vLLM | WSL venv | engine | `uv pip install vllm` (pin what you smoke-test) |
| Ray | WSL venv | Serve + Data | `uv pip install "ray[serve,data]"` (+ `ray[llm]` if using `ray.serve.llm`) |
| k3s + kubectl + helm | WSL | KServe substrate | `curl -sfL https://get.k3s.io \| sh -` |
| KServe (RawDeployment mode) | k3s | InferenceService | quick-install script / helm |
| cloudflared | WSL | public demo tunnel | cloudflare pkg repo |
| Prometheus + Grafana + dcgm-exporter | compose | dashboards | `docker compose up` |
| vhs or asciinema+agg | WSL | README GIFs | go install / cargo |
| llm-compressor | WSL venv | AWQ-int4 quant for KServe week | `uv pip install llmcompressor` |

## 4. Week-by-week

### Week 1 — Phase-1 polish: "deployed, not just trained"

| Day | Work | Exit criteria |
|---|---|---|
| 1 | WSL2 + CUDA smoke (`nvidia-smi` in WSL); `uv sync` this repo; **pick the model**; `deploy/vllm/serve.sh` up; `benchmarks/harness.py` v0 runs against it | TTFT/tokens-per-sec printed for 5 requests |
| 2 | Wire gateway + Redis queue + CPU worker (`docker compose up` infra; `make gateway`, `make worker`); single-image job end-to-end | `POST /v1/ocr` → `202` → SSE tokens → `GET` result JSON |
| 3 | Streaming **PDF** OCR: pymupdf incremental render → per-page events → stitched result; blank/dup-page skip | 10-page PDF streams page-by-page over SSE |
| 4 | Observability: Prometheus + Grafana + dcgm-exporter; panels: TTFT, tokens/s, queue depth, GPU util, job rate | Grafana screenshot with live inference traffic |
| 5 | Cloudflare Tunnel to gateway; record SSE token-streaming GIF (vhs); README real-time section: GIF + TTFT p50/p95 + tokens/s table from harness | README shows live demo + measured table |

### Week 2 — Ray: Serve + Data (~the "1–2 day" v6 item, done properly)

| Day | Work | Exit criteria |
|---|---|---|
| 1 | Ray Serve quickstart; `src/ocr_serving/serving/ray/serve_app.py` via `ray.serve.llm` → OpenAI-compatible app; `serve run` locally; harness passes unchanged | parity bench vs standalone vLLM |
| 2 | Autoscaling: `target_ongoing_requests`, min/max replicas; **fractional GPU** (`num_gpus: 0.5`, `gpu_memory_utilization≈0.42` each) — show 2 replicas of the 0.9B model on one 3060 scale 1→2 under load; Ray dashboard screenshots | scale event captured in dashboard + bench |
| 3 | `benchmarks/build_corpus.py` → 1k-page corpus; `src/ocr_serving/serving/ray/batch_inference.py` Ray Data `map_batches` job over it | pages/min + GPU util recorded, parquet written |
| 4 | Bench matrix (c = 1/4/8/16), cold-start measurement, README "Ray Serve" section + numbers | `results/ray_serve.json`, `results/ray_batch.json` |
| 5 | Buffer / deep-dive: `@serve.batch` manual deployment (educational alt in `serve_app.py`), placement groups reading | — |

### Week 3 — KServe on k3s (~the "2 day" v6 item + k8s substrate reality)

| Day | Work | Exit criteria |
|---|---|---|
| 1 | k3s in WSL2 (`deploy/kserve/setup/install-k3s-gpu.sh`): nvidia-container-toolkit, RuntimeClass `nvidia`, device plugin; CUDA smoke pod | `kubectl` pod runs `nvidia-smi` |
| 2 | KServe **RawDeployment** install (cert-manager + CRDs; skip Knative/Istio); sanity `sklearn-iris` InferenceService | `kubectl get isvc` READY |
| 3 | Quantize model (llm-compressor AWQ-int4) or pick prequantized; deploy `deploy/kserve/inferenceservice.yaml` (huggingface runtime, vLLM backend); hostPath model cache to skip re-downloads | OCR request served through k8s |
| 4 | HPA (min/max replicas, `scaleMetric`), canary manifest + note (traffic-split needs Serverless/Gateway API — document, don't build), cold start: pod delete → ready → first token | `results/kserve.json` + cold-start numbers |
| 5 | README KServe section; ops-overhead ledger: install wall-time, control-plane RAM (`free -g` delta), config LOC, moving parts count | ledger table committed |

### Week 4 — Four-stack comparison + optional + writeup

| Day | Work | Exit criteria |
|---|---|---|
| 1 | Triton parity: vLLM backend (`serving/triton/`), same harness (reuse Month-1 Triton work if it exists) | `results/triton.json` |
| 2 | *(optional)* BentoML wrap (`src/ocr_serving/serving/bentoml/service.py`) + bench; else start writing | `results/bentoml.json` or blog draft |
| 3 | `benchmarks/report.py` → `docs/comparison.md` four-stack table; re-run anything stale on the fixed eval set | comparison table complete |
| 4 | Blog 6 draft from `docs/blog6-outline.md`; README final: stack matrix, architecture diagram, GIFs, numbers | blog draft done |
| 5 | *(optional, ₹1–2k)* SageMaker real-time endpoint weekend (LMI/DJL vLLM container) — only if chasing enterprise JDs. TFX stays skipped | — |

## 5. Benchmark protocol

- **Eval set:** fixed 20 pages (seed 42) from the 1k corpus; same prompt
  (`src/ocr_serving/common/engine.py::OCR_PROMPT`); `temperature=0`, `max_tokens=512`/page.
- **Sweep:** concurrency {1, 4, 8, 16}; 3 warmup + 30 measured requests per point.
- **Serving metrics:** TTFT p50/p95 (first content delta), e2e p50/p95, per-request output
  tokens/s (mean), aggregate tokens/s, pages/min, error rate.
- **Cold start:** (a) process/engine: launch → first `200 /health` → first token;
  (b) k8s: pod deleted → Ready → first token. Report both, labeled.
- **Batch (Ray Data):** pages/min over 1k pages, mean GPU util (dcgm), wall time.
- **Ops overhead (qualitative ledger):** install wall-time, control-plane RAM, lines of config,
  number of moving parts, docs quality note.

## 6. Reading list

*(URLs current as of Jan 2026 — if a path 404s, search the site; concepts stay.)*

**Ray** — getting started <https://docs.ray.io/en/latest/serve/getting_started.html> ·
key concepts <https://docs.ray.io/en/latest/serve/key-concepts.html> ·
autoscaling <https://docs.ray.io/en/latest/serve/autoscaling-guide.html> ·
fractional GPUs / resource allocation <https://docs.ray.io/en/latest/serve/resource-allocation.html> ·
dynamic batching <https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html> ·
serve.llm <https://docs.ray.io/en/latest/serve/llm/serving-llms.html> ·
Ray Data batch inference <https://docs.ray.io/en/latest/data/batch_inference.html> ·
Ray Data + LLMs <https://docs.ray.io/en/latest/data/working-with-llms.html>

**Kubernetes/KServe** — k3s quick start <https://docs.k3s.io/quick-start> ·
k3s NVIDIA runtime <https://docs.k3s.io/advanced#nvidia-container-runtime-support> ·
CUDA on WSL2 <https://docs.nvidia.com/cuda/wsl-user-guide/index.html> ·
nvidia-container-toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html> ·
device plugin (+ time-slicing!) <https://github.com/NVIDIA/k8s-device-plugin> ·
KServe get started <https://kserve.github.io/website/latest/get_started/> ·
RawDeployment mode <https://kserve.github.io/website/latest/admin/kubernetes_deployment/> ·
first InferenceService <https://kserve.github.io/website/latest/get_started/first_isvc/> ·
huggingface runtime (vLLM backend) <https://kserve.github.io/website/latest/modelserving/v1beta1/llm/huggingface/> ·
raw-deployment autoscaling <https://kserve.github.io/website/latest/modelserving/autoscaling/raw_deployment_autoscaling/> ·
canary rollout <https://kserve.github.io/website/latest/modelserving/v1beta1/rollout/canary/>

**vLLM** — OpenAI server <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html> ·
metrics <https://docs.vllm.ai/en/latest/serving/metrics.html> ·
quantization <https://docs.vllm.ai/en/latest/features/quantization/> ·
llm-compressor <https://github.com/vllm-project/llm-compressor> ·
Grafana example <https://github.com/vllm-project/vllm/tree/main/examples/online_serving/prometheus_grafana>

**Triton** — vLLM backend <https://github.com/triton-inference-server/vllm_backend> ·
tutorials <https://github.com/triton-inference-server/tutorials>

**BentoML** — docs <https://docs.bentoml.com/en/latest/> · BentoVLLM <https://github.com/bentoml/BentoVLLM>

**Observability** — dcgm-exporter <https://github.com/NVIDIA/dcgm-exporter> ·
Grafana provisioning <https://grafana.com/docs/grafana/latest/administration/provisioning/>

**Demo plumbing** — Cloudflare Tunnel <https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-remote-tunnel/> ·
vhs <https://github.com/charmbracelet/vhs>

**Models** — PaddleOCR-VL <https://huggingface.co/PaddlePaddle/PaddleOCR-VL> ·
HunyuanOCR <https://huggingface.co/tencent/HunyuanOCR> ·
Qwen2.5-VL-3B AWQ <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ>

**SageMaker (optional)** — real-time endpoints <https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html> ·
LMI containers <https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html>

## 7. Risk register

| Risk | Mitigation |
|---|---|
| k3s GPU inside WSL2 is the flakiest link | Fallbacks in order: k3d with CUDA-enabled k3s image → `minikube --driver=docker --gpus=all`. Budget Day 1 of Week 3 entirely for this. |
| Chosen OCR model unsupported by pinned vLLM | Smoke-test Day 1; fallback `Qwen2.5-VL-3B-Instruct-AWQ`. Pin `vllm` version once it works. |
| 12 GB VRAM contention | One stack at a time; `make down-gpu`; fractional-GPU demo sized for the 0.9B model only. |
| `ray.serve.llm` API drift across 2.4x | Pin `ray` version Day 1 of Week 2; manual `@serve.batch` deployment kept as fallback in `serve_app.py`. |
| KServe huggingfaceserver image is huge | `docker pull` / `ctr images pull` it Day 2 evening, before you need it. |
| WSL2 memory pressure (k3s + engine + compose) | `.wslconfig`: give WSL 24 GB+ if available; run only the active stack. |
| Canary traffic-split needs Knative/Gateway API | Ship the manifest + written note (v6 asks for "canary note", not a mesh). |

## 8. Deliverables checklist (the signals)

- [ ] README real-time section: SSE streaming GIF + TTFT p50/p95 + tokens/s table
- [ ] Grafana dashboard screenshot under live load
- [ ] Public demo via Cloudflare Tunnel (even if temporary)
- [ ] Ray: Serve deployment w/ autoscaling + fractional GPU; Ray Data 1k-page job; numbers
- [ ] KServe: `kubectl get isvc` READY screenshot; HPA config; canary manifest + note; cold-start numbers
- [ ] `docs/comparison.md`: four-stack table (latency / throughput / cold start / ops overhead)
- [ ] Blog 6 draft
- [ ] (opt) BentoML numbers; (opt) SageMaker endpoint writeup
