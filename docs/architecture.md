# Architecture

**Diagrams**

| File | What it is |
|---|---|
| [`diagrams/architecture.drawio`](diagrams/architecture.drawio) | Three pages: **1 · System architecture** (components, numbered request flow, deployment boundary), **2 · Worker page pipeline** (per-page flow with the three GPU-bypass branches), **3 · Lifecycle & failure paths** (job state machine + failure taxonomy). Open at [diagrams.net](https://app.diagrams.net) or with the VS Code Draw.io extension. |
| [`diagrams/architecture-overview.mmd`](diagrams/architecture-overview.mmd) | Compact overview; the same source is embedded in the README so GitHub renders it inline. |
| [`diagrams/`](diagrams/) | The original design sketches (`other.svg`, `planflow-1..3.svg`) this repo implements a slice of. |

Distilled from `docs/diagrams/*.svg`, with the honest mapping from each diagram box to the code that
implements it — and to the boxes deliberately left as notes.

## Request lifecycle (`diagrams/other.svg`)

1. **`POST /v1/ocr`** — the gateway authenticates the key, resolves its tenant, applies the
   token-bucket rate limit, streams the upload to the blob store in 1 MiB chunks (hashing as it
   goes), counts the document's pages without rendering them, checks the tenant's daily page
   quota, writes `job:{id}` state, and `XADD`s the id onto `ocr:jobs`. Responds **202 + job_id**.
2. **Client subscribes** to `GET /v1/ocr/{id}/stream` (SSE) or the WebSocket. The gateway tails
   `job:{id}:events`, a capped Redis Stream, from `Last-Event-ID` or the beginning.
3. **A worker reserves the job** via `XREADGROUP` on the consumer group. The entry stays in the
   pending list until the result is durable, so a crashed worker loses nothing.
4. **Per page, streamed** — classify (native text layer / blank / duplicate) → render at an
   adaptive DPI → deskew → CLAHE → layout detection → tile oversized regions → assign reading
   order → stream each region through the active serving stack → publish token batches.
5. **Assembly** — stitch regions (dropping tile overlap), normalise text, truncate degenerate
   generations, merge pages (repairing hyphenation across the break).
6. **Persist** — JSON + Markdown + searchable PDF to the blob store, the record to PostgreSQL,
   final state to Redis, `job_complete` to the event stream, then `XACK`.
7. **`GET /v1/ocr/{id}`** returns the result; `/text`, `/markdown`, `/pdf` return artifacts.

## Component map (`diagrams/planflow-1.svg`, `diagrams/planflow-2.svg`) → this repo

| Diagram box | Implementation | Notes |
|---|---|---|
| FastAPI Gateway, upload handler, job id | `src/ocr_serving/gateway/main.py` | streaming upload, idempotency keys |
| Auth / rate limit / tenant quotas | `src/ocr_serving/gateway/auth.py`, `src/ocr_serving/common/ratelimit.py` | Lua token bucket + rolling daily page quota, both fail open |
| Object storage / local blob store | `src/ocr_serving/common/storage.py` | S3-shaped API over the filesystem |
| Redis job state | `src/ocr_serving/common/events.py` | hash with TTL, progress counters |
| Redis PubSub / Streams, WS/SSE gateway | `src/ocr_serving/common/events.py`, `src/ocr_serving/gateway/main.py` | Streams, not pub/sub — resumable and replayable |
| Ray internal queue | `src/ocr_serving/common/queue.py` | Redis Stream consumer group: reclaim, redelivery counting, dead-letter |
| CPU worker orchestrator | `src/ocr_serving/workers/cpu_worker.py` | bounded concurrency at job / page / region level |
| PDF intake, metadata, page classifier, native text extractor, incremental renderer | `src/ocr_serving/workers/documents.py` | native-text pages skip the GPU entirely |
| Deskew, denoise, CLAHE, adaptive DPI, blank detector, duplicate hash | `src/ocr_serving/workers/preprocess.py` | duplicates copy the original's text rather than losing it |
| DocYOLO layout, region segmentation, table/text split, tiling, reading order | `src/ocr_serving/workers/layout.py` | ONNX when `models/doclayout_yolo.onnx` exists, whole-page regions otherwise |
| GPU admission control (leaky bucket, priority, model router) | `src/ocr_serving/common/engine.py` | client-side semaphore + bounded retries; priority queue not built |
| GPU runtime: encoder + vLLM decoder, microbatcher, token streaming | `deploy/` + `src/ocr_serving/serving/` | the four stacks — the point of the repo |
| Text normalizer, confidence filter, patch/page stitcher, document merger | `src/ocr_serving/workers/postprocess.py` | repetition filter for looping generations |
| PostgreSQL results | `src/ocr_serving/common/db.py` | optional; degrades to filesystem-only |
| Searchable PDF generator | `src/ocr_serving/workers/pdf_writer.py` | invisible text layer (see caveat below) |
| Observability | `src/ocr_serving/common/metrics.py`, `deploy/monitoring/` | Prometheus + provisioned Grafana + alert rules |
| Admin dashboard | Grafana + `GET /v1/jobs` | no bespoke admin UI |

## Deliberate gaps

* **Word-level PDF positioning.** A page-level OCR VLM returns text, not per-word boxes. The
  searchable PDF therefore carries an invisible page-level text layer: search, copy and indexing
  work; selection does not highlight the exact word. Word-accurate placement needs a detector
  that emits boxes — a different model than the one being benchmarked.
* **CER/WER evaluation.** The corpus has no ground truth yet, so accuracy is not reported. The
  results schema and PostgreSQL store keep the per-page text needed to add it.
* **Priority scheduling / model router.** One model, one queue today. The admission point exists
  (`OCRClient`'s semaphore); priority classes would slot in at `src/ocr_serving/common/queue.py`.
* **TensorRT vision encoder.** The stacks use vLLM end to end so the comparison isolates
  orchestration. A TRT encoder would change the engine, not the orchestrator.

## Target state (`diagrams/planflow-3.svg`)

The GPU runtime replaced by a **Ray Serve OCR endpoint** with dynamic batching inside the
deployment, autoscaling replicas and fractional GPU (two replicas of a 0.9B model on one 12 GB
card). `src/ocr_serving/serving/ray/serve_app.py` is that box; nothing upstream of `OCR_ENGINE_BASE_URL` changes.

## Scaling model

* **Gateway** — stateless; scale with replicas behind any load balancer. All state is in Redis.
* **Worker** — scale with replicas or `docker compose --scale worker=N`; the consumer group hands
  each job to exactly one worker, and `OCR_WORKER_CONCURRENCY` sets per-process parallelism.
* **Engine** — the single 3060 is the hard constraint: one serving stack at a time
  (`make down-gpu` before switching). Ray Serve's fractional GPU is how two replicas coexist.
* **Redis** — job state and events are capped and TTL'd; the queue is the only unbounded
  structure, and its depth is alerted on.
