# Operations runbook

Everything an on-call reader needs: how to start it, what breaks, and what to do about it.

## Start / stop

```bash
docker compose up -d                    # redis, postgres, prometheus, grafana
docker compose --profile app up -d      # + containerised gateway and worker
docker compose --profile mock up -d     # + CPU mock engine (no GPU)
docker compose --profile gpu up -d      # + dcgm-exporter
make down                               # stop everything compose owns
make down-gpu                           # release the GPU before switching stacks
```

Local processes instead of containers: `make gateway`, `make worker`, `make mock`.

**One serving stack at a time.** The 3060 has 12 GB; two engines will OOM each other. `make
down-gpu` kills vLLM, Ray Serve, a Triton container and a KServe `isvc`, then prints
`nvidia-smi` memory so you can confirm the card is free.

## Health

| Check | Meaning |
|---|---|
| `GET /healthz` | Process is alive. Dependency-free — use it as the container liveness probe. |
| `GET /readyz` | Redis (fatal), engine and PostgreSQL (reported). Use as the readiness probe. |
| `GET /metrics` (gateway :8080, worker :9101) | Prometheus scrape. |
| `python scripts/smoke.py` | Full round trip: submit → stream → result → artifacts. Exit code is the verdict. |

## Symptoms → cause → action

**Jobs stay `queued`, queue depth climbs.**
No worker is consuming, or every worker is busy. Check `ocr_jobs_inflight` and the worker's
`/metrics`. Scale workers (`docker compose --profile app up -d --scale worker=3`) or raise
`OCR_WORKER_CONCURRENCY`. If depth climbs with idle workers, the consumer group is missing —
restart a worker, which recreates it.

**Jobs fail immediately with `unreadable or encrypted document`.**
Working as intended: corrupt or password-protected input is a permanent failure and is not
retried. The upload is deleted and no queue entry is made.

**A job is retried forever / lands in the dead-letter stream.**
```bash
redis-cli XRANGE ocr:jobs:dead - +          # job_id + reason
redis-cli HGETALL job:<id>                  # attempts, error, path
```
`ocr_jobs_dead_lettered_total` fires an alert. After fixing the cause, re-submit the file; there
is no automatic replay from the dead-letter stream (deliberate — replaying a poison job loops).

**Engine error rate high / TTFT regression.**
`ocr_engine_requests_total{outcome="error"}` and the TTFT p95 panel. Usual causes, in order:
another stack still holding VRAM (`nvidia-smi`), `max_model_len` too large for the KV cache,
concurrency above what the card sustains (`OCR_ENGINE_MAX_CONCURRENCY`), or the model fighting
the pinned vLLM version — fall back to `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` per PLAN.md §7.

**Streaming clients see nothing.**
Check the job actually started (`GET /v1/ocr/{id}`), then that events exist:
`redis-cli XLEN job:<id>:events`. A proxy that buffers `text/event-stream` will also hide
tokens — the gateway sets `X-Accel-Buffering: no`, but check any layer in front of it.

**`503 too many open streams`.**
More than `MAX_STREAM_CLIENTS` (64) concurrent SSE/WebSocket connections. Each blocking read
holds a Redis connection; raise both that constant and the Redis pool if you genuinely need more.

**429s.**
`ocr_rate_limited_total{reason}` separates `rate_limit` (per-key token bucket), `quota` (tenant
pages/day), `too_large` (upload cap) and `auth`. Tune `OCR_RATE_LIMIT_RPS`,
`OCR_QUOTA_PAGES_PER_DAY`, `OCR_MAX_UPLOAD_MB`.

**Redis restarted / flushed.**
In-flight jobs are lost: state, queue entries and events all live there. Uploads and results on
disk survive, and completed jobs remain readable from PostgreSQL through `GET /v1/ocr/{id}`.

**Disk filling.**
`storage/uploads` and `storage/artifacts` grow without bound. `LocalBlobStore.purge_older_than`
implements the sweep; schedule it against `OCR_RESULT_TTL_DAYS`.

## Useful Redis commands

```bash
redis-cli XINFO GROUPS ocr:jobs                   # depth, pending, consumers
redis-cli XPENDING ocr:jobs workers - + 10        # what is stuck, and with whom
redis-cli HGETALL job:<id>                        # one job's state
redis-cli XRANGE job:<id>:events - + COUNT 20     # its event stream
redis-cli SET job:<id>:cancel 1 EX 3600           # cancel out of band (same as DELETE /v1/ocr/{id})
```

## Deploying a change

1. `make lint && make test` — no GPU required.
2. `docker compose --profile app up -d --build` (or your registry push).
3. Workers drain on SIGTERM: in-flight jobs finish, nothing is re-delivered mid-flight. Give the
   container a `stop_grace_period` longer than your slowest document.
4. `python scripts/smoke.py` against the new deployment.

## Benchmark hygiene

* One stack on the GPU, nothing else running (close the browser — the desktop compositor uses VRAM).
* Same corpus and eval subset (`benchmarks/corpus/manifest.json` pins it).
* `make report` writes the tables; never hand-edit numbers into docs.
* Record the ops ledger while the install is fresh — see `benchmarks/ops_ledger.example.json`.
