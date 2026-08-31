# Testing walkthrough

Three levels, each useful on its own. Level 0 and 1 need **no GPU, no model
download and no Kubernetes**; only level 2 needs the 3060.

| Level | What it proves | Needs |
|---|---|---|
| 0 · test suite | Every stage in isolation + the pipeline end to end | Python only |
| 1 · live stack on CPU | The real service: upload → queue → stream → artifacts | Python (+ Docker for monitoring) |
| 2 · real model on GPU | OCR accuracy and the benchmark numbers | WSL2 + RTX 3060 |

---

## What is present, and what gets fetched

| Thing | State | How it arrives |
|---|---|---|
| Pipeline, gateway, worker, API, artifacts | **in the repo** | — |
| Mock OCR engine (CPU) | **in the repo** | `ocr_serving.serving.mock.server` |
| Grafana dashboard, alert rules, scrape config | **in the repo** | provisioned by compose |
| OCR model weights (PaddleOCR-VL ≈ 2 GB) | **not present** | vLLM downloads from HuggingFace on first run |
| Layout model `models/doclayout_yolo.onnx` | **not present, optional** | one `curl`, see below; without it each page is one region |
| Benchmark corpus | **not present** | `benchmarks/build_corpus.py` builds it |
| PostgreSQL | **optional** | compose; the service degrades to filesystem-only without it |

Nothing is stubbed on the CPU path: with the mock engine every stage runs for
real except the model itself.

### Fetching the layout model (optional, 75 MB)

Without it every page is sent to the engine as a single region, which is a valid
way to run. With it, pages are segmented into titles, columns, tables and
captions, and each region becomes its own streamed request.

```bash
mkdir -p models
curl -L -o models/doclayout_yolo.onnx   https://huggingface.co/wybxc/DocLayout-YOLO-DocStructBench-onnx/resolve/main/doclayout_yolo_docstructbench_imgsz1024.onnx
```

Apache-2.0, an ONNX export of the official DocLayout-YOLO DocStructBench
weights, `imgsz=1024` — which is what `OCR_LAYOUT_MODEL_PATH` and the detector's
default input size expect. The worker logs `layout model loaded` at startup and
`/readyz` is unaffected; without the file it logs `no layout model, using
whole-page regions`.

Verified on a two-column A4 page at 150 dpi: 7 regions (title, 4 text blocks,
table, table caption) in ~2.6 s on CPU, ordered the way a person reads them.

## Environment

`make` is not available on a Windows host, so the raw commands are given first;
inside WSL2 the `make` targets in the [Makefile](../Makefile) do the same thing.

```bash
# Windows host (PowerShell or Git Bash)
py -3 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[worker,bench,dev,db]"

# WSL2 / Linux / macOS
uv sync --frozen --extra worker --extra bench --extra dev --extra db
```

Every command below uses `.venv/Scripts/python.exe` (Windows). Substitute
`uv run python` or an activated venv elsewhere.

---

## Level 0 — the test suite

```bash
.venv/Scripts/python.exe -m pytest -q          # 149 pass, 11 skipped
.venv/Scripts/python.exe -m ruff check .
```

The 11 skips are the PostgreSQL integration tests. To run them too, start a
throwaway server and point the suite at it:

```bash
docker run -d --rm --name ocr-pg -e POSTGRES_USER=ocr -e POSTGRES_PASSWORD=ocr \
  -e POSTGRES_DB=ocr_test -p 5432:5432 postgres:16-alpine

OCR_TEST_POSTGRES_DSN=postgresql://ocr:ocr@localhost:5432/ocr_test \
  .venv/Scripts/python.exe -m pytest -q       # 160 pass
```

CI runs all 160 on every push.

---

## Level 1 — the live stack on CPU

Four processes. Use four terminals, or run the first three in the background.

### 1. Redis

```bash
docker run -d --rm --name ocr-redis -p 6379:6379 redis:7-alpine
```

**Port already in use?** A native PostgreSQL or Redis service on the machine
will already own 5432/6379, and the published container port silently loses —
connections then hit the wrong server with confusing auth errors. The compose
ports are configurable:

```bash
POSTGRES_PORT=5433 REDIS_PORT=6380 docker compose up -d
# and point the services at them:
export OCR_POSTGRES_DSN=postgresql://ocr:ocr@127.0.0.1:5433/ocr
```

Check what holds a port: `Get-NetTCPConnection -LocalPort 5432 -State Listen`
(PowerShell) or `ss -lptn 'sport = :5432'` (Linux).

**No Docker?** `fakeredis` ships a real TCP server and speaks streams and
consumer groups, which is all the pipeline uses:

```bash
.venv/Scripts/python.exe -c "from fakeredis import TcpFakeServer; \
  TcpFakeServer(('127.0.0.1', 6399), server_type='redis').serve_forever()"
```

…then use `OCR_REDIS_URL=redis://127.0.0.1:6399/0` below. State is in memory
only, so it disappears when the process stops.

### 2. Mock engine — :8001

```bash
MOCK_TTFT_MS=250 MOCK_TOKEN_MS=12 MOCK_TOKENS=160 \
  .venv/Scripts/python.exe -m uvicorn ocr_serving.serving.mock.server:app --port 8001
```

Those three variables are the whole cost model: prefill latency, per-token
decode latency, response length. `MOCK_FAIL_RATE=0.2` makes one request in five
return 503, which exercises the retry path.

### 3. Gateway — :8080

```bash
export OCR_REDIS_URL=redis://127.0.0.1:6379/0
export OCR_ENGINE_BASE_URL=http://127.0.0.1:8001/v1
export OCR_MODEL_ID=mock-ocr-vl
export OCR_API_KEYS=dev-key:demo
export OCR_POSTGRES_ENABLED=false
export OCR_LOG_FORMAT=text

.venv/Scripts/python.exe -m uvicorn ocr_serving.gateway.main:app --port 8080
```

### 4. Worker

```bash
# same exports, plus:
export OCR_WORKER_METRICS_PORT=9101
.venv/Scripts/python.exe -m ocr_serving.workers.cpu_worker
```

### Check it is alive

```bash
curl -s localhost:8080/readyz
# {"status":"ok","version":"1.0.0","checks":{"redis":"ok","engine":"ok","postgres":"disabled"}}
```

`redis` must be `ok`; `engine` `down` means step 2 is not running.

### Drive it

**One command, end to end** — submits a generated PDF, consumes the SSE stream,
fetches every artifact, exits non-zero on any failure:

```bash
.venv/Scripts/python.exe scripts/smoke.py --gateway http://localhost:8080 --api-key dev-key
```

**In the browser** — <http://localhost:8080> is a drag-and-drop demo page that
shows tokens arriving live, with TTFT and chars/s. API key `dev-key`.

**By hand:**

```bash
JOB=$(curl -s -X POST localhost:8080/v1/ocr -H 'X-API-Key: dev-key' \
        -F file=@yourfile.pdf | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

curl -N localhost:8080/v1/ocr/$JOB/stream -H 'X-API-Key: dev-key'   # live tokens
curl -s  localhost:8080/v1/ocr/$JOB       -H 'X-API-Key: dev-key'   # result JSON
curl -s  localhost:8080/v1/ocr/$JOB/text  -H 'X-API-Key: dev-key'   # plain text
curl -s  localhost:8080/v1/ocr/$JOB/markdown -H 'X-API-Key: dev-key'
curl -so out.pdf localhost:8080/v1/ocr/$JOB/pdf -H 'X-API-Key: dev-key'   # searchable PDF
curl -X DELETE localhost:8080/v1/ocr/$JOB -H 'X-API-Key: dev-key'         # cancel
```

Interactive API docs: <http://localhost:8080/docs>.

### Worth testing deliberately

| Behaviour | How to trigger | Expected |
|---|---|---|
| Content sniffing | rename a PNG to `.pdf` and submit | `415`, naming the real type |
| Native-text fast path | submit a digital PDF | pages have `"source": "native"`, `tokens: 0` |
| Duplicate detection | a PDF with the same scan twice | second page `"source": "duplicate"`, text copied |
| Blank skip | a PDF with an empty page | `"source": "blank"` |
| Stream resume | kill `curl -N` mid-stream, reconnect with `-H "Last-Event-ID: <id>"` | resumes, no repeats |
| Crash recovery | `kill -9` the worker mid-job, restart it | job reclaimed after `OCR_VISIBILITY_TIMEOUT_S` |
| Rate limit | `OCR_RATE_LIMIT_RPS=0.1`, submit twice | `429` with `Retry-After` |
| Quota | `OCR_QUOTA_PAGES_PER_DAY=1`, submit a 2-page PDF | `429` mentioning the quota |
| Engine failure | restart the mock with `MOCK_FAIL_RATE=1.0` | page carries an error, job still completes |

---

## Output formats

All three artifacts are produced for every completed job.

**`GET /v1/ocr/{id}` — result JSON**

```
job_id  status  filename  tenant  page_count  pages[]  full_text
error   model   engine    attempts  timings{}  artifacts{}

pages[] : index  text  source  regions  chars  duration_ms  ttft_s  tokens
          dpi  width  height  skew_deg  duplicate_of  error
```

`source` is one of `ocr | native | duplicate | blank` — the field that shows how
much work the GPU actually did. A real two-page run:

```
page 0: source=native  chars= 959  tokens=  0  dpi=  0  ttft=None
page 1: source=ocr     chars=2473  tokens=346  dpi=150  ttft=0.229
timings: {queue_wait_s: 0.005, process_s: 1.931, ...}
```

**Markdown** — `# <filename>` then `<!-- page N -->` anchors per page.

**Searchable PDF** — original pages plus an invisible text layer
(`render_mode=3`). Verify:

```bash
python -c "import pymupdf; d=pymupdf.open('out.pdf'); print(d.load_page(0).get_text('text')[:200])"
```

Search, copy and indexing work; selection does not highlight individual words —
a page-level OCR VLM returns text, not per-word boxes. See
[architecture.md](architecture.md#deliberate-gaps).

**SSE frames** — `status`, `job_started`, `page_started`, `token`,
`page_complete`, `progress`, `job_complete` / `job_failed` / `job_cancelled`,
`end`. Every frame carries an `id:` (the Redis stream id) for resuming.

---

## Monitoring

```bash
docker compose up -d                      # redis, postgres, prometheus, grafana
docker compose --profile gpu up -d        # + dcgm-exporter, GPU util and VRAM
```

- Grafana <http://localhost:3000> (`admin` / `admin`) opens on the **OCR
  pipeline** dashboard — provisioned from `deploy/monitoring/`, not clicked
  together by hand.
- Prometheus <http://localhost:9090>; alert rules at
  <http://localhost:9090/alerts>.

Scrape targets are `gateway:8080` / `worker:9101` inside compose and
`host.docker.internal` for processes on the host, so both layouts work. To see
the dashboard move, keep the stack from level 1 running and generate load:

```bash
for i in $(seq 1 20); do
  curl -s -X POST localhost:8080/v1/ocr -H 'X-API-Key: dev-key' -F file=@sample.pdf > /dev/null &
done; wait
```

Panels to watch: **TTFT p50/p95**, **Pages by source** (the native/duplicate/
blank share is the throughput story), **Queue** depth vs pending, **Engine
tokens/s and in-flight**.

Verify the wiring rather than eyeballing the charts:

```bash
# every target that should be up, is
curl -s "localhost:9090/api/v1/targets?state=active" | jq -r   '.data.activeTargets[] | "\(.labels.job) \(.scrapeUrl) \(.health)"'

# rules loaded, and nothing firing while the stack is healthy
curl -s localhost:9090/api/v1/rules  | jq -r '.data.groups[].rules[] | "\(.name) \(.state)"'
curl -s localhost:9090/api/v1/alerts | jq '.data.alerts | length'

# Grafana provisioning
curl -s -u admin:admin localhost:3000/api/datasources/uid/prometheus/health
curl -s -u admin:admin "localhost:3000/api/search?query=" | jq -r '.[].title'
```

Expect `gateway` and `worker` **up** on `host.docker.internal` when you run them
as host processes, and on the service names when you use
`docker compose --profile app up`. Both targets are listed for each job so
either layout works, so *the other one always reads down* — that is why the
availability alerts aggregate with `max(up{...})` instead of matching a single
target.

The `vllm`, `triton`, `ray` and `kserve` targets read down by design: only one
serving stack runs at a time.

Metrics directly:

```bash
curl -s localhost:8080/metrics | grep ocr_http
curl -s localhost:9101/metrics | grep ocr_pages_total
```

---

## Level 2 — the real model on the GPU

Inside WSL2, with nothing else holding the card:

```bash
make down-gpu                 # free the 3060 first — one stack at a time
make vllm                     # downloads the model on first run (~2 GB), serves :8001
curl localhost:8001/v1/models # ready check
```

Then point the worker at it and re-run anything from level 1:

```bash
OCR_ENGINE_BASE_URL=http://localhost:8001/v1 OCR_MODEL_ID=PaddlePaddle/PaddleOCR-VL \
  make worker
```

If the model fights the pinned vLLM, fall back to
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ` and pin both — then never change the model
again, or the four-stack comparison stops being a comparison.

### Benchmarks

```bash
make corpus                                              # ~1k pages from arXiv PDFs
#   or: python benchmarks/build_corpus.py --pdf-dir ~/my-pdfs --max-pages 500

make bench LABEL=vllm BASE_URL=http://localhost:8001/v1  # concurrency 1/4/8/16
make coldstart LABEL=vllm BASE_URL=http://localhost:8001/v1
make report                                              # -> docs/comparison.md
```

`build_corpus.py` writes a manifest that pins the eval subset (seed 42) so every
stack is measured on byte-identical pages.

**Never publish numbers measured against the mock engine.** It is a load
generator with a fixed cost model; it says nothing about the model or the GPU.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `readyz` shows `"engine":"down"` | mock/vLLM not running, or wrong URL | check `OCR_ENGINE_BASE_URL` |
| `readyz` returns 503 | Redis unreachable | start Redis; check `OCR_REDIS_URL` |
| Job sits in `queued` | no worker consuming | start the worker; check its log for the consumer name |
| `make: command not found` | Windows host | use the raw commands above, or run inside WSL2 |
| `ModuleNotFoundError: ocr_serving` | package not installed | `pip install -e .` |
| Stream returns nothing | job already finished | it replays from the start by default; check `GET /v1/ocr/{id}` |
| `415` on a valid file | extension does not match content | the sniffer names the real type in the error |
| Grafana panels empty | Prometheus cannot reach the host | confirm targets at <http://localhost:9090/targets> |
| `password authentication failed` against compose PostgreSQL | a native PostgreSQL owns port 5432 | `POSTGRES_PORT=5433 docker compose up -d` and update the DSN |
| `promtool: path 'M:/alerts.yml' does not exist` | Git Bash rewrites container paths | prefix the command with `MSYS_NO_PATHCONV=1` |

Deeper failure modes, Redis inspection commands and the deploy checklist are in
the [operations runbook](operations.md).
