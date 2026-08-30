# Everything GPU-touching runs inside WSL2. `make help` lists targets.
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= uv run
LABEL ?= vllm
BASE_URL ?= http://localhost:8001/v1

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- dev setup
install: ## sync python deps (CPU dev set)
	uv sync --extra worker --extra bench --extra dev

install-gpu: ## sync everything, inside WSL2 only
	uv sync --all-extras

lint: ## ruff
	$(PY) ruff check .

fmt: ## ruff --fix
	$(PY) ruff check --fix .

test: ## pytest (no redis, no GPU, no model needed)
	$(PY) pytest

# --------------------------------------------------------------------- infra
up: ## start infra: redis, postgres, prometheus, grafana
	docker compose up -d

up-app: ## + containerised gateway and worker
	docker compose --profile app up -d --build

up-mock: ## + CPU mock engine on :8001 (no GPU)
	docker compose --profile mock up -d --build

up-gpu-metrics: ## + dcgm-exporter :9400
	docker compose --profile gpu up -d

down: ## stop everything in compose
	docker compose --profile app --profile mock --profile gpu down

logs: ## tail gateway + worker logs
	docker compose logs -f gateway worker

# ------------------------------------------------------------------ services
gateway: ## run FastAPI gateway :8080
	$(PY) uvicorn ocr_serving.gateway.main:app --host 0.0.0.0 --port 8080 --reload

worker: ## run CPU OCR worker
	$(PY) python -m ocr_serving.workers.cpu_worker

mock: ## run the CPU mock engine :8001 (develop without a GPU)
	$(PY) uvicorn ocr_serving.serving.mock.server:app --host 0.0.0.0 --port 8001

demo: ## open the streaming demo page
	@echo "http://localhost:8080/  (API key: the first entry of OCR_API_KEYS)"

# ------------------------------------------------------- serving stacks (GPU)
vllm: ## serve model with standalone vLLM :8001
	bash deploy/vllm/serve.sh

ray-serve: ## serve model with Ray Serve :8000
	$(PY) serve run deploy/ray/serve_config.yaml

ray-batch: ## Ray Data batch inference over the corpus
	$(PY) python -m ocr_serving.serving.ray.batch_inference --corpus benchmarks/corpus/pages

triton: ## serve model with Triton (vLLM backend) :8000/:8002
	bash deploy/triton/serve.sh

down-gpu: ## kill whatever owns the GPU (one stack at a time!)
	-pkill -f "vllm serve" || true
	-$(PY) serve shutdown -y || true
	-docker rm -f triton 2>/dev/null || true
	-kubectl delete isvc ocr-vlm --ignore-not-found 2>/dev/null || true
	@nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

# ---------------------------------------------------------------- benchmarks
corpus: ## build the 1k-page benchmark corpus
	$(PY) python benchmarks/build_corpus.py --max-pages 1000

bench: ## bench the active stack: make bench LABEL=vllm BASE_URL=http://localhost:8001/v1
	$(PY) python benchmarks/harness.py --label $(LABEL) --base-url $(BASE_URL)

coldstart: ## measure cold start: make coldstart LABEL=vllm BASE_URL=...
	$(PY) python benchmarks/coldstart.py --label $(LABEL) --base-url $(BASE_URL)

report: ## results/*.json -> docs/comparison.md tables
	$(PY) python benchmarks/report.py

smoke: ## end-to-end check against a running gateway (make smoke FILE=doc.pdf)
	$(PY) python scripts/smoke.py --file $(FILE)

.PHONY: help install install-gpu lint fmt test up up-app up-mock up-gpu-metrics down logs \
        gateway worker mock demo vllm ray-serve ray-batch triton down-gpu corpus bench \
        coldstart report smoke
