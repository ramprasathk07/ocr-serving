#!/usr/bin/env bash
# Triton Inference Server with the vLLM backend, plus Triton's OpenAI-compatible
# frontend so the same harness and the same gateway work unchanged.
#
# Engine args live in model_repository/ocr_vlm/1/model.json; config.pbtxt sets
# decoupled mode, which is what makes token streaming possible.
set -euo pipefail

IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:25.01-vllm-python-py3}"
REPO="$(cd "$(dirname "$0")" && pwd)/model_repository"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# Ports: 8000 OpenAI frontend (what the pipeline talks to), 8002 Prometheus metrics.
docker run --rm --name triton --gpus all --shm-size=2g \
  -p 8000:9000 -p 8002:8002 \
  -v "${REPO}:/models:ro" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  "${IMAGE}" \
  bash -lc '
    tritonserver --model-repository=/models --allow-metrics=true &
    # The OpenAI frontend ships in the server image (python/openai in recent releases);
    # it translates /v1/chat/completions onto the decoupled generate stream.
    python3 /opt/tritonserver/python/openai/openai_frontend/main.py \
      --model-repository /models --tokenizer "${OCR_MODEL_ID:-PaddlePaddle/PaddleOCR-VL}" \
      --openai-port 9000
  '
