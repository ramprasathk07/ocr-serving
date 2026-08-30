#!/usr/bin/env bash
# Baseline stack: standalone vLLM OpenAI server on :8001. Run inside WSL2.
# Prometheus metrics come free at :8001/metrics (already a scrape target).
#
#   bash serving/vllm/serve.sh                 # defaults below
#   OCR_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct-AWQ bash serving/vllm/serve.sh
set -euo pipefail

MODEL="${OCR_MODEL_ID:-PaddlePaddle/PaddleOCR-VL}"
PORT="${VLLM_PORT:-8001}"
# 3060 12 GB: 0.85 leaves headroom for the desktop compositor. Drop to ~0.42 when
# running two fractional-GPU replicas under Ray Serve.
GPU_FRACTION="${VLLM_GPU_FRACTION:-0.85}"
# 8192 is plenty for one page and keeps the KV cache small.
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
# One image per request — the pipeline sends a page or a region, never a gallery.
MM_LIMIT="${VLLM_MM_LIMIT:-image=1}"

echo "serving ${MODEL} on :${PORT} (gpu-util=${GPU_FRACTION}, max-len=${MAX_LEN})"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

# Add --quantization awq for the AWQ-int4 build used in the KServe week.
exec vllm serve "${MODEL}" \
  --host 0.0.0.0 --port "${PORT}" \
  --gpu-memory-utilization "${GPU_FRACTION}" \
  --max-model-len "${MAX_LEN}" \
  --limit-mm-per-prompt "${MM_LIMIT}" \
  --disable-log-requests \
  --trust-remote-code
