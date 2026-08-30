"""Mock OpenAI-compatible OCR engine — CPU only, no GPU, no model download.

Why this exists: the pipeline (gateway, queue, worker, streaming, artifacts,
dashboards) must be developable and testable on a laptop and in CI, where a
12 GB VLM is not available. It speaks exactly the wire protocol the four real
stacks speak, so nothing in the pipeline knows the difference.

    uvicorn serving.mock.server:app --port 8001

Tunables (env):
    MOCK_TTFT_MS=120       simulated prefill latency
    MOCK_TOKEN_MS=8        per-token decode latency
    MOCK_TOKENS=180        tokens per response
    MOCK_FAIL_RATE=0.0     fraction of requests that 503 (retry-path testing)

It is a load generator with a fixed cost model, not an OCR engine: use it to
exercise plumbing and dashboards, never to produce benchmark numbers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

TTFT_MS = float(os.environ.get("MOCK_TTFT_MS", 120))
TOKEN_MS = float(os.environ.get("MOCK_TOKEN_MS", 8))
TOKENS = int(os.environ.get("MOCK_TOKENS", 180))
FAIL_RATE = float(os.environ.get("MOCK_FAIL_RATE", 0.0))
MODEL = os.environ.get("OCR_MODEL_ID", "mock-ocr-vl")

app = FastAPI(title="mock OCR engine", version="1.0.0")

_LEXICON = [
    "invoice", "total", "amount", "due", "date", "customer", "address", "line", "item",
    "quantity", "unit", "price", "subtotal", "tax", "shipping", "handling", "payment",
    "terms", "net", "thirty", "reference", "number", "purchase", "order", "description",
    "account", "balance", "signature", "authorised", "representative",
]


def _deterministic_text(seed_material: bytes, tokens: int) -> list[str]:
    """Same image in, same text out — so benchmarks and tests are reproducible."""
    rng = random.Random(hashlib.sha256(seed_material).hexdigest())
    words = []
    for i in range(tokens):
        if i and i % 12 == 0:
            words.append("\n")
        words.append(rng.choice(_LEXICON))
    return [w if w == "\n" else (" " + w if i else w) for i, w in enumerate(words)]


def _image_seed(messages: list) -> bytes:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    return part["image_url"]["url"][-256:].encode()
    return b"no-image"


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]}


@app.get("/health")
@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if FAIL_RATE and random.random() < FAIL_RATE:
        raise HTTPException(503, "mock engine overloaded")

    messages = body.get("messages", [])
    max_tokens = min(int(body.get("max_tokens") or TOKENS), TOKENS)
    pieces = _deterministic_text(_image_seed(messages), max_tokens)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if not body.get("stream"):
        await asyncio.sleep(TTFT_MS / 1000 + len(pieces) * TOKEN_MS / 1000)
        text = "".join(pieces)
        return {
            "id": completion_id, "object": "chat.completion", "created": created, "model": MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 512, "completion_tokens": len(pieces),
                      "total_tokens": 512 + len(pieces)},
        }

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def generate():
        def frame(delta: dict, finish=None) -> str:
            payload = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload)}\n\n"

        await asyncio.sleep(TTFT_MS / 1000)
        yield frame({"role": "assistant", "content": ""})
        for piece in pieces:
            await asyncio.sleep(TOKEN_MS / 1000)
            yield frame({"content": piece})
        yield frame({}, finish="stop")
        if include_usage:
            usage = {
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": MODEL, "choices": [],
                "usage": {"prompt_tokens": 512, "completion_tokens": len(pieces),
                          "total_tokens": 512 + len(pieces)},
            }
            yield f"data: {json.dumps(usage)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})
