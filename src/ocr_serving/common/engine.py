"""Client for any OpenAI-compatible OCR serving endpoint.

Every stack in this repo (vLLM standalone, Triton vLLM backend, Ray Serve via
``serve.llm``, KServe huggingface runtime, and the CPU mock) exposes
``/v1/chat/completions``. The worker and the benchmark harness talk through this
one client, so switching stacks is a ``base_url`` change and nothing else.

Production behaviour beyond a bare POST:

* **client-side admission control** — a semaphore caps in-flight requests so a
  burst of pages cannot bury the scheduler queue inside the engine;
* **bounded retries with jittered backoff**, and only while no token has been
  emitted yet (a resumed stream would otherwise duplicate text);
* **accurate token accounting** via ``stream_options.include_usage`` when the
  backend supports it, falling back to counting deltas;
* every request timed and counted into Prometheus.
"""
from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from dataclasses import dataclass, field

import httpx

from ocr_serving.common.logging import get_logger
from ocr_serving.common.metrics import (
    ENGINE_INFLIGHT,
    ENGINE_REQUESTS,
    ENGINE_RETRIES,
    ENGINE_SECONDS,
    ENGINE_TOKENS,
    TTFT_SECONDS,
)

log = get_logger(__name__)

OCR_PROMPT = (
    "OCR this document page. Output all text in markdown, preserving reading order. "
    "Render tables as markdown tables."
)

#: Status codes worth retrying: engine overloaded / restarting / gateway hiccup.
RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class EngineError(RuntimeError):
    """Engine call failed after exhausting retries."""


def png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


@dataclass
class StreamStats:
    ttft_s: float = 0.0
    e2e_s: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    text: str = ""
    error: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tokens_per_s(self) -> float:
        gen = self.e2e_s - self.ttft_s
        return self.completion_tokens / gen if gen > 0 and self.completion_tokens else 0.0


@dataclass
class OCRClient:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 300.0
    connect_timeout: float = 10.0
    max_tokens: int = 512
    max_retries: int = 2
    max_concurrency: int = 16
    prompt: str = OCR_PROMPT
    _client: httpx.AsyncClient = field(init=False, repr=False)
    _sem: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            limits=httpx.Limits(
                max_connections=self.max_concurrency * 2,
                max_keepalive_connections=self.max_concurrency,
            ),
        )
        self._sem = asyncio.Semaphore(self.max_concurrency)

    # ------------------------------------------------------------------ calls
    def _payload(self, png_bytes: bytes, stream: bool, prompt: str | None = None) -> dict:
        body: dict = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": png_to_data_uri(png_bytes)}},
                        {"type": "text", "text": prompt or self.prompt},
                    ],
                }
            ],
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def ocr_page_stream(
        self, png_bytes: bytes, on_delta=None, prompt: str | None = None
    ) -> StreamStats:
        """Stream one page/region. TTFT is measured to the first content delta.

        Never raises: transport and protocol failures are returned in
        ``StreamStats.error`` so a benchmark run records them instead of dying.
        """
        stats = StreamStats()
        for attempt in range(1, self.max_retries + 2):
            stats.attempts = attempt
            emitted = False
            start = time.perf_counter()
            try:
                async with self._sem:
                    ENGINE_INFLIGHT.inc()
                    try:
                        emitted = await self._consume_stream(
                            png_bytes, stats, on_delta, prompt, start
                        )
                    finally:
                        ENGINE_INFLIGHT.dec()
                stats.e2e_s = time.perf_counter() - start
                stats.error = None
                ENGINE_REQUESTS.labels(outcome="ok").inc()
                ENGINE_SECONDS.observe(stats.e2e_s)
                ENGINE_TOKENS.inc(stats.completion_tokens)
                if stats.ttft_s:
                    TTFT_SECONDS.observe(stats.ttft_s)
                return stats
            except Exception as exc:
                stats.e2e_s = time.perf_counter() - start
                stats.error = f"{type(exc).__name__}: {exc}"
                retryable = self._retryable(exc) and not emitted
                if not retryable or attempt > self.max_retries:
                    ENGINE_REQUESTS.labels(outcome="error").inc()
                    log.warning(
                        "engine request failed",
                        extra={"engine_error": stats.error, "attempt": attempt},
                    )
                    return stats
                ENGINE_RETRIES.inc()
                backoff = min(2 ** (attempt - 1), 8) * (0.5 + random.random())
                log.info(
                    "retrying engine request",
                    extra={"attempt": attempt, "sleep_s": round(backoff, 2)},
                )
                stats.text, stats.completion_tokens, stats.ttft_s = "", 0, 0.0
                await asyncio.sleep(backoff)
        return stats

    async def _consume_stream(
        self, png_bytes: bytes, stats: StreamStats, on_delta, prompt: str | None, start: float
    ) -> bool:
        """Drive one SSE response into ``stats``; returns True once a delta was emitted."""
        emitted = False
        async with self._client.stream(
            "POST", "/chat/completions", json=self._payload(png_bytes, True, prompt)
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if usage := chunk.get("usage"):
                    stats.completion_tokens = (
                        usage.get("completion_tokens") or stats.completion_tokens
                    )
                    stats.prompt_tokens = usage.get("prompt_tokens") or stats.prompt_tokens
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if not delta:
                    continue
                if not emitted:
                    stats.ttft_s = time.perf_counter() - start
                    emitted = True
                stats.text += delta
                if not stats.prompt_tokens:  # only count chunks when usage is absent
                    stats.completion_tokens += 1
                if on_delta is not None:
                    await on_delta(delta)
        return emitted

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in RETRY_STATUS
        return isinstance(
            exc, (httpx.TransportError, httpx.RemoteProtocolError, asyncio.TimeoutError)
        )

    # ----------------------------------------------------------------- health
    async def health(self) -> bool:
        try:
            resp = await self._client.get("/models", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OCRClient:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()


def client_from_settings(settings, **overrides) -> OCRClient:
    """Build a client from :class:`common.config.Settings` (single source of truth)."""
    kwargs = {
        "base_url": settings.engine_base_url,
        "model": settings.model_id,
        "api_key": settings.engine_api_key,
        "timeout": settings.engine_timeout_s,
        "connect_timeout": settings.engine_connect_timeout_s,
        "max_tokens": settings.engine_max_tokens,
        "max_retries": settings.engine_max_retries,
        "max_concurrency": settings.engine_max_concurrency,
    }
    kwargs.update(overrides)
    return OCRClient(**kwargs)
