# Mock engine — CPU stand-in for the serving stack

Speaks the same OpenAI-compatible streaming API as vLLM, Triton, Ray Serve and KServe, so the
gateway, worker, dashboards and tests run on a laptop with no GPU and no model download.

```bash
make mock                                   # :8001
MOCK_TTFT_MS=800 MOCK_TOKEN_MS=25 make mock # simulate a slow stack
MOCK_FAIL_RATE=0.2 make mock                # exercise the retry path
```

| Variable | Default | Effect |
|---|---|---|
| `MOCK_TTFT_MS` | 120 | simulated prefill latency |
| `MOCK_TOKEN_MS` | 8 | per-token decode latency |
| `MOCK_TOKENS` | 180 | tokens per response |
| `MOCK_FAIL_RATE` | 0 | fraction of requests that return 503 |

Output is deterministic per input image, so tests and repeated runs agree.

**Never publish numbers measured against this.** It is a load generator with a fixed cost model,
not an OCR engine: it tests plumbing, backpressure and dashboards, not the model.
