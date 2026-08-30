# Triton — vLLM backend

```bash
make triton                                 # deploy/triton/serve.sh
curl localhost:8000/v1/models
make bench LABEL=triton BASE_URL=http://localhost:8000/v1
```

Two files define the deployment: `model_repository/ocr_vlm/config.pbtxt` (backend, and
`decoupled: True`, which is what allows token streaming) and `1/model.json` (engine args —
keep them numerically identical to `deploy/vllm/serve.sh`, or the comparison is measuring
different engines).

Triton's native API is `/v2/models/ocr_vlm/generate`, not OpenAI chat. Recent server images ship
an OpenAI-compatible frontend, which `serve.sh` starts so the gateway and harness work unchanged;
if your image lacks it, either pull a newer `-vllm-python-py3` tag or run the frontend from the
[server repo](https://github.com/triton-inference-server/server/tree/main/python/openai).

Pick the `<yy.mm>` image whose bundled vLLM supports the chosen model — see the
[vllm_backend release matrix](https://github.com/triton-inference-server/vllm_backend).
Metrics: `:8002/metrics`.
