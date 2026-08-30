# KServe on k3s (Week 3)

```bash
bash deploy/kserve/setup/install-k3s-gpu.sh        # step through it, don't blind-run
kubectl apply -f deploy/kserve/inferenceservice.yaml
kubectl get isvc ocr-vlm -w                          # READY=True -> screenshot
kubectl port-forward svc/ocr-vlm-predictor 8005:80
make bench LABEL=kserve BASE_URL=http://localhost:8005/openai/v1
make coldstart LABEL=kserve --mode k8s BASE_URL=http://localhost:8005/openai/v1
```

Quantize first (Day 3, before deploy): llm-compressor AWQ-int4 recipe → push to HF →
point `--model_id` at it. Fallback: skip quant, serve the bf16 0.9B (fits 12 GB fine) and
note quantization as the prod story.

Cold start protocol: `kubectl delete pod -l serving.kserve.io/inferenceservice=ocr-vlm`,
measure delete → Ready → first token (`benchmarks/coldstart.py --base-url ...`).

Ops ledger (Day 5): install wall-time, `free -g` delta with k3s+KServe idle, `wc -l` of yaml,
moving parts (k3s, cert-manager, kserve controller, device plugin, HPA). Record it into
`benchmarks/results/ops_ledger.json` (template: `benchmarks/ops_ledger.example.json`) so
`make report` renders it next to the latency table — that pairing is the whole argument.

Fallbacks if WSL2 GPU fights (budget Day 1 for this): k3d with CUDA k3s image →
`minikube --driver=docker --gpus=all`.
