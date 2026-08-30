# Monitoring

```bash
docker compose up -d                     # prometheus :9090, grafana :3000 (admin/admin)
docker compose --profile gpu up -d       # + dcgm-exporter :9400 (GPU util, VRAM)
```

Grafana opens on the **OCR pipeline** dashboard. Both the dashboard and the datasource are
provisioned from this directory, so changes belong in git rather than in Grafana's database:

```
monitoring/
  prometheus.yml                                    scrape targets (gateway, worker, active stack, dcgm)
  alerts.yml                                        alert rules, loaded by prometheus
  grafana/provisioning/datasources/prometheus.yml
  grafana/provisioning/dashboards/ocr-pipeline.json
```

## What is exported

| Metric | Source | Answers |
|---|---|---|
| `ocr_ttft_seconds` | worker | how fast does the first token arrive |
| `ocr_page_seconds`, `ocr_job_seconds`, `ocr_queue_wait_seconds` | worker | where the wall time goes |
| `ocr_pages_total{source}` | worker | how many pages avoided the GPU (native / duplicate / blank) |
| `ocr_queue_depth`, `_pending`, `_dead` | both | is the queue keeping up; is anything stuck |
| `ocr_engine_*` | worker | engine throughput, retries, in-flight, error rate |
| `ocr_http_*`, `ocr_stream_clients`, `ocr_rate_limited_total` | gateway | API health and who is being throttled |
| `DCGM_FI_DEV_*` | dcgm-exporter | GPU utilisation and VRAM |

Alerts cover: gateway down, no worker consuming, queue backlog, dead-lettered jobs, engine error
rate, TTFT regression, GPU memory nearly full. See `alerts.yml` for thresholds and the runbook
line each one carries.

## Third-party dashboards worth importing alongside

- **DCGM GPU**: Grafana.com dashboard ID `12239`
- **vLLM**: `vllm/examples/online_serving/prometheus_grafana` (engine-side queue and KV cache)
