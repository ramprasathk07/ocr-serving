# Cloudflare Tunnel (Week 1 Day 5) — public demo without opening ports

```bash
# quick throwaway (no domain needed):
cloudflared tunnel --url http://localhost:8080

# named tunnel (stable URL on your domain):
cloudflared tunnel login
cloudflared tunnel create xfinite-ocr
cloudflared tunnel route dns xfinite-ocr ocr.yourdomain.com
cloudflared tunnel run xfinite-ocr
```

config (`~/.cloudflared/config.yml`):

```yaml
tunnel: xfinite-ocr
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: ocr.yourdomain.com
    service: http://localhost:8080
  - service: http_status:404
```

SSE works through the tunnel out of the box. Keep `x-api-key` auth on — this is public.
