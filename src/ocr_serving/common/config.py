"""Central settings — every process (gateway, worker, benchmarks) reads the same env.

All variables use the ``OCR_`` prefix and are documented in ``.env.example``.
Settings are validated once at import of :func:`get_settings` and cached, so a
typo in the environment fails fast at startup instead of at request time.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ service
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # ------------------------------------------------------------------- engine
    model_id: str = "PaddlePaddle/PaddleOCR-VL"
    #: OpenAI-compatible endpoint of whichever serving stack is active.
    engine_base_url: str = "http://localhost:8001/v1"
    engine_api_key: str = "EMPTY"
    engine_timeout_s: float = 300.0
    engine_connect_timeout_s: float = 10.0
    engine_max_retries: int = 2
    engine_max_tokens: int = 512
    #: Client-side admission control: in-flight requests the worker allows to the engine.
    engine_max_concurrency: int = 16

    # --------------------------------------------------------------- datastores
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://ocr:ocr@localhost:5432/ocr"
    postgres_enabled: bool = True
    storage_dir: Path = Path("./storage")

    # --------------------------------------------------------------- gateway API
    api_keys: str = "dev-key"
    max_upload_mb: int = 64
    cors_origins: str = "*"
    #: Token-bucket rate limit applied per API key.
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 20
    #: Rolling 24 h page budget per tenant. 0 disables the quota.
    quota_pages_per_day: int = 5000

    # ------------------------------------------------------------------- queue
    queue_stream: str = "ocr:jobs"
    queue_group: str = "workers"
    #: A job whose worker stops heartbeating for this long is reclaimed by a peer.
    visibility_timeout_s: int = 300
    max_attempts: int = 3
    dead_letter_stream: str = "ocr:jobs:dead"

    # ------------------------------------------------------------------ events
    #: Per-job event stream cap (approximate trim) and retention after completion.
    event_stream_maxlen: int = 10_000
    event_ttl_s: int = 3600
    sse_keepalive_s: float = 15.0

    # ------------------------------------------------------------------ worker
    worker_concurrency: int = 2
    page_concurrency: int = 2
    region_concurrency: int = 2
    worker_metrics_port: int = 9101
    worker_prefetch: int = 1

    # ------------------------------------------------------------- page pipeline
    render_dpi: int = 150
    adaptive_dpi: bool = True
    min_dpi: int = 110
    max_dpi: int = 250
    max_pages: int = 200
    max_image_px: int = 4_000_000
    #: Pages whose embedded text layer is this long skip the GPU entirely.
    native_text_min_chars: int = 200
    native_text_extraction: bool = True
    blank_std_threshold: float = 4.0
    deduplicate_pages: bool = True
    deskew: bool = True
    deskew_max_angle: float = 15.0
    denoise: bool = False
    clahe: bool = True

    # ------------------------------------------------------------------- layout
    layout_enabled: bool = True
    layout_model_path: Path = Path("models/doclayout_yolo.onnx")
    layout_score_threshold: float = 0.3
    layout_iou_threshold: float = 0.45
    #: Regions taller than this many pixels are tiled before hitting the encoder.
    tile_max_height: int = 1600
    tile_overlap: int = 64

    # ---------------------------------------------------------------- artifacts
    searchable_pdf: bool = True
    result_ttl_days: int = 7
    #: Periodic disk cleanup in the worker. 0 disables the sweep.
    retention_sweep_hours: float = 6.0

    api_title: str = "XFinite-OCR Gateway"
    api_version: str = "1.0.0"

    hf_token: str = Field(default="", repr=False)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level {v!r}")
        return v

    # ------------------------------------------------------------------- keys
    @property
    def api_key_map(self) -> dict[str, str]:
        """``key`` or ``key:tenant`` entries -> ``{key: tenant}``."""
        out: dict[str, str] = {}
        for raw in self.api_keys.split(","):
            raw = raw.strip()
            if not raw:
                continue
            key, _, tenant = raw.partition(":")
            out[key] = tenant or "default"
        return out

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    # -------------------------------------------------------------- redis keys
    def job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def events_stream(self, job_id: str) -> str:
        return f"job:{job_id}:events"

    def cancel_key(self, job_id: str) -> str:
        return f"job:{job_id}:cancel"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.uploads_dir, s.results_dir, s.artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
