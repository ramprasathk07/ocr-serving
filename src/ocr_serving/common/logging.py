"""Structured JSON logging with request/job correlation.

Every process (gateway, worker, benchmarks) calls :func:`setup_logging` once at
startup. Log records are emitted as one JSON object per line so Loki/Promtail or
`docker logs | jq` can consume them directly; set ``OCR_LOG_FORMAT=text`` for a
human-friendly console during local development.

Correlation ids live in :mod:`contextvars`, so anything logged inside a request
or a job automatically carries ``request_id`` / ``job_id`` without threading the
value through every call.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import traceback
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)

# LogRecord attributes that are not user-supplied "extra" fields.
_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render a record as a single-line JSON object."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if (rid := request_id_var.get()) is not None:
            payload["request_id"] = rid
        if (jid := job_id_var.get()) is not None:
            payload["job_id"] = jid
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED and not key.startswith("_")
            }
        )
        if record.exc_info:
            exc_type, exc, tb = record.exc_info
            payload["error"] = f"{exc_type.__name__}: {exc}" if exc_type else str(exc)
            payload["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Compact console format: ``12:00:01 INFO worker job=abc msg``."""

    def format(self, record: logging.LogRecord) -> str:
        ids = " ".join(
            f"{k}={v}"
            for k, v in (("req", request_id_var.get()), ("job", job_id_var.get()))
            if v
        )
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<5} {record.name} {ids} {record.getMessage()}".rstrip()
        )
        extras = " ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")
        )
        if extras:
            base = f"{base} | {extras}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


class SafeLogger(logging.Logger):
    """Logger whose ``extra=`` can never crash the caller.

    ``logging`` raises ``KeyError`` if an ``extra`` key collides with a
    LogRecord attribute (``filename``, ``module``, ``args``, ...). A structured
    logger is used from every code path in this service, so a field name chosen
    innocently in a request handler must not be able to 500 the request: the
    colliding key is renamed instead.
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra:
            extra = {(f"{k}_" if k in _RESERVED else k): v for k, v in extra.items()}
        return super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)


logging.setLoggerClass(SafeLogger)


def setup_logging(service: str, level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service) if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own colourised handlers; make them use ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
