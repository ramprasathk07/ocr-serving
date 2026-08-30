"""API-key auth, tenant resolution, and per-key rate limiting.

The "Auth / Rate Limit / Tenant Quotas" box. Keys are configured as
``OCR_API_KEYS=key1:tenant-a,key2:tenant-b`` (a bare ``key`` maps to tenant
``default``); real deployments swap this dependency for the identity provider
without touching the routes.

Key comparison uses :func:`secrets.compare_digest` — a plain ``==`` on a secret
leaks its prefix through timing.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status

from ocr_serving.common.config import get_settings
from ocr_serving.common.metrics import RATE_LIMITED


@dataclass(frozen=True, slots=True)
class Principal:
    api_key: str
    tenant: str

    @property
    def masked(self) -> str:
        return f"{self.api_key[:4]}***" if len(self.api_key) > 4 else "***"


async def require_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
    authorization: str = Header(default=""),
) -> Principal:
    """Accept ``X-API-Key: k`` or ``Authorization: Bearer k``."""
    presented = x_api_key.strip()
    if not presented and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    for key, tenant in get_settings().api_key_map.items():
        if secrets.compare_digest(presented, key):
            return Principal(api_key=key, tenant=tenant)

    RATE_LIMITED.labels(reason="auth").inc()
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def rate_limited(request: Request, principal: Principal) -> None:
    """Token-bucket check; raises 429 with ``Retry-After`` when exhausted."""
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is None:
        return
    decision = await limiter.check(principal.api_key)
    if not decision.allowed:
        RATE_LIMITED.labels(reason="rate_limit").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded, retry in {decision.retry_after_s}s",
            headers={"Retry-After": str(max(int(decision.retry_after_s), 1))},
        )
