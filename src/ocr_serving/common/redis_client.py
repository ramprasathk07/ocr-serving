"""One shared asyncio Redis connection pool per process.

The previous iteration opened (and leaked) a connection per HTTP request. A
single pool created at startup and closed on shutdown is both faster and the
only way blocking reads (XREAD/XREADGROUP) can be used safely alongside normal
commands: blocking calls take a dedicated connection from the same pool.
"""
from __future__ import annotations

import redis.asyncio as aioredis

from ocr_serving.common.config import get_settings

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Process-wide client. Lazily created; safe to call from anywhere."""
    global _client
    if _client is None:
        s = get_settings()
        _client = aioredis.from_url(
            s.redis_url,
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True,
            retry_on_timeout=True,
            max_connections=64,
        )
    return _client


def set_redis(client: aioredis.Redis | None) -> None:
    """Inject a client (tests use fakeredis); pass ``None`` to reset."""
    global _client
    _client = client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def ping() -> bool:
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False
