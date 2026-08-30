"""Rate limiting and tenant quotas — the "Auth / Rate Limit / Tenant Quotas" box.

Two independent controls, both server-side in Redis so they hold across gateway
replicas:

* **token bucket** per API key — smooths bursts, atomic via a Lua script so two
  concurrent requests cannot both consume the last token;
* **rolling daily page quota** per tenant — counts *pages*, the unit that costs
  GPU time, not requests.

Both fail open: if Redis is unreachable the gateway serves traffic rather than
returning 500s, and the failure is logged and counted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import redis.asyncio as aioredis

from ocr_serving.common.logging import get_logger

log = get_logger(__name__)

# Classic token bucket. Returns {allowed, tokens_left, retry_after_ms}.
_BUCKET_LUA = """
local key   = KEYS[1]
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = burst
  ts = now
end

local elapsed = math.max(0, now - ts) / 1000.0
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry_after = math.ceil(((cost - tokens) / rate) * 1000)
end

redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, math.ceil((burst / rate) * 1000) + 1000)
return {allowed, math.floor(tokens), retry_after}
"""


@dataclass(slots=True)
class LimitDecision:
    allowed: bool
    remaining: int = 0
    retry_after_s: float = 0.0
    reason: str = ""


class RateLimiter:
    def __init__(self, redis: aioredis.Redis, rate: float, burst: int) -> None:
        self.r = redis
        self.rate = rate
        self.burst = burst
        self._script = redis.register_script(_BUCKET_LUA)

    async def check(self, key: str, cost: int = 1) -> LimitDecision:
        if self.rate <= 0:
            return LimitDecision(True)
        try:
            allowed, tokens, retry_ms = await self._script(
                keys=[f"rl:{key}"],
                args=[self.rate, self.burst, int(time.time() * 1000), cost],
            )
        except Exception as exc:  # never let the limiter take the API down
            log.warning("rate limiter unavailable, failing open", extra={"limiter_error": str(exc)})
            return LimitDecision(True)
        return LimitDecision(
            allowed=bool(allowed),
            remaining=int(tokens),
            retry_after_s=round(int(retry_ms) / 1000, 2),
            reason="" if allowed else "rate_limit",
        )


class PageQuota:
    """Rolling 24 h page budget per tenant."""

    def __init__(self, redis: aioredis.Redis, limit_per_day: int) -> None:
        self.r = redis
        self.limit = limit_per_day

    @staticmethod
    def _key(tenant: str) -> str:
        return f"quota:{tenant}:{time.strftime('%Y%m%d', time.gmtime())}"

    async def used(self, tenant: str) -> int:
        try:
            return int(await self.r.get(self._key(tenant)) or 0)
        except Exception:
            return 0

    async def check(self, tenant: str, pages: int = 1) -> LimitDecision:
        if self.limit <= 0:
            return LimitDecision(True)
        used = await self.used(tenant)
        if used + pages > self.limit:
            return LimitDecision(False, remaining=max(self.limit - used, 0), reason="quota")
        return LimitDecision(True, remaining=self.limit - used - pages)

    async def consume(self, tenant: str, pages: int) -> int:
        """Charge pages after the fact (the real page count is known post-render)."""
        if self.limit <= 0 or pages <= 0:
            return 0
        try:
            key = self._key(tenant)
            async with self.r.pipeline(transaction=True) as pipe:
                pipe.incrby(key, pages)
                pipe.expire(key, 172_800)
                used, _ = await pipe.execute()
            return int(used)
        except Exception as exc:
            log.warning("quota accounting failed", extra={"quota_error": str(exc)})
            return 0
