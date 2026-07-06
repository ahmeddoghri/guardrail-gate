"""Per-client token bucket rate limiting.

A guardrail service that itself has no abuse protection is a liability --
a misbehaving client hammering /v1/guard shouldn't be able to degrade it for
everyone else.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    def __init__(self, capacity: int = 20, refill_per_second: float = 5.0, clock=time.time) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, client_id: str) -> bool:
        now = self._clock()
        bucket = self._buckets.get(client_id)
        if bucket is None:
            bucket = _Bucket(tokens=self.capacity - 1, last_refill=now)
            self._buckets[client_id] = bucket
            return True

        elapsed = max(0.0, now - bucket.last_refill)
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False
