"""Per-host adaptive token bucket rate limiter."""

from __future__ import annotations
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from qps_discovery import RateLimitInfo, parse_rate_limit_headers

DEFAULT_QPS = 2.0
DEFAULT_BURST = 5
MIN_QPS = 0.1
MAX_QPS = 500.0


@dataclass
class TokenBucket:
    capacity: float
    refill_rate: float        # tokens per second (= QPS)
    tokens: float = 0.0
    last_refill: float = 0.0
    consecutive_429s: int = 0

    def __post_init__(self):
        if self.last_refill == 0.0:
            self.last_refill = time.monotonic()
        self.tokens = self.capacity

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_acquire(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def wait_time(self) -> float:
        self._refill()
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate

    def apply_429(self, retry_after: float | None = None):
        self.consecutive_429s += 1
        self.refill_rate = max(MIN_QPS, self.refill_rate * 0.5)
        self.tokens = 0.0
        if retry_after:
            self.last_refill = time.monotonic() + retry_after

    def apply_success(self):
        if self.consecutive_429s > 0:
            self.consecutive_429s = max(0, self.consecutive_429s - 1)
            if self.consecutive_429s == 0:
                self.refill_rate = min(self.capacity, self.refill_rate * 1.2)

    def update_rate(self, qps: float, burst: int | None = None):
        self.refill_rate = max(MIN_QPS, min(MAX_QPS, qps))
        if burst is not None:
            self.capacity = max(1, burst)
        self.consecutive_429s = 0


class AdaptiveRateLimiter:
    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _host_key(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc.lower()

    def _get_or_create_bucket(self, host: str, qps: float = DEFAULT_QPS, burst: int = DEFAULT_BURST) -> TokenBucket:
        if host not in self._buckets:
            self._buckets[host] = TokenBucket(capacity=burst, refill_rate=qps)
        return self._buckets[host]

    async def _get_lock(self, host: str) -> asyncio.Lock:
        async with self._global_lock:
            if host not in self._locks:
                self._locks[host] = asyncio.Lock()
            return self._locks[host]

    async def acquire(self, url: str) -> None:
        host = self._host_key(url)
        bucket = self._get_or_create_bucket(host)

        while True:
            if bucket.try_acquire():
                return
            wait = bucket.wait_time()
            await asyncio.sleep(max(0.01, wait))

    def update_from_headers(self, url: str, headers: dict[str, str], status_code: int) -> None:
        host = self._host_key(url)
        info = parse_rate_limit_headers(headers, status_code)

        bucket = self._get_or_create_bucket(host)

        if status_code == 429:
            bucket.apply_429(info.retry_after)
            return

        bucket.apply_success()

        if info.qps and info.source in ("headers", "headers-partial"):
            bucket.update_rate(info.qps, info.burst)

    def handle_429(self, url: str, retry_after: float | None = None) -> float:
        host = self._host_key(url)
        bucket = self._get_or_create_bucket(host)
        bucket.apply_429(retry_after)
        return retry_after or (2 ** min(bucket.consecutive_429s, 6))

    def load_from_servers_json(self, servers_path: str) -> int:
        with open(servers_path) as f:
            servers = json.load(f)

        loaded = 0
        for entry in servers:
            if entry.get("health") != "ok":
                continue
            url = entry.get("url", "")
            if not url:
                continue

            rl = entry.get("rate_limit", {})
            qps = rl.get("qps", DEFAULT_QPS)
            burst = rl.get("burst", DEFAULT_BURST)

            host = self._host_key(url)
            self._buckets[host] = TokenBucket(capacity=burst, refill_rate=qps)
            loaded += 1

        return loaded

    def save(self, path: str) -> None:
        data = {}
        for host, bucket in self._buckets.items():
            data[host] = {
                "qps": round(bucket.refill_rate, 4),
                "burst": int(bucket.capacity),
                "consecutive_429s": bucket.consecutive_429s,
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            data = json.load(f)

        for host, info in data.items():
            self._buckets[host] = TokenBucket(
                capacity=info.get("burst", DEFAULT_BURST),
                refill_rate=info.get("qps", DEFAULT_QPS),
            )
            self._buckets[host].consecutive_429s = info.get("consecutive_429s", 0)

        return len(data)

    def get_stats(self) -> dict:
        return {
            "total_hosts": len(self._buckets),
            "hosts": {
                host: {
                    "qps": round(b.refill_rate, 2),
                    "burst": int(b.capacity),
                    "tokens": round(b.tokens, 1),
                    "consecutive_429s": b.consecutive_429s,
                }
                for host, b in sorted(self._buckets.items())
            },
        }
