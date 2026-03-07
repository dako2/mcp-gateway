"""Parse rate-limit headers from HTTP responses and compute QPS."""

from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class RateLimitInfo:
    qps: float | None = None
    burst: int | None = None
    window_seconds: float | None = None
    remaining: int | None = None
    reset_at: float | None = None          # absolute epoch time
    retry_after: float | None = None       # seconds to wait
    source: str = "default"                # "headers", "429-detected", "default"
    raw_headers: dict[str, str] = field(default_factory=dict)


_HEADER_NAMES = {
    "limit":     ["x-ratelimit-limit", "ratelimit-limit", "x-rate-limit-limit"],
    "remaining": ["x-ratelimit-remaining", "ratelimit-remaining", "x-rate-limit-remaining"],
    "reset":     ["x-ratelimit-reset", "ratelimit-reset", "x-rate-limit-reset"],
    "retry":     ["retry-after"],
}


def _get_header(headers: dict[str, str], keys: list[str]) -> str | None:
    for k in keys:
        v = headers.get(k)
        if v is not None:
            return v
    return None


def parse_rate_limit_headers(headers: dict[str, str], status_code: int = 200) -> RateLimitInfo:
    """Extract rate limit information from response headers.

    Works with both legacy X-RateLimit-* and IETF RateLimit-* headers.
    """
    lower_headers = {k.lower(): v for k, v in headers.items()}

    limit_str = _get_header(lower_headers, _HEADER_NAMES["limit"])
    remaining_str = _get_header(lower_headers, _HEADER_NAMES["remaining"])
    reset_str = _get_header(lower_headers, _HEADER_NAMES["reset"])
    retry_str = _get_header(lower_headers, _HEADER_NAMES["retry"])

    raw = {}
    for k, v in lower_headers.items():
        if "ratelimit" in k or "rate-limit" in k or k == "retry-after":
            raw[k] = v

    info = RateLimitInfo(raw_headers=raw)

    if status_code == 429:
        info.source = "429-detected"
        if retry_str:
            try:
                info.retry_after = float(retry_str)
            except ValueError:
                info.retry_after = 5.0
        else:
            info.retry_after = 5.0
        info.qps = 0.5
        info.burst = 2

    if limit_str:
        try:
            info.burst = int(limit_str.split(",")[0].strip())
        except ValueError:
            pass

    if remaining_str:
        try:
            info.remaining = int(remaining_str.split(",")[0].strip())
        except ValueError:
            pass

    if reset_str:
        try:
            val = float(reset_str)
            if val > 1_000_000_000:
                info.reset_at = val
                info.window_seconds = max(val - time.time(), 1.0)
            else:
                info.window_seconds = max(val, 1.0)
                info.reset_at = time.time() + val
        except ValueError:
            pass

    if info.burst and info.window_seconds and info.source != "429-detected":
        info.qps = info.burst / info.window_seconds
        info.source = "headers"
    elif info.burst and not info.window_seconds and info.source != "429-detected":
        info.window_seconds = 60.0
        info.qps = info.burst / 60.0
        info.source = "headers"

    if raw and info.source == "default":
        info.source = "headers-partial"

    return info
