"""MCP Gateway -- rate-limit-aware proxy for remote MCP servers.

Usage:
    cd src && uvicorn gateway:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /mcp           Forward an MCP request to a target server (rate-limited)
    POST /proxy/{b64}   Transparent MCP proxy (use as MCPServerStreamableHttp URL)
    GET  /stats         View per-host rate limiter stats
    GET  /health        Gateway health check
    GET  /servers       List known healthy servers
"""

from __future__ import annotations
import asyncio
import json
import os
import time
import functools
import base64

from dotenv import load_dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse, Response
from starlette.routing import Route

from rate_limiter import AdaptiveRateLimiter

SERVERS_JSON = os.path.join(_project_root, "data", "servers.json")
LEARNED_RATES = os.path.join(_project_root, "data", "rate_limits.json")
AUTH_KEYS_JSON = os.path.join(_project_root, "data", "auth_keys.json")

SMITHERY_KEY = os.environ.get("SMITHERY_API_KEY", "")
CONFIG_B64 = base64.b64encode(json.dumps({"debug": False}).encode()).decode()

MAX_RETRIES = 3
FORWARD_TIMEOUT = 15.0
PROXY_TIMEOUT = 120.0

limiter = AdaptiveRateLimiter()
http_client: httpx.AsyncClient | None = None
streaming_client: httpx.AsyncClient | None = None

_servers_cache: list[dict] = []
_auth_keys: list[dict] = []


def _load_auth_keys():
    global _auth_keys
    if not os.path.exists(AUTH_KEYS_JSON):
        return
    with open(AUTH_KEYS_JSON) as f:
        data = json.load(f)
    _auth_keys = [k for k in data.get("keys", []) if not k.get("value", "").startswith("YOUR_")]
    configured = len(_auth_keys)
    total = len(data.get("keys", []))
    print(f"[gateway] Auth keys: {configured}/{total} configured (edit data/auth_keys.json to add more)", flush=True)


def _match_auth(hostname: str) -> dict | None:
    """Find an auth key entry matching this hostname."""
    for key_entry in _auth_keys:
        pattern = key_entry.get("pattern", "")
        if pattern.startswith("*."):
            if hostname.endswith(pattern[1:]):
                return key_entry
        elif hostname == pattern:
            return key_entry
    return None


def _build_target_url(url: str) -> str:
    if "server.smithery.ai" in url and SMITHERY_KEY and "api_key=" not in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}config={CONFIG_B64}&api_key={SMITHERY_KEY}"
    return url


async def _startup():
    global http_client, streaming_client, _servers_cache

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
        timeout=FORWARD_TIMEOUT,
    )
    streaming_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
        timeout=httpx.Timeout(PROXY_TIMEOUT, connect=10.0),
    )

    n = limiter.load_from_servers_json(SERVERS_JSON)
    print(f"[gateway] Loaded rate limits for {n} hosts from servers.json", flush=True)

    n2 = limiter.load(LEARNED_RATES)
    if n2:
        print(f"[gateway] Merged {n2} learned rates from rate_limits.json", flush=True)

    with open(SERVERS_JSON) as f:
        _servers_cache = json.load(f)

    _load_auth_keys()

    print("[gateway] Ready.", flush=True)


async def _shutdown():
    global http_client, streaming_client
    limiter.save(LEARNED_RATES)
    print(f"[gateway] Saved learned rates to {LEARNED_RATES}", flush=True)
    if http_client:
        await http_client.aclose()
    if streaming_client:
        await streaming_client.aclose()


async def mcp_proxy(request: Request) -> JSONResponse:
    """Forward an MCP JSON-RPC request to a target server, with rate limiting.

    Body: {"target": "https://mcp.example.com/mcp", "request": {jsonrpc payload}}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    target = body.get("target")
    mcp_request = body.get("request")

    if not target or not mcp_request:
        return JSONResponse(
            {"error": "Body must have 'target' (server URL) and 'request' (MCP JSON-RPC payload)"},
            status_code=400,
        )

    target_url = _build_target_url(target)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    from urllib.parse import urlparse
    hostname = urlparse(target).hostname or ""
    auth_entry = _match_auth(hostname)
    if auth_entry:
        headers[auth_entry["header"]] = auth_entry["value"]

    for attempt in range(MAX_RETRIES):
        await limiter.acquire(target)

        t0 = time.monotonic()
        try:
            resp = await http_client.post(
                target_url,
                json=mcp_request,
                headers=headers,
                follow_redirects=True,
            )
        except httpx.TimeoutException:
            return JSONResponse(
                {"error": "Target server timed out", "target": target},
                status_code=504,
            )
        except httpx.ConnectError as e:
            return JSONResponse(
                {"error": f"Cannot connect to target: {e}", "target": target},
                status_code=502,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        limiter.update_from_headers(target, dict(resp.headers), resp.status_code)

        if resp.status_code == 429:
            retry_after = limiter.handle_429(target)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(retry_after)
                continue
            return JSONResponse(
                {"error": "Rate limited by target server after retries", "target": target, "retry_after": retry_after},
                status_code=429,
            )

        try:
            resp_body = resp.json()
        except Exception:
            resp_body = resp.text

        return JSONResponse({
            "result": resp_body,
            "meta": {
                "target": target,
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
            },
        })

    return JSONResponse({"error": "Max retries exceeded"}, status_code=502)


def _decode_proxy_target(target_b64: str) -> str | None:
    """Decode a URL-safe base64 target URL from the proxy path."""
    try:
        padded = target_b64 + "=" * (-len(target_b64) % 4)
        return base64.urlsafe_b64decode(padded).decode()
    except Exception:
        return None


async def mcp_transparent_proxy(request: Request) -> Response:
    """Transparent MCP proxy -- acts as a standard MCP server endpoint.

    MCPServerStreamableHttp can connect to http://gateway/proxy/{b64_url}
    and the gateway decodes the target, applies rate limiting + auth, and
    forwards the request as-is.
    """
    target_b64 = request.path_params.get("target_b64", "")
    target = _decode_proxy_target(target_b64)
    if not target:
        return JSONResponse({"error": "Invalid base64-encoded target URL"}, status_code=400)

    target_url = _build_target_url(target)

    from urllib.parse import urlparse
    hostname = urlparse(target).hostname or ""

    # Build headers: forward most incoming headers, inject auth
    fwd_headers = {
        "Content-Type": request.headers.get("content-type", "application/json"),
        "Accept": request.headers.get("accept", "application/json, text/event-stream"),
    }
    for h in ("mcp-session-id", "last-event-id"):
        if h in request.headers:
            fwd_headers[h] = request.headers[h]

    auth_entry = _match_auth(hostname)
    if auth_entry:
        fwd_headers[auth_entry["header"]] = auth_entry["value"]

    body = await request.body()

    await limiter.acquire(target)

    t0 = time.monotonic()
    try:
        resp = await streaming_client.request(
            request.method,
            target_url,
            content=body if body else None,
            headers=fwd_headers,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return JSONResponse({"error": "Target server timed out", "target": target}, status_code=504)
    except httpx.ConnectError as e:
        return JSONResponse({"error": f"Cannot connect to target: {e}", "target": target}, status_code=502)

    latency_ms = int((time.monotonic() - t0) * 1000)
    limiter.update_from_headers(target, dict(resp.headers), resp.status_code)

    if resp.status_code == 429:
        retry_after = limiter.handle_429(target)
        return JSONResponse(
            {"error": "Rate limited by target", "target": target, "retry_after": retry_after},
            status_code=429,
        )

    # Build response headers to pass through
    resp_headers = {}
    for h in ("mcp-session-id", "content-type"):
        if h in resp.headers:
            resp_headers[h] = resp.headers[h]

    content_type = resp.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        async def _stream_sse():
            async for chunk in resp.aiter_bytes():
                yield chunk

        return StreamingResponse(
            _stream_sse(),
            status_code=resp.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
        )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


async def stats(request: Request) -> JSONResponse:
    return JSONResponse(limiter.get_stats())


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "hosts_tracked": len(limiter._buckets)})


async def servers(request: Request) -> JSONResponse:
    healthy = [
        {
            "name": s.get("name", ""),
            "url": s["url"],
            "health": s.get("health"),
            "rate_limit": s.get("rate_limit"),
            "tools": s.get("tools", []),
        }
        for s in _servers_cache
        if s.get("health") == "ok"
    ]
    return JSONResponse({"count": len(healthy), "servers": healthy})


app = Starlette(
    routes=[
        Route("/mcp", mcp_proxy, methods=["POST"]),
        Route("/proxy/{target_b64}", mcp_transparent_proxy, methods=["POST", "GET", "DELETE"]),
        Route("/stats", stats, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/servers", servers, methods=["GET"]),
    ],
    on_startup=[_startup],
    on_shutdown=[_shutdown],
)
