"""Health check each MCP endpoint by sending an MCP initialize request."""

from __future__ import annotations
import asyncio
import json
import time
import httpx

MCP_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "mcp-gateway-crawler", "version": "0.1.0"},
    },
}

MCP_TOOLS_LIST = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {},
}

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


async def _check_one(
    client: httpx.AsyncClient,
    entry: dict,
    semaphore: asyncio.Semaphore,
    timeout: float = 8.0,
) -> dict:
    """Send MCP initialize to the endpoint; record health + latency."""
    url = entry["url"]
    result = dict(entry)
    result["health"] = "error"
    result["latency_ms"] = None
    result["tools"] = []

    async with semaphore:
        try:
            httpx.URL(url)
        except Exception:
            return result

        t0 = time.monotonic()
        try:
            resp = await client.post(
                url,
                json=MCP_INITIALIZE,
                headers=HEADERS,
                timeout=timeout,
                follow_redirects=True,
            )
            latency = int((time.monotonic() - t0) * 1000)
            result["latency_ms"] = latency

            if resp.status_code < 500:
                body = resp.text.strip()
                if "jsonrpc" in body or "result" in body or "event:" in body:
                    result["health"] = "ok"
                elif resp.status_code in (200, 201, 202, 204):
                    result["health"] = "ok"
                elif resp.status_code in (401, 403):
                    result["health"] = "auth-required"
                elif resp.status_code == 405:
                    try:
                        resp2 = await client.get(url, headers=HEADERS, timeout=timeout, follow_redirects=True)
                        if resp2.status_code < 500:
                            result["health"] = "ok" if resp2.status_code == 200 else "auth-required"
                    except Exception:
                        pass

        except httpx.TimeoutException:
            result["health"] = "timeout"
            result["latency_ms"] = int(timeout * 1000)
        except Exception:
            result["health"] = "error"

        if result["health"] == "ok":
            try:
                resp = await client.post(
                    url,
                    json=MCP_TOOLS_LIST,
                    headers=HEADERS,
                    timeout=5,
                    follow_redirects=True,
                )
                if resp.status_code == 200:
                    body = resp.text.strip()
                    if body.startswith("event:") or body.startswith("data:"):
                        for line in body.splitlines():
                            if line.startswith("data:"):
                                body = line[5:].strip()
                                break
                    try:
                        data = json.loads(body)
                        tools_result = data.get("result", {}) if isinstance(data, dict) else {}
                        tools = tools_result.get("tools", []) if isinstance(tools_result, dict) else []
                        result["tools"] = [t.get("name", "") for t in tools if isinstance(t, dict) and t.get("name")]
                    except (json.JSONDecodeError, AttributeError):
                        pass
            except Exception:
                pass

    return result


async def check_all(
    entries: list[dict],
    max_concurrent: int = 20,
    timeout: float = 8.0,
) -> list[dict]:
    """Run health checks on all entries concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    client = httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))

    try:
        tasks = [_check_one(client, e, semaphore, timeout) for e in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.aclose()

    cleaned: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            fallback = dict(entries[i])
            fallback["health"] = "error"
            fallback["latency_ms"] = None
            fallback["tools"] = []
            cleaned.append(fallback)
        else:
            cleaned.append(r)

    order = {"ok": 0, "auth-required": 1, "timeout": 2, "error": 3}
    cleaned.sort(key=lambda r: (order.get(r["health"], 9), -(r.get("latency_ms") or 99999)))

    healthy = sum(1 for r in cleaned if r["health"] == "ok")
    auth_req = sum(1 for r in cleaned if r["health"] == "auth-required")
    print(f"[health-check] done: {healthy} ok, {auth_req} auth-required, {len(cleaned) - healthy - auth_req} error/timeout", flush=True)
    return cleaned
