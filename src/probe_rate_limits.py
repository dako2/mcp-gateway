"""Phase 1: Probe all healthy servers to discover and catalog rate limits."""

from __future__ import annotations
import asyncio
import json
import os
import time
import base64
import functools

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import httpx
from qps_discovery import parse_rate_limit_headers

print = functools.partial(print, flush=True)

MCP_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "mcp-gateway-prober", "version": "0.1.0"},
    },
}

HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

SMITHERY_KEY = os.environ.get("SMITHERY_API_KEY", "")
CONFIG_B64 = base64.b64encode(json.dumps({"debug": False}).encode()).decode()


def _build_url(entry: dict) -> str:
    url = entry["url"]
    if "server.smithery.ai" in url and SMITHERY_KEY and "api_key=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}config={CONFIG_B64}&api_key={SMITHERY_KEY}"
    return url


async def _probe_one(
    client: httpx.AsyncClient,
    entry: dict,
    sem: asyncio.Semaphore,
) -> dict:
    url = _build_url(entry)

    async with sem:
        try:
            resp = await client.post(
                url,
                json=MCP_INITIALIZE,
                headers=HEADERS,
                timeout=10,
                follow_redirects=True,
            )
            resp_headers = dict(resp.headers)
            info = parse_rate_limit_headers(resp_headers, resp.status_code)

            return {
                "qps": info.qps,
                "burst": info.burst,
                "window_seconds": info.window_seconds,
                "source": info.source,
                "raw_headers": info.raw_headers if info.raw_headers else None,
                "retry_after": info.retry_after,
                "status_code": resp.status_code,
            }

        except httpx.TimeoutException:
            return {"qps": 1.0, "burst": 3, "source": "timeout-default", "status_code": None}
        except Exception:
            return {"qps": 2.0, "burst": 5, "source": "error-default", "status_code": None}


async def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    servers_path = os.path.join(project_root, "data", "servers.json")

    with open(servers_path) as f:
        all_servers = json.load(f)

    healthy = [(i, e) for i, e in enumerate(all_servers) if e.get("health") == "ok"]
    print(f"Probing {len(healthy)} healthy servers for rate limits...")

    sem = asyncio.Semaphore(20)
    client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))

    try:
        tasks = [_probe_one(client, entry, sem) for _, entry in healthy]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.aclose()

    from collections import Counter
    source_counts = Counter()

    for (idx, entry), result in zip(healthy, results):
        if isinstance(result, Exception):
            rl = {"qps": 2.0, "burst": 5, "source": "error-default"}
        else:
            rl = result

        clean = {
            "qps": rl.get("qps") or 2.0,
            "burst": rl.get("burst") or 5,
            "source": rl.get("source", "default"),
        }
        if rl.get("window_seconds"):
            clean["window_seconds"] = rl["window_seconds"]
        if rl.get("raw_headers"):
            clean["raw_headers"] = rl["raw_headers"]
        if rl.get("retry_after"):
            clean["retry_after"] = rl["retry_after"]

        all_servers[idx]["rate_limit"] = clean
        source_counts[clean["source"]] += 1

    # Set defaults for non-healthy servers
    for e in all_servers:
        if "rate_limit" not in e:
            e["rate_limit"] = {"qps": 2.0, "burst": 5, "source": "not-probed"}

    with open(servers_path, "w") as f:
        json.dump(all_servers, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Updated {len(healthy)} entries in servers.json")
    print(f"\nRate limit sources:")
    for src, count in source_counts.most_common():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
