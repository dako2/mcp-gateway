"""Connect to all healthy servers and fetch full tool definitions via tools/list."""

from __future__ import annotations
import asyncio
import json
import os
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


def _parse_jsonrpc(text: str) -> dict | None:
    """Extract JSON-RPC result from plain JSON or SSE-wrapped response."""
    text = text.strip()
    if text.startswith("event:") or text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return None


async def _fetch_tools(
    client: httpx.AsyncClient,
    entry: dict,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Initialize + tools/list on a single server. Return full tool defs."""
    url = entry["url"]
    async with sem:
        try:
            # Step 1: initialize
            resp = await client.post(url, json=MCP_INITIALIZE, headers=HEADERS, timeout=10, follow_redirects=True)
            if resp.status_code >= 400:
                return None

            # Step 2: tools/list
            resp = await client.post(url, json=MCP_TOOLS_LIST, headers=HEADERS, timeout=10, follow_redirects=True)
            if resp.status_code != 200:
                return None

            parsed = _parse_jsonrpc(resp.text)
            if not parsed:
                return None

            result = parsed.get("result", {})
            if not isinstance(result, dict):
                return None

            tools = result.get("tools", [])
            if not isinstance(tools, list) or not tools:
                return None

            return {
                "server_name": entry.get("name", ""),
                "server_url": url,
                "transport": entry.get("transport", "unknown"),
                "sources": entry.get("sources", []),
                "tool_count": len(tools),
                "tools": tools,
            }

        except Exception:
            return None


async def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    servers_path = os.path.join(project_root, "data", "servers.json")

    with open(servers_path) as f:
        data = json.load(f)

    healthy = [e for e in data if e["health"] == "ok"]
    print(f"Querying {len(healthy)} healthy servers for full tool definitions...", flush=True)

    sem = asyncio.Semaphore(20)
    client = httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))

    try:
        tasks = [_fetch_tools(client, e, sem) for e in healthy]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.aclose()

    servers_with_tools = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            servers_with_tools.append(r)

    servers_with_tools.sort(key=lambda s: -s["tool_count"])

    total_tools = sum(s["tool_count"] for s in servers_with_tools)

    output_path = os.path.join(project_root, "data", "tools.json")
    with open(output_path, "w") as f:
        json.dump(servers_with_tools, f, indent=2, ensure_ascii=False)

    print(f"\nDone!", flush=True)
    print(f"  Servers with tools: {len(servers_with_tools)} / {len(healthy)}", flush=True)
    print(f"  Total tools: {total_tools}", flush=True)
    print(f"  Output: {output_path}", flush=True)

    print(f"\nTop 20 servers by tool count:", flush=True)
    for s in servers_with_tools[:20]:
        name = s["server_name"] or s["server_url"]
        tool_names = [t.get("name", "?") for t in s["tools"][:5]]
        more = f"... +{len(s['tools'])-5} more" if len(s["tools"]) > 5 else ""
        print(f"  {name}: {s['tool_count']} tools [{', '.join(tool_names)}{more}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
