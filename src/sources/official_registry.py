"""Technique 1 – API Polling: Official MCP Registry at registry.modelcontextprotocol.io"""

from __future__ import annotations
import httpx

BASE = "https://registry.modelcontextprotocol.io/v0.1/servers"


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    """Paginate the official registry and return entries that have remote URLs."""
    entries: list[dict] = []
    cursor: str | None = None

    while True:
        params: dict = {"limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            resp = await client.get(BASE, params=params, timeout=15)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"[official-registry] HTTP error: {exc}")
            break

        data = resp.json()
        servers = data.get("servers", data) if isinstance(data, dict) else data

        for item in servers:
            server = item.get("server", item) if isinstance(item, dict) else item
            if not isinstance(server, dict):
                continue

            remotes = server.get("remotes") or []
            name = server.get("title") or server.get("name") or ""

            for remote in remotes:
                url = remote.get("url", "")
                if not url.startswith("http"):
                    continue
                # skip self-hosted template URLs like https://{host}/mcp
                if "{" in url:
                    continue
                transport = remote.get("type", "unknown")
                entries.append({
                    "name": name,
                    "url": url,
                    "transport": transport,
                    "source": "official-registry",
                })

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        cursor = metadata.get("nextCursor")
        if not cursor:
            break

    print(f"[official-registry] collected {len(entries)} remote endpoints")
    return entries
