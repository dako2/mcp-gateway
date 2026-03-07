"""Technique 1 – API Polling: Smithery Registry at registry.smithery.ai (needs API key)."""

from __future__ import annotations
import os
import httpx

BASE = "https://registry.smithery.ai/servers"


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    api_key = os.environ.get("SMITHERY_API_KEY", "")
    if not api_key:
        print("[smithery] SMITHERY_API_KEY not set – skipping")
        return []

    headers = {"Authorization": f"Bearer {api_key}"}
    entries: list[dict] = []
    page = 1

    while True:
        params = {"q": "is:deployed", "pageSize": 100, "page": page}
        try:
            resp = await client.get(BASE, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"[smithery] HTTP error on page {page}: {exc}")
            break

        data = resp.json()
        servers = data.get("servers", [])
        if not servers:
            break

        for srv in servers:
            if not srv.get("isDeployed"):
                continue
            qualified = srv.get("qualifiedName", "")
            if not qualified:
                continue
            url = f"https://server.smithery.ai/{qualified}"
            name = srv.get("displayName") or qualified
            entries.append({
                "name": name,
                "url": url,
                "transport": "streamable-http",
                "source": "smithery",
            })

        pagination = data.get("pagination", {})
        total_pages = pagination.get("totalPages", page)
        if page >= total_pages:
            break
        page += 1

    print(f"[smithery] collected {len(entries)} deployed endpoints")
    return entries
