"""Technique 3 – HTML Scraping: mcpservers.org (6,500+ servers)."""

from __future__ import annotations
import asyncio
import re
import httpx
from bs4 import BeautifulSoup

BASE = "https://mcpservers.org"
MAX_PAGES = 15  # 30 per page → 450 candidates


async def _fetch_list_page(client: httpx.AsyncClient, page: int) -> list[str]:
    """Return detail-page paths from a single list page."""
    url = f"{BASE}/all?page={page}&sort=newest"
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    paths: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/servers/"):
            paths.append(href)
    return list(set(paths))


def _extract_urls_from_detail(html: str) -> list[str]:
    urls: list[str] = []
    for match in re.finditer(r'https?://[^\s"\'<>\)]+(?:/mcp|/sse|/v\d|/api)[^\s"\'<>\)]*', html):
        url = match.group(0).rstrip("/.,;:)")
        if "github.com" not in url and "npmjs.com" not in url:
            urls.append(url)
    return urls


async def _fetch_detail(client: httpx.AsyncClient, path: str, sem: asyncio.Semaphore) -> list[tuple[str, str]]:
    async with sem:
        try:
            resp = await client.get(f"{BASE}{path}", timeout=10, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        name = path.split("/")[-1].replace("-", " ").title() if "/" in path else ""
        return [(url, name) for url in _extract_urls_from_detail(resp.text)]


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    all_detail_paths: list[str] = []

    # Fetch list pages concurrently in batches of 5
    for batch_start in range(1, MAX_PAGES + 1, 5):
        batch = range(batch_start, min(batch_start + 5, MAX_PAGES + 1))
        results = await asyncio.gather(*[_fetch_list_page(client, p) for p in batch])
        for paths in results:
            all_detail_paths.extend(paths)
        if any(not r for r in results):
            break

    all_detail_paths = list(set(all_detail_paths))
    print(f"[mcpservers.org] found {len(all_detail_paths)} detail pages, scraping...", flush=True)

    sem = asyncio.Semaphore(10)
    detail_results = await asyncio.gather(*[_fetch_detail(client, p, sem) for p in all_detail_paths])

    entries: list[dict] = []
    seen: set[str] = set()
    for detail_urls in detail_results:
        for url, name in detail_urls:
            if url in seen:
                continue
            seen.add(url)
            transport = "sse" if "/sse" in url else "streamable-http" if "/mcp" in url else "unknown"
            entries.append({"name": name, "url": url, "transport": transport, "source": "mcpservers.org"})

    print(f"[mcpservers.org] collected {len(entries)} endpoint candidates", flush=True)
    return entries
