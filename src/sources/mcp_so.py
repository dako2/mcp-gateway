"""Technique 3 – HTML Scraping: mcp.so (18,000+ servers)."""

from __future__ import annotations
import asyncio
import re
import httpx
from bs4 import BeautifulSoup

BASE = "https://mcp.so"
MAX_PAGES = 15


async def _fetch_list_page(client: httpx.AsyncClient, page: int) -> list[str]:
    url = f"{BASE}/servers?page={page}"
    try:
        resp = await client.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    paths: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/server/" in href or "/servers/" in href:
            if href.startswith("/"):
                paths.append(href)
            elif href.startswith(BASE):
                paths.append(href.replace(BASE, ""))
    return list(set(paths))


def _extract_urls(html: str) -> list[str]:
    urls: list[str] = []
    for m in re.finditer(r'https?://[^\s"\'<>\)]+(?:/mcp|/sse|/v\d|/api)[^\s"\'<>\)]*', html):
        url = m.group(0).rstrip("/.,;:)")
        if "github.com" not in url and "npmjs.com" not in url and "mcp.so" not in url:
            urls.append(url)
    return urls


async def _fetch_detail(client: httpx.AsyncClient, path: str, sem: asyncio.Semaphore) -> list[str]:
    async with sem:
        try:
            resp = await client.get(f"{BASE}{path}", timeout=10, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        return _extract_urls(resp.text)


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    all_paths: list[str] = []

    for batch_start in range(1, MAX_PAGES + 1, 5):
        batch = range(batch_start, min(batch_start + 5, MAX_PAGES + 1))
        results = await asyncio.gather(*[_fetch_list_page(client, p) for p in batch])
        for paths in results:
            all_paths.extend(paths)
        if any(not r for r in results):
            break

    all_paths = list(set(all_paths))
    print(f"[mcp.so] found {len(all_paths)} detail pages, scraping...", flush=True)

    sem = asyncio.Semaphore(10)
    detail_results = await asyncio.gather(*[_fetch_detail(client, p, sem) for p in all_paths])

    entries: list[dict] = []
    seen: set[str] = set()
    for urls in detail_results:
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            transport = "sse" if "/sse" in url else "streamable-http" if "/mcp" in url else "unknown"
            entries.append({"name": "", "url": url, "transport": transport, "source": "mcp.so"})

    print(f"[mcp.so] collected {len(entries)} endpoint candidates", flush=True)
    return entries
