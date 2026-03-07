"""Technique 2 – Markdown Parsing: punkpeye + wong2 awesome-mcp-servers READMEs."""

from __future__ import annotations
import re
import httpx

READMES = [
    "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md",
    "https://raw.githubusercontent.com/wong2/awesome-mcp-servers/main/README.md",
]

# Patterns that suggest an actual remote MCP endpoint (not a GitHub repo)
ENDPOINT_HINTS = re.compile(
    r"https?://(?!github\.com|raw\.githubusercontent\.com|npmjs\.com|pypi\.org)"
    r"[^\s\)>\]\"'`]+(?:/mcp|/sse|/v\d|/api)",
    re.IGNORECASE,
)

GITHUB_LINK = re.compile(r"https?://github\.com/\S+")


def _extract_endpoints(md: str) -> list[dict]:
    entries: list[dict] = []
    seen: set[str] = set()

    for match in ENDPOINT_HINTS.finditer(md):
        url = match.group(0).rstrip("/.,;:)")
        if url in seen:
            continue
        seen.add(url)

        transport = "sse" if "/sse" in url else "streamable-http" if "/mcp" in url else "unknown"
        entries.append({
            "name": "",
            "url": url,
            "transport": transport,
            "source": "awesome-lists",
        })

    return entries


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    all_entries: list[dict] = []

    for readme_url in READMES:
        try:
            resp = await client.get(readme_url, timeout=15, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"[awesome-lists] error fetching {readme_url}: {exc}")
            continue

        entries = _extract_endpoints(resp.text)
        all_entries.extend(entries)

    print(f"[awesome-lists] collected {len(all_entries)} endpoint candidates")
    return all_entries
