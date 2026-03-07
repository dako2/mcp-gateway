"""Technique 2 – Markdown Parsing: jaw9c/awesome-remote-mcp-servers (~110 verified URLs)."""

from __future__ import annotations
import re
import httpx

RAW_URL = "https://raw.githubusercontent.com/jaw9c/awesome-remote-mcp-servers/main/README.md"


def _parse_table(md: str) -> list[dict]:
    """Extract rows from the markdown table that has columns: Name | Category | URL | Authentication | Maintainer."""
    entries: list[dict] = []
    in_table = False

    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break
            continue

        cols = [c.strip() for c in line.split("|")]
        # skip separator rows like |---|---|...|
        if all(set(c) <= {"-", ":"} for c in cols if c):
            in_table = True
            continue
        if not in_table:
            # header row
            if "URL" in line or "Name" in line:
                continue
            continue

        if len(cols) < 5:
            continue

        name = re.sub(r"[\[\]]", "", cols[1]).strip()
        category = cols[2].strip()
        url_raw = cols[3].strip()
        auth = cols[4].strip()

        url_match = re.search(r"`(https?://[^`]+)`", url_raw)
        if not url_match:
            url_match = re.search(r"(https?://\S+)", url_raw)
        if not url_match:
            continue

        url = url_match.group(1).rstrip("/")
        entries.append({
            "name": name,
            "url": url,
            "transport": "sse" if "/sse" in url else "streamable-http" if "/mcp" in url else "unknown",
            "auth": auth.lower().replace("oauth2.1", "oauth").replace(" ", ""),
            "source": "awesome-remote",
        })

    return entries


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    try:
        resp = await client.get(RAW_URL, timeout=15, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[awesome-remote] HTTP error: {exc}")
        return []

    entries = _parse_table(resp.text)
    print(f"[awesome-remote] collected {len(entries)} remote endpoints")
    return entries
