"""Normalize and deduplicate collected MCP server entries."""

from __future__ import annotations
from urllib.parse import urlparse, urlunparse


def normalize(entries: list[dict]) -> list[dict]:
    """Deduplicate by canonical URL, filter to https://, merge metadata from multiple sources."""
    seen: dict[str, dict] = {}

    for entry in entries:
        url = entry.get("url", "")
        if not url:
            continue

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            continue
        # skip localhost / private IPs
        host = parsed.hostname or ""
        if host in ("localhost", "127.0.0.1", "0.0.0.0") or host.startswith("192.168."):
            continue
        # skip template URLs and invalid ports
        if "{" in url or "{{" in url:
            continue
        raw_netloc = parsed.netloc
        if ":" in raw_netloc:
            port_str = raw_netloc.rsplit(":", 1)[-1]
            if not port_str.isdigit():
                continue

        canonical = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",  # params
            "",  # query
            "",  # fragment
        ))

        if canonical in seen:
            existing = seen[canonical]
            # merge: prefer entries with a name
            if not existing.get("name") and entry.get("name"):
                existing["name"] = entry["name"]
            # track all sources
            sources = set(existing.get("sources", [existing.get("source", "")]))
            sources.add(entry.get("source", ""))
            existing["sources"] = sorted(s for s in sources if s)
        else:
            entry_copy = dict(entry)
            entry_copy["url"] = canonical
            entry_copy["sources"] = [entry.get("source", "")]
            seen[canonical] = entry_copy

    result = list(seen.values())

    # clean up: remove singular 'source' key, keep 'sources' list
    for r in result:
        r.pop("source", None)
        r.pop("_status_code", None)

    # infer transport from URL if still unknown
    for r in result:
        if r.get("transport", "unknown") == "unknown":
            path = urlparse(r["url"]).path
            if path.endswith("/sse"):
                r["transport"] = "sse"
            elif path.endswith("/mcp"):
                r["transport"] = "streamable-http"

    print(f"[normalize] {len(entries)} raw → {len(result)} unique endpoints")
    return result
