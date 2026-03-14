"""Cross-reference mcp_servers/*.json with data/servers.json to build a URL map.

For each Toucan server file, resolves the best direct URL (bypassing Smithery
where possible) and records health status from the crawled server list.

Usage:
    cd src && python resolve_urls.py
    cd src && python resolve_urls.py --output ../data/server_url_map.json

Output: JSON mapping  server_id -> {url, health, source, server_name, ...}
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from urllib.parse import urlparse

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MCP_SERVERS_DIR = os.path.join(_project_root, "mcp_servers")
SERVERS_JSON = os.path.join(_project_root, "data", "servers.json")
DEFAULT_OUTPUT = os.path.join(_project_root, "data", "server_url_map.json")


def _normalize_url(url: str) -> str:
    """Strip trailing slashes and lowercase the host for comparison."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{host}{path}"


def _load_crawled_servers(path: str) -> dict[str, dict]:
    """Load data/servers.json and index by normalized URL."""
    if not os.path.exists(path):
        print(f"[warn] {path} not found, skipping crawled server data")
        return {}
    with open(path) as f:
        servers = json.load(f)
    index = {}
    for s in servers:
        url = s.get("url", "")
        if url:
            index[_normalize_url(url)] = s
    return index


def resolve(mcp_servers_dir: str = MCP_SERVERS_DIR,
            servers_json: str = SERVERS_JSON) -> list[dict]:
    crawled = _load_crawled_servers(servers_json)
    results = []

    files = sorted(glob.glob(os.path.join(mcp_servers_dir, "*.json")))
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        labels = data.get("labels", {})
        server_id = meta.get("server_id")
        server_name = meta.get("server_name", "")
        fname = os.path.basename(fpath)

        remote_resp = meta.get("remote_server_response", {})
        crawled_info = meta.get("server_info_crawled", {})

        # Candidate URLs (prefer direct over Smithery)
        direct_url = remote_resp.get("url", "")
        sdk_url = crawled_info.get("python_sdk_url", "")

        is_smithery_sdk = "server.smithery.ai" in sdk_url
        is_connected = labels.get("is_connected", False)
        is_valid = labels.get("is_remote_tool_valid", False)

        # Pick best URL: direct > non-Smithery sdk > Smithery sdk
        chosen_url = ""
        source = ""
        if direct_url and "smithery" not in direct_url.lower():
            chosen_url = direct_url
            source = "direct"
        elif sdk_url and not is_smithery_sdk:
            chosen_url = sdk_url
            source = "sdk_direct"
        elif direct_url:
            chosen_url = direct_url
            source = "direct_smithery"
        elif sdk_url:
            chosen_url = sdk_url
            source = "smithery"
        else:
            source = "none"

        # Cross-reference with crawled health
        health = "unknown"
        crawled_entry = None
        if chosen_url:
            norm = _normalize_url(chosen_url)
            crawled_entry = crawled.get(norm)
            if crawled_entry:
                health = crawled_entry.get("health", "unknown")

        tool_count = remote_resp.get("tool_count", 0)
        tool_names = remote_resp.get("tool_names", [])

        results.append({
            "server_id": server_id,
            "server_name": server_name,
            "file": fname,
            "url": chosen_url,
            "source": source,
            "health": health,
            "is_connected": is_connected,
            "is_valid": is_valid,
            "tool_count": tool_count,
            "tool_names": tool_names,
            "is_smithery_only": source in ("smithery", "direct_smithery"),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Resolve MCP server URLs")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mcp_servers_dir", default=MCP_SERVERS_DIR)
    parser.add_argument("--servers_json", default=SERVERS_JSON)
    args = parser.parse_args()

    results = resolve(args.mcp_servers_dir, args.servers_json)

    # Stats
    total = len(results)
    direct = sum(1 for r in results if r["source"] in ("direct", "sdk_direct"))
    smithery = sum(1 for r in results if r["is_smithery_only"])
    no_url = sum(1 for r in results if r["source"] == "none")
    healthy = sum(1 for r in results if r["health"] == "ok")
    connected = sum(1 for r in results if r["is_connected"])
    valid = sum(1 for r in results if r["is_valid"])

    print(f"Total servers:    {total}")
    print(f"Direct URLs:      {direct}")
    print(f"Smithery-only:    {smithery}")
    print(f"No URL:           {no_url}")
    print(f"Health OK:        {healthy}")
    print(f"Connected:        {connected}")
    print(f"Valid tools:      {valid}")

    # Write indexed by server_id for fast lookup
    url_map = {}
    for r in results:
        sid = r["server_id"]
        if sid is not None:
            url_map[str(sid)] = r

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(url_map, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(url_map)} entries to {args.output}")


if __name__ == "__main__":
    main()
