"""Main crawler orchestrator – runs all sources, normalizes, health-checks, exports."""

from __future__ import annotations
import asyncio
import functools
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import httpx

print = functools.partial(print, flush=True)

from sources import (
    official_registry,
    smithery,
    awesome_remote,
    awesome_lists,
    mcpservers_org,
    mcp_so,
    github_search,
    dns_probe,
)
from normalize import normalize
from health_check import check_all
from export import export


async def collect_all() -> list[dict]:
    """Run all source fetchers concurrently and merge results."""
    async with httpx.AsyncClient(follow_redirects=True) as client:
        results = await asyncio.gather(
            official_registry.fetch(client),
            smithery.fetch(client),
            awesome_remote.fetch(client),
            awesome_lists.fetch(client),
            github_search.fetch(client),
            dns_probe.fetch(client),
            # HTML scrapers run sequentially within themselves but concurrently with others
            mcpservers_org.fetch(client),
            mcp_so.fetch(client),
            return_exceptions=True,
        )

    all_entries: list[dict] = []
    source_names = [
        "official_registry", "smithery", "awesome_remote", "awesome_lists",
        "github_search", "dns_probe", "mcpservers_org", "mcp_so",
    ]

    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            print(f"[crawl] {name} failed: {result}")
            continue
        all_entries.extend(result)

    return all_entries


async def main() -> None:
    print("=" * 60)
    print("MCP Gateway Crawler")
    print("=" * 60)

    # 1. Collect
    print("\n--- Phase 1: Collecting URLs from all sources ---")
    raw = await collect_all()
    print(f"\nTotal raw entries: {len(raw)}")

    # 2. Normalize & dedupe
    print("\n--- Phase 2: Normalizing and deduplicating ---")
    unique = normalize(raw)

    # 3. Health check
    print(f"\n--- Phase 3: Health-checking {len(unique)} endpoints ---")
    checked = await check_all(unique)

    # 4. Export
    print("\n--- Phase 4: Exporting ---")
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = export(checked, os.path.join(project_root, "data", "servers.json"))

    # Summary
    healthy = sum(1 for e in checked if e["health"] == "ok")
    auth_req = sum(1 for e in checked if e["health"] == "auth-required")
    total = len(checked)

    print("\n" + "=" * 60)
    print(f"DONE: {total} total endpoints")
    print(f"  - {healthy} healthy (ok)")
    print(f"  - {auth_req} auth-required (live but need credentials)")
    print(f"  - {total - healthy - auth_req} error/timeout")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
