"""Technique 4 – GitHub API Search: find server.json with remotes + mcp-server repos."""

from __future__ import annotations
import os
import re
import httpx

GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    h: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def _search_code(client: httpx.AsyncClient) -> list[dict]:
    """Search GitHub for server.json files containing 'remotes' and 'streamable-http' or 'sse'."""
    entries: list[dict] = []

    for query in [
        "remotes streamable-http filename:server.json",
        "remotes sse filename:server.json",
    ]:
        params = {"q": query, "per_page": 100}
        try:
            resp = await client.get(
                f"{GITHUB_API}/search/code",
                params=params,
                headers=_headers(),
                timeout=15,
            )
            if resp.status_code == 403:
                print("[github-search] rate-limited, try setting GITHUB_TOKEN")
                break
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"[github-search] code search error: {exc}")
            break

        items = resp.json().get("items", [])
        for item in items:
            raw_url = (
                item.get("html_url", "")
                .replace("github.com", "raw.githubusercontent.com")
                .replace("/blob/", "/")
            )
            if not raw_url:
                continue

            try:
                file_resp = await client.get(raw_url, timeout=10, follow_redirects=True)
                file_resp.raise_for_status()
                data = file_resp.json()
            except Exception:
                continue

            remotes = data.get("remotes", [])
            name = data.get("title") or data.get("name") or ""
            for remote in remotes:
                url = remote.get("url", "")
                if url.startswith("http") and "{" not in url:
                    entries.append({
                        "name": name,
                        "url": url,
                        "transport": remote.get("type", "unknown"),
                        "source": "github-search",
                    })

    return entries


async def _search_repos(client: httpx.AsyncClient) -> list[dict]:
    """Search repos with topic mcp-server, then scan READMEs for endpoint URLs."""
    entries: list[dict] = []
    params = {"q": "topic:mcp-server", "sort": "stars", "per_page": 100}

    try:
        resp = await client.get(
            f"{GITHUB_API}/search/repositories",
            params=params,
            headers=_headers(),
            timeout=15,
        )
        if resp.status_code == 403:
            print("[github-search] rate-limited on repo search")
            return entries
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"[github-search] repo search error: {exc}")
        return entries

    repos = resp.json().get("items", [])
    endpoint_re = re.compile(
        r"https?://(?!github\.com|raw\.githubusercontent)[^\s\"'<>\)]+(?:/mcp|/sse)[^\s\"'<>\)]*"
    )

    for repo in repos[:50]:
        full_name = repo.get("full_name", "")
        default_branch = repo.get("default_branch", "main")
        readme_url = f"https://raw.githubusercontent.com/{full_name}/{default_branch}/README.md"

        try:
            r = await client.get(readme_url, timeout=8, follow_redirects=True)
            if r.status_code != 200:
                continue
        except httpx.HTTPError:
            continue

        for m in endpoint_re.finditer(r.text):
            url = m.group(0).rstrip("/.,;:)")
            if "{" in url:
                continue
            entries.append({
                "name": full_name.split("/")[-1].replace("-", " ").title(),
                "url": url,
                "transport": "sse" if "/sse" in url else "streamable-http",
                "source": "github-search",
            })

    return entries


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    code_entries = await _search_code(client)
    repo_entries = await _search_repos(client)
    all_entries = code_entries + repo_entries

    seen: set[str] = set()
    deduped = []
    for e in all_entries:
        if e["url"] not in seen:
            seen.add(e["url"])
            deduped.append(e)

    print(f"[github-search] collected {len(deduped)} endpoints (code: {len(code_entries)}, repos: {len(repo_entries)})")
    return deduped
