# MCP Gateway — Crawl Plan

Plan for crawling **public remote MCP servers** (no API keys or auth). Output: a JSON list of servers with health status.

---

## 1. Goals

- **Discover** public MCP server endpoints (HTTP/SSE/Streamable HTTP only; skip stdio-only).
- **No auth** — only servers that work without API keys or tokens (e.g. DeepWiki-style).
- **Health check** — test liveness/readiness and optionally list tools.
- **Export** — one or more JSON lists (e.g. `servers.json`) for the gateway.

---

## 2. Discovery Sources (where to get server URLs)

| Source | Type | Auth? | How to crawl |
|--------|------|-------|----------------|
| **GitHub awesome lists** | Markdown | No | Parse README for URLs (e.g. `wong2/awesome-mcp-servers`, `punkpeye/awesome-mcp-servers`) |
| **GitHub mcp-servers-hub** | Repo / markdown | No | Parse repo structure and README for server entries |
| **DeepWiki** | Single endpoint | No | Add known URL: `https://mcp.deepwiki.com/mcp` |
| **MCPList.ai** | Web directory | Unknown | Check for public API or sitemap; else optional scrape |
| **MCPServers.org / PulseMCP** | Web directory | Unknown | Same as above |
| **toolsdk-mcp-registry** | Registry/API | May need key | Prefer only if they expose a public list or API |

**Priority for v1:** GitHub awesome lists + DeepWiki + any other public JSON/markdown list. Add directory sites only if they offer a clear, legal way to get URLs (API, export, or documented scrape).

---

## 3. Crawl Pipeline (high level)

```
[Sources] → [URL extractor] → [Dedupe & filter] → [Health checker] → [JSON exporter]
```

### Step 1 — Collect URLs

- **GitHub**
  - Clone or fetch README (and maybe `servers.json` if present) from:
    - `wong2/awesome-mcp-servers`
    - `punkpeye/awesome-mcp-servers`
    - `apappascs/mcp-servers-hub`
  - Parse markdown for:
    - Links that look like MCP endpoints (e.g. `https://.../mcp`, `https://.../sse`, `mcp.` subdomains).
    - Code blocks or tables that list server URLs.
- **Curated list**
  - Maintain a small `sources/known-public.json` (or similar) with well-known public endpoints (e.g. DeepWiki).
- **Optional**
  - If a directory exposes an API or a single page with all links, add a fetcher for that URL and extract links.

### Step 2 — Normalize and filter

- **Normalize**
  - One canonical URL per server (e.g. prefer `https://.../mcp` over `/sse` when both exist).
  - Deduplicate by host + path.
- **Filter**
  - Keep only **remote** URLs: `http://` or `https://`.
  - Drop stdio-only entries (e.g. `npx`, `python -m`, or local paths).
  - Optionally drop entries that mention "API key" or "auth" in the same list entry (heuristic; improve over time).

### Step 3 — Health check

- **HTTP(S) endpoints**
  - If server has a documented health endpoint: `GET /health` or `GET /ready` (with timeout, e.g. 5s).
  - Else: send a minimal MCP JSON-RPC request (e.g. `initialize` or `tools/list`) and check for 2xx and valid JSON-RPC response.
- **Result**
  - For each URL store: `ok`, `error`, or `timeout` (and optionally `status_code`, `latency_ms`).
- **Rate limiting**
  - Throttle requests (e.g. 1–2 req/s per host) to avoid overloading servers.

### Step 4 — Export to JSON

- **Schema (minimal)**
  - `url` (string)
  - `name` or `id` (optional, from source)
  - `transport` (optional: `streamable_http`, `sse`, etc.)
  - `health` (e.g. `ok` | `error` | `timeout`)
  - `source` (e.g. `github:awesome-mcp-servers`, `curated`)
  - `last_checked` (ISO timestamp)
- **Optional**
  - `tools` (if we call `tools/list`) for downstream use.

---

## 4. Implementation order

1. **URL extraction**
   - Script or module that:
     - Reads `sources/known-public.json` (curated list including DeepWiki).
     - Fetches one or two awesome-list READMEs and parses URLs with a simple regex or markdown parser.
   - Output: raw list of candidate URLs (e.g. `candidates.txt` or `candidates.json`).

2. **Dedupe and filter**
   - Normalize URL (lowercase host, default path like `/mcp` if missing).
   - Remove duplicates; drop non-http(s) and obvious stdio-only entries.

3. **Health checker**
   - For each URL: HTTP GET to `/health` or MCP `initialize`/`tools/list`; record result and optional latency.
   - Throttle and timeout; retry once on timeout.

4. **JSON exporter**
   - Merge health results with metadata (source, name) and write `servers.json` (or `public-servers.json`).

5. **Agent/skills (later)**
   - Wrap the above in an agent or skill so Cursor (or another client) can “run crawl” or “refresh server list” on demand.

---

## 5. File layout (suggested)

```
mcp-gateway/
├── docs/
│   └── CRAWL_PLAN.md          # this file
├── sources/
│   └── known-public.json      # curated public endpoints (e.g. DeepWiki)
├── scripts/                   # or src/
│   ├── fetch-sources.ts       # fetch GitHub READMEs, known list
│   ├── extract-urls.ts        # parse markdown/JSON → URL list
│   ├── health-check.ts        # check each URL, output status
│   └── export-json.ts         # build final servers.json
├── data/
│   ├── candidates.json        # after extract, before health
│   └── servers.json           # final list with health
└── readme.md
```

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|-------------|
| Awesome lists list many stdio/local servers | Filter strictly to `http://` / `https://` and known path patterns |
| Directory sites block or rate-limit | Use only public APIs or documented methods; cache responses |
| Health checks overload small servers | Throttle (e.g. 1 req/s per host), short timeouts, optional backoff |
| False “public” (server later adds auth) | Re-check health periodically; mark `health: error` and keep in list with note |

---

## 7. Success criteria for “crawl v1”

- At least one source (e.g. `known-public.json` + one awesome list) is used.
- Output is a single JSON file with: `url`, `health`, `source`, `last_checked`.
- All listed URLs are remote (HTTP/HTTPS) and intended to be public (no API key in the crawl step).
- Health check runs without requiring any API keys or auth tokens.

Next step: implement **Step 1 (URL extraction)** with `sources/known-public.json` and one GitHub awesome list, then add **Steps 2–4** in order.

---

## 8. Path to 400 servers

To reach **400** public remote MCP servers, use these sources in order. Combine and dedupe, then filter to HTTP(S) only and run health checks; keep expanding until you have ≥400 entries (or 400 healthy).

| Source | Est. count | Auth? | How to get URLs | Notes |
|--------|------------|-------|------------------|--------|
| **Apify MCP Directory Scraper** | **28K+** | Apify free tier | Run actor, get JSON/CSV output; use “Deployment URL” or equivalent field | Aggregates Glama.ai, PulseMCP, mcp.so. Best single source for 400+. |
| **mcpservers.org** | **6,522** | No | Scrape `/all?page=N`, then each server detail page (e.g. `/servers/...`) for endpoint/try URL | List has 30 per page; detail pages often show remote URL (e.g. `mcp.alphavantage.co`). |
| **GitHub awesome lists** | Hundreds | No | Parse README from wong2/awesome-mcp-servers, punkpeye/awesome-mcp-servers for `https://` links | Many stdio/local; filter to `http://`/`https://` only. |
| **0x7c2f mcp-servers.json** | ~40 | No | `GET https://0x7c2f.github.io/api/mcp-servers.json` | Gives **repos** (GitHub), not endpoints; use for metadata or to cross-check names. |
| **Curated list** | Small | No | Maintain `sources/known-public.json` (e.g. DeepWiki) | Guaranteed public, no-auth endpoints. |

**Recommended order to reach 400:**

1. **Primary:** Use **Apify MCP Directory Scraper** (e.g. `lovely_sequoia/mcp-directory-scraper`). Run once, export JSON. Filter rows that have a deployment/remote URL; normalize to `http://`/`https://`; dedupe; run health check; take first 400 that pass (or 400 by source priority). If Apify is not an option, skip to (2).
2. **Secondary:** Scrape **mcpservers.org**:  
   - Fetch list pages `all?page=1` … `page=ceil(400/30)` (e.g. 14 pages for 400).  
   - For each server card, open the detail page URL, parse HTML for “endpoint”, “try it”, “URL”, or similar; collect `https?://` URLs.  
   - Dedupe and filter to remote only; health check; add to pool until ≥400 candidates (or 400 healthy).
3. **Tertiary:** Add **GitHub awesome lists**: fetch READMEs, extract all `https://` links that look like MCP (e.g. contain `mcp`, `sse`, or known host patterns). Dedupe with existing pool; health check; fill remaining slots toward 400.
4. **Always:** Merge in **curated** list (e.g. DeepWiki) and **0x7c2f** for extra metadata (name, repo) where the URL matches.

**Target output:** `data/servers.json` (or `public-servers.json`) with ≥400 entries: `url`, `health`, `source`, `last_checked`, and optionally `name`, `transport`, `tools`. Prefer “healthy” servers first when trimming to exactly 400.
