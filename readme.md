# MCP Gateway

Crawl, catalog, and proxy 2,200+ remote MCP servers with smart rate limiting and Grok-powered trace generation.

## What's Inside

| Directory | Contents |
|-----------|----------|
| `data/servers.json` | 2,246 MCP endpoints (739 healthy, 1,507 auth-required) |
| `data/tools.json` | 267 servers with 3,356 tool definitions (full JSON schemas) |
| `mcp_servers/` | 267 Toucan-format server files with labels, categories, popularity |
| `src/` | Crawler, gateway, rate limiter, trace generator |
| `datagen/` | Toucan-based data generation pipeline (modified for Grok) |

## Quick Start

```bash
pip install -r requirements.txt
```

Add keys to `.env`:
```
SMITHERY_API_KEY=your-smithery-key
XAI_API_KEY=xai-your-grok-key
```

## Commands

### 1. Crawl MCP Servers

```bash
# Full crawl: 8 techniques, 10+ sources, health check all
cd src && python -u crawl.py

# Fetch full tool definitions from healthy servers
cd src && python -u list_tools.py

# Probe rate limits (adds rate_limit field to servers.json)
cd src && python -u probe_rate_limits.py

# Convert tools.json to Toucan format (for datagen pipeline)
cd src && python -u convert_to_toucan.py
```

### 2. Start the Gateway

```bash
cd src && uvicorn gateway:app --host 0.0.0.0 --port 8000
```

Endpoints:
```bash
# Health check
curl http://localhost:8000/health

# List healthy servers
curl http://localhost:8000/servers

# Proxy an MCP request (rate-limited per host)
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "target": "https://mcp.deepwiki.com/mcp",
    "request": {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
  }'

# View rate limiter stats
curl http://localhost:8000/stats
```

### 3. Generate Traces (with Grok)

```bash
# Direct trace generation (simple, fast)
cd src && python -u trace_gen.py --count 100 --concurrent 20

# High throughput
cd src && python -u trace_gen.py --count 5000 --concurrent 100

# Toucan pipeline (full quality control)
cd datagen

# Step 1.1: Generate question prompts from server tools
python step1.1_gen_questions.py --total_prompts 1000 --sampling_strategy random

# Step 1.2: LLM generates questions (using Grok)
bash run_with_grok.sh completion <file_from_step1.1>

# Step 1.3: Deduplicate and clean
python step1.3_process_completion.py --input_file <file_from_step1.2>

# Step 2.1-2.3: Question quality check
python step2.1_question_quality_check.py --input_file <*_3sanitized.jsonl>
bash run_with_grok.sh completion <file_from_step2.1>
python step2.3_process_completion.py --input_file <file_from_step2.2>

# Step 3.1-3.2: Agent trace generation (calls real MCP tools)
bash run_with_grok.sh trace_gen <*_2prepared.jsonl>
python step3.2_process_completion.py --input_file <file_from_step3.1>

# Step 4.1-4.3: Response quality check
python step4.1_response_quality_check.py --input_file <file_from_step3.2>
bash run_with_grok.sh completion <file_from_step4.1>
python step4.3_process_completion.py --input_file <file_from_step4.2>

# Multi-turn conversations
bash ext_completion_multi_turn.sh <input_file> grok-4-1-fast-reasoning xai 5
```

### 4. Configure Auth for Servers

Edit `data/auth_keys.json` to add API keys for auth-required servers:
```json
{
  "pattern": "mcp.notion.com",
  "header": "Authorization",
  "value": "Bearer ntn_your_actual_key"
}
```

## Crawling Sources

| Source | Technique | Yield |
|--------|-----------|-------|
| Official MCP Registry | API polling | 1,069 |
| Smithery Registry | API polling | 700 |
| mcpservers.org | HTML scraping | 159 |
| Toucan dataset | GitHub clone + health check | 126 |
| DNS endpoint probing | Pattern guessing | 100 |
| awesome-remote-mcp-servers | Markdown parsing | 94 |
| mcp.so | HTML scraping | 28 |
| GitHub API search | Code + repo search | 25 |
| awesome-mcp-servers lists | Markdown parsing | 23 |

## Gateway Features

- Per-host adaptive token bucket rate limiting
- QPS discovery from response headers (49 servers with known limits)
- Automatic 429 backoff learning (halves rate, exponential backoff)
- API key injection for auth-required servers (24 services pre-configured)
- Learned rates persisted to `data/rate_limits.json`

## Trace Generation

- Uses `grok-4-1-fast-reasoning` ($0.20/$0.50 per 1M tokens)
- Built-in xAI rate limiter (460 RPM)
- Tool calls routed through gateway (QPS-protected per MCP server)
- Supports single-turn, multi-turn, single-server, multi-server scenarios
- ~14,000 traces/hour theoretical throughput
