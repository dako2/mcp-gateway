"""Generate Toucan-style multi-turn tool-use traces using xAI Grok + real MCP servers.

Usage:
    # Make sure gateway is running first:
    #   cd src && uvicorn gateway:app --port 8000
    # Then:
    cd src && python -u trace_gen.py --count 100 --output ../data/traces.jsonl
    cd src && python -u trace_gen.py --count 1000 --concurrent 50 --output ../data/traces.jsonl

    # High-throughput: 1000 concurrent (xAI RPM = 480, so ~8 req/s sustained)
    cd src && python -u trace_gen.py --count 5000 --concurrent 100 --output ../data/traces.jsonl

Env vars:
    XAI_API_KEY     -- xAI API key (required)
    GATEWAY_URL     -- gateway URL (default: http://127.0.0.1:8000)
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import time
import functools
from datetime import datetime, timezone

from dotenv import load_dotenv

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env"))

import httpx

print = functools.partial(print, flush=True)

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")
MODEL = "grok-4-1-fast-reasoning"
MAX_TURNS = 5
MAX_TOOLS_PER_SERVER = 15

# xAI rate limits: 480 RPM for grok-4-1-fast-reasoning
# We use a token bucket to stay under this
XAI_RPM = 460  # leave 20 RPM headroom
XAI_TOKENS_PER_SEC = XAI_RPM / 60.0  # ~7.67 req/s

TASK_PROMPT_SYSTEM = """You are a task generator for testing MCP (Model Context Protocol) tool servers.
Given a list of available tools with their descriptions and input schemas, generate a realistic user task
that would require using 1-3 of these tools. The task should be specific and actionable.

Respond with ONLY a JSON object:
{"task": "the user task description", "tools_to_use": ["tool1", "tool2"]}"""

AGENT_SYSTEM = """You are an AI assistant with access to MCP tools. Complete the user's task by calling
the appropriate tools. Be concise. When you have enough information, summarize the result."""


def _load_servers() -> list[dict]:
    tools_path = os.path.join(_project_root, "data", "tools.json")
    with open(tools_path) as f:
        servers = json.load(f)
    usable = [s for s in servers if s.get("tool_count", 0) > 0 and s.get("tools")]
    return usable


def _server_to_openai_tools(server: dict) -> list[dict]:
    """Convert MCP tool schemas to OpenAI function-calling format."""
    tools = server.get("tools", [])[:MAX_TOOLS_PER_SERVER]
    openai_tools = []
    for t in tools:
        if not t.get("name"):
            continue
        schema = t.get("inputSchema") or t.get("input_schema") or {"type": "object", "properties": {}}
        fn = {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t.get("description") or "")[:500],
                "parameters": schema,
            },
        }
        openai_tools.append(fn)
    return openai_tools


class _XaiRateLimiter:
    """Simple token bucket to stay under xAI's 480 RPM."""

    def __init__(self, rpm: float = XAI_RPM):
        self._rate = rpm / 60.0
        self._tokens = min(rpm / 60.0 * 5, 30)  # small burst buffer
        self._capacity = self._tokens
        self._last = time.monotonic()
        self._lock = asyncio.Lock()
        self._total_calls = 0
        self._start_time = time.monotonic()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
            self._last = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0
            self._total_calls += 1

    @property
    def effective_rps(self) -> float:
        elapsed = time.monotonic() - self._start_time
        return self._total_calls / max(elapsed, 0.1)


_xai_limiter = _XaiRateLimiter()


async def _call_grok(
    client: httpx.AsyncClient,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
) -> dict:
    """Call xAI Grok API (OpenAI-compatible) with rate limiting."""
    await _xai_limiter.acquire()

    body: dict = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    for attempt in range(5):
        try:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=120,
            )
            if resp.status_code == 429:
                body_text = resp.text[:200]
                if "credits" in body_text or "spending limit" in body_text:
                    raise Exception(f"xAI account out of credits: {body_text}")
                retry_after = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                print(f"  [xai] 429 rate limited, waiting {retry_after:.0f}s (attempt {attempt+1}/5)")
                await asyncio.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            if attempt < 4:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    raise Exception("xAI API: max retries exceeded")


async def _call_tool_via_gateway(
    client: httpx.AsyncClient,
    server_url: str,
    tool_name: str,
    arguments: dict,
) -> dict:
    """Call an MCP tool through the gateway."""
    mcp_request = {
        "jsonrpc": "2.0",
        "id": random.randint(1, 99999),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    try:
        resp = await client.post(
            f"{GATEWAY_URL}/mcp",
            json={"target": server_url, "request": mcp_request},
            timeout=20,
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


async def _generate_task(client: httpx.AsyncClient, server: dict) -> dict | None:
    """Use Grok to generate a realistic task for a server's tools."""
    tools_summary = []
    for t in server["tools"][:MAX_TOOLS_PER_SERVER]:
        schema = t.get("inputSchema") or t.get("input_schema") or {}
        tools_summary.append({
            "name": t.get("name", ""),
            "description": (t.get("description") or "")[:200],
            "params": list((schema.get("properties") or {}).keys()),
        })

    messages = [
        {"role": "system", "content": TASK_PROMPT_SYSTEM},
        {"role": "user", "content": f"Server: {server.get('server_name', 'Unknown')}\n\nTools:\n{json.dumps(tools_summary, indent=2)}"},
    ]

    try:
        resp = await _call_grok(client, messages, temperature=0.9)
        content = resp["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        print(f"  [task-gen] error: {e}")
        return None


async def _run_trace(
    client: httpx.AsyncClient,
    server: dict,
    task: str,
) -> dict:
    """Run a multi-turn agent trace: Grok decides which tools to call, we execute via gateway."""
    openai_tools = _server_to_openai_tools(server)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": task},
    ]
    trace_steps = []
    tool_calls_made = 0

    for turn in range(MAX_TURNS):
        try:
            resp = await _call_grok(client, messages, tools=openai_tools)
        except Exception as e:
            trace_steps.append({"turn": turn, "error": f"grok error: {e}"})
            break

        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "")

        reasoning = msg.get("reasoning_content", "")

        if msg.get("tool_calls"):
            messages.append(msg)

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_name = fn["name"]
                try:
                    arguments = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    arguments = {}

                tool_result = await _call_tool_via_gateway(
                    client, server["server_url"], tool_name, arguments,
                )

                result_str = json.dumps(tool_result)[:2000]

                trace_steps.append({
                    "turn": turn,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": tool_result,
                    "reasoning": reasoning[:500] if reasoning else None,
                })
                tool_calls_made += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        elif msg.get("content"):
            trace_steps.append({
                "turn": turn,
                "type": "assistant_message",
                "content": msg["content"],
                "reasoning": reasoning[:500] if reasoning else None,
            })
            if finish == "stop":
                break
        else:
            break

    usage = resp.get("usage", {}) if "resp" in dir() else {}

    return {
        "server_name": server.get("server_name", ""),
        "server_url": server["server_url"],
        "task": task,
        "tool_count": len(openai_tools),
        "tools_available": [t["function"]["name"] for t in openai_tools],
        "tool_calls_made": tool_calls_made,
        "turns": len(trace_steps),
        "steps": trace_steps,
        "model": MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "usage": usage,
    }


async def main():
    parser = argparse.ArgumentParser(description="Generate tool-use traces with Grok + MCP servers")
    parser.add_argument("--count", type=int, default=10, help="Number of traces to generate")
    parser.add_argument("--output", type=str, default=os.path.join(_project_root, "data", "traces.jsonl"))
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent trace generations")
    args = parser.parse_args()

    if not XAI_API_KEY:
        print("ERROR: Set XAI_API_KEY in .env")
        return

    servers = _load_servers()
    print(f"Loaded {len(servers)} servers with tools")
    print(f"Generating {args.count} traces using {MODEL}")
    print(f"Output: {args.output}\n")

    # Check gateway is running
    async with httpx.AsyncClient() as check:
        try:
            r = await check.get(f"{GATEWAY_URL}/health", timeout=3)
            print(f"Gateway: {r.json()}")
        except Exception:
            print(f"ERROR: Gateway not running at {GATEWAY_URL}")
            print("Start it first: cd src && uvicorn gateway:app --port 8000")
            return

    # Spread across diverse servers, weighted by tool count
    weights = [s["tool_count"] for s in servers]
    selected = random.choices(servers, weights=weights, k=args.count)
    sem = asyncio.Semaphore(args.concurrent)

    # Shared httpx client for all Grok calls (connection pooling)
    grok_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrent + 10, max_keepalive_connections=args.concurrent),
        timeout=120,
    )

    completed = 0
    failed = 0
    t_start = time.monotonic()

    async def _gen_one(idx: int, server: dict) -> dict | None:
        async with sem:
            name = server.get('server_name', '?')[:40]
            print(f"[{idx+1}/{args.count}] {name} ({server['tool_count']} tools)")

            task_data = await _generate_task(grok_client, server)
            if not task_data:
                print(f"  skipped (task generation failed)")
                return None

            task = task_data.get("task", "")
            print(f"  task: {task[:80]}")

            trace = await _run_trace(grok_client, server, task)
            print(f"  done: {trace['tool_calls_made']} tool calls, {trace['turns']} turns")
            return trace

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        with open(args.output, "w") as f:
            tasks = [_gen_one(i, s) for i, s in enumerate(selected)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    print(f"  error: {r}")
                    failed += 1
                elif r is not None:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    completed += 1
                else:
                    failed += 1
    finally:
        await grok_client.aclose()

    elapsed = time.monotonic() - t_start

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Traces: {completed} saved, {failed} failed")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Throughput: {completed/max(elapsed,1):.2f} traces/s")
    print(f"  xAI effective RPS: {_xai_limiter.effective_rps:.2f}")
    print(f"  Output: {args.output}")

    if completed > 0:
        total_calls = 0
        total_turns = 0
        with open(args.output) as f:
            for line in f:
                t = json.loads(line)
                total_calls += t.get("tool_calls_made", 0)
                total_turns += t.get("turns", 0)
        print(f"  Total tool calls: {total_calls}")
        print(f"  Avg calls/trace: {total_calls/completed:.1f}")
        print(f"  Avg turns/trace: {total_turns/completed:.1f}")
        print(f"  Est traces/hour at this rate: {completed/max(elapsed,1)*3600:.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
