"""Verify MCP server tool health by calling one tool per server through the gateway.

Usage:
    # Make sure gateway is running first:
    #   cd src && uvicorn gateway:app --port 8000
    # Then:
    cd src && python -u verify_servers.py
    cd src && python -u verify_servers.py --concurrent 30 --timeout 20
    cd src && python -u verify_servers.py --update-labels
    cd src && python -u verify_servers.py --retry-errors
"""

from __future__ import annotations
import argparse
import asyncio
import glob
import json
import os
import random
import re
import time
import functools
from datetime import datetime, timezone

import httpx

print = functools.partial(print, flush=True)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000")

# Tool name patterns ordered by "safety" (read-only, likely to work with dummy args)
_SAFE_TOOL_PATTERNS = [
    re.compile(r"(^|_|-)(list|search|get|fetch|query|find|lookup)($|_|-)", re.I),
    re.compile(r"(^|_|-)(status|health|ping|version|info|help|about)($|_|-)", re.I),
    re.compile(r"(^|_|-)(calculate|convert|check|count|stats|read)($|_|-)", re.I),
]

# Auth-related error strings in responses
_AUTH_ERROR_STRINGS = [
    "unauthorized", "unauthenticated", "authentication required",
    "invalid api key", "api key required", "access denied",
    "forbidden", "not authorized", "login required",
    "missing.*token", "invalid.*token", "expired.*token",
    "missing.*credential", "invalid.*credential",
]
_AUTH_ERROR_RE = re.compile("|".join(_AUTH_ERROR_STRINGS), re.I)


def _load_all_servers(servers_dir: str) -> list[dict]:
    """Load all mcp_servers/*.json files with their file paths."""
    pattern = os.path.join(servers_dir, "*.json")
    files = sorted(glob.glob(pattern))
    servers = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            data["_filepath"] = fpath
            data["_filename"] = os.path.basename(fpath)
            servers.append(data)
        except Exception as e:
            print(f"  [load] error reading {fpath}: {e}")
    return servers


def _get_server_url(server: dict) -> str:
    """Extract the server URL from metadata."""
    meta = server.get("metadata", {})
    resp = meta.get("remote_server_response", {})
    return resp.get("url", "") or meta.get("server_info_crawled", {}).get("python_sdk_url", "")


def _get_tools(server: dict) -> list[dict]:
    """Extract tool definitions from server metadata."""
    resp = server.get("metadata", {}).get("remote_server_response", {})
    return resp.get("tools", [])


def _pick_test_tool(tools: list[dict]) -> dict | None:
    """Pick the safest tool to test: read-only patterns first, fewest required params."""
    if not tools:
        return None

    def _required_count(tool: dict) -> int:
        schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        return len(schema.get("required", []))

    # Try each safety tier in order
    for pattern in _SAFE_TOOL_PATTERNS:
        candidates = [t for t in tools if t.get("name") and pattern.search(t["name"])]
        if candidates:
            candidates.sort(key=_required_count)
            return candidates[0]

    # Fallback: any tool with 0 required params
    zero_req = [t for t in tools if t.get("name") and _required_count(t) == 0]
    if zero_req:
        return zero_req[0]

    # Last resort: tool with fewest required params
    named = [t for t in tools if t.get("name")]
    if named:
        named.sort(key=_required_count)
        return named[0]

    return None


def _generate_test_args(tool: dict) -> dict:
    """Generate minimal test arguments from the tool's JSON Schema."""
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    args = {}
    for param_name in required:
        prop = properties.get(param_name, {})
        args[param_name] = _default_for_schema(prop, param_name)

    return args


def _default_for_schema(prop: dict, name: str = "") -> object:
    """Generate a safe default value from a JSON Schema property."""
    if "enum" in prop and prop["enum"]:
        return prop["enum"][0]

    if "default" in prop:
        return prop["default"]

    typ = prop.get("type", "string")
    if typ == "string":
        name_lower = name.lower()
        if any(kw in name_lower for kw in ("query", "search", "keyword", "term", "text", "q")):
            return "test"
        if any(kw in name_lower for kw in ("url", "uri", "link")):
            return "https://example.com"
        if any(kw in name_lower for kw in ("id", "identifier")):
            return "1"
        if any(kw in name_lower for kw in ("name",)):
            return "test"
        if any(kw in name_lower for kw in ("email",)):
            return "test@example.com"
        if "format" in prop:
            fmt = prop["format"]
            if fmt == "uri":
                return "https://example.com"
            if fmt == "email":
                return "test@example.com"
            if fmt in ("date", "date-time"):
                return "2025-01-01"
        return "test"
    elif typ in ("number", "float"):
        return 1.0
    elif typ == "integer":
        return 1
    elif typ == "boolean":
        return True
    elif typ == "array":
        return []
    elif typ == "object":
        obj_props = prop.get("properties", {})
        obj_required = set(prop.get("required", []))
        if obj_required and obj_props:
            return {k: _default_for_schema(obj_props[k], k) for k in obj_required if k in obj_props}
        return {}
    else:
        return "test"


def _classify_response(http_status: int, gateway_resp: dict) -> tuple[str, str]:
    """Classify the gateway response into a status and reason.

    Returns (status, reason) where status is one of:
      verified, auth_required, broken, tool_error
    """
    # Gateway-level errors
    if "error" in gateway_resp and "meta" not in gateway_resp:
        error_msg = str(gateway_resp.get("error", ""))
        error_lower = error_msg.lower()
        if "timed out" in error_lower:
            return "broken", f"gateway timeout: {error_msg[:100]}"
        if "connect" in error_lower:
            return "broken", f"connection error: {error_msg[:100]}"
        if "rate limited" in error_lower:
            return "broken", f"rate limited: {error_msg[:100]}"
        return "broken", f"gateway error: {error_msg[:100]}"

    meta = gateway_resp.get("meta", {})
    status_code = meta.get("status_code", http_status)
    result = gateway_resp.get("result", {})
    result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)

    if status_code in (401, 403):
        return "auth_required", f"HTTP {status_code}"

    if status_code in (502, 504):
        return "broken", f"HTTP {status_code}"

    if _AUTH_ERROR_RE.search(result_str[:2000]):
        return "auth_required", f"auth error in response: {result_str[:100]}"

    if status_code == 200:
        # Check for MCP-level errors
        if isinstance(result, dict):
            error = result.get("error")
            if error:
                error_str = json.dumps(error) if isinstance(error, dict) else str(error)
                if _AUTH_ERROR_RE.search(error_str):
                    return "auth_required", f"MCP auth error: {error_str[:100]}"
                return "tool_error", f"MCP error: {error_str[:100]}"
            if "result" in result:
                return "verified", "tool call succeeded"
            # Some servers return the result directly without the jsonrpc wrapper
            return "verified", "response received (non-standard format)"
        return "verified", f"response received"

    if status_code >= 400:
        if _AUTH_ERROR_RE.search(result_str[:2000]):
            return "auth_required", f"HTTP {status_code} with auth error"
        return "tool_error", f"HTTP {status_code}: {result_str[:100]}"

    return "verified", f"HTTP {status_code}"


async def _test_server(
    client: httpx.AsyncClient,
    server: dict,
    gateway_url: str,
    timeout: float,
) -> dict:
    """Test a single server by calling one tool through the gateway."""
    name = server.get("metadata", {}).get("server_name", "unknown")
    url = _get_server_url(server)
    tools = _get_tools(server)
    mode = server.get("metadata", {}).get("mode", "unknown")
    is_smithery = "smithery" in url.lower()

    result = {
        "server_name": name,
        "server_url": url,
        "file": server.get("_filename", ""),
        "status": "broken",
        "reason": "",
        "tested_tool": None,
        "tool_args": None,
        "response_snippet": None,
        "latency_ms": 0,
        "tool_count": len(tools),
        "mode": mode,
        "is_smithery": is_smithery,
    }

    if not url:
        result["reason"] = "no server URL"
        return result

    if not tools:
        result["reason"] = "no tools defined"
        return result

    tool = _pick_test_tool(tools)
    if not tool:
        result["reason"] = "no suitable tool found"
        return result

    tool_name = tool["name"]
    test_args = _generate_test_args(tool)
    result["tested_tool"] = tool_name
    result["tool_args"] = test_args

    mcp_request = {
        "jsonrpc": "2.0",
        "id": random.randint(1, 99999),
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": test_args},
    }

    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{gateway_url}/mcp",
            json={"target": url, "request": mcp_request},
            timeout=timeout,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        result["latency_ms"] = latency_ms

        gateway_resp = resp.json()
        status, reason = _classify_response(resp.status_code, gateway_resp)
        result["status"] = status
        result["reason"] = reason

        # Store response snippet
        resp_result = gateway_resp.get("result", {})
        snippet = json.dumps(resp_result) if isinstance(resp_result, (dict, list)) else str(resp_result)
        result["response_snippet"] = snippet[:300]

    except httpx.TimeoutException:
        result["latency_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"] = "broken"
        result["reason"] = f"client timeout after {result['latency_ms']}ms"
    except httpx.ConnectError as e:
        result["status"] = "broken"
        result["reason"] = f"connection error: {str(e)[:100]}"
    except Exception as e:
        result["status"] = "broken"
        result["reason"] = f"unexpected error: {str(e)[:100]}"

    return result


def _update_server_labels(server: dict, test_result: dict) -> None:
    """Update a server's labels in its JSON file with verification results."""
    fpath = server.get("_filepath")
    if not fpath or not os.path.exists(fpath):
        return

    try:
        with open(fpath) as f:
            data = json.load(f)

        if "labels" not in data:
            data["labels"] = {}

        data["labels"]["tool_verified"] = test_result["status"] == "verified"
        data["labels"]["tool_verified_status"] = test_result["status"]
        data["labels"]["tool_verified_at"] = datetime.now(timezone.utc).isoformat()
        data["labels"]["tool_verified_tool"] = test_result.get("tested_tool", "")
        data["labels"]["tool_verified_reason"] = test_result.get("reason", "")

        with open(fpath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [label] error updating {fpath}: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Verify MCP server tool health via gateway")
    parser.add_argument("--concurrent", type=int, default=20, help="Max concurrent checks")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout per tool call (seconds)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_project_root, "data", "verified_servers.json"))
    parser.add_argument("--servers-dir", type=str,
                        default=os.path.join(_project_root, "mcp_servers"))
    parser.add_argument("--gateway-url", type=str, default=GATEWAY_URL)
    parser.add_argument("--update-labels", action="store_true",
                        help="Update mcp_servers/*.json with verification results")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Only re-test servers that previously failed")
    args = parser.parse_args()

    # Check gateway
    async with httpx.AsyncClient() as check:
        try:
            r = await check.get(f"{args.gateway_url}/health", timeout=3)
            print(f"Gateway: {r.json()}")
        except Exception:
            print(f"ERROR: Gateway not running at {args.gateway_url}")
            print("Start it first: cd src && uvicorn gateway:app --port 8000")
            return

    servers = _load_all_servers(args.servers_dir)
    print(f"Loaded {len(servers)} server files from {args.servers_dir}")

    # Filter to retry-only if requested
    if args.retry_errors:
        prev_results = {}
        if os.path.exists(args.output):
            with open(args.output) as f:
                prev = json.load(f)
            for s in prev.get("servers", []):
                prev_results[s["file"]] = s
        retry_servers = []
        skip_count = 0
        for s in servers:
            fname = s.get("_filename", "")
            prev = prev_results.get(fname)
            if prev and prev["status"] == "verified":
                skip_count += 1
            else:
                retry_servers.append(s)
        print(f"Retry mode: skipping {skip_count} verified, re-testing {len(retry_servers)}")
        servers = retry_servers

    sem = asyncio.Semaphore(args.concurrent)
    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrent + 10, max_keepalive_connections=args.concurrent),
        timeout=args.timeout + 5,
    )

    results = []
    completed = 0
    t_start = time.monotonic()

    async def _check_one(idx: int, server: dict) -> dict:
        nonlocal completed
        async with sem:
            name = server.get("metadata", {}).get("server_name", "?")[:45]
            url = _get_server_url(server)[:60]
            print(f"[{idx+1}/{len(servers)}] {name}")

            result = await _test_server(client, server, args.gateway_url, args.timeout)

            status_tag = {
                "verified": "OK ",
                "auth_required": "AUTH",
                "broken": "FAIL",
                "tool_error": "ERR ",
            }.get(result["status"], "????")

            tool_info = f"tool={result['tested_tool']}" if result["tested_tool"] else "no tool"
            print(f"  [{status_tag}] {tool_info} ({result['latency_ms']}ms) {result['reason'][:60]}")

            if args.update_labels:
                _update_server_labels(server, result)

            completed += 1
            return result

    tasks = [_check_one(i, s) for i, s in enumerate(servers)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    await client.aclose()

    # Process results
    final_results = []
    errors = 0
    for r in results:
        if isinstance(r, Exception):
            errors += 1
            print(f"  [error] unexpected: {r}")
        else:
            final_results.append(r)

    # If retry mode, merge with previous verified results
    if args.retry_errors and os.path.exists(args.output):
        with open(args.output) as f:
            prev = json.load(f)
        prev_verified = {s["file"]: s for s in prev.get("servers", []) if s["status"] == "verified"}
        retested_files = {r["file"] for r in final_results}
        for fname, prev_result in prev_verified.items():
            if fname not in retested_files:
                final_results.append(prev_result)

    # Sort by status then tool count
    status_order = {"verified": 0, "tool_error": 1, "auth_required": 2, "broken": 3}
    final_results.sort(key=lambda r: (status_order.get(r["status"], 9), -r.get("tool_count", 0)))

    # Summary
    summary = {
        "total": len(final_results),
        "verified": sum(1 for r in final_results if r["status"] == "verified"),
        "tool_error": sum(1 for r in final_results if r["status"] == "tool_error"),
        "auth_required": sum(1 for r in final_results if r["status"] == "auth_required"),
        "broken": sum(1 for r in final_results if r["status"] == "broken"),
    }

    elapsed = time.monotonic() - t_start

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "gateway_url": args.gateway_url,
        "summary": summary,
        "servers": final_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Total servers: {summary['total']}")
    print(f"  Verified:      {summary['verified']}")
    print(f"  Tool error:    {summary['tool_error']}")
    print(f"  Auth required: {summary['auth_required']}")
    print(f"  Broken:        {summary['broken']}")
    print(f"  Output: {args.output}")
    if args.update_labels:
        print(f"  Labels updated in {args.servers_dir}/*.json")
    print(f"{'='*60}")

    # Print verified servers summary
    verified = [r for r in final_results if r["status"] == "verified"]
    if verified:
        total_tools = sum(r.get("tool_count", 0) for r in verified)
        print(f"\nVerified servers ({len(verified)}) -- {total_tools} total tools:")
        for r in verified[:30]:
            sm = "S" if r.get("is_smithery") else "D"
            print(f"  [{sm}] {r['server_name'][:42]:<44} tools={r['tool_count']:<4} tested={r['tested_tool']}")
        if len(verified) > 30:
            print(f"  ... and {len(verified) - 30} more")


if __name__ == "__main__":
    asyncio.run(main())
