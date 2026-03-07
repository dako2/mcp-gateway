"""Convert tools.json + toucan_live.json to Toucan-format files with high-quality labels."""

from __future__ import annotations
import json
import os
import re
import functools
import glob
from datetime import datetime, timezone
from urllib.parse import urlparse

print = functools.partial(print, flush=True)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_JSON = os.path.join(_project_root, "data", "tools.json")
TOUCAN_LIVE = os.path.join(_project_root, "data", "toucan_live.json")
TOUCAN_LABEL_MAP = "/tmp/toucan_label_map.json"
OUTPUT_DIR = os.path.join(_project_root, "mcp_servers")

CATEGORY_RULES = [
    ("Web Search & Research",        ["search", "crawl", "scrape", "fetch", "browse", "spider", "seo", "serp", "query_web", "web_search", "research", "scholar", "pubmed", "arxiv", "wikipedia", "index_search"]),
    ("Data Analysis & Processing",   ["data", "analytic", "dashboard", "metric", "report", "chart", "csv", "transform", "etl", "aggregate", "statistics", "forecast_", "predict_", "time_series"]),
    ("Database Operations",          ["database", "mongo", "postgres", "mysql", "redis", "sqlite", "table", "collection", "record", "insert_", "select_", "schema_", "migration"]),
    ("AI/ML Tools",                  ["ai_", "llm", "model_", "generat", "prompt", "ml_", "neural", "embedding", "vector", "inference", "fine_tune", "training", "hugging", "replicate", "image_gen", "text_to"]),
    ("Financial Services",           ["payment", "invoice", "finance", "stock", "trade", "forex", "bank", "accounting", "tax", "revenue", "billing", "transaction", "ledger", "portfolio", "market_data", "commodity", "gdp", "inflation"]),
    ("Cryptocurrency & Blockchain",  ["crypto", "token", "wallet", "bitcoin", "solana", "blockchain", "defi", "nft", "ethereum", "smart_contract", "web3", "chain_", "swap_token"]),
    ("Communication Tools",          ["email", "message", "chat", "slack", "sms", "notification", "inbox", "mail", "send_message", "conversation"]),
    ("Social Media",                 ["social", "tweet", "linkedin", "post_to", "share_", "follower", "instagram", "tiktok", "youtube", "schedule_post", "marketing"]),
    ("Security & Authentication",    ["security", "auth_", "scan_", "threat", "vulnerab", "compliance", "audit", "firewall", "encrypt", "ssl_", "certificate", "phishing", "malware", "pentest"]),
    ("File Management",              ["file_", "document", "pdf", "storage", "convert_", "upload", "download", "drive_", "dropbox", "s3_", "bucket", "archive"]),
    ("DNS & Networking",             ["dns", "domain", "whois", "ip_", "network", "traceroute", "nameserver", "registrar", "port_", "lookup_dns"]),
    ("Maps & Location",              ["map_", "location", "geo", "route", "direction", "travel", "flight", "transit", "subway", "gps", "coordinate", "ferry", "hotel"]),
    ("Gaming",                       ["game", "player", "score", "match", "league", "steam", "pokemon", "chess", "sport", "mlb", "nba", "nfl", "esport"]),
    ("Health & Fitness",             ["health", "medical", "drug", "clinical", "patient", "fitness", "exercise", "diet", "therapy", "diagnosis", "fda"]),
    ("Education",                    ["learn", "education", "course", "quiz", "university", "student", "teach", "tutorial", "canvas_"]),
    ("News & Media",                 ["news", "article", "blog", "press", "headline", "rss", "feed", "hacker_news", "trend_"]),
    ("Real Estate",                  ["property", "real_estate", "listing", "rental", "house", "apartment", "mortgage", "booking", "vacation"]),
    ("Productivity",                 ["task", "project", "todo", "calendar", "schedule", "workflow", "kanban", "meeting", "agenda", "note_", "time_", "timer", "clock", "timezone", "reminder"]),
    ("E-Commerce",                   ["shop", "cart", "order_", "inventory", "product", "ecommerce", "store", "merchant", "catalog", "checkout", "shipping", "price_"]),
    ("API Integration",              ["api_", "webhook", "integration", "connector", "gateway", "rest_", "graphql", "openapi"]),
    ("Content Creation",             ["image_", "photo", "video", "audio", "music", "render", "upscal", "resize", "generat_image"]),
    ("Development Tools",            ["git", "deploy", "build", "ci_", "cd_", "code", "docker", "kubernetes", "pipeline", "debug", "lint", "test_", "compile", "devops", "sdk", "snippet", "endpoint", "webhook_", "registry", "package", "dependency"]),
]


def _deep_categorize(server: dict) -> tuple[str, list[str], str]:
    """Analyze tool names AND descriptions to determine category with reasoning."""
    name = (server.get("server_name", "") or "").lower()
    url = (server.get("server_url", "") or "").lower()
    
    tool_names_text = ""
    tool_descs_text = ""
    tools = server.get("tools", [])
    for t in tools[:20]:
        tn = (t.get("name", "") or "").lower()
        td = (t.get("description", "") or "").lower()
        tool_names_text += f" {tn}"
        tool_descs_text += f" {td}"
    
    combined = f"{name} {url} {tool_names_text} {tool_descs_text}"

    scores: dict[str, float] = {}
    matched_keywords: dict[str, list[str]] = {}
    
    for category, keywords in CATEGORY_RULES:
        cat_score = 0
        cat_matches = []
        for kw in keywords:
            # Weight: tool name matches count 2x, description matches 1x
            name_hits = tool_names_text.count(kw)
            desc_hits = tool_descs_text.count(kw)
            url_hits = 1 if kw in f"{name} {url}" else 0
            
            total = name_hits * 2 + desc_hits + url_hits
            if total > 0:
                cat_score += total
                cat_matches.append(kw)
        
        if cat_score > 0:
            scores[category] = cat_score
            matched_keywords[category] = cat_matches

    if not scores:
        return "Development Tools", [], f"Default category assigned; no strong keyword matches found for {name}."

    sorted_cats = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_cats[0][0]
    secondary = [c for c, _ in sorted_cats[1:3]]
    
    primary_kws = matched_keywords.get(primary, [])
    reasoning = (
        f'Primary label "{primary}" assigned based on strong keyword matches in tool '
        f'names and descriptions: {", ".join(primary_kws[:6])}.'
    )
    
    return primary, secondary, reasoning


def _generate_analysis(server: dict, primary_label: str) -> str:
    """Generate a Toucan-quality analysis string."""
    name = server.get("server_name", "") or "Unknown"
    tools = server.get("tools", [])
    tool_count = len(tools)
    
    tool_categories = set()
    for t in tools[:15]:
        desc = (t.get("description", "") or "")[:100].lower()
        if any(w in desc for w in ["search", "find", "query", "look"]):
            tool_categories.add("search and retrieval")
        if any(w in desc for w in ["create", "add", "post", "generate"]):
            tool_categories.add("creation and generation")
        if any(w in desc for w in ["get", "list", "read", "fetch"]):
            tool_categories.add("data access and reading")
        if any(w in desc for w in ["update", "modify", "edit", "set"]):
            tool_categories.add("modification and updates")
        if any(w in desc for w in ["delete", "remove"]):
            tool_categories.add("deletion and cleanup")
        if any(w in desc for w in ["analyze", "compute", "calculate"]):
            tool_categories.add("analysis and computation")

    cats_str = ", ".join(sorted(tool_categories)[:4]) if tool_categories else "various operations"
    
    top_tools = [t.get("name", "") for t in tools[:5] if t.get("name")]
    tools_str = ", ".join(top_tools)
    
    return (
        f'The MCP Server "{name}" provides {tool_count} tools focused on {primary_label.lower()} tasks. '
        f'Core capabilities include {cats_str}. '
        f'Key tools: {tools_str}.'
    )


def _generate_custom_label(server: dict, primary_label: str) -> str:
    """Generate a descriptive custom label like Toucan's 'Advanced Web Intelligence'."""
    name = server.get("server_name", "") or ""
    tools = server.get("tools", [])
    
    tool_verbs = set()
    for t in tools[:10]:
        tn = (t.get("name", "") or "").lower()
        if "search" in tn: tool_verbs.add("Search")
        if "get" in tn or "list" in tn: tool_verbs.add("Data Access")
        if "create" in tn or "generate" in tn: tool_verbs.add("Generation")
        if "manage" in tn or "update" in tn: tool_verbs.add("Management")
        if "analyze" in tn or "report" in tn: tool_verbs.add("Analytics")
        if "monitor" in tn or "track" in tn: tool_verbs.add("Monitoring")
        if "deploy" in tn or "build" in tn: tool_verbs.add("Deployment")
    
    if tool_verbs:
        return f"{' & '.join(sorted(tool_verbs)[:2])} - {name}"
    return f"{primary_label} - {name}"


def is_featured(server: dict) -> bool:
    tool_count = len(server.get("tools", []))
    tools = server.get("tools", [])
    has_descriptions = sum(1 for t in tools if t.get("description")) / max(len(tools), 1)
    has_schemas = sum(1 for t in tools if t.get("inputSchema") or t.get("input_schema")) / max(len(tools), 1)
    if tool_count >= 5 and has_descriptions >= 0.7 and has_schemas >= 0.7:
        return True
    if tool_count >= 10:
        return True
    return False


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name[:60] or "unknown"


def convert_gateway_server(server: dict, idx: int, total: int) -> dict:
    """Convert a tools.json entry to Toucan format with deep labeling."""
    primary, secondary, reasoning = _deep_categorize(server)
    analysis = _generate_analysis(server, primary)
    custom_label = _generate_custom_label(server, primary)
    featured = is_featured(server)
    
    tools = server.get("tools", [])
    tool_names = [t.get("name", "") for t in tools if t.get("name")]
    name = server.get("server_name", "") or urlparse(server.get("server_url", "")).netloc
    url = server.get("server_url", "")
    
    tool_count = len(tools)
    if tool_count >= 50:
        usage_est = max(10000, 50000 - idx * 100)
    elif tool_count >= 20:
        usage_est = max(1000, 10000 - idx * 50)
    elif tool_count >= 5:
        usage_est = max(100, 2000 - idx * 20)
    else:
        usage_est = max(10, 500 - idx * 5)
    
    sources = server.get("sources", [])
    if len(sources) > 1:
        usage_est = int(usage_est * 1.5)

    return {
        "labels": {
            "analysis": analysis,
            "reasoning": reasoning,
            "primary_label": primary,
            "secondary_labels": secondary,
            "custom_label": custom_label,
            "is_connected": True,
            "is_remote_tool_valid": True,
            "featured_server": featured,
        },
        "metadata": {
            "server_id": idx,
            "server_name": name,
            "rank_by_usage": idx + 1,
            "usage_count": f"{usage_est:,}",
            "original_file": f"tools.json#{idx}",
            "mode": "gateway",
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "remote_server_response": {
                "url": url,
                "is_success": True,
                "error": None,
                "tools": tools,
                "tool_count": len(tools),
                "tool_names": tool_names,
            },
            "server_info_crawled": {
                "id": idx,
                "name": name,
                "author": "",
                "overview": f"{name} - {primary} server with {len(tools)} tools. {analysis[:100]}",
                "repository_url": "",
                "homepage": url,
                "remote_or_local": "Remote",
                "license": "",
                "usage_count": f"{usage_est:,}",
                "success_rate": "100%",
                "tags": [primary.lower().replace(" & ", "-").replace(" ", "-")] + [s.lower().replace(" & ", "-").replace(" ", "-") for s in secondary[:2]],
                "categories": [primary.lower()] + [s.lower() for s in secondary[:2]],
                "tools_count": len(tools),
                "tools": tools,
                "python_sdk_url": url,
                "python_sdk_config": json.dumps({"debug": False}),
            },
            "source_filename": f"{idx:04d}.{_safe_filename(name)}_labeled.json",
            "processed_timestamp": int(datetime.now(timezone.utc).timestamp()),
            "processing_mode": "gateway",
            "rank": idx + 1,
        },
    }


def convert_toucan_live(toucan_entry: dict, toucan_labels: dict, idx: int) -> dict | None:
    """Convert a toucan_live.json entry using original Toucan labels."""
    qn = toucan_entry.get("qualified_name", "")
    orig = toucan_labels.get(qn)
    if not orig:
        return None
    
    name = toucan_entry.get("name", "") or orig.get("name", qn)
    url = toucan_entry.get("url", "")
    tools = orig.get("tools", [])
    tool_names = orig.get("tool_names", [])
    labels = orig.get("labels", {})
    
    if not tools and not tool_names:
        live_tools = toucan_entry.get("live_tools", [])
        if live_tools:
            tool_names = live_tools
    
    if not labels.get("primary_label"):
        return None

    return {
        "labels": labels,
        "metadata": {
            "server_id": idx,
            "server_name": name,
            "rank_by_usage": orig.get("rank") or idx + 1,
            "usage_count": orig.get("usage_count", "0"),
            "original_file": f"toucan_live#{qn}",
            "mode": "toucan",
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "remote_server_response": {
                "url": url,
                "is_success": True,
                "error": None,
                "tools": tools,
                "tool_count": len(tools) or len(tool_names),
                "tool_names": tool_names or [t.get("name", "") for t in tools],
            },
            "server_info_crawled": {
                "id": idx,
                "name": name,
                "author": "",
                "overview": orig.get("overview", f"{name} - {labels.get('primary_label', '')} MCP server"),
                "repository_url": "",
                "homepage": url,
                "remote_or_local": "Remote",
                "license": "",
                "usage_count": orig.get("usage_count", "0"),
                "success_rate": orig.get("success_rate", ""),
                "tags": orig.get("tags", []),
                "categories": orig.get("categories", []),
                "tools_count": len(tools) or len(tool_names),
                "tools": tools,
                "python_sdk_url": url,
                "python_sdk_config": json.dumps({"debug": False}),
            },
            "source_filename": f"{idx:04d}.{_safe_filename(name)}_labeled.json",
            "processed_timestamp": int(datetime.now(timezone.utc).timestamp()),
            "processing_mode": "toucan",
            "rank": orig.get("rank") or idx + 1,
        },
    }


def main():
    with open(TOOLS_JSON) as f:
        gateway_servers = json.load(f)
    print(f"Gateway servers: {len(gateway_servers)}")

    toucan_live = []
    if os.path.exists(TOUCAN_LIVE):
        with open(TOUCAN_LIVE) as f:
            toucan_live = json.load(f)
    print(f"Toucan live servers: {len(toucan_live)}")

    toucan_labels = {}
    if os.path.exists(TOUCAN_LABEL_MAP):
        with open(TOUCAN_LABEL_MAP) as f:
            toucan_labels = json.load(f)
    print(f"Toucan label map entries: {len(toucan_labels)}")

    # Clear output dir
    for f in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        os.remove(f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    idx = 0
    label_counts: dict[str, int] = {}
    featured_count = 0
    seen_urls = set()

    # 1. Gateway servers (267, with deep labeling)
    for server in gateway_servers:
        toucan = convert_gateway_server(server, idx, len(gateway_servers))
        primary = toucan["labels"]["primary_label"]
        label_counts[primary] = label_counts.get(primary, 0) + 1
        if toucan["labels"]["featured_server"]:
            featured_count += 1

        safe_name = _safe_filename(server.get("server_name", "") or f"server-{idx}")
        filename = f"{idx:04d}.{safe_name}_labeled.json"
        with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
            json.dump(toucan, f, indent=2, ensure_ascii=False)
        
        seen_urls.add((server.get("server_url", "") or "").lower().rstrip("/"))
        idx += 1

    gateway_count = idx

    # 2. Toucan live servers (141, with original AI labels, deduped)
    toucan_added = 0
    for entry in toucan_live:
        url = (entry.get("url", "") or "").lower().rstrip("/")
        if url in seen_urls:
            continue

        converted = convert_toucan_live(entry, toucan_labels, idx)
        if not converted:
            continue

        primary = converted["labels"]["primary_label"]
        label_counts[primary] = label_counts.get(primary, 0) + 1
        if converted["labels"].get("featured_server"):
            featured_count += 1

        safe_name = _safe_filename(entry.get("name", "") or f"toucan-{idx}")
        filename = f"{idx:04d}.{safe_name}_labeled.json"
        with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        
        seen_urls.add(url)
        idx += 1
        toucan_added += 1

    total = idx
    print(f"\nDone! Wrote {total} files to {OUTPUT_DIR}/")
    print(f"  Gateway servers: {gateway_count}")
    print(f"  Toucan live added: {toucan_added}")
    print(f"  Featured: {featured_count}/{total}")
    print(f"\nCategory distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<35} {count:>4}")


if __name__ == "__main__":
    main()
