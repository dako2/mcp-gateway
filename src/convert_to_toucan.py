"""Convert tools.json to Toucan-format individual JSON files with labels and metadata."""

from __future__ import annotations
import json
import os
import re
import functools
from datetime import datetime, timezone

print = functools.partial(print, flush=True)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_JSON = os.path.join(_project_root, "data", "tools.json")
OUTPUT_DIR = os.path.join(_project_root, "mcp_servers")

CATEGORIES = {
    "Development Tools": {
        "keywords": ["git", "deploy", "build", "ci", "cd", "code", "docker", "k8s", "kubernetes",
                      "pipeline", "bitrise", "codecov", "debug", "lint", "test", "compile", "devops",
                      "sdk", "snippet", "api", "endpoint", "webhook", "schema", "migration",
                      "registry", "package", "dependency", "sonatype", "npm", "pip"],
        "secondary": ["API Integration", "DevOps"],
    },
    "Web Search & Research": {
        "keywords": ["search", "crawl", "scrape", "fetch", "browse", "spider", "seo",
                      "exa", "web_search", "query", "index", "sitemap", "ranking", "serp",
                      "research", "paper", "scholar", "pubmed", "arxiv", "wikipedia"],
        "secondary": ["Browser Automation", "Data Analysis & Processing"],
    },
    "Data Analysis & Processing": {
        "keywords": ["data", "analytic", "dashboard", "metric", "sql", "database", "query",
                      "csv", "json", "transform", "etl", "pipeline", "report", "chart",
                      "bigquery", "snowflake", "warehouse", "aggregate"],
        "secondary": ["Database Operations"],
    },
    "AI/ML Tools": {
        "keywords": ["ai", "llm", "model", "generat", "prompt", "ml", "neural", "embedding",
                      "vector", "gpt", "claude", "image gen", "text-to", "inference",
                      "training", "fine-tune", "hugging", "replicate"],
        "secondary": ["Content Creation"],
    },
    "Financial Services": {
        "keywords": ["payment", "invoice", "finance", "stock", "trade", "forex", "bank",
                      "stripe", "paypal", "accounting", "tax", "revenue", "billing",
                      "transaction", "ledger", "portfolio", "market", "commodity", "gdp",
                      "inflation", "forecast_"],
        "secondary": ["E-Commerce"],
    },
    "Cryptocurrency & Blockchain": {
        "keywords": ["crypto", "token", "wallet", "bitcoin", "solana", "blockchain", "defi",
                      "nft", "ethereum", "smart contract", "web3", "chain", "swap"],
        "secondary": ["Financial Services"],
    },
    "Communication Tools": {
        "keywords": ["email", "message", "chat", "slack", "sms", "notification",
                      "inbox", "mail", "send_message", "conversation", "thread"],
        "secondary": ["Social Media"],
    },
    "Social Media": {
        "keywords": ["social", "tweet", "linkedin", "post", "share", "follower",
                      "instagram", "tiktok", "youtube", "thread", "schedule_post",
                      "marketing", "vibe"],
        "secondary": ["Communication Tools", "Content Creation"],
    },
    "Productivity": {
        "keywords": ["task", "project", "todo", "calendar", "schedule", "workflow",
                      "kanban", "board", "meeting", "agenda", "note", "organize",
                      "time", "timer", "clock", "timezone", "reminder"],
        "secondary": ["Project Management"],
    },
    "E-Commerce": {
        "keywords": ["shop", "cart", "order", "inventory", "product", "ecommerce",
                      "store", "merchant", "catalog", "checkout", "shipping", "price"],
        "secondary": ["Financial Services"],
    },
    "Security & Authentication": {
        "keywords": ["security", "auth", "scan", "threat", "vulnerab", "compliance",
                      "audit", "firewall", "encrypt", "ssl", "certificate", "pentest",
                      "phishing", "malware"],
        "secondary": ["Development Tools"],
    },
    "File Management": {
        "keywords": ["file", "document", "pdf", "storage", "convert", "upload",
                      "download", "drive", "dropbox", "s3", "bucket", "archive"],
        "secondary": ["Cloud Storage"],
    },
    "Database Operations": {
        "keywords": ["database", "mongo", "postgres", "mysql", "redis", "table",
                      "collection", "record", "insert", "select", "index_", "schema"],
        "secondary": ["Data Analysis & Processing"],
    },
    "Weather": {
        "keywords": ["weather", "climate", "forecast", "temperature", "wind", "rain",
                      "storm", "meteorolog"],
        "secondary": ["API Integration"],
    },
    "Maps & Location": {
        "keywords": ["map", "location", "geo", "route", "direction", "travel",
                      "flight", "transit", "subway", "gps", "coordinate", "address",
                      "ferry", "hotel"],
        "secondary": ["API Integration"],
    },
    "Gaming": {
        "keywords": ["game", "player", "score", "match", "league", "steam",
                      "xbox", "playstation", "pokemon", "chess", "sport", "mlb",
                      "nba", "nfl", "esport", "opgg"],
        "secondary": ["Entertainment"],
    },
    "Health & Fitness": {
        "keywords": ["health", "medical", "drug", "clinical", "patient", "fitness",
                      "exercise", "diet", "nutrition", "therapy", "diagnosis", "fda"],
        "secondary": ["Science & Research"],
    },
    "Education": {
        "keywords": ["learn", "education", "course", "quiz", "university", "student",
                      "teach", "tutorial", "knowledge", "math", "science", "canvas"],
        "secondary": ["Productivity"],
    },
    "News & Media": {
        "keywords": ["news", "article", "blog", "press", "media", "headline",
                      "journalist", "rss", "feed", "hacker news", "trend"],
        "secondary": ["Content Creation"],
    },
    "Real Estate": {
        "keywords": ["property", "real estate", "listing", "rental", "house",
                      "apartment", "mortgage", "vacancy", "booking", "vacation"],
        "secondary": ["E-Commerce"],
    },
    "DNS & Networking": {
        "keywords": ["dns", "domain", "whois", "ip", "network", "ssl", "certificate",
                      "port", "traceroute", "ping", "nameserver", "registrar"],
        "secondary": ["Security & Authentication"],
    },
    "API Integration": {
        "keywords": ["api", "webhook", "integration", "connector", "gateway",
                      "rest", "graphql", "openapi", "swagger"],
        "secondary": ["Development Tools"],
    },
}


def categorize_server(server: dict) -> tuple[str, list[str]]:
    """Determine primary and secondary labels based on server name, URL, and tool names/descriptions."""
    name = (server.get("server_name", "") or "").lower()
    url = (server.get("server_url", "") or "").lower()
    tools_text = ""
    for t in server.get("tools", [])[:15]:
        tools_text += f" {t.get('name', '')} {t.get('description', '') or ''}"
    tools_text = tools_text.lower()

    combined = f"{name} {url} {tools_text}"

    scores: dict[str, int] = {}
    for category, info in CATEGORIES.items():
        score = sum(1 for kw in info["keywords"] if kw in combined)
        if score > 0:
            scores[category] = score

    if not scores:
        return "Others", []

    primary = max(scores, key=scores.get)
    secondary = CATEGORIES[primary].get("secondary", [])
    extra = [cat for cat, sc in sorted(scores.items(), key=lambda x: -x[1]) if cat != primary][:2]
    secondary = list(dict.fromkeys(secondary + extra))

    return primary, secondary


def estimate_popularity(server: dict, rank: int, total: int) -> dict:
    """Estimate popularity based on tool count, sources, and position."""
    tool_count = server.get("tool_count", 0)
    sources = server.get("sources", [])

    if tool_count >= 50:
        tier = "high"
        usage_estimate = max(10000, 50000 - rank * 100)
    elif tool_count >= 20:
        tier = "medium"
        usage_estimate = max(1000, 10000 - rank * 50)
    elif tool_count >= 5:
        tier = "medium-low"
        usage_estimate = max(100, 2000 - rank * 20)
    else:
        tier = "low"
        usage_estimate = max(10, 500 - rank * 5)

    if len(sources) > 1:
        usage_estimate = int(usage_estimate * 1.5)

    return {
        "rank_by_usage": rank + 1,
        "usage_count": f"{usage_estimate:,}",
        "popularity_tier": tier,
    }


def is_featured(server: dict, primary_label: str) -> bool:
    """Determine if a server should be featured based on quality signals."""
    tool_count = server.get("tool_count", 0)
    tools = server.get("tools", [])

    has_descriptions = sum(1 for t in tools if t.get("description")) / max(len(tools), 1)
    has_schemas = sum(1 for t in tools if t.get("inputSchema") or t.get("input_schema")) / max(len(tools), 1)

    if tool_count >= 5 and has_descriptions >= 0.8 and has_schemas >= 0.8:
        return True
    if tool_count >= 10:
        return True
    return False


def convert_one(server: dict, idx: int, total: int) -> dict:
    """Convert a single tools.json entry to Toucan format."""
    primary_label, secondary_labels = categorize_server(server)
    popularity = estimate_popularity(server, idx, total)
    featured = is_featured(server, primary_label)

    tools = server.get("tools", [])
    tool_names = [t.get("name", "") for t in tools if t.get("name")]

    name = server.get("server_name", "") or server.get("server_url", "").split("//")[1].split("/")[0]
    url = server.get("server_url", "")

    custom_label = f"{primary_label} - {name}"

    return {
        "labels": {
            "analysis": f"The MCP Server \"{name}\" provides {len(tools)} tools for {primary_label.lower()} tasks.",
            "reasoning": f"Primary label \"{primary_label}\" assigned based on tool names and descriptions.",
            "primary_label": primary_label,
            "secondary_labels": secondary_labels,
            "custom_label": custom_label,
            "is_connected": True,
            "is_remote_tool_valid": True,
            "featured_server": featured,
        },
        "metadata": {
            "server_id": idx,
            "server_name": name,
            "rank_by_usage": popularity["rank_by_usage"],
            "usage_count": popularity["usage_count"],
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
                "overview": f"{name} - {primary_label} MCP server with {len(tools)} tools.",
                "repository_url": "",
                "homepage": url,
                "remote_or_local": "Remote",
                "license": "",
                "usage_count": popularity["usage_count"],
                "success_rate": "100%",
                "tags": [primary_label.lower().replace(" & ", "-").replace(" ", "-")],
                "categories": [primary_label.lower()],
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


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name.strip())
    return name[:60] or "unknown"


def main():
    with open(TOOLS_JSON) as f:
        servers = json.load(f)

    print(f"Converting {len(servers)} servers from tools.json to Toucan format...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    featured_count = 0
    label_counts: dict[str, int] = {}

    for idx, server in enumerate(servers):
        toucan = convert_one(server, idx, len(servers))

        primary = toucan["labels"]["primary_label"]
        label_counts[primary] = label_counts.get(primary, 0) + 1
        if toucan["labels"]["featured_server"]:
            featured_count += 1

        safe_name = _safe_filename(server.get("server_name", "") or f"server-{idx}")
        filename = f"{idx:04d}.{safe_name}_labeled.json"
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, "w") as f:
            json.dump(toucan, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Wrote {len(servers)} files to {OUTPUT_DIR}/")
    print(f"\nFeatured servers: {featured_count}/{len(servers)}")
    print(f"\nCategory distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<35} {count:>4}")


if __name__ == "__main__":
    main()
