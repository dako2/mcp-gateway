"""Technique 5 – DNS/Endpoint Probing: try mcp.{company}.com/mcp for known domains."""

from __future__ import annotations
import asyncio
import httpx

# Known tech companies / services likely to host MCP servers.
# We'll try common URL patterns for each.
DOMAINS = [
    "anthropic.com", "openai.com", "google.com", "microsoft.com",
    "aws.amazon.com", "cloudflare.com", "vercel.com", "netlify.com",
    "supabase.com", "neon.tech", "planetscale.com", "turso.tech",
    "railway.app", "render.com", "fly.io", "heroku.com",
    "digitalocean.com", "linode.com", "vultr.com",
    "stripe.com", "paypal.com", "square.com", "shopify.com",
    "twilio.com", "sendgrid.com", "mailgun.com", "postmark.com",
    "slack.com", "discord.com", "telegram.org",
    "notion.so", "linear.app", "asana.com", "monday.com",
    "github.com", "gitlab.com", "bitbucket.org",
    "jira.atlassian.com", "confluence.atlassian.com",
    "sentry.io", "datadog.com", "newrelic.com", "grafana.com",
    "elastic.co", "mongodb.com", "redis.com", "cockroachlabs.com",
    "snowflake.com", "databricks.com", "fivetran.com",
    "hubspot.com", "salesforce.com", "zendesk.com", "intercom.com",
    "figma.com", "canva.com", "adobe.com", "sketch.com",
    "spotify.com", "youtube.com", "tiktok.com",
    "dropbox.com", "box.com", "onedrive.com",
    "zoom.us", "webex.com", "teams.microsoft.com",
    "airtable.com", "coda.io", "retool.com",
    "algolia.com", "meilisearch.com", "typesense.org",
    "auth0.com", "okta.com", "clerk.com",
    "prisma.io", "hasura.io", "apollographql.com",
    "docker.com", "kubernetes.io",
    "terraform.io", "pulumi.com",
    "deepwiki.com", "exa.ai", "perplexity.ai", "cohere.com",
    "mistral.ai", "huggingface.co", "replicate.com",
    "langchain.com", "llamaindex.ai",
    "semgrep.dev", "snyk.io", "sonarcloud.io",
    "pagerduty.com", "opsgenie.com",
    "launchdarkly.com", "split.io",
    "segment.com", "mixpanel.com", "amplitude.com",
    "plaid.com", "ramp.com", "brex.com",
    "zapier.com", "make.com", "n8n.io",
    "wix.com", "webflow.com", "squarespace.com",
    "wordpress.com", "ghost.org",
    "deno.com", "bun.sh", "nodejs.org",
]

URL_PATTERNS = [
    "https://mcp.{domain}/mcp",
    "https://mcp.{domain}/sse",
    "https://mcp.{domain}",
    "https://{domain}/mcp",
    "https://{domain}/api/mcp",
]


async def _probe_one(client: httpx.AsyncClient, url: str, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        try:
            resp = await client.head(url, timeout=3, follow_redirects=True)
            if resp.status_code < 500:
                return {
                    "name": "",
                    "url": url,
                    "transport": "sse" if "/sse" in url else "streamable-http" if "/mcp" in url else "unknown",
                    "source": "dns-probe",
                    "_status_code": resp.status_code,
                }
        except (httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException):
            pass
    return None


async def fetch(client: httpx.AsyncClient) -> list[dict]:
    candidates: list[str] = []
    for domain in DOMAINS:
        base = domain.split(".")[-2] if "." in domain else domain
        for pattern in URL_PATTERNS:
            candidates.append(pattern.format(domain=domain))
            if "." in domain:
                candidates.append(pattern.format(domain=base + "." + domain.split(".")[-1]))

    candidates = list(set(candidates))
    print(f"[dns-probe] probing {len(candidates)} candidate URLs...")

    sem = asyncio.Semaphore(50)
    tasks = [_probe_one(client, url, sem) for url in candidates]
    results = await asyncio.gather(*tasks)

    entries = [r for r in results if r is not None]

    # dedupe, prefer /mcp over /sse
    seen_hosts: dict[str, dict] = {}
    for e in entries:
        from urllib.parse import urlparse
        host = urlparse(e["url"]).netloc
        if host not in seen_hosts or "/mcp" in e["url"]:
            seen_hosts[host] = e

    deduped = list(seen_hosts.values())
    print(f"[dns-probe] discovered {len(deduped)} live endpoints")
    return deduped
