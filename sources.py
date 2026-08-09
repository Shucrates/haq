"""Tier B retrieval — web search over government domains only.

The hand-authored corpus in data/statutes.json covers twelve situations. Everything
else got no grounding at all. This module fills that gap without touching the
guarantee the product is built on.

  Tier A  data/statutes.json, human-verified   -> CITABLE in a filing
  Tier B  fetched here, gov domains only       -> CONTEXT ONLY, never citable

Enforcement needs no new safety code. drafting.validate_citations() builds its
allowed set from the Tier A snippets retrieved this session; Tier B ids are simply
never added, so any web-derived citation the model emits is deleted by the renderer
that already exists.

That makes the Q&A answer stronger, not weaker:

    "We do search the web — government domains only. And the renderer still deletes
     every citation that isn't human-verified. Here's a draft where it stripped one."
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass

import httpx

import sarvam
from demo_cache import cached, key_for

log = logging.getLogger("haq.sources")

FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
TIMEOUT = httpx.Timeout(45.0, connect=10.0)

# Retrieval is restricted at the API level, not filtered afterwards. This list is
# the entire universe HAQ will read law from.
GOV_DOMAINS = [
    "rbi.org.in",
    "indiacode.nic.in",
    "rti.gov.in",
    "consumerhelpline.gov.in",
    "sebi.gov.in",
    "trai.gov.in",
    "irdai.gov.in",
    "npci.org.in",
    "pgportal.gov.in",
    "egazette.gov.in",
]

MAX_SNIPPET_CHARS = 400
WEB_ID_PREFIX = "web_"

SNIPPET_PROMPT = """You extract one short legal statement from an official Indian
government page.

Reply with ONLY a JSON object:
{"text": "<one or two sentences stating the rule, verbatim where possible>",
 "section_ref": "<section/clause number if the page states one, else null>",
 "relevance": "<high|medium|low>"}

Do not summarise the whole page. Do not invent a section number. If the page states
no rule relevant to the query, set relevance to "low"."""


@dataclass
class WebSource:
    id: str
    url: str
    title: str
    text: str
    section_ref: str | None
    relevance: str
    provenance: str = "web"      # never "human" — that is Tier A's word

    def as_dict(self) -> dict:
        return asdict(self)


def enabled() -> bool:
    return bool(os.getenv("FIRECRAWL_API_KEY"))


def _slug(url: str) -> str:
    """Stable, readable id. The web_ prefix makes Tier B obvious in a draft and in
    the logs, and guarantees it can never collide with a statutes.json id."""
    tail = re.sub(r"[^a-z0-9]+", "_", url.lower().split("://")[-1]).strip("_")
    return f"{WEB_ID_PREFIX}{tail[-48:]}"


def search(query: str, limit: int = 4) -> list[dict]:
    """Firecrawl /v2/search restricted to GOV_DOMAINS, returning page markdown in
    the same call. Cached like every Sarvam call so HAQ_MODE=replay still runs with
    the wifi off — an uncached network call on the demo path is exactly the failure
    demo mode exists to prevent."""
    if not enabled():
        return []

    payload = {
        "query": query[:500],
        "limit": limit,
        "country": "IN",
        "includeDomains": GOV_DOMAINS,
        "scrapeOptions": {"formats": [{"type": "markdown"}]},
    }

    def call() -> dict:
        headers = {"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}"}
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(FIRECRAWL_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"Firecrawl -> {response.status_code}: {response.text[:300]}")
        return response.json()

    try:
        out = cached("websearch", key_for(payload), call)
    except Exception as exc:  # noqa: BLE001 — context is a bonus, never a blocker
        log.warning("web_search_failed: %s", exc)
        return []

    return (out.get("data") or {}).get("web") or []


def extract_snippet(markdown: str, url: str, query: str) -> dict | None:
    """The LLM parsing layer: turn a page of markdown into one short, quotable
    statement. Retries once via sarvam.chat's own path, then falls back to the head
    of the page — the same never-let-a-parse-error-surface rule as agent.py."""
    if not markdown:
        return None

    try:
        raw = sarvam.chat(
            [
                {"role": "system", "content": SNIPPET_PROMPT},
                {"role": "user", "content": f"Query: {query}\n\nPage ({url}):\n{markdown[:6000]}"},
            ],
            max_tokens=300,
        )
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            import json

            parsed = json.loads(match.group(0))
            text = (parsed.get("text") or "").strip()
            if text:
                return {
                    "text": text[:MAX_SNIPPET_CHARS],
                    "section_ref": parsed.get("section_ref") or None,
                    "relevance": parsed.get("relevance") or "medium",
                }
    except Exception as exc:  # noqa: BLE001
        log.warning("snippet_extract_failed: %s", exc)

    plain = re.sub(r"[#*`>\[\]()]", " ", markdown)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        return None
    return {"text": plain[:300], "section_ref": None, "relevance": "low"}


def context_for(query: str, limit: int = 3) -> list[WebSource]:
    """Search, parse, return Tier B context. Low-relevance hits are dropped: half a
    page of unrelated statute is worse than nothing in a drafting prompt."""
    out: list[WebSource] = []

    for hit in search(query, limit=limit):
        url = hit.get("url") or ""
        if not url:
            continue
        snippet = extract_snippet(hit.get("markdown") or "", url, query)
        if not snippet or snippet["relevance"] == "low":
            continue
        out.append(WebSource(
            id=_slug(url),
            url=url,
            title=hit.get("title") or url,
            text=snippet["text"],
            section_ref=snippet["section_ref"],
            relevance=snippet["relevance"],
        ))

    return out
