"""Tier B safety tests.

The product's central claim is that every citation in a filing is human-verified
with a source URL. Adding web retrieval is exactly the change that could quietly
make that false, so these tests assert the boundary rather than the plumbing.

If `test_web_citation_is_stripped_from_the_filing` ever goes red, the Q&A answer
has become a lie. Treat it as a release blocker, not a flaky test.
"""

import pytest

import drafting
import sources
from drafting import Retrieved, validate_citations


@pytest.fixture
def tier_a():
    return [Retrieved(
        id="rbi_bsbda_no_amc",
        text="No annual maintenance charge may be levied on a BSBDA.",
        source_url="https://rbi.org.in/example",
        verified_on="2026-08-08",
    )]


@pytest.fixture
def tier_b():
    return sources.WebSource(
        id="web_rbi_org_in_faq_page",
        url="https://rbi.org.in/faq",
        title="RBI FAQ",
        text="Banks should respond to complaints within thirty days.",
        section_ref=None,
        relevance="high",
    )


# ------------------------------------------------------- the boundary itself


def test_web_citation_is_stripped_from_the_filing(tier_a, tier_b):
    """The headline safety property. A web-sourced id must never survive into a
    document the user sends to a bank."""
    body = (
        "The charge was wrongly levied. [rbi_bsbda_no_amc] "
        "The bank must reply within thirty days. [web_rbi_org_in_faq_page]"
    )
    cleaned, stripped = validate_citations(body, tier_a)

    assert "web_rbi_org_in_faq_page" not in cleaned
    assert "web_rbi_org_in_faq_page" in stripped
    assert "rbi_bsbda_no_amc" in cleaned, "verified citations must survive"


def test_passing_web_sources_as_citable_would_break_it(tier_a, tier_b):
    """Guards the wiring, not just the function: this is what MUST NOT happen in
    drafting.build_body(). Documented as a test so the mistake is obvious."""
    body = "Claim. [web_rbi_org_in_faq_page]"

    _, stripped_correctly = validate_citations(body, tier_a)
    assert stripped_correctly == ["web_rbi_org_in_faq_page"]

    # If someone ever passes web sources into the allowed set, the id survives —
    # which is precisely the regression this file exists to catch.
    smuggled = tier_a + [Retrieved(tier_b.id, tier_b.text, tier_b.url, "unverified")]
    _, stripped_wrongly = validate_citations(body, smuggled)
    assert stripped_wrongly == [], "demonstrates why build_body passes only Tier A"


def test_build_body_never_lets_web_context_be_cited(monkeypatch, tier_b):
    """End to end: force retrieval to be thin, force the model to cite a web id,
    and assert the renderer deletes it."""
    monkeypatch.setattr(drafting, "retrieve", lambda q, limit=4: [])
    monkeypatch.setattr(sources, "context_for", lambda q, limit=3: [tier_b])
    monkeypatch.setattr(
        drafting.sarvam, "chat",
        lambda *a, **k: "The bank must reply in thirty days. [web_rbi_org_in_faq_page]",
    )

    case = {"facts": {}, "transcript": "bank ignored me", "grievance_class": "banking/x",
            "language": "en-IN"}
    out = drafting.build_body(case, None, 1)

    assert "web_rbi_org_in_faq_page" not in out["body_text"]
    assert out["grounded"] is False
    assert out["stripped"] == ["web_rbi_org_in_faq_page"]
    # ...but the source is still shown to the user as background reading.
    assert out["web_context"][0]["url"] == "https://rbi.org.in/faq"


def test_web_sources_are_never_labelled_human():
    """provenance is what a reviewer reads when deciding whether to promote."""
    assert sources.WebSource("web_x", "u", "t", "s", None, "high").provenance == "web"


# ------------------------------------------------------------- retrieval


def test_search_is_disabled_without_a_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert sources.enabled() is False
    assert sources.search("anything") == []


def test_search_failure_returns_empty_rather_than_raising(monkeypatch):
    """Context is a bonus. A Firecrawl outage must never stop a letter being drafted."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(sources, "cached",
                        lambda name, key, fn: (_ for _ in ()).throw(RuntimeError("boom")))
    assert sources.search("bank charge") == []


def test_only_government_domains_are_searched():
    """Retrieval is restricted at the API level. If a non-government host appears
    here, HAQ starts quoting blogs at banks."""
    for domain in sources.GOV_DOMAINS:
        assert domain.endswith((".gov.in", ".nic.in", ".org.in")), domain


def test_slug_always_marks_web_provenance_in_the_id():
    slug = sources._slug("https://rbi.org.in/Scripts/BS_ViewMasCirculardetails.aspx")
    assert slug.startswith(sources.WEB_ID_PREFIX)
    assert len(slug) <= len(sources.WEB_ID_PREFIX) + 48


def test_low_relevance_hits_are_dropped(monkeypatch):
    """Half a page of unrelated statute is worse than nothing in a prompt."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
    monkeypatch.setattr(sources, "search",
                        lambda q, limit=4: [{"url": "https://rbi.org.in/a", "markdown": "text"}])
    monkeypatch.setattr(sources, "extract_snippet",
                        lambda md, url, q: {"text": "x", "section_ref": None, "relevance": "low"})
    assert sources.context_for("query") == []


def test_snippet_extraction_falls_back_when_the_model_fails(monkeypatch):
    """Same discipline as agent.py: a parse error must never surface."""
    monkeypatch.setattr(sources.sarvam, "chat",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    out = sources.extract_snippet("# Heading\n\nBanks **must** reply.", "https://rbi.org.in/a", "q")
    assert out["text"]
    assert out["relevance"] == "low"
    assert "**" not in out["text"], "markdown syntax should be cleaned out"


def test_empty_markdown_yields_nothing():
    assert sources.extract_snippet("", "https://rbi.org.in/a", "q") is None
