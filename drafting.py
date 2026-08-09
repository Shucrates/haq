"""Document generation + the grounding validator.

The validator is the line that makes the safety story real (PRD 3.5):

    "This is code, not a prompt. The model literally cannot emit an ungrounded
     section number, because the renderer deletes it."

Retrieval is keyword overlap over ~12 hand-verified statute snippets. No embeddings,
no vector DB, no RAG pipeline. At this scale it is more accurate and infinitely more
debuggable — and that is a defensible engineering choice, not a shortcut.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fpdf import FPDF

import sarvam
import sources

DATA = Path(__file__).parent / "data"
PDF_DIR = DATA / "drafts"
STATUTES = json.loads((DATA / "statutes.json").read_text())

# Below this many verified snippets, reach for Tier B context.
MIN_CITABLE_BEFORE_WEB = 2

DRAFT_PROMPT = """You draft formal complaint letters for Indian consumers.

Write ONLY the body of the letter — no address block, no date, no subject line, no
signature. Three or four short paragraphs, plain factual tone, first person.

You will be given two blocks of material and they are NOT equivalent:

  CITABLE — human-verified. Cite these by id in square brackets, like
            [rbi_bsbda_no_amc], immediately after the sentence they support.

  CONTEXT — retrieved from government websites but NOT verified. It may inform how
            you phrase things. You must NEVER cite it, quote a section number from
            it, or present it as settled law.

Never state a rule, section number or scheme name that is not in CITABLE. If you
have no citable snippet for a point, make the point without a citation."""


@dataclass(frozen=True)
class Retrieved:
    id: str
    text: str
    source_url: str
    verified_on: str


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def retrieve(query: str, limit: int = 4) -> list[Retrieved]:
    """Keyword overlap scoring. Ten lines, zero embeddings (PRD 3.5)."""
    query_tokens = _tokens(query)
    scored = []
    for entry in STATUTES:
        keyword_tokens = _tokens(" ".join(entry["keywords"]))
        score = len(query_tokens & keyword_tokens)
        # A keyword phrase appearing verbatim is worth more than a token overlap.
        for phrase in entry["keywords"]:
            if phrase in query.lower():
                score += 2
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [
        Retrieved(e["id"], e["text"], e["source_url"], e["verified_on"])
        for _, e in scored[:limit]
    ]


CITATION_MARKER = re.compile(r"\[([a-z0-9_]+)\]")

# A numbered legal reference: "Section 7(1)", "Regulation 12", "Clause 4.2", "Act 2005".
# The number is what makes a claim falsifiable, and what a judge checks.
LEGAL_CLAIM = re.compile(
    r"\b(section|regulation|clause|rule|article|paragraph|act|scheme)\b[^.]{0,40}?\d",
    re.I,
)
# A sentence, plus any citation markers trailing it. DRAFT_PROMPT asks for the
# marker *after* the sentence it supports, so it sits past the full stop — splitting
# on punctuation alone would orphan the marker and delete the sentence that earned it.
SENTENCE = re.compile(r"[^.!?]+[.!?]*(?:[ \t]*\[[a-z0-9_]+\])*")


def validate_citations(body: str, retrieved: list[Retrieved]) -> tuple[str, list[str], list[str]]:
    """Strip any citation lacking a source_url retrieved this session, then strip any
    sentence that states a numbered legal rule without one.

    Two passes, because the first alone was not the guarantee it was sold as. It
    deleted `[unknown_id]` markers, so the model could not attribute a claim to an
    unverified source — but a sentence reading "Section 7(1) of the RTI Act requires
    disposal within thirty days", with no bracket at all, passed through untouched
    and still reported grounded=True. The marker was never there to strip.

    Returns (cleaned, stripped_markers, stripped_claims).
    """
    allowed = {r.id for r in retrieved if r.source_url}
    stripped: list[str] = []

    def replace(match: re.Match) -> str:
        cited = match.group(1)
        if cited in allowed:
            return match.group(0)
        stripped.append(cited)
        return ""

    cleaned = CITATION_MARKER.sub(replace, body)

    # Any marker still standing after the first pass is an allowed one, so a sentence
    # carrying one is grounded by definition. Line structure is preserved because
    # build_pdf splits the body on newlines to lay out paragraphs.
    stripped_claims: list[str] = []
    kept_lines: list[str] = []
    for line in cleaned.split("\n"):
        kept = []
        for sentence in SENTENCE.findall(line):
            if LEGAL_CLAIM.search(sentence) and not CITATION_MARKER.search(sentence):
                stripped_claims.append(sentence.strip())
                continue
            kept.append(sentence)
        kept_lines.append("".join(kept))
    cleaned = "\n".join(kept_lines)

    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;])", r"\1", cleaned)
    return cleaned.strip(), stripped, stripped_claims


def build_body(case: dict, verdict, tier: int) -> dict:
    """Retrieve -> draft -> validate -> translate. Returns the pieces the API needs."""
    facts = case.get("facts", {})
    query = " ".join(
        str(v) for v in [case.get("transcript"), case.get("grievance_class"), facts.get("institution")] if v
    )
    retrieved = retrieve(query)

    # Tier B only when the verified corpus is thin. This keeps the common path
    # offline and fast, and means the rehearsed demo case never hits the network.
    web_sources: list = []
    if len(retrieved) < MIN_CITABLE_BEFORE_WEB:
        web_sources = sources.context_for(query)

    snippets = "\n".join(f"[{r.id}] {r.text} (source: {r.source_url})" for r in retrieved)
    web_block = "\n".join(
        f"- {w.text} (unverified, from {w.url})" for w in web_sources
    )
    context = (
        f"Grievance: {case.get('grievance_class')}\n"
        f"Institution: {facts.get('institution') or 'the bank'}\n"
        f"Amount in rupees: {facts.get('amount_inr') or 'not stated'}\n"
        f"Written complaint already made: {facts.get('tier1_written_complaint')}\n"
        f"Date of that complaint: {facts.get('tier1_filed_on') or 'not stated'}\n"
        f"Last response from the institution: {facts.get('last_communication_on') or 'none'}\n"
        f"Filing at tier {tier}.\n"
        f"What the person said: {case.get('transcript') or ''}\n\n"
        f"CITABLE (human-verified — cite these by id):\n{snippets or '(none)'}\n\n"
        f"CONTEXT (unverified, never cite):\n{web_block or '(none)'}"
    )

    try:
        raw = sarvam.chat(
            [{"role": "system", "content": DRAFT_PROMPT}, {"role": "user", "content": context}],
            max_tokens=900,
            reasoning_effort="medium",
        )
    except Exception:
        raw = _fallback_body(case, facts, retrieved)

    # NOTE: `retrieved` only — web_sources are deliberately NOT passed here, so any
    # Tier B id the model emits falls outside the allowed set and gets deleted.
    grounded_body, stripped, stripped_claims = validate_citations(raw, retrieved)

    # If the validator took everything, send the template rather than a blank page.
    # The fallback is our own text plus the user's own words plus Tier A snippets
    # that carry their markers — it contains no model-invented law by construction.
    if not grounded_body.strip():
        grounded_body = _fallback_body(case, facts, retrieved)

    # The filing goes out in formal English regardless of the language spoken.
    if case.get("language") and not str(case["language"]).startswith("en"):
        try:
            grounded_body = sarvam.translate(grounded_body, target="en-IN", source="auto")
        except Exception:
            pass  # keep the English the model already produced

    return {
        "body_text": grounded_body,
        "citations": [
            {"id": r.id, "text": r.text, "source_url": r.source_url, "verified_on": r.verified_on}
            for r in retrieved
        ],
        "stripped": stripped,
        "stripped_claims": stripped_claims,
        "grounded": not stripped and not stripped_claims,
        # Shown to the user as background reading, clearly separated from citations.
        "web_context": [w.as_dict() for w in web_sources],
    }


def _fallback_body(case: dict, facts: dict, retrieved: list[Retrieved]) -> str:
    """If 105B is unreachable, still produce a filable letter. Never show an error
    where a document should be."""
    institution = facts.get("institution") or "the bank"
    amount = facts.get("amount_inr")
    lines = [
        f"I am writing regarding a grievance against {institution}.",
        case.get("transcript") or "",
    ]
    if amount:
        lines.append(f"The amount in dispute is Rs. {amount:,.0f}.")
    for r in retrieved[:2]:
        lines.append(f"{r.text} [{r.id}]")
    lines.append("I request that this be examined and the amount reversed.")
    return "\n\n".join(line for line in lines if line.strip())


# ------------------------------------------------------------------- the PDF


HEADER = "COMPLAINT"


def _latin(text: str) -> str:
    """Core PDF fonts are latin-1 only. Filings are English-only by design (PRD 11),
    so this is a guard, not a limitation — it stops one stray Devanagari character
    from rendering the whole document as boxes."""
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(case_id: str, body: str, citations: list[dict], tier_name: str,
              today: date | None = None) -> Path:
    today = today or date.today()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{case_id}.pdf"

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(20, 18, 20)

    pdf.set_font("helvetica", "B", 15)
    pdf.cell(0, 10, _latin(HEADER), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, _latin(f"To: {tier_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _latin(f"Date: {today.isoformat()}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, _latin(f"Reference: HAQ/{case_id}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("helvetica", "", 11)
    for paragraph in body.split("\n"):
        if paragraph.strip():
            # new_x must be LMARGIN: multi_cell otherwise leaves the cursor at the
            # right edge, so the next cell gets zero width and fpdf raises.
            pdf.multi_cell(0, 6, _latin(paragraph.strip()), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    if citations:
        pdf.ln(4)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, "Sources relied on", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 8)
        for c in citations:
            pdf.multi_cell(
                0, 4,
                _latin(f"[{c['id']}] {c['source_url']} (verified {c['verified_on']})"),
                new_x="LMARGIN", new_y="NEXT",
            )

    pdf.output(str(path))
    return path
