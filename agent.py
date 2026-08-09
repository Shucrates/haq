"""The interrogation agent — a state machine with an LLM inside it (PRD 4.3).

Four states, no autonomous planning, hard cap of six turns:

    COLLECTING  105B gets (fact_sheet, ladder_requirements) and returns ONE question
    RESOLVING   ladder_engine.resolve() — no LLM involved
    DRAFTING    template + statutes + facts -> 105B fills the body
    WATCHING    deadlines in SQLite, /api/advance simulates the clock

Deterministic where it matters, generative only where language is the output.
Every LLM call has a rule-based fallback, because "105B returns malformed JSON" is
a listed risk (PRD 11) and a parse error must never reach the user.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime

import onboarding
import sarvam
import store

MAX_TURNS = 6

# The facts we chase, in order. The ladder decides which are REQUIRED; these are
# the ones we know how to ask for in plain language.
# A required fact missing from this list is never asked — pending_facts() filters
# `required` through it — so the interrogation would report done with the fact still
# pending and the case could never resolve. Keep it a superset of every fact any
# ladder can require (ladder_engine.RULE_FACTS included).
ASK_ORDER = [
    "tier1_written_complaint",
    "last_communication_on",
    "tier1_filed_on",
    "amount_inr",
    "institution",
    "pending_in_court",
    "commercial_decision",
]

# Rule-based fallback questions. If the model is unavailable or returns junk, the
# interrogation still works — it just sounds less natural.
FALLBACK_QUESTIONS = {
    "tier1_written_complaint": "Have you already made a written complaint to the bank about this?",
    "last_communication_on": "When did the bank last respond to you? A rough date is fine.",
    "tier1_filed_on": "On what date did you send that written complaint?",
    "amount_inr": "How much money is involved, in rupees?",
    "institution": "Which bank or institution is this about?",
    "pending_in_court": "Is this matter already in a court, tribunal or arbitration?",
    "commercial_decision": (
        "Is this about a business decision the bank made — refusing you a loan, "
        "or closing an account it is allowed to close?"
    ),
}

FACT_TYPES = {
    "tier1_written_complaint": "boolean",
    "pending_in_court": "boolean",
    "commercial_decision": "boolean",
    "last_communication_on": "date",
    "tier1_filed_on": "date",
    "amount_inr": "number",
    "institution": "text",
}

CLASSIFY_PROMPT = """You classify Indian consumer grievances. Reply with ONLY a JSON object:
{"grievance_class": "<one of: banking/unauthorised_charge, banking/unauthorised_debit, banking/service_deficiency, rti/no_response, other>", "institution": "<name or null>"}
No prose, no markdown fence."""

QUESTION_PROMPT = """You are helping a person in India escalate a grievance. You already know
some facts and need exactly ONE more.

Ask a single short question, in the person's language, for the fact named below.
Rules: one question only, under 25 words, no preamble, no explanation, no lists.
Write it in that language's own script — never transliterated into Latin letters.
Reply with ONLY the question text."""

REFUSAL_PROMPT = """Explain, kindly and in under 45 words, why this complaint cannot be filed
yet and what the person must do first. Do not invent any rule or section number — use only the
reason given. Reply with only the explanation."""


# ------------------------------------------------------------------ helpers


def _first_json(text: str) -> dict:
    """105B occasionally wraps JSON in prose or a fence. Pull the first object out."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in model output")
    return json.loads(match.group(0))


def _language_name(code: str | None) -> str:
    """Read off the offered list rather than a second copy of it — a local map went
    stale the moment onboarding added Malayalam, Punjabi and Odia, and the only
    symptom was those three quietly getting their questions in English."""
    language = onboarding.by_code(code)
    return language.english if language else "the same language as the user"


# --------------------------------------------------------------- classify


def classify(transcript: str) -> dict:
    """Grievance class + institution. Falls back to keyword rules on any failure."""
    try:
        raw = sarvam.chat(
            [
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=120,
        )
        out = _first_json(raw)
        grievance = out.get("grievance_class") or "other"
        return {"grievance_class": grievance, "institution": out.get("institution")}
    except Exception:
        return _classify_by_keyword(transcript)


def _classify_by_keyword(transcript: str) -> dict:
    lowered = transcript.lower()
    if any(w in lowered for w in ["rti", "information", "माहिती"]):
        grievance = "rti/no_response"
    elif any(w in lowered for w in ["charge", "fee", "amc", "कापले", "पैसे", "कटौती"]):
        grievance = "banking/unauthorised_charge"
    elif any(w in lowered for w in ["bank", "account", "बँक", "खाते"]):
        grievance = "banking/service_deficiency"
    else:
        grievance = "other"
    return {"grievance_class": grievance, "institution": None}


# -------------------------------------------------------------- COLLECTING


def _applicable(name: str, facts: dict) -> bool:
    """Some facts only make sense in light of others. Never ask when a written
    complaint was filed after the person has just told you they never filed one —
    that is the difference between an interrogation and a form."""
    if name == "tier1_filed_on" and facts.get("tier1_written_complaint") is False:
        return False
    return True


def missing_required(facts: dict, required: list[str]) -> list[str]:
    """Required facts the ladder still needs, skipping ones made moot by answers
    already given. If this never empties, the interrogation never ends."""
    return [f for f in required if facts.get(f) is None and _applicable(f, facts)]


def pending_facts(facts: dict, required: list[str]) -> list[str]:
    """Required facts first, then the merely useful ones, in a stable order."""
    ordered = [f for f in required if f in ASK_ORDER] + [f for f in ASK_ORDER if f not in required]
    seen: set[str] = set()
    out = []
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        if facts.get(name) is None and _applicable(name, facts):
            out.append(name)
    return out


def next_question(case_id: str, required: list[str]) -> dict:
    """One question per turn. Returns done=True when the ladder has what it needs."""
    case = store.get_case(case_id)
    if case is None:
        raise KeyError(case_id)

    facts = case["facts"]
    still_needed = missing_required(facts, required)
    pending = pending_facts(facts, required)

    # Stop as soon as the ladder is satisfied, or at the hard cap.
    if not still_needed or case["turns"] >= MAX_TURNS or not pending:
        return {
            "question": None,
            "question_native": None,
            "facts_known": facts,
            "facts_pending": still_needed,
            "done": True,
        }

    target = pending[0]
    english = FALLBACK_QUESTIONS[target]
    native = english

    try:
        native = sarvam.chat(
            [
                {"role": "system", "content": QUESTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Known facts: {json.dumps(facts, default=str)}\n"
                        f"Fact needed: {target} ({FACT_TYPES[target]})\n"
                        f"Ask in: {_language_name(case['language'])}\n"
                        f"English version for reference: {english}"
                    ),
                },
            ],
            max_tokens=80,
        ).strip() or english
    except Exception:
        native = english  # never let a model failure stall the interrogation

    store.add_message(case_id, "assistant", english, native)
    return {
        "question": english,
        "question_native": native,
        "asking_for": target,
        "facts_known": facts,
        "facts_pending": still_needed,
        "done": False,
    }


def record_answer(case_id: str, target: str, text: str) -> dict:
    """Parse one answer into the fact sheet. Rule-based first — a date or a yes/no
    does not need a language model — with the LLM only as a fallback."""
    case = store.get_case(case_id)
    if case is None:
        raise KeyError(case_id)

    facts = dict(case["facts"])
    value = _parse_value(target, text)

    if value is None:
        value = _parse_with_model(target, text)

    if value is not None:
        facts[target] = value

    store.update_case(case_id, facts=facts)
    store.add_message(case_id, "user", text)
    store.bump_turns(case_id)
    return facts


_YES = ["yes", "yeah", "yep", "haan", "हो", "हाँ", "हा", "ho", "kela", "केला", "done", "did"]
_NO = ["no", "nope", "nahi", "नाही", "नहीं", "never", "not yet", "nai"]

_DATE_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), lambda m: date(int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})"), lambda m: date(int(m[3]), int(m[2]), int(m[1]))),
]


def _parse_value(target: str, text: str):
    kind = FACT_TYPES[target]
    lowered = text.lower().strip()

    if kind == "boolean":
        if any(w in lowered for w in _NO):
            return False
        if any(w in lowered for w in _YES):
            return True
        return None

    if kind == "date":
        for pattern, build in _DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return build(match).isoformat()
                except ValueError:
                    return None
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(text.strip(), fmt).date().isoformat()
            except ValueError:
                continue
        return None

    if kind == "number":
        match = re.search(r"(\d[\d,]*\.?\d*)", text.replace(" ", ""))
        return float(match.group(1).replace(",", "")) if match else None

    return text.strip() or None


def _parse_with_model(target: str, text: str):
    """Last resort: ask 105B to normalise. Still validated by _parse_value."""
    kind = FACT_TYPES[target]
    instruction = {
        "boolean": 'Reply with only {"value": true} or {"value": false}.',
        "date": 'Reply with only {"value": "YYYY-MM-DD"}. Today is ' + date.today().isoformat() + ".",
        "number": 'Reply with only {"value": <number>}.',
        "text": 'Reply with only {"value": "<short text>"}.',
    }[kind]

    try:
        raw = sarvam.chat(
            [
                {"role": "system", "content": f"Extract {target} from the user's reply. {instruction}"},
                {"role": "user", "content": text},
            ],
            max_tokens=60,
        )
        value = _first_json(raw).get("value")
    except Exception:
        return None

    if kind == "boolean":
        return bool(value) if isinstance(value, bool) else None
    if kind == "date":
        try:
            return date.fromisoformat(str(value)).isoformat()
        except ValueError:
            return None
    if kind == "number":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return str(value) if value else None


# --------------------------------------------------------------- RESOLVING


def phrase_refusal(verdict, language_code: str | None) -> str:
    """The refusal is decided by ladder_engine, never by the model. The model only
    puts the already-decided reason into the user's language."""
    reason = " ".join(verdict.blocked_messages) or verdict.next_action
    try:
        return sarvam.chat(
            [
                {"role": "system", "content": REFUSAL_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Reason (do not add to it): {reason}\n"
                        f"Reply in: {_language_name(language_code)}"
                    ),
                },
            ],
            max_tokens=160,
        ).strip() or reason
    except Exception:
        return reason
