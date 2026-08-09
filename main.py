"""FastAPI app — six endpoints, that's all (PRD 2.3).

Every route here is a thin wrapper: parse the request, call a plain function in a
module, return JSON. That rule is what makes the WhatsApp adapter ~120 lines instead
of a duplicate of the whole product.
"""

from __future__ import annotations

import base64
import os
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
import demo_cache
import drafting
import ladder_engine
import sarvam
import store
from ladder_engine import Facts, load_ladders, resolve

BASE = Path(__file__).parent
LADDERS = load_ladders()

app = FastAPI(title="HAQ", docs_url="/docs")
store.init()


# ------------------------------------------------------------------ helpers


def _case_or_404(case_id: str) -> dict:
    case = store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"unknown case {case_id}")
    return case


def _today_for(case: dict) -> date:
    """The time machine: /api/advance moves this case's clock, nothing else's."""
    return date.today() + timedelta(days=case.get("clock_offset_days") or 0)


def _ladder_for(case: dict) -> dict | None:
    facts = Facts(grievance_class=case.get("grievance_class") or "other")
    for ladder_id in sorted(LADDERS):
        spec = LADDERS[ladder_id]
        for prefix in spec.get("grievance_classes", []):
            if facts.grievance_class == prefix or facts.grievance_class.startswith(prefix + "/"):
                return spec
    return None


def required_facts_for(case: dict) -> list[str]:
    ladder = _ladder_for(case)
    return ladder.get("required_facts", []) if ladder else []


def facts_from_case(case: dict) -> Facts:
    stored = case.get("facts", {})
    return Facts(
        grievance_class=case.get("grievance_class") or "other",
        tier1_written_complaint=stored.get("tier1_written_complaint"),
        tier1_filed_on=stored.get("tier1_filed_on"),
        last_communication_on=stored.get("last_communication_on"),
        pending_in_court=bool(stored.get("pending_in_court") or False),
        amount_inr=float(stored.get("amount_inr") or 0),
        account_type=stored.get("account_type"),
        institution=stored.get("institution"),
        commercial_decision=bool(stored.get("commercial_decision") or False),
    )


def verdict_for(case: dict):
    """Resolve this case against the ladders. No LLM on this path — ever."""
    return resolve(facts_from_case(case), LADDERS, _today_for(case))


def verdict_json(verdict) -> dict:
    return {
        "ladder_id": verdict.ladder_id,
        "ladder_version": verdict.ladder_version,
        "current_tier": verdict.current_tier,
        "maintainable": verdict.maintainable,
        "blocked_by": verdict.blocked_by,
        "blocked_messages": verdict.blocked_messages,
        "redirect_to": verdict.redirect_to,
        "next_action": verdict.next_action,
        "haq_score": verdict.haq_score,
        "source_url": verdict.source_url,
        "verified_on": verdict.verified_on,
        "facts_pending": verdict.facts_pending,
        "deadlines": [
            {"id": d.id, "label": d.label, "due_on": d.due_on.isoformat(), "tier": d.tier}
            for d in verdict.deadlines
        ],
        "citations": [
            {"id": c.id, "text": c.text, "source_url": c.source_url, "verified_on": c.verified_on}
            for c in verdict.citations
        ],
    }


# ------------------------------------------------------------ 1. /api/intake


@app.post("/api/intake")
def intake(audio: UploadFile = File(...), lang: str = Form("unknown")):
    """Voice in -> transcript, language, classification. Opens the demo."""
    raw = audio.file.read()
    heard = sarvam.stt(raw, filename=audio.filename or "audio.webm", language_code=lang)

    case_id = store.create_case(channel="web")
    classified = agent.classify(heard["transcript"])

    facts = {}
    if classified.get("institution"):
        facts["institution"] = classified["institution"]

    store.update_case(
        case_id,
        transcript=heard["transcript"],
        language=heard["language_code"],
        grievance_class=classified["grievance_class"],
        facts=facts,
    )
    store.add_message(case_id, "user", heard["transcript"])

    return {
        "case_id": case_id,
        "transcript": heard["transcript"],
        "language": heard["language_code"],
        "classification": classified["grievance_class"],
        "confidence": heard["confidence"],
    }


class TextIntake(BaseModel):
    text: str
    language: str | None = None


@app.post("/api/intake_text")
def intake_text(payload: TextIntake):
    """Same as /api/intake but for typed text — used by WhatsApp and by anyone
    testing without a microphone."""
    case_id = store.create_case(channel="web")
    classified = agent.classify(payload.text)
    store.update_case(
        case_id,
        transcript=payload.text,
        language=payload.language or "en-IN",
        grievance_class=classified["grievance_class"],
        facts={"institution": classified["institution"]} if classified.get("institution") else {},
    )
    store.add_message(case_id, "user", payload.text)
    return {
        "case_id": case_id,
        "transcript": payload.text,
        "language": payload.language or "en-IN",
        "classification": classified["grievance_class"],
        "confidence": None,
    }


# -------------------------------------------------------------- 2. /api/turn


class Turn(BaseModel):
    case_id: str
    text: str | None = None
    asking_for: str | None = None


@app.post("/api/turn")
def turn(payload: Turn):
    """One question per turn, filling the fact sheet. STATE: COLLECTING."""
    case = _case_or_404(payload.case_id)
    required = required_facts_for(case)

    if payload.text and payload.asking_for:
        agent.record_answer(payload.case_id, payload.asking_for, payload.text)
        case = _case_or_404(payload.case_id)

    return agent.next_question(payload.case_id, required)


# ----------------------------------------------------------- 3. /api/resolve


class CaseRef(BaseModel):
    case_id: str


@app.post("/api/resolve")
def resolve_case(payload: CaseRef):
    """The refusal lives here. Pure function, no LLM. STATE: RESOLVING."""
    case = _case_or_404(payload.case_id)
    verdict = verdict_for(case)
    store.save_deadlines(payload.case_id, verdict.deadlines)

    out = verdict_json(verdict)
    if not verdict.maintainable and verdict.blocked_by:
        # The model only phrases a refusal the engine already decided.
        out["refusal_native"] = agent.phrase_refusal(verdict, case.get("language"))
        store.add_event(payload.case_id, "refusal", ", ".join(verdict.blocked_by))
    return out


# ------------------------------------------------------------- 4. /api/draft


class DraftRequest(BaseModel):
    case_id: str
    tier: int | None = None


@app.post("/api/draft")
def draft(payload: DraftRequest):
    """template -> 105B -> validate_citations -> Mayura -> PDF. STATE: DRAFTING."""
    case = _case_or_404(payload.case_id)
    verdict = verdict_for(case)
    tier = payload.tier or verdict.current_tier

    built = drafting.build_body(case, verdict, tier)

    ladder = _ladder_for(case) or {}
    tier_name = next(
        (t["name"] for t in ladder.get("tiers", []) if t["tier"] == tier),
        f"tier {tier}",
    )

    drafting.build_pdf(
        payload.case_id, built["body_text"], built["citations"], tier_name, _today_for(case)
    )
    store.update_case(payload.case_id, draft_text=built["body_text"])
    store.add_event(payload.case_id, "draft", f"tier {tier}")

    return {
        "body_text": built["body_text"],
        "citations": built["citations"],
        "stripped_citations": built["stripped"],
        "grounded": built["grounded"],
        "tier": tier,
        "tier_name": tier_name,
        "pdf_url": f"/api/draft/{payload.case_id}.pdf",
    }


@app.get("/api/draft/{case_id}.pdf")
def draft_pdf(case_id: str):
    path = drafting.PDF_DIR / f"{case_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="draft not generated yet")
    return FileResponse(path, media_type="application/pdf", filename=f"haq-{case_id}.pdf")


# ------------------------------------------------------------- 5. /api/speak


class Speak(BaseModel):
    text: str
    lang: str = "mr-IN"
    speaker: str = "ritu"


@app.post("/api/speak")
def speak(payload: Speak):
    """Bulbul reads the English draft aloud in her language. She approves a document
    she cannot read, by hearing it in one she can."""
    audio_base64 = sarvam.tts(payload.text, language_code=payload.lang, speaker=payload.speaker)
    return {"audio_base64": audio_base64}


class Approve(BaseModel):
    case_id: str


@app.post("/api/approve")
def approve(payload: Approve):
    """We never file autonomously. Every outbound document needs a recorded human
    approval — this endpoint is that record (Q&A answer 3)."""
    _case_or_404(payload.case_id)
    from datetime import datetime

    stamp = datetime.utcnow().isoformat(timespec="seconds")
    store.update_case(payload.case_id, approved_at=stamp)
    store.add_event(payload.case_id, "approved", stamp)
    return {"approved_at": stamp}


# ----------------------------------------------------------- 6. /api/advance


class Advance(BaseModel):
    case_id: str
    days: int = 31


@app.post("/api/advance")
def advance(payload: Advance):
    """The time machine. Nobody clicked anything — silence became a legal event.
    STATE: WATCHING."""
    case = _case_or_404(payload.case_id)
    offset = (case.get("clock_offset_days") or 0) + payload.days
    store.update_case(payload.case_id, clock_offset_days=offset)

    case = _case_or_404(payload.case_id)
    today = _today_for(case)

    verdict = verdict_for(case)
    store.save_deadlines(payload.case_id, verdict.deadlines)
    fired = store.due_deadlines(payload.case_id, today)

    new_messages = []
    for item in fired:
        text = f"{item['label']} — this deadline has now arrived."
        store.add_event(payload.case_id, "deadline_fired", item["deadline_id"])
        store.add_message(payload.case_id, "assistant", text)
        new_messages.append({"deadline_id": item["deadline_id"], "text": text})

    return {
        "today": today.isoformat(),
        "clock_offset_days": offset,
        "events": store.events(payload.case_id),
        "new_messages": new_messages,
        "new_deadlines": store.all_deadlines(payload.case_id),
        "verdict": verdict_json(verdict),
    }


# --------------------------------------------------------------- case + health


@app.get("/api/case/{case_id}")
def get_case(case_id: str):
    case = _case_or_404(case_id)
    verdict = verdict_for(case)
    return {
        "case": {k: v for k, v in case.items() if k != "facts"},
        "facts": case["facts"],
        "messages": store.messages(case_id),
        "deadlines": store.all_deadlines(case_id),
        "events": store.events(case_id),
        "verdict": verdict_json(verdict),
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "mode": demo_cache.mode(),
        "ladders": sorted(LADDERS),
        "cached_calls": len(demo_cache.recorded_keys()),
        "whatsapp": os.getenv("CHANNEL_WHATSAPP") == "1",
        "sarvam": sarvam.health(),
    }


# ------------------------------------------------------- optional channels

# Disabled by default. With CHANNEL_WHATSAPP unset the module is never imported,
# so a broken WhatsApp file cannot take the demo down. This is the PRD's "cut #1"
# reduced to one character.
#
# Must be registered BEFORE the static mount: mounting at "/" is greedy and would
# otherwise swallow /webhook and hand Meta a 404.
if os.getenv("CHANNEL_WHATSAPP") == "1":
    from whatsapp import router as whatsapp_router

    app.include_router(whatsapp_router)


# ------------------------------------------------------------------- static

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
