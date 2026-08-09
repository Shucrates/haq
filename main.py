"""FastAPI app — six endpoints, that's all (PRD 2.3).

Every route here is a thin wrapper: parse the request, call a plain function in a
module, return JSON. That rule is what makes the WhatsApp adapter ~120 lines instead
of a duplicate of the whole product.
"""

from __future__ import annotations

import base64
import os
import re
import secrets
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agent
import demo_cache
import documents
import drafting
import ladder_engine
import onboarding
import sarvam
import store
from ladder_engine import Facts, load_ladders, resolve

BASE = Path(__file__).parent
LADDERS = load_ladders()

app = FastAPI(title="HAQ", docs_url="/docs")
store.init()


# ------------------------------------------------------------------ helpers


COOKIE_NAME = "haq_sid"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # a legal case outlives a browser session

# case_id is uuid4().hex[:12] (store.create_case). Anything else cannot name a case,
# and rejecting it early keeps unvalidated input out of the PDF path lookup.
CASE_ID = re.compile(r"[0-9a-f]{12}")


def _session_token(request: Request) -> str:
    """This browser's token, minted on first contact. One token per browser, not per
    case — a browser owns every case it opened."""
    return request.cookies.get(COOKIE_NAME) or secrets.token_urlsafe(32)


def _set_session_cookie(response, token: str, request: Request) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        max_age=COOKIE_MAX_AGE,
    )


def _owned_case(case_id: str, request: Request) -> dict:
    """The case this caller is allowed to touch, or 404.

    404 and not 403, deliberately: a 403 confirms the id names a real case, which
    turns a blind guess into a probe. Missing and forbidden must look identical.

    Fails closed on a NULL owner_token — a case from before ownership existed is
    unreachable rather than public.
    """
    token = request.cookies.get(COOKIE_NAME)
    case = store.get_case(case_id) if CASE_ID.fullmatch(case_id or "") else None
    if case is None or not token or case.get("owner_token") != token:
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
    return ladder_engine.required_facts(ladder) if ladder else []


def _tri_state(value) -> bool | None:
    """Pass an unanswered boolean through as None. `bool(x or False)` turned every
    unasked exclusion into a "no", which is the permissive answer."""
    return None if value is None else bool(value)


def facts_from_case(case: dict) -> Facts:
    stored = case.get("facts", {})
    return Facts(
        grievance_class=case.get("grievance_class") or "other",
        tier1_written_complaint=stored.get("tier1_written_complaint"),
        tier1_filed_on=stored.get("tier1_filed_on"),
        last_communication_on=stored.get("last_communication_on"),
        pending_in_court=_tri_state(stored.get("pending_in_court")),
        amount_inr=float(stored.get("amount_inr") or 0),
        account_type=stored.get("account_type"),
        institution=stored.get("institution"),
        commercial_decision=_tri_state(stored.get("commercial_decision")),
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
def intake(request: Request, response: Response, audio: UploadFile = File(...),
           lang: str = Form("unknown"), case_id: str = Form(None)):
    """Voice in -> transcript, language, classification. Opens the demo.

    The chosen language wins; Saaras' detection is kept as a cross-check and
    surfaced when the two disagree, rather than silently overriding the choice.
    """
    token = _session_token(request)
    _set_session_cookie(response, token, request)

    case = _owned_case(case_id, request) if case_id else None
    chosen = (case or {}).get("language")

    raw = audio.file.read()
    heard = sarvam.stt(
        raw,
        filename=audio.filename or "audio.webm",
        language_code=chosen if chosen and lang == "unknown" else lang,
    )

    if case_id is None:
        case_id = store.create_case(channel="web", owner_token=token)

    classified = agent.classify(heard["transcript"])

    facts = {}
    if classified.get("institution"):
        facts["institution"] = classified["institution"]

    language = chosen or heard["language_code"]
    store.update_case(
        case_id,
        transcript=heard["transcript"],
        language=language,
        grievance_class=classified["grievance_class"],
        facts=facts,
    )
    store.add_message(case_id, "user", heard["transcript"])

    return {
        "case_id": case_id,
        "transcript": heard["transcript"],
        "language": language,
        "detected_language": heard["language_code"],
        "language_mismatch": bool(chosen and heard["language_code"] not in (chosen, "unknown")),
        "classification": classified["grievance_class"],
        "confidence": heard["confidence"],
    }


class TextIntake(BaseModel):
    text: str
    language: str | None = None
    case_id: str | None = None


@app.post("/api/intake_text")
def intake_text(payload: TextIntake, request: Request, response: Response):
    """Same as /api/intake but for typed text — used by WhatsApp and by anyone
    testing without a microphone.

    A language must already be chosen. Passing one inline onboards the case in the
    same call, which is what the tests and the WhatsApp adapter do.
    """
    token = _session_token(request)
    _set_session_cookie(response, token, request)

    if payload.case_id:
        case = _owned_case(payload.case_id, request)
        case_id = payload.case_id
    else:
        case_id = store.create_case(channel="web", owner_token=token)
        case = store.get_case(case_id)

    language = payload.language or case.get("language")
    if not language:
        # Never guess. An unonboarded case gets the picker, not an English default.
        picker = JSONResponse(
            status_code=409,
            content={"needs_language": True, "case_id": case_id,
                     "languages": onboarding.as_json()},
        )
        # Returning a Response directly bypasses the injected `response`, so the
        # cookie has to be set here or the case we just minted is unreachable.
        _set_session_cookie(picker, token, request)
        return picker

    classified = agent.classify(payload.text)
    store.update_case(
        case_id,
        transcript=payload.text,
        language=language,
        grievance_class=classified["grievance_class"],
        facts={"institution": classified["institution"]} if classified.get("institution") else {},
    )
    store.add_message(case_id, "user", payload.text)
    return {
        "case_id": case_id,
        "transcript": payload.text,
        "language": language,
        "classification": classified["grievance_class"],
        "confidence": None,
    }


# ------------------------------------------------------------- 0. onboarding


@app.get("/api/languages")
def languages():
    """Served so the UI never hardcodes a language list. These are exactly the
    languages Bulbul can speak, because the read-back must never fail."""
    return {"prefill": onboarding.PREFILL_TEXT, "languages": onboarding.as_json()}


@app.post("/api/start")
def start(request: Request, response: Response):
    """First contact. Creates a case with no language yet — the picker comes next."""
    token = _session_token(request)
    _set_session_cookie(response, token, request)
    case_id = store.create_case(channel="web", owner_token=token)
    return {
        "case_id": case_id,
        "prefill": onboarding.PREFILL_TEXT,
        "languages": onboarding.as_json(),
    }


class Onboard(BaseModel):
    case_id: str
    language: str


@app.post("/api/onboard")
def onboard(payload: Onboard, request: Request):
    """Record the chosen language and hand back the first prompt in it."""
    _owned_case(payload.case_id, request)
    if not onboarding.is_supported(payload.language):
        raise HTTPException(status_code=400, detail=f"unsupported language {payload.language}")
    store.update_case(payload.case_id, language=payload.language)
    store.add_event(payload.case_id, "language", payload.language)
    return {
        "case_id": payload.case_id,
        "language": payload.language,
        "greeting": onboarding.greeting(payload.language),
    }


# -------------------------------------------------------------- 2. /api/turn


class Turn(BaseModel):
    case_id: str
    text: str | None = None
    asking_for: str | None = None


@app.post("/api/turn")
def turn(payload: Turn, request: Request):
    """One question per turn, filling the fact sheet. STATE: COLLECTING."""
    case = _owned_case(payload.case_id, request)
    required = required_facts_for(case)

    if payload.text and payload.asking_for:
        agent.record_answer(payload.case_id, payload.asking_for, payload.text)
        case = _owned_case(payload.case_id, request)

    return agent.next_question(payload.case_id, required)


# ----------------------------------------------------------- 3. /api/resolve


class CaseRef(BaseModel):
    case_id: str


@app.post("/api/resolve")
def resolve_case(payload: CaseRef, request: Request):
    """The refusal lives here. Pure function, no LLM. STATE: RESOLVING."""
    case = _owned_case(payload.case_id, request)
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
def draft(payload: DraftRequest, request: Request):
    """template -> 105B -> validate_citations -> Mayura -> PDF. STATE: DRAFTING.

    The refusal is ENFORCED here, not merely reported by /api/resolve. A blocked case
    that still walks away with a filable PDF is the product's central claim broken —
    and /api/resolve is advisory, so nothing stopped a client from skipping it.
    """
    case = _owned_case(payload.case_id, request)
    verdict = verdict_for(case)

    if not verdict.maintainable:
        store.add_event(
            payload.case_id,
            "draft_blocked",
            ", ".join(verdict.blocked_by) or ", ".join(verdict.facts_pending) or "not maintainable",
        )
        out = verdict_json(verdict)
        out["refusal_native"] = agent.phrase_refusal(verdict, case.get("language"))
        # jsonable_encoder, not the raw dict: PyYAML parses `verified_on: 2026-08-08`
        # into a date, and JSONResponse (unlike a returned dict) will not encode one.
        return JSONResponse(status_code=409, content=jsonable_encoder(out))

    # A maintainable verdict always came from a ladder, so `tiers` is never empty here.
    ladder = _ladder_for(case) or {}
    tiers = {t["tier"]: t["name"] for t in ladder.get("tiers", [])}
    tier = verdict.current_tier if payload.tier is None else payload.tier
    if tier not in tiers:
        raise HTTPException(
            status_code=400,
            detail=f"tier {tier} does not exist in ladder {verdict.ladder_id}",
        )
    tier_name = tiers[tier]

    built = drafting.build_body(case, verdict, tier)

    drafting.build_pdf(
        payload.case_id, built["body_text"], built["citations"], tier_name, _today_for(case)
    )
    store.update_case(payload.case_id, draft_text=built["body_text"])
    store.add_event(payload.case_id, "draft", f"tier {tier}")

    # Tier B is recorded against the case but kept out of the PDF and out of the
    # citation list. It is shown to the user as background reading, clearly labelled
    # unverified, and a human can later promote it into statutes.json.
    if built.get("web_context"):
        store.save_sources(payload.case_id, built["web_context"], case.get("transcript") or "")

    return {
        "body_text": built["body_text"],
        "citations": built["citations"],
        "stripped_citations": built["stripped"],
        "grounded": built["grounded"],
        "web_context": built.get("web_context", []),
        "tier": tier,
        "tier_name": tier_name,
        "pdf_url": f"/api/draft/{payload.case_id}.pdf",
    }


@app.post("/api/document")
def upload_document(request: Request, document: UploadFile = File(...),
                    case_id: str = Form(...)):
    """Read a bank letter or RTI reply with Sarvam Doc AI and merge what it says
    into the fact sheet — but only the fields it is confident about. Anything
    below the threshold comes back for the user to confirm, because a wrong date
    here silently changes which tier a complaint is maintainable at."""
    case = _owned_case(case_id, request)
    raw = document.file.read()

    try:
        extracted = documents.extract(raw, document.filename or "document.pdf",
                                      case.get("language"))
    except Exception as exc:  # noqa: BLE001 — a bad scan must not close the case
        store.save_document(case_id, document.filename, "failed", {"error": str(exc)})
        return {"status": "failed", "facts": {}, "confirm": [],
                "message": "I could not read that document. You can tell me the details instead."}

    split = documents.to_facts(extracted)
    store.save_document(case_id, document.filename, extracted.get("status", "unknown"), split["raw"])

    if split["facts"]:
        facts = {**case["facts"], **split["facts"]}
        store.update_case(case_id, facts=facts)
        store.add_event(case_id, "document", ", ".join(split["facts"]))

    return {
        "status": extracted.get("status"),
        "facts": split["facts"],
        "confirm": split["confirm"],
        "summary": documents.summarise(extracted),
    }


@app.get("/api/draft/{case_id}.pdf")
def draft_pdf(case_id: str, request: Request):
    """The filing itself. _owned_case validates the id shape before it is ever
    joined onto a filesystem path."""
    _owned_case(case_id, request)
    path = drafting.PDF_DIR / f"{case_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="draft not generated yet")
    return FileResponse(path, media_type="application/pdf", filename=f"haq-{case_id}.pdf")


# ------------------------------------------------------------- 5. /api/speak


class Speak(BaseModel):
    # case_id is required: without it this endpoint was an unauthenticated
    # text-to-speech proxy anyone could point at our Sarvam quota.
    text: str
    case_id: str
    lang: str | None = None
    speaker: str = "ritu"


@app.post("/api/speak")
def speak(payload: Speak, request: Request):
    """Bulbul reads the English draft aloud in her language. She approves a document
    she cannot read, by hearing it in one she can.

    The language comes from the case, not from the caller. Hardcoding it here is how
    a Hindi speaker ends up being read to in Marathi.
    """
    case = _owned_case(payload.case_id, request)
    language = payload.lang or case.get("language") or "en-IN"

    if not onboarding.is_supported(language):
        raise HTTPException(
            status_code=400,
            detail=f"{language} can be transcribed but not spoken; no read-back available",
        )

    audio_base64 = sarvam.tts(payload.text, language_code=language, speaker=payload.speaker)
    return {"audio_base64": audio_base64, "language": language}


class Approve(BaseModel):
    case_id: str


@app.post("/api/approve")
def approve(payload: Approve, request: Request):
    """We never file autonomously. Every outbound document needs a recorded human
    approval — this endpoint is that record (Q&A answer 3)."""
    _owned_case(payload.case_id, request)
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
def advance(payload: Advance, request: Request):
    """The time machine. Nobody clicked anything — silence became a legal event.
    STATE: WATCHING."""
    case = _owned_case(payload.case_id, request)
    offset = (case.get("clock_offset_days") or 0) + payload.days
    store.update_case(payload.case_id, clock_offset_days=offset)

    case = _owned_case(payload.case_id, request)
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
def get_case(case_id: str, request: Request):
    case = _owned_case(case_id, request)
    verdict = verdict_for(case)
    return {
        # owner_token is a credential, not case data. It must never leave the server.
        "case": {k: v for k, v in case.items() if k not in ("facts", "owner_token")},
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
