"""WhatsApp channel — Meta Cloud API.

Ported from vinaypokharkar/turfbot, keeping the four patterns that make that bot
production-grade, because HAQ needs all of them more than a turf booking did:

  * fast-ack then async   Meta retries if you don't answer in ~5s, and
                          Saaras -> 105B -> ladder -> Mayura -> Bulbul never fits
  * HMAC verification     an unsigned public webhook is an endpoint anyone can
                          POST complaints into
  * message-id dedupe     Meta redelivers; without this one voice note files two
                          complaints
  * state in the DB       turfbot's conversations table, minus its STALE_MINUTES
                          reset — a legal case is not a 30-minute booking session

Deliberately dropped: multi-tenancy. turfbot resolves a tenant by phone_number_id;
HAQ has one number, so that table becomes two env vars.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import date

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import agent
import documents
import drafting
import onboarding
import sarvam
import store

log = logging.getLogger("haq.whatsapp")
router = APIRouter()

GRAPH_VERSION = "v21.0"
GRAPH = "https://graph.facebook.com"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "")
APP_SECRET = os.getenv("APP_SECRET", "")
WA_TOKEN = os.getenv("WA_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")

# Saaras accepts ogg/opus directly, which is exactly what WhatsApp sends. No ffmpeg.
STT_LIMIT_NOTE = "Voice notes must be under 30 seconds — the REST STT endpoint rejects longer."


# ------------------------------------------------------------ verification


def verify_meta(raw_body: bytes, header: str | None, app_secret: str) -> bool:
    """x-hub-signature-256 = 'sha256=' + HMAC(appSecret, rawBody).
    Same contract as turfbot's src/wa/verify.js, constant-time compared."""
    if not header or not app_secret or not raw_body:
        return False
    expected = "sha256=" + hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header, expected)


@router.get("/webhook")
def meta_verify(request: Request):
    """Meta's subscribe handshake — echo hub.challenge."""
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(params.get("hub.challenge") or "")
    return PlainTextResponse("forbidden", status_code=403)


@router.post("/webhook")
async def meta_webhook(request: Request, background: BackgroundTasks):
    """Ack fast, work later. The raw body must be read BEFORE parsing JSON or the
    signature can't be checked."""
    raw = await request.body()

    if not verify_meta(raw, request.headers.get("x-hub-signature-256"), APP_SECRET):
        log.warning("meta_signature_fail")
        return JSONResponse({"error": "bad signature"}, status_code=401)

    try:
        payload = json.loads(raw)
    except ValueError:
        return JSONResponse({"ok": True})

    value = (
        payload.get("entry", [{}])[0]
        .get("changes", [{}])[0]
        .get("value", {})
    )
    message = (value.get("messages") or [None])[0]

    # Status callbacks (delivered/read) arrive here too — ignore them.
    if message:
        if not store.already_processed(message.get("id", "")):
            background.add_task(handle_inbound, value, message)

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------- inbound


def parse_inbound(value: dict, message: dict) -> dict:
    """Normalise a Cloud API message into a flat shape. turfbot's parseInbound plus
    an `audio` branch, which is the whole point of HAQ on WhatsApp."""
    base = {
        "from": message.get("from"),
        "message_id": message.get("id"),
        "name": (value.get("contacts") or [{}])[0].get("profile", {}).get("name"),
        "type": message.get("type"),
        "text": "",
        "value": None,
        "media_id": None,
        "filename": None,
    }
    kind = message.get("type")

    if kind == "text":
        base["text"] = message.get("text", {}).get("body", "")
    elif kind == "audio":
        base["media_id"] = message.get("audio", {}).get("id")
    elif kind in ("document", "image"):
        # The bank's rejection letter, photographed or forwarded. Densest source of
        # facts we ever get — Doc AI reads it instead of us asking six questions.
        payload = message.get(kind, {})
        base["media_id"] = payload.get("id")
        base["filename"] = payload.get("filename") or f"{kind}.jpg"
        base["text"] = payload.get("caption", "")
    elif kind == "interactive":
        interactive = message.get("interactive", {})
        if interactive.get("type") == "button_reply":
            reply = interactive["button_reply"]
            base.update(type="button", value=reply["id"], text=reply["title"])
        elif interactive.get("type") == "list_reply":
            reply = interactive["list_reply"]
            base.update(type="list", value=reply["id"], text=reply["title"])
    elif kind == "button":
        base.update(value=message.get("button", {}).get("payload"),
                    text=message.get("button", {}).get("text", ""))

    return base


def download_media(media_id: str) -> bytes:
    """Two hops: the id resolves to a URL, and the URL needs the Bearer token too."""
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    with httpx.Client(timeout=TIMEOUT) as client:
        meta = client.get(f"{GRAPH}/{GRAPH_VERSION}/{media_id}", headers=headers)
        meta.raise_for_status()
        url = meta.json()["url"]
        blob = client.get(url, headers=headers)
        blob.raise_for_status()
        return blob.content


def handle_inbound(value: dict, message: dict) -> None:
    """Runs in the background after the 200. Any exception here is logged, never
    surfaced to Meta — a 500 would make it retry the same message forever."""
    try:
        inbound = parse_inbound(value, message)
        phone = inbound["from"]
        if not phone:
            return

        text = inbound["text"] or ""
        action = inbound.get("value")
        case_id = store.case_for_phone(phone)
        case = store.get_case(case_id)

        if text.strip().lower() in {"reset", "start over", "restart"}:
            case_id = store.reset_phone(phone)
            _handle_language(phone, case_id, None)  # a reset returns to onboarding
            return

        # A bare "hi" is an opener, not a complaint. Without this it gets classified,
        # lands in `other`, matches no ladder, and the person is answered with a
        # refusal. Only before a grievance exists — mid-case, "hi" is an answer.
        if onboarding.is_greeting(text) and not case.get("grievance_class"):
            _handle_language(phone, case_id, None)
            return

        # Onboarding first. A grievance interrogated in the wrong language is worse
        # than a slow one, and this must run BEFORE any media is transcribed —
        # otherwise a voice note's detected language silently overrides the choice.
        # A language row is also routed here once a language is already set, because
        # the picker can be re-shown by a greeting; otherwise the tap's title text
        # ("हिंदी") would fall through and be classified as the grievance.
        if not case.get("language") or (action or "").startswith(onboarding.LANG_PREFIX):
            _handle_language(phone, case_id, action)
            return

        if inbound["media_id"] and inbound["type"] == "audio":
            audio = download_media(inbound["media_id"])
            # Hint Saaras with the language she chose rather than re-detecting it.
            heard = sarvam.stt(audio, filename="voice.ogg", language_code=case["language"])
            text = heard["transcript"]

        elif inbound["media_id"] and inbound["type"] in ("document", "image"):
            _handle_document(phone, case_id, inbound, case.get("language"))
            return

        if not text:
            _say(phone, case_id, "Send a message or a voice note describing your problem.")
            return

        # A draft is already on the table, so this message is a decision about it,
        # not an answer to a fact question. Without this branch a button tap falls
        # through to the interrogation, gets read as a fact, and the bot re-drafts.
        if _awaiting_approval(case):
            _handle_decision(phone, case_id, action, text)
            return

        _advance_case(phone, text)

    except Exception as exc:  # noqa: BLE001 — a channel must not crash the app
        log.exception("whatsapp_inbound_failed: %s", exc)


def _say(phone: str, case_id: str, english: str) -> bool:
    """Send one message in the language this case chose.

    Every literal HAQ says goes through here, so adding a message never means
    remembering to translate it. The language is read from the case rather than
    passed in because the callers already have the case id and nothing else may
    decide what language a person is spoken to in.
    """
    language = (store.get_case(case_id) or {}).get("language")
    return send_text(phone, agent.localise(english, language))


def _debug(exc: Exception) -> str:
    """Echo the real error to the chat when HAQ_DEBUG=1.

    Off by default: users must never see a stack trace, and the surrounding handlers
    deliberately swallow failures so one bad scan cannot close a case. But that same
    silence makes a misconfigured deployment indistinguishable from a blurry photo,
    and the logs are not always where the person testing is looking. Read at call
    time so it can be toggled without a code change.
    """
    if os.getenv("HAQ_DEBUG") != "1":
        return ""
    return f"\n\n_debug: {str(exc)[:500]}_"


def _handle_document(phone: str, case_id: str, inbound: dict, language: str | None) -> None:
    """Read a letter the user sent. Doc AI is slow, so tell them it's happening —
    silence after sending a document reads as the bot being broken."""
    _say(phone, case_id, "Reading your document…")

    try:
        blob = download_media(inbound["media_id"])
        extracted = documents.extract(blob, inbound.get("filename") or "document.jpg", language)
    except Exception as exc:  # noqa: BLE001 — a bad scan must not close the case
        log.warning("document_failed: %s", exc)
        store.save_document(case_id, inbound.get("filename"), "failed", {"error": str(exc)})
        # The debug tail is appended AFTER localisation — it is a stack trace for us,
        # not a sentence for her, and translating it would only make it useless.
        send_text(
            phone,
            agent.localise("I couldn't read that. Just tell me the details instead.", language)
            + _debug(exc),
        )
        return

    split = documents.to_facts(extracted)
    store.save_document(case_id, inbound.get("filename"),
                        extracted.get("status", "unknown"), split["raw"])

    if split["facts"]:
        store.merge_facts(case_id, split["facts"])
        store.add_event(case_id, "document", ", ".join(split["facts"]))

    lines = [documents.summarise(extracted)]
    # Low-confidence fields are asked about, never assumed. A wrong date here
    # silently changes which tier the complaint is maintainable at.
    for item in split["confirm"]:
        lines.append(f"Is this right — {item['field'].replace('_', ' ')}: {item['value']}?")
    # The values inside are what she has to check, and Mayura carries dates and
    # amounts through unchanged (numerals_format=international).
    send_text(phone, agent.localise("\n".join(lines), language))


def _handle_language(phone: str, case_id: str, action: str | None) -> None:
    """The whole of onboarding: send the picker, or record the pick.

    The user's entire first action is pressing send on a prefilled message, so this
    is the first thing they see. It must never require typing.
    """
    code = onboarding.code_from_option(action)

    if code:
        store.update_case(case_id, language=code)
        store.add_event(case_id, "language", code)
        send_text(phone, onboarding.greeting(code))
        return

    # "More languages" -> page two. Anything else -> page one.
    page = 1 if action == onboarding.MORE_LANGUAGES else 0
    rows, _ = onboarding.wa_rows(page)
    send_language_list(phone, rows)


# What HAQ can actually escalate: the ladders in data/ladders, and nothing else.
OUT_OF_SCOPE = (
    "I can only take this further for complaints against a bank, or an RTI request "
    "that got no reply. Tell me if you have one of those and I will help."
)

APPROVE = "APPROVE"
REJECT = "REJECT"

# Typed replies, for people who answer instead of tapping. "हो" is the word from the
# demo script; the rest are the usual ways people say yes and no in this context.
_YES_TEXT = {"yes", "y", "ok", "okay", "haan", "ha", "हो", "हाँ", "हा", "file it", "approve"}
_NO_TEXT = {"no", "n", "nahi", "नाही", "नहीं", "change", "reject", "wait"}


def _awaiting_approval(case: dict) -> bool:
    """True once a draft exists and no human has approved it yet."""
    return bool(case.get("draft_text")) and not case.get("approved_at")


def _decision_from(action: str | None, text: str) -> str | None:
    """Button id wins; fall back to what they typed."""
    if action in (APPROVE, REJECT):
        return action
    lowered = (text or "").strip().lower()
    if lowered in _YES_TEXT:
        return APPROVE
    if lowered in _NO_TEXT:
        return REJECT
    return None


def _handle_decision(phone: str, case_id: str, action: str | None, text: str) -> None:
    """The approval gate. We never file autonomously: every outbound document needs
    a recorded human approval, and this function is where that record is made."""
    from datetime import datetime

    decision = _decision_from(action, text)

    if decision == APPROVE:
        stamp = datetime.utcnow().isoformat(timespec="seconds")
        store.update_case(case_id, approved_at=stamp)
        store.add_event(case_id, "approved", f"whatsapp {phone}")
        store.add_message(case_id, "user", text or "APPROVE")

        # The deadline labels come from the ladder YAML and are English too, so the
        # whole block goes through together rather than a translated sentence
        # followed by a list she cannot read.
        lines = ["Filed. I will watch the clock for you."]
        for deadline in store.all_deadlines(case_id):
            lines.append(f"• {deadline['label']} — {deadline['due_on']}")
        _say(phone, case_id, "\n".join(lines))
        return

    if decision == REJECT:
        # Drop the draft so the next message is treated as input again.
        store.update_case(case_id, draft_text=None)
        store.add_event(case_id, "rejected", f"whatsapp {phone}")
        _say(
            phone,
            case_id,
            "Not sending it. Tell me what to change, or send *reset* to start over.",
        )
        return

    # Anything else while a draft is pending: re-ask rather than guess. Approval is
    # the one place in this product where an LLM must not interpret intent.
    _ask_to_file(phone, case_id, "Please tap one — should I prepare this for filing?")


# WhatsApp truncates a button title past 20 characters, and a half-word in Devanagari
# is worse than none — so the limit is given to the model rather than to a slice.
BUTTON_TITLE_CHARS = 20


def _ask_to_file(phone: str, case_id: str, body: str) -> bool:
    """The approval buttons, in her language. The ids stay APPROVE / REJECT: the
    words are translated, the decision is still read off a button id, never off
    text a model produced."""
    language = (store.get_case(case_id) or {}).get("language")
    return send_buttons(
        phone,
        agent.localise(body, language),
        [
            (APPROVE, agent.localise("Yes, file it", language, BUTTON_TITLE_CHARS)),
            (REJECT, agent.localise("No, change it", language, BUTTON_TITLE_CHARS)),
        ],
    )


def _advance_case(phone: str, text: str) -> None:
    """Drive the same state machine the browser drives. This function calls the
    module-level functions directly — not the HTTP endpoints — which is exactly why
    the routes in main.py stay thin."""
    import main  # imported lazily so this module stays importable on its own

    case_id = store.case_for_phone(phone)
    case = store.get_case(case_id)

    # First contact on this case: classify, then start interrogating.
    if not case.get("grievance_class"):
        classified = agent.classify(text)

        # Nothing has gone wrong yet — she is asking how a loan works or which
        # documents a bank wants. Answer it. grievance_class is deliberately left
        # unset so the next message is classified fresh: most people ask a question
        # first and describe the actual problem second.
        if classified.get("intent") == "question":
            store.add_message(case_id, "user", text)
            answer = agent.answer_question(text, case.get("language"))
            send_text(phone, answer)
            store.add_message(case_id, "assistant", answer)
            store.add_event(case_id, "guidance", classified.get("grievance_class") or "other")
            return

        store.update_case(
            case_id,
            transcript=text,
            grievance_class=classified["grievance_class"],
        )
        # merge, never replace: a document sent before the description already put
        # institution, amount and dates on this fact sheet.
        store.merge_facts(case_id, {"institution": classified.get("institution")})
        store.add_message(case_id, "user", text)
    else:
        pending = agent.pending_facts(case["facts"], main.required_facts_for(case))
        if pending:
            agent.record_answer(case_id, pending[0], text)

    case = store.get_case(case_id)
    required = main.required_facts_for(case)
    step = agent.next_question(case_id, required)

    if not step["done"]:
        send_text(phone, step["question_native"] or step["question"])
        return

    # Fact sheet complete -> resolve. The refusal is decided by the pure function.
    verdict = main.verdict_for(case)
    store.save_deadlines(case_id, verdict.deadlines)

    if not verdict.maintainable:
        blocked = ", ".join(verdict.blocked_by)
        store.add_event(case_id, "refusal", blocked)

        # No ladder is not a legal refusal, it is the edge of what HAQ covers. Saying
        # "we have no verified escalation ladder" to someone with an insurance
        # problem answers a question they did not ask.
        if verdict.blocked_by == ["no_ladder_for_grievance"]:
            _say(phone, case_id, OUT_OF_SCOPE)
            return

        reason = agent.phrase_refusal(verdict, case.get("language"))
        # The rule id is for us. It was going out to every user as `Blocked:
        # tier1_not_exhausted`, which is a variable name, not a sentence.
        tail = f"\n\n_Blocked: {blocked}_" if os.getenv("HAQ_DEBUG") == "1" else ""
        send_text(phone, f"{reason}{tail}\n{verdict.source_url}")
        return

    built = drafting.build_body(case, verdict, verdict.current_tier)

    # Persist before asking: _awaiting_approval() reads draft_text to know that the
    # next inbound message is a decision rather than a fact.
    store.update_case(case_id, draft_text=built["body_text"])
    store.add_event(case_id, "draft", f"tier {verdict.current_tier}")

    # The filing itself stays in formal English — it is addressed to the bank, not
    # to her — so it is introduced in her language rather than dropped on her as a
    # wall of text she has no way to place.
    _say(phone, case_id, "This is the complaint I have written for you. Listen to it below.")
    send_text(phone, built["body_text"][:3500])

    # She approves a document she cannot read, by hearing it in one she can. A TTS
    # failure must not block the approval, so this is best-effort.
    try:
        language = case.get("language") or "mr-IN"
        send_voice(phone, _spoken_readback(built["body_text"], language), language_code=language)
    except Exception as exc:  # noqa: BLE001
        log.warning("readback_failed: %s", exc)

    _ask_to_file(phone, case_id, "Shall I prepare this for filing?")


def _spoken_readback(body: str, language: str | None) -> str:
    """The English filing, in the language it will be read aloud in.

    Mayura rather than 105B: this is the one text that must say exactly what the
    document says, and a reasoning model asked to "rewrite" a legal body is a
    reasoning model given the chance to change it. Bulbul was previously handed the
    English body with a mr-IN voice, which produced Marathi-accented English — the
    read-back is the whole approval mechanism, so it has to be her language.
    """
    if not language or language.startswith("en"):
        return body
    try:
        return sarvam.translate(body, target=language, source="en-IN") or body
    except Exception as exc:  # noqa: BLE001 — she still hears something
        log.warning("readback_translate_failed: %s", exc)
        return body


# --------------------------------------------------------------- outbound


def _send(payload: dict) -> bool:
    """One sender for every message shape, like turfbot's src/wa/send.js."""
    if not (WA_TOKEN and PHONE_NUMBER_ID):
        log.error("wa_not_configured")
        return False
    url = f"{GRAPH}/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            log.error("wa_send_failed %s %s", response.status_code, response.text[:300])
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("wa_send_error %s", exc)
        return False


def send_text(to: str, body: str) -> bool:
    return _send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body[:4096]},
    })


def send_buttons(to: str, body: str, buttons: list[tuple[str, str]]) -> bool:
    """Interactive buttons take the LLM out of the approval path: 'Yes, file it' is
    a button id, not free text we have to interpret."""
    return _send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body[:1024]},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": bid, "title": title[:20]}}
                    for bid, title in buttons[:3]
                ]
            },
        },
    })


def send_language_list(to: str, rows: list[dict]) -> bool:
    """Interactive list, capped at 10 rows by the Cloud API — onboarding.wa_rows()
    pages for us. A list beats buttons here: buttons max out at three."""
    return _send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {"text": onboarding.PICKER_PROMPT},
            "action": {
                "button": "Choose",
                "sections": [{"title": "Languages", "rows": rows[:onboarding.WA_LIST_MAX_ROWS]}],
            },
        },
    })


def send_voice(to: str, text: str, language_code: str = "mr-IN") -> bool:
    """Bulbul read-back as a real WhatsApp voice note. opus is what WhatsApp wants,
    and Bulbul emits it directly — no conversion step."""
    audio_base64 = sarvam.tts(text, language_code=language_code, output_audio_codec="opus")
    media_id = upload_media(audio_base64, "voice.ogg", "audio/ogg")
    if not media_id:
        return False
    return _send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "audio",
        "audio": {"id": media_id},
    })


def upload_media(audio_base64: str, filename: str, mime: str) -> str | None:
    import base64

    url = f"{GRAPH}/{GRAPH_VERSION}/{PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {WA_TOKEN}"}
    files = {
        "file": (filename, base64.b64decode(audio_base64), mime),
        "messaging_product": (None, "whatsapp"),
        "type": (None, mime),
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.post(url, headers=headers, files=files)
        if response.status_code >= 400:
            log.error("wa_upload_failed %s %s", response.status_code, response.text[:300])
            return None
        return response.json().get("id")
    except Exception as exc:  # noqa: BLE001
        log.error("wa_upload_error %s", exc)
        return None


def send_deadline_template(to: str, institution: str, tier: str, due_on: date) -> bool:
    """Proactive escalation OUTSIDE the 24-hour customer-service window needs an
    approved template. Inside the window, send_text works and this is unnecessary.
    Submit haq_deadline_alert to Meta on day 0 — approval has lead time."""
    return _send({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": {
            "name": "haq_deadline_alert",
            "language": {"code": "en"},
            "components": [{
                "type": "body",
                "parameters": [
                    {"type": "text", "text": institution},
                    {"type": "text", "text": tier},
                    {"type": "text", "text": due_on.isoformat()},
                ],
            }],
        },
    })
