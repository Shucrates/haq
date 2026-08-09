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
import drafting
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
    }
    kind = message.get("type")

    if kind == "text":
        base["text"] = message.get("text", {}).get("body", "")
    elif kind == "audio":
        base["media_id"] = message.get("audio", {}).get("id")
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

        text = inbound["text"]

        if inbound["media_id"]:
            audio = download_media(inbound["media_id"])
            heard = sarvam.stt(audio, filename="voice.ogg")
            text = heard["transcript"]
            case_id = store.case_for_phone(phone)
            store.update_case(case_id, language=heard["language_code"])

        if not text:
            send_text(phone, "Send a message or a voice note describing your problem.")
            return

        if text.strip().lower() in {"reset", "start over", "restart"}:
            store.reset_phone(phone)
            send_text(phone, "Starting a fresh case. Tell me what happened.")
            return

        _advance_case(phone, text)

    except Exception as exc:  # noqa: BLE001 — a channel must not crash the app
        log.exception("whatsapp_inbound_failed: %s", exc)


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
        store.update_case(
            case_id,
            transcript=text,
            grievance_class=classified["grievance_class"],
            language=case.get("language") or "en-IN",
            facts={"institution": classified["institution"]} if classified.get("institution") else {},
        )
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
        reason = agent.phrase_refusal(verdict, case.get("language"))
        blocked = ", ".join(verdict.blocked_by)
        send_text(phone, f"{reason}\n\n_Blocked: {blocked}_\n{verdict.source_url}")
        store.add_event(case_id, "refusal", blocked)
        return

    built = drafting.build_body(case, verdict, verdict.current_tier)
    send_text(phone, built["body_text"][:3500])
    send_buttons(
        phone,
        "Shall I prepare this for filing?",
        [("APPROVE", "Yes, file it"), ("REJECT", "No, change it")],
    )


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
