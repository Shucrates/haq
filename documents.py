"""Document intake — Sarvam Doc AI Extract.

The bank's rejection letter is the densest object in the whole process: it carries
the institution, the reference number, the date the clock starts from, and whether
this was a final reply. Reading it beats asking six questions.

Doc AI over a generic OCR because it handles 22 Indian languages, scans and
handwriting — and because every extracted leaf comes back with a `confidence` and
`sources`. We use that: a field we are unsure about is *confirmed with the user*,
never silently written into a fact sheet that decides a legal deadline.

    POST /doc-ai/v1/job/extract          -> { job_id }              async
    GET  /doc-ai/v1/job/{id}/status      -> poll to a terminal state
    GET  /doc-ai/v1/job/{id}/results     -> { result, annotations }  409 if running
"""

from __future__ import annotations

import json
import time
from typing import Any

import sarvam
from demo_cache import cached, key_for

# Above this we trust the extraction; below it we ask. Deliberately conservative —
# a wrong date here silently changes which tier a complaint is maintainable at.
CONFIDENCE_THRESHOLD = 0.75

TERMINAL = {"completed", "partially_completed", "failed", "rejected"}
POLL_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2.0

# Every field needs a non-empty description — the API rejects the schema otherwise.
EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "institution": {
            "type": "string",
            "description": "Name of the bank, company or public authority that sent this letter.",
        },
        "reference_number": {
            "type": "string",
            "description": "Complaint, docket or reference number quoted in the letter.",
        },
        "letter_date": {
            "type": "string",
            "description": "Date the letter was written or issued, in YYYY-MM-DD format.",
        },
        "decision": {
            "type": "string",
            "description": "What the institution decided: rejected, partly allowed, allowed, or no decision.",
        },
        "amount_inr": {
            "type": "number",
            "description": "Any amount of money in Indian rupees mentioned as disputed, charged or refunded.",
        },
        "is_final_reply": {
            "type": "boolean",
            "description": "True if the letter states it is the institution's final response to the complaint.",
        },
        "deadline_mentioned": {
            "type": "string",
            "description": "Any deadline or time limit the letter tells the recipient to act within.",
        },
    },
}

# Extracted field -> the fact sheet name the ladder engine understands.
FACT_MAP = {
    "institution": "institution",
    "amount_inr": "amount_inr",
    "letter_date": "last_communication_on",
}


def _submit(file_bytes: bytes, filename: str, language: str | None) -> str:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {"schema": json.dumps(EXTRACT_SCHEMA)}
    if language:
        data["language"] = language
    out = sarvam._post("/doc-ai/v1/job/extract", files=files, data=data)
    return out["job_id"]


def _await_terminal(job_id: str) -> str:
    """Poll until the job settles. A document that will not parse must never hold
    the case open, so this gives up rather than blocking forever."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    status = "unknown"
    while time.monotonic() < deadline:
        status = sarvam._get(f"/doc-ai/v1/job/{job_id}/status").get("status", "unknown")
        if status in TERMINAL:
            return status
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Doc AI job {job_id} still {status} after {POLL_TIMEOUT_SECONDS}s")


def extract(file_bytes: bytes, filename: str = "document.pdf",
            language: str | None = None) -> dict:
    """Submit, poll, fetch. Cached like every other network call so the demo can
    run offline. Returns {result, annotations, status}."""

    def call() -> dict:
        job_id = _submit(file_bytes, filename, language)
        status = _await_terminal(job_id)
        if status in {"failed", "rejected"}:
            return {"result": {}, "annotations": {}, "status": status}
        out = sarvam._get(f"/doc-ai/v1/job/{job_id}/results")
        return {
            "result": out.get("result", {}),
            "annotations": out.get("annotations", {}),
            "status": status,
        }

    return cached("docai", key_for(file_bytes, language), call)


def _confidence(annotations: dict, field: str) -> float | None:
    """annotations mirrors result, with each leaf replaced by its confidence and
    sources. Shapes vary, so read defensively rather than assuming."""
    leaf = (annotations or {}).get(field)
    if isinstance(leaf, dict):
        value = leaf.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def to_facts(extracted: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Split the extraction into facts we trust and fields we should confirm.

    Returns {facts, confirm, ignored} where `confirm` is a list of
    {field, fact, value, confidence} the user should be asked about.
    """
    result = extracted.get("result") or {}
    annotations = extracted.get("annotations") or {}

    facts: dict[str, Any] = {}
    confirm: list[dict] = []
    ignored: list[str] = []

    for field, fact in FACT_MAP.items():
        value = result.get(field)
        if value in (None, "", []):
            continue

        score = _confidence(annotations, field)
        entry = {"field": field, "fact": fact, "value": value, "confidence": score}

        # No confidence reported is not the same as high confidence — ask.
        if score is not None and score >= threshold:
            facts[fact] = value
        else:
            confirm.append(entry)

    for field in result:
        if field not in FACT_MAP and result.get(field) not in (None, "", []):
            ignored.append(field)

    return {"facts": facts, "confirm": confirm, "ignored": ignored, "raw": result}


def summarise(extracted: dict) -> str:
    """One line for a WhatsApp reply — what we read out of their document."""
    result = extracted.get("result") or {}
    bits = []
    if result.get("institution"):
        bits.append(str(result["institution"]))
    if result.get("letter_date"):
        bits.append(f"dated {result['letter_date']}")
    if result.get("reference_number"):
        bits.append(f"ref {result['reference_number']}")
    if result.get("is_final_reply"):
        bits.append("final reply")
    return "I read: " + ", ".join(bits) if bits else "I could not read anything useful from that."
