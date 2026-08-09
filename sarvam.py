"""Thin Sarvam API client — five functions, one per capability.

Verified against docs.sarvam.ai on 2026-08-08. Two things that cost teams an hour
each and are handled here:

  1. TWO AUTH SCHEMES. /speech-to-text, /text-to-speech and /translate take only
     `api-subscription-key`. /v1/chat/completions ALSO needs `Authorization: Bearer`.
  2. TTS returns `audios` as a LIST of base64 strings, not a single string.

Everything is wrapped in demo_cache.cached() so the demo can run with the wifi off.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

import httpx

from demo_cache import cached, key_for

BASE_URL = "https://api.sarvam.ai"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# Model choices are deliberate — see README "Why these models".
STT_MODEL = "saaras:v3"          # v4 is newer but drops `mode`, and we need codemix
STT_MODE = "codemix"             # Marathi in Devanagari, English words in Latin
CHAT_MODEL = "sarvam-105b"       # 128K context, for reasoning
TTS_MODEL = "bulbul:v3"
TRANSLATE_MODEL = "mayura:v1"

# Hard limits from the docs. Exceeding them is a 4xx, not a truncation.
STT_MAX_SECONDS = 30
TTS_MAX_CHARS = 2500
MAYURA_MAX_CHARS = 1000

CHAT_SEED = 20260808  # fixed so replays and reruns look the same


def _key() -> str:
    key = os.getenv("SARVAM_API_KEY", "")
    if not key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Copy .env.example to .env and fill it in. "
            "(In HAQ_MODE=replay you never reach this line.)"
        )
    return key


def _headers(bearer: bool = False) -> dict[str, str]:
    headers = {"api-subscription-key": _key()}
    if bearer:
        headers["Authorization"] = f"Bearer {_key()}"
    return headers


def _post(path: str, *, bearer: bool = False, **kwargs) -> dict:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(f"{BASE_URL}{path}", headers=_headers(bearer), **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"Sarvam {path} -> {response.status_code}: {response.text[:400]}")
        return response.json()


# ---------------------------------------------------------------- C1: voice in


def stt(audio: bytes, filename: str = "audio.webm", language_code: str = "unknown") -> dict:
    """Speech to text. Saaras accepts WebM/OGG/Opus/WAV/MP3 directly — no ffmpeg,
    no browser-side WAV encoder. REST is for clips UNDER 30 SECONDS.

    Returns {transcript, language_code, language_probability}.
    """

    def call() -> dict:
        files = {"file": (filename, audio, "application/octet-stream")}
        data = {"model": STT_MODEL, "mode": STT_MODE, "language_code": language_code}
        return _post("/speech-to-text", files=files, data=data)

    out = cached("stt", key_for(audio, language_code), call)
    return {
        "transcript": out.get("transcript", ""),
        "language_code": out.get("language_code") or "unknown",
        # Comes free from the API when language_code=unknown — this is the
        # confidence number the UI shows. No separate classifier needed.
        "confidence": out.get("language_probability"),
    }


# ------------------------------------------------------------ C2: the 105B loop


def chat(messages: list[dict], *, max_tokens: int = 512, reasoning_effort: str = "low") -> str:
    """Chat completion. Note the extra Bearer header and wiki_grounding=False:
    grounding comes from statutes.json and validate_citations(), never from the
    model's own world knowledge."""

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "wiki_grounding": False,
        "seed": CHAT_SEED,
        "n": 1,
    }

    def call() -> dict:
        return _post("/v1/chat/completions", bearer=True, json=payload)

    out = cached("chat", key_for(payload), call)
    return out["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------- C4: formal English out


def _chunk(text: str, limit: int) -> list[str]:
    """Split on paragraph then sentence boundaries so nothing is cut mid-clause."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(para) > limit:
            cut = para.rfind(". ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        current = para
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def translate(text: str, target: str = "en-IN", source: str = "auto") -> str:
    """Mayura, formal mode — she speaks Marathi, the bank receives formal English.

    mayura:v1 caps at 1000 characters, so a real complaint body must be chunked.
    numerals_format=international keeps rupee amounts as 9840, not ९८४०.
    """
    parts = []
    for piece in _chunk(text, MAYURA_MAX_CHARS):
        payload = {
            "input": piece,
            "source_language_code": source,
            "target_language_code": target,
            "model": TRANSLATE_MODEL,
            "mode": "formal",
            "numerals_format": "international",
        }

        def call(p=payload) -> dict:
            return _post("/translate", json=p)

        out = cached("translate", key_for(payload), call)
        parts.append(out.get("translated_text") or out.get("output") or "")
    return "\n".join(parts).strip()


# ------------------------------------------------------------- C5: Bulbul reads


_NUMBER = re.compile(r"(?<![\d,])(\d{5,})(?![\d,])")


def format_numbers_for_speech(text: str) -> str:
    """Bulbul reads numbers over 4 digits digit-by-digit unless they carry commas.
    A read-back of '10000' becomes 'one zero zero zero zero', which sounds broken
    on stage. Insert Indian-format commas first."""

    def comma(match: re.Match) -> str:
        digits = match.group(1)
        if len(digits) <= 3:
            return digits
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        return ",".join(groups + [tail])

    return _NUMBER.sub(comma, text)


def tts(text: str, language_code: str = "mr-IN", speaker: str = "ritu",
        output_audio_codec: str = "wav") -> str:
    """Text to speech. Returns ONE base64 string (chunks concatenated by the caller
    where needed). bulbul:v3 does not support pitch/loudness/enable_preprocessing.

    output_audio_codec='wav' for the browser, 'opus' for WhatsApp voice notes.
    """
    spoken = format_numbers_for_speech(text)
    audios: list[str] = []

    for piece in _chunk(spoken, TTS_MAX_CHARS):
        payload = {
            "text": piece,
            "language_code": language_code,
            "model": TTS_MODEL,
            "speaker": speaker,          # lowercase — names are case-sensitive
            "pace": 1.0,
            "temperature": 0.6,
            "output_audio_codec": output_audio_codec,
        }

        def call(p=payload) -> dict:
            return _post("/text-to-speech", json=p)

        out = cached("tts", key_for(payload), call)
        audios.extend(out["audios"])  # NOTE: a LIST, not a string

    if len(audios) == 1:
        return audios[0]
    # Concatenate raw payloads; fine for opus/mp3 and good enough for a demo read-back.
    return base64.b64encode(b"".join(base64.b64decode(a) for a in audios)).decode()


def health() -> dict[str, Any]:
    """Cheap self-check for /api/health — never calls the network."""
    return {
        "stt_model": STT_MODEL,
        "stt_mode": STT_MODE,
        "chat_model": CHAT_MODEL,
        "tts_model": TTS_MODEL,
        "translate_model": TRANSLATE_MODEL,
        "key_present": bool(os.getenv("SARVAM_API_KEY")),
    }
