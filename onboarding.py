"""Onboarding — one source of truth for the first thirty seconds.

Both channels share this module so they cannot drift: the wa.me deep link and the
web input show the same prefilled sentence, and both offer the same languages.

Why only eleven languages when Saaras transcribes twenty-two: HAQ's thesis is that
she approves a document she cannot read *by hearing it*. Bulbul speaks eleven. A
language we cannot speak back is a language we should not offer as an interface —
the extra eleven stay available for transcription, they are just not on this menu.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

# A WhatsApp interactive list allows at most 10 rows in total. With more languages
# than that we page: 9 real rows plus a "More languages" row that sends page two.
WA_LIST_MAX_ROWS = 10
_ROWS_PER_PAGE = WA_LIST_MAX_ROWS - 1

PREFILL_TEXT = "Hi HAQ, I need help with a complaint"

LANG_PREFIX = "lang_"
MORE_LANGUAGES = "lang_more"


@dataclass(frozen=True)
class Language:
    code: str      # BCP-47, exactly what Sarvam wants
    native: str    # shown to the user, in their own script
    english: str   # shown to us, in logs and the operator console


# Ordered by expected demand — Marathi first because it is the demo language.
LANGUAGES: list[Language] = [
    Language("mr-IN", "मराठी", "Marathi"),
    Language("hi-IN", "हिंदी", "Hindi"),
    Language("en-IN", "English", "English"),
    Language("bn-IN", "বাংলা", "Bengali"),
    Language("ta-IN", "தமிழ்", "Tamil"),
    Language("te-IN", "తెలుగు", "Telugu"),
    Language("kn-IN", "ಕನ್ನಡ", "Kannada"),
    Language("gu-IN", "ગુજરાતી", "Gujarati"),
    Language("ml-IN", "മലയാളം", "Malayalam"),
    Language("pa-IN", "ਪੰਜਾਬੀ", "Punjabi"),
    Language("od-IN", "ଓଡ଼ିଆ", "Odia"),
]

BY_CODE = {lang.code: lang for lang in LANGUAGES}

# The first question, written out per language rather than translated at runtime.
# Onboarding must work with the wifi off and must not cost an LLM call.
GREETINGS = {
    "mr-IN": "काय झालं ते सांगा. बोला किंवा लिहा.",
    "hi-IN": "बताइए क्या हुआ। बोलिए या लिखिए।",
    "en-IN": "Tell me what happened. Speak or type.",
    "bn-IN": "কী হয়েছে বলুন। বলুন বা লিখুন।",
    "ta-IN": "என்ன நடந்தது என்று சொல்லுங்கள். பேசுங்கள் அல்லது தட்டச்சு செய்யுங்கள்.",
    "te-IN": "ఏమి జరిగిందో చెప్పండి. మాట్లాడండి లేదా టైప్ చేయండి.",
    "kn-IN": "ಏನಾಯಿತು ಎಂದು ಹೇಳಿ. ಮಾತನಾಡಿ ಅಥವಾ ಟೈಪ್ ಮಾಡಿ.",
    "gu-IN": "શું થયું તે કહો. બોલો અથવા લખો.",
    "ml-IN": "എന്താണ് സംഭവിച്ചതെന്ന് പറയൂ. സംസാരിക്കുക അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക.",
    "pa-IN": "ਦੱਸੋ ਕੀ ਹੋਇਆ। ਬੋਲੋ ਜਾਂ ਲਿਖੋ।",
    "od-IN": "କଣ ଘଟିଲା କୁହନ୍ତୁ। କୁହନ୍ତୁ କିମ୍ବା ଲେଖନ୍ତୁ।",
}

# Shown above the language list. Bilingual so it reads to anyone.
PICKER_PROMPT = "Choose your language / भाषा निवडा"

# A bare hello is how most people open a chat, and it is not a grievance. Repeated
# letters are squeezed before matching, so "hii" and "heyyy" need no entry here.
_SQUEEZE = re.compile(r"(.)\1+")
_GREETING_WORDS = {
    _SQUEEZE.sub(r"\1", word)
    for word in (
        "hi", "hey", "hello", "hlo", "yo", "start", "hola",
        "namaste", "namaskar", "namaskaar", "salaam", "vanakkam",
        "नमस्ते", "नमस्कार", "हाय", "हॅलो", "हैलो", "வணக்கம்", "నమస్కారం",
    )
}
# Trailing decoration people add to a greeting. Kept off the Devanagari danda and
# other marks that only ever appear inside real words.
_GREETING_TRIM = " \t!.,?;:'\"…-~*"


def is_greeting(text: str | None) -> bool:
    """True only for a bare hello — one word and nothing else.

    Single-word is the whole guard: PREFILL_TEXT starts with "Hi" but goes on to
    describe a complaint, and that must still reach the interrogation.
    """
    word = (text or "").strip().lower().strip(_GREETING_TRIM)
    if not word or any(ch.isspace() for ch in word):
        return False
    return _SQUEEZE.sub(r"\1", word) in _GREETING_WORDS


# "Send it as a voice note" — in the eleven languages we speak back in. Substrings,
# so inflections and the nukta variants ("आवाज़", "ਆਵਾਜ਼") match the base form here.
# Latin first: half of these users type Hinglish even when they speak Marathi.
_VOICE_WORDS = (
    "voice", "audio", "awaaz", "awaz", "aawaj", "sunao", "sunau", "bol ke", "bolke",
    "आवाज", "व्हॉइस", "वॉइस", "ऑडिओ", "ऑडियो", "ऐकव", "सुनाओ", "बोलून",   # mr, hi
    "ভয়েস", "অডিও",          # bn
    "குரல்", "ஆடியோ",         # ta
    "వాయిస్", "ఆడియో",        # te
    "ಧ್ವನಿ", "ಆಡಿಯೋ",         # kn
    "વોઇસ", "ઓડિયો", "અવાજ",  # gu
    "ശബ്ദ", "ഓഡിയോ",          # ml
    "ਆਵਾਜ", "ਆਡੀਓ",          # pa
    "ଅଡିଓ", "ଭଏସ",           # od
)


def wants_voice(text: str | None) -> bool:
    """True when she asked to be answered in a voice note.

    ponytail: bare keyword match, so "the bank gave me a voice call" also flips it
    on. Harmless — she still gets the text — and cheaper than an LLM call on every
    inbound message, which is what we would pay to tell the two apart.
    """
    lowered = (text or "").lower()
    return any(word in lowered for word in _VOICE_WORDS)


def by_code(code: str | None) -> Language | None:
    return BY_CODE.get(code or "")


def is_supported(code: str | None) -> bool:
    """True when we can both understand and speak this language."""
    return (code or "") in BY_CODE


def greeting(code: str | None) -> str:
    return GREETINGS.get(code or "", GREETINGS["en-IN"])


def option_id(code: str) -> str:
    """List-row id, e.g. 'lang_mr-IN'. Prefixed so it can never collide with the
    APPROVE / REJECT button ids."""
    return f"{LANG_PREFIX}{code}"


def code_from_option(value: str | None) -> str | None:
    """Inverse of option_id(). Returns None for anything that isn't a language pick."""
    if not value or not value.startswith(LANG_PREFIX):
        return None
    code = value[len(LANG_PREFIX):]
    return code if code in BY_CODE else None


def wa_rows(page: int = 0) -> tuple[list[dict], bool]:
    """Rows for a WhatsApp interactive list, respecting the 10-row cap.

    Returns (rows, has_more). When there are more languages than fit, the last row
    is a "More languages" entry that requests the next page.
    """
    start = page * _ROWS_PER_PAGE
    chunk = LANGUAGES[start:start + _ROWS_PER_PAGE]
    rows = [
        # WhatsApp caps row titles at 24 characters.
        {"id": option_id(lang.code), "title": lang.native[:24], "description": lang.english}
        for lang in chunk
    ]
    has_more = start + _ROWS_PER_PAGE < len(LANGUAGES)
    if has_more:
        rows.append({"id": MORE_LANGUAGES, "title": "More languages", "description": "Other options"})
    return rows, has_more


def wa_link(phone_number: str) -> str:
    """The click-to-chat deep link that carries the prefilled message. Put this
    behind a QR code — tapping it opens WhatsApp with the text already typed, so
    the user's entire onboarding action is pressing send."""
    digits = "".join(ch for ch in phone_number if ch.isdigit())
    return f"https://wa.me/{digits}?text={urllib.parse.quote(PREFILL_TEXT)}"


def as_json() -> list[dict]:
    """For GET /api/languages, so the web UI never hardcodes the list."""
    return [{"code": l.code, "native": l.native, "english": l.english} for l in LANGUAGES]
