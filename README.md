# HAQ

**India has a thousand AIs that explain the system. This one stands inside it, next to you.**

On 1 July 2026 the RBI cut the window to escalate a bank complaint from one year to
ninety days. Nobody told the people it affects. HAQ is the alarm clock: speak your
grievance in Marathi, and it works out which rung of which legal ladder you are on,
**refuses to file when the complaint would be thrown out**, drafts the correct letter
in formal English, reads it back to you in your own language, and then watches the
clock so that silence becomes a legal event.

The interesting part is the refusal. Every other AI product would cheerfully generate
an Ombudsman complaint that gets rejected, burning the user's one chance. HAQ knows
the rules well enough to say no — and that is not a prompt, it is a tested function.

```
pytest -v   ->  25 passed
```

---

## Setup

Three commands from a clean clone:

```bash
uv venv --python 3.11              # 3.11 specifically — see "Python version" below
uv pip install -r requirements.txt
cp .env.example .env               # then put your Sarvam key in it
```

Run it:

```bash
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000>. Prove the engine works without touching the network:

```bash
pytest -v
```

### Without `uv`

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### The only required secret

```bash
SARVAM_API_KEY=sk_xxx      # required
FIRECRAWL_API_KEY=fc_xxx   # optional — Tier B context only, see "Grounding"
```

Everything else in `.env.example` has a working default. The key is **server-side only** —
the browser talks to FastAPI and never to `api.sarvam.ai`. Do not reintroduce a
`NEXT_PUBLIC_`-style variable for it; that ships your key to every visitor.

---

## Demo mode — read this before demoing

Never demo live against a third-party API on venue wifi. Venue wifi fails, Sarvam
rate-limits, and a cold model call takes nine seconds while the room goes quiet.

```bash
HAQ_MODE=record uvicorn main:app     # run the full demo once, locally, recording
git add data/cache && git commit     # the recordings are part of the repo
HAQ_MODE=replay uvicorn main:app     # on stage. always.
```

`replay` reads from `data/cache/` and never opens a socket. It sleeps 400 ms per call
so it still feels real.

Keep a second machine on `HAQ_MODE=live` on a phone hotspot. When a judge asks "is this
hardcoded?", volunteer it before they ask: *"The demo runs from cache so it never fails
on venue wifi. Here's the same call live — watch."* That converts suspicion into
credibility. Never say "hardcoded" defensively; say "cached, and here's the live one."

---

## What runs

**One process.** `main.py` is the only entrypoint — everything else is a library it
imports. You never run `python ladder_engine.py`.

```
uvicorn main:app
  └── main.py            routes: parse and return, nothing else
        ├── ladder_engine.resolve()   pure function, no I/O, no LLM   ← the moat
        ├── agent.py                  4-state machine with an LLM inside it
        ├── drafting.py               retrieve → draft → validate → translate → PDF
        ├── sarvam.py                 stt / chat / translate / tts
        ├── demo_cache.py             live | record | replay
        ├── store.py                  SQLite, no ORM
        └── whatsapp.py               optional channel, off by default
```

Route handlers stay thin on purpose: because all the logic lives in plain functions,
the WhatsApp adapter reuses it instead of duplicating it.

### API

| Endpoint | Does |
|---|---|
| `GET /api/languages` | the eleven offered languages + the prefilled opening message |
| `POST /api/start` | create a case; language is NULL until chosen |
| `POST /api/onboard` | record the chosen language, return the greeting in it |
| `POST /api/document` | read a letter with Doc AI, merge confident fields into the facts |
| `POST /api/intake` | audio → transcript, language, classification, confidence |
| `POST /api/intake_text` | same, for typing (no microphone needed) |
| `POST /api/turn` | one question per turn, filling the fact sheet |
| `POST /api/resolve` | **the refusal** — pure function, no LLM |
| `POST /api/draft` | grounded letter + PDF |
| `POST /api/speak` | Bulbul reads it aloud |
| `POST /api/approve` | records human approval — we never file autonomously |
| `POST /api/advance` | the time machine |

---

## The ladder engine

`ladder_engine.resolve(facts, ladders, today) -> Verdict` is deterministic: no network,
no LLM, no randomness, same input → same output. A test asserts it opens no sockets.

Ladders are **versioned YAML data, not code** (`data/ladders/`). When RBI changes the
scheme it is a pull request, not a retrain. Two ship today — RB-IOS and RTI 2005 — and
the second exists purely to prove the engine is general rather than hardcoded to one law.

Every `Verdict` carries the `source_url` and `verified_on` of the ladder it used, and
`blocked_by` is never empty when `maintainable` is false — each entry maps to a
plain-language message a human can read.

### Grounding — two tiers

| Tier | Source | Citable in a filing? |
|---|---|---|
| **A** | `data/statutes.json`, human-verified | **Yes**, with `source_url` + `verified_on` |
| **B** | Firecrawl, government domains only | **No** — background context only |

Tier A retrieval is keyword overlap over ~12 hand-verified snippets. No embeddings, no
vector DB. At this scale that is more accurate than RAG and infinitely more debuggable.

Tier B fires only when Tier A returns fewer than two snippets, so the common path never
touches the network. `sources.search()` calls Firecrawl with `includeDomains` set to
`sources.GOV_DOMAINS` — retrieval is restricted **at the API level**, not filtered
afterwards — and `sources.extract_snippet()` uses 105B to reduce each page to one short
statement.

`drafting.validate_citations()` builds its allowed set from Tier A **only**. Tier B ids
are never added, so any web-derived citation the model emits is deleted by the renderer
that was already there. No new safety code — and it makes the Q&A answer stronger:

> *"We do search the web — government domains only. And the renderer still deletes every
> citation that isn't human-verified. Here's a draft where it stripped one."*

`test_sources.py::test_web_citation_is_stripped_from_the_filing` asserts exactly this.
If it goes red, that claim has become false — treat it as a release blocker.

Promotion is how the corpus grows: a human reviews a Tier B snippet and writes it into
`data/statutes.json` with their name in `verified_by`. Editorial work, not engineering.

### Documents

`POST /api/document`, or just send a photo of the letter on WhatsApp. Sarvam **Doc AI
Extract** reads it — 22 Indian languages, scans and handwriting — and returns every field
with a `confidence`. Fields above `documents.CONFIDENCE_THRESHOLD` become facts; anything
below is **confirmed with the user**, never written in silently. A misread date would move
a legal deadline, so the product asks.

---

## WhatsApp channel (optional, off by default)

Meta Cloud API — a real WhatsApp Business number, not the Twilio sandbox, so there is
no `join <code>` ritual and judges can simply message it.

```bash
CHANNEL_WHATSAPP=1     # 0 = module never imported; a broken file cannot crash the demo
VERIFY_TOKEN=...       # any random string, same value in the Meta dashboard
APP_SECRET=...         # Meta App → Settings → Basic
WA_TOKEN=...           # permanent access token
PHONE_NUMBER_ID=...    # the id, not the phone number
```

Point the Meta webhook at `https://<your-app>/webhook`. Verify it:

```bash
curl "localhost:8000/webhook?hub.mode=subscribe&hub.verify_token=$VERIFY_TOKEN&hub.challenge=OK"
# -> OK        (wrong token -> 403, bad signature on POST -> 401)
```

Ported from `vinaypokharkar/turfbot`, keeping the four patterns that make it
production-grade: fast-ack then async work, HMAC signature verification, message-id
dedupe, and conversation state in the database. Voice notes arrive as ogg/opus, which
**Saaras accepts directly** — no ffmpeg anywhere in this project.

Proactive escalation inside 24 hours of an inbound message is free-form. Outside it,
WhatsApp requires an approved template — submit `haq_deadline_alert` (UTILITY) to Meta
on day zero, because approval has lead time and can be rejected.

---

## Deployment (Railway)

`Procfile` and `railway.json` are included.

**Attach a volume and set `HAQ_DB=/data/haq.db`.** Railway's container filesystem is
ephemeral; with the default `./haq.db` every redeploy wipes live cases mid-demo.

Record the cache locally and commit it — never run `HAQ_MODE=record` on Railway, since
the recordings would vanish on the next deploy.

---

## Python version

Pinned to **3.11** (`.python-version`). This is not superstition: the machine this was
built on runs 3.14, where wheels for the PDF stack are still patchy. Debugging a build
failure at hour 30 of a hackathon is not a good use of the hour.

---

## Legal verification — READ BEFORE DEMOING

Every ladder and statute entry currently carries `verified_by: UNVERIFIED`.

The legal content was drafted from the PRD, **not** confirmed against primary sources.
Before this is shown to judges, a named human must check every `source_url` and every
date, and replace `UNVERIFIED` with their own name. A judge who finds a wrong legal
claim will remember it, and "we verified each clause" is only an answer if it is true.

Files to verify: `data/ladders/*.yaml`, `data/statutes.json`.

---

## Onboarding

The user's entire first action is pressing send — the message is already written.

**WhatsApp:** a `wa.me` deep link carries the prefilled text (`onboarding.wa_link()`).
Sending it returns an interactive language list; picking one stores it and greets in it.
Put the link behind a QR code on the poster.

**Web:** the first panel is a chat opener with the same prefilled sentence, then language
chips from `GET /api/languages`.

Both surfaces read `onboarding.py` so they cannot drift, and **`cases.language IS NULL`
is the onboarding state** — no extra column, and nothing reaches the interrogation until
a language exists. HAQ never guesses one.

**Why eleven languages** when Saaras transcribes twenty-two: Bulbul speaks eleven. The
thesis is that she approves a document she cannot read *by hearing it*, so a language we
cannot speak back is one we should not offer as an interface. The other eleven remain
available for transcription. `test_onboarding.py` asserts the offered set equals the
speakable set.

## What is not built yet

Answer this honestly in Q&A — judges trust teams that name their gaps, and "nothing" is
never the right answer.

- **No Sarvam call has run against the live API yet.** The client is written to the
  verified specs and every path has a fallback, but until a key is present the code has
  only ever exercised those fallbacks. Doc AI and Firecrawl are in the same position.
- **Two ladders, not six.** RB-IOS and RTI.
- **Precedent data is seeded**, not real. `data/precedent.json` says so in the file, and
  `sample_size` is 0 on every row.
- **The Haq score is a rule-based formula**, not a model.
- **No auth, no accounts, no multi-tenancy.** One case per browser, one case per phone.
- **Legal content is unverified.** See above.

Deliberately unused from the Sarvam surface: Voice Agents, realtime STT streaming,
pronunciation dictionaries, transliteration, dubbing. Naming what you skipped and why
reads as judgment, not ignorance.

---

## Why these models

| Job | Model | Why |
|---|---|---|
| Speech in | `saaras:v3`, `mode=codemix` | v4 is newer but drops `mode`; we need code-mixed Marathi with English words in Latin script |
| Reasoning | `sarvam-105b` | 128K context; `wiki_grounding` is off so grounding comes only from our verified snippets |
| Translation | `mayura:v1`, `mode=formal` | She speaks Marathi, the bank receives formal English |
| Speech out | `bulbul:v3` | `opus` output feeds WhatsApp voice notes directly |

Two API details that cost an hour each if you miss them, both handled in `sarvam.py`:
chat completions needs **both** `Authorization: Bearer` and `api-subscription-key`,
while the other endpoints take only the latter; and TTS returns `audios` as a **list**.
Bulbul also reads numbers over four digits digit-by-digit unless they carry commas, so
draft text is comma-formatted before it is spoken.
