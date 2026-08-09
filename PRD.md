# HAQ — BUILD PRD v2

*Supersedes `HAQ BUILD PRD.pdf` (v1), which is kept in the repo as the original
planning artifact. This version describes what was actually built and what is
actually true. Where v1 was wrong, that is stated rather than quietly corrected —
a PRD nobody trusts is worse than no PRD.*

**Governing rule (unchanged):** if it is not on stage, it does not get built.

| | |
|---|---|
| Team | Hack_Overflow (4) |
| Theme | AI Innovation with Sarvam AI |
| Deliverable | One live demo, one repo, one 3-min video |
| Status | 11 modules · 3,268 lines · **145 tests passing** · 15 endpoints |
| Repo | https://github.com/Shucrates/haq |

---

## What changed since v1

Seven things. Four are corrections — v1 asserted things that turned out to be false.

| v1 said | Reality |
|---|---|
| **The audio trap**: browsers emit webm/opus, Saaras wants wav/mp3; build a browser WAV encoder; "the single most likely cause of a dead demo" | **Wrong.** Saaras accepts WebM, OGG, Opus, WAV, MP3, AAC, FLAC, M4A, AMR and PCM directly. No encoder, no ffmpeg anywhere in the project. This also made WhatsApp voice notes free — they arrive as ogg/opus |
| TTS: chunk to 500 chars, returns base64 | bulbul:v3 caps at **2500**, and returns `audios` as a **list** — reading it as a string is a one-line bug that looks like a broken demo |
| WhatsApp = Twilio Sandbox, 4 hrs, **cut #1** | Built on **Meta Cloud API** by porting patterns from `vinaypokharkar/turfbot`. A real Business number, so judges can message it — no `join <code>` ritual |
| Sarvam Vision OCR — P1, cut | **Built.** Sarvam Doc AI Extract, with per-field confidence |
| No RAG pipeline, ever | **Built, in a constrained form.** Government-domain-only retrieval, context tier, never citable |
| — | **Onboarding added.** Did not exist in v1; the product assumed a language |
| — | **Deployment target: Railway.** v1 assumed a laptop |

Two new API constraints v1 did not know about, both of which shape the demo:

- **STT REST is for audio under 30 seconds.** Sunita's "rambling grievance" must be
  scripted to ~20s or it needs the Batch API, which is far too slow for stage.
- **Chat completions needs two auth headers** (`Authorization: Bearer` *and*
  `api-subscription-key`); the other endpoints need only the second. Getting this wrong
  produces a 401 that looks like a bad key.

---

## Part 1 — Scope

### What is built

| # | Component | State |
|---|---|---|
| C0 | **Onboarding** — prefilled message, then language selection, both channels | Built |
| C1 | **Voice intake** — record → Saaras v3 (`mode=codemix`) → transcript | Built |
| C2 | **Interrogation agent** — 105B, one question per turn, fills a fact sheet | Built |
| C3 | **Ladder Engine** — pure function, 2 ladders, unit-tested | Built · the moat |
| C4 | **Document generator** — template → 105B → grounding validator → Mayura → PDF | Built |
| C5 | **Bulbul read-back** — English draft spoken in her language | Built |
| C6 | **Time-travel daemon** — advance clock, escalation fires | Built |
| C7 | **Web UI** — single file, vanilla JS, no build step | Built |
| C8 | **WhatsApp channel** — Meta Cloud API | Built |
| C9 | **Document intake** — Sarvam Doc AI Extract, confidence-gated | Built |
| C10 | **Web retrieval** — Firecrawl, government domains only, context tier | Built |

### What is still not built — say this in Q&A

- **Two ladders, not six.** RB-IOS 2026 and RTI 2005.
- **Precedent data is seeded.** `data/precedent.json` declares `sample_size: 0` on every row.
- **Haq score is a rule-based formula**, not a model. Deliberate.
- **No accounts or multi-tenancy.** Cases are *owned* — bound to the browser session that
  opened them (httponly `haq_sid` cookie) or to the WhatsApp number that started them, and
  every case route answers 404 to anyone else (`test_auth.py`). That is ownership, not
  identity: no login, no recovery, and clearing cookies orphans the case.
- **Legal content is `UNVERIFIED`.** See Part 9 — this is the blocking item.
- **No Sarvam, Doc AI or Firecrawl call has run against a live endpoint.** Every path
  has a fallback; so far only the fallbacks have executed.

Still explicitly not built: user accounts, database migrations, Docker, CI/CD, embeddings
or a vector DB, an operator console, a mobile app, payments, Next.js.

**The Next.js decision held.** Vanilla HTML against FastAPI, served as a static file.

---

## Part 2 — Architecture

**One process.** `main.py` is the only entrypoint; everything else is a library it
imports. Route handlers parse and return — all logic lives in plain functions. That rule
is why the WhatsApp adapter reuses the product instead of duplicating it.

```
uvicorn main:app
  ├── ladder_engine.py   269   pure function — no network, no LLM, no randomness
  ├── agent.py           339   4-state machine with an LLM inside it
  ├── main.py            519   15 endpoints, all thin
  ├── whatsapp.py        522   Meta Cloud API, off by default
  ├── store.py           334   SQLite, no ORM
  ├── sarvam.py          250   stt / chat / translate / tts / doc-ai
  ├── drafting.py        246   retrieve → draft → validate → translate → PDF
  ├── sources.py         184   Tier B web retrieval
  ├── documents.py       176   Doc AI Extract, confidence-gated
  ├── onboarding.py      128   languages + prefill, shared by both channels
  └── demo_cache.py       74   live | record | replay
```

Stack: Python 3.11, FastAPI, SQLite (plain `sqlite3`), PyYAML, fpdf2, httpx. Frontend:
one HTML file, no framework, no npm.

### API — 15 endpoints

```
GET  /api/languages              the 11 offered languages + prefilled opening message
POST /api/start                  create a case; language is NULL until chosen
POST /api/onboard                record chosen language, greet in it
POST /api/intake                 audio → transcript, language, classification, confidence
POST /api/intake_text            same, typed
POST /api/document               Doc AI → confident fields become facts, rest confirmed
POST /api/turn                   one question per turn
POST /api/resolve                THE REFUSAL — pure function, no LLM
POST /api/draft                  grounded letter + PDF
GET  /api/draft/{id}.pdf
POST /api/speak                  Bulbul, in the case's language
POST /api/approve                records human approval — we never file autonomously
POST /api/advance                the time machine
GET  /api/case/{id}
GET  /api/health
GET/POST /webhook                WhatsApp, only when CHANNEL_WHATSAPP=1
```

---

## Part 3 — The Ladder Engine (unchanged, and still the product)

`resolve(facts, ladders, today) -> Verdict`. No network, no LLM, no randomness. Same
input, same output, always. A test asserts it opens no sockets — the "pure function"
claim is enforced by the suite, not by good intentions.

- `blocked_by` is never empty when `maintainable` is False, and each entry maps to a
  plain-language message from the YAML.
- Every `Verdict` carries `source_url` and `verified_on`.
- Ladders are **versioned YAML data, not code**: when RBI changes the scheme it is a pull
  request, not a retrain.

**Judge moment:** run `pytest -v` on the projector. 145 green. Do not explain the tests.

---

## Part 4 — Sarvam integration (corrected against the live docs)

| Job | Model | The thing that bites |
|---|---|---|
| `stt` | `saaras:v3`, `mode=codemix` | v4 drops `mode`; **REST caps at 30 seconds** |
| `chat` | `sarvam-105b` | needs **both** auth headers; `wiki_grounding` stays False |
| `translate` | `mayura:v1`, `mode=formal` | **1000-char cap** forces chunking on a real letter |
| `tts` | `bulbul:v3` | returns `audios` as a **list**; no pitch/loudness; `opus` feeds WhatsApp |
| `doc-ai` | Extract | async: submit → poll status → fetch results (409 if not terminal) |

Two details that decide whether the demo sounds right:

- **`language_probability`** comes back free when `language_code=unknown`. That is the
  confidence number on screen — no separate classifier.
- **Numbers over four digits are read digit-by-digit unless they carry commas.** Draft
  text is comma-formatted before it is spoken, or "₹10000" becomes "one zero zero zero zero".

Docs trick worth keeping: append `.md` to any `docs.sarvam.ai` page for clean Markdown,
`/llms.txt` for a section index. There is also an MCP server at
`https://docs.sarvam.ai/_mcp/server`.

---

## Part 5 — Onboarding

The user's entire first action is pressing send. The message is already written.

**WhatsApp:** a `wa.me` deep link carries the prefilled text; sending it returns an
interactive language list (capped at 10 rows by the Cloud API, so it pages); picking one
stores it and greets in that language. Put the link behind a QR code.

**Web:** the first panel is a chat opener with the same sentence, then language chips.

Both read `onboarding.py`, and **`cases.language IS NULL` is the onboarding state** — no
extra column, and nothing reaches the interrogation until a language exists.

**Eleven languages, not twenty-two.** Saaras transcribes 22; Bulbul speaks 11. The thesis
is that she approves a document she cannot read *by hearing it*, so a language we cannot
speak back is one we should not offer as an interface. A test asserts the offered set
equals the speakable set.

---

## Part 6 — Grounding, in two tiers

| Tier | Source | Citable in a filing? |
|---|---|---|
| **A** | `data/statutes.json` — 12 snippets, human-verified | **Yes**, with `source_url` + `verified_on` |
| **B** | Firecrawl, 10 government domains | **No** — background context only |

Tier A retrieval is keyword overlap. No embeddings, no vector DB: at this scale it is
more accurate and infinitely more debuggable.

Tier B fires only when Tier A returns fewer than two snippets, so the common path never
touches the network. Search is restricted with `includeDomains` **at the API level**, not
filtered afterwards.

`validate_citations()` runs two passes. The first builds its allowed set from Tier A alone,
so any web-derived `[citation]` the model emits is deleted. The second deletes any sentence
stating a *numbered* legal reference — `Section 7(1)`, `Regulation 12` — that carries no
surviving citation, and `grounded` is false if either pass removed anything.

Pass two exists because pass one was weaker than this document used to claim. It could only
delete a marker, so an invented "Section 7(1) of the RTI Act requires disposal within thirty
days", with no bracket anywhere, went through untouched and still reported `grounded: true`.
The marker was never there to strip.

> **On stage:** "We do search the web — government domains only. The renderer deletes every
> citation that isn't human-verified, and every legal claim that isn't cited at all. Here's
> a draft where it stripped both."

`test_sources.py::test_web_citation_is_stripped_from_the_filing` asserts pass one and
`test_an_uncited_section_number_does_not_survive` asserts pass two. If either goes red, that
sentence has become a lie. Release blocker.

**Promotion** is how the corpus grows: a human reviews a Tier B snippet and writes it into
`statutes.json` with their name in `verified_by`. Editorial work, not engineering.

---

## Part 7 — Documents

A photo of the bank's letter is the densest object in the process. Sarvam Doc AI Extract
reads it — 22 languages, scans, handwriting — and returns **a confidence per field**.

Above `CONFIDENCE_THRESHOLD` (0.75) a field becomes a fact. Below it, the user is asked.
**Missing confidence counts as uncertain, not as high.** A misread date moves a legal
deadline, so the product asks rather than assumes.

Works from the web upload and from a photo sent on WhatsApp.

---

## Part 8 — Demo mode (unchanged, still the most important section)

Never demo live against a third-party API on venue wifi.

```bash
HAQ_MODE=record   # run the demo once, locally, recording
HAQ_MODE=replay   # on stage. always.
```

Every network call — Sarvam, Doc AI, **and Firecrawl** — goes through `demo_cache.cached()`.
`replay` never opens a socket and sleeps 400ms per call so it still feels real.

**Deployment note:** the target is Railway, which puts the demo on venue wifi — the exact
failure this section exists to prevent. Keep a laptop on `HAQ_MODE=replay` as the backup;
it is the same repo and one command. Railway's filesystem is ephemeral, so `HAQ_DB` must
point at a mounted volume or every redeploy wipes live cases.

---

## Part 9 — Q&A drill

Answers under 25 seconds. The first four are the ones that decide it.

1. **"How do you stop it hallucinating law?"** → Two layers. The ladder is a tested pure
   function with no LLM in it, and the renderer deletes any citation without a source URL
   retrieved this session. Show `validate_citations`.
2. **"You search the web — so how is that verified?"** *(new)* → It isn't, and we never
   claim it is. Web text is context the drafter may read; only human-verified snippets can
   be cited. The renderer enforces it. Here is a draft where it stripped one.
3. **"What if the document scan is wrong?"** *(new)* → Doc AI returns a confidence per
   field. Below threshold we ask instead of assuming — a misread date would move a legal
   deadline.
4. **"Is this legal advice?"** → No. Document preparation and deadline tracking. The user
   files; we prepare. We never file autonomously — every outbound document has a recorded
   human approval.
5. **"What happens when the law changes?"** → It already did. That is why ladders are
   versioned YAML with `effective_from` and `verified_on`, not code.
6. **"Why Sarvam?"** → Code-mixed Marathi over a phone mic, native numerals, telephony
   audio, Devanagari handwriting. Our entire input distribution is what Sarvam is built for.
7. **"How do you scale to hundreds of ladders?"** → A few hours of a paralegal's time each.
   An editorial pipeline, not an engineering one. Deliberate.
8. **"What's your moat?"** → The ladders take a month to copy. The outcome corpus — what
   works against which institution at which tier — compounds and can't be.
9. **"How do people find you?"** → CSC operators, ASHA workers, gig unions, tenant
   associations. One operator serves hundreds.
10. **"Could this be weaponised?"** → Respondents must be institutions, enforced at the
    schema level.
11. **"How do you make money?"** → Free for individuals. Lawyer referrals, operator
    subscriptions, union licensing, aggregate data for regulators.
12. **"What's not working yet?"** → Answer from Part 1 honestly. Never say "nothing".

---

## Part 10 — Blocking items before demo

1. **Verify the legal content.** Every ladder and statute entry says
   `verified_by: UNVERIFIED`. A named human must check each `source_url` and date and put
   their name in. The repo is **public** — anyone can read this content as if it were
   authoritative. This is the single largest risk to credibility.
2. **Run every Sarvam path once against the live API.** Nothing has yet executed against a
   real endpoint; only fallbacks have run.
3. **Record the cache**, commit it, switch to `replay`.
4. **Rotate the keys leaked in the old TGBH repo** — Groq, Mistral, ngrok, Cloudinary, and
   a Sarvam key that was shipped to the browser as `NEXT_PUBLIC_`.
5. Attach the Railway volume. Submit the `haq_deadline_alert` template to Meta.
6. Rehearse the grievance at **~20 seconds** — the STT limit is 30.

---

## Part 11 — Risks

| Risk | Mitigation |
|---|---|
| A judge finds a wrong legal claim | Part 10 item 1. Currently unmitigated |
| Live API behaves differently from the docs | Part 10 item 2 — find out before the venue, not on stage |
| Venue wifi / Railway outage | `HAQ_MODE=replay` on a backup laptop |
| Web retrieval erodes the verified-citation story | Tier B never enters the allowed set; a test asserts stripping |
| 105B returns malformed JSON | Retry once, then rule-based fallback. Never surfaces |
| Scope creep | 11 components is already past v1's list. Nothing further before the demo |
| Team exhaustion | Sleep in the last two hours. Enforced by the lead |

---

## The one-sentence version

The engine refuses, the tests prove it, the cache means it cannot fail on stage — and the
only thing standing between this and a credible submission is a human verifying the law.
