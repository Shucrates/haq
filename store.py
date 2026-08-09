"""SQLite helpers — plain sqlite3, no ORM (PRD 2.1).

DB path comes from HAQ_DB so Railway can point it at a mounted volume. Railway's
container filesystem is ephemeral: with the default ./haq.db every redeploy wipes
live cases mid-demo. Set HAQ_DB=/data/haq.db and attach a volume.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("HAQ_DB", Path(__file__).parent / "haq.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    channel       TEXT NOT NULL DEFAULT 'web',
    language      TEXT,
    transcript    TEXT,
    grievance_class TEXT,
    facts         TEXT NOT NULL DEFAULT '{}',
    turns         INTEGER NOT NULL DEFAULT 0,
    clock_offset_days INTEGER NOT NULL DEFAULT 0,
    draft_text    TEXT,
    approved_at   TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    native     TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS deadlines (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    deadline_id TEXT NOT NULL,
    label      TEXT NOT NULL,
    due_on     TEXT NOT NULL,
    tier       INTEGER NOT NULL,
    fired_at   TEXT,
    UNIQUE (case_id, deadline_id)
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL,
    at         TEXT NOT NULL
);

-- WhatsApp: phone -> case. Replaces turfbot's in-memory dict so state survives
-- a reload. Note there is no staleness reset: a legal case is not a turf booking.
CREATE TABLE IF NOT EXISTS wa_sessions (
    phone      TEXT PRIMARY KEY,
    case_id    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Meta redelivers webhooks. Without this, one voice note files two complaints.
CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    at         TEXT NOT NULL
);

-- Tier B: retrieved from government websites, NOT human-verified. Kept separate
-- from data/statutes.json on purpose — nothing in here may be cited in a filing
-- until a human promotes it (promoted=1) and writes it into statutes.json.
CREATE TABLE IF NOT EXISTS sources (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    url         TEXT NOT NULL,
    title       TEXT,
    snippet     TEXT NOT NULL,
    query       TEXT,
    provenance  TEXT NOT NULL DEFAULT 'web',
    promoted    INTEGER NOT NULL DEFAULT 0,
    retrieved_at TEXT NOT NULL,
    UNIQUE (case_id, source_id)
);

-- What we read out of a document the user sent us.
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    filename   TEXT,
    status     TEXT NOT NULL,
    extracted  TEXT NOT NULL DEFAULT '{}',
    at         TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# ------------------------------------------------------------------- cases


def create_case(channel: str = "web") -> str:
    case_id = uuid.uuid4().hex[:12]
    with connect() as conn:
        conn.execute(
            "INSERT INTO cases (id, created_at, channel) VALUES (?, ?, ?)",
            (case_id, _now(), channel),
        )
    return case_id


def get_case(case_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
    if row is None:
        return None
    case = dict(row)
    case["facts"] = json.loads(case["facts"] or "{}")
    return case


def update_case(case_id: str, **fields: Any) -> None:
    if not fields:
        return
    if "facts" in fields and not isinstance(fields["facts"], str):
        fields["facts"] = json.dumps(fields["facts"], default=str)
    columns = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE cases SET {columns} WHERE id = ?", (*fields.values(), case_id))


def bump_turns(case_id: str) -> int:
    with connect() as conn:
        conn.execute("UPDATE cases SET turns = turns + 1 WHERE id = ?", (case_id,))
        row = conn.execute("SELECT turns FROM cases WHERE id = ?", (case_id,)).fetchone()
    return row["turns"] if row else 0


# ---------------------------------------------------------------- messages


def add_message(case_id: str, role: str, text: str, native: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO messages (case_id, role, text, native, created_at) VALUES (?,?,?,?,?)",
            (case_id, role, text, native, _now()),
        )


def messages(case_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, text, native, created_at FROM messages WHERE case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- deadlines


def save_deadlines(case_id: str, deadlines: list) -> None:
    with connect() as conn:
        for d in deadlines:
            conn.execute(
                """INSERT INTO deadlines (case_id, deadline_id, label, due_on, tier)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT (case_id, deadline_id)
                   DO UPDATE SET label=excluded.label, due_on=excluded.due_on, tier=excluded.tier""",
                (case_id, d.id, d.label, d.due_on.isoformat(), d.tier),
            )


def due_deadlines(case_id: str, as_of: date) -> list[dict]:
    """Unfired deadlines whose date has arrived. Marks them fired so the escalation
    only ever happens once."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM deadlines WHERE case_id = ? AND fired_at IS NULL AND due_on <= ?"
            " ORDER BY due_on",
            (case_id, as_of.isoformat()),
        ).fetchall()
        for row in rows:
            conn.execute("UPDATE deadlines SET fired_at = ? WHERE id = ?", (_now(), row["id"]))
    return [dict(r) for r in rows]


def all_deadlines(case_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT deadline_id, label, due_on, tier, fired_at FROM deadlines"
            " WHERE case_id = ? ORDER BY due_on",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------- events


def add_event(case_id: str, kind: str, detail: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (case_id, kind, detail, at) VALUES (?,?,?,?)",
            (case_id, kind, detail, _now()),
        )


def events(case_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT kind, detail, at FROM events WHERE case_id = ? ORDER BY id", (case_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# -------------------------------------------------------------- whatsapp


def case_for_phone(phone: str) -> str:
    """Map a WhatsApp number to a case, creating one on first contact."""
    with connect() as conn:
        row = conn.execute("SELECT case_id FROM wa_sessions WHERE phone = ?", (phone,)).fetchone()
        if row:
            return row["case_id"]
    case_id = create_case(channel="whatsapp")
    with connect() as conn:
        conn.execute(
            "INSERT INTO wa_sessions (phone, case_id, updated_at) VALUES (?,?,?)",
            (phone, case_id, _now()),
        )
    return case_id


def phone_for_case(case_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT phone FROM wa_sessions WHERE case_id = ? ORDER BY updated_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
    return row["phone"] if row else None


def reset_phone(phone: str) -> str:
    """Start a fresh case for this number — used by the demo reset command."""
    with connect() as conn:
        conn.execute("DELETE FROM wa_sessions WHERE phone = ?", (phone,))
    return case_for_phone(phone)


# -------------------------------------------------- tier B sources + documents


def save_sources(case_id: str, web_sources: list, query: str = "") -> None:
    """Record what we read off the web. Never merged into statutes.json — promotion
    is a human act, and that is the whole point of keeping the tiers apart."""
    with connect() as conn:
        for s in web_sources:
            conn.execute(
                """INSERT INTO sources
                       (case_id, source_id, url, title, snippet, query, provenance, retrieved_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT (case_id, source_id) DO UPDATE SET snippet=excluded.snippet""",
                (case_id, s["id"], s["url"], s.get("title"), s["text"], query,
                 s.get("provenance", "web"), _now()),
            )


def sources_for(case_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT source_id, url, title, snippet, provenance, promoted FROM sources"
            " WHERE case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_document(case_id: str, filename: str | None, status: str, extracted: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (case_id, filename, status, extracted, at) VALUES (?,?,?,?,?)",
            (case_id, filename, status, json.dumps(extracted, default=str), _now()),
        )


def documents_for(case_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT filename, status, extracted, at FROM documents WHERE case_id = ? ORDER BY id",
            (case_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["extracted"] = json.loads(item["extracted"] or "{}")
        out.append(item)
    return out


def already_processed(message_id: str) -> bool:
    """True if we've seen this Meta message id before (and records it if new)."""
    if not message_id:
        return False
    with connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO processed_messages (message_id, at) VALUES (?,?)",
            (message_id, _now()),
        )
    return cur.rowcount == 0
