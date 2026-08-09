"""Offline replay (PRD 7) — the most important 40 lines in the repo.

Never demo live against a third-party API on venue wifi. Every Sarvam call goes
through cached(). Three modes:

    live    call the API, cache nothing
    record  call the API and write the response to data/cache/
    replay  read from data/cache/ and never touch the network

Demo-day protocol: HAQ_MODE=replay on the presenting laptop, always.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

CACHE = Path(__file__).parent / "data" / "cache"
REPLAY_LATENCY_SECONDS = 0.4  # keep it feeling real, not instant


def mode() -> str:
    return os.getenv("HAQ_MODE", "live").lower()


def key_for(*parts: Any) -> str:
    """Stable key from arbitrary inputs. Bytes are hashed, everything else is
    JSON-serialised, so the same call always lands on the same file."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, (bytes, bytearray)):
            h.update(hashlib.sha256(part).hexdigest().encode())
        else:
            h.update(json.dumps(part, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()[:24]


class CacheMiss(RuntimeError):
    """Raised in replay mode when a call was never recorded."""


def cached(name: str, key: str, fn: Callable[[], Any]) -> Any:
    """Wrap one API call. `name` groups by function so cache files stay readable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.{key}.json"
    current = mode()

    if current == "replay":
        if not path.exists():
            raise CacheMiss(
                f"No cached response for {name} ({key}). "
                f"Run the demo once with HAQ_MODE=record before switching to replay."
            )
        time.sleep(REPLAY_LATENCY_SECONDS)
        return json.loads(path.read_text())

    out = fn()

    if current == "record":
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    return out


def recorded_keys() -> list[str]:
    """Used by /api/health so you can see on stage what is cached."""
    if not CACHE.exists():
        return []
    return sorted(p.name for p in CACHE.glob("*.json"))
