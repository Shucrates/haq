"""WhatsApp adapter tests — the security surface, not the happy path.

An unsigned webhook is an endpoint anyone can POST complaints into, and a missing
dedupe means one voice note files two complaints. Both are cheap to get wrong and
invisible until they matter.
"""

import hashlib
import hmac
import json

import pytest

import store
import whatsapp


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init()


SECRET = "test_app_secret"


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_valid_signature_accepted():
    body = b'{"entry":[]}'
    assert whatsapp.verify_meta(body, sign(body), SECRET) is True


def test_tampered_body_rejected():
    body = b'{"entry":[]}'
    header = sign(body)
    assert whatsapp.verify_meta(b'{"entry":[1]}', header, SECRET) is False


def test_wrong_secret_rejected():
    body = b'{"entry":[]}'
    assert whatsapp.verify_meta(body, sign(body, "other_secret"), SECRET) is False


def test_missing_signature_rejected():
    assert whatsapp.verify_meta(b"{}", None, SECRET) is False


def test_missing_app_secret_rejects_rather_than_allows():
    """Fail closed. An unset APP_SECRET must not mean 'accept everything'."""
    body = b"{}"
    assert whatsapp.verify_meta(body, sign(body), "") is False


def test_dedupe_fires_once_per_message_id():
    assert store.already_processed("wamid.ABC") is False  # first sighting
    assert store.already_processed("wamid.ABC") is True   # Meta redelivered
    assert store.already_processed("wamid.XYZ") is False


def test_parse_inbound_text():
    value = {"contacts": [{"profile": {"name": "Sunita"}}]}
    message = {"from": "919", "id": "m1", "type": "text", "text": {"body": "hello"}}
    out = whatsapp.parse_inbound(value, message)
    assert out["text"] == "hello"
    assert out["name"] == "Sunita"
    assert out["media_id"] is None


def test_parse_inbound_voice_note_carries_media_id():
    """The audio branch is the whole point of HAQ on WhatsApp — turfbot had no
    equivalent, so this is new code and worth pinning."""
    message = {"from": "919", "id": "m2", "type": "audio", "audio": {"id": "media-123"}}
    out = whatsapp.parse_inbound({}, message)
    assert out["media_id"] == "media-123"
    assert out["type"] == "audio"


def test_parse_inbound_button_reply():
    message = {
        "from": "919", "id": "m3", "type": "interactive",
        "interactive": {"type": "button_reply", "button_reply": {"id": "APPROVE", "title": "Yes"}},
    }
    out = whatsapp.parse_inbound({}, message)
    assert out["type"] == "button"
    assert out["value"] == "APPROVE"


def test_phone_maps_to_a_stable_case():
    first = store.case_for_phone("919999")
    assert store.case_for_phone("919999") == first  # same number, same case
    assert store.case_for_phone("918888") != first  # different number, new case


def test_reset_starts_a_new_case_for_the_same_phone():
    original = store.case_for_phone("917777")
    assert store.reset_phone("917777") != original
