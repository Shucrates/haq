"""WhatsApp adapter tests — the security surface, not the happy path.

An unsigned webhook is an endpoint anyone can POST complaints into, and a missing
dedupe means one voice note files two complaints. Both are cheap to get wrong and
invisible until they matter.
"""

import hashlib
import hmac
import json

import pytest

import agent
import store
import whatsapp


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init()


@pytest.fixture(autouse=True)
def localised(monkeypatch):
    """Record what was put through the translator and hand back the English.

    Autouse so no test can reach Sarvam: these cases carry language='mr-IN', and a
    developer with SARVAM_API_KEY exported would otherwise translate every assertion
    in this file over the network.
    """
    calls = []

    def spy(text, language, limit=None):
        calls.append((text, language))
        return text

    agent.localise.cache_clear()
    monkeypatch.setattr(agent, "localise", spy)
    return calls


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


# ----------------------------------------------------- the approval gate


@pytest.fixture
def sent(monkeypatch):
    """Capture outbound messages instead of calling Meta."""
    log = []
    monkeypatch.setattr(whatsapp, "send_text", lambda to, body: log.append(("text", to, body)))
    monkeypatch.setattr(whatsapp, "send_buttons",
                        lambda to, body, buttons: log.append(("buttons", to, body)))
    return log


@pytest.fixture
def drafted():
    """An onboarded case with a draft on the table, awaiting a human decision."""
    case_id = store.case_for_phone("91555")
    store.update_case(case_id, language="mr-IN",
                      draft_text="Dear Sir, I am writing regarding...")
    store.save_deadlines(case_id, [])
    return case_id


def test_awaiting_approval_only_between_draft_and_approval(drafted):
    assert whatsapp._awaiting_approval(store.get_case(drafted)) is True
    store.update_case(drafted, approved_at="2026-08-09T10:00:00")
    assert whatsapp._awaiting_approval(store.get_case(drafted)) is False


def test_no_draft_means_not_awaiting_approval():
    case_id = store.create_case()
    assert whatsapp._awaiting_approval(store.get_case(case_id)) is False


@pytest.mark.parametrize("action,text,expected", [
    ("APPROVE", "Yes, file it", "APPROVE"),   # button id wins
    ("REJECT", "No, change it", "REJECT"),
    (None, "हो", "APPROVE"),                  # the word from the demo script
    (None, "haan", "APPROVE"),
    (None, "नाही", "REJECT"),
    (None, "no", "REJECT"),
    (None, "what does this mean?", None),     # ambiguous -> never assume consent
    (None, "", None),
])
def test_decision_parsing(action, text, expected):
    assert whatsapp._decision_from(action, text) == expected


def test_approve_records_a_human_approval(drafted, sent):
    whatsapp._handle_decision("91555", drafted, "APPROVE", "Yes, file it")
    case = store.get_case(drafted)
    assert case["approved_at"], "approval must be recorded — we never file autonomously"
    assert any(e["kind"] == "approved" for e in store.events(drafted))
    assert sent[0][0] == "text"


def test_reject_clears_the_draft_so_input_flows_again(drafted, sent):
    whatsapp._handle_decision("91555", drafted, "REJECT", "No, change it")
    case = store.get_case(drafted)
    assert case["draft_text"] is None
    assert case["approved_at"] is None
    assert whatsapp._awaiting_approval(case) is False


def test_ambiguous_reply_re_asks_instead_of_approving(drafted, sent):
    """Approval is the one place an LLM must not interpret intent."""
    whatsapp._handle_decision("91555", drafted, None, "hmm not sure")
    assert store.get_case(drafted)["approved_at"] is None
    assert sent[-1][0] == "buttons"


def test_what_haq_says_is_translated_before_it_is_sent(drafted, sent, localised):
    """The gap this closes: the questions were in Marathi and everything around them
    — the confirmation, the approval prompt, the deadline list — was in English."""
    whatsapp._handle_decision("91555", drafted, "APPROVE", "Yes, file it")

    assert localised, "the reply went out without passing through the translator"
    text, language = localised[-1]
    assert language == "mr-IN", "the language must come from the case, never a default"
    assert "Filed." in text


def test_the_approval_buttons_are_translated_too(drafted, localised, monkeypatch):
    """A Marathi prompt over two English buttons is the same bug, one line lower."""
    buttons = []
    monkeypatch.setattr(whatsapp, "send_buttons",
                        lambda to, body, btns: buttons.append(btns))

    whatsapp._ask_to_file("91555", drafted, "Shall I prepare this for filing?")

    titles = [title for _, title in buttons[0]]
    assert titles == ["Yes, file it", "No, change it"]  # the spy returns the English
    assert [ids for ids, _ in buttons[0]] == [whatsapp.APPROVE, whatsapp.REJECT], \
        "the ids must stay untranslated — the decision is read off the id"
    assert ("Yes, file it", "mr-IN") in localised


def test_button_tap_does_not_get_recorded_as_a_fact(drafted, sent, monkeypatch):
    """The bug this fixes: a button tap used to fall through to the interrogation,
    be parsed as an answer, and trigger a re-draft."""
    called = []
    monkeypatch.setattr(whatsapp, "_advance_case", lambda *a: called.append(a))

    whatsapp.handle_inbound(
        {"contacts": [{"profile": {"name": "Sunita"}}]},
        {"from": "91555", "id": "btn-1", "type": "interactive",
         "interactive": {"type": "button_reply",
                         "button_reply": {"id": "APPROVE", "title": "Yes, file it"}}},
    )

    assert called == [], "a decision must not reach the fact collector"
    assert store.get_case(drafted)["approved_at"]


# ------------------------------------------------- debug echo (off by default)


def test_document_failure_hides_the_error_by_default(monkeypatch, sent):
    """A user must never be shown a stack trace."""
    monkeypatch.delenv("HAQ_DEBUG", raising=False)
    monkeypatch.setattr(whatsapp, "download_media",
                        lambda mid: (_ for _ in ()).throw(RuntimeError("Sarvam 422: boom")))
    case_id = store.case_for_phone("91666")

    whatsapp._handle_document("91666", case_id, {"media_id": "m", "filename": "x.pdf"}, "en-IN")

    assert "422" not in sent[-1][2]
    assert "couldn't read that" in sent[-1][2]


def test_debug_flag_surfaces_the_real_error(monkeypatch, sent):
    """The whole point: a misconfigured deployment must be distinguishable from a
    blurry photo by the person holding the phone."""
    monkeypatch.setenv("HAQ_DEBUG", "1")
    monkeypatch.setattr(whatsapp, "download_media",
                        lambda mid: (_ for _ in ()).throw(RuntimeError("Sarvam 422: boom")))
    case_id = store.case_for_phone("91667")

    whatsapp._handle_document("91667", case_id, {"media_id": "m", "filename": "x.pdf"}, "en-IN")

    assert "Sarvam 422: boom" in sent[-1][2]
