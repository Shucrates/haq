"""Onboarding tests.

Two properties matter here and both are easy to regress:
  1. we never guess a language, and
  2. we never offer a language we cannot speak back in.
"""

import pytest

import onboarding
import store
import whatsapp


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init()


@pytest.fixture
def sent(monkeypatch):
    log = []
    monkeypatch.setattr(whatsapp, "send_text", lambda to, body: log.append(("text", to, body)))
    monkeypatch.setattr(whatsapp, "send_language_list",
                        lambda to, rows: log.append(("list", to, rows)))
    monkeypatch.setattr(whatsapp, "send_buttons",
                        lambda to, body, buttons: log.append(("buttons", to, body)))
    return log


# ------------------------------------------------------------ the language set


def test_every_offered_language_can_be_spoken_back():
    """The read-back is the product thesis. Offering a language Bulbul cannot
    speak would mean she approves a document she never heard."""
    bulbul_supported = {
        "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN",
        "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
    }
    offered = {lang.code for lang in onboarding.LANGUAGES}
    assert offered <= bulbul_supported
    assert offered == bulbul_supported, "all eleven speakable languages should be offered"


def test_every_language_has_a_greeting_in_its_own_script():
    for lang in onboarding.LANGUAGES:
        greeting = onboarding.GREETINGS[lang.code]
        assert greeting.strip()
        if lang.code != "en-IN":
            assert not greeting.isascii(), f"{lang.code} greeting is not in its own script"


def test_option_id_round_trips():
    for lang in onboarding.LANGUAGES:
        assert onboarding.code_from_option(onboarding.option_id(lang.code)) == lang.code


@pytest.mark.parametrize("value", [None, "", "APPROVE", "lang_xx-XX", "lang_", "REJECT"])
def test_non_language_values_are_not_mistaken_for_a_pick(value):
    """'lang_' prefixing is what stops APPROVE being read as a language."""
    assert onboarding.code_from_option(value) is None


# --------------------------------------------------------------- the WA list


def test_list_never_exceeds_the_ten_row_cap():
    page = 0
    seen = 0
    while True:
        rows, has_more = onboarding.wa_rows(page)
        assert len(rows) <= onboarding.WA_LIST_MAX_ROWS, "Cloud API rejects >10 rows"
        seen += sum(1 for r in rows if r["id"] != onboarding.MORE_LANGUAGES)
        if not has_more:
            break
        page += 1
        assert page < 10, "paging did not terminate"
    assert seen == len(onboarding.LANGUAGES), "every language must be reachable"


def test_first_page_offers_the_demo_language_first():
    rows, _ = onboarding.wa_rows(0)
    assert rows[0]["id"] == onboarding.option_id("mr-IN")


def test_row_titles_fit_whatsapps_limit():
    rows, _ = onboarding.wa_rows(0)
    assert all(len(r["title"]) <= 24 for r in rows)


def test_wa_link_carries_the_prefilled_message():
    link = onboarding.wa_link("+91 98765 43210")
    assert link.startswith("https://wa.me/919876543210?text=")
    assert "Hi%20HAQ" in link


# ------------------------------------------------------------- the WA flow


def test_first_contact_gets_the_picker_not_the_interrogation(sent, monkeypatch):
    advanced = []
    monkeypatch.setattr(whatsapp, "_advance_case", lambda *a: advanced.append(a))

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text",
             "text": {"body": onboarding.PREFILL_TEXT}},
    )

    assert advanced == [], "nothing may proceed before a language is chosen"
    assert sent[-1][0] == "list"


def test_choosing_a_language_stores_it_and_greets_in_it(sent):
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text", "text": {"body": "hi"}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m2", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": "lang_mr-IN", "title": "मराठी"}}},
    )

    case = store.get_case(store.case_for_phone("9111"))
    assert case["language"] == "mr-IN"
    assert sent[-1][2] == onboarding.GREETINGS["mr-IN"]


def test_more_languages_pages_rather_than_dead_ends(sent):
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text", "text": {"body": "hi"}},
    )
    first_page = sent[-1][2]

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m2", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": onboarding.MORE_LANGUAGES, "title": "More"}}},
    )
    second_page = sent[-1][2]

    assert sent[-1][0] == "list"
    assert second_page != first_page
    assert store.get_case(store.case_for_phone("9111"))["language"] is None


def test_voice_note_before_onboarding_is_not_transcribed(sent, monkeypatch):
    """STT must not run before a language exists — otherwise the detected language
    silently overrides a choice the user never got to make."""
    called = []
    monkeypatch.setattr(whatsapp, "download_media", lambda mid: called.append(mid) or b"")

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "audio", "audio": {"id": "media-1"}},
    )

    assert called == [], "no media should be fetched before onboarding"
    assert sent[-1][0] == "list"


@pytest.mark.parametrize(
    "text", ["hi", "Hii", "heyy", "HELLO", "hello!", " hey ", "namaste", "नमस्कार"],
)
def test_bare_greetings_are_greetings(text):
    assert onboarding.is_greeting(text)


@pytest.mark.parametrize(
    "text",
    [onboarding.PREFILL_TEXT, "hi my bank charged me", "", None, "reset",
     "बँकेने पैसे कापले"],
)
def test_real_messages_are_not_greetings(text):
    assert not onboarding.is_greeting(text)


def test_greeting_reopens_the_picker_instead_of_refusing(sent, monkeypatch):
    """The bug this guards: "hi" was classified as a grievance, matched no ladder,
    and the person's hello was answered with `no_ladder_for_grievance`."""
    advanced = []
    monkeypatch.setattr(whatsapp, "_advance_case", lambda *a: advanced.append(a))

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text", "text": {"body": "hi"}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m2", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": "lang_mr-IN", "title": "मराठी"}}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m3", "type": "text", "text": {"body": "Heyy"}},
    )

    assert advanced == [], "a greeting is never a grievance"
    assert sent[-1][0] == "list", "a greeting re-offers the language picker"


def test_repicking_a_language_is_not_read_as_a_grievance(sent, monkeypatch):
    """The picker can be re-shown after a language exists, so the row tap must still
    route to onboarding — otherwise the row title becomes the complaint."""
    advanced = []
    monkeypatch.setattr(whatsapp, "_advance_case", lambda *a: advanced.append(a))

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text", "text": {"body": "hi"}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m2", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": "lang_mr-IN", "title": "मराठी"}}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m3", "type": "text", "text": {"body": "hi"}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m4", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": "lang_hi-IN", "title": "हिंदी"}}},
    )

    assert advanced == []
    assert store.get_case(store.case_for_phone("9111"))["language"] == "hi-IN"
    assert sent[-1][2] == onboarding.GREETINGS["hi-IN"]


def test_reset_returns_to_onboarding(sent):
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m1", "type": "text", "text": {"body": "hi"}},
    )
    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m2", "type": "interactive",
             "interactive": {"type": "list_reply",
                             "list_reply": {"id": "lang_hi-IN", "title": "हिंदी"}}},
    )
    assert store.get_case(store.case_for_phone("9111"))["language"] == "hi-IN"

    whatsapp.handle_inbound(
        {}, {"from": "9111", "id": "m3", "type": "text", "text": {"body": "reset"}},
    )
    assert store.get_case(store.case_for_phone("9111"))["language"] is None
    assert sent[-1][0] == "list"
