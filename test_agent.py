"""Interrogation completeness.

A ladder can require a fact the interrogation has no way to ask for. When that
happens nothing errors: pending_facts() silently drops the fact, next_question()
reports done, and the case sits unresolvable forever. These tests are the guard.
"""

import pytest

import agent
import store
from ladder_engine import load_ladders, required_facts

L = load_ladders()


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    store.init()


def test_every_required_fact_is_askable():
    """The trap: pending_facts() filters `required` through ASK_ORDER, so a required
    fact missing from it is never asked and never fills. One assertion stops the
    whole class of bug, for every ladder, forever."""
    for ladder_id, ladder in L.items():
        for name in required_facts(ladder):
            assert name in agent.ASK_ORDER, f"{ladder_id} requires {name}, nothing asks for it"
            assert name in agent.FALLBACK_QUESTIONS, f"no fallback question for {name}"
            assert name in agent.FACT_TYPES, f"no parser type for {name}"


def test_exclusion_facts_are_actually_reached():
    required = required_facts(L["rbi_rbios_2026"])
    pending = agent.pending_facts({}, required)
    assert "pending_in_court" in pending
    assert "commercial_decision" in pending


def test_every_ladder_fits_inside_the_turn_cap():
    """MAX_TURNS is a hard stop. If a ladder needs more facts than turns, the case
    can never become maintainable no matter how honestly the user answers."""
    for ladder_id, ladder in L.items():
        needed = len(required_facts(ladder))
        assert needed <= agent.MAX_TURNS, f"{ladder_id} needs {needed} facts, cap is {agent.MAX_TURNS}"


def test_interrogation_terminates_once_every_required_fact_is_known():
    case_id = store.create_case()
    facts = {name: False for name in required_facts(L["rbi_rbios_2026"])}
    facts.update(tier1_written_complaint=True, tier1_filed_on="2026-06-01",
                 last_communication_on="2026-06-15")
    store.update_case(case_id, facts=facts)

    step = agent.next_question(case_id, required_facts(L["rbi_rbios_2026"]))

    assert step["done"] is True
    assert step["facts_pending"] == []


def test_hitting_the_turn_cap_reports_what_is_still_missing(monkeypatch):
    """Failing closed only works if the caller can see why. An empty facts_pending
    at the cap would read as 'ready to resolve'."""
    case_id = store.create_case()
    store.update_case(case_id, facts={"tier1_written_complaint": True})
    monkeypatch.setattr(agent, "MAX_TURNS", 0)

    step = agent.next_question(case_id, required_facts(L["rbi_rbios_2026"]))

    assert step["done"] is True
    assert "commercial_decision" in step["facts_pending"]


# ------------------------------------------------------------------ guidance


@pytest.mark.parametrize("text,intent", [
    ("bank ne mere account se paise kaat liye", "grievance"),  # a keyword fires
    ("mala loan sathi arj karaycha aahe", "question"),         # nothing to escalate
])
def test_the_keyword_fallback_still_picks_an_intent(text, intent):
    """The fallback runs when 105B is unreachable. Marching someone who asked how to
    apply for a loan through a six-question interrogation is the bug it must avoid."""
    assert agent._classify_by_keyword(text)["intent"] == intent


@pytest.fixture
def guidance(monkeypatch):
    """Answer with whatever the test wants 105B to have said, and keep Tier B off
    the network."""
    monkeypatch.setattr(agent.sources, "context_for", lambda *a, **k: [])

    def reply(text):
        monkeypatch.setattr(agent.sarvam, "chat", lambda *a, **k: text)

    return reply


def test_guidance_deletes_an_invented_section_number(guidance):
    """The grounding guarantee is code, not a prompt: an answer about which documents
    a bank wants may be general, but it may not invent law."""
    guidance("You need Aadhaar and PAN. Section 12(4) of the Banking Act guarantees "
             "a loan within thirty days.")

    out = agent.answer_question("what documents do I need for a home loan", "en-IN")

    assert "Aadhaar" in out
    assert "Section 12(4)" not in out, "an uncited numbered rule must not survive"


def test_guidance_deletes_a_borrowed_citation_id(guidance):
    """Tier B ids are never in the allowed set, so a model that cites a web page it
    read gets the marker deleted rather than the claim dressed up as verified."""
    guidance("Banks usually ask for proof of income [web_rbi_org_in_some_page].")

    out = agent.answer_question("what documents do I need", "en-IN")

    assert "web_rbi_org_in_some_page" not in out


def test_guidance_falls_back_rather_than_answering_from_nothing(monkeypatch):
    agent.localise.cache_clear()
    monkeypatch.setattr(agent.sources, "context_for", lambda *a, **k: [])
    monkeypatch.setattr(agent.sarvam, "chat", lambda *a, **k: "Section 9 says you get a loan.")

    out = agent.answer_question("do I get a loan", "en-IN")

    assert out == agent.GUIDANCE_FALLBACK, "an answer stripped to nothing is not an answer"


# ------------------------------------------------------------------ localise


@pytest.fixture
def model(monkeypatch):
    """Count the calls as well as stubbing them — the cache is what keeps this off
    the wire on every message, so it has to be asserted, not assumed."""
    calls = []

    def fake_translate(text, target="en-IN", source="auto"):
        calls.append((text, target))
        return "अनुवादित"

    agent.localise.cache_clear()
    monkeypatch.setattr(agent.sarvam, "translate", fake_translate)
    return calls


def test_localise_puts_a_message_into_the_users_language(model):
    assert agent.localise("Filed. I will watch the clock for you.", "mr-IN") == "अनुवादित"
    assert model[0][1] == "mr-IN"


@pytest.mark.parametrize("language", [None, "", "en-IN"])
def test_english_speakers_cost_nothing(model, language):
    text = "Filed. I will watch the clock for you."
    assert agent.localise(text, language) == text
    assert model == [], "no call belongs on the English path"


def test_repeated_chrome_is_translated_once(model):
    for _ in range(3):
        agent.localise("Reading your document…", "mr-IN")
    assert len(model) == 1, "the fixed strings must not cost a call per message"


def test_a_list_keeps_its_lines(model):
    """Mayura returns one flowing paragraph, so the deadline list has to be fed to
    it a line at a time or it stops being a list."""
    out = agent.localise("Filed.\n• Reply due — 2026-11-07\n• Escalate — 2027-02-05", "mr-IN")
    assert out.count("\n") == 2
    assert len(model) == 3, "each line is translated, and cached, on its own"


def test_a_translation_failure_falls_back_to_english(monkeypatch):
    agent.localise.cache_clear()

    def boom(*args, **kwargs):
        raise RuntimeError("sarvam down")

    monkeypatch.setattr(agent.sarvam, "translate", boom)
    text = "Not sending it. Tell me what to change, or send *reset* to start over."
    assert agent.localise(text, "mr-IN") == text


def test_a_button_title_is_cut_to_fit(monkeypatch):
    """WhatsApp rejects nothing here — it truncates, so an over-long title becomes a
    half word."""
    agent.localise.cache_clear()
    monkeypatch.setattr(agent.sarvam, "translate", lambda *a, **k: "हो" * 40)
    assert len(agent.localise("Yes, file it", "mr-IN", 20)) == 20


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
