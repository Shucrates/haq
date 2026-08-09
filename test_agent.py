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


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
