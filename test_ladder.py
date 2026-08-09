"""The tests that win Q&A (PRD 3.3).

Run `pytest -v` on the projector when a judge asks how you know the AI isn't
hallucinating the law. Twelve green dots beats any explanation.
"""

from datetime import date, timedelta

import pytest

from ladder_engine import Facts, load_ladders, resolve

L = load_ladders()
TODAY = date(2026, 8, 8)


def banking(**kw) -> Facts:
    """A banking grievance that is maintainable unless a test says otherwise."""
    base = dict(
        grievance_class="banking/unauthorised_charge",
        tier1_written_complaint=True,
        tier1_filed_on=date(2026, 6, 1),
        last_communication_on=date(2026, 6, 15),
        pending_in_court=False,
        amount_inr=1200.0,
        account_type="bsbda",
    )
    base.update(kw)
    return Facts(**base)


def test_tier1_not_exhausted_blocks_ombudsman():
    v = resolve(banking(tier1_written_complaint=False), L, TODAY)
    assert v.maintainable is False
    assert "tier1_not_exhausted" in v.blocked_by
    assert v.redirect_to == 1


def test_ninety_day_window_computed_from_later_of_two_dates():
    # Filed 1 June, bank last spoke 15 June -> the clock runs from 15 June.
    v = resolve(banking(), L, TODAY)
    due = {d.id: d.due_on for d in v.deadlines}
    assert due["rbios_90_day_window"] == date(2026, 9, 13)


def test_time_barred_after_ninety_one_days():
    # Both dates must be old: the clock runs from the LATER of the two.
    v = resolve(
        banking(tier1_filed_on=date(2026, 4, 1), last_communication_on=date(2026, 5, 1)),
        L,
        TODAY,
    )
    assert v.maintainable is False
    assert "time_barred" in v.blocked_by


def test_exactly_ninety_days_is_still_maintainable():
    # Boundary: 90 days is inside the window, 91 is outside.
    anchor = TODAY - timedelta(days=90)
    v = resolve(banking(tier1_filed_on=anchor, last_communication_on=anchor), L, TODAY)
    assert "time_barred" not in v.blocked_by


def test_pending_in_court_hard_stops():
    v = resolve(banking(pending_in_court=True), L, TODAY)
    assert v.maintainable is False
    assert "pending_in_court" in v.blocked_by
    assert v.redirect_to is None


def test_commercial_decision_excluded():
    v = resolve(banking(commercial_decision=True), L, TODAY)
    assert v.maintainable is False
    assert "commercial_decision" in v.blocked_by


def test_waiting_period_not_elapsed_is_premature():
    v = resolve(
        banking(tier1_filed_on=date(2026, 8, 1), last_communication_on=date(2026, 8, 1)),
        L,
        TODAY,
    )
    assert v.maintainable is False
    assert "waiting_period_not_elapsed" in v.blocked_by
    assert v.redirect_to == 1


def test_deadline_is_deterministic_across_runs():
    a = resolve(banking(), L, TODAY)
    b = resolve(banking(), L, TODAY)
    assert [(d.id, d.due_on) for d in a.deadlines] == [(d.id, d.due_on) for d in b.deadlines]
    assert a.haq_score == b.haq_score


def test_blocked_by_always_has_a_plain_language_message():
    """The refusal must be explainable to a human, in every blocking case."""
    for facts in [
        banking(tier1_written_complaint=False),
        banking(pending_in_court=True),
        banking(commercial_decision=True),
        banking(tier1_filed_on=date(2026, 1, 1), last_communication_on=date(2026, 1, 1)),
    ]:
        v = resolve(facts, L, TODAY)
        assert v.blocked_by, "expected a block"
        assert len(v.blocked_messages) == len(v.blocked_by)
        assert all(m.strip() for m in v.blocked_messages)


def test_maintainable_case_carries_source_and_verification_date():
    v = resolve(banking(), L, TODAY)
    assert v.maintainable is True
    assert v.blocked_by == []
    assert v.source_url.startswith("http")
    assert v.verified_on


def test_second_ladder_proves_engine_is_general():
    """Different law, different tiers, different clock — same resolve()."""
    v = resolve(
        Facts(
            grievance_class="rti/no_response",
            tier1_written_complaint=True,
            # PIO's 30 days have lapsed, and the 30-day appeal window is still open.
            tier1_filed_on=date(2026, 7, 1),
            last_communication_on=date(2026, 7, 20),
        ),
        L,
        TODAY,
    )
    assert v.ladder_id == "rti_2005"
    assert v.maintainable is True
    due = {d.id: d.due_on for d in v.deadlines}
    assert due["pio_reply_due"] == date(2026, 7, 31)


def test_unknown_grievance_refuses_rather_than_guessing():
    v = resolve(Facts(grievance_class="tenancy/deposit"), L, TODAY)
    assert v.maintainable is False
    assert v.blocked_by == ["no_ladder_for_grievance"]
    assert v.citations == []


def test_engine_makes_no_network_calls():
    """Guard the 'pure function' claim: fail loudly if anyone adds a socket."""
    import socket

    original = socket.socket

    def forbidden(*args, **kwargs):
        raise AssertionError("ladder_engine must not open sockets")

    socket.socket = forbidden
    try:
        resolve(banking(), L, TODAY)
    finally:
        socket.socket = original


def test_missing_required_fact_is_pending_not_maintainable():
    v = resolve(banking(last_communication_on=None), L, TODAY)
    assert v.maintainable is False
    assert "last_communication_on" in v.facts_pending


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
