"""The refusal must be enforced, not just reported.

/api/resolve is advisory — nothing ever stopped a client from skipping it and
POSTing straight to /api/draft. Until this file existed, a time-barred complaint
still got a formatted, citation-bearing PDF, which is the one outcome the whole
product claims to prevent.
"""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import drafting
import main
import store

TODAY = date.today()


@pytest.fixture(autouse=True)
def temp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(drafting, "PDF_DIR", tmp_path / "drafts")
    store.init()


@pytest.fixture
def client():
    return TestClient(main.app)


def _case(**facts) -> str:
    """A banking case, maintainable unless the caller says otherwise."""
    base = {
        "tier1_written_complaint": True,
        "tier1_filed_on": (TODAY - timedelta(days=45)).isoformat(),
        "last_communication_on": (TODAY - timedelta(days=45)).isoformat(),
        "pending_in_court": False,
        "commercial_decision": False,
        "amount_inr": 9840.0,
        "institution": "Bank of Maharashtra",
    }
    base.update(facts)
    case_id = store.create_case(channel="web")
    store.update_case(
        case_id,
        language="en-IN",
        transcript="the bank charged me a fee I never agreed to",
        grievance_class="banking/unauthorised_charge",
        facts=base,
    )
    return case_id


def _pdf_exists(case_id: str) -> bool:
    return (drafting.PDF_DIR / f"{case_id}.pdf").exists()


# ------------------------------------------------------------- the gate


def test_blocked_case_gets_no_pdf(client):
    """The headline. No written complaint -> the Ombudsman would throw it out, so
    HAQ must not hand over a document that says otherwise."""
    case_id = _case(tier1_written_complaint=False)

    response = client.post("/api/draft", json={"case_id": case_id})

    assert response.status_code == 409
    assert "tier1_not_exhausted" in response.json()["blocked_by"]
    assert not _pdf_exists(case_id), "a refused case must never produce a filing"
    assert store.get_case(case_id)["draft_text"] is None


def test_time_barred_case_gets_no_pdf(client):
    old = (TODAY - timedelta(days=200)).isoformat()
    case_id = _case(tier1_filed_on=old, last_communication_on=old)

    response = client.post("/api/draft", json={"case_id": case_id})

    assert response.status_code == 409
    assert "time_barred" in response.json()["blocked_by"]
    assert not _pdf_exists(case_id)


def test_refusal_is_explained_and_recorded(client):
    case_id = _case(pending_in_court=True)

    body = client.post("/api/draft", json={"case_id": case_id}).json()

    assert body["refusal_native"].strip(), "a refusal the user cannot read is not a refusal"
    assert body["maintainable"] is False
    assert any(e["kind"] == "draft_blocked" for e in store.events(case_id))


def test_pending_facts_block_the_draft_too(client):
    """Not-maintainable-yet is not the same as blocked, and neither may draft."""
    case_id = _case(last_communication_on=None)

    response = client.post("/api/draft", json={"case_id": case_id})

    assert response.status_code == 409
    assert "last_communication_on" in response.json()["facts_pending"]
    assert not _pdf_exists(case_id)


# ------------------------------------------------------------ the tier param


def test_invented_tier_is_rejected(client):
    """`tier` was free-form user input printed straight into the PDF's To: line."""
    case_id = _case()

    response = client.post("/api/draft", json={"case_id": case_id, "tier": 99})

    assert response.status_code == 400
    assert not _pdf_exists(case_id)


def test_maintainable_case_still_drafts(client):
    """The gate must not break the happy path."""
    case_id = _case()

    response = client.post("/api/draft", json={"case_id": case_id})

    assert response.status_code == 200
    assert response.json()["tier_name"] == "the RBI Ombudsman (RB-IOS)"
    assert _pdf_exists(case_id)
    assert store.get_case(case_id)["draft_text"]


def test_explicit_valid_tier_is_honoured(client):
    case_id = _case()

    body = client.post("/api/draft", json={"case_id": case_id, "tier": 1}).json()

    assert body["tier"] == 1
    assert body["tier_name"] == "the bank's own Internal Ombudsman / Nodal Officer"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
