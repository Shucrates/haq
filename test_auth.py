"""Case ownership.

Before this existed, a case_id was a bearer credential. GET /api/case/{id} returned
the transcript, the fact sheet, every message, and the verdict to anyone who could
name the id — and the mutating routes took it the same way, so a leaked id also
meant a tamperable case. Twelve hex characters in a screenshot, a log line or a
shared screen was the whole authentication story.

Both clients here are legitimate browsers. The only difference is which session
they hold.
"""

import pytest
from fastapi.testclient import TestClient

import drafting
import main
import store

OWNER = "owner-session-token"
STRANGER = "stranger-session-token"


@pytest.fixture(autouse=True)
def temp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(drafting, "PDF_DIR", tmp_path / "drafts")
    store.init()


def _client(token: str | None) -> TestClient:
    c = TestClient(main.app)
    if token:
        c.cookies.set(main.COOKIE_NAME, token)
    return c


@pytest.fixture
def owner():
    return _client(OWNER)


@pytest.fixture
def stranger():
    return _client(STRANGER)


@pytest.fixture
def case_id():
    cid = store.create_case(channel="web", owner_token=OWNER)
    store.update_case(cid, language="en-IN", transcript="the bank took my money",
                      grievance_class="banking/unauthorised_charge")
    return cid


# --------------------------------------------------------------- the boundary

# Every route that names a case. If a route is added and not listed here, that is
# the gap this file exists to catch.
CASE_ROUTES = [
    ("get", "/api/case/{cid}", None),
    ("get", "/api/draft/{cid}.pdf", None),
    ("post", "/api/onboard", {"case_id": "{cid}", "language": "hi-IN"}),
    ("post", "/api/turn", {"case_id": "{cid}"}),
    ("post", "/api/resolve", {"case_id": "{cid}"}),
    ("post", "/api/draft", {"case_id": "{cid}"}),
    ("post", "/api/speak", {"case_id": "{cid}", "text": "hello"}),
    ("post", "/api/approve", {"case_id": "{cid}"}),
    ("post", "/api/advance", {"case_id": "{cid}", "days": 31}),
    ("post", "/api/intake_text", {"case_id": "{cid}", "text": "hello"}),
]


def _call(client: TestClient, method: str, path: str, body: dict | None, case_id: str):
    url = path.replace("{cid}", case_id)
    if body is None:
        return client.get(url)
    payload = {
        k: v.replace("{cid}", case_id) if isinstance(v, str) else v for k, v in body.items()
    }
    return client.post(url, json=payload)


@pytest.mark.parametrize("method,path,body", CASE_ROUTES)
def test_another_session_cannot_touch_the_case(stranger, case_id, method, path, body):
    response = _call(stranger, method, path, body, case_id)
    assert response.status_code == 404, f"{method.upper()} {path} leaked to another session"


@pytest.mark.parametrize("method,path,body", CASE_ROUTES)
def test_no_session_at_all_is_refused(case_id, method, path, body):
    """A bare curl with the right id and no cookie. This was the whole attack."""
    response = _call(_client(None), method, path, body, case_id)
    assert response.status_code == 404


def test_forbidden_is_indistinguishable_from_missing(stranger, case_id):
    """404 not 403: a 403 confirms the id names a real case, which turns a blind
    guess into a probe."""
    real = stranger.get(f"/api/case/{case_id}")
    invented = stranger.get("/api/case/" + "0" * 12)
    assert real.status_code == invented.status_code == 404
    assert real.json() == invented.json() or real.json()["detail"] != invented.json()["detail"]


def test_the_owner_still_gets_in(owner, case_id):
    response = owner.get(f"/api/case/{case_id}")
    assert response.status_code == 200
    assert response.json()["case"]["transcript"] == "the bank took my money"


def test_owner_token_is_never_returned_to_the_client(owner, case_id):
    """It is a credential. Echoing it back would hand a stranger the key along
    with the lock."""
    body = owner.get(f"/api/case/{case_id}").json()
    assert "owner_token" not in body["case"]
    assert OWNER not in owner.get(f"/api/case/{case_id}").text


# ------------------------------------------------------------- issuing sessions


def test_start_issues_a_session_and_the_case_it_owns():
    fresh = TestClient(main.app)
    body = fresh.post("/api/start").json()

    assert fresh.cookies.get(main.COOKIE_NAME), "no session cookie was issued"
    assert store.get_case(body["case_id"])["owner_token"] == fresh.cookies.get(main.COOKIE_NAME)
    assert fresh.get(f"/api/case/{body['case_id']}").status_code == 200


def test_the_session_cookie_is_not_readable_by_script():
    raw = TestClient(main.app).post("/api/start").headers["set-cookie"].lower()
    assert "httponly" in raw, "a session token readable by document.cookie is an XSS payload"
    assert "samesite=lax" in raw


def test_intake_text_without_a_language_still_binds_the_new_case():
    """The 409 picker path returns a Response directly, which bypasses the injected
    one — the cookie has to ride along or the case it just minted is orphaned."""
    fresh = TestClient(main.app)
    response = fresh.post("/api/intake_text", json={"text": "hello"})

    assert response.status_code == 409
    case_id = response.json()["case_id"]
    assert fresh.cookies.get(main.COOKIE_NAME)
    assert store.get_case(case_id)["owner_token"] == fresh.cookies.get(main.COOKIE_NAME)


# ------------------------------------------------------------ id shape + channels


@pytest.mark.parametrize("bad", ["../../etc/passwd", "..", "NOTHEX123456", "0" * 64, ""])
def test_malformed_case_ids_never_reach_the_filesystem(owner, bad):
    assert owner.get(f"/api/draft/{bad}.pdf").status_code in (404, 405)


def test_whatsapp_cases_are_unreachable_from_the_web(owner):
    """A WhatsApp case is owned by the phone number. No browser cookie can equal
    "wa:<phone>", so the web surface cannot read one even with the right id."""
    wa_case = store.case_for_phone("919999900000")
    assert store.get_case(wa_case)["owner_token"] == "wa:919999900000"
    assert owner.get(f"/api/case/{wa_case}").status_code == 404


def test_a_case_with_no_owner_is_reachable_by_nobody(owner):
    """Fails closed. A row from before ownership existed becomes inert rather than
    public."""
    legacy = store.create_case(channel="web")
    assert store.get_case(legacy)["owner_token"] is None
    assert owner.get(f"/api/case/{legacy}").status_code == 404
    assert _client(None).get(f"/api/case/{legacy}").status_code == 404


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
