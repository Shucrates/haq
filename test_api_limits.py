"""Limits on the three routes that spend money.

/api/intake, /api/document and /api/speak each hand their input to a paid API.
Before this, every one of them read the whole upload into memory first and had no
ceiling on how often it could be called, so an oversized or repeated request was
somebody else's budget and our RSS.
"""

import pytest
from fastapi.testclient import TestClient

import drafting
import main
import store

TOKEN = "limits-session-token"


@pytest.fixture(autouse=True)
def temp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(drafting, "PDF_DIR", tmp_path / "drafts")
    store.init()
    main._hits.clear()


@pytest.fixture
def client():
    c = TestClient(main.app)
    c.cookies.set(main.COOKIE_NAME, TOKEN)
    return c


@pytest.fixture
def case_id():
    cid = store.create_case(channel="web", owner_token=TOKEN)
    store.update_case(cid, language="en-IN")
    return cid


@pytest.fixture
def no_network(monkeypatch):
    """Any limit that lets a request through to Sarvam has failed."""
    def boom(*a, **k):
        raise AssertionError("request reached the paid API despite the limit")
    monkeypatch.setattr(main.sarvam, "stt", boom)
    monkeypatch.setattr(main.sarvam, "tts", boom)
    monkeypatch.setattr(main.documents, "extract", boom)


# ------------------------------------------------------------------ size caps


def test_oversized_audio_is_refused_before_transcription(client, case_id, no_network):
    huge = b"\x00" * (main.MAX_AUDIO_BYTES + 1024)
    response = client.post(
        "/api/intake",
        files={"audio": ("big.webm", huge, "audio/webm")},
        data={"case_id": case_id, "lang": "en-IN"},
    )
    assert response.status_code == 413


def test_oversized_document_is_refused_before_doc_ai(client, case_id, no_network):
    huge = b"%PDF-1.4" + b"\x00" * (main.MAX_DOC_BYTES + 1024)
    response = client.post(
        "/api/document",
        files={"document": ("big.pdf", huge, "application/pdf")},
        data={"case_id": case_id},
    )
    assert response.status_code == 413


def test_oversized_speak_text_is_refused(client, case_id, no_network):
    response = client.post(
        "/api/speak",
        json={"case_id": case_id, "text": "a" * (main.MAX_SPEAK_CHARS + 1)},
    )
    assert response.status_code == 413


# ------------------------------------------------------------ type allowlist


@pytest.mark.parametrize("mime", ["application/zip", "text/html", "application/x-msdownload", ""])
def test_document_rejects_types_doc_ai_cannot_read(client, case_id, no_network, mime):
    response = client.post(
        "/api/document",
        files={"document": ("payload.bin", b"nope", mime)},
        data={"case_id": case_id},
    )
    assert response.status_code == 415


def test_intake_rejects_non_audio(client, case_id, no_network):
    response = client.post(
        "/api/intake",
        files={"audio": ("not-audio.pdf", b"%PDF", "application/pdf")},
        data={"case_id": case_id, "lang": "en-IN"},
    )
    assert response.status_code == 415


def test_a_normal_document_still_gets_through(client, case_id, monkeypatch):
    """The cap must not become a wall the demo walks into."""
    monkeypatch.setattr(main.documents, "extract",
                        lambda *a, **k: {"result": {}, "annotations": {}, "status": "completed"})
    response = client.post(
        "/api/document",
        files={"document": ("letter.pdf", b"%PDF-1.4 small", "application/pdf")},
        data={"case_id": case_id},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------- rate limit


def test_repeated_calls_are_throttled(client, case_id, monkeypatch):
    """Counting the calls that land, not asserting none do: the first RATE_LIMIT
    are legitimate. What matters is that the throttle caps our Sarvam spend."""
    calls = []
    monkeypatch.setattr(main.sarvam, "tts", lambda *a, **k: calls.append(1) or "")

    codes = [
        client.post("/api/speak", json={"case_id": case_id, "text": "hello"}).status_code
        for _ in range(main.RATE_LIMIT + 5)
    ]

    assert 429 in codes, "an unthrottled paid endpoint is an open tab on our Sarvam quota"
    assert codes.index(429) == main.RATE_LIMIT, "throttle must not fire early"
    assert len(calls) == main.RATE_LIMIT, "throttled requests must not reach the paid API"


def test_the_free_routes_are_not_throttled(client, case_id, no_network):
    """Rate limiting the whole app would break the interrogation, which is chatty
    and costs nothing."""
    codes = [
        client.post("/api/resolve", json={"case_id": case_id}).status_code
        for _ in range(main.RATE_LIMIT + 5)
    ]
    assert 429 not in codes


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
