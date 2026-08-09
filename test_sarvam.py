"""Sarvam client tests.

Only the one property that was silently wrong in production: sarvam-105b is a
reasoning model, so max_tokens has to pay for the thinking as well as the answer.
"""

import logging

import pytest

import sarvam


@pytest.fixture
def posted(monkeypatch):
    """Capture the payload and hand back a canned completion."""
    calls = []
    reply = {"choices": [{"finish_reason": "stop",
                          "message": {"content": "तुम्ही तक्रार केली आहे का?"}}]}

    def fake_post(path, *, bearer=False, **kwargs):
        calls.append(kwargs.get("json"))
        return reply

    monkeypatch.setattr(sarvam, "_post", fake_post)
    monkeypatch.setenv("HAQ_MODE", "live")
    return calls, reply


def test_max_tokens_pays_for_the_reasoning_trace(posted):
    """The bug: an 80-token budget was spent entirely on reasoning, the API returned
    finish_reason='length' with content=null, and every caller fell back to English."""
    calls, _ = posted
    sarvam.chat([{"role": "user", "content": "hi"}], max_tokens=80)
    assert calls[0]["max_tokens"] == 80 + sarvam.REASONING_BUDGET


def test_reasoning_budget_covers_an_observed_trace():
    """Measured against the live API: 'low' effort spent 1015-1690 tokens thinking
    about a one-line question. A budget near that is the same bug again."""
    assert sarvam.REASONING_BUDGET >= 2000


def test_the_budget_never_pushes_a_caller_over_the_tier_cap(posted):
    """The starter tier 400s above 4096, so the headroom must clamp — the longest
    caller in the repo is drafting's 900-token body."""
    calls, _ = posted
    sarvam.chat([{"role": "user", "content": "hi"}], max_tokens=900)
    assert calls[0]["max_tokens"] <= sarvam.MAX_COMPLETION_TOKENS


def test_an_empty_answer_is_logged_not_swallowed(posted, caplog):
    calls, reply = posted
    reply["choices"][0].update(finish_reason="length",
                               message={"content": None})

    with caplog.at_level(logging.WARNING, logger="haq.sarvam"):
        assert sarvam.chat([{"role": "user", "content": "hi"}]) == ""

    assert "sarvam_chat_empty" in caplog.text


def test_opus_asks_for_a_sample_rate_opus_accepts(monkeypatch):
    """The bug: WhatsApp voice notes never sent. Bulbul synthesises at 22050 Hz by
    default and opus 400s on it — "OPUS codec requires one of these sample rates:
    8000, 12000, 16000, 24000, 48000 Hz." Every read-back died in a logged warning."""
    calls = []

    def fake_post(path, *, bearer=False, **kwargs):
        calls.append(kwargs.get("json"))
        return {"audios": ["Zm9v"]}

    monkeypatch.setattr(sarvam, "_post", fake_post)
    monkeypatch.setenv("HAQ_MODE", "live")

    sarvam.tts("नमस्कार", language_code="mr-IN", output_audio_codec="opus")
    assert calls[0]["speech_sample_rate"] in (8000, 12000, 16000, 24000, 48000)

    # wav is left alone: it works at the default, and adding a key would miss every
    # response already recorded in data/cache.
    sarvam.tts("नमस्कार", language_code="mr-IN")
    assert "speech_sample_rate" not in calls[1]
