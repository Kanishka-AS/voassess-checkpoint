"""
Route-level tests: proving pronunciation_provider selection actually flows
from the API request through to score_free_speech() and back into the
response, and that invalid providers are rejected before any work happens.

Stubs whisper the same way tests/test_app_scoring_integration.py does.
Auth (require_user) is bypassed by monkeypatching it, same rationale as
that file — these tests are about provider wiring, not PocketBase.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "whisper" not in sys.modules:
    _stub = types.ModuleType("whisper")

    class _StubModel:
        def transcribe(self, *a, **kw):
            return {"text": "hello world", "segments": [{"avg_logprob": -0.2, "words": []}]}

    _stub.load_model = lambda name: _StubModel()
    sys.modules["whisper"] = _stub

import app  # noqa: E402


def _no_languagetool(transcript):
    return None, None, {"check": "stub", "analyze": "stub"}


def test_score_free_speech_default_provider(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("hello world", [{"avg_logprob": -0.2}], duration=5.0)
    assert r["pronunciation"]["provider"] == "whisper_confidence"
    assert r["pronunciation"]["requested_provider"] == "whisper_confidence"
    assert r["pronunciation"]["available"] is True
    assert r["pronunciation"]["detail"] is None


def test_score_free_speech_explicit_gop_falls_back_but_reports_honestly(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("hello world", [{"avg_logprob": -0.2}], duration=5.0,
                               pronunciation_provider="gop")
    # The score/issues came from whisper_confidence (fallback)...
    assert r["pronunciation"]["provider"] == "whisper_confidence"
    # ...but the response never pretends gop actually ran.
    assert r["pronunciation"]["requested_provider"] == "gop"
    assert r["pronunciation"]["available"] is False
    assert r["pronunciation"]["detail"] == "Custom GOP pronunciation assessment is not currently available."


def test_score_free_speech_explicit_saaras_not_configured_falls_back(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    # Rebuild the module-level registry's saaras entry with no key, in case
    # the test environment happens to have one set.
    import pronunciation_provider as pp
    monkeypatch.setattr(pp.pronunciation_registry, "_providers", {
        **pp.pronunciation_registry._providers,
        "saaras": pp.SaarasPronunciationProvider(api_key=""),
    })
    r = app.score_free_speech("hello world", [{"avg_logprob": -0.2}], duration=5.0,
                               pronunciation_provider="saaras")
    assert r["pronunciation"]["provider"] == "whisper_confidence"
    assert r["pronunciation"]["requested_provider"] == "saaras"
    assert r["pronunciation"]["available"] is False


def test_validate_pronunciation_provider_accepts_known_names():
    for name in ("whisper_confidence", "saaras", "local_llm", "gop"):
        assert app.validate_pronunciation_provider(name) == name


def test_validate_pronunciation_provider_rejects_unknown_name():
    from fastapi import HTTPException
    try:
        app.validate_pronunciation_provider("made_up")
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 422


def test_pronunciation_providers_status_endpoint_shape():
    import asyncio
    result = asyncio.run(app.pronunciation_providers())
    assert result["default"] == "whisper_confidence"
    assert set(result["providers"]) == {"whisper_confidence", "saaras", "local_llm", "gop"}
    assert result["providers"]["whisper_confidence"]["available"] is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
