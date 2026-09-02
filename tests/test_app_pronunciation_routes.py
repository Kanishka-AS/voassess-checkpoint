"""
Route-level tests: proving pronunciation_provider selection actually flows
from the API request through to score_free_speech() and back into the
response, and that invalid providers are rejected before any work happens.

Stubs whisper the same way tests/test_app_scoring_integration.py does —
importantly, this stub's transcribe() raises if ever called, so these tests
also double as a guard that pronunciation scoring (default: allosaurus_g2p)
never triggers a Whisper transcription. Auth (require_user) is bypassed by
monkeypatching it, same rationale as that file — these tests are about
provider wiring, not PocketBase.
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
import pronunciation_provider as pp  # noqa: E402


def _forbid_whisper_transcribe(monkeypatch):
    """Scoped (per-test, auto-restored by monkeypatch) guard: makes the
    already-loaded Whisper model explode if .transcribe() is ever called,
    without touching the shared module-level `whisper` stub other test
    files in this same process also rely on for their own (legitimate)
    Whisper STT tests."""
    def _boom(*a, **kw):
        raise AssertionError(
            "Whisper.transcribe() must never be called by pronunciation "
            "scoring — Sarvam is the single source of truth for the "
            "transcript, and the default pronunciation provider "
            "(allosaurus_g2p) must not depend on Whisper."
        )
    monkeypatch.setattr(app._whisper, "transcribe", _boom)


def _no_languagetool(transcript):
    return None, None, {"check": "stub", "analyze": "stub"}


class _FakeAllosaurusG2P(pp.PronunciationProvider):
    """Fast, deterministic stand-in for AllosaurusG2PPronunciationProvider —
    these tests are about routing/fallback mechanics, not about actually
    running the Allosaurus model (that's covered separately, with real
    audio, in tests/test_pronunciation_provider.py and the end-to-end test
    below). Mirrors the real provider's honest contract: issues is always
    [], and it requires wav_path."""

    name = "allosaurus_g2p"

    def is_available(self) -> bool:
        return True

    def assess(self, transcript, segments, wav_path):
        if not wav_path:
            return pp.PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Allosaurus/G2P pronunciation assessment requires the "
                       "recorded audio, which wasn't provided to this call.",
            )
        return pp.PronunciationResult(
            score=55.0, issues=[], provider=self.name, available=True,
            methodology="fake allosaurus_g2p result for routing tests",
        )


def _install_fake_allosaurus_g2p(monkeypatch):
    monkeypatch.setattr(pp.pronunciation_registry, "_providers", {
        **pp.pronunciation_registry._providers,
        "allosaurus_g2p": _FakeAllosaurusG2P(),
    })


def test_score_free_speech_default_provider(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    _install_fake_allosaurus_g2p(monkeypatch)
    _forbid_whisper_transcribe(monkeypatch)
    r = app.score_free_speech("hello world", [], duration=5.0, wav_path="/tmp/fake.wav")
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["requested_provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["available"] is True
    assert r["pronunciation"]["detail"] is None
    assert r["pronunciation"]["score"] == 55.0


def test_score_free_speech_default_provider_without_audio_reports_unavailable(monkeypatch):
    """No wav_path -> the default provider honestly reports unavailable
    (score 0.0, available False) rather than silently substituting a
    Whisper-derived number — there is no Whisper fallback anymore."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    _install_fake_allosaurus_g2p(monkeypatch)
    r = app.score_free_speech("hello world", [], duration=5.0, wav_path=None)
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["requested_provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["available"] is False
    assert r["pronunciation"]["score"] == 0.0


def test_score_free_speech_explicit_gop_falls_back_but_reports_honestly(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    _install_fake_allosaurus_g2p(monkeypatch)
    r = app.score_free_speech("hello world", [], duration=5.0, wav_path="/tmp/fake.wav",
                               pronunciation_provider="gop")
    # The score/issues came from allosaurus_g2p (fallback) — not Whisper...
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    # ...but the response never pretends gop actually ran.
    assert r["pronunciation"]["requested_provider"] == "gop"
    assert r["pronunciation"]["available"] is False
    assert r["pronunciation"]["detail"] == "Custom GOP pronunciation assessment is not currently available."


def test_score_free_speech_explicit_saaras_not_configured_falls_back(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    _install_fake_allosaurus_g2p(monkeypatch)
    # Rebuild the module-level registry's saaras entry with no key, in case
    # the test environment happens to have one set.
    monkeypatch.setattr(pp.pronunciation_registry, "_providers", {
        **pp.pronunciation_registry._providers,
        "saaras": pp.SaarasPronunciationProvider(api_key=""),
    })
    r = app.score_free_speech("hello world", [], duration=5.0, wav_path="/tmp/fake.wav",
                               pronunciation_provider="saaras")
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["requested_provider"] == "saaras"
    assert r["pronunciation"]["available"] is False


def test_score_free_speech_default_unavailable_does_not_double_call(monkeypatch):
    """When the requested provider IS the default and it fails, app.py must
    not re-resolve the exact same provider a second time (see the
    'fallback == pronun_result' short-circuit in score_free_speech)."""
    calls = []

    class _CountingFailingProvider(pp.PronunciationProvider):
        name = "allosaurus_g2p"

        def is_available(self):
            return True

        def assess(self, transcript, segments, wav_path):
            calls.append(1)
            return pp.PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="simulated failure",
            )

    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    monkeypatch.setattr(pp.pronunciation_registry, "_providers", {
        **pp.pronunciation_registry._providers,
        "allosaurus_g2p": _CountingFailingProvider(),
    })
    r = app.score_free_speech("hello world", [], duration=5.0, wav_path="/tmp/fake.wav")
    assert len(calls) == 1
    assert r["pronunciation"]["available"] is False
    assert r["pronunciation"]["detail"] == "simulated failure"


def test_validate_pronunciation_provider_accepts_known_names():
    for name in ("allosaurus_g2p", "whisper_confidence", "saaras", "local_llm", "gop"):
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
    assert result["default"] == "allosaurus_g2p"
    assert set(result["providers"]) == {"allosaurus_g2p", "whisper_confidence", "saaras", "local_llm", "gop"}
    assert result["providers"]["whisper_confidence"]["available"] is True


# ---- End-to-end: real Allosaurus + G2P, real synthesized audio --------------

def test_score_free_speech_default_provider_real_pipeline_no_whisper(monkeypatch):
    """Not a fake — runs the actual default provider (real G2P, real
    Allosaurus model) against real synthesized audio, through the real
    app.score_free_speech(), with Whisper.transcribe() wired to explode if
    it's ever called. Skips cleanly if the optional pronunciation
    dependencies or espeak-ng aren't installed in this environment."""
    import shutil
    import subprocess
    import tempfile
    import pytest

    provider = pp.AllosaurusG2PPronunciationProvider()
    if not provider.is_available():
        pytest.skip("allosaurus/panphon/eng-to-ipa not installed in this environment")
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed; cannot synthesize test audio")

    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    _forbid_whisper_transcribe(monkeypatch)

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "t.wav")
        subprocess.run(["espeak-ng", "-v", "en-us", "-w", wav, "I like apples"],
                        check=True, capture_output=True)
        r = app.score_free_speech("I like apples", [], duration=2.0, wav_path=wav)

    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["available"] is True
    assert r["pronunciation"]["issues"] == []
    assert 0.0 <= r["pronunciation"]["score"] <= 100.0
    assert "Whisper" not in r["pronunciation"]["methodology"] or \
           "No Whisper" in r["pronunciation"]["methodology"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
