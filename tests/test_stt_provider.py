"""
Tests for stt_provider.py — standalone, no app.py/Whisper-model-loading
dependency (WhisperSTTProvider takes an injected model, so a stub suffices).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stt_provider as sp


class _StubWhisperModel:
    def __init__(self, text="hello world", segments=None, raise_err=None):
        self._text = text
        self._segments = segments if segments is not None else [{"avg_logprob": -0.2, "words": []}]
        self._raise_err = raise_err

    def transcribe(self, path, **kwargs):
        if self._raise_err:
            raise self._raise_err
        return {"text": self._text, "segments": self._segments}


# ---- Registry -----------------------------------------------------------------

def test_registry_has_both_providers():
    reg = sp.STTProviderRegistry({
        "whisper": sp.WhisperSTTProvider(_StubWhisperModel()),
        "saaras": sp.SaarasSTTProvider(api_key=""),
    })
    assert reg.get("whisper").name == "whisper"
    assert reg.get("saaras").name == "saaras"


def test_registry_unknown_provider_raises():
    reg = sp.STTProviderRegistry({"whisper": sp.WhisperSTTProvider(_StubWhisperModel())})
    try:
        reg.get("nonexistent")
        assert False, "expected STTProviderError"
    except sp.STTProviderError as e:
        assert "nonexistent" in str(e)


def test_registry_status_reports_availability():
    reg = sp.STTProviderRegistry({
        "whisper": sp.WhisperSTTProvider(_StubWhisperModel()),
        "saaras": sp.SaarasSTTProvider(api_key=""),
    })
    status = reg.status()
    assert status["whisper"]["available"] is True
    assert status["saaras"]["available"] is False


# ---- WhisperSTTProvider — existing behavior unchanged --------------------------

def test_whisper_stt_always_available():
    assert sp.WhisperSTTProvider(_StubWhisperModel()).is_available() is True


def test_whisper_stt_returns_transcript_and_segments():
    model = _StubWhisperModel(text="testing one two", segments=[{"avg_logprob": -0.1}])
    result = sp.WhisperSTTProvider(model).transcribe("fake.wav")
    assert result.transcript == "testing one two"
    assert result.segments == [{"avg_logprob": -0.1}]
    assert result.provider == "whisper"
    assert result.available is True


def test_whisper_stt_empty_transcript_reports_unavailable_not_raise():
    model = _StubWhisperModel(text="   ", segments=[])
    result = sp.WhisperSTTProvider(model).transcribe("fake.wav")
    assert result.available is False
    assert "speak clearly" in result.detail


# ---- SaarasSTTProvider ----------------------------------------------------------

def test_saaras_stt_not_configured_without_api_key():
    provider = sp.SaarasSTTProvider(api_key="")
    assert provider.is_available() is False
    result = provider.transcribe("fake.wav")
    assert result.available is False
    assert "not configured" in result.detail


def test_saaras_stt_configured_flag_reflects_api_key():
    assert sp.SaarasSTTProvider(api_key="fake-key").is_available() is True


def test_saaras_stt_missing_wav_path_reports_unavailable():
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    result = provider.transcribe(None)
    assert result.available is False
    assert result.provider == "saaras"


def test_saaras_stt_rejects_audio_over_30s(tmp_path, monkeypatch):
    """The REST endpoint's documented 30s cap — must refuse cleanly rather
    than send an oversized request."""
    wav = tmp_path / "long.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    monkeypatch.setattr(provider, "_wav_duration_seconds", staticmethod(lambda p: 45.0))
    result = provider.transcribe(wav)
    assert result.available is False
    assert "30" in result.detail


def test_saaras_stt_request_actually_reaches_the_http_call(monkeypatch, tmp_path):
    calls = []

    class _FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": "hello from saaras", "language_code": "en-IN"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        calls.append({"url": url, "headers": headers, "data": data})
        return _FakeResponse()

    monkeypatch.setattr(sp.httpx, "post", fake_post)

    wav = tmp_path / "short.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")

    provider = sp.SaarasSTTProvider(api_key="fake-key")
    monkeypatch.setattr(provider, "_wav_duration_seconds", staticmethod(lambda p: 5.0))
    result = provider.transcribe(wav)

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/speech-to-text")
    assert calls[0]["headers"]["api-subscription-key"] == "fake-key"
    assert calls[0]["data"]["model"] == "saaras:v3"
    assert result.available is True
    assert result.transcript == "hello from saaras"
    # No per-word confidence data in Saaras's documented response shape —
    # segments must stay empty rather than fabricate word-level probabilities.
    assert result.segments == []
    assert result.provider == "saaras"


def test_saaras_stt_empty_transcript_response_reports_unavailable(monkeypatch, tmp_path):
    class _FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": ""}

    monkeypatch.setattr(sp.httpx, "post", lambda *a, **kw: _FakeResponse())
    wav = tmp_path / "short.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    monkeypatch.setattr(provider, "_wav_duration_seconds", staticmethod(lambda p: 5.0))
    result = provider.transcribe(wav)
    assert result.available is False


def test_saaras_stt_http_error_reports_unavailable_not_raise(monkeypatch, tmp_path):
    def fake_post(*a, **kw):
        raise ConnectionError("network down")

    monkeypatch.setattr(sp.httpx, "post", fake_post)
    wav = tmp_path / "short.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    monkeypatch.setattr(provider, "_wav_duration_seconds", staticmethod(lambda p: 5.0))
    result = provider.transcribe(wav)
    assert result.available is False
    assert "network down" in result.detail


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
