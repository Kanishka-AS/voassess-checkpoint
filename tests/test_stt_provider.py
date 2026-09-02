"""
Tests for stt_provider.py — standalone, no app.py/Whisper-model-loading
dependency (WhisperSTTProvider takes an injected model, so a stub suffices).
"""
import os
import sys
from pathlib import Path

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


def _write_real_wav(path, seconds: float, rate: int = 16000, channels: int = 1, sampwidth: int = 2):
    """Real (silent) WAV file at a known duration, for chunking tests that
    need split_wav_into_chunks() to actually read valid WAV headers/frames
    — a fake `RIFF....` byte string (used by the other tests here, which
    only exercise the HTTP call) won't parse as a WAV."""
    import wave
    n_frames = round(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(b"\x00" * n_frames * sampwidth * channels)
    return n_frames


def test_saaras_stt_over_30s_is_chunked_not_rejected(tmp_path, monkeypatch):
    """Audio over the REST endpoint's 30s cap is split into sequential
    chunks and transcribed successfully, not refused outright."""
    wav = tmp_path / "long.wav"
    _write_real_wav(wav, seconds=70.0)  # -> 3 chunks at 25s each (25/25/20)

    calls = []

    class _FakeResponse:
        status_code = 200
        def __init__(self, index):
            self._index = index
        def raise_for_status(self): pass
        def json(self):
            return {"transcript": f"chunk {self._index} um yeah"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        index = len(calls)
        calls.append(1)
        return _FakeResponse(index)

    monkeypatch.setattr(sp.httpx, "post", fake_post)
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    result = provider.transcribe(wav)

    assert len(calls) == 3
    assert result.available is True
    assert result.provider == "saaras"
    # Chunks joined in order, raw text untouched (no dedup/cleanup):
    assert result.transcript == "chunk 0 um yeah chunk 1 um yeah chunk 2 um yeah"
    assert result.segments == []


def test_saaras_stt_chunk_failure_reports_clear_error_not_partial(tmp_path, monkeypatch):
    """If one chunk's request fails, the whole call reports available=False
    with a detail naming the failing chunk — earlier chunks' text is not
    silently returned as if it were the complete transcript."""
    wav = tmp_path / "long.wav"
    _write_real_wav(wav, seconds=70.0)  # 3 chunks

    call_count = {"n": 0}

    class _FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": "chunk ok"}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ConnectionError("network down mid-recording")
        return _FakeResponse()

    monkeypatch.setattr(sp.httpx, "post", fake_post)
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    result = provider.transcribe(wav)

    assert result.available is False
    assert result.transcript == ""
    assert "chunk 2/3" in result.detail
    assert "network down mid-recording" in result.detail


def test_saaras_stt_chunk_temp_files_cleaned_up(tmp_path, monkeypatch):
    """Temp chunk files must not linger after a successful (or failed)
    chunked transcription."""
    import os as _os
    wav = tmp_path / "long.wav"
    _write_real_wav(wav, seconds=70.0)

    seen_dirs = []

    class _FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": "ok"}

    real_open = open
    def tracking_post(url, headers=None, data=None, files=None, timeout=None):
        # files=("name", filehandle, mimetype) — record the chunk dir while
        # the file still exists, so we can confirm it's gone afterward.
        seen_dirs.append(Path(files["file"][1].name).parent)
        return _FakeResponse()

    monkeypatch.setattr(sp.httpx, "post", tracking_post)
    provider = sp.SaarasSTTProvider(api_key="fake-key")
    result = provider.transcribe(wav)

    assert result.available is True
    assert seen_dirs, "expected at least one chunk request"
    for d in set(seen_dirs):
        assert not d.exists(), f"chunk temp dir {d} was not cleaned up"


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
