"""
Route-level tests for stt_provider selection — mirrors
tests/test_app_pronunciation_routes.py. Stubs whisper the same way.
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


def test_validate_stt_provider_accepts_known_names():
    for name in ("whisper", "saaras"):
        assert app.validate_stt_provider(name) == name


def test_validate_stt_provider_rejects_unknown_name():
    from fastapi import HTTPException
    try:
        app.validate_stt_provider("made_up")
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 422


def test_resolve_stt_default_whisper(tmp_path):
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    requested, used = app.resolve_stt("whisper", wav)
    assert requested.available is True
    assert used.provider == "whisper"
    assert used.transcript == "hello world"


def test_resolve_stt_saaras_not_configured_falls_back_to_whisper(monkeypatch, tmp_path):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    import stt_provider as sp
    monkeypatch.setitem(app.stt_registry._providers, "saaras", sp.SaarasSTTProvider(api_key=""))
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    requested, used = app.resolve_stt("saaras", wav)
    assert requested.available is False
    assert requested.provider == "saaras"
    # Fallback actually produced the transcript, and honestly reports whisper
    # as the provider that ran.
    assert used.provider == "whisper"
    assert used.transcript == "hello world"


def test_stt_providers_status_endpoint_shape():
    import asyncio
    result = asyncio.run(app.stt_providers())
    assert result["default"] == "whisper"
    assert set(result["providers"]) == {"whisper", "saaras"}
    assert result["providers"]["whisper"]["available"] is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
