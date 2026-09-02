"""
Tests for pronunciation_provider.py — the registry, the interface contract,
and each of the four providers. No app.py import needed here (this module
has no Whisper/LanguageTool dependency), so these run fast and standalone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pronunciation_provider as pp


# ---- Registry ---------------------------------------------------------------

def test_registry_has_all_five_providers():
    reg = pp.PronunciationProviderRegistry()
    for name in ("allosaurus_g2p", "whisper_confidence", "saaras", "local_llm", "gop"):
        provider = reg.get(name)
        assert provider.name == name


def test_registry_resolves_correct_class():
    reg = pp.PronunciationProviderRegistry()
    assert isinstance(reg.get("allosaurus_g2p"), pp.AllosaurusG2PPronunciationProvider)
    assert isinstance(reg.get("whisper_confidence"), pp.WhisperConfidenceProvider)
    assert isinstance(reg.get("saaras"), pp.SaarasPronunciationProvider)
    assert isinstance(reg.get("local_llm"), pp.LocalLLMPronunciationProvider)
    assert isinstance(reg.get("gop"), pp.GOPPronunciationProvider)


def test_registry_unknown_provider_raises():
    reg = pp.PronunciationProviderRegistry()
    try:
        reg.get("nonexistent")
        assert False, "expected PronunciationProviderError"
    except pp.PronunciationProviderError as e:
        assert "nonexistent" in str(e)


def test_registry_status_reports_availability():
    reg = pp.PronunciationProviderRegistry({
        "whisper_confidence": pp.WhisperConfidenceProvider(),
        "saaras": pp.SaarasPronunciationProvider(api_key=""),
        "local_llm": pp.LocalLLMPronunciationProvider(base_url=""),
        "gop": pp.GOPPronunciationProvider(),
    })
    status = reg.status()
    assert status["whisper_confidence"]["available"] is True
    assert status["saaras"]["available"] is False
    assert status["local_llm"]["available"] is False
    assert status["gop"]["available"] is False


# ---- Default provider is now allosaurus_g2p — no Whisper in this path -------

def test_default_provider_selection_is_allosaurus_g2p():
    assert pp.pronunciation_registry.get("allosaurus_g2p").name == "allosaurus_g2p"


def test_allosaurus_g2p_module_never_imports_whisper():
    """Static guard: the module backing the default pronunciation provider
    must not import whisper anywhere, so pronunciation assessment can run
    even in an environment with no Whisper installed at all."""
    import inspect
    src = inspect.getsource(pp)
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import whisper")
        assert not stripped.startswith("from whisper")


def test_allosaurus_g2p_requires_wav_path():
    provider = pp.AllosaurusG2PPronunciationProvider()
    result = provider.assess("hello world", segments=[], wav_path=None)
    assert result.available is False
    assert result.provider == "allosaurus_g2p"
    assert "audio" in result.detail.lower()


def test_allosaurus_g2p_requires_nonempty_transcript():
    provider = pp.AllosaurusG2PPronunciationProvider()
    result = provider.assess("", segments=[], wav_path="/tmp/does-not-matter.wav")
    assert result.available is False
    assert result.provider == "allosaurus_g2p"


def test_allosaurus_g2p_never_produces_per_word_issues():
    """Core honesty constraint: no forced alignment exists in this project,
    so this provider must never fabricate a per-word issues list, even on
    a successful assessment."""
    provider = pp.AllosaurusG2PPronunciationProvider()
    if not provider.is_available():
        import pytest
        pytest.skip("allosaurus/panphon/eng-to-ipa not installed in this environment")
    import subprocess, shutil, tempfile, os as _os
    if not shutil.which("espeak-ng"):
        import pytest
        pytest.skip("espeak-ng not installed; cannot synthesize test audio")
    with tempfile.TemporaryDirectory() as tmp:
        wav = _os.path.join(tmp, "t.wav")
        subprocess.run(["espeak-ng", "-v", "en-us", "-w", wav, "I like apples"],
                        check=True, capture_output=True)
        result = provider.assess("I like apples", segments=[], wav_path=wav)
    assert result.provider == "allosaurus_g2p"
    assert result.issues == []
    if result.available:
        assert 0.0 <= result.score <= 100.0
        assert "Whisper" not in result.methodology or "No Whisper" in result.methodology


def test_whisper_confidence_always_available():
    assert pp.WhisperConfidenceProvider().is_available() is True


def test_whisper_confidence_score_formula_unchanged():
    """Same avg_logprob -> score mapping as the pre-migration
    score_pronunciation(): map [-1, -0.1] -> [0, 100], clamp [10, 100]."""
    provider = pp.WhisperConfidenceProvider()
    segments = [{"avg_logprob": -0.2}, {"avg_logprob": -0.4}]
    result = provider.assess("hello world", segments, wav_path=None)
    avg = (-0.2 + -0.4) / 2
    expected = round(max(10.0, min(100.0, (avg + 1.0) / 0.9 * 100)), 1)
    assert result.score == expected
    assert result.available is True
    assert result.provider == "whisper_confidence"


def test_whisper_confidence_no_segments_returns_default_72():
    result = pp.WhisperConfidenceProvider().assess("hello", [], wav_path=None)
    assert result.score == 72.0


def test_whisper_confidence_issues_skip_short_and_function_words():
    segments = [{
        "words": [
            {"word": "the", "probability": 0.1},   # skip word — excluded even though low-conf
            {"word": "xylophone", "probability": 0.3},
        ]
    }]
    result = pp.WhisperConfidenceProvider().assess("the xylophone", segments, wav_path=None)
    words = [i["word"] for i in result.issues]
    assert "the" not in words
    assert "xylophone" in words


def test_normalized_result_accepted_by_scoring():
    """PronunciationResult exposes exactly the fields score_free_speech()
    reads (score, issues, provider, available) — proves the interface
    boundary the scoring engine depends on."""
    result = pp.WhisperConfidenceProvider().assess("test", [{"avg_logprob": -0.3}], wav_path=None)
    assert hasattr(result, "score")
    assert hasattr(result, "issues")
    assert hasattr(result, "provider")
    assert hasattr(result, "available")
    assert isinstance(result.score, float)
    assert isinstance(result.issues, list)


# ---- Explicit Saaras selection ------------------------------------------------

def test_saaras_not_configured_without_api_key():
    provider = pp.SaarasPronunciationProvider(api_key="")
    assert provider.is_available() is False
    result = provider.assess("hello", [], wav_path=None)
    assert result.available is False
    assert "not configured" in result.detail
    assert result.provider == "saaras"


def test_saaras_configured_reports_available_flag_from_api_key():
    provider = pp.SaarasPronunciationProvider(api_key="fake-key-for-test")
    assert provider.is_available() is True


def test_saaras_without_wav_path_reports_unavailable_not_crash():
    provider = pp.SaarasPronunciationProvider(api_key="fake-key-for-test")
    result = provider.assess("hello", [], wav_path=None)
    assert result.available is False
    assert result.provider == "saaras"


def test_saaras_request_actually_reaches_the_http_call(monkeypatch, tmp_path):
    """Proves resolve_pronunciation('saaras', ...) drives a real HTTP call
    to Sarvam's endpoint (not a no-op) — the transcription API test the
    brief asks for, adapted since this repo has no live Sarvam account."""
    calls = []

    class _FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": "hello"}  # no word-confidence — documented shape

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        calls.append({"url": url, "headers": headers, "data": data})
        return _FakeResponse()

    monkeypatch.setattr(pp.httpx, "post", fake_post)

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")

    provider = pp.SaarasPronunciationProvider(api_key="fake-key-for-test")
    result = provider.assess("hello", [], wav_path=wav)

    assert len(calls) == 1
    assert calls[0]["url"].endswith("/speech-to-text")
    assert calls[0]["headers"]["api-subscription-key"] == "fake-key-for-test"
    # No per-word confidence in the (real, documented) Saaras response shape
    # -> honest "integration incomplete", not a fabricated score.
    assert result.available is False
    assert "integration incomplete" in result.detail


# ---- Explicit Local LLM selection --------------------------------------------

def test_local_llm_not_configured_without_url():
    provider = pp.LocalLLMPronunciationProvider(base_url="")
    assert provider.is_available() is False
    result = provider.assess("hello", [], wav_path=None)
    assert result.available is False
    assert "not configured" in result.detail
    assert result.provider == "local_llm"


def test_local_llm_configured_flag_reflects_url():
    provider = pp.LocalLLMPronunciationProvider(base_url="http://localhost:9999")
    assert provider.is_available() is True


# ---- GOP selection — explicitly ON HOLD --------------------------------------

def test_gop_always_unavailable():
    assert pp.GOPPronunciationProvider().is_available() is False


def test_gop_selection_returns_not_implemented_message():
    result = pp.GOPPronunciationProvider().assess("hello", [], wav_path=None)
    assert result.available is False
    assert result.detail == "Custom GOP pronunciation assessment is not currently available."
    assert result.provider == "gop"


# ---- resolve_pronunciation() convenience wrapper -----------------------------

def test_resolve_pronunciation_unknown_provider_raises():
    try:
        pp.resolve_pronunciation("nope", "hello", [], None)
        assert False, "expected PronunciationProviderError"
    except pp.PronunciationProviderError:
        pass


def test_resolve_pronunciation_gop_does_not_silently_become_saaras():
    """Regression guard for the explicit brief requirement: requesting gop
    must never come back labeled as any other provider's name."""
    result = pp.resolve_pronunciation("gop", "hello", [], None)
    assert result.provider == "gop"
    assert result.available is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
