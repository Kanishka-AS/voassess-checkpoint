"""
Regression tests for the 2026-08-31 consistency/debug audit.

Covers the 8 minimum cases from the audit request, plus the two concrete
UI/backend contract bugs found while tracing them:

  1. Whisper STT -> transcript + word timings
  2. Saaras STT -> transcript with no fabricated word timings
  3. Explicit Whisper request failure -> fallback to Saaras (the default;
     provider distinguishable); Saaras (default) failure -> clear error,
     no silent Whisper fallback
  4. LanguageTool /v2/analyze -> linguistic_analysis reaches final response,
     AND languagetool_errors reaches it too (the actual bug: it didn't)
  5. filler detected by backend -> reaches final API response's
     filler.occurrences (provider-independent — text-based, not audio-based)
  6. pronunciation provider selection stays independent of STT selection
  7. requested vs actual provider always distinguishable (STT + pronunciation)
  8. no stale "Whisper"-only field is populated when Saaras was actually used
     (_build_word_timings reports source=saaras, available=False, words=[])

Same stubbing pattern as tests/test_app_scoring_integration.py and
tests/test_app_stt_routes.py: `whisper` is stubbed in sys.modules before
`import app` so no real model loads; LanguageToolProvider.check_and_analyze
is monkeypatched per-test to avoid a real network call.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "whisper" not in sys.modules:
    _stub = types.ModuleType("whisper")

    class _StubModel:
        def transcribe(self, *a, **kw):
            return {
                "text": "um this is a test",
                "segments": [{
                    "avg_logprob": -0.2,
                    "words": [
                        {"word": "um", "start": 0.0, "end": 0.3, "probability": 0.4},
                        {"word": "this", "start": 0.3, "end": 0.5, "probability": 0.95},
                        {"word": "is", "start": 0.5, "end": 0.6, "probability": 0.97},
                        {"word": "a", "start": 0.6, "end": 0.65, "probability": 0.99},
                        {"word": "test", "start": 0.65, "end": 1.0, "probability": 0.9},
                    ],
                }],
            }

    _stub.load_model = lambda name: _StubModel()
    sys.modules["whisper"] = _stub

import app  # noqa: E402


def _lt_unavailable(transcript):
    """Simulates a real /v2/analyze failure with a real reason attached —
    the exact case the debug UI's Token Analysis panel needs to explain."""
    return None, None, {"analyze": "connect timeout after 3.0s to http://localhost:8010"}


def _lt_ok(transcript):
    return (
        {"errors": 0, "issues": []},
        {"language": "en-US", "sentences": [{"text": transcript, "tokens": []}]},
        {},
    )


# ---- 4. linguistic_analysis / languagetool_errors reach the final response --

def test_linguistic_analysis_reaches_score_free_speech_response(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_ok)
    r = app.score_free_speech("Hello there.", [], duration=5.0)
    assert r["linguistic_analysis"] is not None
    assert r["linguistic_analysis"]["sentences"][0]["text"] == "Hello there."


def test_languagetool_errors_reaches_score_free_speech_response(monkeypatch):
    """The actual bug found in this audit: this key was silently discarded
    (bound to `_lt_errors` and dropped) even though debug_analyze_text()
    already returned the identical tuple's third element under this name.
    /assess, /assess/stage, and /debug/analyze-audio could never explain
    *why* linguistic_analysis was null — this locks that fix in."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_unavailable)
    r = app.score_free_speech("Hello there.", [], duration=5.0)
    assert r["linguistic_analysis"] is None
    assert r["languagetool_errors"] == {
        "analyze": "connect timeout after 3.0s to http://localhost:8010"
    }


# ---- 5. filler detection reaches the final response, independent of STT ----

def test_filler_occurrences_reach_final_response_with_no_segments(monkeypatch):
    """detect_fillers() runs on transcript text (+ optional linguistic_analysis),
    not on audio segments — must produce real occurrences even when segments=[]
    (i.e. even when the STT provider was Saaras, which never returns segments)."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_unavailable)
    r = app.score_free_speech("Um, this is a test.", segments=[], duration=5.0)
    assert r["filler"]["count"] == 1
    assert len(r["filler"]["occurrences"]) == 1
    # Occurrence text preserves original transcript casing ("Um", not "um") —
    # the lowercased comparison here matches how summarize_words()/filler.words
    # already normalize it elsewhere in app.py.
    assert r["filler"]["occurrences"][0]["word"].lower() == "um"


# ---- 6. pronunciation provider selection is independent of STT selection ---

def test_pronunciation_provider_independent_of_stt_provider(monkeypatch):
    """Selecting Saaras as STT must not silently select (or fabricate) a
    Saaras pronunciation score. With no SARVAM_API_KEY configured and
    pronunciation_provider left at the default, the score must still come
    from the default pronunciation provider (allosaurus_g2p — Whisper-free;
    see pronunciation_provider.py), never from "saaras" — the two
    selections are orthogonal. No audio is passed here, so the default
    provider honestly reports unavailable too (there is no Whisper fallback
    to silently paper over that) — the point of this test is that the
    *provider name* is never "saaras", not that a score of 0 is meaningful."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_unavailable)
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    wav = None
    segments_from_saaras = []  # SaarasSTTProvider always returns segments=[]
    r = app.score_free_speech("This is a test.", segments_from_saaras, duration=5.0,
                               pronunciation_provider=app.DEFAULT_PRONUNCIATION_PROVIDER)
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["requested_provider"] == "allosaurus_g2p"
    assert r["pronunciation"]["provider"] != "saaras"


def test_saaras_pronunciation_never_fabricated(monkeypatch):
    """Explicitly requesting the (incomplete) Saaras pronunciation provider
    must fall back honestly, never invent a score under Saaras's name."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_unavailable)
    r = app.score_free_speech("This is a test.", [], duration=5.0,
                               pronunciation_provider="saaras")
    assert r["pronunciation"]["requested_provider"] == "saaras"
    assert r["pronunciation"]["provider"] == "allosaurus_g2p"  # honest fallback, not saaras
    assert r["pronunciation"]["available"] is False
    assert r["pronunciation"]["detail"]  # a real reason, not blank


# ---- 3 & 7. STT requested vs actual provider always distinguishable --------

def test_stt_requested_vs_actual_distinguishable_on_fallback(monkeypatch, tmp_path):
    """Saaras is now DEFAULT_STT_PROVIDER, so an explicit *whisper* request
    that fails is the fallback case (falls back to Saaras, the default) —
    the reverse of when Whisper was the default."""
    import stt_provider as sp

    class _EmptyWhisperModel:
        def transcribe(self, *a, **kw):
            return {"text": "   ", "segments": []}

    monkeypatch.setitem(app.stt_registry._providers, "whisper",
                         sp.WhisperSTTProvider(_EmptyWhisperModel()))

    class _StubSaarasResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"transcript": "hello from saaras"}

    monkeypatch.setattr(sp.httpx, "post", lambda *a, **kw: _StubSaarasResponse())
    monkeypatch.setitem(app.stt_registry._providers, "saaras", sp.SaarasSTTProvider(api_key="fake-key"))

    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    requested, used = app.resolve_stt("whisper", wav)
    assert requested.provider == "whisper"
    assert requested.available is False
    assert used.provider == "saaras"
    assert used.available is True
    # A caller building an /assess-style response can always tell what was
    # asked for vs what actually ran:
    stt_info = {
        "provider": used.provider,
        "requested_provider": "whisper",
        "available": requested.available,
        "detail": requested.detail,
    }
    assert stt_info["requested_provider"] != stt_info["provider"]
    assert stt_info["detail"]


def test_stt_default_saaras_failure_is_not_silently_replaced_by_whisper(monkeypatch, tmp_path):
    """The DEFAULT provider (Saaras) failing must surface as a clear error,
    not a silent switch to Whisper's (filler-normalizing) transcript."""
    from fastapi import HTTPException
    import stt_provider as sp
    monkeypatch.setitem(app.stt_registry._providers, "saaras", sp.SaarasSTTProvider(api_key=""))
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")
    try:
        app.resolve_stt("saaras", wav)
        assert False, "expected HTTPException"
    except HTTPException as e:
        assert e.status_code == 400
        assert "not configured" in e.detail


# ---- 1, 2, 8. word_timings is provider-neutral and never mislabels Saaras --

def test_word_timings_whisper_source_has_real_words():
    segments = [{
        "avg_logprob": -0.1,
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.4, "probability": 0.9},
            {"word": "world", "start": 0.4, "end": 0.8, "probability": 0.85},
        ],
    }]
    wt = app._build_word_timings(segments, "whisper")
    assert wt["source"] == "whisper"
    assert wt["available"] is True
    assert len(wt["words"]) == 2
    assert wt["words"][0]["word"] == "hello"
    assert wt["words"][0]["duration"] == 0.4


def test_word_timings_saaras_source_is_honestly_empty_not_whisper():
    """The audit's core Problem 2: Saaras returns segments=[] (no word-level
    timestamps — see stt_provider.py). word_timings must report
    source='saaras' and available=False, never silently look like an empty
    Whisper result (which the old 'whisper_words' field name implied)."""
    wt = app._build_word_timings([], "saaras")
    assert wt["source"] == "saaras"
    assert wt["available"] is False
    assert wt["words"] == []


def test_legacy_filler_occurrences_empty_when_no_word_timings():
    """The legacy Whisper-word-match cross-check is correctly empty when
    there's no word-level timing data to match against (Saaras) — this is
    expected, not a sign that filler detection itself failed (see
    filler.occurrences in the tests above for the real, always-populated
    signal)."""
    assert app._find_filler_occurrences([]) == []


# ---- "Grammar looks clean!" vs grammar score contradiction -----------------
#
# The exact case from the 2026-08-31 debug trace: 13-word transcript, 1
# LanguageTool match -> score_grammar() rate = 1/13 ≈ 0.077, landing in the
# `elif rate < 0.08: return 56` bracket (score 56/100) -- but build_feedback()
# used to receive only the raw error count `ge` (=1), which fell into its
# `else` branch ("ge >= 2" was false) and printed "Grammar looks clean!",
# directly contradicting the 56 score shown in the same response.

def _lt_one_error(transcript):
    """Simulates a real /v2/check response with exactly one match, mirroring
    the 'She go ... yesterday' trace (go -> goes, subject-verb agreement)."""
    return (
        {
            "errors": 1,
            "issues": [{
                "wrong": "go", "correct": "goes",
                "message": "The verb form is not correct for a third-person subject.",
                "context": "She go to the store yesterday...",
            }],
        },
        None,
        {},
    )


def test_grammar_score_56_is_not_reported_as_clean(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_one_error)
    transcript = "She go to the store yesterday and me and him was very tired."
    assert len(transcript.split()) == 13  # pins the exact audit case
    r = app.score_free_speech(transcript, [], duration=5.0)
    assert r["grammar"]["errors"] == 1
    assert r["grammar"]["score"] == 56  # score_grammar(1, 13) -- unchanged
    # Exact phrase check, not a bare "clean" substring match -- the "minor"
    # branch's own wording ("cleaner delivery") legitimately contains
    # "clean" as a substring, so a substring check would false-positive.
    assert "grammar looks clean!" not in r["feedback"].lower()
    assert "1" in r["feedback"]  # still surfaces the real error count


def test_build_feedback_grammar_thresholds_follow_score_not_raw_count():
    """Direct unit coverage of the new score-based branching in
    build_feedback(), independent of the LanguageTool/network path above."""
    common = dict(wpm=130, pace_s=100, fc=0, filler_words=[], overall=80, clarity_s=80)

    # score 56 (the audit case) -> not "clean" (exact-phrase check -- the
    # "minor" branch's own wording legitimately contains "clean" as a
    # substring, via "cleaner delivery").
    msg = app.build_feedback(ge=1, pronun_s=90, grammar_s=56, **common)
    assert "grammar looks clean!" not in msg.lower()
    assert "minor grammar issues" in msg.lower()

    # score 100 (zero errors) -> still "clean".
    msg = app.build_feedback(ge=0, pronun_s=90, grammar_s=100, **common)
    assert "grammar looks clean!" in msg.lower()

    # score 90 (very low error rate on a long transcript) -> still "clean".
    msg = app.build_feedback(ge=1, pronun_s=90, grammar_s=90, **common)
    assert "grammar looks clean!" in msg.lower()

    # score 36 (high error rate) -> the severe branch, not "minor"/"clean".
    msg = app.build_feedback(ge=3, pronun_s=90, grammar_s=36, **common)
    assert "grammar looks clean!" not in msg.lower()
    assert "minor grammar issues" not in msg.lower()
    assert "grammar issues detected" in msg.lower()


def test_build_feedback_without_grammar_s_falls_back_to_old_count_thresholds():
    """grammar_s defaults to None so any caller still using the old 8-arg
    signature keeps its previous (count-based) behavior rather than breaking."""
    common = dict(wpm=130, pace_s=100, fc=0, filler_words=[], overall=80, clarity_s=80)
    msg = app.build_feedback(ge=1, pronun_s=90, **common)
    assert "grammar looks clean!" in msg.lower()  # old behavior: ge=1 < 2 -> "clean"


# ---- grammar_source threaded through score_free_speech() -------------------
#
# debug_analyze_text() already returned `grammar_source` (which of
# languagetool_http / language_tool_python_local / regex_fallback actually
# produced `grammar`), but score_free_speech() -- shared by /assess,
# /assess/stage, and /debug/analyze-audio -- did not, so none of those three
# routes could self-report which grammar path ran.

def test_grammar_source_is_languagetool_http_when_lt_available(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_one_error)
    r = app.score_free_speech("She go to the store yesterday.", [], duration=5.0)
    assert r["grammar_source"] == "languagetool_http"


def test_grammar_source_falls_back_when_lt_unavailable(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _lt_unavailable)
    r = app.score_free_speech("This is a test.", [], duration=5.0)
    # Whichever fallback actually ran (local language_tool_python if
    # GRAMMAR_OK, else the naive regex scan) -- either way it must be
    # reported, and it must not claim the HTTP path ran.
    assert r["grammar_source"] in ("language_tool_python_local", "regex_fallback")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
