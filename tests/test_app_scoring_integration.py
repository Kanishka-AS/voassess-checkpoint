"""
Integration tests for score_free_speech() in app.py — proving the Phase 1
wiring is real, not just present in the diff:

  1. Filler counting now comes from filler_detector.detect_fillers()
     (context-aware), not the old flat FILLERS regex scan — verified with
     a case the two approaches disagree on ("I like programming" — old
     regex counts "like" as a filler; the real detector does not, since
     it's a verb).
  2. Repetitions ("I I") are surfaced separately as `hesitations`, never
     folded into the filler count.
  3. `score_free_speech()` takes `duration` as a parameter and does not
     independently re-derive it from anything client-side (the duration
     fix itself — using the real WAV duration instead of client wall-clock
     — lives in the /assess, /assess/stage, /debug/analyze-audio route
     handlers, which call save_and_convert() + wav_duration_seconds();
     that part needs a real audio file and Whisper and is exercised
     manually per the instructions given alongside this change, not here).

app.py loads a real Whisper model and constructs a LanguageTool HTTP
client at import time. Neither is needed for these tests (no audio is
transcribed here — segments are supplied directly), so `whisper` is
stubbed in sys.modules before import, exactly enough for
`whisper.load_model("base")` to succeed without a network call/model
download. LanguageToolProvider.check_and_analyze() is monkeypatched
per-test to avoid real network calls to a LanguageTool server and to
control whether POS/lemma data is "available" for a given test case —
its own HTTP behavior is already covered by tests/test_languagetool_provider.py.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- Stub `whisper` before app.py imports it -------------------------------
if "whisper" not in sys.modules:
    _stub = types.ModuleType("whisper")

    class _StubModel:
        def transcribe(self, *a, **kw):
            raise NotImplementedError("stub Whisper model — not used by these tests")

    _stub.load_model = lambda name: _StubModel()
    sys.modules["whisper"] = _stub

import app  # noqa: E402  (must come after the whisper stub above)


def _no_languagetool(transcript):
    """Simulates 'LanguageTool server unreachable' — (None, None, {errors}).
    filler_detector degrades gracefully to punctuation/word-list-only rules
    in this mode (see filler_detector.py docstring); grammar falls back to
    the existing local/regex path exactly as it did before this change."""
    return None, None, {"check": "stub: no LT server", "analyze": "stub: no LT server"}


def test_verb_like_not_counted_as_filler(monkeypatch):
    """Old behavior (flat FILLERS regex) would have counted 'like' here.
    The real detector correctly excludes it (used as a verb)."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("I like programming and I like reading.", [], duration=5.0)
    words = [w.split("×")[0] for w in r["filler"]["words"]]
    assert "like" not in words
    assert r["filler"]["count"] == 0


def test_comma_bracketed_like_is_still_counted_as_filler(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("It was, like, really difficult.", [], duration=5.0)
    words = [w.split("×")[0] for w in r["filler"]["words"]]
    assert "like" in words
    assert r["filler"]["count"] == 1


def test_uh_um_still_counted_as_fillers(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("Uh, I think, um, this is good.", [], duration=5.0)
    assert r["filler"]["count"] == 2


def test_repetition_is_hesitation_not_filler(monkeypatch):
    """'I I' must not inflate filler_count/filler_score, and must show up
    separately under the new top-level `hesitations` key."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("I I think the project is good.", [], duration=5.0)
    assert r["filler"]["count"] == 0
    assert len(r["hesitations"]) == 1
    assert r["hesitations"][0]["type"] == "repetition"


def test_filler_score_formula_unchanged(monkeypatch):
    """score_fillers()'s own thresholds are untouched by this change —
    spot-check the exact same ratio→score mapping that existed before."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    # 20 words, 0 fillers -> ratio 0 -> 100 (score_fillers unchanged)
    transcript = " ".join(["word"] * 20)
    r = app.score_free_speech(transcript, [], duration=10.0)
    assert r["filler"]["score"] == 100
    assert r["filler"]["count"] == 0


def test_filler_occurrences_and_rate_per_min_are_additive_fields(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("Uh, this has one filler in it.", [], duration=10.0)
    assert isinstance(r["filler"]["occurrences"], list)
    assert len(r["filler"]["occurrences"]) == 1
    assert r["filler"]["rate_per_min"] == 6.0  # 1 filler / (10s/60)


def test_fluency_is_present_and_not_folded_into_overall(monkeypatch):
    """New `fluency` field must exist and must not change the pre-existing
    `overall`/`clarity` formulas (informational only, same contract as
    vocabulary/cefr)."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    transcript = " ".join(["word"] * 20)
    r = app.score_free_speech(transcript, [], duration=10.0)
    assert "fluency" in r
    assert 0 <= r["fluency"]["score"] <= 100
    # overall must still equal the documented five-metric blend — fluency
    # must not have been mixed in.
    expected_overall = round(
        r["pace"]["score"] * 0.20 + r["filler"]["score"] * 0.20 +
        r["pronunciation"]["score"] * 0.25 + r["grammar"]["score"] * 0.20 +
        r["clarity"]["score"] * 0.15, 1)
    assert r["overall"] == expected_overall


def test_fluency_reports_no_pause_data_when_segments_empty(monkeypatch):
    """No word-timestamp segments (e.g. Saaras STT, or these unit tests
    which pass segments=[]) -> fluency still returns a score, but honestly
    flags that pause evidence wasn't available rather than fabricating
    pause numbers."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("This is a short test sentence.", [], duration=5.0)
    assert r["fluency"]["pause_data_available"] is False
    assert r["fluency"]["pause_count"] == 0


def test_fluency_penalizes_real_pauses_from_word_timestamps(monkeypatch):
    """With real Whisper-style word timestamps, a long silent gap between
    words must lower the fluency score relative to the same transcript
    spoken without any gap — proves analyze_pauses()'s output actually
    drives the score, not just decoration."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    transcript = "I think the answer is yes"

    smooth_segments = [{"words": [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "think", "start": 0.25, "end": 0.6},
        {"word": "the", "start": 0.65, "end": 0.8},
        {"word": "answer", "start": 0.85, "end": 1.2},
        {"word": "is", "start": 1.25, "end": 1.4},
        {"word": "yes", "start": 1.45, "end": 1.7},
    ]}]
    choppy_segments = [{"words": [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "think", "start": 0.25, "end": 0.6},
        {"word": "the", "start": 2.6, "end": 2.75},    # 2s hesitation pause
        {"word": "answer", "start": 4.8, "end": 5.15},  # another 2s+ pause
        {"word": "is", "start": 5.2, "end": 5.35},
        {"word": "yes", "start": 5.4, "end": 5.65},
    ]}]

    r_smooth = app.score_free_speech(transcript, smooth_segments, duration=2.0)
    r_choppy = app.score_free_speech(transcript, choppy_segments, duration=6.0)

    assert r_smooth["fluency"]["pause_data_available"] is True
    assert r_choppy["fluency"]["long_pause_count"] == 2
    assert r_choppy["fluency"]["score"] < r_smooth["fluency"]["score"]


def test_low_evidence_flag_on_short_response(monkeypatch):
    """A very short/brief response must be flagged low_evidence=True
    without altering any of the actual metric scores."""
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    r = app.score_free_speech("Fine.", [], duration=1.0)
    assert r["evidence"]["low_evidence"] is True
    assert r["evidence"]["reason"] is not None
    assert "short" in r["feedback"].lower()


def test_low_evidence_flag_absent_on_substantial_response(monkeypatch):
    monkeypatch.setattr(app._lt_provider, "check_and_analyze", _no_languagetool)
    transcript = " ".join(["word"] * 40)
    r = app.score_free_speech(transcript, [], duration=20.0)
    assert r["evidence"]["low_evidence"] is False
    assert r["evidence"]["reason"] is None


def test_only_one_languagetool_call_per_scoring_pass(monkeypatch):
    """Regression guard: filler detection and grammar checking must share
    the single check_and_analyze() call, not trigger two network round
    trips per assessment."""
    calls = []

    def counting_stub(transcript):
        calls.append(transcript)
        return _no_languagetool(transcript)

    monkeypatch.setattr(app._lt_provider, "check_and_analyze", counting_stub)
    app.score_free_speech("This is a short test sentence.", [], duration=5.0)
    assert len(calls) == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
