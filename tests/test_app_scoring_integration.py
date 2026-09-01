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
