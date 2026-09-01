"""
Tests for filler_detector.py.

No LanguageTool server or network access required — `linguistic_analysis`
fixtures below are hand-built in the same AnalyzeResponse shape
languagetool_provider.analyze() returns (see
tests/test_languagetool_provider.py's ANALYZE_RESPONSE_FIXTURE), with
startOffset/endOffset matching each fixture's transcript exactly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filler_detector import detect_fillers


def _tok(text, start, transcript, pos=None, lemma=None):
    """Build one AnalyzeResponse-shaped token, offsets computed from `text`."""
    return {"text": text, "lemma": lemma or text.lower(), "partOfSpeech": pos or "",
            "posTag": pos or "", "startOffset": start, "endOffset": start + len(text)}


def _analysis(transcript, tokens):
    """tokens: list of (text, start, pos) — start is the char offset in
    `transcript` (use transcript.index(...) at call sites to avoid drift)."""
    toks = [_tok(text, start, transcript, pos=pos) for (text, start, pos) in tokens]
    return {"language": "en-US", "sentences": [{"text": transcript, "tokens": toks}]}


def _offsets(transcript, words):
    """Helper: find each word's char offset in transcript, occurrence by occurrence."""
    out = []
    cursor = 0
    for w in words:
        idx = transcript.index(w, cursor)
        out.append(idx)
        cursor = idx + len(w)
    return out


# ---- Clear fillers (filled pauses) ------------------------------------------

def test_uh_is_filler():
    transcript = "Uh, I think this is good."
    r = detect_fillers(transcript, None)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "uh" in words
    occ = next(o for o in r["occurrences"] if o["word"].lower() == "uh")
    assert occ["type"] == "filled_pause"
    assert occ["confidence"] >= 0.9


def test_um_is_filler():
    transcript = "I was, um, working on the project."
    r = detect_fillers(transcript, None)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "um" in words


def test_er_is_filler():
    transcript = "I, er, don't know."
    r = detect_fillers(transcript, None)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "er" in words


# ---- Normal usage (must NOT be flagged) -------------------------------------

def test_like_as_verb_is_not_filler():
    transcript = "I like programming."
    start = transcript.index("like")
    analysis = _analysis(transcript, [
        ("I", 0, "PRP"), ("like", start, "VBP"), ("programming", start + 5, "NN"),
    ])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


def test_i_mean_followed_by_object_is_not_filler():
    transcript = "I mean what I said."
    start = transcript.index("mean")
    analysis = _analysis(transcript, [
        ("I", 0, "PRP"), ("mean", start, "VBP"), ("what", start + 5, "WP"),
    ])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


def test_actually_as_plain_adverb_is_not_filler():
    transcript = "That is actually important."
    start = transcript.index("actually")
    analysis = _analysis(transcript, [
        ("That", 0, "DT"), ("is", 5, "VBZ"), ("actually", start, "RB"),
    ])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


def test_so_as_connective_is_not_filler():
    transcript = "So I went home."
    analysis = _analysis(transcript, [("So", 0, "RB"), ("I", 3, "PRP")])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


def test_well_opening_a_response_is_not_filler():
    transcript = "Well, this is the answer."
    analysis = _analysis(transcript, [("Well", 0, "UH"), ("this", 6, "DT")])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


# ---- Ambiguous usage ---------------------------------------------------------

def test_like_comma_bracketed_is_filler():
    transcript = "It was, like, really difficult."
    start = transcript.index("like")
    analysis = _analysis(transcript, [
        ("It", 0, "PRP"), ("was", 3, "VBD"), ("like", start, "UH"),
        ("really", start + 6, "RB"), ("difficult", start + 13, "JJ"),
    ])
    r = detect_fillers(transcript, analysis)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "like" in words
    occ = next(o for o in r["occurrences"] if o["word"].lower() == "like")
    assert occ["type"] == "discourse_filler"


def test_i_mean_uh_result_flags_uh_and_mean():
    transcript = "I mean, uh, the result was good."
    mean_start = transcript.index("mean")
    uh_start = transcript.index("uh")
    analysis = _analysis(transcript, [
        ("I", 0, "PRP"), ("mean", mean_start, "VBP"), ("uh", uh_start, "UH"),
        ("the", uh_start + 4, "DT"),
    ])
    r = detect_fillers(transcript, analysis)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "uh" in words
    assert "mean" in words


def test_you_know_standalone_is_filler():
    transcript = "You know, I think this is useful."
    analysis = _analysis(transcript, [
        ("You", 0, "PRP"), ("know", 4, "VBP"), ("I", 10, "PRP"),
    ])
    r = detect_fillers(transcript, analysis)
    phrases = [o["word"].lower() for o in r["occurrences"]]
    assert "you know" in phrases


def test_i_know_you_is_not_filler():
    """Reversed word order ('know' before 'you') must not match the
    'you know' phrase detector, and 'know' isn't an ambiguous filler word
    on its own."""
    transcript = "I know you."
    analysis = _analysis(transcript, [
        ("I", 0, "PRP"), ("know", 2, "VBP"), ("you", 7, "PRP"),
    ])
    r = detect_fillers(transcript, analysis)
    assert r["count"] == 0


# ---- Repetition / hesitation (separate signal, not counted as fillers) -----

def test_immediate_word_repetition_is_hesitation_not_filler():
    transcript = "I I think the project is good."
    r = detect_fillers(transcript, None)
    assert r["count"] == 0  # "I I" must not be counted as a filler word
    assert len(r["hesitations"]) == 1
    assert r["hesitations"][0]["type"] == "repetition"


def test_ellipsis_repetition_is_hesitation():
    transcript = "I... I think..."
    r = detect_fillers(transcript, None)
    assert len(r["hesitations"]) == 1
    assert r["hesitations"][0]["phrase"].lower().startswith("i")


def test_the_the_repetition_is_hesitation():
    transcript = "The the project was completed."
    r = detect_fillers(transcript, None)
    assert len(r["hesitations"]) == 1
    assert r["count"] == 0


# ---- No linguistic_analysis (LanguageTool unavailable) — graceful fallback --

def test_falls_back_gracefully_without_linguistic_analysis():
    transcript = "Uh, I think this is good, like, really good."
    r = detect_fillers(transcript, None)
    words = [o["word"].lower() for o in r["occurrences"]]
    assert "uh" in words
    # "like" has no POS data to rule out verb usage, but IS comma-bracketed,
    # so the punctuation-only signal should still catch it.
    assert "like" in words


def test_empty_transcript_returns_zero_evidence():
    r = detect_fillers("", None)
    assert r["count"] == 0
    assert r["occurrences"] == []
    assert r["hesitations"] == []


# ---- rate_per_min ------------------------------------------------------------

def test_rate_per_min_computed_from_duration():
    transcript = "Uh, um, er, this has three fillers in thirty seconds."
    r = detect_fillers(transcript, None, duration_seconds=30)
    assert r["count"] == 3
    assert r["rate_per_min"] == 6.0


def test_rate_per_min_none_without_duration():
    transcript = "Uh, this has one filler."
    r = detect_fillers(transcript, None)
    assert r["rate_per_min"] is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, repr(e)))
    print(f"PASSED: {passed}/{len(tests)}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
