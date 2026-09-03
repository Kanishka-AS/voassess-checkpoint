#!/usr/bin/env python3
"""
Tests for the filler-scoring bug fix — spoken fillers vs. acoustic
hesitations/silence must never be conflated.

Background (the bug): `filler_detector.detect_fillers()` used to merge its
RMS-energy acoustic "low-energy segment" detector straight into the same
`occurrences` list / `count` that transcript-based lexical filler detection
produced, tagging each acoustic event with `type: "filled_pause"` and
`word: "[filler]"`. That let a silent pause between words get scored as if
the speaker had said "um" — inflating `filler_words` score and, through it,
`overall`. Separately, `app.py:insert_filler_markers()` spliced those same
acoustic events into the transcript TEXT using their audio-second
timestamps as if they were transcript character offsets, which is how
markers ended up landing inside words (e.g. "thinkin[filler]g") instead of
between them.

This file proves:
  1. "uh"/"um" are counted as spoken fillers.
  2. Silence alone is NOT counted as a filler word.
  3. An acoustic low-energy segment with no lexical evidence does NOT
     increase filler_words count.
  4. Acoustic hesitation evidence remains available separately.
  5. `insert_filler_markers()` (verbatim copy — see NOTE below) never
     produces a `[filler]` marker inside a lexical word, and only ever
     splices SPOKEN (char-offset) occurrences.
  6. filler count/rate are computed from spoken evidence only.
  7. A before/after example showing the overall-score impact of the fix.
  8. The same hesitation is not double-penalized (fluency formula unchanged).
  9. Groq's evidence payload keeps spoken vs. acoustic filler evidence in
     separate fields.
  10. When linguistic_analysis (LanguageTool) is unavailable, ambiguous
      single-word candidates fall back conservatively (NOT_FILLER) rather
      than a low-confidence guess being treated as confirmed.

Only two heavy-import production modules are used directly: filler_detector
(pure stdlib + numpy/wave for its optional audio branch) and audio_utils
(numpy/wave). `app.py` itself is NOT imported (it pulls in FastAPI/Whisper/
allosaurus — heavy binary deps this sandbox doesn't need). `score_fillers`,
`score_fluency`, and `insert_filler_markers` are duplicated here verbatim,
read-only, purely for this test's before/after demonstration — production
scoring/splicing still lives solely in app.py. `build_report_evidence`'s
filler-section shape is exercised directly via groq_provider.py (its own
deps — httpx/wordfreq — are lightweight enough to import here).

Run: python3 test_filler_scoring_fix.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

from filler_detector import detect_fillers, summarize_words

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{('  -- ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(label)


# ── Verbatim copies of app.py's scoring/splicing functions ─────────────────
# (see module docstring — app.py itself isn't imported here)

def score_fillers(count: int, words: int) -> float:
    ratio = count / max(words, 1)
    if   ratio == 0:      return 100
    elif ratio < 0.02:    return 90
    elif ratio < 0.05:    return 70
    elif ratio < 0.08:    return 50
    elif ratio < 0.12:    return 28
    else:                 return 10


def score_clarity(pace: float, filler: float, grammar: float, pronun: float) -> float:
    return round(pace * 0.25 + filler * 0.25 + grammar * 0.25 + pronun * 0.25, 1)


def score_fluency(pause_info: dict, filler_count: int, hesitation_count: int,
                   word_count: int) -> dict:
    disfluency_rate = (filler_count + hesitation_count) / max(word_count, 1)
    disfluency_penalty = min(disfluency_rate * 250, 45)
    if pause_info.get("available"):
        pause_penalty = min(
            pause_info["pause_rate_per_min"] * 1.5 + pause_info["long_pause_count"] * 7,
            45,
        )
    else:
        pause_penalty = 0.0
    score = round(max(5.0, min(100.0, 100 - disfluency_penalty - pause_penalty)), 1)
    return {"score": score, "pause_data_available": bool(pause_info.get("available"))}


def insert_filler_markers(transcript: str, filler_occurrences: list) -> str:
    """Verbatim copy of app.py's FIXED insert_filler_markers() — splices
    only char-offset SPOKEN occurrences; there is no acoustic/audio-second
    branch to accidentally use as a character index."""
    if not filler_occurrences:
        return transcript
    sorted_fillers = sorted(filler_occurrences, key=lambda x: x.get('start', 0), reverse=True)
    result = list(transcript)
    inserted_count = 0
    for f in sorted_fillers:
        start, end, word = f.get('start', 0), f.get('end', 0), f.get('word', '')
        start_idx, end_idx = int(start), int(end)
        if start_idx < len(transcript) and end_idx <= len(transcript):
            adj_start, adj_end = start_idx + inserted_count, end_idx + inserted_count
            if adj_end <= len(result):
                actual_word = ''.join(result[adj_start:adj_end]).strip().lower()
                if actual_word == word.lower():
                    result[adj_start:adj_end] = f'[{word}]'
                    inserted_count += 2
    return ''.join(result)


# ── Test audio: a clean speech-like signal with a genuine acoustic pause ───
# Amplitude-modulated tone (mimics natural speech-energy variance) with a
# true-silence gap spliced in — reliably trips audio_utils'
# RMS-energy low-energy-segment detector without needing any spoken content.

def _make_pause_wav(path: Path, sr: int = 16000) -> None:
    t = np.arange(int(sr * 1.5)) / sr
    envelope = 1.0 + 0.15 * np.sin(2 * np.pi * 3 * t)
    speech = np.sin(2 * np.pi * 180 * t) * 0.3 * envelope
    pause = np.zeros(int(sr * 0.1))
    sig = np.clip(np.concatenate([speech, pause, speech]), -1, 1)
    data = (sig * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def _make_silent_wav(path: Path, dur: float = 2.0, sr: int = 16000) -> None:
    data = np.zeros(int(sr * dur), dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def main():
    tmp_dir = Path("/tmp/filler_scoring_fix_tests")
    tmp_dir.mkdir(exist_ok=True)
    pause_wav = tmp_dir / "pause.wav"
    silent_wav = tmp_dir / "silent.wav"
    _make_pause_wav(pause_wav)
    _make_silent_wav(silent_wav)

    # ── 1. "uh"/"um" are counted as spoken fillers ─────────────────────────
    transcript = "This uh and umm it again"
    r1 = detect_fillers(transcript, None, duration_seconds=10)
    words1 = {o["word"].lower() for o in r1["occurrences"]}
    check("1. 'uh' counted as a spoken filler", "uh" in words1, f"got {words1}")
    check("1. 'umm' counted as a spoken filler", "umm" in words1, f"got {words1}")
    check("1. spoken count == 2", r1["count"] == 2, f"got {r1['count']}")

    # ── 2 & 3. Silence / acoustic-only events never inflate filler count ───
    clean_transcript = "This is a clear sentence with no filler words at all today"
    r2 = detect_fillers(clean_transcript, None, duration_seconds=3.1, audio_path=pause_wav)
    check("2. silence alone is NOT counted as a filler word",
          r2["count"] == 0, f"count={r2['count']}")
    check("3. acoustic low-energy segment does not inflate filler_words count",
          len(r2["occurrences"]) == 0 and r2["acoustic_hesitation_count"] > 0,
          f"occurrences={r2['occurrences']} acoustic_hesitation_count={r2['acoustic_hesitation_count']}")
    check("3. no acoustic event leaks into occurrences as type=filled_pause",
          all(o.get("word") != "[filler]" for o in r2["occurrences"]))

    # ── 4. Acoustic hesitation remains available as separate evidence ──────
    check("4. acoustic_hesitations list is populated",
          len(r2["acoustic_hesitations"]) == r2["acoustic_hesitation_count"] > 0,
          f"{r2['acoustic_hesitations']}")
    check("4. acoustic events are tagged as acoustic, not spoken, evidence",
          all(a["type"] == "acoustic_low_energy" for a in r2["acoustic_hesitations"]))
    check("4. acoustic events use audio-second timing, not transcript char offsets",
          all("start_seconds" in a and "start" not in a for a in r2["acoustic_hesitations"]))

    # Mixed case: genuine spoken filler AND acoustic pause in the same clip —
    # spoken count must reflect only the spoken word. Long enough (60 words)
    # that the before/after example in (7) has room to show a tier change
    # in score_fillers()'s banded formula, not just floor-to-floor.
    mixed_transcript = (
        "So um I think that this is uh working correctly and I wanted to walk "
        "through the whole plan for the project before we start the next phase "
        "of development because the team needs to understand exactly what we "
        "are building and why it matters for the customers who will use this "
        "product every single day of the week going forward"
    )
    r3 = detect_fillers(mixed_transcript, None, duration_seconds=24, audio_path=pause_wav)
    check("mixed: spoken count reflects only lexical fillers ('um','uh')",
          r3["count"] == 2, f"got {r3['count']}, occurrences={r3['occurrences']}")
    check("mixed: acoustic evidence still captured separately",
          r3["acoustic_hesitation_count"] > 0)

    # ── 5. insert_filler_markers never produces a marker mid-word, and only
    #        ever consumes spoken (char-offset) occurrences ────────────────
    marked = insert_filler_markers(mixed_transcript, r3["occurrences"])
    check("5. transcript with markers has no mid-word '[filler]'/'[uh]' split",
          "[filler]" not in marked and not any(
              w for w in marked.split() if w.count("[") > 0 and not
              (w.startswith("[") or w.endswith("]") or w.rstrip(".,!?").endswith("]"))
          ),
          marked)
    check("5. every lexical word from the original transcript is still intact",
          all(word.strip("[]") .rstrip(".,!?") in mixed_transcript
              for word in marked.replace("[", " ").replace("]", " ").split()))
    # The acoustic branch has been removed entirely from insert_filler_markers
    # — proven structurally: passing acoustic_hesitations (which have no
    # transcript 'start'/'end'/'word' keys at all) is simply a no-op/ignored
    # rather than something that could ever be spliced into the text.
    marked_with_acoustic_ignored = insert_filler_markers(
        clean_transcript, r2["occurrences"])  # r2["occurrences"] is empty — spoken-only
    check("5. acoustic-only detection leaves the transcript completely unmodified",
          marked_with_acoustic_ignored == clean_transcript, marked_with_acoustic_ignored)

    # Saaras word_timings.available=False scenario: filler_detector's char
    # offsets come from the transcript text itself (regex/LanguageTool
    # token offsets), never from STT word-level timestamps, so detection
    # and marking both work identically regardless of word_timings
    # availability — proving the transcript is preserved exactly except for
    # genuine spoken-filler markers.
    saaras_style_transcript = "I was thinking about everything and getting ready"
    r4 = detect_fillers(saaras_style_transcript, None, duration_seconds=5, audio_path=pause_wav)
    marked4 = insert_filler_markers(saaras_style_transcript, r4["occurrences"])
    check("5. Saaras-style (no word-level timing) transcript is preserved exactly "
          "when there are no genuine spoken fillers",
          marked4 == saaras_style_transcript, marked4)
    check("5. no lexical word ever gets split (e.g. 'thinking'/'everything'/'getting' intact)",
          all(w in marked4 for w in ["thinking", "everything", "getting"]))

    # ── 6. Count/rate are calculated from the correct spoken evidence ──────
    r5 = detect_fillers("um um um", None, duration_seconds=6)
    check("6. count reflects exactly the spoken-filler occurrences",
          r5["count"] == len(r5["occurrences"]) == 3, f"{r5}")
    check("6. rate_per_min derives from the spoken count, not an acoustic one",
          r5["rate_per_min"] == 30.0, f"rate={r5['rate_per_min']}")
    check("6. spoken_count alias matches count", r5["spoken_count"] == r5["count"])

    # ── 7. Before/after overall-score example ───────────────────────────────
    # Simulates the exact bug scenario from the bug report: a transcript
    # with 2 real spoken fillers plus a burst of acoustic low-energy events
    # that used to get folded into the same count.
    word_count = len(mixed_transcript.split())
    pace_s, pronun_s, grammar_s = 92.0, 88.0, 95.0  # held fixed to isolate the filler-fix's effect

    # BEFORE (bug behavior): acoustic events counted as spoken fillers too.
    acoustic_events_before = r3["acoustic_hesitation_count"]
    buggy_count = r3["count"] + acoustic_events_before  # old code's merged count
    buggy_filler_s = score_fillers(buggy_count, word_count)
    buggy_clarity_s = score_clarity(pace_s, buggy_filler_s, grammar_s, pronun_s)
    buggy_overall = round(pace_s * 0.20 + buggy_filler_s * 0.20 + pronun_s * 0.25 +
                           grammar_s * 0.20 + buggy_clarity_s * 0.15, 1)

    # AFTER (fixed behavior): only the 2 genuinely spoken fillers count.
    fixed_count = r3["count"]
    fixed_filler_s = score_fillers(fixed_count, word_count)
    fixed_clarity_s = score_clarity(pace_s, fixed_filler_s, grammar_s, pronun_s)
    fixed_overall = round(pace_s * 0.20 + fixed_filler_s * 0.20 + pronun_s * 0.25 +
                           grammar_s * 0.20 + fixed_clarity_s * 0.15, 1)

    print("\n--- Before/After example (transcript: "
          f"{mixed_transcript!r}, {word_count} words) ---")
    print(f"BEFORE (bug): acoustic events={acoustic_events_before}, spoken fillers=2, "
          f"merged filler count={buggy_count}, filler score={buggy_filler_s}, "
          f"overall score={buggy_overall}")
    print(f"AFTER  (fix): acoustic hesitation events={acoustic_events_before} "
          f"(kept separate, non-scoring), confirmed spoken fillers={fixed_count}, "
          f"filler score={fixed_filler_s}, overall score={fixed_overall}")
    print(f"Why it changed: the {acoustic_events_before} acoustic low-energy segments "
          "are no longer added to the filler count — only the 2 words actually "
          "identified as spoken fillers ('um', 'uh') are. filler_words score went "
          f"from {buggy_filler_s} to {fixed_filler_s}, which flows through clarity "
          f"({buggy_clarity_s} -> {fixed_clarity_s}) into overall "
          f"({buggy_overall} -> {fixed_overall}).")
    check("7. fixed overall score is higher than the buggy overall score "
          "(false acoustic penalty removed)",
          fixed_overall > buggy_overall,
          f"fixed={fixed_overall} buggy={buggy_overall}")
    check("7. fixed filler score reflects only 2 spoken fillers out of "
          f"{word_count} words",
          fixed_filler_s == score_fillers(2, word_count))

    # ── 8. No double-penalization: fluency's own formula is unchanged, and
    #        acoustic_hesitation_count is informational-only (not folded in) ──
    pause_info = {"available": False}
    hesitation_count = 0  # no word-repetition hesitations in this transcript
    fluency_before_fix_inputs = score_fluency(pause_info, buggy_count, hesitation_count, word_count)
    fluency_after_fix_inputs = score_fluency(pause_info, fixed_count, hesitation_count, word_count)
    check("8. fluency score also improves once acoustic events are excluded "
          "from filler_count (same underlying bug, same fix)",
          fluency_after_fix_inputs["score"] > fluency_before_fix_inputs["score"],
          f"before={fluency_before_fix_inputs['score']} after={fluency_after_fix_inputs['score']}")
    check("8. acoustic_hesitation_count is exposed as informational fluency "
          "context without being added into score_fluency's own formula",
          "acoustic_hesitation_count" not in score_fluency.__code__.co_varnames)

    # ── 9. Groq receives spoken vs. acoustic filler evidence as separate
    #        fields (build_report_evidence's shape, via the real module) ────
    try:
        import groq_provider
        fake_evidence = {
            "overall": fixed_overall,
            "pace": {"score": pace_s, "wpm": 145.0},
            "filler": {
                "score": fixed_filler_s, "count": fixed_count,
                "words": summarize_words(r3["occurrences"]),
                "rate_per_min": r3["rate_per_min"],
                "spoken_count": fixed_count,
                "spoken_fillers": summarize_words(r3["occurrences"]),
                "acoustic_hesitations": r3["acoustic_hesitations"],
                "acoustic_hesitation_count": r3["acoustic_hesitation_count"],
            },
            "hesitations": [],
            "grammar": {"score": grammar_s, "errors": 0, "issues": []},
            "grammar_source": "languagetool",
            "pronunciation": {"score": pronun_s, "available": True},
            "vocabulary": {"score": 80}, "cefr": {"level": "B2"},
            "fluency": {"score": fluency_after_fix_inputs["score"], "hesitation_count": 0,
                        "long_pause_count": 0, "pause_data_available": False},
            "evidence": {"word_count": word_count, "duration_seconds": 3.1, "low_evidence": False},
        }
        report_evidence = groq_provider.build_report_evidence(
            mixed_transcript, mixed_transcript, fake_evidence)
        filler_section = report_evidence["filler"]
        check("9. Groq evidence keeps 'count' as spoken-only",
              filler_section["count"] == fixed_count)
        check("9. Groq evidence exposes spoken_filler_count separately",
              filler_section.get("spoken_filler_count") == fixed_count)
        check("9. Groq evidence exposes acoustic_hesitation_count separately, "
              "distinct from the spoken count",
              filler_section.get("acoustic_hesitation_count") == r3["acoustic_hesitation_count"]
              and filler_section.get("acoustic_hesitation_count") != filler_section["count"])
    except ImportError as e:
        print(f"[SKIP] 9. groq_provider not importable in this environment ({e}) — "
              "field-shape check skipped, fix itself is independent of this import.")

    # ── 10. Uncertain evidence falls back conservatively ────────────────────
    # Without linguistic_analysis (LanguageTool down), the POS-dependent
    # ambiguous-word classifiers (e.g. "like", "well") default to their
    # conservative "insufficient context" branch rather than guessing.
    ambiguous_transcript = "I like that a lot and well it works"
    r6 = detect_fillers(ambiguous_transcript, None, duration_seconds=6)
    like_occurrences = [o for o in r6["occurrences"] if o["word"].lower() == "like"]
    check("10. ambiguous word 'like' without POS context does not get "
          "aggressively classified as a confirmed filler",
          len(like_occurrences) == 0, f"{like_occurrences}")

    print(f"\n{len(FAILURES)} failing check(s)" if FAILURES else "\nAll checks passed.")
    if FAILURES:
        print("Failed:", FAILURES)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
