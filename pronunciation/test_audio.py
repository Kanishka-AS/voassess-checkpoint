"""
test_audio.py
-------------
STANDALONE, arbitrary-audio test harness for AllosaurusProvider.

This is deliberately separate from evaluation.py. evaluation.py's --real
mode is hardcoded to the three MediaRepeat clips (a different activity's
recordings, with sentences copied from app.py's ASSESSMENT_MANIFEST) --
it must not be used for pronunciation-test audio. This script accepts any
WAV/audio file + any expected text, so a real pronunciation-test
recording can be evaluated without touching MediaRepeat, evaluation.py,
or app.py at all.

It does NOT modify, replace, or import anything from the existing
production pronunciation implementation used by app.py (app.py does not
import this `pronunciation` package at all -- verified separately). It
only exercises the experimental AllosaurusProvider built in this package.

PIPELINE (unchanged from allosaurus_provider.py -- this script does not
reimplement any of it, only calls it):

    audio + text -> AllosaurusProvider.analyze()
        -> g2p_sentence(text)                      [g2p.py]
        -> AllosaurusPhoneNormalizer.normalize_sequence(...)   [allosaurus_normalizer.py]
        -> acoustic evidence per normalized phone   [phoneme_analyzer.py]

SCIENTIFIC GUARDRAIL: this script prints and saves raw acoustic evidence
(posteriors, margins, decode membership) and simple descriptive counts
(e.g. "N phonemes with max_posterior > 0.5"). It does NOT compute or
print any single overall "pronunciation score" derived from these
numbers -- schemas.py's own docstring prohibits silently rescaling
posteriors into a calibrated score, and no such calibration exists yet.
The "summary" section below is a description of the evidence distribution,
not a grade.

Usage:
    python3 -m pronunciation.test_audio \\
        --audio pronunciation_test/test.wav \\
        --text "I want to improve my English pronunciation." \\
        --out pronunciation_test_result.json
"""

import argparse
import json
import os
import sys

from .allosaurus_provider import AllosaurusProvider
from .schemas import PronunciationResult


# Evidence-strength thresholds used ONLY for the descriptive summary
# counts below (Task 8: "strong evidence" / "weak evidence" /
# "phonemes with competing predictions"). These are arbitrary, clearly-
# labeled bucket boundaries for descriptive reporting -- they are NOT a
# validated pronunciation-correctness threshold, and this script never
# claims otherwise.
STRONG_EVIDENCE_THRESHOLD = 0.5
WEAK_EVIDENCE_THRESHOLD = 0.1


def build_provider() -> AllosaurusProvider:
    return AllosaurusProvider()


def run_test(audio_path: str, text: str) -> PronunciationResult:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            f"This script does not fall back to MediaRepeat or any bundled "
            f"audio -- point --audio at your own recording."
        )
    provider = build_provider()
    print(f"[info] provider resource_info: {provider.resource_info()}")
    result = provider.analyze(audio_path, text)
    return result


def summarize(result: PronunciationResult) -> dict:
    """Descriptive counts over the evidence -- see module docstring's
    SCIENTIFIC GUARDRAIL. No score is computed here."""
    evs = result.phoneme_evidence
    n = len(evs)
    posteriors = [e.max_posterior for e in evs if e.max_posterior is not None]

    strong = [e for e in evs if e.max_posterior is not None and e.max_posterior >= STRONG_EVIDENCE_THRESHOLD]
    weak = [e for e in evs if e.max_posterior is not None and e.max_posterior < WEAK_EVIDENCE_THRESHOLD]
    competing = [
        e for e in evs
        if e.competing_symbol is not None
        and e.competing_symbol != e.expected
        and e.competing_posterior is not None
        and e.evidence_margin is not None
        and e.evidence_margin < 0.3
    ]
    not_in_greedy = [e for e in evs if e.appears_in_greedy_decode is False]

    return {
        "num_expected_phonemes": n,
        "num_normalized_phonemes": len(result.normalized_phonemes),
        "num_strong_evidence (max_posterior >= %.2f)" % STRONG_EVIDENCE_THRESHOLD: len(strong),
        "num_weak_evidence (max_posterior < %.2f)" % WEAK_EVIDENCE_THRESHOLD: len(weak),
        "num_with_close_competing_phone (margin < 0.3)": len(competing),
        "num_not_in_greedy_decode": len(not_in_greedy),
        "mean_max_posterior": round(sum(posteriors) / len(posteriors), 4) if posteriors else None,
        "max_max_posterior": round(max(posteriors), 4) if posteriors else None,
        "min_max_posterior": round(min(posteriors), 4) if posteriors else None,
        "note": (
            "These are descriptive counts over raw acoustic evidence, "
            "NOT a pronunciation score. See schemas.py and this module's "
            "docstring."
        ),
    }


def print_report(audio_path: str, text: str, result: PronunciationResult, summary: dict) -> None:
    print()
    print("Pronunciation Test")
    print("-" * 60)
    print(f"Audio: {audio_path}")
    print(f"Text:  {text}")
    print()
    print("Expected phonemes (generic, from G2P):")
    print(" ", " ".join(result.expected_phonemes) if result.expected_phonemes else "(none -- G2P failed, see warnings)")
    print()
    print("Normalized phonemes (after AllosaurusPhoneNormalizer):")
    print(" ", " ".join(result.normalized_phonemes) if result.normalized_phonemes else "(none)")
    print()
    print("Greedy decode (model's own whole-utterance output):")
    print(" ", " ".join(result.greedy_decoded_phonemes) if result.greedy_decoded_phonemes else "(empty)")
    print()

    print("Phoneme evidence:")
    header = f"{'expected':<10}{'normalized':<12}{'posterior':<11}{'top1':<8}{'competing':<11}{'margin':<9}{'in_decode':<10}{'rule'}"
    print(header)
    print("-" * len(header))
    for ev in result.phoneme_evidence:
        expected_generic = "+".join(ev.generic_phones) if ev.generic_phones else ev.expected
        post = f"{ev.max_posterior:.4f}" if ev.max_posterior is not None else "n/a"
        top1 = ev.top1_phone if ev.top1_phone is not None else "n/a"
        comp = ev.competing_symbol if ev.competing_symbol is not None else "n/a"
        margin = f"{ev.evidence_margin:.4f}" if ev.evidence_margin is not None else "n/a"
        in_decode = "yes" if ev.appears_in_greedy_decode else "no"
        rule = ev.normalization_rule if ev.normalization_rule else "-"
        print(f"{expected_generic:<10}{ev.expected:<12}{post:<11}{top1:<8}{comp:<11}{margin:<9}{in_decode:<10}{rule}")
    print()

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
        print()

    print("Summary (descriptive only -- not a pronunciation score):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone AllosaurusProvider test on an arbitrary audio file + expected text. "
                     "Does NOT use MediaRepeat and does NOT touch app.py or the existing pronunciation implementation."
    )
    parser.add_argument("--audio", required=True, help="Path to your own WAV/audio file (NOT MediaRepeat)")
    parser.add_argument("--text", required=True, help="Expected sentence/text spoken in the audio")
    parser.add_argument("--out", default="pronunciation_test_result.json", help="Path to write the JSON result")
    args = parser.parse_args()

    try:
        result = run_test(args.audio, args.text)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)

    summary = summarize(result)
    print_report(args.audio, args.text, result, summary)

    out_obj = result.to_dict()
    out_obj["_summary"] = summary
    out_obj["_meta"] = {
        "source": "standalone test_audio.py (user-supplied audio, not MediaRepeat)",
        "note": "Independent of evaluation.py and app.py. Uses the same AllosaurusProvider "
                "implementation, unmodified.",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"[info] wrote full result to {args.out}")


if __name__ == "__main__":
    main()
