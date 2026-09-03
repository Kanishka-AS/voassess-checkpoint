#!/usr/bin/env python3
"""
Regression tests for groq_provider.py's Teacher Report consistency
guarantees — specifically the merge_report() locks added to close the
"zero validated grammar errors but the model hallucinated content anyway"
loophole.

These tests exercise merge_report()/build_report_evidence() directly and
never call the network (no GROQ_API_KEY needed, no httpx call made) — they
test the deterministic Python-side enforcement, which is what guarantees
consistency regardless of what a real Groq response contains.

Run: python3 test_teacher_report_consistency.py
"""
import json
import sys

from groq_provider import build_report_evidence, merge_report

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{('  -- ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(label)


# ── Shared fixture: score_free_speech()-shaped evidence dict ────────────────
# Mirrors exactly the shape app.py's score_free_speech() returns (the
# `evidence` argument generate_teacher_report()/build_report_evidence()
# receive), for the acceptance-test transcript from this conversation:
# "wanted to know, I wanted to know that if any of the check checking is
# done well. If if you know, the the transcription is done well."
# Groq's contextual validation classified all 3 raw candidates as NOT
# true_grammar_error (check checking -> spoken_usage_issue, if ->
# not_an_error, the the -> spoken_usage_issue), so the validated
# grammar.issues list app.py:resolve_grammar() would have produced is
# EMPTY and grammar.errors is 0 — exactly what build_report_evidence() is
# given here (nothing in this test recomputes that; it's asserting the
# report layer honors it correctly).

TRANSCRIPT = ("wanted to know, I wanted to know that if any of the check checking is "
              "done well. If if you know, the the transcription is done well.")

ZERO_ERROR_EVIDENCE = {
    "evidence": {"word_count": 24, "duration_seconds": 9.0, "low_evidence": True},
    "overall": 82.0,
    "pace": {"score": 80.0, "wpm": 130.0},
    "filler": {"score": 100.0, "count": 0, "words": [], "rate_per_min": 0.0},
    "hesitations": [],
    "grammar": {"score": 100.0, "errors": 0, "issues": []},
    "grammar_source": "languagetool_http",
    # The three raw candidates, all reclassified AWAY from scoring by
    # grammar_context_validator.apply_contextual_validation() — this is
    # exactly grammar_context["context_notes"] as app.py sets it.
    "grammar_context": {
        "candidates_evaluated": 3,
        "reclassified_away_from_score": 3,
        "validation_source": "llm_groq",
        "context_notes": [
            {"wrong": "check checking", "correct": "checking", "classification": "spoken_usage_issue"},
            {"wrong": "if", "correct": "if", "classification": "not_an_error"},
            {"wrong": "the the", "correct": "the", "classification": "spoken_usage_issue"},
        ],
    },
    "pronunciation": {"score": 88.0, "available": True, "issues": [], "provider": "allosaurus_g2p"},
    "clarity": {"score": 85.0},
    "fluency": {"score": 90.0, "pause_data_available": True, "hesitation_count": 0, "long_pause_count": 0},
    "vocabulary": {"score": 70.0, "unique_words": 15, "total_words": 24, "advanced_ratio": 5.0},
    "cefr": {"score": 55.0, "level": "B1"},
    "archetype": {"archetype": "The Communicator"},
}

REAL_ERROR_EVIDENCE = {
    "evidence": {"word_count": 40, "duration_seconds": 15.0, "low_evidence": False},
    "overall": 78.0,
    "pace": {"score": 80.0, "wpm": 130.0},
    "filler": {"score": 90.0, "count": 1, "words": ["um"], "rate_per_min": 4.0},
    "hesitations": [],
    "grammar": {
        "score": 74.0, "errors": 1,
        "issues": [{
            "wrong": "go", "correct": "went",
            "message": "Past tense expected after 'Yesterday'.",
            "context": "Yesterday I go to the store and bought some milk.",
            "rule_id": "LEARNER_MISSING_PAST_TENSE", "category": "Verb Tense",
        }],
    },
    "grammar_source": "languagetool_http",
    "grammar_context": {
        "candidates_evaluated": 1, "reclassified_away_from_score": 0,
        "validation_source": "llm_groq", "context_notes": [],
    },
    "pronunciation": {"score": 88.0, "available": True, "issues": [], "provider": "allosaurus_g2p"},
    "clarity": {"score": 83.0},
    "fluency": {"score": 90.0, "pause_data_available": True, "hesitation_count": 0, "long_pause_count": 0},
    "vocabulary": {"score": 70.0, "unique_words": 25, "total_words": 40, "advanced_ratio": 5.0},
    "cefr": {"score": 58.0, "level": "B1"},
    "archetype": {"archetype": "The Communicator"},
}


# ─────────────────────────────────────────────────────────────────────────
# A. 3 raw candidates -> 0 validated grammar errors -> Grammar Errors == 0
#    (this is app.py's own field, sourced straight from score_free_speech();
#    asserting the fixture models it correctly, then that build_report_evidence
#    passes it through unchanged into what Groq/merge_report sees).
print("=== A. 0 validated grammar errors is what the report layer receives ===")
evidence_a = build_report_evidence(TRANSCRIPT, None, ZERO_ERROR_EVIDENCE)
check("grammar.error_count == 0", evidence_a["grammar"]["error_count"] == 0)
check("grammar.issues == []", evidence_a["grammar"]["issues"] == [])
check("grammar.non_scoring_notes carries the 3 reclassified candidates",
      len(evidence_a["grammar"]["non_scoring_notes"]) == 3)
check("locked_rubric.accuracy.score == 100 (matches score_grammar(0 errors))",
      evidence_a["locked_rubric"]["accuracy"]["score"] == 100.0)
check("locked_rubric.accuracy.band == High",
      evidence_a["locked_rubric"]["accuracy"]["band"] == "High")


# ─────────────────────────────────────────────────────────────────────────
# B. Genuine grammar error remains validated and affects Grammar score
print("\n=== B. Genuine grammar error stays validated, affects Grammar score ===")
evidence_b = build_report_evidence(TRANSCRIPT, None, REAL_ERROR_EVIDENCE)
check("grammar.error_count == 1", evidence_b["grammar"]["error_count"] == 1)
check("grammar.issues has exactly the 1 real error",
      len(evidence_b["grammar"]["issues"]) == 1 and evidence_b["grammar"]["issues"][0]["wrong"] == "go")
check("locked_rubric.accuracy.score == 74 (matches evidence['grammar']['score'])",
      evidence_b["locked_rubric"]["accuracy"]["score"] == 74.0)
check("locked_rubric.accuracy.band == Medium",
      evidence_b["locked_rubric"]["accuracy"]["band"] == "Medium")


# ─────────────────────────────────────────────────────────────────────────
# C/D. Teacher Report uses exactly the validated grammar issues, and does
# NOT contain hallucinated grammar/vocabulary content when there is no real
# grammar error — even when the (simulated) model response disagrees with
# the evidence. This is the actual bug this change fixes: previously
# merge_report() only intervened when real_issue_count > 0; a
# well-formed-but-wrong model response for the zero-error case passed
# through untouched.
print("\n=== C/D. merge_report() enforces consistency even against a hallucinating model ===")

HALLUCINATING_MODEL_REPORT = {
    "overview": {
        "overall_assessment": "Good effort overall.",
        # Exactly the kind of text the bug report described:
        "grammar_accuracy_summary": "The system recorded a single grammatical error count in this response.",
        "advanced_grammar_constructions_detected": "The student used a relative clause correctly.",
        "complex_sentence_usage": "Sentences were reasonably complex.",
        "strong_vocabulary_observations": "Some good word choices.",
        "strong_language_use_observations": "Clear overall.",
    },
    "growth_areas": {
        "repeated_overused_words": "The word 'know' was used too often.",
        "fillers": {"summary": "A few filler words were used.", "why_it_matters": "They break flow.",
                    "how_to_reduce": ["Pause instead."]},
        "linking_word_suggestions": "Try using 'however' to connect your ideas.",
        # Hallucinated grammar_breakdown despite real_issue_count == 0:
        "grammar_breakdown": [{
            "you_said": "if any of the check checking is done well",
            "what_went_wrong": "Repeated word.",
            "why_its_wrong": "Grammar rule violation.",
            "correct_version": "if any of the checking is done well",
            "how_to_avoid_next_time": "Speak more slowly.",
        }],
        "vocabulary_improvements": "Try 'wanted' -> 'desired' for a more advanced tone.",
        "fluency_pacing_improvements": "Good pace.",
        "other_weaknesses": "None noted.",
    },
    "vocabulary": {
        "active_vocabulary_note": "Decent range.",
        "useful_higher_level_words_used": [],
        "suggestions_for_improving_vocabulary": "Use more advanced words.",
    },
    "repetitions": [{"word_or_phrase": "know", "frequency": 2, "better_alternatives": ["understand"]}],
    "advanced_grammar_used": [],  # empty list, but overview text above claims one exists
    "performance_summary_elaboration": {
        "accuracy_extra": "Solid grammar overall.",
        "fluency_extra": "Good flow.",
        "use_of_english_extra": "Clear pronunciation.",
    },
}

report_evidence_zero = build_report_evidence(TRANSCRIPT, None, ZERO_ERROR_EVIDENCE)
merged = merge_report(HALLUCINATING_MODEL_REPORT, report_evidence_zero)

check("grammar_breakdown is forced empty despite model hallucinating one",
      merged["growth_areas"]["grammar_breakdown"] == [])
check("grammar_accuracy_summary does not contain 'system' (banned internal language)",
      "system" not in merged["overview"]["grammar_accuracy_summary"].lower())
check("grammar_accuracy_summary does not claim an error occurred (it may say errors "
      "weren't counted, which is a correct negation, not a false claim)",
      "grammatical error count" not in merged["overview"]["grammar_accuracy_summary"].lower()
      and "you made" not in merged["overview"]["grammar_accuracy_summary"].lower())
check("grammar_accuracy_summary references the real reclassified candidates",
      "check checking" in merged["overview"]["grammar_accuracy_summary"]
      and "the the" in merged["overview"]["grammar_accuracy_summary"])
check("performance_summary.accuracy.score == 100 (locked, matches validated 0 errors)",
      merged["performance_summary"]["accuracy"]["score"] == 100.0)
check("performance_summary.accuracy.teacher_explanation uses the locked rubric sentence",
      merged["performance_summary"]["accuracy"]["teacher_explanation"].startswith(
          "You made very few grammatical errors."))
check("advanced_grammar_constructions_detected forced to 'None clearly evidenced' "
      "(advanced_grammar_used list was empty)",
      merged["overview"]["advanced_grammar_constructions_detected"] == "None clearly evidenced in this sample.")
check("advanced_grammar_used stays the empty list the model gave (not invented content)",
      merged["advanced_grammar_used"] == [])

print("\n--- Merged overview.grammar_accuracy_summary ---")
print(json.dumps(merged["overview"]["grammar_accuracy_summary"], indent=2))
print("\n--- Merged growth_areas.grammar_breakdown ---")
print(json.dumps(merged["growth_areas"]["grammar_breakdown"], indent=2))


# ─────────────────────────────────────────────────────────────────────────
# D (continued). No hallucinated content when the model DID behave — the
# locks must be no-ops (not silently discard a well-behaved model's real
# work) when the model already agreed with the evidence.
print("\n=== D (cont). Locks are no-ops for a well-behaved zero-error response ===")

WELL_BEHAVED_MODEL_REPORT = {
    "overview": {
        "overall_assessment": "Solid, understandable response.",
        "grammar_accuracy_summary": (
            "Your grammar was understandable in this response. Moments like 'check checking' "
            "and 'the the' sound like spoken hesitations rather than grammar mistakes."
        ),
        "advanced_grammar_constructions_detected": "None clearly evidenced in this sample.",
        "complex_sentence_usage": "Not clearly measurable from this sample.",
        "strong_vocabulary_observations": "Reasonable word choice for a short response.",
        "strong_language_use_observations": "Ideas came through clearly.",
    },
    "growth_areas": {
        "repeated_overused_words": "No significant repetition detected.",
        "fillers": {"summary": "No significant filler use detected.", "why_it_matters": "", "how_to_reduce": []},
        "linking_word_suggestions": "",
        "grammar_breakdown": [],
        "vocabulary_improvements": "No specific vocabulary changes needed for this response.",
        "fluency_pacing_improvements": "Pace was comfortable.",
        "other_weaknesses": "None noted.",
    },
    "vocabulary": {"active_vocabulary_note": "Fine for a short sample.",
                   "useful_higher_level_words_used": [],
                   "suggestions_for_improving_vocabulary": "None needed."},
    "repetitions": [],
    "advanced_grammar_used": [],
    "performance_summary_elaboration": {"accuracy_extra": "", "fluency_extra": "", "use_of_english_extra": ""},
}

merged_good = merge_report(WELL_BEHAVED_MODEL_REPORT, report_evidence_zero)
check("well-behaved model's grammar_accuracy_summary is still the grounded fallback "
      "(deterministic lock, same content either way)",
      "check checking" in merged_good["overview"]["grammar_accuracy_summary"])
check("well-behaved model's grammar_breakdown stays empty", merged_good["growth_areas"]["grammar_breakdown"] == [])
check("vocabulary_improvements preserved (nothing to lock here, model already conservative)",
      merged_good["growth_areas"]["vocabulary_improvements"] == "No specific vocabulary changes needed for this response.")


# ─────────────────────────────────────────────────────────────────────────
# Sanity: real-error case is NOT wiped out by the zero-error lock path.
print("\n=== Sanity: real-error case still produces a full grammar_breakdown ===")
GOOD_REAL_ERROR_MODEL_REPORT = {
    "overview": {
        "overall_assessment": "Good attempt with one tense slip.",
        "grammar_accuracy_summary": "You made one grammar mistake involving verb tense.",
        "advanced_grammar_constructions_detected": "None clearly evidenced in this sample.",
        "complex_sentence_usage": "Not clearly measurable from this sample.",
        "strong_vocabulary_observations": "Reasonable range.",
        "strong_language_use_observations": "Clear overall.",
    },
    "growth_areas": {
        "repeated_overused_words": "No significant repetition detected.",
        "fillers": {"summary": "1 filler instance was detected in this sample.",
                    "why_it_matters": "It slightly interrupts flow.",
                    "how_to_reduce": ["Pause instead of saying 'um'."]},
        "linking_word_suggestions": "",
        "grammar_breakdown": [{
            "you_said": "Yesterday I go to the store and bought some milk.",
            "what_went_wrong": "'go' should be 'went' — this happened yesterday, so it needs the past tense.",
            "why_its_wrong": "'Yesterday' signals the past, so the verb must be in the past tense.",
            "correct_version": "Yesterday I went to the store and bought some milk.",
            "how_to_avoid_next_time": "When a sentence has a past-time word like 'yesterday', check the verb is in the past tense.",
        }],
        "vocabulary_improvements": "No specific vocabulary changes needed for this response.",
        "fluency_pacing_improvements": "Pace was comfortable.",
        "other_weaknesses": "None noted.",
    },
    "vocabulary": {"active_vocabulary_note": "Fine.", "useful_higher_level_words_used": [],
                   "suggestions_for_improving_vocabulary": "None needed."},
    "repetitions": [],
    "advanced_grammar_used": [],
    "performance_summary_elaboration": {"accuracy_extra": "", "fluency_extra": "", "use_of_english_extra": ""},
}
report_evidence_real = build_report_evidence(TRANSCRIPT, None, REAL_ERROR_EVIDENCE)
merged_real = merge_report(GOOD_REAL_ERROR_MODEL_REPORT, report_evidence_real)
check("real-error grammar_breakdown has exactly 1 entry", len(merged_real["growth_areas"]["grammar_breakdown"]) == 1)
check("real-error grammar_accuracy_summary is the model's own grounded text (not the zero-error fallback)",
      merged_real["overview"]["grammar_accuracy_summary"] == "You made one grammar mistake involving verb tense.")
check("real-error performance_summary.accuracy.score == 74",
      merged_real["performance_summary"]["accuracy"]["score"] == 74.0)


# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
