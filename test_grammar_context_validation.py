#!/usr/bin/env python3
"""
Tests for grammar_context_validator.py — the contextual (Groq) validation
layer that sits between the deterministic grammar candidates
(LanguageTool + grammar_heuristics + grammar_pos_rules) and the grammar
score / learner-facing issue list.

Two paths are exercised, since this sandbox has no network access to
api.groq.com:

  1. HEURISTIC FALLBACK PATH (no GROQ_API_KEY set) — exercises the exact
     code path production hits if Groq is ever unreachable. This is what
     proves the "forty one" / "gonna" examples are fixed even with zero
     network dependency.

  2. MOCKED LLM PATH — monkeypatches GroqGrammarValidator.call() to return
     the JSON a real Groq response would produce for this exact prompt,
     exercising the full parse/merge/scoring-adjustment logic that runs in
     production when GROQ_API_KEY is set.

Run: python3 test_grammar_context_validation.py
"""
import json
import sys

from grammar_context_validator import (
    GroqGrammarValidator,
    classify_candidate_heuristic,
    validate_grammar_context,
    apply_contextual_validation,
)

# NOTE: this test deliberately does NOT `import app` — app.py's import chain
# pulls in Whisper/allosaurus/language_tool_python (heavy binary deps, plus
# a live LanguageTool server / model downloads), none of which this module
# needs or touches. `apply_contextual_validation()` is the exact function
# app.py:resolve_grammar() calls after augment_grammar_issues(), so testing
# it directly exercises the real production code path. score_grammar()'s
# formula (errors/word-count rate -> banded score) is duplicated here
# verbatim, read-only, purely so this test can show a before/after score
# comparison without importing app.py.


def score_grammar(errors: int, words: int) -> float:
    """Verbatim copy of app.py:score_grammar() for this test's before/after
    comparison only — production scoring still lives solely in app.py."""
    rate = errors / max(words, 1)
    if   rate == 0:    return 100
    elif rate < 0.02:  return 90
    elif rate < 0.05:  return 74
    elif rate < 0.08:  return 56
    elif rate < 0.12:  return 36
    else:              return 16

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{('  -- ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(label)


# ─────────────────────────────────────────────────────────────────────────
# Fixture: the exact candidate the user's debug output currently shows.
# ─────────────────────────────────────────────────────────────────────────
FORTY_ONE_SENTENCE = "You can see there is a quick tab showing forty one."
FORTY_ONE_CANDIDATE = {
    "wrong": "forty one",
    "correct": "forty-one",
    "message": "This word is normally spelled with a hyphen.",
    "context": FORTY_ONE_SENTENCE,
    "rule_id": "EN_COMPOUNDS",
    "category": "Compounding",
    "offset": 39,
    "length": 9,
}

GONNA_SENTENCE = "I don't know how this is gonna work."
GONNA_CANDIDATE = {
    "wrong": "gonna",
    "correct": "going to",
    "message": "'gonna' is informal; consider 'going to'.",
    "context": GONNA_SENTENCE,
    "rule_id": "GONNA_INFORMAL",
    "category": "Informal Language",
    "offset": 26,
    "length": 5,
}

# A genuine grammar mistake that MUST still be counted, so we can confirm
# the validator isn't just rubber-stamping everything as "not an error".
REAL_ERROR_SENTENCE = "Yesterday I go to college and I seen my friend."
REAL_ERROR_CANDIDATE = {
    "wrong": "go",
    "correct": "went",
    "message": "Past tense expected after 'Yesterday'.",
    "context": REAL_ERROR_SENTENCE,
    "rule_id": "LEARNER_MISSING_PAST_TENSE",
    "category": "Verb Tense",
    "offset": 10,
    "length": 2,
    "source": "learner_heuristic",
}


# ─────────────────────────────────────────────────────────────────────────
# 1. Offline heuristic classifier — unit-level checks
# ─────────────────────────────────────────────────────────────────────────
print("\n=== 1. classify_candidate_heuristic() unit checks ===")

r = classify_candidate_heuristic(FORTY_ONE_CANDIDATE)
check("'forty one' -> 'forty-one' classified written_only_issue",
      r["classification"] == "written_only_issue", r["classification"])

r = classify_candidate_heuristic(GONNA_CANDIDATE)
check("'gonna' classified style_or_register",
      r["classification"] == "style_or_register", r["classification"])

r = classify_candidate_heuristic(REAL_ERROR_CANDIDATE)
check("'go' (missing past tense) classified true_grammar_error",
      r["classification"] == "true_grammar_error", r["classification"])


# ─────────────────────────────────────────────────────────────────────────
# 2. Full validate_grammar_context() — HEURISTIC FALLBACK PATH
#    (no GROQ_API_KEY -> GroqGrammarValidator.is_available() is False)
# ─────────────────────────────────────────────────────────────────────────
print("\n=== 2. validate_grammar_context() — heuristic fallback path (no GROQ_API_KEY) ===")

no_key_validator = GroqGrammarValidator(api_key="")
result = validate_grammar_context(
    FORTY_ONE_SENTENCE, [FORTY_ONE_CANDIDATE, GONNA_CANDIDATE, REAL_ERROR_CANDIDATE],
    validator=no_key_validator,
)

check("validation_source is heuristic_fallback", result.validation_source == "heuristic_fallback")
check("true_error_count == 1 (only the real error counts)", result.true_error_count == 1,
      str(result.true_error_count))
check("learner_facing_issues has exactly 1 entry", len(result.learner_facing_issues) == 1)
check("learner_facing_issues[0] is the 'go'->'went' issue",
      result.learner_facing_issues and result.learner_facing_issues[0]["wrong"] == "go")
check("context_notes has exactly 2 entries (forty one + gonna)", len(result.context_notes) == 2)

notes_by_wrong = {n["wrong"]: n for n in result.context_notes}
check("'forty one' note classified written_only_issue",
      notes_by_wrong.get("forty one", {}).get("classification") == "written_only_issue")
check("'forty one' note does not count toward score",
      notes_by_wrong.get("forty one", {}).get("counts_toward_grammar_score") is False)
check("'gonna' note classified style_or_register",
      notes_by_wrong.get("gonna", {}).get("classification") == "style_or_register")

print("\n--- context_notes (non-scoring) ---")
print(json.dumps(result.context_notes, indent=2))
print("\n--- learner_facing_issues (scoring) ---")
print(json.dumps(result.learner_facing_issues, indent=2))


# ─────────────────────────────────────────────────────────────────────────
# 3. apply_contextual_validation() end-to-end — does the score actually
#    stop being hurt? This is the exact function app.py:resolve_grammar()
#    calls right after augment_grammar_issues() and right before
#    score_grammar() — i.e. the real production integration point.
# ─────────────────────────────────────────────────────────────────────────
print("\n=== 3. apply_contextual_validation() end-to-end (heuristic-fallback validation) ===")

# Force a validator with no API key, deterministically exercising the
# fallback path regardless of the environment this test happens to run in
# (no network access to api.groq.com is required for this section at all).
no_key_validator_2 = GroqGrammarValidator(api_key="")

transcript = f"{FORTY_ONE_SENTENCE} {GONNA_SENTENCE} {REAL_ERROR_SENTENCE}"
word_count = len(transcript.split())

# Simulates exactly what augment_grammar_issues() hands to
# apply_contextual_validation() in production: a combined LanguageTool +
# heuristic candidate list, plus the pre-validation error count.
ge_before = 3
grammar_issues_before = [FORTY_ONE_CANDIDATE, GONNA_CANDIDATE, REAL_ERROR_CANDIDATE]

ge, grammar_issues, grammar_context = apply_contextual_validation(
    transcript, ge_before, grammar_issues_before, validator=no_key_validator_2)

check("ge (validated error count) == 1", ge == 1, str(ge))
check("grammar_issues has exactly 1 entry (the real error)", len(grammar_issues) == 1)
check("grammar_context.reclassified_away_from_score == 2",
      grammar_context["reclassified_away_from_score"] == 2,
      str(grammar_context["reclassified_away_from_score"]))
check("grammar_context.context_notes has 2 entries", len(grammar_context["context_notes"]) == 2)

grammar_s_before = score_grammar(ge_before, word_count)   # naive/old behavior: all 3 counted
grammar_s_after = score_grammar(ge, word_count)           # new behavior: only the real error counts
check("validated grammar score is not worse than the naive (pre-validation) score",
      grammar_s_after >= grammar_s_before,
      f"before={grammar_s_before} after={grammar_s_after}")
print(f"\nGrammar score WITHOUT contextual validation (3 counted): {grammar_s_before}")
print(f"Grammar score WITH contextual validation ({ge} counted):    {grammar_s_after}")

print("\n--- Full 'grammar' + 'grammar_context' JSON (as /debug/analyze-text would return it) ---")
debug_style_output = {
    "grammar": {"score": round(grammar_s_after, 1), "errors": ge, "issues": grammar_issues},
    "grammar_context": grammar_context,
}
print(json.dumps(debug_style_output, indent=2))


# ─────────────────────────────────────────────────────────────────────────
# 4. MOCKED LLM PATH — simulate what a real Groq response would look like
#    for this exact prompt, to exercise the full parse/merge logic.
# ─────────────────────────────────────────────────────────────────────────
print("\n=== 4. validate_grammar_context() — mocked Groq LLM response ===")


class FakeLLMValidator(GroqGrammarValidator):
    """Stands in for a real Groq call — returns exactly the JSON shape
    build_validation_prompt()/GroqGrammarValidator.call() expects, as if
    the model had reasoned about this specific transcript."""

    def is_available(self):
        return True

    def call(self, transcript, candidates):
        prompt = [{"role": "system", "content": "(mocked)"}]
        judgments = [
            {
                "id": 0,
                "classification": "written_only_issue",
                "learner_explanation": (
                    "You said \"forty one\" perfectly clearly — hyphenating "
                    "compound numbers like \"forty-one\" is only a rule for "
                    "writing them down, so this isn't a spoken grammar mistake."
                ),
                "written_note": "When writing this number, use \"forty-one\" with a hyphen.",
            },
            {
                "id": 1,
                "classification": "style_or_register",
                "learner_explanation": (
                    "\"Gonna\" is just the natural, relaxed way people say "
                    "\"going to\" when speaking casually — it's not a grammar error."
                ),
                "written_note": None,
            },
            {
                "id": 2,
                "classification": "true_grammar_error",
                "learner_explanation": (
                    "You said \"Yesterday I go to college\", but \"yesterday\" "
                    "tells us this already happened, so the verb needs the "
                    "past tense: \"Yesterday I went to college\"."
                ),
                "written_note": None,
            },
        ]
        raw_text = json.dumps({"judgments": judgments})
        parsed = {"judgments": judgments, "_raw_text": raw_text}
        return parsed, prompt, None


mock_result = validate_grammar_context(
    transcript, [FORTY_ONE_CANDIDATE, GONNA_CANDIDATE, REAL_ERROR_CANDIDATE],
    validator=FakeLLMValidator(),
)

check("mocked path: validation_source == llm_groq", mock_result.validation_source == "llm_groq")
check("mocked path: true_error_count == 1", mock_result.true_error_count == 1)
check("mocked path: learner_facing_issues[0] carries LLM's learner_explanation",
      "isn't a spoken grammar mistake" not in mock_result.learner_facing_issues[0].get("learner_explanation", "")
      and "past tense" in mock_result.learner_facing_issues[0].get("learner_explanation", ""))
mock_notes_by_wrong = {n["wrong"]: n for n in mock_result.context_notes}
check("mocked path: 'forty one' note carries LLM's written_note",
      mock_notes_by_wrong.get("forty one", {}).get("written_note") ==
      "When writing this number, use \"forty-one\" with a hyphen.")

print("\n--- Mocked-LLM-path 'grammar' + 'grammar_context' JSON ---")
mocked_debug_style_output = {
    "grammar": {
        "score": round(score_grammar(mock_result.true_error_count, word_count), 1),
        "errors": mock_result.true_error_count,
        "issues": mock_result.learner_facing_issues,
    },
    "grammar_context": {
        "candidates_evaluated": 3,
        "reclassified_away_from_score": 3 - mock_result.true_error_count,
        "validation_source": mock_result.validation_source,
        "model": mock_result.model,
        "detail": mock_result.detail,
        "context_notes": mock_result.context_notes,
        "debug_trail": mock_result.validated_issues,
    },
}
print(json.dumps(mocked_debug_style_output, indent=2))


# ─────────────────────────────────────────────────────────────────────────
# 4. Spec section 11/12 acceptance case — immediate word repetitions
#    ("check check", "the the") must NOT automatically become grammar
#    errors, on the heuristic-fallback path (no GROQ_API_KEY, exactly the
#    path production hits if Groq is ever unreachable).
print("\n=== 4. Repeated-word acceptance case (heuristic fallback) ===")

CHECK_CHECK_CANDIDATE = {
    "wrong": "check check", "correct": "check",
    "message": "Possible typo: you repeated a word.",
    "context": "check check if the what to say the the transcription is working well.",
    "rule_id": "ENGLISH_WORD_REPEAT_RULE", "category": "Miscellaneous",
}
THE_WHAT_CANDIDATE = {
    "wrong": "the what", "correct": "what",
    "message": "Did you mean 'what'?",
    "context": "if the what to say the the transcription is working well.",
    "rule_id": "THE_HOW", "category": "Grammar",
}
THE_THE_CANDIDATE = {
    "wrong": "the the", "correct": "the",
    "message": "Possible typo: you repeated a word.",
    "context": "the what to say the the transcription is working well.",
    "rule_id": "ENGLISH_WORD_REPEAT_RULE", "category": "Miscellaneous",
}

check("'check check' classified spoken_usage_issue",
      classify_candidate_heuristic(CHECK_CHECK_CANDIDATE)["classification"] == "spoken_usage_issue")
check("'the the' classified spoken_usage_issue",
      classify_candidate_heuristic(THE_THE_CANDIDATE)["classification"] == "spoken_usage_issue")
check("'the what' (genuine error) still classified true_grammar_error",
      classify_candidate_heuristic(THE_WHAT_CANDIDATE)["classification"] == "true_grammar_error")

acceptance_transcript = (
    "check check if the what to say the the transcription is working well "
    "and if I want to check that the contextual grammar is also working perfectly."
)
acc_ge, acc_issues, acc_ctx = apply_contextual_validation(
    acceptance_transcript, 3,
    [CHECK_CHECK_CANDIDATE, THE_WHAT_CANDIDATE, THE_THE_CANDIDATE],
)
check("acceptance case: validated grammar errors == 1", acc_ge == 1, f"got {acc_ge}")
check("acceptance case: only 'the what' remains scoring",
      [i["wrong"] for i in acc_issues] == ["the what"], f"got {[i['wrong'] for i in acc_issues]}")
check("acceptance case: 2 candidates reclassified away",
      acc_ctx["reclassified_away_from_score"] == 2, f"got {acc_ctx['reclassified_away_from_score']}")


# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
