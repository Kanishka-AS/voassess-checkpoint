"""
Unit tests for grammar_heuristics.py.

Pure-function tests — no network, no app.py import, no Whisper/LanguageTool
stubbing needed. Covers:
  * The exact problematic transcript from the bug report.
  * Each of the five representative learner-error categories named in the
    requirements.
  * A battery of grammatically correct sentences that must NOT be flagged
    (the false-positive guard — a heuristic layer that starts flagging
    normal English is worse than the bug it fixes).
  * augment_grammar_issues()'s de-duplication against an existing
    LanguageTool match for the same mistake.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar_heuristics import detect_learner_errors, augment_grammar_issues


# ---- The exact bug-report transcript ---------------------------------------

def test_problematic_transcript_is_no_longer_silently_clean():
    transcript = ("Hi, this is a person from the south of India. "
                  "I wear eating, study and apple.")
    issues = detect_learner_errors(transcript)
    assert len(issues) >= 1
    wrongs = {i["wrong"].lower() for i in issues}
    assert any("wear" in w for w in wrongs)


def test_problematic_transcript_end_to_end_through_augment():
    """Simulates LanguageTool reporting 0 errors (the actual bug) and checks
    the augmented result is no longer '0 errors' / implicitly 100 score."""
    transcript = ("Hi, this is a person from the south of India. "
                  "I wear eating, study and apple.")
    combined_errors, combined_issues, added = augment_grammar_issues(transcript, 0, [])
    assert combined_errors > 0
    assert added > 0
    assert len(combined_issues) > 0
    assert combined_issues[0]["source"] == "learner_heuristic"


# ---- The five representative categories from the requirements -------------

def test_missing_be_auxiliary_wear_eating():
    issues = detect_learner_errors("I wear eating.")
    assert any(i["rule_id"] == "LEARNER_VERB_STACKING" for i in issues)
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_VERB_STACKING")
    assert hit["correct"] == "I am"


def test_subject_verb_agreement_she_go():
    issues = detect_learner_errors("She go to school.")
    assert any(i["wrong"] == "go" and i["correct"] == "goes" for i in issues)


def test_past_tense_required_yesterday_i_go():
    issues = detect_learner_errors("Yesterday I go to college.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_PAST_TENSE_REQUIRED")
    assert hit["wrong"] == "go"
    assert hit["correct"] == "went"


def test_do_does_negation_he_dont():
    issues = detect_learner_errors("He don't like coffee.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_DO_AUX_AGREEMENT")
    assert hit["wrong"].lower() == "don't"
    assert hit["correct"] == "doesn't"


def test_missing_be_auxiliary_direct_gerund():
    issues = detect_learner_errors("I studying computer science.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_MISSING_BE_AUX")
    assert hit["correct"] == "I am studying"


# ---- "be" past-tense agreement (Pass 2b) -----------------------------------

def test_they_was_is_flagged():
    issues = detect_learner_errors("They was very surprised.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_BE_PAST_AGREEMENT")
    assert hit["wrong"] == "was"
    assert hit["correct"] == "were"


def test_he_were_is_flagged():
    issues = detect_learner_errors("He were tired yesterday.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_BE_PAST_AGREEMENT")
    assert hit["wrong"] == "were"
    assert hit["correct"] == "was"


def test_they_were_is_not_flagged():
    assert detect_learner_errors("They were surprised.") == []


def test_subjunctive_if_he_were_is_not_flagged():
    assert detect_learner_errors("If he were rich, he would travel.") == []


def test_i_was_is_not_flagged():
    assert detect_learner_errors("Yesterday I was very tired.") == []


# ---- Additional representative learner errors ------------------------------

def test_reverse_agreement_they_goes():
    issues = detect_learner_errors("They goes to the market.")
    assert any(i["wrong"] == "goes" and i["correct"] == "go" for i in issues)


def test_reverse_do_negation_i_doesnt():
    issues = detect_learner_errors("I doesn't want that.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_DO_AUX_AGREEMENT")
    assert hit["correct"] == "don't"


def test_past_tense_with_ago():
    issues = detect_learner_errors("I visit my grandmother two years ago.")
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_PAST_TENSE_REQUIRED")
    assert hit["wrong"] == "visit"
    assert hit["correct"] == "visited"


def test_verb_stacking_generalizes_beyond_the_example_verb():
    """Same pattern as 'wear eating' but with a different, unrelated verb —
    proves this isn't a hard-coded match on the bug-report sentence."""
    issues = detect_learner_errors("I come working every day.")
    assert any(i["rule_id"] == "LEARNER_VERB_STACKING" for i in issues)


# ---- False-positive guard: normal, correct English must stay clean --------

CORRECT_SENTENCES = [
    "I am eating breakfast.",
    "She goes to school every day.",
    "He doesn't like coffee.",
    "I went to college yesterday.",
    "I am studying computer science.",
    "I enjoy reading books.",
    "She keeps talking about it.",
    "I go swimming every weekend.",
    "It is raining this morning.",
    "This is a very interesting building.",
    "I have a strange feeling about this.",
    "They started working on the project last month.",
    "We are planning a trip next week.",
    "He suggested reading that book.",
    "My brother works at a bank.",
    "The weather is nice today.",
    "I can speak three languages.",
    "I would like to visit Japan.",
    "Yesterday I was very tired.",
    "I have finished my homework.",
    "We enjoy hiking in the mountains.",
    "The children are playing in the park.",
]


def test_correct_sentences_are_not_flagged():
    for sentence in CORRECT_SENTENCES:
        issues = detect_learner_errors(sentence)
        assert issues == [], f"False positive on correct sentence: {sentence!r} -> {issues}"


def test_empty_and_blank_text_returns_no_issues():
    assert detect_learner_errors("") == []
    assert detect_learner_errors("   ") == []


# ---- augment_grammar_issues() de-duplication -------------------------------

def test_augment_deduplicates_against_existing_languagetool_match():
    """The exact audit scenario: LanguageTool already caught 'go' -> 'goes'
    for 'She go ... yesterday' as a single match. Our heuristics would
    independently flag the same word (for either agreement or, since
    'yesterday' is present, past tense) — the merge must not double-count
    the same real mistake."""
    transcript = "She go to the store yesterday and me and him was very tired."
    existing_issues = [{
        "wrong": "go", "correct": "goes",
        "message": "The verb form is not correct for a third-person subject.",
        "context": "She go to the store yesterday...",
    }]
    combined_errors, combined_issues, added = augment_grammar_issues(
        transcript, existing_errors=1, existing_issues=existing_issues)
    assert combined_errors == 1
    assert added == 0
    assert combined_issues == existing_issues


def test_augment_adds_new_issue_not_covered_by_existing():
    transcript = "He don't like coffee."
    combined_errors, combined_issues, added = augment_grammar_issues(
        transcript, existing_errors=0, existing_issues=[])
    assert combined_errors == 1
    assert added == 1
    assert combined_issues[0]["correct"] == "doesn't"


def test_augment_caps_combined_issues_but_not_error_count():
    transcript = " ".join([f"He don't like item{i}." for i in range(20)])
    combined_errors, combined_issues, added = augment_grammar_issues(
        transcript, existing_errors=0, existing_issues=[], max_issues=8)
    assert combined_errors == added == 20
    assert len(combined_issues) == 8


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))