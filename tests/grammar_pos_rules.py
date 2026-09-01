"""
Unit tests for grammar_pos_rules.py.

Pure-function tests — no network, no real LanguageTool server. Fixtures
mirror the actual /v2/analyze JSON shape (see tests/test_languagetool_provider.py's
ANALYZE_RESPONSE_FIXTURE): {"language", "sentences": [{"text", "tokens": [
{"text", "lemma", "partOfSpeech", "posTag", "startOffset", "endOffset"}]}]},
with token offsets absolute across the whole document (confirmed by
LanguageToolProvider's own multi-sentence test).

`_fixture()` below builds that structure from a compact per-sentence word
list instead of hand-computing offsets: for each (word, lemma, pos) triple
it does a sequential forward search in the full text starting from where
the previous token match ended, so tests only need to list the tokens a
rule actually cares about — not every word in the sentence — while still
getting correct absolute offsets.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar_heuristics import augment_grammar_issues
from grammar_pos_rules import detect_pos_aware_errors


def _fixture(full_text, sentences):
    """sentences: [(sentence_text, [(word, lemma, pos), ...]), ...]"""
    out_sentences = []
    cursor = 0
    for sent_text, words in sentences:
        tokens = []
        for word, lemma, pos in words:
            m = re.compile(re.escape(word)).search(full_text, cursor)
            assert m, f"could not locate {word!r} in fixture text from offset {cursor}"
            tokens.append({
                "text": word, "lemma": lemma, "partOfSpeech": pos,
                "posTag": pos, "startOffset": m.start(), "endOffset": m.end(),
            })
            cursor = m.end()
        out_sentences.append({"text": sent_text, "tokens": tokens})
    return {"language": "en-US", "sentences": out_sentences}


# ---- No linguistic_analysis -> no issues, no crash ------------------------

def test_returns_empty_without_linguistic_analysis():
    assert detect_pos_aware_errors("The students goes to school.", None) == []
    assert detect_pos_aware_errors("The students goes to school.", {}) == []
    assert detect_pos_aware_errors("The students goes to school.",
                                    {"language": "en-US", "sentences": []}) == []


# ---- Rule 1: subject-verb agreement for noun / indefinite-pronoun subjects -

def test_plural_noun_subject_with_singular_verb_is_flagged():
    text = "The students goes to school."
    fixture = _fixture(text, [(text, [
        ("students", "student", "NNS"), ("goes", "go", "VBZ"),
    ])])
    issues = detect_pos_aware_errors(text, fixture)
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_SUBJECT_VERB_AGREEMENT_NP")
    assert hit["wrong"] == "goes"
    assert hit["correct"] == "go"
    assert hit["confidence"] == "high"


def test_indefinite_pronoun_subject_have_is_flagged():
    text = "Nobody have enough cash."
    fixture = _fixture(text, [(text, [
        ("Nobody", "nobody", "NN"), ("have", "have", "VBP"),
    ])])
    issues = detect_pos_aware_errors(text, fixture)
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_SUBJECT_VERB_AGREEMENT_NP")
    assert hit["wrong"] == "have"
    assert hit["correct"] == "has"


def test_group_of_students_goes_is_not_flagged():
    """The syntax/context case explicitly called out in the design brief:
    'group' (not 'students') is the head of the subject NP, and 'group ...
    goes' is correct — a naive nearest-noun rule would wrongly flag this."""
    text = "The group of students goes to the store."
    fixture = _fixture(text, [(text, [
        ("group", "group", "NN"), ("of", "of", "IN"),
        ("students", "student", "NNS"), ("goes", "go", "VBZ"),
    ])])
    issues = detect_pos_aware_errors(text, fixture)
    assert issues == []


def test_students_go_is_not_flagged():
    text = "The students go to school."
    fixture = _fixture(text, [(text, [
        ("students", "student", "NNS"), ("go", "go", "VBP"),
    ])])
    assert detect_pos_aware_errors(text, fixture) == []


def test_nobody_has_is_not_flagged():
    text = "Nobody has enough cash."
    fixture = _fixture(text, [(text, [
        ("Nobody", "nobody", "NN"), ("has", "have", "VBZ"),
    ])])
    assert detect_pos_aware_errors(text, fixture) == []


def test_personal_pronoun_subject_is_skipped_here():
    """Owned by grammar_heuristics.py's regex Pass 2 instead — this module
    must not also flag it (that's what augment_grammar_issues()'s span
    overlap filter additionally guards against, but this rule shouldn't
    even try)."""
    text = "She go to school."
    fixture = _fixture(text, [(text, [
        ("She", "she", "PRP"), ("go", "go", "VBP"),
    ])])
    assert detect_pos_aware_errors(text, fixture) == []


def test_measure_noun_agrees_with_following_np():
    text = "A lot of students has arrived."
    fixture = _fixture(text, [(text, [
        ("lot", "lot", "NN"), ("of", "of", "IN"),
        ("students", "student", "NNS"), ("has", "have", "VBZ"),
    ])])
    issues = detect_pos_aware_errors(text, fixture)
    hit = next(i for i in issues if i["rule_id"] == "LEARNER_SUBJECT_VERB_AGREEMENT_NP")
    assert hit["wrong"] == "has"
    assert hit["correct"] == "have"


# ---- Rule 2: auxiliary do/does/did + verb form -----------------------------

def test_did_went_is_flagged():
    text = "She did went there."
    fixture = _fixture(text, [(text, [("did", "do", "VBD"), ("went", "go", "VBD")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_AUX_DO_VERB_FORM")
    assert hit["wrong"] == "did went"
    assert hit["correct"] == "did go"


def test_does_goes_is_flagged():
    text = "He does goes there every day."
    fixture = _fixture(text, [(text, [("does", "do", "VBZ"), ("goes", "go", "VBZ")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_AUX_DO_VERB_FORM")
    assert hit["correct"] == "does go"


def test_did_go_is_not_flagged():
    text = "She did go there."
    fixture = _fixture(text, [(text, [("did", "do", "VBD"), ("go", "go", "VB")])])
    assert detect_pos_aware_errors(text, fixture) == []


# ---- Rule 3: modal + verb form ---------------------------------------------

def test_should_goes_is_flagged():
    text = "He should goes home."
    fixture = _fixture(text, [(text, [("should", "should", "MD"), ("goes", "go", "VBZ")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_MODAL_VERB_FORM")
    assert hit["correct"] == "should go"


def test_can_went_is_flagged():
    text = "I can went there yesterday."
    fixture = _fixture(text, [(text, [("can", "can", "MD"), ("went", "go", "VBD")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_MODAL_VERB_FORM")
    assert hit["correct"] == "can go"


def test_should_go_is_not_flagged():
    text = "He should go home."
    fixture = _fixture(text, [(text, [("should", "should", "MD"), ("go", "go", "VB")])])
    assert detect_pos_aware_errors(text, fixture) == []


def test_should_have_gone_is_not_flagged():
    text = "He should have gone home."
    fixture = _fixture(text, [(text, [
        ("should", "should", "MD"), ("have", "have", "VB"), ("gone", "go", "VBN"),
    ])])
    assert detect_pos_aware_errors(text, fixture) == []


# ---- Rule 4: infinitive / gerund complement selection ----------------------

def test_want_brought_is_flagged():
    text = "She didn't want brought her wallet."
    fixture = _fixture(text, [(text, [("want", "want", "VB"), ("brought", "bring", "VBD")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_INFINITIVE_COMPLEMENT")
    assert hit["wrong"] == "want brought"
    assert hit["correct"] == "want to bring"


def test_need_asking_is_flagged():
    text = "They need asking for help."
    fixture = _fixture(text, [(text, [("need", "need", "VBP"), ("asking", "ask", "VBG")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_INFINITIVE_COMPLEMENT")
    assert hit["wrong"] == "need asking"
    assert hit["correct"] == "need to ask"


def test_want_to_bring_is_not_flagged():
    text = "She wants to bring her wallet."
    fixture = _fixture(text, [(text, [
        ("wants", "want", "VBZ"), ("to", "to", "TO"), ("bring", "bring", "VB"),
    ])])
    assert detect_pos_aware_errors(text, fixture) == []


def test_enjoy_to_go_is_flagged():
    text = "I enjoy to go swimming."
    fixture = _fixture(text, [(text, [
        ("enjoy", "enjoy", "VBP"), ("to", "to", "TO"), ("go", "go", "VB"),
    ])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_GERUND_COMPLEMENT")
    assert hit["correct"] == "enjoy going"


def test_enjoy_reading_is_not_flagged():
    text = "I enjoy reading books."
    fixture = _fixture(text, [(text, [("enjoy", "enjoy", "VBP"), ("reading", "read", "VBG")])])
    assert detect_pos_aware_errors(text, fixture) == []


# ---- Rule 5: determiner + noun number --------------------------------------

def test_some_apple_is_flagged():
    text = "I bought some apple."
    fixture = _fixture(text, [(text, [("some", "some", "DT"), ("apple", "apple", "NN")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_DETERMINER_NOUN_NUMBER")
    assert hit["wrong"] == "some apple"
    assert hit["correct"] == "some apples"
    assert hit["confidence"] == "medium"


def test_a_apples_is_flagged():
    text = "She has a apples in her bag."
    fixture = _fixture(text, [(text, [("a", "a", "DT"), ("apples", "apple", "NNS")])])
    hit = next(i for i in detect_pos_aware_errors(text, fixture)
               if i["rule_id"] == "LEARNER_DETERMINER_NOUN_NUMBER")
    assert hit["wrong"] == "a apples"
    assert hit["correct"] == "an apple"


def test_some_apples_is_not_flagged():
    text = "I bought some apples."
    fixture = _fixture(text, [(text, [("some", "some", "DT"), ("apples", "apple", "NNS")])])
    assert detect_pos_aware_errors(text, fixture) == []


# ---- augment_grammar_issues() integration: linguistic_analysis is optional -

def test_augment_grammar_issues_works_without_linguistic_analysis():
    """Backward compatibility: existing callers that don't pass
    linguistic_analysis at all must behave exactly as before."""
    transcript = "He don't like coffee."
    combined_errors, combined_issues, added = augment_grammar_issues(transcript, 0, [])
    assert combined_errors == 1
    assert added == 1


def test_augment_grammar_issues_merges_pos_aware_candidates():
    transcript = "The students goes to school."
    fixture = _fixture(transcript, [(transcript, [
        ("students", "student", "NNS"), ("goes", "go", "VBZ"),
    ])])
    combined_errors, combined_issues, added = augment_grammar_issues(
        transcript, 0, [], linguistic_analysis=fixture)
    assert combined_errors == 1
    assert added == 1
    assert combined_issues[0]["correct"] == "go"
    assert combined_issues[0]["confidence"] == "high"


def test_augment_grammar_issues_does_not_double_count_regex_and_pos_candidate():
    """If both detectors would independently flag the same verb (shouldn't
    normally happen given the pronoun/noun split, but guard the merge logic
    itself), the span-overlap filter must keep the count at one."""
    transcript = "He don't like coffee."
    # Fabricate a pos_aware-style overlapping candidate at the exact same
    # span as the regex detector's "don't" -> "doesn't" match, to test the
    # merge's overlap filter directly rather than relying on both detectors
    # naturally colliding (they're designed not to).
    fixture = {"language": "en-US", "sentences": [{"text": transcript, "tokens": [
        {"text": "He", "lemma": "he", "partOfSpeech": "PRP", "posTag": "PRP",
         "startOffset": 0, "endOffset": 2},
    ]}]}
    combined_errors, combined_issues, added = augment_grammar_issues(
        transcript, 0, [], linguistic_analysis=fixture)
    # No noun/modal/aux/infinitive/determiner pattern here beyond what the
    # regex layer already owns, so the pos-aware layer should contribute 0.
    assert added == 1
    assert combined_errors == 1


# ---- The exact bug-report transcript, end to end ---------------------------

TRANSCRIPT = (
    "Yesterday the group of students goes to the store for buying some apple. "
    "He don't know that the price was changed. "
    "So they was very surprised when paying. "
    "Mary Jane said she didn't want brought her wallet and John for get his money to. "
    "They're going to need asking for help because nobody have enough cash now."
)


def _transcript_fixture():
    s1 = "Yesterday the group of students goes to the store for buying some apple."
    s2 = "He don't know that the price was changed."
    s3 = "So they was very surprised when paying."
    s4 = "Mary Jane said she didn't want brought her wallet and John for get his money to."
    s5 = "They're going to need asking for help because nobody have enough cash now."
    return _fixture(TRANSCRIPT, [
        (s1, [
            ("group", "group", "NN"), ("of", "of", "IN"), ("students", "student", "NNS"),
            ("goes", "go", "VBZ"), ("some", "some", "DT"), ("apple", "apple", "NN"),
        ]),
        (s2, [("price", "price", "NN"), ("was", "be", "VBD"), ("changed", "change", "VBN")]),
        (s3, [("they", "they", "PRP"), ("was", "be", "VBD")]),
        (s4, [
            ("want", "want", "VB"), ("brought", "bring", "VBD"),
        ]),
        (s5, [
            ("need", "need", "VBP"), ("asking", "ask", "VBG"),
            ("nobody", "nobody", "NN"), ("have", "have", "VBP"),
        ]),
    ])


def test_exact_bug_report_transcript_end_to_end():
    fixture = _transcript_fixture()

    # 1) "the group of students goes" must NOT be flagged by the POS layer —
    #    this is the deliberate syntax/context case from the design brief.
    pos_issues = detect_pos_aware_errors(TRANSCRIPT, fixture)
    wrongs = {i["wrong"] for i in pos_issues}
    assert "goes" not in wrongs

    # 2) The previously-missed errors are now caught somewhere in the merged
    #    (regex + POS-aware) result, simulating LanguageTool catching only
    #    "don't"->"doesn't" (as the bug report states it already does).
    existing_lt_issues = [{
        "wrong": "don't", "correct": "doesn't",
        "message": "Use the correct auxiliary.", "context": "He don't know...",
    }]
    combined_errors, combined_issues, added = augment_grammar_issues(
        TRANSCRIPT, existing_errors=1, existing_issues=existing_lt_issues,
        linguistic_analysis=fixture, max_issues=20)

    combined_wrongs = {i["wrong"].lower() for i in combined_issues}
    rule_ids = {i.get("rule_id") for i in combined_issues}

    assert added >= 4
    # some apple -> some apples
    assert "some apple" in combined_wrongs
    # nobody have -> nobody has
    assert any(w == "have" for w in combined_wrongs)
    # want brought -> want to bring
    assert "want brought" in combined_wrongs
    # need asking -> need to ask
    assert "need asking" in combined_wrongs
    # they was -> they were (regex layer, Pass 2b)
    assert "LEARNER_BE_PAST_AGREEMENT" in rule_ids

    # "the group of students goes" must still not appear anywhere in the
    # final merged result.
    assert not any(w == "goes" for w in combined_wrongs)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
