"""
Precision / recall / F1 benchmark for the learner-grammar detection layer
(grammar_heuristics.detect_learner_errors + grammar_pos_rules.detect_pos_aware_errors,
combined via grammar_heuristics.augment_grammar_issues).

There is no existing labeled grammar-error corpus in this repository, so
this benchmark is built from the transcript given in the design brief plus
the brief's own POSITIVE_CASES (sentences that contain exactly one known
learner error each and MUST be flagged) and NEGATIVE_CASES (grammatically
correct sentences, including the intervening-noun-phrase cases, that MUST
NOT be flagged). This is a small, hand-labeled sanity benchmark, not a
claim of general-domain accuracy — see the printed caveat at the bottom.

Run: python3 tests/benchmark_grammar_learner.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar_heuristics import detect_learner_errors

# Each positive case is (sentence, linguistic_analysis_or_None). Cases that
# need POS data to be caught (noun-subject agreement, modal/aux+verb form,
# infinitive/gerund selection, determiner+number) build a minimal fixture
# inline; cases the regex layer already owns pass None.
from tests.test_grammar_pos_rules import _fixture
from grammar_pos_rules import detect_pos_aware_errors


def _combined(sentence, linguistic_analysis=None):
    issues = detect_learner_errors(sentence)
    if linguistic_analysis:
        issues = issues + detect_pos_aware_errors(sentence, linguistic_analysis)
    return issues


POSITIVE_CASES = [
    ("The students goes to school.",
     _fixture("The students goes to school.",
              [("The students goes to school.", [("students", "student", "NNS"), ("goes", "go", "VBZ")])])),
    ("Nobody have enough cash.",
     _fixture("Nobody have enough cash.",
              [("Nobody have enough cash.", [("Nobody", "nobody", "NN"), ("have", "have", "VBP")])])),
    ("He don't know.", None),
    ("They was surprised.", None),
    ("She did went there.",
     _fixture("She did went there.",
              [("She did went there.", [("did", "do", "VBD"), ("went", "go", "VBD")])])),
    ("He should goes home.",
     _fixture("He should goes home.",
              [("He should goes home.", [("should", "should", "MD"), ("goes", "go", "VBZ")])])),
    ("She didn't want brought her wallet.",
     _fixture("She didn't want brought her wallet.",
              [("She didn't want brought her wallet.", [("want", "want", "VB"), ("brought", "bring", "VBD")])])),
    ("They need asking for help.",
     _fixture("They need asking for help.",
              [("They need asking for help.", [("need", "need", "VBP"), ("asking", "ask", "VBG")])])),
    ("I bought some apple.",
     _fixture("I bought some apple.",
              [("I bought some apple.", [("some", "some", "DT"), ("apple", "apple", "NN")])])),
]

NEGATIVE_CASES = [
    ("The group of students goes to school.",
     _fixture("The group of students goes to school.",
              [("The group of students goes to school.",
                [("group", "group", "NN"), ("of", "of", "IN"),
                 ("students", "student", "NNS"), ("goes", "go", "VBZ")])])),
    ("The students go to school.",
     _fixture("The students go to school.",
              [("The students go to school.", [("students", "student", "NNS"), ("go", "go", "VBP")])])),
    ("Nobody has enough cash.",
     _fixture("Nobody has enough cash.",
              [("Nobody has enough cash.", [("Nobody", "nobody", "NN"), ("has", "have", "VBZ")])])),
    ("He doesn't know.", None),
    ("They were surprised.", None),
    ("She did go there.",
     _fixture("She did go there.",
              [("She did go there.", [("did", "do", "VBD"), ("go", "go", "VB")])])),
    ("He should go home.",
     _fixture("He should go home.",
              [("He should go home.", [("should", "should", "MD"), ("go", "go", "VB")])])),
    ("I bought some apples.",
     _fixture("I bought some apples.",
              [("I bought some apples.", [("some", "some", "DT"), ("apples", "apple", "NNS")])])),
    ("The book on the shelf belongs to her.", None),
    ("The woman with the two children is waiting.", None),
]


def run():
    tp = fn = fp = tn = 0

    for sentence, la in POSITIVE_CASES:
        issues = _combined(sentence, la)
        if issues:
            tp += 1
        else:
            fn += 1
            print(f"[MISS]  no issue detected for positive case: {sentence!r}")

    for sentence, la in NEGATIVE_CASES:
        issues = _combined(sentence, la)
        if issues:
            fp += 1
            print(f"[FALSE POSITIVE] flagged a correct sentence: {sentence!r} -> {issues}")
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else float("nan"))

    print()
    print(f"Positive cases : {len(POSITIVE_CASES)}  (detected: {tp}, missed: {fn})")
    print(f"Negative cases : {len(NEGATIVE_CASES)}  (correctly clean: {tn}, false positives: {fp})")
    print(f"Precision      : {precision:.3f}")
    print(f"Recall         : {recall:.3f}")
    print(f"F1             : {f1:.3f}")
    print()
    print("Caveat: this is a small hand-labeled sanity benchmark (18 sentences drawn")
    print("from the design brief's own positive/negative lists), not a claim about")
    print("accuracy on open-domain learner speech. It exists to catch regressions in")
    print("this rule set, not to certify a general accuracy number.")


if __name__ == "__main__":
    run()
