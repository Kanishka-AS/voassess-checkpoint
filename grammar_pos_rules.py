"""
POS-aware learner grammar rules (LanguageTool /v2/analyze layer)
==================================================================

grammar_heuristics.py's detect_learner_errors() is a pure regex/lexicon
scan that only understands PERSONAL PRONOUN subjects (I/you/he/she/it/we/
they) — it has no idea what the grammatical subject of a sentence actually
is, so it can't safely reason about NOUN subjects at all (a naive "nearest
noun" rule would wrongly flag "The group of students goes to the store" —
see the module docstring there).

This module adds a second, independent detection pass that uses the token/
lemma/POS-tag data LanguageTool's /v2/analyze endpoint already returns (see
languagetool_provider.py) to do a little real syntax: find the head of the
subject noun phrase (skipping over prepositional-phrase modifiers), tell a
missing "to" from a wrong verb form after a modal, etc. It is still fully
deterministic, offline, and CPU-cheap — no ML model, just POS-tag pattern
matching over LanguageTool's own tagger output.

    detect_pos_aware_errors(text, linguistic_analysis) -> [issue, ...]

Pure function of (text, linguistic_analysis). Degrades to "no additional
issues" whenever linguistic_analysis is None/empty — /v2/analyze is an
optional call that can fail independently of /v2/check (see
LanguageToolProvider.check_and_analyze), and grammar scoring must never
depend on it being available. grammar_heuristics.augment_grammar_issues()
is the only intended caller; it merges this module's output with its own
regex-based candidates and de-duplicates both against LanguageTool.

Rule families implemented here
-------------------------------
  1. Subject-verb agreement for NOUN / indefinite-pronoun subjects
     ("the students goes" -> "go", "nobody have" -> "has"), using a
     lightweight NP-head heuristic: the head is the last noun *before* the
     first preposition in the subject window, so "the group of students
     goes" is correctly left alone (head = "group", singular, agrees with
     "goes") while "the students goes" (no PP) is correctly flagged.
     Personal-pronoun subjects are explicitly skipped here — Pass 2 in
     grammar_heuristics.py already owns those, and skipping avoids the two
     detectors ever double-flagging the same word for the same reason.
  2. Auxiliary "do/does/did" + verb form ("did went" -> "did go",
     "does goes" -> "does go").
  3. Modal + verb form ("should goes" -> "should go", "can went" -> "can go").
  4. Infinitive/gerund complement selection ("want brought" -> "want to
     bring", "need asking" -> "need to ask", "enjoy to go" -> "enjoy going").
  5. Determiner + noun number, restricted to a curated countable-noun
     whitelist to keep false positives near zero ("some apple" -> "some
     apples", "a apples" -> "an apple").

Confidence
----------
Every issue carries a "confidence" field: "high" for rule families backed
by an unambiguous, closed-class trigger (a specific modal/auxiliary/
pronoun set), "medium" for the two lexicon-dependent families (#4 and #5),
where the correction is right whenever the pattern fires but the curated
verb/noun lists can't claim to be exhaustive. Nothing below "medium" is
ever returned — if a candidate isn't at least medium-confidence, it's
simply not emitted (see the design brief: low-confidence guesses should
not be counted automatically, so this module never generates them in the
first place rather than emitting-then-filtering).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from grammar_heuristics import VERB_FORMS, _regular_3sg

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# ── Penn Treebank tag groups (LanguageTool's /v2/analyze posTag values) ────
NOUN_TAGS_SG = {"NN", "NNP"}
NOUN_TAGS_PL = {"NNS", "NNPS"}
NOUN_TAGS = NOUN_TAGS_SG | NOUN_TAGS_PL
VERB_3SG_TAG = "VBZ"          # goes, has, does, is
VERB_BASE_PRESENT_TAG = "VBP"  # go, have, do, are/am
MODAL_TAG = "MD"
PREP_TAG = "IN"
CC_TAG = "CC"

INDEFINITE_SINGULAR = {
    "nobody", "somebody", "anybody", "everybody", "someone", "anyone",
    "everyone", "no one", "nothing", "something", "anything", "everything",
    "each", "either", "neither",
}

# Measure/collective nouns that, in "a NOUN of X" constructions, actually
# agree with X ("a lot of students are..."), not with the measure word
# itself — the inverse of the "group of students" case, which correctly
# agrees with "group". Deliberately a short, curated list.
MEASURE_NOUNS = {"lot", "lots", "number", "couple", "majority", "plenty",
                  "bunch", "ton", "load"}

# Verbs that take a "to" + base-verb complement, never a bare finite/gerund
# verb directly after them. Curated, high-frequency, unambiguous (verbs like
# "like"/"start"/"begin" that can take either a gerund or infinitive are
# deliberately excluded to avoid false positives).
WANT_TO_VERBS = {
    "want", "need", "hope", "plan", "decide", "refuse", "agree", "manage",
    "fail", "offer", "promise", "wish", "expect", "choose", "afford",
    "tend", "pretend", "attempt", "aim", "intend", "learn",
}

# Verbs that take a bare gerund complement, never "to" + base verb.
GERUND_ONLY_VERBS = {
    "enjoy", "avoid", "finish", "mind", "suggest", "keep", "quit",
    "practice", "practise", "deny", "admit", "delay", "postpone", "resist",
    "risk", "recommend",
}

# Determiners that require a plural countable noun right after them.
SOME_ANY_DETERMINERS = {"some", "any", "several", "many", "few"}

# Curated whitelist of common countable nouns for the determiner/number
# rule — deliberately narrow (not "any noun") so mass nouns ("some water",
# "some advice", "some information") are never misflagged.
COUNTABLE_NOUNS = {
    "apple", "banana", "orange", "book", "car", "dog", "cat", "student",
    "idea", "problem", "question", "chair", "table", "phone", "picture",
    "photo", "friend", "house", "room", "job", "project", "meeting",
    "email", "message", "file", "report", "pen", "pencil", "cup", "bottle",
    "bag", "shoe", "shirt", "ticket", "gift", "present", "toy", "game",
    "movie", "song", "story", "letter", "word", "sentence", "mistake",
    "error", "reason", "plan", "goal", "dream", "memory", "lesson",
}


@dataclass
class _Tok:
    text: str
    lower: str
    start: int
    end: int
    lemma: str
    pos: str  # posTag, "" if unknown/missing


def _tokens_for_sentence(sentence: dict) -> list:
    toks = []
    for t in sentence.get("tokens", []):
        text = t.get("text", "")
        if not text or not _WORD_RE.fullmatch(text):
            continue
        toks.append(_Tok(
            text=text, lower=text.lower(),
            start=t.get("startOffset", 0), end=t.get("endOffset", 0),
            lemma=(t.get("lemma") or text).lower(),
            pos=t.get("posTag") or "",
        ))
    toks.sort(key=lambda tk: tk.start)
    return toks


def _make_issue(wrong, correct, offset, length, message, rule_id, category, context,
                 confidence):
    return {
        "wrong": wrong,
        "correct": correct,
        "message": message,
        "context": context,
        "rule_id": rule_id,
        "category": category,
        "offset": offset,
        "length": length,
        "source": "learner_heuristic",
        "confidence": confidence,
    }


def _to_gerund(base: str) -> str:
    if base.endswith("ie"):
        return base[:-2] + "ying"
    if base.endswith("e") and not base.endswith("ee"):
        return base[:-1] + "ing"
    return base + "ing"


# ── Rule 1: subject-verb agreement for noun / indefinite-pronoun subjects ──

def _subject_verb_agreement_np(toks: list, context: str) -> list:
    issues = []
    for i, tok in enumerate(toks):
        if tok.pos not in (VERB_3SG_TAG, VERB_BASE_PRESENT_TAG):
            continue
        lemma = tok.lemma
        if lemma == "be":
            # "is"/"are"/"am" agreement has enough irregular forms (and
            # overlap with the progressive-aux rules already covered
            # elsewhere) that it's left out here to keep this rule precise.
            continue

        # Walk backward collecting the subject window, stopping at a
        # preceding finite verb (a clause boundary) or after a handful of
        # tokens (a sentence-initial subject rarely needs more context).
        window = []
        j = i - 1
        while j >= 0:
            wtok = toks[j]
            if wtok.pos and wtok.pos.startswith("V") and wtok.pos not in ("VBG", "VBN"):
                break
            window.append(wtok)
            j -= 1
            if len(window) >= 8:
                break
        if not window:
            continue
        window.reverse()  # left-to-right, ending right before the verb

        # Personal-pronoun subjects belong to grammar_heuristics.py's own
        # Pass 2 — skip so the two detectors never double-flag the same verb.
        if len(window) == 1 and window[0].pos == "PRP":
            continue

        last_word = window[-1].lower
        two_word = " ".join(t.lower for t in window[-2:]) if len(window) >= 2 else last_word

        if last_word in INDEFINITE_SINGULAR or two_word in INDEFINITE_SINGULAR:
            expected = "singular"
            head_tok = window[-1]
        else:
            first_prep_idx = next((k for k, t in enumerate(window) if t.pos == PREP_TAG), None)
            core = window[:first_prep_idx] if first_prep_idx is not None else window
            core_nouns = [t for t in core if t.pos in NOUN_TAGS]
            if not core_nouns:
                continue  # no identifiable noun head in the core NP — don't guess
            head_tok = core_nouns[-1]
            if head_tok.lower in MEASURE_NOUNS and first_prep_idx is not None:
                after_nouns = [t for t in window[first_prep_idx + 1:] if t.pos in NOUN_TAGS]
                if after_nouns:
                    head_tok = after_nouns[-1]

            has_top_level_and = any(t.pos == CC_TAG and t.lower == "and" for t in window)
            if has_top_level_and:
                expected = "plural"
            elif head_tok.pos in NOUN_TAGS_PL:
                expected = "plural"
            elif head_tok.pos in NOUN_TAGS_SG:
                expected = "singular"
            else:
                continue

        if expected == "singular" and tok.pos == VERB_BASE_PRESENT_TAG:
            if lemma in VERB_FORMS:
                correct_verb = VERB_FORMS[lemma]["3sg"]
                confidence = "high"
            elif lemma == "have":
                correct_verb = "has"
                confidence = "high"
            elif lemma == "do":
                correct_verb = "does"
                confidence = "high"
            else:
                correct_verb = _regular_3sg(lemma)
                confidence = "medium"
            if correct_verb == tok.lower:
                continue
            issues.append(_make_issue(
                wrong=tok.text, correct=correct_verb, offset=tok.start, length=tok.end - tok.start,
                message=f"'{head_tok.text}' is singular here — use '{correct_verb}' instead of '{tok.text}'.",
                rule_id="LEARNER_SUBJECT_VERB_AGREEMENT_NP", category="Subject-Verb Agreement",
                context=context, confidence=confidence,
            ))
        elif expected == "plural" and tok.pos == VERB_3SG_TAG:
            correct_verb = lemma
            if correct_verb == tok.lower:
                continue
            issues.append(_make_issue(
                wrong=tok.text, correct=correct_verb, offset=tok.start, length=tok.end - tok.start,
                message=f"'{head_tok.text}' is plural here — use '{correct_verb}' instead of '{tok.text}'.",
                rule_id="LEARNER_SUBJECT_VERB_AGREEMENT_NP", category="Subject-Verb Agreement",
                context=context, confidence="high",
            ))
    return issues


# ── Rule 2: auxiliary "do/does/did" + verb form ─────────────────────────────

def _aux_do_plus_verb_form(toks: list, context: str) -> list:
    issues = []
    for i in range(len(toks) - 1):
        aux, verb = toks[i], toks[i + 1]
        if aux.lower in ("do", "does", "did") and verb.pos in ("VBD", "VBZ"):
            correct_verb = verb.lemma
            if correct_verb == verb.lower:
                continue
            issues.append(_make_issue(
                wrong=f"{aux.text} {verb.text}", correct=f"{aux.text} {correct_verb}",
                offset=aux.start, length=verb.end - aux.start,
                message=f"After '{aux.text}', use the base verb form — '{correct_verb}', not '{verb.text}'.",
                rule_id="LEARNER_AUX_DO_VERB_FORM", category="Auxiliary + Verb Form",
                context=context, confidence="high",
            ))
    return issues


# ── Rule 3: modal + verb form ───────────────────────────────────────────────

def _modal_plus_verb_form(toks: list, context: str) -> list:
    issues = []
    for i in range(len(toks) - 1):
        modal, verb = toks[i], toks[i + 1]
        if modal.pos == MODAL_TAG and verb.pos in ("VBD", "VBZ") and verb.lemma != "have":
            correct_verb = verb.lemma
            if correct_verb == verb.lower:
                continue
            issues.append(_make_issue(
                wrong=f"{modal.text} {verb.text}", correct=f"{modal.text} {correct_verb}",
                offset=modal.start, length=verb.end - modal.start,
                message=(f"After a modal verb ('{modal.text}'), use the base form — "
                         f"'{correct_verb}', not '{verb.text}'."),
                rule_id="LEARNER_MODAL_VERB_FORM", category="Modal + Verb Form",
                context=context, confidence="high",
            ))
    return issues


# ── Rule 4: infinitive / gerund complement selection ───────────────────────

def _infinitive_gerund_selection(toks: list, context: str) -> list:
    issues = []
    n = len(toks)
    for i in range(n - 1):
        head, nxt = toks[i], toks[i + 1]
        lemma = head.lemma

        if lemma in WANT_TO_VERBS and nxt.pos in ("VBD", "VBG", "VBZ", "VBN"):
            base = nxt.lemma
            issues.append(_make_issue(
                wrong=f"{head.text} {nxt.text}", correct=f"{head.text} to {base}",
                offset=head.start, length=nxt.end - head.start,
                message=(f"'{head.text}' needs a 'to' + base-verb complement — say "
                         f"'{head.text} to {base}', not '{head.text} {nxt.text}'."),
                rule_id="LEARNER_INFINITIVE_COMPLEMENT", category="Verb Complement",
                context=context, confidence="medium",
            ))
            continue

        if lemma in GERUND_ONLY_VERBS and nxt.lower == "to" and i + 2 < n:
            verb2 = toks[i + 2]
            if verb2.pos in ("VB", "VBP"):
                gerund = _to_gerund(verb2.lemma)
                issues.append(_make_issue(
                    wrong=f"{head.text} to {verb2.text}", correct=f"{head.text} {gerund}",
                    offset=head.start, length=verb2.end - head.start,
                    message=(f"'{head.text}' takes a gerund, not 'to' + verb — say "
                             f"'{head.text} {gerund}', not '{head.text} to {verb2.text}'."),
                    rule_id="LEARNER_GERUND_COMPLEMENT", category="Verb Complement",
                    context=context, confidence="medium",
                ))
    return issues


# ── Rule 5: determiner + noun number ────────────────────────────────────────

def _determiner_noun_number(toks: list, context: str) -> list:
    issues = []
    for i in range(len(toks) - 1):
        det, noun = toks[i], toks[i + 1]
        dl, nl = det.lower, noun.lower

        if dl in SOME_ANY_DETERMINERS and noun.pos == "NN" and nl in COUNTABLE_NOUNS:
            plural_form = _regular_3sg(nl)
            issues.append(_make_issue(
                wrong=f"{det.text} {noun.text}", correct=f"{det.text} {plural_form}",
                offset=det.start, length=noun.end - det.start,
                message=f"'{det.text}' takes a plural noun — use '{plural_form}' instead of '{noun.text}'.",
                rule_id="LEARNER_DETERMINER_NOUN_NUMBER", category="Noun Number",
                context=context, confidence="medium",
            ))
        elif dl == "a" and noun.pos == "NNS":
            singular_form = nl[:-3] + "y" if nl.endswith("ies") else (
                nl[:-1] if nl.endswith("s") and not nl.endswith("ss") else nl)
            if singular_form not in COUNTABLE_NOUNS:
                continue
            article = "an" if singular_form[:1] in "aeiou" else "a"
            issues.append(_make_issue(
                wrong=f"{det.text} {noun.text}", correct=f"{article} {singular_form}",
                offset=det.start, length=noun.end - det.start,
                message=(f"'{det.text}' takes a singular noun — use "
                         f"'{article} {singular_form}' instead of '{det.text} {noun.text}'."),
                rule_id="LEARNER_DETERMINER_NOUN_NUMBER", category="Noun Number",
                context=context, confidence="medium",
            ))
    return issues


def detect_pos_aware_errors(text: str, linguistic_analysis: Optional[dict]) -> list:
    """
    Runs the POS-aware learner-error detectors over each sentence in
    `linguistic_analysis` (LanguageTool's /v2/analyze response — see
    languagetool_provider.py) and returns a list of issue dicts shaped like
    grammar_heuristics.detect_learner_errors()'s output, plus a "confidence"
    field (see module docstring).

    `text` is accepted for interface symmetry with detect_learner_errors()
    and as a fallback context source; every rule here works off the tokens
    LanguageTool already extracted; it does not re-tokenize `text` itself.

    Returns [] when linguistic_analysis is None/empty/has no sentences —
    this layer is purely additive and never runs its own tokenizer, so
    there's nothing it can safely do without real POS data.
    """
    if not linguistic_analysis or not linguistic_analysis.get("sentences"):
        return []

    issues = []
    for sentence in linguistic_analysis["sentences"]:
        toks = _tokens_for_sentence(sentence)
        if not toks:
            continue
        context = (sentence.get("text") or text or "").strip()
        issues.extend(_subject_verb_agreement_np(toks, context))
        issues.extend(_aux_do_plus_verb_form(toks, context))
        issues.extend(_modal_plus_verb_form(toks, context))
        issues.extend(_infinitive_gerund_selection(toks, context))
        issues.extend(_determiner_noun_number(toks, context))
    return issues
