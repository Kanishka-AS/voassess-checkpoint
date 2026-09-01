"""
Learner-focused grammar heuristics — a deterministic, offline, CPU-cheap
detection layer that sits *alongside* LanguageTool, not instead of it.

Why this exists
----------------
LanguageTool (whether the HTTP server, the local Java fallback, or the naive
regex fallback) is a general-purpose proofreading tool. It's tuned for
"typo/style" mistakes in otherwise well-formed text, and can genuinely
report 0 errors on sentences that are grammatically broken in ways typical
of English learners — wrong/missing auxiliary verbs, subject-verb
agreement, missing past tense, etc. — because the words involved are all
individually valid and LanguageTool's own rule set doesn't happen to cover
that exact pattern.

This module adds a small set of hand-written, high-precision pattern
detectors for the error types that matter most for a speaking-assessment
tool aimed at English learners:

  1. Missing/incorrect "be" auxiliary before a progressive verb
     ("I wear eating" / "I studying computer science" -> "I am studying").
  2. Subject-verb agreement for third-person singular
     ("She go to school" -> "She goes to school").
  3. "do/does" negation agreement ("He don't like coffee" -> "doesn't").
  4. Missing past tense when an explicit past-time expression is present
     ("Yesterday I go to college" -> "went").

Design constraints (see project requirements):
  * CPU-friendly, low-RAM, fully offline, deterministic — no ML model, no
    network call, no LLM. Just regex + small hand-built lexicons.
  * High precision over high recall: every list here is intentionally
    curated (not "any word ending in -ing") so normal, correct English is
    not flagged. False positives are worse than missed errors for a score
    that learners will trust.
  * Additive: this module never decides the final grammar score by itself.
    app.py merges its output with whatever LanguageTool already found (see
    `augment_grammar_issues` below), de-duplicating so the same real
    mistake is never counted twice.
"""
import re

# ── Lexicons ──────────────────────────────────────────────────────────────
PRONOUNS_1 = {"i"}
PRONOUNS_2PL = {"you", "we", "they"}
PRONOUNS_3SG = {"he", "she", "it"}
SUBJECT_PRONOUNS = PRONOUNS_1 | PRONOUNS_2PL | PRONOUNS_3SG

MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being"}
DO_FORMS = {"do", "does", "did", "don't", "doesn't", "didn't", "dont", "doesnt", "didnt"}
HAVE_AUX = {"have", "has", "had"}

def _regular_3sg(v: str) -> str:
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    if len(v) > 1 and v[-1] == "y" and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _regular_past(v: str) -> str:
    if v.endswith("e"):
        return v + "d"
    if len(v) > 1 and v[-1] == "y" and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


# Verbs that legitimately take a bare gerund complement right after the
# subject+verb ("I enjoy reading", "She keeps talking") — excluded from the
# "verb stacking" check below so they don't get misflagged as missing "be".
# Expanded to include their common inflections ("keeps", "kept", "started",
# ...) since the stacking check looks at the actual (conjugated) surface form.
_CATENATIVE_BASE_VERBS = {
    "enjoy", "love", "like", "dislike", "hate", "avoid", "consider", "suggest",
    "recommend", "practice", "practise", "keep", "start", "stop", "finish",
    "quit", "resist", "risk", "mind", "miss", "deny", "admit", "delay",
    "postpone", "imagine", "involve", "appreciate", "anticipate", "continue",
    "begin", "prefer", "try", "remember", "forget", "regret", "propose",
    "discuss", "report", "dread", "tolerate", "contemplate", "fancy",
    "adore", "detest", "endure", "resent",
}
_CATENATIVE_IRREGULAR_FORMS = {
    "begin": {"begins", "began"},
    "forget": {"forgets", "forgot"},
    "keep": {"keeps", "kept"},
}


def _expand_verb_forms(bases: set) -> set:
    expanded = set()
    for b in bases:
        expanded.add(b)
        expanded |= _CATENATIVE_IRREGULAR_FORMS.get(b, {_regular_3sg(b), _regular_past(b)})
    return expanded


CATENATIVE_GERUND_VERBS = _expand_verb_forms(_CATENATIVE_BASE_VERBS)

# Fixed "go + gerund" collocations ("go swimming") that are correct as-is.
GO_FIXED_GERUNDS = {
    "shopping", "swimming", "running", "jogging", "hiking", "camping",
    "fishing", "bowling", "skiing", "dancing", "sightseeing", "surfing",
}
GO_VERB_SURFACE_FORMS = {"go", "goes", "went", "going", "gone"}

# Curated set of common action-verb gerunds. Deliberately a whitelist (not a
# generic "*ing" regex) to avoid flagging ordinary -ing nouns/adjectives
# like "morning", "something", "building", "interesting", "ring", etc.
GERUND_WHITELIST = {
    "eating", "studying", "working", "playing", "going", "doing", "running",
    "sleeping", "reading", "writing", "talking", "walking", "driving",
    "cooking", "cleaning", "watching", "listening", "thinking", "trying",
    "learning", "teaching", "helping", "dancing", "swimming", "shopping",
    "traveling", "travelling", "waiting", "living", "staying", "calling",
    "texting", "coding", "programming", "testing", "building", "making",
    "taking", "coming", "sitting", "standing", "laughing", "smiling",
    "speaking", "singing", "drawing", "painting", "jogging", "exercising",
    "practicing", "practising", "preparing", "planning", "organizing",
    "managing", "leading", "developing", "creating", "designing", "growing",
    "improving", "understanding", "explaining", "discussing", "arguing",
    "complaining", "worrying", "hoping", "wishing", "dreaming", "feeling",
    "believing", "knowing", "wondering", "considering", "enjoying", "loving",
    "liking", "hating", "missing", "needing", "wanting", "using", "buying",
    "selling", "paying", "spending", "saving", "earning", "training",
    "coaching", "mentoring", "guiding", "supporting", "caring", "raising",
    "asking", "answering", "checking", "fixing", "cutting", "drinking",
    "jumping", "climbing", "riding", "flying", "sailing", "rowing",
    "skating", "meeting", "visiting", "touring", "exploring",
}

# base -> {"3sg": ..., "past": ...}. Curated list of ~90 high-frequency
# verbs. Intentionally NOT exhaustive/generic — restricting agreement/tense
# checks to a known-safe list keeps false positives near zero.
VERB_FORMS = {
    "go": {"3sg": "goes", "past": "went"},
    "do": {"3sg": "does", "past": "did"},
    "have": {"3sg": "has", "past": "had"},
    "study": {"3sg": "studies", "past": "studied"},
    "try": {"3sg": "tries", "past": "tried"},
    "watch": {"3sg": "watches", "past": "watched"},
    "wish": {"3sg": "wishes", "past": "wished"},
    "come": {"3sg": "comes", "past": "came"},
    "take": {"3sg": "takes", "past": "took"},
    "make": {"3sg": "makes", "past": "made"},
    "get": {"3sg": "gets", "past": "got"},
    "know": {"3sg": "knows", "past": "knew"},
    "think": {"3sg": "thinks", "past": "thought"},
    "say": {"3sg": "says", "past": "said"},
    "eat": {"3sg": "eats", "past": "ate"},
    "play": {"3sg": "plays", "past": "played"},
    "work": {"3sg": "works", "past": "worked"},
    "live": {"3sg": "lives", "past": "lived"},
    "speak": {"3sg": "speaks", "past": "spoke"},
    "walk": {"3sg": "walks", "past": "walked"},
    "run": {"3sg": "runs", "past": "ran"},
    "sleep": {"3sg": "sleeps", "past": "slept"},
    "read": {"3sg": "reads", "past": "read"},
    "write": {"3sg": "writes", "past": "wrote"},
    "teach": {"3sg": "teaches", "past": "taught"},
    "help": {"3sg": "helps", "past": "helped"},
    "stay": {"3sg": "stays", "past": "stayed"},
    "enjoy": {"3sg": "enjoys", "past": "enjoyed"},
    "believe": {"3sg": "believes", "past": "believed"},
    "feel": {"3sg": "feels", "past": "felt"},
    "want": {"3sg": "wants", "past": "wanted"},
    "need": {"3sg": "needs", "past": "needed"},
    "like": {"3sg": "likes", "past": "liked"},
    "love": {"3sg": "loves", "past": "loved"},
    "see": {"3sg": "sees", "past": "saw"},
    "buy": {"3sg": "buys", "past": "bought"},
    "give": {"3sg": "gives", "past": "gave"},
    "find": {"3sg": "finds", "past": "found"},
    "leave": {"3sg": "leaves", "past": "left"},
    "meet": {"3sg": "meets", "past": "met"},
    "drive": {"3sg": "drives", "past": "drove"},
    "call": {"3sg": "calls", "past": "called"},
    "ask": {"3sg": "asks", "past": "asked"},
    "use": {"3sg": "uses", "past": "used"},
    "tell": {"3sg": "tells", "past": "told"},
    "start": {"3sg": "starts", "past": "started"},
    "stop": {"3sg": "stops", "past": "stopped"},
    "finish": {"3sg": "finishes", "past": "finished"},
    "cook": {"3sg": "cooks", "past": "cooked"},
    "clean": {"3sg": "cleans", "past": "cleaned"},
    "listen": {"3sg": "listens", "past": "listened"},
    "wait": {"3sg": "waits", "past": "waited"},
    "visit": {"3sg": "visits", "past": "visited"},
    "travel": {"3sg": "travels", "past": "traveled"},
    "arrive": {"3sg": "arrives", "past": "arrived"},
    "understand": {"3sg": "understands", "past": "understood"},
    "bring": {"3sg": "brings", "past": "brought"},
    "build": {"3sg": "builds", "past": "built"},
    "send": {"3sg": "sends", "past": "sent"},
    "spend": {"3sg": "spends", "past": "spent"},
    "pay": {"3sg": "pays", "past": "paid"},
    "sit": {"3sg": "sits", "past": "sat"},
    "stand": {"3sg": "stands", "past": "stood"},
    "begin": {"3sg": "begins", "past": "began"},
    "break": {"3sg": "breaks", "past": "broke"},
    "choose": {"3sg": "chooses", "past": "chose"},
    "grow": {"3sg": "grows", "past": "grew"},
    "hold": {"3sg": "holds", "past": "held"},
    "keep": {"3sg": "keeps", "past": "kept"},
    "lose": {"3sg": "loses", "past": "lost"},
    "sell": {"3sg": "sells", "past": "sold"},
    "wear": {"3sg": "wears", "past": "wore"},
    "win": {"3sg": "wins", "past": "won"},
    "fly": {"3sg": "flies", "past": "flew"},
    "cut": {"3sg": "cuts", "past": "cut"},
    "hear": {"3sg": "hears", "past": "heard"},
    "put": {"3sg": "puts", "past": "put"},
    "set": {"3sg": "sets", "past": "set"},
    "show": {"3sg": "shows", "past": "showed"},
    "sing": {"3sg": "sings", "past": "sang"},
    "swim": {"3sg": "swims", "past": "swam"},
    "catch": {"3sg": "catches", "past": "caught"},
    "draw": {"3sg": "draws", "past": "drew"},
    "drink": {"3sg": "drinks", "past": "drank"},
    "forget": {"3sg": "forgets", "past": "forgot"},
    "ride": {"3sg": "rides", "past": "rode"},
    "throw": {"3sg": "throws", "past": "threw"},
    "cook": {"3sg": "cooks", "past": "cooked"},
    "join": {"3sg": "joins", "past": "joined"},
    "move": {"3sg": "moves", "past": "moved"},
    "open": {"3sg": "opens", "past": "opened"},
    "close": {"3sg": "closes", "past": "closed"},
    "explain": {"3sg": "explains", "past": "explained"},
    "practice": {"3sg": "practices", "past": "practiced"},
}
# Reverse index: conjugated 3sg surface form -> base verb (for catching
# "I likes" / "they goes" — a subject that should take the base form but is
# instead using the third-person-singular form).
_THIRD_SG_TO_BASE = {forms["3sg"]: base for base, forms in VERB_FORMS.items()}

TIME_MARKER_RE = re.compile(
    r"\byesterday\b|\blast\s+(?:night|week|month|year|summer|winter|weekend)\b"
    r"|\bago\b|\bwhen\s+i\s+was\s+(?:a\s+)?(?:child|young|kid|little)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _tokenize(text):
    """Returns [(word, start, end), ...] for word-ish tokens in `text`."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _sentence_spans(text):
    spans = []
    start = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _make_issue(wrong, correct, offset, length, message, rule_id, category, context,
                 confidence="high"):
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
        # HIGH/MEDIUM/LOW evidence level for this candidate (see module
        # docstring in grammar_pos_rules.py). Every detector in *this* file
        # is a hand-curated, narrow pattern match, so "high" by default —
        # only the newer POS-aware layer (grammar_pos_rules.py) currently
        # emits "medium" for lexicon-dependent rules.
        "confidence": confidence,
    }


def detect_learner_errors(text: str) -> list:
    """
    Runs the deterministic learner-error detectors over `text` and returns a
    list of issue dicts shaped like LanguageTool's own issues (wrong/correct/
    message/context/rule_id/category/offset/length), plus an additive
    "source": "learner_heuristic" so callers/UI can tell them apart from real
    LanguageTool matches if they want to.

    Pure function of the transcript text — no network call, no state, same
    input always produces the same output.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    n = len(tokens)
    sentence_spans = _sentence_spans(text)
    issues = []
    # Token *indices* already used by an issue, so later passes don't
    # double-flag the same verb for a second, lower-priority reason.
    flagged = set()

    def sentence_for(pos):
        for s, e in sentence_spans:
            if s <= pos < e:
                return text[s:e].strip()
        return text.strip()

    # ── Pass 1: missing past tense with an explicit past-time expression ──
    # Highest priority: "Yesterday I go to college" should be corrected to
    # "went", not flagged as a present-tense agreement issue.
    for i in range(n - 1):
        word, start, _ = tokens[i]
        wl = word.lower()
        if wl not in SUBJECT_PRONOUNS:
            continue
        vword, vstart, vend = tokens[i + 1]
        vl = vword.lower()
        if vl in VERB_FORMS and TIME_MARKER_RE.search(sentence_for(start)):
            correct_verb = VERB_FORMS[vl]["past"]
            if correct_verb != vl:
                issues.append(_make_issue(
                    wrong=vword, correct=correct_verb, offset=vstart, length=vend - vstart,
                    message=(f"A past-time expression is present, but '{vword}' is in the "
                             f"present tense — use '{correct_verb}'."),
                    rule_id="LEARNER_PAST_TENSE_REQUIRED", category="Verb Tense",
                    context=sentence_for(start),
                ))
                flagged.add(i + 1)

    # ── Pass 2: subject-verb agreement (third-person singular) ────────────
    for i in range(n - 1):
        if (i + 1) in flagged:
            continue
        word, start, _ = tokens[i]
        wl = word.lower()
        vword, vstart, vend = tokens[i + 1]
        vl = vword.lower()
        if wl in PRONOUNS_3SG and vl in VERB_FORMS:
            correct_verb = VERB_FORMS[vl]["3sg"]
            if correct_verb != vl:
                issues.append(_make_issue(
                    wrong=vword, correct=correct_verb, offset=vstart, length=vend - vstart,
                    message=(f"'{word}' needs a third-person-singular verb — "
                             f"use '{correct_verb}' instead of '{vword}'."),
                    rule_id="LEARNER_SUBJECT_VERB_AGREEMENT", category="Subject-Verb Agreement",
                    context=sentence_for(start),
                ))
                flagged.add(i + 1)
        elif wl in (PRONOUNS_1 | PRONOUNS_2PL) and vl in _THIRD_SG_TO_BASE:
            base = _THIRD_SG_TO_BASE[vl]
            issues.append(_make_issue(
                wrong=vword, correct=base, offset=vstart, length=vend - vstart,
                message=f"'{word}' takes the base verb form — use '{base}' instead of '{vword}'.",
                rule_id="LEARNER_SUBJECT_VERB_AGREEMENT", category="Subject-Verb Agreement",
                context=sentence_for(start),
            ))
            flagged.add(i + 1)

    # ── Pass 2b: "be" past-tense agreement (was/were) ──────────────────────
    # "they was" -> "they were" / "he were" -> "he was". Kept separate from
    # Pass 2 because "be" isn't in VERB_FORMS (irregular in a way that
    # doesn't fit the base/3sg/past shape the rest of that table uses).
    # "I" is deliberately excluded on both sides: "I was" is always correct,
    # and "I were" is often legitimate subjunctive ("I wish I were taller")
    # rather than a mistake, so it's left to LanguageTool rather than risk a
    # false positive here. The same subjunctive concern applies to 3rd-person
    # singular "were" ("If he were rich..."), so that direction is only
    # flagged when the immediately preceding context doesn't look
    # subjunctive.
    _SUBJUNCTIVE_TRIGGERS = ("if", "wish", "as if", "as though")
    for i in range(n - 1):
        if (i + 1) in flagged:
            continue
        word, start, _ = tokens[i]
        wl = word.lower()
        vword, vstart, vend = tokens[i + 1]
        vl = vword.lower()
        if wl in PRONOUNS_2PL and vl == "was":
            issues.append(_make_issue(
                wrong=vword, correct="were", offset=vstart, length=vend - vstart,
                message=f"'{word}' takes 'were', not 'was'.",
                rule_id="LEARNER_BE_PAST_AGREEMENT", category="Subject-Verb Agreement",
                context=sentence_for(start),
            ))
            flagged.add(i + 1)
        elif wl in PRONOUNS_3SG and vl == "were":
            preceding = text[max(0, start - 20):start].lower()
            if any(trig in preceding for trig in _SUBJUNCTIVE_TRIGGERS):
                continue
            issues.append(_make_issue(
                wrong=vword, correct="was", offset=vstart, length=vend - vstart,
                message=f"'{word}' takes 'was', not 'were'.",
                rule_id="LEARNER_BE_PAST_AGREEMENT", category="Subject-Verb Agreement",
                context=sentence_for(start),
            ))
            flagged.add(i + 1)

    # ── Pass 3: do/does negation agreement ─────────────────────────────────
    for i in range(n - 1):
        word, start, _ = tokens[i]
        wl = word.lower()
        vword, vstart, vend = tokens[i + 1]
        vl = vword.lower().replace("\u2019", "'")
        if wl in PRONOUNS_3SG and vl == "don't":
            issues.append(_make_issue(
                wrong=vword, correct="doesn't", offset=vstart, length=vend - vstart,
                message=f"Use \"doesn't\" with '{word}', not \"don't\".",
                rule_id="LEARNER_DO_AUX_AGREEMENT", category="Auxiliary Agreement",
                context=sentence_for(start),
            ))
        elif wl in (PRONOUNS_1 | PRONOUNS_2PL) and vl == "doesn't":
            issues.append(_make_issue(
                wrong=vword, correct="don't", offset=vstart, length=vend - vstart,
                message=f"Use \"don't\" with '{word}', not \"doesn't\".",
                rule_id="LEARNER_DO_AUX_AGREEMENT", category="Auxiliary Agreement",
                context=sentence_for(start),
            ))

    # ── Pass 4: missing/incorrect "be" auxiliary before a gerund ───────────
    for i in range(n - 1):
        word, start, _ = tokens[i]
        wl = word.lower()
        if wl not in SUBJECT_PRONOUNS:
            continue
        correct_be = "am" if wl == "i" else ("is" if wl in PRONOUNS_3SG else "are")

        v2, v2start, v2end = tokens[i + 1]
        v2l = v2.lower()

        # 4a: pronoun directly followed by a gerund — "be" is missing entirely.
        if v2l in GERUND_WHITELIST:
            phrase_end = v2end
            issues.append(_make_issue(
                wrong=f"{word} {v2}", correct=f"{word} {correct_be} {v2}",
                offset=start, length=phrase_end - start,
                message=(f"The progressive tense needs a 'be' verb — say "
                         f"'{word} {correct_be} {v2}', not '{word} {v2}'."),
                rule_id="LEARNER_MISSING_BE_AUX", category="Verb Form",
                context=sentence_for(start),
            ))
            continue

        # 4b: pronoun + another finite verb + gerund — two main verbs
        # stacked with no linking construction (e.g. "I wear eating").
        if (i + 1) not in flagged and i + 2 < n:
            v3, v3start, v3end = tokens[i + 2]
            v3l = v3.lower()
            if (v3l in GERUND_WHITELIST
                    and v2l not in BE_FORMS
                    and v2l not in MODALS
                    and v2l not in DO_FORMS
                    and v2l not in HAVE_AUX
                    and v2l not in CATENATIVE_GERUND_VERBS
                    and not (v2l in GO_VERB_SURFACE_FORMS and v3l in GO_FIXED_GERUNDS)):
                issues.append(_make_issue(
                    wrong=f"{word} {v2}", correct=f"{word} {correct_be}",
                    offset=start, length=v2end - start,
                    message=(f"'{v2}' can't be followed directly by '{v3}'. "
                             f"Did you mean '{word} {correct_be} {v3}'?"),
                    rule_id="LEARNER_VERB_STACKING", category="Verb Form",
                    context=sentence_for(start),
                ))
                flagged.add(i + 1)

    return issues


def _wrong_key(text_value: str) -> str:
    return re.sub(r"\s+", " ", (text_value or "").strip().lower())


def _overlaps(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _spans_overlap(a_offset: int, a_length: int, b_offset: int, b_length: int) -> bool:
    a_end = a_offset + (a_length or 0)
    b_end = b_offset + (b_length or 0)
    return a_offset < b_end and b_offset < a_end


def augment_grammar_issues(transcript: str, existing_errors: int, existing_issues: list,
                            linguistic_analysis: dict = None, max_issues: int = 8):
    """
    Runs detect_learner_errors() — and, when `linguistic_analysis` (the
    LanguageTool /v2/analyze token/lemma/POS data — see
    LanguageToolProvider.check_and_analyze) is available,
    grammar_pos_rules.detect_pos_aware_errors() alongside it — then merges
    the combined result into whatever LanguageTool (or its fallbacks)
    already found, de-duplicating so a mistake LanguageTool already caught
    is never counted twice.

    `linguistic_analysis` is optional and additive: when it's None (the
    /v2/analyze call failed or wasn't attempted), this behaves exactly as
    before — only the regex-based, personal-pronoun-only detectors in this
    file run. When it's available, grammar_pos_rules.py's noun/indefinite-
    pronoun-subject and lexicon-based rules run too, giving broader
    coverage (see that module's docstring) without needing a POS-free
    detector to guess syntax it can't see. See grammar_pos_rules.py for why
    this is a separate module rather than more passes here — it needs its
    own lexicons and a different (POS-driven, not purely-regex) traversal.

    De-duplication happens in two stages:
      1. Between the two detectors' own candidates: any POS-aware candidate
         whose token span overlaps a regex-based candidate's span is
         dropped — the regex layer is more narrowly tested, so it wins ties
         rather than emitting two issues for the same mistake.
      2. Against `existing_issues` (LanguageTool's own findings): a
         candidate is dropped if its flagged word/phrase overlaps (as a
         case-insensitive substring, either direction) with any existing
         issue's "wrong" text — robust even when an existing issue's
         `wrong` field doesn't carry exact offsets (e.g. minimal test
         fixtures / the local language_tool_python fallback path).

    Returns (combined_errors: int, combined_issues: list, added_count: int).
    `combined_errors` is the true total (not capped); `combined_issues` is
    capped at `max_issues` to match the existing per-response issue-list cap.
    """
    regex_candidates = detect_learner_errors(transcript)

    pos_candidates = []
    if linguistic_analysis:
        # Local import: grammar_pos_rules.py imports lexicons/helpers from
        # this module, so importing it back at module load time here would
        # be circular. It's only needed inside this branch anyway.
        from grammar_pos_rules import detect_pos_aware_errors
        raw_pos_candidates = detect_pos_aware_errors(transcript, linguistic_analysis)
        pos_candidates = [
            c for c in raw_pos_candidates
            if not any(_spans_overlap(c["offset"], c["length"], r["offset"], r["length"])
                       for r in regex_candidates)
        ]

    candidates = regex_candidates + pos_candidates
    if not candidates:
        return existing_errors, list(existing_issues)[:max_issues], 0

    # Only de-duplicate against issues LanguageTool (or its fallbacks)
    # already found — NOT against other heuristic candidates. A transcript
    # can legitimately repeat the exact same mistake ("he don't ... he
    # don't ...") several times; each occurrence is a real, separate error
    # and must still be counted. detect_learner_errors() itself already
    # avoids emitting two heuristic issues for the same token position, and
    # the span-overlap filter above does the equivalent job across the two
    # detectors.
    existing_wrongs = {_wrong_key(i.get("wrong", "")) for i in existing_issues}
    existing_wrongs.discard("")

    added = []
    for cand in candidates:
        cand_key = _wrong_key(cand["wrong"])
        # Also check just the last word of a multi-word candidate phrase
        # (e.g. "wear eating" -> "eating") against existing single-word
        # LanguageTool matches, and vice versa.
        last_word = cand_key.split()[-1] if " " in cand_key else cand_key
        if any(_overlaps(cand_key, w) or _overlaps(last_word, w) for w in existing_wrongs):
            continue
        added.append(cand)

    combined_issues = list(existing_issues) + added
    combined_errors = existing_errors + len(added)
    return combined_errors, combined_issues[:max_issues], len(added)
