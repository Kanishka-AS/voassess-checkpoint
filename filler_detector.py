"""
VoAssess — Context-Aware Filler Detection
==========================================

Replaces the old flat word-list/substring filler scan with a deterministic,
rule-based detector that uses the LanguageTool `/v2/analyze` token/lemma/POS
data (already integrated in languagetool_provider.py) to tell filler usage
of a word apart from ordinary usage of the same word.

    detect_fillers(transcript, linguistic_analysis, duration_seconds) → evidence dict

No LLM, no new NLP model, no network calls of its own — this module is a
pure function over the transcript text plus whatever `/v2/analyze` already
returned for this request (or `None`, in which case it degrades to
punctuation/word-list heuristics only; see `_build_tokens`).

Two kinds of signal are produced:

  1. **Fillers** — filled pauses ("uh", "um", ...) and discourse fillers
     ("like", "you know", "I mean", ...) that are genuinely functioning as
     hesitation/hedging devices in context. These count toward `count` /
     `rate_per_min` and are what `app.py` feeds into the existing
     `score_fillers()` formula.

  2. **Hesitations** — immediate word repetitions ("I I", "the the",
     "I... I...") reported separately (`hesitations`), since a repeated
     word is a distinct disfluency phenomenon from a filler word and the
     brief explicitly asks not to fold every repetition into the filler
     count.

Design notes
------------
* Ambiguous single words (like/mean/so/well/actually/basically/just/right)
  are only ever classified using signals LanguageTool actually provides for
  that token (POS tag, and the POS of its immediate neighbors) plus
  surrounding punctuation read straight from the transcript (comma-
  bracketing, sentence position, following "?"). Nothing about a token's
  classification is invented.
* Punctuation alone is NOT a reliable signal for this: natural continuous
  speech transcripts (this project's actual input — Whisper output, not
  hand-punctuated text) very often omit the commas that would otherwise set
  a discourse marker off ("Actually we was planning..." rather than
  "Actually, we was planning..."). Every ambiguous-word and phrase
  classifier therefore also has a punctuation-independent path — sentence
  position, adjacent POS tags (e.g. a copula immediately before "like"
  followed by an adjective/adverb), or a fixed phrase adjacency (e.g. "so
  basically") — so detection does not silently fail just because the
  transcript has no commas.
* When `linguistic_analysis` is unavailable (LanguageTool's `/v2/analyze`
  down — see LanguageToolProvider.check_and_analyze), tokens still get
  built from the raw transcript, so filled pauses, phrase fillers, and
  punctuation-only rules still work; only the POS-dependent branches fall
  back to their conservative "insufficient context" default.
* Confidence is a plain heuristic (more independent signals agreeing = more
  confidence) — not a probability from a model.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

# ── Token model ────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class _Tok:
    text: str
    lower: str
    start: int
    end: int
    lemma: Optional[str] = None
    pos: Optional[str] = None


def _build_tokens(transcript: str, linguistic_analysis: Optional[dict]) -> list[_Tok]:
    """Flatten LanguageTool's sentences/tokens into one offset-ordered list
    with lemma/POS attached. Falls back to plain regex word tokenization
    (no lemma/POS) when `/v2/analyze` data isn't available or is empty —
    the caller is expected to already know that means "no LanguageTool
    server" (see LanguageToolProvider), this function just degrades."""
    if linguistic_analysis and linguistic_analysis.get("sentences"):
        toks = []
        for sent in linguistic_analysis["sentences"]:
            for t in sent.get("tokens", []):
                text = t.get("text", "")
                if not text or not _WORD_RE.fullmatch(text):
                    continue  # skip punctuation/whitespace tokens, if any
                toks.append(_Tok(
                    text=text, lower=text.lower(),
                    start=t.get("startOffset", 0), end=t.get("endOffset", 0),
                    lemma=t.get("lemma"), pos=(t.get("posTag") or None),
                ))
        if toks:
            toks.sort(key=lambda tok: tok.start)
            return toks
        # analysis was present but had no usable word tokens — fall through

    return [
        _Tok(text=m.group(0), lower=m.group(0).lower(), start=m.start(), end=m.end())
        for m in _WORD_RE.finditer(transcript)
    ]


def _prev_nonspace_char(text: str, idx: int) -> str:
    idx -= 1
    while idx >= 0 and text[idx].isspace():
        idx -= 1
    return text[idx] if idx >= 0 else ""


def _next_nonspace_char(text: str, idx: int) -> str:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return text[idx] if idx < len(text) else ""


def _is_sentence_initial(tok: _Tok, transcript: str) -> bool:
    if tok.start == 0:
        return True
    return _prev_nonspace_char(transcript, tok.start) in (".", "!", "?", "")


def _occ(toks: list[_Tok], ftype: str, confidence: float, reason: str) -> dict:
    """Build one occurrence record spanning one or more consecutive tokens."""
    words = " ".join(t.text for t in toks)
    return {
        "word": words,
        "start": toks[0].start,
        "end": toks[-1].end,
        "type": ftype,
        "confidence": round(confidence, 2),
        "reason": reason,
    }


# ── Filled pauses — near-unambiguous, always a hesitation marker ───────────

FILLED_PAUSES = {"uh", "uhh", "uhm", "um", "umm", "er", "err", "erm"}


# ── Ambiguous single-word discourse markers ─────────────────────────────────
# Each classifier looks only at: this token's POS (if any), the previous
# token, and the raw punctuation immediately before/after the token in the
# transcript. Returns (is_filler, filler_type, confidence, reason).

_ClassifyFn = Callable[[list[_Tok], int, str], tuple[bool, Optional[str], float, str]]


def _classify_like(tokens: list[_Tok], i: int, transcript: str):
    tok = tokens[i]
    pos = (tok.pos or "").upper()
    prev_c = _prev_nonspace_char(transcript, tok.start)
    next_c = _next_nonspace_char(transcript, tok.end)
    prev_tok = tokens[i - 1] if i > 0 else None
    next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
    prev_pos = (prev_tok.pos or "").upper() if prev_tok else ""
    next_pos = (next_tok.pos or "").upper() if next_tok else ""

    if pos.startswith("VB"):
        return False, None, 0.85, "used as a main verb (VB* POS) — lexical usage, not a discourse marker"
    # Sentence-initial "Like" opening a clause with a pronoun subject
    # ("Like I was thinking...") is the classic no-comma discourse-marker
    # opener. Genuine comparisons ("Like most people, ...", "Like your
    # sister, ...") are not followed directly by a personal pronoun, so
    # this stays narrow. Checked before the preposition exclusion below,
    # since LanguageTool commonly (mis)tags this exact discourse use as
    # IN (preposition) — the sentence-initial+pronoun pattern overrides
    # that tag rather than being masked by it.
    if _is_sentence_initial(tok, transcript) and next_pos == "PRP":
        return True, "discourse_filler", 0.65, "sentence-initial 'Like' immediately followed by a pronoun subject — discourse-marker opener, not a comparison (which would name a noun/subject other than a pronoun)"
    if pos == "IN" and next_c != "," and next_pos not in ("RB", "JJ"):
        return False, None, 0.7, "used as a preposition (IN POS), not set off by commas and not a quotative 'was like <adj/adv>' pattern"
    if prev_c == "," and next_c == ",":
        return True, "discourse_filler", 0.9, "comma-bracketed parenthetical 'like' — discourse-marker position"
    if prev_c == "," or next_c == ",":
        return True, "discourse_filler", 0.72, "adjacent to a comma with no verb/preposition usage — likely a discourse marker"
    # Quotative/hedge "was/is/were like <adjective/adverb>" — e.g. "I was
    # like really confused" — no commas at all in natural continuous
    # speech, but the copula-before + adjective/adverb-after pattern is a
    # reliable signal LanguageTool's POS tags already expose without
    # needing punctuation.
    if prev_tok and prev_tok.lower in ("was", "is", "were", "are", "am", "been", "being", "be") \
            and next_pos in ("RB", "JJ"):
        return True, "discourse_filler", 0.75, "quotative/hedge use after a copula ('was/is like') followed by an adjective/adverb, not a noun phrase — no comma needed for this pattern"
    if not pos:
        return False, None, 0.5, "no POS data available to disambiguate; defaulting to non-filler"
    return False, None, 0.55, "no discourse-marker signal (comma-bracketing, sentence-initial+pronoun, or copula+adjective pattern) found"


def _classify_mean(tokens: list[_Tok], i: int, transcript: str):
    tok = tokens[i]
    prev_tok = tokens[i - 1] if i > 0 else None
    next_c = _next_nonspace_char(transcript, tok.end)

    if not (prev_tok and prev_tok.lower == "i"):
        return False, None, 0.6, "not part of an 'I mean' construction"
    if next_c == ",":
        return True, "discourse_filler", 0.8, "'I mean,' followed by a pause — hedging usage rather than 'I mean X'"
    return False, None, 0.6, "'I mean' followed directly by its object — literal usage"


def _classify_so(tokens: list[_Tok], i: int, transcript: str):
    tok = tokens[i]
    next_c = _next_nonspace_char(transcript, tok.end)
    sentence_initial = _is_sentence_initial(tok, transcript)

    if next_c == "," and sentence_initial:
        return True, "discourse_filler", 0.65, "sentence-initial 'so,' set off by a comma — hedge rather than logical connective"
    return False, None, 0.65, "functioning as a connective/adverb ('therefore'/'to that extent'), not set off as a hedge"


def _classify_well(tokens: list[_Tok], i: int, transcript: str):
    # "Well," opening a response ("Well, this is the answer.") is
    # conventional discourse structure, not a hesitation marker on its own —
    # only flag it when doubled or adjacent to another filler, which the
    # phrase/repetition passes already catch. Kept as an explicit no-op
    # classifier (rather than omitted) to document that decision.
    return False, None, 0.55, "sentence-initial 'well' introducing a response — conventional usage, not treated as a filler on its own"


def _classify_bracketed_adverb(word_label: str) -> _ClassifyFn:
    def _classify(tokens: list[_Tok], i: int, transcript: str):
        tok = tokens[i]
        prev_c = _prev_nonspace_char(transcript, tok.start)
        next_c = _next_nonspace_char(transcript, tok.end)
        sentence_initial = _is_sentence_initial(tok, transcript)
        # A parenthetical use only needs a comma on ONE side to be
        # legitimately "set off" — mid-sentence it's usually bracketed on
        # both sides ("we, actually, started"), but sentence-initial it
        # structurally CANNOT have a preceding comma (there's no token
        # before it), so requiring both sides made this branch unreachable
        # for the single most common real pattern ("Actually, we started
        # yesterday.").
        if prev_c == "," or next_c == ",":
            return True, "discourse_filler", 0.68, f"comma-set-off parenthetical '{word_label}'"
        # Real (especially ASR) speech transcripts frequently omit the
        # comma entirely. A sentence-initial hedge adverb opening an
        # utterance ("Actually we was planning...", "So basically I told
        # them...") is functioning the same way whether or not a comma
        # happens to be transcribed — only mid-sentence, un-bracketed use
        # ("The result is basically correct.") stays non-filler.
        if sentence_initial:
            return True, "discourse_filler", 0.6, f"sentence-initial '{word_label}' opening the utterance — hedge usage even without a transcribed comma"
        return False, None, 0.55, f"'{word_label}' functioning as an ordinary adverb, not set off as parenthetical and not sentence-initial"
    return _classify


def _classify_right(tokens: list[_Tok], i: int, transcript: str):
    tok = tokens[i]
    prev_c = _prev_nonspace_char(transcript, tok.start)
    next_c = _next_nonspace_char(transcript, tok.end)
    if next_c == "?":
        return True, "discourse_filler", 0.72, "tag-question 'right?' seeking agreement — hedge usage"
    if prev_c == "," and next_c in (",", "."):
        return True, "discourse_filler", 0.6, "comma-set-off 'right' functioning as a filler/agreement marker"
    return False, None, 0.6, "used as an adjective ('correct') or direction, not a hedge marker"


_SINGLE_WORD_CLASSIFIERS: dict[str, _ClassifyFn] = {
    "like": _classify_like,
    "mean": _classify_mean,
    "so": _classify_so,
    "well": _classify_well,
    "actually": _classify_bracketed_adverb("actually"),
    "basically": _classify_bracketed_adverb("basically"),
    "just": _classify_bracketed_adverb("just"),
    "right": _classify_right,
}


# ── Multi-word phrase fillers ────────────────────────────────────────────────

def _scan_phrases(tokens: list[_Tok], transcript: str, occurrences: list[dict], used: set[int]) -> None:
    i = 0
    n = len(tokens)
    while i < n - 1:
        a, b = tokens[i], tokens[i + 1]

        if a.lower == "you" and b.lower == "know":
            next_c = _next_nonspace_char(transcript, b.end)
            next_tok = tokens[i + 2] if i + 2 < n else None
            next2_tok = tokens[i + 3] if i + 3 < n else None
            next_pos = (next_tok.pos or "").upper() if next_tok else ""
            next2_pos = (next2_tok.pos or "").upper() if next2_tok else ""
            # "you know the answer" (direct object follows) is literal
            # knowledge, not a hedge; "you know," / "you know." / end of
            # utterance is the discourse-marker use.
            if next_c in (",", ".", "!", "?", ""):
                occurrences.append(_occ(
                    [a, b], "discourse_filler", 0.82,
                    "'you know' with no object following — hedge/filler usage, not literal knowledge",
                ))
                used.update({i, i + 1})
                i += 2
                continue
            # No punctuation at all (common in continuous ASR transcripts),
            # but "you know" is immediately followed by a new subject +
            # verb ("you know I have a meeting") rather than a noun-phrase
            # object of "know" ("you know the answer" / "you know him") —
            # that pattern means "know" isn't taking what follows as its
            # object, so this is the hedge use too.
            if next_pos == "PRP" and next2_pos.startswith("VB"):
                occurrences.append(_occ(
                    [a, b], "discourse_filler", 0.68,
                    "'you know' immediately followed by a new subject+verb clause, not a noun-phrase object of 'know' — hedge usage even without a transcribed pause",
                ))
                used.update({i, i + 1})
                i += 2
                continue

        if a.lower == "so" and b.lower == "basically":
            # A fixed two-word hedge phrase in its own right (distinct from
            # the standalone 'so'/'basically' classifiers, which each
            # require their own comma or sentence-initial signal that a
            # combined "so basically I ..." opener won't individually
            # satisfy — "so" isn't set off by a comma and "basically" isn't
            # the first word of the sentence). Treated as filler whenever
            # the two words are adjacent, matching how filled pauses are
            # handled; documented limitation: a genuine non-hedge use of
            # this exact adjacency is not expected in ordinary speech.
            occurrences.append(_occ(
                [a, b], "discourse_filler", 0.75,
                "'so basically' — fixed two-word hedge phrase, treated as filler regardless of surrounding punctuation",
            ))
            used.update({i, i + 1})
            i += 2
            continue

        if a.lower in ("kind", "sort") and b.lower == "of" and i + 2 < n:
            nxt = tokens[i + 2]
            pos_next = (nxt.pos or "").upper()
            if pos_next.startswith("NN"):
                # e.g. "kind of animal" / "sort of cake" — literal partitive use
                i += 1
                continue
            occurrences.append(_occ(
                [a, b], "discourse_filler", 0.65,
                f"'{a.text.lower()} of' used as a hedging downtoner before a non-noun",
            ))
            used.update({i, i + 1})
            i += 2
            continue

        i += 1


# ── Repetition / hesitation detection (separate from fillers) ───────────────
# Matches an immediately repeated word, optionally separated by a short run
# of pause punctuation ("...", ",") — "I I", "the the", "I... I...".

_REPEAT_RE = re.compile(r"\b(\w+)\b([.,]{0,3}\s+)\1\b", re.IGNORECASE)


def _detect_hesitations(transcript: str) -> list[dict]:
    hesitations = []
    for m in _REPEAT_RE.finditer(transcript):
        hesitations.append({
            "phrase": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "type": "repetition",
            "confidence": 0.8,
            "reason": "immediate word repetition — hesitation/self-correction signal, not counted as a filler word",
        })
    return hesitations


# ── Public API ────────────────────────────────────────────────────────────

def _surrounding_text(tokens: list[_Tok], i: int, window: int = 4) -> str:
    lo = max(0, i - window)
    hi = min(len(tokens), i + window + 1)
    return " ".join(t.text for t in tokens[lo:hi])


def _punct_context(tok: _Tok, transcript: str) -> str:
    return f"prev={_prev_nonspace_char(transcript, tok.start)!r} next={_next_nonspace_char(transcript, tok.end)!r}"


def detect_fillers(transcript: str, linguistic_analysis: Optional[dict] = None,
                    duration_seconds: Optional[float] = None, debug: bool = False) -> dict:
    """
    Context-aware filler detection.

    Args:
        transcript: the full transcript text.
        linguistic_analysis: the `/v2/analyze` response dict from
            LanguageToolProvider.analyze() / check_and_analyze() for this
            same transcript, or None if it wasn't available. When present,
            it's used for POS/lemma-based disambiguation; when absent, only
            the POS-independent rules (filled pauses, phrase fillers,
            punctuation-only rules) are applied.
        duration_seconds: recording duration, for rate_per_min. None/0
            yields `rate_per_min: None` (matches text-only callers that
            have no audio duration, e.g. the debug text-analysis route).
        debug: when True, prints one line per candidate (accepted AND
            rejected) in the form:
                Candidate: <text>
                POS: <tag>
                Context: "<surrounding text>"
                Decision: FILLER | NOT_FILLER
                Reason: <why>
            and includes the same records under the returned "debug" key.
            False by default — no behavior/shape change for existing
            callers (app.py does not pass this).

    Returns:
        {
          "count": int,
          "rate_per_min": float | None,
          "occurrences": [ {word, start, end, type, confidence, reason}, ... ],
          "hesitations": [ {phrase, start, end, type, confidence, reason}, ... ],
          "debug": [ {...} , ... ],   # only present when debug=True
        }
    """
    debug_log: list[dict] = []

    def _log(candidate_tokens: list[_Tok], pos: str, decision: bool, reason: str) -> None:
        if not debug:
            return
        text = " ".join(t.text for t in candidate_tokens)
        i0 = tokens.index(candidate_tokens[0])
        entry = {
            "candidate": text,
            "start": candidate_tokens[0].start,
            "end": candidate_tokens[-1].end,
            "pos": pos,
            "lemma": candidate_tokens[0].lemma,
            "surrounding": _surrounding_text(tokens, i0),
            "punctuation": _punct_context(candidate_tokens[0], transcript),
            "decision": "FILLER" if decision else "NOT_FILLER",
            "reason": reason,
        }
        debug_log.append(entry)
        print(f"Candidate: {text}")
        print(f"POS: {pos or '(none)'}")
        print(f'Context: "{entry["surrounding"]}"')
        print(f"Decision: {entry['decision']}")
        print(f"Reason: {reason}")
        print()

    if not transcript or not transcript.strip():
        result = {"count": 0, "rate_per_min": 0.0 if duration_seconds else None,
                   "occurrences": [], "hesitations": []}
        if debug:
            result["debug"] = []
        return result

    tokens = _build_tokens(transcript, linguistic_analysis)
    occurrences: list[dict] = []
    used: set[int] = set()

    _scan_phrases(tokens, transcript, occurrences, used)
    if debug:
        for occ in occurrences:
            debug_log.append({
                "candidate": occ["word"], "start": occ["start"], "end": occ["end"],
                "pos": None, "lemma": None, "surrounding": occ["word"],
                "punctuation": None, "decision": "FILLER", "reason": occ["reason"],
            })
            print(f"Candidate: {occ['word']}")
            print("POS: (phrase match)")
            print(f"Decision: FILLER")
            print(f"Reason: {occ['reason']}")
            print()

    for i, tok in enumerate(tokens):
        if i in used:
            continue
        w = tok.lower
        if w in FILLED_PAUSES:
            occurrences.append(_occ(
                [tok], "filled_pause", 0.97,
                "standalone hesitation marker (uh/um/er class) — treated as filler regardless of context",
            ))
            _log([tok], tok.pos or "", True, "standalone hesitation marker (uh/um/er class) — treated as filler regardless of context")
            continue
        classifier = _SINGLE_WORD_CLASSIFIERS.get(w)
        if classifier is None:
            continue
        is_filler, ftype, confidence, reason = classifier(tokens, i, transcript)
        _log([tok], tok.pos or "", is_filler, reason)
        if is_filler:
            occurrences.append(_occ([tok], ftype, confidence, reason))

    occurrences.sort(key=lambda o: o["start"])
    count = len(occurrences)

    rate_per_min = None
    if duration_seconds and duration_seconds > 0:
        rate_per_min = round(count / (duration_seconds / 60), 1)

    result = {
        "count": count,
        "rate_per_min": rate_per_min,
        "occurrences": occurrences,
        "hesitations": _detect_hesitations(transcript),
    }
    if debug:
        result["debug"] = debug_log
    return result


def summarize_words(occurrences: list[dict]) -> list[str]:
    """`["uh×2", "like×1"]`-style summary — matches the shape app.py's
    existing `filler["words"]` field / build_feedback() already expect, so
    the richer `occurrences` list can be added without changing that
    existing contract."""
    counts = Counter(o["word"].lower() for o in occurrences)
    return [f"{word}×{n}" for word, n in counts.items()]
