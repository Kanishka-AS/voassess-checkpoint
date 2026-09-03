"""
Contextual grammar-issue validation layer (Groq).

────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS

VOAssess is a SPOKEN-English assessment tool. Its deterministic grammar
detectors — LanguageTool (languagetool_provider.py), the learner-error regex
heuristics (grammar_heuristics.py), and the POS-aware rules
(grammar_pos_rules.py) — are proofreading tools built for WRITTEN text. They
regularly flag things that are simply not grammar mistakes when someone is
*speaking*:

    "forty one"  -> "forty-one"   (hyphenation is a spelling convention;
                                    you cannot hear a hyphen)
    "gonna"      -> "going to"    (a normal informal spoken contraction,
                                    not a grammatical error)

Before this module existed, every candidate those tools produced was
counted as a real grammar error and folded straight into the grammar score
and the learner-facing issue list. That penalizes learners for transcript-
formatting artifacts and written-only orthography, not for anything they
actually got wrong when speaking.

This module does NOT replace or re-implement LanguageTool/the heuristics —
they remain exactly as they are, running exactly as before, as candidate
detectors (see grammar_heuristics.py's own docstring: "high precision over
high recall"). This module is a second, independent pass that takes their
output and asks a strict-but-fair judgment question for every candidate,
using Groq's hosted LLM as the reasoning layer:

    "In THIS sentence, spoken by a learner, is this candidate actually a
     spoken-grammar mistake — or is it a written-only/orthography/style/
     register artifact, or not an error at all?"

Classification taxonomy (exactly one per candidate):
    true_grammar_error   -> a genuine grammar mistake, relevant whether
                             spoken or written. Counts toward the grammar
                             score and the learner-facing issue count.
    spoken_usage_issue    -> worth a note for a spoken-English learner, but
                             not a hard grammar-rule violation.
    written_only_issue   -> purely an orthography/transcript-formatting
                             convention (hyphenation, punctuation,
                             capitalization, spacing/compounding). Not
                             something a listener could ever "hear".
    style_or_register     -> informal-but-legitimate spoken forms (gonna,
                             wanna, kinda, ...) or a tone/register choice.
    not_an_error           -> the rule engine's candidate doesn't actually
                             apply here (false positive).

ONLY "true_grammar_error" counts toward the grammar score / error count.
Everything else is preserved as a "context note" — visible for debugging
and, where useful, as a written-English aside — but it never reduces the
grammar score and never appears in the learner's spoken grammar mistake
count.

Hard boundaries this module enforces:
  - It NEVER invents a new correction. `wrong`/`correct` always come from
    the candidate exactly as the deterministic tool produced them — the
    model only judges relevance and writes a contextual explanation, it
    does not get to propose its own fix.
  - It NEVER adds a grammar issue that wasn't already a candidate from
    LanguageTool/grammar_heuristics/grammar_pos_rules. This is a filter/
    re-labeling layer, not a new detector.
  - If GROQ_API_KEY is not set, the request fails, or the response isn't
    parseable, this module falls back to a small, transparent, offline
    heuristic classifier (see classify_candidate_heuristic()) rather than
    silently defaulting every candidate to "true_grammar_error" again —
    that heuristic is what already fixes the "forty one" / "gonna" cases
    even with no LLM configured. `validation_source` on the result always
    says which path actually produced the judgments being served
    ("llm_groq" or "heuristic_fallback"), so nothing is silently degraded
    without a trace.

Configuration (reuses the same Groq env vars as groq_provider.py, plus an
optional dedicated override so the two Groq call sites can be pointed at
different models/deployments independently if ever needed):
    GROQ_API_KEY               Required for the LLM path. Groq API key.
    GROQ_API_URL               Optional. Defaults to https://api.groq.com/openai/v1
    GROQ_GRAMMAR_MODEL          Optional. Defaults to GROQ_MODEL, then to
                               "llama-3.3-70b-versatile".
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_GRAMMAR_MODEL = os.environ.get(
    "GROQ_GRAMMAR_MODEL", os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
)
DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=4.0)

CLASSIFICATIONS = (
    "true_grammar_error",
    "spoken_usage_issue",
    "written_only_issue",
    "style_or_register",
    "not_an_error",
)
SCORING_CLASSIFICATION = "true_grammar_error"


# ── Offline heuristic classifier (fallback, and a sanity floor for the LLM) ──
# Deliberately narrow and high-precision, same design philosophy as
# grammar_heuristics.py: it only reclassifies candidates it can be CONFIDENT
# are written-only or register/style issues; everything else defaults to
# true_grammar_error, exactly as the pipeline behaved before this module
# existed, so this never silently makes the pipeline more lenient than it
# used to be — it only removes the specific, well-understood false
# positives spoken-English assessment should never have counted.

_INFORMAL_CONTRACTIONS = {
    "gonna": "going to", "wanna": "want to", "gotta": "got to / have got to",
    "kinda": "kind of", "sorta": "sort of", "dunno": "don't know",
    "lemme": "let me", "gimme": "give me", "outta": "out of",
    "y'all": "you all", "ain't": "am not / is not / are not",
    "cuz": "because", "cause": "because", "coulda": "could have",
    "shoulda": "should have", "woulda": "would have", "kinda-": "kind of",
}

_WRITTEN_ONLY_TEXT_HINTS = (
    "hyphen", "hyphenat", "capitaliz", "capitalis", "punctuat", "comma",
    "apostrophe", "casing", "compound word", "spelled with", "spelling of",
    "spaced", "one word or two", "should be written as one word",
)
_WRITTEN_ONLY_FIELD_HINTS = (
    "punctuation", "casing", "typograph", "compound", "hyphen", "whitespace",
)

_WORD_ONLY_RE = re.compile(r"[a-z']+")

# Spec section 3 / 12 (case 1): "check check", "the the" — an immediate,
# exact repetition of one word. In SPOKEN English this is overwhelmingly a
# self-correction or disfluency (the speaker re-starting the word), not a
# grammar mistake, so — like the informal-contraction list above — it gets
# its own explicit, high-precision heuristic rather than falling through to
# the "true_grammar_error" safe default. This only matches an EXACT,
# adjacent, single-word duplication (not any two-word repeated-word rule
# match in general), which keeps it narrow enough not to mask a genuine
# duplication-shaped grammar error (e.g. "I want want to go" still isn't
# covered here since the intent is caution, not blanket exemption — the LLM
# path still sees the real context and can override either way).
_REPEAT_TEXT_HINTS = ("repeated a word", "repeated word", "word repetition",
                      "duplicate word", "you repeated")
_REPEAT_FIELD_HINTS = ("repeat", "duplicat")


def _is_immediate_word_repeat(wrong: str) -> bool:
    tokens = _WORD_ONLY_RE.findall((wrong or "").lower())
    return len(tokens) == 2 and tokens[0] == tokens[1]


def _norm_collapsed(s: str) -> str:
    """Lowercase and strip everything except letters/apostrophes, so
    'forty one' / 'forty-one' / 'forty—one' all collapse to 'fortyone'."""
    return re.sub(r"[^a-z']", "", (s or "").lower())


def _norm_word_token(s: str) -> str:
    s = (s or "").strip().lower()
    return s.strip(".,!?;:()[]\"'\u2014-")


def classify_candidate_heuristic(candidate: dict) -> dict:
    """Pure-Python, offline, deterministic classification for one candidate
    issue dict (wrong/correct/message/context/rule_id/category). Used as
    the fallback path when Groq is unavailable, and also surfaced to the
    LLM as a non-authoritative hint (see build_validation_prompt()) so the
    model has a starting point it's free to override with real context.

    Returns {"classification": ..., "reason": "..."}."""
    wrong = candidate.get("wrong") or ""
    correct = candidate.get("correct") or ""
    message = (candidate.get("message") or "").lower()
    category = (candidate.get("category") or "").lower()
    rule_id = (candidate.get("rule_id") or "").lower()

    if _is_immediate_word_repeat(wrong) or \
       any(h in message for h in _REPEAT_TEXT_HINTS) or \
       any(h in category for h in _REPEAT_FIELD_HINTS) or \
       any(h in rule_id for h in _REPEAT_FIELD_HINTS):
        return {
            "classification": "spoken_usage_issue",
            "reason": f"'{wrong}' is an immediate repetition of one word — in spoken "
                      f"English that's typically a self-correction or disfluency "
                      f"(the speaker re-starting the word), not a grammar-rule "
                      f"violation, so it isn't automatically counted as one.",
        }

    wrong_token = _norm_word_token(wrong)
    if wrong_token in _INFORMAL_CONTRACTIONS:
        expansion = _INFORMAL_CONTRACTIONS[wrong_token]
        return {
            "classification": "style_or_register",
            "reason": f"'{wrong_token}' is a common informal spoken contraction of "
                      f"'{expansion}', not a grammar mistake.",
        }

    if wrong and correct and _norm_collapsed(wrong) == _norm_collapsed(correct):
        return {
            "classification": "written_only_issue",
            "reason": f"'{wrong}' vs '{correct}' differ only in spelling/hyphenation/"
                      f"spacing — the spoken words are identical, so there is nothing "
                      f"a listener could have gotten wrong.",
        }

    if any(h in message for h in _WRITTEN_ONLY_TEXT_HINTS) or \
       any(h in category for h in _WRITTEN_ONLY_FIELD_HINTS) or \
       any(h in rule_id for h in _WRITTEN_ONLY_FIELD_HINTS):
        return {
            "classification": "written_only_issue",
            "reason": "The rule engine's own message/category identifies this as a "
                      "punctuation/casing/hyphenation/spacing convention, which is a "
                      "written-English-only distinction.",
        }

    return {
        "classification": "true_grammar_error",
        "reason": "No written-only or informal-register pattern matched; treated as "
                  "a genuine grammar issue (safe default).",
    }


def _heuristic_learner_explanation(candidate: dict, classification: str, reason: str) -> tuple[str, Optional[str]]:
    """Builds (learner_explanation, written_note) for the heuristic fallback
    path, in the same grounded, no-invented-facts spirit as
    groq_provider.py's _fallback_grammar_breakdown()."""
    wrong = candidate.get("wrong") or ""
    correct = candidate.get("correct") or ""

    if classification == "written_only_issue":
        explanation = (
            f"Saying \"{wrong}\" out loud is completely correct — {reason} "
            f"This is not counted as a spoken grammar mistake."
        )
        written_note = (
            f"If you're writing this down, the standard written form is \"{correct}\"."
            if correct else None
        )
        return explanation, written_note

    if classification == "style_or_register":
        explanation = (
            f"\"{wrong}\" is natural, correct spoken English — {reason} "
            f"It's a register choice, not a grammar mistake, so it's not counted here."
        )
        return explanation, None

    if classification == "spoken_usage_issue":
        explanation = (
            f"\"{wrong}\" isn't a hard grammar-rule violation, but it's worth noticing "
            f"in spoken delivery: {candidate.get('message') or reason}"
        )
        return explanation, None

    if classification == "not_an_error":
        return "On closer look, this isn't actually an error in this sentence.", None

    # true_grammar_error
    if wrong and correct:
        explanation = (
            f"You said \"{wrong}\", but the correct spoken form here is \"{correct}\". "
            f"{candidate.get('message') or ''}".strip()
        )
    else:
        explanation = candidate.get("message") or "This is a genuine grammar issue in this sentence."
    return explanation, None


# ── Result shape ──────────────────────────────────────────────────────────

@dataclass
class GrammarContextValidationResult:
    """`learner_facing_issues` is the ONLY list that should ever feed the
    grammar score / learner-facing "grammar mistakes" count downstream —
    it contains exactly the candidates judged `true_grammar_error`, each
    carrying the same wrong/correct/message/context/rule_id/category
    fields the candidate always had (untouched, for backward
    compatibility) PLUS `classification`/`learner_explanation`.

    `context_notes` holds every other candidate (spoken_usage_issue /
    written_only_issue / style_or_register / not_an_error) — never scored,
    always kept for transparency/debugging and for an optional
    "written-English notes" UI section.

    `validated_issues` is every candidate, in original order, with its
    judgment attached — the full evidence trail for debugging: candidate
    (as detected) + judgment (LLM or heuristic) in one place."""

    available: bool
    validation_source: str  # "llm_groq" | "heuristic_fallback" | "no_candidates"
    validated_issues: list = field(default_factory=list)
    learner_facing_issues: list = field(default_factory=list)
    context_notes: list = field(default_factory=list)
    true_error_count: int = 0
    model: Optional[str] = None
    raw_text: Optional[str] = None
    prompt: Optional[list] = field(default=None, repr=False)
    detail: Optional[str] = None


# ── Prompt ──────────────────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = """
You are a contextual grammar-validation layer inside a SPOKEN-English
(not written) assessment tool. A deterministic rule engine (LanguageTool
plus hand-written learner-error heuristics) has already produced a list of
CANDIDATE grammar issues from a transcript of something a learner SAID
out loud. Those tools are built for proofreading written text, so they
regularly flag things that are not actually spoken-grammar mistakes.

Your job: for EVERY candidate, read the full transcript and the
candidate's own sentence context, then classify it into EXACTLY ONE of:

  - "true_grammar_error": a genuine grammar mistake, relevant whether
    spoken or written (wrong verb tense, subject-verb agreement, wrong
    preposition, missing article, etc).
  - "spoken_usage_issue": worth a note for a spoken-English learner, but
    not a hard grammar-rule violation.
  - "written_only_issue": purely a spelling/punctuation/hyphenation/
    capitalization/compounding convention — something that could never be
    "heard" in speech (e.g. "forty one" vs "forty-one").
  - "style_or_register": an informal-but-legitimate spoken form (e.g.
    "gonna", "wanna", "kinda") or a tone/register choice, not an error.
  - "not_an_error": the candidate is simply wrong here — a false positive.

Absolute rules:
1. Only "true_grammar_error" may ever be treated as a scoring grammar
   mistake. Every other label must NOT be counted against the learner.
2. Do NOT penalize spoken learners for punctuation, hyphenation,
   capitalization, or other transcript-formatting artifacts unless the
   candidate is genuinely a spoken-grammar problem in its own right.
3. Informal spoken forms ("gonna", "wanna", "kinda", "gotta", etc.) are
   NOT automatically grammar errors — classify them as "style_or_register"
   unless the surrounding sentence has an actual, separate grammar problem.
3b. An immediate, exact repetition of one word (e.g. "check check",
   "the the") is NOT automatically a grammar error — it is typically a
   spoken self-correction or disfluency. Classify it as
   "spoken_usage_issue" unless the repetition itself is embedded in a
   separate, genuine grammar problem.
4. NEVER invent a new correction. If you reference a fix, it must be
   exactly the candidate's own "correct" value — you are validating and
   contextualizing an existing candidate, not creating a new one.
5. Write learner_explanation the way a good teacher would talk to the
   student about THIS sentence — specific, plain, and immediately
   clear about what happened and why it does or doesn't matter. Never
   describe it as "a rule violation" or reference rule IDs/categories.
6. A `pattern_hint` is included with each candidate from a simple offline
   heuristic. It is NOT authoritative — override it whenever the actual
   sentence context justifies a different classification.

Respond with ONLY a single JSON object, no markdown fences, no commentary,
matching exactly this shape:

{
  "judgments": [
    {
      "id": <the candidate's integer id, copied exactly>,
      "classification": "<one of the five labels above>",
      "learner_explanation": "<1-3 sentences, teacher voice, specific to this transcript>",
      "written_note": "<short written-English-only aside if genuinely useful, else null>"
    }
  ]
}

Include exactly one judgment object per candidate id given to you.
""".strip()


def build_validation_prompt(transcript: str, candidates: list) -> list:
    payload = []
    for idx, cand in enumerate(candidates):
        hint = classify_candidate_heuristic(cand)
        payload.append({
            "id": idx,
            "wrong": cand.get("wrong"),
            "correct": cand.get("correct"),
            "message": cand.get("message"),
            "context": cand.get("context"),
            "rule_id": cand.get("rule_id"),
            "category": cand.get("category"),
            "source": cand.get("source", "languagetool"),
            "pattern_hint": hint["classification"],
        })

    user = (
        "FULL TRANSCRIPT (verbatim, as spoken):\n"
        f"{transcript or '(empty transcript)'}\n\n"
        "CANDIDATE GRAMMAR ISSUES (from the deterministic rule engine — JSON):\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )

    return [
        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ── Groq client ─────────────────────────────────────────────────────────────

class GroqGrammarValidator:
    """Thin, isolated Groq client for this validation step — mirrors the
    provider pattern used elsewhere in the project (languagetool_provider,
    groq_provider): synchronous httpx, never raises for a merely-
    unavailable/failed call, returns a normalized result instead."""

    def __init__(self, api_key: str | None = None, base_url: str = GROQ_API_URL,
                 model: str = GROQ_GRAMMAR_MODEL, timeout: httpx.Timeout = DEFAULT_TIMEOUT):
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def call(self, transcript: str, candidates: list) -> tuple[Optional[dict], Optional[list], Optional[str]]:
        """Returns (parsed_json_or_None, prompt, error_detail_or_None)."""
        messages = build_validation_prompt(transcript, candidates)
        if not self.is_available():
            return None, messages, "Groq is not configured (GROQ_API_KEY is not set)."

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            try:
                reason = e.response.json().get("error", {}).get("message") or e.response.text
            except Exception:
                reason = e.response.text
            return None, messages, f"Groq request failed ({e.response.status_code}): {reason}"
        except Exception as e:
            return None, messages, f"Groq request failed: {e}"

        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None, messages, "Groq response did not contain the expected message content."

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            return None, messages, f"Groq response was not valid JSON: {e}"

        parsed["_raw_text"] = raw_text
        return parsed, messages, None


_validator = GroqGrammarValidator()


def _build_learner_facing_entry(candidate: dict, classification: str,
                                 learner_explanation: str) -> dict:
    entry = dict(candidate)  # preserve every original field for debugging/backward-compat
    entry["classification"] = classification
    entry["spoken_relevant"] = True
    entry["counts_toward_grammar_score"] = True
    entry["learner_explanation"] = learner_explanation
    return entry


def _build_context_note(candidate: dict, classification: str,
                         learner_explanation: str, written_note: Optional[str]) -> dict:
    return {
        "wrong": candidate.get("wrong"),
        "correct": candidate.get("correct"),
        "context": candidate.get("context"),
        "rule_engine_message": candidate.get("message"),
        "rule_id": candidate.get("rule_id"),
        "category": candidate.get("category"),
        "classification": classification,
        "spoken_relevant": False,
        "counts_toward_grammar_score": False,
        "learner_explanation": learner_explanation,
        "written_note": written_note,
    }


def apply_contextual_validation(transcript: str, ge: int, grammar_issues: list,
                                 validator: Optional[GroqGrammarValidator] = None) -> tuple:
    """The exact integration point app.py:resolve_grammar() calls, pulled
    out into this module (rather than left inline in app.py) so it can be
    exercised directly in tests without importing app.py's whole dependency
    chain (Whisper, allosaurus, etc.).

    Takes the (ge, grammar_issues) pair exactly as produced by
    augment_grammar_issues() — the combined LanguageTool + heuristic
    candidate list — and returns the POST-VALIDATION
    (ge_validated, grammar_issues_validated, grammar_context) triple:

      - ge_validated / grammar_issues_validated: safe drop-in replacements
        for `ge`/`grammar_issues` — only candidates judged
        `true_grammar_error` remain, so callers that don't even look at
        `grammar_context` still get the corrected score/issue list for
        free.
      - grammar_context: additive debug/notes dict — candidates evaluated,
        how many were reclassified away from scoring, which validation
        path actually ran, the non-scoring context notes, and the full
        candidate+judgment evidence trail.
    """
    validation = validate_grammar_context(transcript, grammar_issues, validator=validator)

    if not grammar_issues:
        return ge, grammar_issues, {
            "candidates_evaluated": 0,
            "reclassified_away_from_score": 0,
            "validation_source": validation.validation_source,
            "model": validation.model,
            "detail": validation.detail,
            "context_notes": [],
            "debug_trail": [],
        }

    # `ge` can legitimately exceed len(grammar_issues) (LanguageTool's own
    # `errors` count includes matches filtered out of the issues list, or
    # matches beyond the per-response issues cap — see
    # LanguageToolProvider.check_grammar / augment_grammar_issues). Only the
    # visible candidates were validated, so `ge` is only ever corrected by
    # however many of THOSE were reclassified away — never a guess about
    # candidates we couldn't see.
    reclassified_away = len(grammar_issues) - len(validation.learner_facing_issues)
    ge_validated = max(ge - reclassified_away, 0)
    grammar_issues_validated = validation.learner_facing_issues

    grammar_context = {
        "candidates_evaluated": len(grammar_issues),
        "reclassified_away_from_score": reclassified_away,
        "validation_source": validation.validation_source,
        "model": validation.model,
        "detail": validation.detail,
        "context_notes": validation.context_notes,
        "debug_trail": validation.validated_issues,
    }

    return ge_validated, grammar_issues_validated, grammar_context


def validate_grammar_context(transcript: str, candidates: list,
                              validator: Optional[GroqGrammarValidator] = None) -> GrammarContextValidationResult:
    """Main entry point. `candidates` is the combined LanguageTool +
    grammar_heuristics/grammar_pos_rules issue list exactly as
    resolve_grammar() already produces it (app.py) — nothing here changes
    what gets detected, only how each candidate is judged and counted.
    """
    if not candidates:
        return GrammarContextValidationResult(available=True, validation_source="no_candidates")

    v = validator or _validator
    parsed, prompt, error = v.call(transcript, candidates)

    judgments_by_id: dict[int, dict] = {}
    validation_source = "heuristic_fallback"
    raw_text = None
    detail = error

    if parsed is not None:
        raw_text = parsed.get("_raw_text")
        judgments = parsed.get("judgments")
        if isinstance(judgments, list):
            for j in judgments:
                try:
                    jid = int(j.get("id"))
                except (TypeError, ValueError):
                    continue
                cls = j.get("classification")
                if cls not in CLASSIFICATIONS:
                    continue  # invalid label from the model — fall back to heuristic for this one
                judgments_by_id[jid] = {
                    "classification": cls,
                    "learner_explanation": j.get("learner_explanation") or "",
                    "written_note": j.get("written_note"),
                }
            if judgments_by_id:
                validation_source = "llm_groq"
        if not judgments_by_id:
            detail = detail or "Groq response had no usable judgments; used heuristic fallback."

    validated_issues, learner_facing, context_notes = [], [], []
    for idx, cand in enumerate(candidates):
        judgment = judgments_by_id.get(idx)
        if judgment is None:
            h = classify_candidate_heuristic(cand)
            explanation, written_note = _heuristic_learner_explanation(
                cand, h["classification"], h["reason"])
            judgment = {
                "classification": h["classification"],
                "learner_explanation": explanation,
                "written_note": written_note,
            }

        classification = judgment["classification"]
        explanation = judgment["learner_explanation"] or ""
        written_note = judgment.get("written_note")

        if classification == SCORING_CLASSIFICATION:
            entry = _build_learner_facing_entry(cand, classification, explanation)
            learner_facing.append(entry)
        else:
            entry = _build_context_note(cand, classification, explanation, written_note)
            context_notes.append(entry)

        validated_issues.append({
            "candidate": dict(cand),
            "llm_judgment": {
                "classification": classification,
                "learner_explanation": explanation,
                "written_note": written_note,
                "source": "llm_groq" if idx in judgments_by_id else "heuristic_fallback",
            },
        })

    return GrammarContextValidationResult(
        available=True,
        validation_source=validation_source,
        validated_issues=validated_issues,
        learner_facing_issues=learner_facing,
        context_notes=context_notes,
        true_error_count=len(learner_facing),
        model=v.model if judgments_by_id else None,
        raw_text=raw_text,
        prompt=prompt,
        detail=detail,
    )
