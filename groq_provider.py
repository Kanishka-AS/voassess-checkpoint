"""
Groq-based "strict English teacher" final report generator.

────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE IS

The rest of the pipeline (stt_provider.py, pronunciation_provider.py,
grammar_heuristics.py / grammar_pos_rules.py / languagetool_provider.py,
vocabulary.py, filler_detector.py, audio_utils.py, and the scoring
functions in app.py) is the DETERMINISTIC ENGINE: it measures things and
produces evidence (scores, counts, issue lists, transcripts). None of that
is touched by this module.

This module is the second, separate step: it takes the deterministic
engine's own output (never raw audio, never a second transcription) and
asks Groq's hosted LLM to act as a strict English teacher, reading the
transcript alongside the evidence, and writing the polished, structured
report a mentor would produce by hand. It is purely a report-authoring
layer on top of numbers/evidence that already exist.

Hard boundaries this module enforces (see build_teacher_prompt() and
TeacherReportResult below):

  - Groq NEVER invents or changes a score, count, or statistic. Every
    number in the final report (grammar score, fluency score, vocabulary
    stats, filler count, CEFR level, etc.) comes from `evidence` verbatim,
    not from the model's own text.
  - The Accuracy / Fluency / Use-of-English High/Medium/Low rankings and
    their required rubric sentences (see RUBRIC below) are computed
    deterministically in Python from the project's own existing
    thresholds, never by the model. Groq may only add a short
    teacher-style elaboration alongside the fixed sentence — it cannot
    replace or contradict it.
  - Groq may identify genuine transcript-level English issues (wording,
    naturalness, an advanced construction actually present) that the
    deterministic providers didn't flag — but the prompt requires every
    such observation to quote/point at real transcript text, and the
    response is labeled as teacher observation, not deterministic-scorer
    output. If the model has nothing genuine to point to for a section
    (e.g. no advanced grammar constructions are actually present), it must
    say so rather than fabricate an example.
  - If GROQ_API_KEY is not set, or the request fails, or the model's
    response isn't valid/parseable JSON, this module returns
    `available=False` with a `detail` explaining why — callers (app.py)
    must degrade gracefully (keep serving the deterministic assessment
    exactly as before; the report is additive, not required).

Configuration (see .env.example):
    GROQ_API_KEY     Required. Groq API key.
    GROQ_API_URL     Optional. Defaults to https://api.groq.com/openai/v1
    GROQ_MODEL       Optional. Defaults to "llama-3.3-70b-versatile".
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import httpx

from vocabulary import is_function_word, _WORD_RE  # read-only reuse — no changes to vocabulary.py

GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


# ── Fixed rubric text (verbatim, per mentor's spec — never rewritten) ────────

ACCURACY_RUBRIC = {
    "High": "You made very few grammatical errors.",
    "Medium": "You made some grammatical errors, but your overall meaning was always clear.",
    "Low": "Grammar mistakes made it harder to follow parts of what you said.",
}
FLUENCY_RUBRIC = {
    "High": "Excellent! You had a good fluency speed without many hesitations.",
    "Medium": "You spoke at a steady pace, with a few pauses that broke up your flow.",
    "Low": "Frequent pauses and hesitations made your speech difficult to follow smoothly.",
}
USE_OF_ENGLISH_RUBRIC = {
    "High": "You had very clear pronunciation with only minor errors.",
    "Medium": "Your pronunciation was mostly clear, with a few sounds that need attention.",
    "Low": "Pronunciation clarity needs more practice, especially with key sounds.",
}


# ── Deterministic High/Medium/Low ranking ─────────────────────────────────────
# These reuse thresholds that already exist elsewhere in the project rather
# than inventing new ones:
#   - Accuracy reuses score_grammar()'s own bands, already surfaced in
#     app.py:build_feedback() (grammar_s < 56 -> issues call-out,
#     < 90 -> "minor", else -> "clean"). Same two cut points here.
#   - Use of English reuses build_feedback()'s existing pronunciation bands
#     (pronun_s >= 85 / >= 65).
#   - Fluency has no existing single-score High/Medium/Low band in the
#     project, so it is built from the three concrete metrics the mentor's
#     spec gives explicit thresholds for (speaking rate, hesitations +
#     repetitions, unexpected/long pauses), each banded independently, then
#     combined by simple majority (ties broken toward the lower band, i.e.
#     err toward "needs practice" rather than overstating fluency).

def rank_accuracy(grammar_score: Optional[float]) -> str:
    if grammar_score is None:
        return "Medium"
    if grammar_score >= 90:
        return "High"
    if grammar_score >= 56:
        return "Medium"
    return "Low"


def rank_use_of_english(pronunciation_score: Optional[float]) -> str:
    if pronunciation_score is None:
        return "Medium"
    if pronunciation_score >= 85:
        return "High"
    if pronunciation_score >= 65:
        return "Medium"
    return "Low"


def _band_speaking_rate(wpm: Optional[float]) -> Optional[str]:
    if wpm is None:
        return None
    if 120 <= wpm <= 160:
        return "High"
    if (90 <= wpm < 120) or (160 < wpm <= 180):
        return "Medium"
    return "Low"  # below 90 or above 180


def _band_hesitations_repetitions(count: Optional[int]) -> Optional[str]:
    if count is None:
        return None
    if count <= 2:
        return "High"
    if count <= 5:
        return "Medium"
    return "Low"  # 6+


def _band_unexpected_pauses(count: Optional[int]) -> Optional[str]:
    if count is None:
        return None
    if count <= 2:
        return "High"
    if count <= 4:
        return "Medium"
    return "Low"  # 5+


def rank_fluency(wpm: Optional[float], hesitations_repetitions: Optional[int],
                  unexpected_pauses: Optional[int]) -> dict:
    """Returns the overall band plus each component band, so the report can
    show its work (speaking rate / hesitations & repetitions / unexpected
    pauses) alongside the single Fluency ranking."""
    rate_band = _band_speaking_rate(wpm)
    hes_band = _band_hesitations_repetitions(hesitations_repetitions)
    pause_band = _band_unexpected_pauses(unexpected_pauses)

    order = {"Low": 0, "Medium": 1, "High": 2}
    bands = [b for b in (rate_band, hes_band, pause_band) if b is not None]
    if not bands:
        overall = "Medium"
    else:
        counts = Counter(bands)
        best = max(counts.values())
        tied = sorted([b for b, c in counts.items() if c == best], key=lambda b: order[b])
        overall = tied[0]  # tie -> lower band

    return {
        "overall": overall,
        "speaking_rate_band": rate_band,
        "hesitations_repetitions_band": hes_band,
        "unexpected_pauses_band": pause_band,
    }


# ── Repeated / overused words ─────────────────────────────────────────────────

def find_repeated_words(transcript: str, min_count: int = 2, top_n: int = 8) -> list:
    """Actual repeated content words from the transcript, with real
    frequency counts — grounding for REPETITIONS / GROWTH AREAS so Groq
    doesn't have to (and isn't asked to) invent repeated-word examples.
    Excludes function words (a/the/is/and/...) via vocabulary.py's own
    FUNCTION_WORDS list (imported, not duplicated)."""
    if not transcript or not transcript.strip():
        return []
    tokens = [m.group(0).lower() for m in _WORD_RE.finditer(transcript)]
    content = [t for t in tokens if not is_function_word(t)]
    counts = Counter(content)
    repeated = [{"word": w, "count": c} for w, c in counts.items() if c >= min_count]
    repeated.sort(key=lambda x: (-x["count"], x["word"]))
    return repeated[:top_n]


# ── Evidence assembly ──────────────────────────────────────────────────────

def build_locked_rubric(evidence: dict) -> dict:
    """The parts of the final report that are computed in Python and are
    NEVER to be altered by the model — scores, rankings, and the fixed
    rubric sentence for each ranking. Passed to Groq as read-only context
    and re-merged into the final report after the model responds, so even
    if the model ignored the instruction, the served report still carries
    the correct locked values."""
    grammar_score = evidence.get("grammar", {}).get("score")
    pronun_score = evidence.get("pronunciation", {}).get("score")
    fluency = evidence.get("fluency", {}) or {}
    pace = evidence.get("pace", {}) or {}
    filler = evidence.get("filler", {}) or {}

    hesitations_repetitions = None
    if fluency.get("hesitation_count") is not None or filler.get("count") is not None:
        hesitations_repetitions = (fluency.get("hesitation_count") or 0) + (filler.get("count") or 0)

    accuracy_band = rank_accuracy(grammar_score)
    use_of_english_band = rank_use_of_english(pronun_score)
    fluency_bands = rank_fluency(
        pace.get("wpm"), hesitations_repetitions, fluency.get("long_pause_count"),
    )

    return {
        "accuracy": {
            "score": grammar_score,
            "band": accuracy_band,
            "rubric_sentence": ACCURACY_RUBRIC[accuracy_band],
        },
        "fluency": {
            "score": fluency.get("score"),
            "band": fluency_bands["overall"],
            "rubric_sentence": FLUENCY_RUBRIC[fluency_bands["overall"]],
            "speaking_rate_wpm": pace.get("wpm"),
            "speaking_rate_band": fluency_bands["speaking_rate_band"],
            "hesitations_and_repetitions_count": hesitations_repetitions,
            "hesitations_and_repetitions_band": fluency_bands["hesitations_repetitions_band"],
            "unexpected_pauses_count": fluency.get("long_pause_count"),
            "unexpected_pauses_band": fluency_bands["unexpected_pauses_band"],
            "pause_data_available": fluency.get("pause_data_available"),
        },
        "use_of_english": {
            "score": pronun_score,
            "band": use_of_english_band,
            "rubric_sentence": USE_OF_ENGLISH_RUBRIC[use_of_english_band],
            "pronunciation_provider_available": evidence.get("pronunciation", {}).get("available"),
        },
    }


def build_report_evidence(transcript: str, transcript_with_fillers: Optional[str],
                           evidence: dict, vocab_agg: Optional[dict] = None,
                           cefr_agg: Optional[dict] = None, name: Optional[str] = None) -> dict:
    """Assembles the single JSON object of DETERMINISTIC evidence that gets
    sent to Groq alongside the transcript. `evidence` is score_free_speech()'s
    own output dict for the stage being reported on (i.e. the Full
    Assessment / `final` stage) — nothing here recomputes anything, it only
    re-shapes already-computed values for the prompt, plus the two purely
    local, non-fabricating helpers above (find_repeated_words,
    build_locked_rubric)."""
    vocab = vocab_agg if vocab_agg is not None else evidence.get("vocabulary", {})
    cefr = cefr_agg if cefr_agg is not None else evidence.get("cefr", {})

    return {
        "student_name": name or None,
        "word_count": evidence.get("evidence", {}).get("word_count"),
        "duration_seconds": evidence.get("evidence", {}).get("duration_seconds"),
        "low_evidence_short_sample": evidence.get("evidence", {}).get("low_evidence"),
        "overall_score": evidence.get("overall"),
        "pace": evidence.get("pace"),
        "filler": {
            "score": evidence.get("filler", {}).get("score"),
            "count": evidence.get("filler", {}).get("count"),
            "words_detected": evidence.get("filler", {}).get("words"),
            "rate_per_min": evidence.get("filler", {}).get("rate_per_min"),
        },
        "hesitations": evidence.get("hesitations"),
        "grammar": {
            "score": evidence.get("grammar", {}).get("score"),
            "error_count": evidence.get("grammar", {}).get("errors"),
            "issues": evidence.get("grammar", {}).get("issues"),
            "source": evidence.get("grammar_source"),
        },
        "pronunciation": {
            "score": evidence.get("pronunciation", {}).get("score"),
            "available": evidence.get("pronunciation", {}).get("available"),
            "issues": evidence.get("pronunciation", {}).get("issues"),
            "provider": evidence.get("pronunciation", {}).get("provider"),
        },
        "clarity": evidence.get("clarity"),
        "fluency": evidence.get("fluency"),
        "vocabulary": {
            "score": vocab.get("score"),
            "unique_words": vocab.get("unique_words"),
            "total_words": vocab.get("total_words"),
            "advanced_ratio_percent": vocab.get("advanced_ratio"),
            "diversity": vocab.get("diversity"),
            "sophistication": vocab.get("sophistication"),
            "vocabulary_distribution_by_level": None,  # not computed anywhere in the
            # pipeline (no per-word CEFR tagging exists) — explicitly None so the
            # prompt/report says "not available" instead of a fabricated breakdown.
        },
        "cefr": cefr,
        "archetype": evidence.get("archetype"),
        "repeated_words": find_repeated_words(transcript),
        "locked_rubric": build_locked_rubric(evidence),
    }


# ── Prompt ──────────────────────────────────────────────────────────────────

REPORT_JSON_SCHEMA_INSTRUCTIONS = """
Respond with ONLY a single JSON object (no markdown fences, no commentary
before or after) matching exactly this shape:

{
  "overview": {
    "overall_assessment": "2-4 sentences, teacher voice",
    "grammar_accuracy_summary": "grounded in grammar evidence given",
    "advanced_grammar_constructions_detected": "prose summary, or 'None clearly evidenced in this sample.' if none",
    "complex_sentence_usage": "grounded in avg sentence length / transcript, or 'Not clearly measurable from this sample.'",
    "strong_vocabulary_observations": "grounded in vocabulary evidence",
    "strong_language_use_observations": "grounded in transcript/evidence"
  },
  "growth_areas": {
    "repeated_overused_words": "reference the actual repeated_words list given, or 'No significant repetition detected.'",
    "fillers": "reference the actual filler words/count given, or 'No significant filler use detected.'",
    "linking_word_suggestions": "concrete, e.g. suggest 'however', 'in addition' etc. where it would help THIS transcript",
    "grammar_issues": "reference the actual grammar issues given, or 'No grammar issues detected.'",
    "vocabulary_improvements": "concrete suggestions",
    "fluency_pacing_improvements": "grounded in the fluency/pace evidence given",
    "other_weaknesses": "any other genuine, evidence-grounded observation, or 'None noted.'"
  },
  "vocabulary": {
    "active_vocabulary_note": "1-2 sentences using the unique_words/total_words/CEFR values given",
    "useful_higher_level_words_used": ["actual words pulled from the transcript that are genuinely higher-level; empty list if none"],
    "suggestions_for_improving_vocabulary": "concrete suggestions"
  },
  "repetitions": [
    {"word_or_phrase": "...", "frequency": <int, from repeated_words given>, "better_alternatives": ["...", "..."]}
  ],
  "advanced_grammar_used": [
    {"construction": "e.g. Reported speech / Relative clause / Passive voice / Conditional / Modal / Phrasal verb / Intensifier / a specific tense",
     "quoted_example": "the EXACT phrase from the transcript that shows it",
     "note": "brief teacher note"}
  ],
  "performance_summary_elaboration": {
    "accuracy_extra": "1-2 teacher-style sentences ADDING to (not replacing) the fixed rubric sentence, grounded in the grammar evidence",
    "fluency_extra": "1-2 teacher-style sentences ADDING to the fixed rubric sentence, grounded in the fluency/pace evidence",
    "use_of_english_extra": "1-2 teacher-style sentences ADDING to the fixed rubric sentence, grounded ONLY in the pronunciation evidence given (if pronunciation.available is false, say pronunciation evidence is limited rather than describing specific sounds)"
  }
}

Rules:
- "repetitions" MUST be built only from the repeated_words array given (same words, same counts) — do not add words not in that list, and if that list is empty return an empty "repetitions" array.
- "advanced_grammar_used" entries MUST each have a quoted_example that is real, verbatim text pulled from the transcript. If you cannot find a genuine example of a construction, do not include an entry for it. An empty list is the correct output when nothing is clearly evidenced.
- Never state a numeric score, count, or statistic anywhere in your response other than by referencing the ones given to you — do not compute or restate rounded/alternate versions of them.
- Every "quoted_example" and every reference to repeated words/fillers/grammar issues must be traceable to the transcript or evidence given below, not invented.
""".strip()


def build_teacher_prompt(transcript: str, transcript_with_fillers: Optional[str],
                          report_evidence: dict) -> list:
    system = (
        "You are a strict, experienced English-language teacher writing a structured "
        "spoken-English assessment report for a student, in the style used by this "
        "school's mentors. You are given (A) the student's full transcript and (B) a "
        "JSON object of deterministic measurements and evidence already computed by "
        "the school's scoring system.\n\n"
        "Absolute rules — violating any of these makes your response unusable:\n"
        "1. NEVER invent, change, round differently, or restate-as-different any score, "
        "count, or statistic. Use only the numbers given to you, and only by reference "
        "(e.g. 'your grammar score of {score}'), never a number you calculated yourself.\n"
        "2. NEVER invent a pronunciation error, grammar error, vocabulary statistic, "
        "filler count, or repeated word that is not present in the evidence given.\n"
        "3. NEVER claim a grammar construction (passive voice, reported speech, a "
        "relative clause, a conditional, a modal, an intensifier, a phrasal verb, a "
        "specific tense, etc.) is present unless you can quote the exact transcript "
        "text that demonstrates it.\n"
        "4. If a metric is unavailable or empty in the evidence, say plainly that it "
        "is unavailable/not detected — never fill the gap with a plausible-sounding "
        "invented value or example.\n"
        "5. The Accuracy/Fluency/Use-of-English rankings and their fixed rubric "
        "sentences are already decided (see locked_rubric in the evidence) — do not "
        "recompute or contradict them; only add brief, evidence-grounded elaboration "
        "in the *_extra fields.\n"
        "6. Write like a real teacher giving direct, specific, encouraging-but-honest "
        "feedback about THIS transcript — not generic, templated praise.\n\n"
        + REPORT_JSON_SCHEMA_INSTRUCTIONS
    )

    user_parts = [
        "STUDENT TRANSCRIPT (verbatim, as transcribed):",
        transcript or "(empty transcript)",
    ]
    if transcript_with_fillers and transcript_with_fillers != transcript:
        user_parts += [
            "",
            "TRANSCRIPT WITH FILLER MARKERS (same transcript, fillers marked inline):",
            transcript_with_fillers,
        ]
    user_parts += [
        "",
        "DETERMINISTIC ASSESSMENT EVIDENCE (JSON — the only source of numbers/facts you may use):",
        json.dumps(report_evidence, indent=2, ensure_ascii=False, default=str),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ── Groq client ─────────────────────────────────────────────────────────────

@dataclass
class TeacherReportResult:
    """Normalized result. `report` is the final, ready-to-serve structured
    report — the model's parsed JSON with `locked_rubric`'s scores/bands/
    rubric sentences re-merged in (see generate_teacher_report()), so a
    caller never has to reach into two places to get the full report."""
    available: bool
    report: Optional[dict] = None
    raw_text: Optional[str] = None
    model: Optional[str] = None
    prompt: Optional[list] = field(default=None, repr=False)
    detail: Optional[str] = None


class GroqReportProvider:
    """Thin, isolated client for Groq's OpenAI-compatible chat completions
    endpoint. Mirrors the existing provider pattern in this project (see
    stt_provider.SaarasSTTProvider / languagetool_provider): synchronous
    httpx, never raises for a merely-unavailable/failed call, returns a
    normalized result with available/detail instead."""

    def __init__(self, api_key: str | None = None, base_url: str = GROQ_API_URL,
                 model: str = GROQ_MODEL, timeout: httpx.Timeout = DEFAULT_TIMEOUT):
        self.api_key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate_report(self, transcript: str, transcript_with_fillers: Optional[str],
                         report_evidence: dict) -> TeacherReportResult:
        if not self.is_available():
            return TeacherReportResult(
                available=False,
                detail="Groq is not configured (GROQ_API_KEY is not set).",
            )
        if not transcript or not transcript.strip():
            return TeacherReportResult(
                available=False,
                detail="No transcript available to build a report from.",
            )

        messages = build_teacher_prompt(transcript, transcript_with_fillers, report_evidence)

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
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            try:
                err_body = e.response.json().get("error", {})
                reason = err_body.get("message") or e.response.text
            except Exception:
                reason = e.response.text
            return TeacherReportResult(
                available=False, prompt=messages,
                detail=f"Groq request failed ({e.response.status_code}): {reason}",
            )
        except Exception as e:
            return TeacherReportResult(
                available=False, prompt=messages,
                detail=f"Groq request failed: {e}",
            )

        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return TeacherReportResult(
                available=False, prompt=messages, raw_text=json.dumps(body)[:2000],
                detail="Groq response did not contain the expected message content.",
            )

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            return TeacherReportResult(
                available=False, prompt=messages, raw_text=raw_text,
                detail=f"Groq response was not valid JSON: {e}",
            )

        merged = merge_report(parsed, report_evidence)
        return TeacherReportResult(
            available=True, report=merged, raw_text=raw_text,
            model=self.model, prompt=messages,
        )


def merge_report(model_report: dict, report_evidence: dict) -> dict:
    """Combines the model's narrative JSON with the locked (Python-computed,
    never-model-touched) scores/bands/rubric sentences. Even if the model's
    JSON included its own performance_summary numbers, they are discarded
    here in favor of locked_rubric — the model's only contribution to
    performance_summary is the *_extra elaboration text."""
    locked = report_evidence.get("locked_rubric", {})
    elaboration = model_report.get("performance_summary_elaboration", {}) or {}

    performance_summary = {
        "accuracy": {
            **locked.get("accuracy", {}),
            "teacher_explanation": _join(
                locked.get("accuracy", {}).get("rubric_sentence"),
                elaboration.get("accuracy_extra"),
            ),
        },
        "fluency": {
            **locked.get("fluency", {}),
            "teacher_explanation": _join(
                locked.get("fluency", {}).get("rubric_sentence"),
                elaboration.get("fluency_extra"),
            ),
        },
        "use_of_english": {
            **locked.get("use_of_english", {}),
            "teacher_explanation": _join(
                locked.get("use_of_english", {}).get("rubric_sentence"),
                elaboration.get("use_of_english_extra"),
            ),
        },
    }

    return {
        "overview": model_report.get("overview", {}),
        "growth_areas": model_report.get("growth_areas", {}),
        "vocabulary": {
            **model_report.get("vocabulary", {}),
            "active_vocabulary_size": report_evidence.get("vocabulary", {}).get("unique_words"),
            "cefr_level": report_evidence.get("cefr", {}).get("level"),
            "unique_words": report_evidence.get("vocabulary", {}).get("unique_words"),
            "vocabulary_distribution_by_level": report_evidence.get("vocabulary", {})
                .get("vocabulary_distribution_by_level"),
        },
        "repetitions": model_report.get("repetitions", []),
        "advanced_grammar_used": model_report.get("advanced_grammar_used", []),
        "performance_summary": performance_summary,
    }


def _join(fixed: Optional[str], extra: Optional[str]) -> str:
    fixed = (fixed or "").strip()
    extra = (extra or "").strip()
    if fixed and extra:
        return f"{fixed} {extra}"
    return fixed or extra


# ── Convenience top-level function (what app.py calls) ────────────────────────

_provider = GroqReportProvider()


def generate_teacher_report(transcript: str, transcript_with_fillers: Optional[str],
                             evidence: dict, vocab_agg: Optional[dict] = None,
                             cefr_agg: Optional[dict] = None, name: Optional[str] = None,
                             provider: Optional[GroqReportProvider] = None) -> TeacherReportResult:
    """Single entry point app.py calls after an assessment's evidence is
    available. `evidence` is score_free_speech()'s full output dict for the
    stage being reported (the `final` stage of a guided assessment).
    `provider` is injectable for tests; defaults to the module-level
    instance (reads GROQ_API_KEY from the environment)."""
    prov = provider or _provider
    report_evidence = build_report_evidence(
        transcript, transcript_with_fillers, evidence, vocab_agg, cefr_agg, name,
    )
    return prov.generate_report(transcript, transcript_with_fillers, report_evidence)