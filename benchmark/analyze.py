"""
Assessment-oriented analysis: NOT WER. WER only tells you whether the
*words* came out right. This module answers the question this project
actually cares about: did the transcript preserve the way the person
spoke — the fillers, repetitions, and self-corrections our downstream
filler/fluency scoring (filler_detector.py, score_fluency) depends on?

"Recognized the word" vs "preserved the way it was spoken":
  - A provider that transcribes "I really think the meeting went well"
    from ground truth "I I really think that the the meeting went well"
    got every *word* right in sequence, but it SILENTLY DELETED the two
    repetitions. WER on the retained words could still look decent; for
    this project, that's a failure, because filler_detector.py never
    sees the repetition to score it.
  - This module checks each ground-truth phenomenon (filler / discourse
    filler / repetition / false-start) individually against the raw
    transcript text, so "silently cleaned up" and "genuinely mis-heard"
    are visible as different failure modes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s']", " ", s.lower()).strip()


def _retention(transcript: str, phrases: list[str]) -> dict:
    """For each ground-truth phrase, was it found (as a substring of the
    normalized transcript)? Returns per-phrase hits plus an overall ratio.
    Empty phrase list => ratio is None (not applicable), not 0 — a clean
    sample retaining zero fillers isn't a retention failure."""
    if not phrases:
        return {"applicable": False, "ratio": None, "hits": {}}
    norm_t = _norm(transcript)
    hits = {p: (_norm(p) in norm_t) for p in phrases}
    ratio = sum(hits.values()) / len(phrases)
    return {"applicable": True, "ratio": round(ratio, 2), "hits": hits}


def _word_count(transcript: str) -> int:
    return len(_norm(transcript).split())


@dataclass
class SampleAnalysis:
    sample: str
    provider: str
    available: bool
    detail: str | None
    transcript: str
    latency_seconds: float
    word_count: int
    ground_truth_word_count: int
    filler_retention: dict
    discourse_filler_retention: dict
    repetition_retention: dict
    false_start_retention: dict
    has_word_timestamps: bool
    has_word_confidence: bool


def analyze_sample(sample_name: str, ground_truth: dict, provider_name: str,
                    timed_result) -> SampleAnalysis:
    r = timed_result.result
    transcript = r.transcript or ""
    return SampleAnalysis(
        sample=sample_name,
        provider=provider_name,
        available=r.available,
        detail=r.detail,
        transcript=transcript,
        latency_seconds=round(timed_result.latency_seconds, 3),
        word_count=_word_count(transcript),
        ground_truth_word_count=_word_count(ground_truth["text"]),
        filler_retention=_retention(transcript, ground_truth.get("fillers", [])),
        discourse_filler_retention=_retention(transcript, ground_truth.get("discourse_fillers", [])),
        repetition_retention=_retention(transcript, ground_truth.get("repetitions", [])),
        false_start_retention=_retention(transcript, ground_truth.get("false_starts", [])),
        has_word_timestamps=timed_result.word_level_timestamps,
        has_word_confidence=timed_result.word_level_confidence,
    )
