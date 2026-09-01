"""
schemas.py
----------
Structured output types for the pronunciation-analysis subsystem.

IMPORTANT SEMANTIC RULE (do not violate this elsewhere in the codebase):

Values called `posterior` here are raw softmax outputs of an acoustic
model for one phone symbol, measured at the single frame where that
symbol's posterior is highest across the inspected span. They are
"acoustic evidence" for a phone hypothesis. They are NOT:

  - a probability that the speaker pronounced the word correctly
  - a calibrated pronunciation score
  - comparable in meaning across different acoustic models (each model's
    posteriors are conditioned on that model's own training distribution)

No function in this package is permitted to silently rescale a posterior
into a 0-100 "score" without an explicit, separately-justified calibration
step. As of this writing, no such calibration exists, because no labeled
pronunciation-error dataset has been collected for this project.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PhonemeEvidence:
    """Acoustic evidence for a single expected phoneme."""

    expected: str
    # Best-matching frame index / time for the EXPECTED phone, within the
    # inspected span (whole-utterance if no alignment is available).
    max_posterior: float
    frame_index: Optional[int] = None
    time_sec: Optional[float] = None

    # What the model's own top-1 decoding says at that same frame.
    top1_phone: Optional[str] = None
    top1_posterior: Optional[float] = None

    # Blank (CTC) posterior at that frame, for diagnosing blank competition.
    blank_posterior: Optional[float] = None

    # Whether the expected phone appears anywhere in the model's own
    # greedy-decoded phone sequence for the whole utterance (weak signal,
    # since this is not aligned to the specific word occurrence).
    appears_in_greedy_decode: Optional[bool] = None

    # Explicit flag: was this measured with real temporal alignment
    # (word/phone boundaries known) or just a whole-utterance max search?
    alignment_quality: str = "none"  # "none" | "word-window" | "forced-alignment"

    # -- Normalization metadata (added for the G2P->Allosaurus vocabulary
    # mismatch experiment). `expected` above is now the MODEL-vocabulary
    # phone actually looked up (e.g. "ɹ", "tʃ"). `generic_phones` records
    # what the generic G2P layer originally produced before normalization
    # (e.g. ["r"], or ["t","ʃ"] for a merged affricate), so callers can see
    # both sides of the mapping. `normalization_rule` is None when the
    # phone passed through unchanged.
    generic_phones: List[str] = field(default_factory=list)
    normalization_rule: Optional[str] = None
    in_model_inventory: Optional[bool] = None

    # -- Simple evidence-comparison metric (NOT a pronunciation score --
    # see schemas.py module docstring). `competing_symbol`/`competing_posterior`
    # is whatever symbol (possibly <blank>) had the second-highest posterior
    # at the expected phone's best frame, i.e. the model's strongest
    # alternative explanation for that frame. `evidence_margin` is simply
    # max_posterior - competing_posterior: how much more (or less) the
    # model favors the expected phone over its next-best alternative at
    # that frame.
    competing_symbol: Optional[str] = None
    competing_posterior: Optional[float] = None
    evidence_margin: Optional[float] = None

    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected": self.expected,
            "max_posterior": round(self.max_posterior, 4) if self.max_posterior is not None else None,
            "frame_index": self.frame_index,
            "time_sec": round(self.time_sec, 3) if self.time_sec is not None else None,
            "top1_phone": self.top1_phone,
            "top1_posterior": round(self.top1_posterior, 4) if self.top1_posterior is not None else None,
            "blank_posterior": round(self.blank_posterior, 4) if self.blank_posterior is not None else None,
            "appears_in_greedy_decode": self.appears_in_greedy_decode,
            "alignment_quality": self.alignment_quality,
            "generic_phones": self.generic_phones,
            "normalization_rule": self.normalization_rule,
            "in_model_inventory": self.in_model_inventory,
            "competing_symbol": self.competing_symbol,
            "competing_posterior": round(self.competing_posterior, 4) if self.competing_posterior is not None else None,
            "evidence_margin": round(self.evidence_margin, 4) if self.evidence_margin is not None else None,
            "notes": self.notes,
        }


@dataclass
class PronunciationResult:
    """Top-level result of a provider's analyze() call."""

    text: str
    provider: str
    model_name: str
    audio_path: str

    expected_phonemes: List[str] = field(default_factory=list)
    # Generic phones after AllosaurusPhoneNormalizer, i.e. what was actually
    # looked up in the acoustic model's vocabulary. Same length as the
    # number of PhonemeEvidence entries in phoneme_evidence (may be SHORTER
    # than expected_phonemes when an affricate merge collapsed two generic
    # phones into one model phone).
    normalized_phonemes: List[str] = field(default_factory=list)
    greedy_decoded_phonemes: List[str] = field(default_factory=list)
    phoneme_evidence: List[PhonemeEvidence] = field(default_factory=list)

    # Explicit disclaimers carried in the object itself, not just in docs,
    # so any downstream consumer (including a future UI) sees them.
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model_name": self.model_name,
            "audio_path": self.audio_path,
            "expected_phonemes": self.expected_phonemes,
            "normalized_phonemes": self.normalized_phonemes,
            "greedy_decoded_phonemes": self.greedy_decoded_phonemes,
            "phonemes": [p.to_dict() for p in self.phoneme_evidence],
            "warnings": self.warnings,
        }
