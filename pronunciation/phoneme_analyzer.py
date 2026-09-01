"""
phoneme_analyzer.py
--------------------
Model-agnostic evidence extraction from CTC frame-level logits.

This module does NOT know anything about Allosaurus specifically. It only
assumes the standard CTC convention: a [T, C] logits matrix where index 0
is the blank symbol and indices 1..C-1 map to phone labels via an
`id2label` dict. Any future CTC-based provider (Wav2Vec2Provider,
L2Wav2VecProvider) can reuse this module by producing logits in the same
shape and supplying its own id2label map -- that's the whole point of
keeping this separate from allosaurus_provider.py.

No alignment is assumed. Without forced alignment or word timestamps, the
only thing we can honestly measure is: "somewhere in this utterance, what
is the strongest evidence for phone X?" That is a weak, whole-utterance
signal -- NOT a claim about a specific word occurrence. This is stated
explicitly in PhonemeEvidence.alignment_quality = "none".
"""

from typing import Dict, List, Optional
import numpy as np

from .schemas import PhonemeEvidence
from .allosaurus_normalizer import NormalizedPhone


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Allosaurus's am(...) output is raw logits, NOT
    log-probabilities -- confirmed by reading allosaurus/lm/decoder.py,
    which itself computes exp(logit - max) / sum(...) before treating
    values as probabilities. Applying exp() directly to raw logits without
    subtracting the max (as an earlier draft in this session did) produces
    nonsense values >> 1; that bug was caught and is fixed here."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def greedy_decode(probs: np.ndarray, id2label: Dict[int, str]) -> List[str]:
    """Standard CTC greedy decode: argmax per frame, collapse repeats,
    drop blank (index 0)."""
    top1 = probs.argmax(axis=1)
    collapsed = []
    prev = None
    for idx in top1:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [id2label[i] for i in collapsed if i != 0]


def evidence_for_phone(
    probs: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    phone: str,
) -> PhonemeEvidence:
    """Whole-utterance max-posterior evidence for a single expected phone.

    alignment_quality is explicitly "none": this searches the ENTIRE
    utterance for the frame of maximum posterior for `phone`, without
    regard to where in the sentence that phone is actually supposed to
    occur. If the phone occurs multiple times in the sentence, this will
    report the strongest occurrence, not a specific one. This is a
    deliberate limitation, not an oversight -- see module docstring.
    """
    if phone not in label2id:
        return PhonemeEvidence(
            expected=phone,
            max_posterior=0.0,
            alignment_quality="none",
            notes=f"'{phone}' is not in this model's phone inventory.",
        )

    pid = label2id[phone]
    target_col = probs[:, pid]
    best_frame = int(np.argmax(target_col))
    best_val = float(target_col[best_frame])

    frame_probs = probs[best_frame]
    top1_idx = int(np.argmax(frame_probs))
    top1_label = id2label.get(top1_idx, f"<id {top1_idx}>")
    top1_val = float(frame_probs[top1_idx])
    blank_val = float(frame_probs[0])

    # Simple evidence-comparison metric (NOT a pronunciation score -- see
    # schemas.py docstring): whatever symbol other than the expected phone
    # itself has the highest posterior at this same frame is the model's
    # strongest competing explanation for that frame. This is usually the
    # same as top1 (when the model does not think the expected phone
    # occurred here at all), but differs when the expected phone *is*
    # top1 -- in that case it shows the runner-up, i.e. how comfortably
    # the expected phone wins.
    order = np.argsort(frame_probs)[::-1]
    competing_idx = next((int(i) for i in order if int(i) != pid), None)
    competing_label = id2label.get(competing_idx, f"<id {competing_idx}>") if competing_idx is not None else None
    competing_val = float(frame_probs[competing_idx]) if competing_idx is not None else None
    margin = (best_val - competing_val) if competing_val is not None else None

    return PhonemeEvidence(
        expected=phone,
        max_posterior=best_val,
        frame_index=best_frame,
        top1_phone=top1_label,
        top1_posterior=top1_val,
        blank_posterior=blank_val,
        alignment_quality="none",
        competing_symbol=competing_label,
        competing_posterior=competing_val,
        evidence_margin=margin,
        notes=None,
    )


def analyze_expected_phonemes(
    probs: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    expected_phones: List[str],
) -> List[PhonemeEvidence]:
    """Run evidence_for_phone for every phone in expected_phones, in order.
    Duplicate phones (e.g. two 'l's in a sentence) will currently return
    the SAME whole-utterance-max evidence for each occurrence -- this is a
    known limitation of not having alignment, and is why
    alignment_quality is always "none" here.

    NOTE: this operates on raw (non-normalized) phones. It is kept around
    unchanged for backward compatibility / direct unit testing of
    phoneme_analyzer in isolation. AllosaurusProvider itself now calls
    analyze_normalized_phonemes (below) so evidence is looked up against
    Allosaurus's actual vocabulary rather than raw G2P output -- see
    allosaurus_normalizer.py for why that distinction matters."""
    decoded = greedy_decode(probs, id2label)
    results = []
    for phone in expected_phones:
        ev = evidence_for_phone(probs, id2label, label2id, phone)
        ev.appears_in_greedy_decode = phone in decoded
        results.append(ev)
    return results


def analyze_normalized_phonemes(
    probs: np.ndarray,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    normalized_phones: List[NormalizedPhone],
) -> List[PhonemeEvidence]:
    """Like analyze_expected_phonemes, but takes AllosaurusPhoneNormalizer
    output instead of raw generic phones. Evidence is looked up using each
    NormalizedPhone's `.model` symbol (the actual Allosaurus vocabulary
    entry), while `.generic` and `.rule` are copied onto the resulting
    PhonemeEvidence purely for transparent reporting -- they play no part
    in the evidence math itself."""
    decoded = greedy_decode(probs, id2label)
    results = []
    for np_phone in normalized_phones:
        ev = evidence_for_phone(probs, id2label, label2id, np_phone.model)
        ev.appears_in_greedy_decode = np_phone.model in decoded
        ev.generic_phones = list(np_phone.generic)
        ev.normalization_rule = np_phone.rule
        ev.in_model_inventory = np_phone.in_inventory
        results.append(ev)
    return results
