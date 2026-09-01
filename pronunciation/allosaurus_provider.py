"""
allosaurus_provider.py
-----------------------
The only PronunciationProvider that is actually implemented and tested as
of this writing. Wraps the Allosaurus uni2005 acoustic model.

VERIFIED IN THIS ENVIRONMENT:
  - allosaurus 1.0.2 installed via pip.
  - uni2005 model (45MB, github.com/xinjli/allosaurus release) downloaded
    successfully.
  - phone.txt inventory has 229 phones, including r, ɹ, ɻ, v, w, θ, ð, ʃ,
    s, l simultaneously as distinct symbols (checked directly from the
    file, not assumed).
  - AM output (`recognizer.am(...)`) is raw logits, confirmed by reading
    allosaurus/lm/decoder.py, which applies its own softmax before use.
  - column 0 of the logits matrix is the CTC blank; columns 1..229
    correspond to phone.txt's 1-indexed phone ids.
"""

import os
from typing import Dict, List

import numpy as np

from allosaurus.app import read_recognizer
from allosaurus.am.utils import move_to_tensor
from allosaurus.audio import read_audio
import allosaurus as _allosaurus_pkg

from .pronunciation_provider import PronunciationProvider
from .schemas import PronunciationResult
from .phoneme_analyzer import softmax, greedy_decode, analyze_normalized_phonemes
from .g2p import g2p_sentence
from .allosaurus_normalizer import AllosaurusPhoneNormalizer

MODEL_NAME = "uni2005"


class AllosaurusProvider(PronunciationProvider):
    name = "allosaurus"

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._recognizer = read_recognizer(model_name)

        phone_path = os.path.join(
            os.path.dirname(_allosaurus_pkg.__file__), "pretrained", model_name, "phone.txt"
        )
        lines = open(phone_path, encoding="utf-8").read().splitlines()
        phones = [l.split()[0] for l in lines if l.strip()]

        self.id2label: Dict[int, str] = {0: "<blank>"}
        for i, p in enumerate(phones):
            self.id2label[i + 1] = p
        self.label2id: Dict[str, int] = {v: k for k, v in self.id2label.items()}
        self._phone_count = len(phones)

        # Built once from this model's ACTUAL inventory (self.label2id
        # keys), not assumed -- see allosaurus_normalizer.py. If uni2005 is
        # ever swapped for a different Allosaurus checkpoint with a
        # different phone.txt, this normalizer's active_rhotic_map /
        # active_affricate_merges will reflect that automatically.
        self.normalizer = AllosaurusPhoneNormalizer(self.label2id.keys())

    # -- internal -----------------------------------------------------

    def _get_probs(self, audio_path: str) -> np.ndarray:
        audio = read_audio(audio_path)
        feat = self._recognizer.pm.compute(audio)
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)
        tensor_feat, tensor_feat_len = move_to_tensor(
            [feats, feat_len], self._recognizer.config.device_id
        )
        logits = self._recognizer.am(tensor_feat, tensor_feat_len)
        logits = logits.detach().numpy()[0]  # [T, C]
        return softmax(logits)

    # -- public interface ----------------------------------------------

    def analyze(self, audio_path: str, expected_text: str) -> PronunciationResult:
        """Standard path: text -> G2P -> generic IPA phones -> normalizer
        -> evidence. Requires the text to be in eng_to_ipa's dictionary."""
        expected_phones = g2p_sentence(expected_text)

        warnings: List[str] = []
        if not expected_phones:
            warnings.append(
                "G2P returned no phonemes for the expected text "
                "(word(s) may not be in eng_to_ipa's dictionary). Use "
                "analyze_manual() / `python3 -m pronunciation.evaluation "
                "--manual` to supply an explicit expected phone sequence "
                "instead, e.g. for nonsense words or minimal pairs."
            )

        return self._analyze_generic_phones(
            audio_path=audio_path,
            text_label=expected_text,
            generic_phones=expected_phones,
            extra_warnings=warnings,
        )

    def analyze_manual(
        self, audio_path: str, expected_phones: List[str], label: str = "(manual)"
    ) -> PronunciationResult:
        """Controlled-experiment path (TASK 2): caller supplies the
        expected phone sequence directly, bypassing G2P entirely. This is
        required for nonsense words / minimal pairs that are not in
        eng_to_ipa's dictionary, and for probing a single phone in
        isolation (e.g. ["r"] or ["tʃ"]).

        `expected_phones` should be GENERIC IPA symbols exactly as G2P
        would produce them (e.g. "r", not "ɹ"; "t","ʃ" separately if you
        want the affricate-merge rule exercised, or "tʃ" directly if you
        want to bypass it). They still pass through
        AllosaurusPhoneNormalizer, same as the G2P path.
        """
        return self._analyze_generic_phones(
            audio_path=audio_path,
            text_label=label,
            generic_phones=list(expected_phones),
            extra_warnings=[
                "Expected phones were supplied manually, not derived from "
                "G2P. This mode exists for controlled experiments "
                "(minimal pairs, nonsense words, single-phone probes) and "
                "is explicitly NOT a claim that any G2P dictionary lookup "
                "occurred for this input."
            ],
        )

    def _analyze_generic_phones(
        self,
        audio_path: str,
        text_label: str,
        generic_phones: List[str],
        extra_warnings: List[str],
    ) -> PronunciationResult:
        """Shared core: generic IPA phones -> AllosaurusPhoneNormalizer ->
        Allosaurus vocabulary -> acoustic evidence. Used by both analyze()
        (G2P path) and analyze_manual() (explicit-phones path) so the
        normalization + evidence-extraction logic exists in exactly one
        place."""
        normalized = self.normalizer.normalize_sequence(generic_phones)

        probs = self._get_probs(audio_path)
        decoded = greedy_decode(probs, self.id2label)

        evidence = analyze_normalized_phonemes(
            probs, self.id2label, self.label2id, normalized
        )

        warnings: List[str] = list(extra_warnings)
        not_in_inventory = [nphone.model for nphone in normalized if not nphone.in_inventory]
        if not_in_inventory:
            warnings.append(
                f"The following normalized phones are not in this model's "
                f"phone inventory and will report zero evidence: "
                f"{sorted(set(not_in_inventory))}"
            )
        warnings.append(
            "No forced alignment or word-level timestamps are used. All "
            "evidence is a whole-utterance maximum-posterior search per "
            "expected phone (alignment_quality='none'). If a phone occurs "
            "more than once in the sentence, all occurrences currently "
            "report the same (strongest) evidence."
        )

        return PronunciationResult(
            text=text_label,
            provider=self.name,
            model_name=self.model_name,
            audio_path=audio_path,
            expected_phonemes=generic_phones,
            normalized_phonemes=[nphone.model for nphone in normalized],
            greedy_decoded_phonemes=decoded,
            phoneme_evidence=evidence,
            warnings=warnings,
        )

    def resource_info(self) -> dict:
        model_dir = os.path.join(
            os.path.dirname(_allosaurus_pkg.__file__), "pretrained", self.model_name
        )
        model_pt = os.path.join(model_dir, "model.pt")
        size_bytes = os.path.getsize(model_pt) if os.path.exists(model_pt) else None
        return {
            "provider": self.name,
            "model_name": self.model_name,
            "model_file_size_mb": round(size_bytes / (1024 * 1024), 2) if size_bytes else None,
            "phone_inventory_size": self._phone_count,
            "measurement": "VERIFIED (os.path.getsize on the actual downloaded model.pt file)",
        }
