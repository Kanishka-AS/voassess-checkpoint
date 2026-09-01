"""
pronunciation_provider.py
--------------------------
Abstract interface all acoustic-model backends must implement, so the rest
of the system (evaluation, and eventually the application) can swap
AllosaurusProvider <-> Wav2Vec2Provider <-> L2Wav2VecProvider without
changing any other code.

Only AllosaurusProvider is actually implemented right now. The others are
listed as NOT IMPLEMENTED so nobody mistakes their absence for a bug.
"""

from abc import ABC, abstractmethod
from .schemas import PronunciationResult


class PronunciationProvider(ABC):
    """Base class for a phoneme-level pronunciation-evidence backend."""

    #: Short machine-readable name, e.g. "allosaurus", "wav2vec2-espeak".
    name: str = "base"

    @abstractmethod
    def analyze(self, audio_path: str, expected_text: str) -> PronunciationResult:
        """
        Run the acoustic model on `audio_path`, compare against the
        phonemes expected for `expected_text`, and return a
        PronunciationResult containing raw acoustic evidence.

        Implementations MUST NOT:
          - invent a 0-100 pronunciation score
          - claim forced alignment was used if it wasn't
          - claim word-level timestamps if none are available
        """
        raise NotImplementedError

    @abstractmethod
    def resource_info(self) -> dict:
        """Return whatever the provider can honestly report about its own
        resource footprint (model size, params, etc). Fields the provider
        cannot measure should be omitted or explicitly set to None, not
        estimated silently."""
        raise NotImplementedError


# Providers that are referenced by the target architecture but do not have
# a real implementation yet. Importing these names raises NotImplementedError
# immediately rather than silently no-op-ing, so nothing downstream can
# accidentally treat them as working.

class Wav2Vec2Provider(PronunciationProvider):
    """NOT IMPLEMENTED. facebook/wav2vec2-lv-60-espeak-cv-ft is hosted on
    huggingface.co, which is not reachable from this environment
    (HTTP 403 host_not_allowed, verified). This class exists only as an
    architectural placeholder so the interface shape is settled."""

    name = "wav2vec2-espeak (NOT IMPLEMENTED)"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Wav2Vec2Provider is not implemented. "
            "facebook/wav2vec2-lv-60-espeak-cv-ft could not be downloaded "
            "in this environment (huggingface.co is network-blocked). "
            "Enable access to huggingface.co, or upload the model weights "
            "directly, then implement this class."
        )

    def analyze(self, audio_path: str, expected_text: str) -> PronunciationResult:
        raise NotImplementedError

    def resource_info(self) -> dict:
        raise NotImplementedError


class L2Wav2VecProvider(PronunciationProvider):
    """NOT IMPLEMENTED. slplab/wav2vec2-large-robust-L2-english-phoneme-recognition
    is hosted on huggingface.co, same blocker as Wav2Vec2Provider. Also
    note: its license has not been verified -- check before using in
    production even once it's downloadable."""

    name = "l2-wav2vec2 (NOT IMPLEMENTED)"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "L2Wav2VecProvider is not implemented. "
            "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition "
            "could not be downloaded in this environment (huggingface.co "
            "is network-blocked), and its license has not been verified."
        )

    def analyze(self, audio_path: str, expected_text: str) -> PronunciationResult:
        raise NotImplementedError

    def resource_info(self) -> dict:
        raise NotImplementedError
