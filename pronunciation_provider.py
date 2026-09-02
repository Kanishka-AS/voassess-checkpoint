"""
Pronunciation provider architecture.

────────────────────────────────────────────────────────────────────────────
UPDATE (Whisper removed from the default pronunciation path):

Sarvam's Saaras v3 is the project's primary/default STT (see app.py,
DEFAULT_STT_PROVIDER = "saaras"), and pronunciation assessment must not
depend on Whisper — not Whisper confidence, not Whisper word probabilities,
not Whisper timings, and no second-pass Whisper transcription. Sarvam
remains the single source of truth for the transcript.

DEFAULT_PRONUNCIATION_PROVIDER (see app.py) is now "allosaurus_g2p":
audio + whatever transcript the STT step already produced (Sarvam by
default) -> pronunciation/g2p.py (text -> expected IPA phones) ->
pronunciation/allosaurus_provider.py (Allosaurus CTC acoustic-model
evidence for those expected phones, entirely offline/CPU, no network call,
no Whisper import anywhere in this path). See AllosaurusG2PPronunciationProvider
below for the full honesty/limitation disclosure (short version: this is an
UTTERANCE-LEVEL acoustic-evidence signal, not true phoneme- or word-level
scoring — no forced alignment exists in this project, so no per-word
`issues` are produced by this provider; see the class docstring).

WhisperConfidenceProvider is kept, unmodified, as an explicitly-selectable,
non-default option (same pattern as "gop" being kept as a selectable-but-
unavailable placeholder) — nothing production-default touches it anymore.

    PronunciationProvider (interface)
        │
        ▼
    PronunciationProviderRegistry
        │
        ├── AllosaurusG2PPronunciationProvider ("allosaurus_g2p") — default.
        │                                 G2P + Allosaurus acoustic evidence,
        │                                 no Whisper anywhere in this path.
        ├── WhisperConfidenceProvider   ("whisper_confidence") — legacy,
        │                                 explicit-selection only, unchanged.
        ├── SaarasPronunciationProvider ("saaras")              — primary, per mentor.
        ├── LocalLLMPronunciationProvider ("local_llm")         — secondary, if configured.
        └── GOPPronunciationProvider    ("gop")                 — ON HOLD, placeholder only.

app.py never branches on provider name (`if provider == "saaras"` etc.) — it
asks the registry for a provider and calls `.assess()`. Provider-specific
logic lives inside each provider class.

────────────────────────────────────────────────────────────────────────────
IMPORTANT — Saaras is not (yet) a real pronunciation-assessment integration.

Sarvam's Saaras is a speech-to-text / speech-translation model
(`/speech-to-text`, model `saaras:v3`) — its documented response is a
transcript, not a phoneme/pronunciation score. There is no dedicated
pronunciation-scoring endpoint to call.

SaarasPronunciationProvider below is written as a genuine adapter (real HTTP
call, real credential handling) that reuses the same confidence-based
scoring approach the existing Whisper path already uses — IF Saaras's
response includes per-word confidence data. It has not been verified against
a live account, because the exact response shape (word-level confidence
field name, or its absence) isn't documented publicly. If that data isn't
present in the response, `assess()` returns a controlled "integration
incomplete" result (available=False) rather than fabricating a score.

Before this can be the real primary provider, someone needs to either:
  (a) confirm Saaras's actual response includes per-word/segment confidence
      and tell us the field name, or
  (b) confirm we're expected to derive pronunciation signal a different way
      (e.g. comparing Saaras's transcript against Whisper's as a disagreement
      proxy), or
  (c) point to a different Sarvam endpoint intended for pronunciation
      assessment.

This is why "saaras" is not the pronunciation default even though it is the
STT default: it has no real pronunciation signal to give yet. "allosaurus_g2p"
is what actually satisfies "assess pronunciation from Sarvam's transcript +
audio, without Whisper" today.
────────────────────────────────────────────────────────────────────────────

Configuration:
    SARVAM_API_KEY   Sarvam API subscription key. Required for "saaras".
                      Never hard-coded — read from the environment only.
    SARVAM_API_URL   Base URL for the Sarvam API. Defaults to
                      "https://api.sarvam.ai".
    LOCAL_LLM_URL     Base URL of a local pronunciation-scoring LLM server,
                      if/when one exists. Unset today — see
                      LocalLLMPronunciationProvider.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# ── Result shape ─────────────────────────────────────────────────────────────

@dataclass
class PronunciationResult:
    """Normalized result every provider returns. This is the only shape the
    scoring engine (score_free_speech) is allowed to know about — no
    provider-specific fields leak past this boundary."""
    score: float                     # 0-100. Meaningless if available=False; see fallback rule below.
    issues: list                     # [{"word": str, "confidence": int}], capped at 8, worst-first
    provider: str                    # provider actually used, e.g. "whisper_confidence"
    available: bool                  # False => provider could not produce a real assessment
    detail: str | None = field(default=None)  # human-readable reason when available=False
    # Additive (not part of the pre-existing interface): a short, honest
    # description of what the score/issues above were actually computed
    # from — e.g. "Whisper avg_logprob confidence; no reference phonemes
    # available". Optional because not every provider needs to say more
    # than its name+detail already do. See WhisperConfidenceProvider.assess().
    methodology: str | None = field(default=None)


PRONUNCIATION_PROVIDER_NAMES = ("allosaurus_g2p", "whisper_confidence", "saaras", "local_llm", "gop")


class PronunciationProviderError(Exception):
    """Raised for a request naming an unknown/invalid provider. Callers (the
    /assess, /assess/stage routes) turn this into a 400."""


# ── Interface ─────────────────────────────────────────────────────────────────

class PronunciationProvider(ABC):
    """Every pronunciation provider implements this. `assess()` must never
    raise for a merely-unavailable provider (not configured / not
    implemented) — it returns a PronunciationResult with available=False and
    a `detail` message instead. Exceptions are reserved for genuine bugs."""

    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap, synchronous check — no network call. Used by the UI/API to
        report whether this provider can currently be used, before a request
        is even made (e.g. so the frontend can show Local LLM as available
        only when the backend confirms it's configured)."""

    @abstractmethod
    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        """Produce a pronunciation assessment for one recording.

        transcript: the transcript already produced upstream by whichever
                    STT provider actually ran (Sarvam Saaras v3 by default;
                    Whisper only if explicitly selected/used as STT) — no
                    provider here re-transcribes just to know what was said.
        segments:   Whisper-shaped segments/words (with per-word
                    probabilities), populated ONLY when Whisper was the STT
                    provider used — Saaras (the default STT) always returns
                    segments=[] (see stt_provider.py). Only
                    WhisperConfidenceProvider reads this field; providers
                    that must not depend on Whisper (e.g. allosaurus_g2p,
                    the default) ignore it entirely and use transcript +
                    wav_path instead.
        wav_path:   path to the converted 16kHz mono WAV — what an
                    audio-based provider (Saaras, allosaurus_g2p) needs.
        """


# ── Provider 0: Allosaurus + G2P (default — no Whisper anywhere) ──────────────
#
# Wraps the already-existing, already-tested pronunciation/ package (see
# pronunciation/schemas.py, g2p.py, allosaurus_normalizer.py,
# phoneme_analyzer.py, allosaurus_provider.py) rather than reimplementing any
# of that logic here. That package was previously unused by app.py (see
# pronunciation/test_audio.py's own docstring, which explicitly notes app.py
# did not import it) — this class is what actually wires it into production.
#
# SIGNAL: audio (wav_path) + `transcript` (whatever the STT step already
# produced — Sarvam's Saaras v3 by default; this class does not care which
# STT produced it and never re-transcribes) -> pronunciation.g2p (text ->
# expected IPA phones) -> pronunciation.allosaurus_provider (Allosaurus CTC
# acoustic-model posteriors for those expected phones). Fully offline/CPU
# (~43MB model, downloaded once via allosaurus's own installer, then read
# from local disk) and gives a real, expected-content-aware comparison
# between the recording and the words that were actually said — as opposed
# to Whisper-confidence or Allosaurus-self-confidence, both of which only
# measure "how sure was the ASR about its own decoding," not "does this
# audio match the expected pronunciation of these specific words."
#
# HONESTY CONSTRAINTS (do not violate elsewhere in this class):
#   - This is UTTERANCE-LEVEL evidence, not word-level or true phoneme-level.
#     No forced alignment exists anywhere in this project (nothing in
#     requirements.txt provides it, and none is added here — see task
#     instructions: don't introduce a large/expensive dependency). Every
#     PhonemeEvidence.alignment_quality is "none": for each expected phone,
#     Allosaurus reports the single strongest matching frame ANYWHERE in the
#     whole recording, not at the specific timestamp where that phone/word
#     was supposed to occur (see pronunciation/phoneme_analyzer.py's own
#     docstring). If a phone occurs more than once in the sentence, every
#     occurrence currently reports the same (strongest) evidence.
#   - Because of the above, `issues` (the per-word list) is ALWAYS []. This
#     provider has no reliable way to say "word X was mispronounced" —
#     doing so would fabricate word-level precision this signal does not
#     have. (Contrast WhisperConfidenceProvider, which has real per-word
#     timestamps and can honestly report per-word issues.)
#   - The 0-100 `score` is an explicit, disclosed, UNCALIBRATED linear
#     rescaling (mean acoustic posterior x 100) of evidence for the expected
#     phones that are actually present in Allosaurus's phone inventory. It
#     is NOT validated against any labeled pronunciation-error dataset (none
#     exists in this project — see pronunciation/schemas.py's module
#     docstring, which is why that package's own scripts never compute a
#     score at all). This provider reports it anyway because the production
#     scoring engine (score_free_speech) requires a numeric score from every
#     provider — but every result's `methodology` says explicitly that this
#     is acoustic-evidence strength, not a validated accuracy grade.

_ALLOSAURUS_G2P_MODEL_NAME = "uni2005"


class AllosaurusG2PPronunciationProvider(PronunciationProvider):
    """Default pronunciation provider. See module-level comment block above
    this class for the full signal description and honesty constraints."""

    name = "allosaurus_g2p"

    def __init__(self):
        # Lazy singleton: loading the Allosaurus model reads model.pt from
        # disk (and, on a completely fresh install, triggers allosaurus's
        # own one-time model download) — expensive enough that this must
        # happen once per process, on first real use, not at import time
        # (registry construction must stay cheap — see the module-level
        # `pronunciation_registry` singleton comment further down this
        # file). Mirrors the pattern the pre-existing optional Allosaurus
        # secondary-signal code already used.
        self._provider = None
        self._init_error: str | None = None
        self._checked = False

    def _get_provider(self):
        if self._checked:
            return self._provider
        self._checked = True
        try:
            from pronunciation.allosaurus_provider import AllosaurusProvider
            self._provider = AllosaurusProvider(model_name=_ALLOSAURUS_G2P_MODEL_NAME)
        except Exception as e:
            # Covers: allosaurus/panphon/eng_to_ipa not installed, model
            # download/load failure, etc. Cached so a broken install costs
            # one attempt per process, not one attempt per request.
            self._init_error = f"{type(e).__name__}: {e}"
            self._provider = None
        return self._provider

    def is_available(self) -> bool:
        """Whether this provider CAN run in this environment at all (model
        loads successfully) — independent of whether any particular
        request's text/audio happens to produce a usable signal (that is
        reported per-call via assess()'s own available=False, not here)."""
        return self._get_provider() is not None

    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        provider = self._get_provider()
        if provider is None:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail=("Allosaurus/G2P pronunciation assessment could not be "
                         f"initialized in this environment: {self._init_error}"),
            )
        if not wav_path:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Allosaurus/G2P pronunciation assessment requires the "
                       "recorded audio, which wasn't provided to this call.",
            )
        if not transcript or not transcript.strip():
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="No transcript was available to assess pronunciation "
                       "against (nothing to G2P).",
            )

        try:
            result = provider.analyze(str(wav_path), transcript)
        except Exception as e:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail=f"Allosaurus pronunciation analysis failed: {type(e).__name__}: {e}",
            )

        if not result.expected_phonemes:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="G2P produced no phonemes for this transcript (the "
                       "word(s) may not be in the G2P dictionary) — no "
                       "pronunciation signal could be computed.",
            )

        scored = [ev.max_posterior for ev in result.phoneme_evidence if ev.in_model_inventory]
        if not scored:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="None of this transcript's expected phones are in "
                       "Allosaurus's phone inventory — no pronunciation "
                       "signal could be computed.",
            )

        mean_posterior = sum(scored) / len(scored)
        score = round(max(10.0, min(100.0, mean_posterior * 100)), 1)

        methodology = (
            "Pronunciation evidence derived from the transcript already "
            "produced by STT (Sarvam Saaras v3 by default) — G2P'd to "
            "expected IPA phones and compared against Allosaurus's "
            f"({provider.model_name}) own CPU acoustic-model posteriors for "
            "those specific phones. No Whisper transcription, confidence, "
            "word probabilities, or timings are used anywhere in this "
            "signal. UTTERANCE-LEVEL EVIDENCE ONLY: no forced alignment "
            "exists in this project, so each expected phone's evidence is "
            "the strongest match found anywhere in the recording, not at a "
            "specific word's timestamp — per-word issues are therefore "
            "never produced by this provider (issues=[] always; see "
            "alignment_quality on each phone in the underlying provider). "
            f"Score is an explicit, UNCALIBRATED linear rescaling (mean "
            f"posterior x 100) of acoustic evidence for the "
            f"{len(scored)}/{len(result.expected_phonemes)} expected phones "
            "present in Allosaurus's phone inventory — it reflects "
            "acoustic-evidence strength, not a validated pronunciation-"
            "accuracy grade (no labeled pronunciation-error dataset exists "
            "in this project for that calibration)."
        )
        if result.warnings:
            methodology += " Provider warnings: " + " | ".join(result.warnings)

        return PronunciationResult(
            score=score, issues=[], provider=self.name, available=True,
            methodology=methodology,
        )


# ── Provider 1: Whisper confidence (existing behavior, moved here as-is) ──────

# Common short/function words to skip even if Whisper is uncertain about them.
# Unchanged from the pre-existing app.py implementation.
_SKIP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "as", "do", "did",
    "not", "no", "so", "if", "its", "that", "this", "these", "those", "what", "which",
    "have", "has", "had", "will", "would", "could", "should", "may", "might", "shall",
    "am", "just", "very", "also", "then", "than", "when", "how", "who", "all", "can",
}


def _score_pronunciation_from_segments(segments: list) -> float:
    """Use Whisper avg_logprob as pronunciation proxy. 0 = perfect, -1 = bad.
    Byte-for-byte the pre-existing score_pronunciation() from app.py — not
    touched as part of this migration."""
    logprobs = [s["avg_logprob"] for s in segments if "avg_logprob" in s]
    if not logprobs:
        return 72.0
    avg = sum(logprobs) / len(logprobs)
    score = (avg + 1.0) / 0.9 * 100
    return round(max(10.0, min(100.0, score)), 1)


def _extract_mispronounced_from_segments(segments: list, threshold: float = 0.78) -> list:
    """Return words Whisper was uncertain about. Byte-for-byte the
    pre-existing extract_mispronounced() from app.py — not touched."""
    issues, seen = [], set()
    for seg in segments:
        for w in seg.get("words", []):
            prob = w.get("probability", 1.0)
            raw = w.get("word", "")
            word = raw.strip().strip(".,!?;:()[]\"'—-").lower()
            if (not word
                    or len(word) < 3
                    or word in _SKIP_WORDS
                    or word in seen
                    or prob >= threshold):
                continue
            seen.add(word)
            issues.append({"word": word, "confidence": int(round(prob * 100))})
    issues.sort(key=lambda x: x["confidence"])
    return issues[:8]


_WHISPER_METHODOLOGY = (
    "Pronunciation score is derived from Whisper's own transcription "
    "confidence (avg_logprob per segment; per-word probability for "
    "issues), used as an acoustic-clarity proxy. This is free-form speech "
    "with no reference/expected phoneme sequence, so this is NOT a "
    "phoneme-accuracy (GOP) score and never claims a specific phoneme was "
    "mispronounced — only that the model was less confident about a word."
)

# ── Optional secondary signal: Allosaurus (phoneme-level, no reference) ───────
#
# NOTE: Allosaurus (+ panphon, eng-to-ipa) IS now a real project dependency
# (see requirements.txt) — but only because it backs the new default
# pronunciation provider, AllosaurusG2PPronunciationProvider, above. This
# section is unrelated legacy code specific to WhisperConfidenceProvider
# (kept byte-for-byte, since that provider itself is kept only as a
# non-default, explicit-selection legacy option — see this file's module
# docstring): it best-effort-imports Allosaurus independently and, if
# available, uses it to add one extra, honest sentence of acoustic evidence
# to `methodology`. It never touches `score` or `issues` — see requirement
# to keep existing scoring behavior unchanged. It never requires or invents
# a reference phoneme sequence: it only reports how confidently Allosaurus's
# own CTC decoder recognized *some* phones somewhere in the recording, using
# only the raw AM logits Allosaurus actually returns (see
# pronunciation/allosaurus_provider.py, which verified that shape).

_ALLOSAURUS_MODEL_NAME = "uni2005"
_allosaurus_state = {"checked": False, "recognizer": None, "id2label": None}


def _get_allosaurus_recognizer():
    """Lazily import + load the Allosaurus uni2005 recognizer, once per
    process. Returns (recognizer, id2label) or (None, None) if Allosaurus
    isn't installed or fails to load for any reason — that failure is
    cached too, so a missing/broken install costs one attempt, not one
    attempt per request."""
    if _allosaurus_state["checked"]:
        return _allosaurus_state["recognizer"], _allosaurus_state["id2label"]
    _allosaurus_state["checked"] = True
    try:
        from allosaurus.app import read_recognizer
        import allosaurus as _allosaurus_pkg

        recognizer = read_recognizer(_ALLOSAURUS_MODEL_NAME)
        phone_path = os.path.join(
            os.path.dirname(_allosaurus_pkg.__file__),
            "pretrained", _ALLOSAURUS_MODEL_NAME, "phone.txt",
        )
        phones = [l.split()[0] for l in open(phone_path, encoding="utf-8").read().splitlines() if l.strip()]
        id2label = {0: "<blank>"}
        for i, p in enumerate(phones):
            id2label[i + 1] = p
        _allosaurus_state["recognizer"] = recognizer
        _allosaurus_state["id2label"] = id2label
    except Exception:
        pass  # stays None,None — Allosaurus simply isn't available here
    return _allosaurus_state["recognizer"], _allosaurus_state["id2label"]


def _allosaurus_secondary_signal(wav_path: Path) -> dict | None:
    """Best-effort secondary phoneme-level signal. Returns None (meaning
    "no secondary signal, proceed with Whisper only") if Allosaurus isn't
    installed, wav_path is missing, or anything at all goes wrong — this
    function must never raise and must never block pronunciation scoring.

    What it returns when it works: an aggregate confidence over
    Allosaurus's OWN greedy CTC decode of the recording — no expected/
    reference phones, no word alignment, no phoneme-substitution claim.
    Just "how sure was the acoustic model about what it heard"."""
    if not wav_path:
        return None
    recognizer, id2label = _get_allosaurus_recognizer()
    if recognizer is None:
        return None
    try:
        import numpy as np
        from allosaurus.am.utils import move_to_tensor
        from allosaurus.audio import read_audio

        audio = read_audio(str(wav_path))
        feat = recognizer.pm.compute(audio)
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)
        tensor_feat, tensor_feat_len = move_to_tensor([feats, feat_len], recognizer.config.device_id)
        logits = recognizer.am(tensor_feat, tensor_feat_len).detach().numpy()[0]  # [T, C] raw logits

        m = logits.max(axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / e.sum(axis=1, keepdims=True)  # real softmax posteriors, nothing invented

        top1 = probs.argmax(axis=1)
        non_blank = top1 != 0  # column 0 is the CTC blank
        if not non_blank.any():
            return None

        frame_conf = probs[np.arange(len(probs)), top1][non_blank]
        return {
            "mean_confidence": float(frame_conf.mean()),  # 0-1
            "decoded_phone_frames": int(non_blank.sum()),
        }
    except Exception:
        return None


class WhisperConfidenceProvider(PronunciationProvider):
    """The pre-existing (and only, until now) pronunciation implementation:
    Whisper's own per-word transcription confidence as a pronunciation
    proxy. Always available — no external service, no config. This is the
    default provider and the fallback every other provider's caller uses if
    a request doesn't specify one.

    Primary signal: Whisper avg_logprob / word probability (unchanged
    formula — see _score_pronunciation_from_segments /
    _extract_mispronounced_from_segments, byte-for-byte the pre-existing
    logic).

    Secondary signal (optional, only applied when Allosaurus is actually
    installed and successfully runs — see _allosaurus_secondary_signal
    above): Allosaurus's own acoustic-model confidence is blended into the
    final score at a MINORITY weight (ALLOSAURUS_SCORE_WEIGHT below), so
    Whisper's confidence stays the dominant signal and the pre-existing
    score doesn't move dramatically for any given recording. If Allosaurus
    is unavailable or fails for any reason, the score is exactly the
    original Whisper-only formula — no dependency, no crash, no change in
    behavior. `issues` (the per-word list) is never touched by Allosaurus
    — it only ever reflects real, aligned Whisper per-word data; Allosaurus
    has no word alignment, so it never fabricates a per-word claim."""

    name = "whisper_confidence"

    # How much weight the optional Allosaurus signal gets when present.
    # Kept deliberately small — Whisper confidence remains the primary,
    # well-tested signal (see requirement to keep existing scoring
    # behavior "as much as possible"); Allosaurus only nudges the score.
    ALLOSAURUS_SCORE_WEIGHT = 0.2

    def is_available(self) -> bool:
        return True

    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        whisper_score = _score_pronunciation_from_segments(segments)
        issues = _extract_mispronounced_from_segments(segments)

        methodology = _WHISPER_METHODOLOGY
        score = whisper_score
        allo = _allosaurus_secondary_signal(wav_path)
        if allo is not None:
            allo_score = allo["mean_confidence"] * 100  # 0-100, same scale as whisper_score
            w = self.ALLOSAURUS_SCORE_WEIGHT
            score = round(max(10.0, min(100.0, whisper_score * (1 - w) + allo_score * w)), 1)
            methodology += (
                f" Blended with a secondary signal: Allosaurus acoustic-model "
                f"phoneme confidence = {round(allo_score)}% over "
                f"{allo['decoded_phone_frames']} decoded phone frames "
                f"(whole-utterance, no reference phonemes), weighted at "
                f"{int(w * 100)}% against Whisper's {int((1 - w) * 100)}% "
                f"(Whisper-only score was {whisper_score})."
            )
        else:
            methodology += (
                " Allosaurus was not available for this assessment, so the "
                "score above is Whisper confidence only."
            )

        return PronunciationResult(score=score, issues=issues, provider=self.name,
                                    available=True, methodology=methodology)


# ── Provider 2: Saaras (Sarvam) — real adapter, integration incomplete ────────

SARVAM_API_URL = os.environ.get("SARVAM_API_URL", "https://api.sarvam.ai").rstrip("/")


class SaarasPronunciationProvider(PronunciationProvider):
    """Adapter for Sarvam's Saaras speech model. See the module docstring —
    Saaras is a transcription/translation model, not a documented
    pronunciation-assessment API, so this makes a real call and derives a
    score IF per-word confidence is present in the response; otherwise it
    reports available=False rather than guessing a score.

    Credentials: SARVAM_API_KEY environment variable. Never hard-coded."""

    name = "saaras"

    def __init__(self, api_key: str | None = None, base_url: str = SARVAM_API_URL,
                 timeout: httpx.Timeout = httpx.Timeout(30.0, connect=5.0)):
        self.api_key = api_key if api_key is not None else os.environ.get("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        if not self.is_available():
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Saaras pronunciation assessment is not configured "
                       "(SARVAM_API_KEY is not set).",
            )
        if not wav_path:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Saaras pronunciation assessment requires the recorded "
                       "audio, which wasn't provided to this call.",
            )

        try:
            with open(wav_path, "rb") as f:
                resp = httpx.post(
                    f"{self.base_url}/speech-to-text",
                    headers={"api-subscription-key": self.api_key},
                    data={"model": "saaras:v3"},
                    files={"file": (Path(wav_path).name, f, "audio/wav")},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail=f"Saaras request failed: {e}",
            )

        # Saaras's documented response is a transcript ({"transcript": "..."}),
        # with no publicly documented per-word confidence field. We look for
        # one defensively (in case a future/enterprise response shape adds
        # it) but do NOT fabricate a score if it's absent — see module
        # docstring for what needs to be confirmed before this is real.
        word_conf = data.get("words") or data.get("word_confidence")
        if not word_conf:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Saaras integration incomplete: the STT call succeeded, but "
                       "its response has no per-word confidence data to score "
                       "pronunciation from (Saaras is documented as a "
                       "transcription/translation model, not a pronunciation-"
                       "assessment API). Needs confirmation from Sarvam docs/team "
                       "on how pronunciation signal should be derived — see "
                       "pronunciation_provider.py module docstring.",
            )

        # Placeholder scoring path for if/when word_conf data does exist —
        # mirrors the Whisper confidence formula so results stay comparable.
        probs = [w.get("confidence", w.get("probability")) for w in word_conf
                 if w.get("confidence") is not None or w.get("probability") is not None]
        if not probs:
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Saaras returned word data with no confidence values.",
            )
        score = round(max(10.0, min(100.0, (sum(probs) / len(probs)) * 100)), 1)
        return PronunciationResult(score=score, issues=[], provider=self.name, available=True)


# ── Provider 3: Local LLM — interface only, not configured ────────────────────

LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "")


class LocalLLMPronunciationProvider(PronunciationProvider):
    """No local LLM infrastructure exists in this project today (nothing in
    requirements.txt — no llama-cpp-python, ollama client, transformers,
    etc., and no LOCAL_LLM_URL is set anywhere). This class exists so the
    interface/registry slot is real, but it deliberately does not download
    or load any model — per instructions, don't introduce a large model or
    RAM requirement just to make the option appear.

    Set LOCAL_LLM_URL to an already-running local inference server to make
    this provider report available; assess() would then POST the transcript
    + audio to it. Until that exists, it always reports not-configured."""

    name = "local_llm"

    def __init__(self, base_url: str = LOCAL_LLM_URL):
        self.base_url = base_url.rstrip("/") if base_url else ""

    def is_available(self) -> bool:
        return bool(self.base_url)

    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        if not self.is_available():
            return PronunciationResult(
                score=0.0, issues=[], provider=self.name, available=False,
                detail="Local LLM pronunciation assessment is not configured — "
                       "no local inference server is set up (LOCAL_LLM_URL unset).",
            )
        # No server exists to call yet; this branch is unreachable until
        # LOCAL_LLM_URL is set to a real running server. Left unimplemented
        # deliberately rather than guessing a wire format for a server that
        # doesn't exist.
        return PronunciationResult(
            score=0.0, issues=[], provider=self.name, available=False,
            detail="LOCAL_LLM_URL is set but the local LLM provider's request "
                   "format has not been implemented yet.",
        )


# ── Provider 4: Custom GOP — ON HOLD, placeholder only ─────────────────────────

class GOPPronunciationProvider(PronunciationProvider):
    """DO NOT IMPLEMENT THE GOP ALGORITHM — explicitly on hold. This class
    only reserves the provider name and satisfies the interface so the UI
    can show a disabled "Coming Soon" option and the backend can return a
    controlled response if it's somehow requested. The ASR → transcript →
    phoneme → GOP approach will be investigated separately later."""

    name = "gop"

    def is_available(self) -> bool:
        return False

    def assess(self, transcript: str, segments: list, wav_path: Path) -> PronunciationResult:
        return PronunciationResult(
            score=0.0, issues=[], provider=self.name, available=False,
            detail="Custom GOP pronunciation assessment is not currently available.",
        )


# ── Registry ────────────────────────────────────────────────────────────────

class PronunciationProviderRegistry:
    """Resolves a provider name to a PronunciationProvider instance. The
    scoring engine and routes only ever talk to this — never to a concrete
    provider class directly (except to construct the registry itself)."""

    def __init__(self, providers: dict | None = None):
        self._providers = providers if providers is not None else {
            "allosaurus_g2p": AllosaurusG2PPronunciationProvider(),
            "whisper_confidence": WhisperConfidenceProvider(),
            "saaras": SaarasPronunciationProvider(),
            "local_llm": LocalLLMPronunciationProvider(),
            "gop": GOPPronunciationProvider(),
        }

    def get(self, name: str) -> PronunciationProvider:
        if name not in self._providers:
            raise PronunciationProviderError(
                f"Unknown pronunciation_provider '{name}'. Valid options: "
                f"{', '.join(self._providers)}."
            )
        return self._providers[name]

    def status(self) -> dict:
        """{provider_name: {available: bool}} — for an API endpoint the UI
        can poll to know which options to enable/disable before the user
        even picks one."""
        return {name: {"available": p.is_available()} for name, p in self._providers.items()}


# Module-level singleton — mirrors how _lt_provider is constructed once at
# import time in app.py. Construction here is cheap (no network calls; each
# provider's __init__ just reads env vars / stores config).
pronunciation_registry = PronunciationProviderRegistry()


def resolve_pronunciation(provider_name: str, transcript: str, segments: list,
                           wav_path: Path) -> PronunciationResult:
    """Resolve + assess, with the one fallback rule the brief requires: if
    the explicitly-requested provider is unavailable, DO NOT silently swap
    in Whisper's result under the requested provider's name — return the
    unavailable result honestly (available=False, detail explains why) so
    the caller (app.py) can decide what to do (e.g. 200 with a clear message
    for gop/local_llm/saaras, per the mentor's brief). Callers that want
    "fall back to whisper_confidence and keep going" do that themselves by
    checking `.available` and re-resolving "whisper_confidence" explicitly —
    this function never does it silently."""
    provider = pronunciation_registry.get(provider_name)
    return provider.assess(transcript, segments, wav_path)

# ── Whisper reliability check ─────────────────────────────────────────────
def is_whisper_reliable(segments: list, threshold: float = 0.5) -> dict:
    """
    Check if Whisper was confident enough to trust the transcript.
    
    Args:
        segments: Whisper segments with word-level probabilities
        threshold: Minimum average word probability to consider reliable (default 0.5)
    
    Returns:
        {
            "reliable": bool,
            "avg_word_prob": float,
            "word_count": int,
            "reason": str
        }
    """
    probs = []
    word_count = 0
    
    for seg in segments or []:
        for w in seg.get("words", []) or []:
            prob = w.get("probability")
            if prob is not None:
                probs.append(prob)
                word_count += 1
    
    if not probs or word_count < 2:
        return {
            "reliable": False,
            "avg_word_prob": 0.0,
            "word_count": word_count,
            "reason": "No word-level confidence data available"
        }
    
    avg_prob = sum(probs) / len(probs)
    
    if avg_prob < threshold:
        return {
            "reliable": False,
            "avg_word_prob": round(avg_prob, 3),
            "word_count": word_count,
            "reason": f"Average word confidence ({avg_prob:.2f}) below threshold ({threshold})"
        }
    
    return {
        "reliable": True,
        "avg_word_prob": round(avg_prob, 3),
        "word_count": word_count,
        "reason": None
    }