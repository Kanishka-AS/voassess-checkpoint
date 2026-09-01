"""
Speech-to-text provider architecture — same pattern as pronunciation_provider.py,
applied one layer earlier in the pipeline (audio -> transcript+segments, instead
of transcript+segments -> pronunciation score).

    STTProvider (interface)
        │
        ▼
    STTProviderRegistry
        │
        ├── WhisperSTTProvider  ("whisper") — default/fallback, the pre-existing
        │                                      transcribe_wav() logic, unchanged.
        └── SaarasSTTProvider   ("saaras")  — Sarvam's Saaras v3 speech-to-text.

app.py never branches on provider name here either — it asks the registry for
a provider and calls `.transcribe()`.

────────────────────────────────────────────────────────────────────────────
Saaras as STT (unlike as a pronunciation source — see pronunciation_provider.py)
is a well-documented, real fit: Sarvam's `/speech-to-text` endpoint (model
`saaras:v3`) IS a transcription API. Confirmed from Sarvam's docs:

  - POST https://api.sarvam.ai/speech-to-text
  - Auth header: api-subscription-key: <SARVAM_API_KEY>
  - multipart/form-data: file=<audio>, model="saaras:v3", mode="transcribe"
    (default), language_code=<BCP-47> (optional — omitted lets Saaras
    auto-detect and return language_probability), with_timestamps=true
    (optional, enables the `timestamps` object in the response)
  - Response (success): {"transcript": "...", "language_code": "...",
    "request_id": "...", "timestamps": {...}}  (timestamps only present
    when with_timestamps=true was sent)
  - Response (error): {"error": {"message": "...", "code": "...",
    "request_id": "..."}}
  - REST endpoint hard cap: audio must be <= 30 seconds. Longer audio needs
    Sarvam's separate Batch API (POST /speech-to-text/job/v1, async,
    poll-then-fetch) — not implemented here; see is_available()/transcribe()
    below, which refuse (available=False) rather than send an oversized
    request that Sarvam will reject.

KNOWN LIMITATION — timestamps are chunk/sentence-level, not word-level, and
carry no confidence value (documented explicitly: "each entry covers a
sentence or phrase, not an individual word" / word-level timestamps are not
in the REST response shape at all). This means WhisperConfidenceProvider
(the pronunciation scorer) cannot compute a real per-word confidence score
from Saaras-sourced segments — SaarasSTTProvider deliberately returns
segments=[] rather than fabricating per-word data, and
WhisperConfidenceProvider's existing default-72.0/no-issues behavior for
empty segments applies. If word-by-word pronunciation flagging needs to work
with Saaras as the transcription source, that requires either the
(separately in-progress) SaarasPronunciationProvider producing its own
signal, or GOP down the line — not something STT alone can supply here.

Configuration:
    SARVAM_API_KEY   Sarvam API subscription key. Required for "saaras".
                      Shared with pronunciation_provider.py's Saaras adapter
                      — same account, same key, two different endpoints.
    SARVAM_API_URL   Base URL for the Sarvam API. Defaults to
                      "https://api.sarvam.ai".
    SARVAM_LANGUAGE_CODE   Optional BCP-47 code (e.g. "en-IN"). If unset,
                      language_code is omitted and Saaras auto-detects.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx

STT_PROVIDER_NAMES = ("whisper", "saaras")
SAARAS_REST_MAX_SECONDS = 30.0


class STTProviderError(Exception):
    """Raised for a request naming an unknown/invalid STT provider."""


@dataclass
class STTResult:
    """Normalized result every STT provider returns."""
    transcript: str
    segments: list          # Whisper-shaped where possible: [{"avg_logprob":..,
                             # "words":[{"word":.., "probability":..}]}]. May be
                             # [] for providers (Saaras) that don't expose that
                             # level of detail — callers must handle that, not
                             # assume Whisper's shape is always populated.
    provider: str
    available: bool
    detail: str | None = None


class STTProvider(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool:
        """Cheap, synchronous check — no network call."""

    @abstractmethod
    def transcribe(self, wav_path: Path) -> STTResult:
        """Transcribe one 16kHz mono WAV file. Must not raise for a merely-
        unavailable provider (not configured, audio too long for this
        provider's mode, etc.) — return available=False with `detail`
        instead. Exceptions are reserved for genuine bugs."""


# ── Provider 1: Whisper (existing behavior, moved here as-is) ─────────────────

class WhisperSTTProvider(STTProvider):
    """The pre-existing (and, until now, only) transcription implementation.
    Always available — local model, no external service. Default/fallback
    provider."""

    name = "whisper"

    def __init__(self, model):
        # The already-loaded Whisper model (app.py's module-level `_whisper`,
        # loaded once at import time) — not reloaded here, just reused, same
        # as before this migration.
        self._model = model

    def is_available(self) -> bool:
        return True

    def transcribe(self, wav_path: Path) -> STTResult:
        # Unchanged from the pre-existing transcribe_wav(): word_timestamps=True
        # for per-word probability.
        #
        # condition_on_previous_text=False: by default Whisper conditions each
        # segment's decoding on its (greedy) transcription of the previous
        # segment. On short/ambiguous clips — especially ones with only one
        # or two segments, where there's little reliable "previous text" to
        # condition on in the first place — this is a documented cause of
        # early truncation/hallucination (the model locks onto a guess for
        # the first segment and that guess suppresses/distorts decoding of
        # what follows). Disabling it makes each segment decode
        # independently. No other transcription setting (model, language,
        # word_timestamps) changed from the pre-migration behavior.
        result = self._model.transcribe(
            str(wav_path), language="en", word_timestamps=True,
            condition_on_previous_text=False,
        )
        transcript = result["text"].strip()
        segments = result.get("segments", [])
        if not transcript:
            return STTResult(transcript="", segments=[], provider=self.name,
                              available=False,
                              detail="Could not transcribe — please speak clearly and try again.")
        return STTResult(transcript=transcript, segments=segments, provider=self.name, available=True)


# ── Provider 2: Saaras (Sarvam) — real, documented STT integration ────────────

SARVAM_API_URL = os.environ.get("SARVAM_API_URL", "https://api.sarvam.ai").rstrip("/")


class SaarasSTTProvider(STTProvider):
    """Adapter for Sarvam's Saaras v3 speech-to-text (REST, synchronous,
    <=30s audio). See module docstring for the documented request/response
    shape and the word-level-timestamp limitation.

    Credentials: SARVAM_API_KEY environment variable. Never hard-coded."""

    name = "saaras"

    def __init__(self, api_key: str | None = None, base_url: str = SARVAM_API_URL,
                 language_code: str | None = None,
                 timeout: httpx.Timeout = httpx.Timeout(30.0, connect=5.0)):
        self.api_key = api_key if api_key is not None else os.environ.get("SARVAM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.language_code = (language_code if language_code is not None
                               else os.environ.get("SARVAM_LANGUAGE_CODE", "") or None)
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def transcribe(self, wav_path: Path) -> STTResult:
        if not self.is_available():
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail="Saaras STT is not configured (SARVAM_API_KEY is not set).",
            )
        if not wav_path or not Path(wav_path).exists():
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail="Saaras STT requires the recorded audio, which wasn't provided.",
            )

        # The synchronous REST endpoint caps audio at 30 seconds — refuse
        # cleanly rather than send a request Sarvam will reject. (Longer
        # audio needs Sarvam's separate async Batch API, not implemented
        # here.) Duration check is dependency-free (stdlib `wave`, same
        # approach as audio_utils.wav_duration_seconds) to avoid a circular
        # import on app.py.
        duration = self._wav_duration_seconds(wav_path)
        if duration and duration > SAARAS_REST_MAX_SECONDS:
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail=f"Saaras's synchronous STT endpoint accepts audio up to "
                       f"{SAARAS_REST_MAX_SECONDS:.0f}s; this recording is "
                       f"{duration:.1f}s. Longer recordings need Sarvam's Batch "
                       f"API, which isn't implemented yet.",
            )

        data = {"model": "saaras:v3", "mode": "transcribe", "with_timestamps": "true"}
        if self.language_code:
            data["language_code"] = self.language_code

        try:
            with open(wav_path, "rb") as f:
                resp = httpx.post(
                    f"{self.base_url}/speech-to-text",
                    headers={"api-subscription-key": self.api_key},
                    data=data,
                    files={"file": (Path(wav_path).name, f, "audio/wav")},
                    timeout=self.timeout,
                )
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail=f"Saaras STT request failed: {e}",
            )

        transcript = (body.get("transcript") or "").strip()
        if not transcript:
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail="Saaras STT returned an empty transcript — please speak "
                       "clearly and try again.",
            )

        # Deliberately segments=[] — see module docstring: Saaras's timestamps
        # are chunk/sentence-level with no confidence value, so there is no
        # honest way to fill Whisper's per-word {"word","probability"} shape
        # from it. Downstream pronunciation scoring (WhisperConfidenceProvider)
        # already handles empty segments gracefully (falls back to its
        # default score, no fabricated per-word issues).
        return STTResult(transcript=transcript, segments=[], provider=self.name, available=True)

    @staticmethod
    def _wav_duration_seconds(wav_path) -> float:
        import wave
        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
        except (wave.Error, EOFError, FileNotFoundError):
            return 0.0
        if rate <= 0 or frames <= 0:
            return 0.0
        return frames / rate


# ── Registry ────────────────────────────────────────────────────────────────

class STTProviderRegistry:
    def __init__(self, providers: dict):
        # No hard-coded default construction here (unlike
        # PronunciationProviderRegistry) — WhisperSTTProvider needs the
        # already-loaded model injected, so app.py constructs this registry
        # explicitly at import time, after `_whisper` exists.
        self._providers = providers

    def get(self, name: str) -> STTProvider:
        if name not in self._providers:
            raise STTProviderError(
                f"Unknown stt_provider '{name}'. Valid options: {', '.join(self._providers)}."
            )
        return self._providers[name]

    def status(self) -> dict:
        return {name: {"available": p.is_available()} for name, p in self._providers.items()}
