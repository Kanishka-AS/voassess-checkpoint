"""
Speech-to-text provider architecture — same pattern as pronunciation_provider.py,
applied one layer earlier in the pipeline (audio -> transcript+segments, instead
of transcript+segments -> pronunciation score).

    STTProvider (interface)
        │
        ▼
    STTProviderRegistry
        │
        ├── WhisperSTTProvider  ("whisper") — fallback only now; the
        │                                      pre-existing transcribe_wav()
        │                                      logic, unchanged.
        └── SaarasSTTProvider   ("saaras")  — Sarvam's Saaras v3 speech-to-text.
                                               Primary/default provider (see
                                               app.py's DEFAULT_STT_PROVIDER).

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
  - REST endpoint hard cap: audio must be <= 30 seconds. LONG AUDIO: rather
    than Sarvam's separate async Batch API (POST /speech-to-text/job/v1,
    poll-then-fetch), SaarasSTTProvider.transcribe() below handles this
    itself by splitting audio over the cap into sequential, non-overlapping
    chunks (audio_utils.split_wav_into_chunks(), pure stdlib `wave`, no
    re-encoding) of SAARAS_CHUNK_SECONDS each, sending each chunk through
    the same synchronous /speech-to-text call in original order, and joining
    the raw per-chunk transcripts with a single space. No dedup/cleanup step
    runs on the joined text — see module docstring point 3 below. If any
    chunk's request fails, transcription stops there and available=False is
    returned with a detail naming which chunk failed and why; prior chunks'
    text is discarded rather than returned as a silently-partial transcript.
    Temp chunk files are removed in a `finally` block regardless of outcome.

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

from audio_utils import split_wav_into_chunks, cleanup_chunk_files

STT_PROVIDER_NAMES = ("whisper", "saaras")
SAARAS_REST_MAX_SECONDS = 30.0
# Per-chunk size used once audio exceeds SAARAS_REST_MAX_SECONDS. Kept
# comfortably under the 30s cap (not right up against it) so that merging a
# too-short trailing chunk into the previous one (see
# audio_utils.split_wav_into_chunks) can never push a chunk over the cap.
SAARAS_CHUNK_SECONDS = 25.0


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
    """The pre-existing (and, until this migration, only) transcription
    implementation. Always available — local model, no external service.
    Saaras is now DEFAULT_STT_PROVIDER (see app.py); Whisper is used when a
    caller explicitly requests "whisper" and it fails (app.py's
    resolve_stt() falls back to the default, Saaras, in that case). If
    Saaras itself (as the default) fails, resolve_stt() does NOT fall back
    to Whisper — that would silently swap in a transcript from a model
    known to normalize away the fillers/disfluencies this pipeline needs to
    preserve, so a Saaras failure is surfaced as a clear error instead."""

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
    <=30s per request — audio longer than that is chunked internally, see
    _transcribe_chunked()). See module docstring for the documented
    request/response shape and the word-level-timestamp limitation.

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

        # Duration check is dependency-free (stdlib `wave`, same approach as
        # audio_utils.wav_duration_seconds) to avoid a circular import on
        # app.py.
        duration = self._wav_duration_seconds(wav_path)
        if duration and duration > SAARAS_REST_MAX_SECONDS:
            return self._transcribe_chunked(wav_path, duration)

        transcript, err = self._transcribe_one(wav_path)
        if err:
            return STTResult(transcript="", segments=[], provider=self.name,
                              available=False, detail=err)
        # Deliberately segments=[] — see module docstring: Saaras's timestamps
        # are chunk/sentence-level with no confidence value, so there is no
        # honest way to fill Whisper's per-word {"word","probability"} shape
        # from it. Downstream pronunciation scoring (WhisperConfidenceProvider)
        # already handles empty segments gracefully (falls back to its
        # default score, no fabricated per-word issues).
        return STTResult(transcript=transcript, segments=[], provider=self.name, available=True)

    def _transcribe_one(self, wav_path: Path) -> tuple[str | None, str | None]:
        """One Saaras REST call for one <=30s WAV file/chunk. Returns
        (transcript, None) on success or (None, detail) on failure. Never
        raises — mirrors transcribe()'s own no-raise contract, since this is
        called both directly and once per chunk from _transcribe_chunked()."""
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

        except httpx.HTTPStatusError as e:
            try:
                err_body = e.response.json().get("error", {})
                reason = err_body.get("message") or e.response.text
            except Exception:
                reason = e.response.text

            return None, (
                f"Saaras STT request failed "
                f"({e.response.status_code}): {reason}"
            )

        except Exception as e:
            return None, f"Saaras STT request failed: {e}"

        transcript = (body.get("transcript") or "").strip()
        if not transcript:
            return None, ("Saaras STT returned an empty transcript — please speak "
                           "clearly and try again.")
        return transcript, None

    def _transcribe_chunked(self, wav_path: Path, duration: float) -> STTResult:
        """Long-audio path: split into sequential <=SAARAS_CHUNK_SECONDS
        chunks, transcribe each in order through _transcribe_one(), and join
        the raw results with a single space. No cleanup/dedup/normalization
        of the joined text — fillers, repetitions, and false starts that
        Saaras returned are passed through untouched (see requirement 3 in
        the module this backs). If a chunk fails, transcription stops there:
        available=False is returned with a detail naming the failing chunk,
        and any earlier chunks' text is discarded rather than returned as a
        transcript with a silent gap in it."""
        chunk_paths = split_wav_into_chunks(wav_path, SAARAS_CHUNK_SECONDS)

        if len(chunk_paths) == 1:
            # split_wav_into_chunks() decided chunking wasn't needed after
            # all (e.g. duration was right at the boundary) — no temp files
            # were created, nothing to clean up.
            transcript, err = self._transcribe_one(chunk_paths[0])
            if err:
                return STTResult(transcript="", segments=[], provider=self.name,
                                  available=False, detail=err)
            return STTResult(transcript=transcript, segments=[], provider=self.name, available=True)

        transcripts: list[str] = []
        try:
            for i, chunk_path in enumerate(chunk_paths, start=1):
                transcript, err = self._transcribe_one(chunk_path)
                if err:
                    return STTResult(
                        transcript="", segments=[], provider=self.name, available=False,
                        detail=(f"Saaras STT failed on chunk {i}/{len(chunk_paths)} of this "
                                f"{duration:.1f}s recording ({err}). No transcript is "
                                f"returned for this recording — the {i - 1} chunk(s) that "
                                f"did transcribe successfully are not silently returned as "
                                f"if they were the complete result."),
                    )
                transcripts.append(transcript)
        finally:
            cleanup_chunk_files(chunk_paths, wav_path)

        merged = " ".join(transcripts).strip()
        if not merged:
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail="Saaras STT returned an empty transcript across all chunks — "
                       "please speak clearly and try again.",
            )
        # Deliberately segments=[] here too, same reasoning as the
        # single-request path above.
        return STTResult(transcript=merged, segments=[], provider=self.name, available=True)

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
