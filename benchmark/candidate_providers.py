"""
Benchmark-only STT provider adapters — Parakeet and Moonshine.

Same reasoning as before: not imported by app.py, doesn't touch the
production pipeline, follows the exact STTProvider/STTResult contract from
stt_provider.py so run_benchmark.py can compare all providers the same way.

REVISED again after checking real vendor/package docs (no hosted inference,
no GPU-heavy stacks):

  * Parakeet: switched from NVIDIA's hosted build.nvidia.com API to a
    genuinely local, CPU-only runtime via `onnx-asr`
    (https://github.com/istupakov/onnx-asr) — a pure-Python package with
    minimal deps (numpy, onnxruntime, huggingface-hub). No PyTorch, no
    NeMo, no GPU. It loads a purpose-built ONNX export of
    nvidia/parakeet-tdt-0.6b-v2 from Hugging Face (istupakov's onnx-asr
    conversion) once and reuses it for every sample. First run needs
    network to download the model (~2.4 GB fp32); after that it's cached
    under the HF cache dir and runs fully offline. No API key involved
    anywhere.
  * Moonshine: fixed the import/API to match what's actually installed —
    `moonshine_voice==0.1.5` (NOT `useful-moonshine`, NOT `moonshine_onnx`).
    That package has no `moonshine.transcribe()` free function; it exposes
    a `Transcriber` class with a `.transcribe(audio, sample_rate=...)`
    method returning a `Transcript` with `.lines` of `TranscriptLine`
    objects, plus `get_model_path()` / `load_wav_file()` helpers. Local,
    offline, no key.

Configuration:
    PARAKEET_ONNX_MODEL   Optional. onnx-asr model name. Default:
                          "nemo-parakeet-tdt-0.6b-v2" (English). Set to
                          "nemo-parakeet-tdt-0.6b-v3" for the multilingual
                          v3 checkpoint instead.
    (Neither provider needs an API key — both run locally/offline.)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from stt_provider import STTProvider, STTResult


@dataclass
class TimedSTTResult:
    result: STTResult
    latency_seconds: float
    word_level_timestamps: bool
    word_level_confidence: bool


# ── Candidate: NVIDIA Parakeet TDT 0.6B — LOCAL, CPU-only, offline ────────────
#
# `pip install onnx-asr[cpu,hub]`
# Real API: `import onnx_asr; model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")`
# -> `model.recognize(wav_path)` -> str.
#
# onnx-asr (https://github.com/istupakov/onnx-asr) is a pure-Python +
# onnxruntime package — no PyTorch, no NeMo, no GPU required. It loads
# istupakov's ONNX conversion of the real NVIDIA checkpoint from Hugging
# Face (downloaded once, then cached locally) and runs greedy TDT decoding
# itself. providers=["CPUExecutionProvider"] pins it to CPU explicitly so
# it never silently reaches for a GPU provider if one happens to exist.
#
# HONESTY NOTE on timestamps/confidence: onnx-asr can return token-level
# timestamps/logprobs via `.with_timestamps()`, but those are sub-word
# tokens, not the word-level offsets the old hosted Riva path returned.
# Rather than fake word boundaries by guessing token->word grouping, this
# adapter reports plain text only (segments=[]) — same honesty policy the
# Moonshine adapter below already uses for its own missing granularity.

DEFAULT_PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"  # English; set PARAKEET_ONNX_MODEL=nemo-parakeet-tdt-0.6b-v3 for multilingual


class ParakeetSTTProvider(STTProvider):
    name = "parakeet"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get("PARAKEET_ONNX_MODEL", DEFAULT_PARAKEET_MODEL)
        self._import_error = None
        self._model = None       # loaded lazily, once, reused for every sample
        self._model_error = None

    def is_available(self) -> bool:
        try:
            import onnx_asr  # noqa: F401
            return True
        except Exception as e:
            self._import_error = str(e)
            return False

    def _load_model(self):
        if self._model is not None:
            return self._model
        import onnx_asr
        # CPU-only per requirements — pin explicitly rather than letting
        # onnxruntime pick whatever execution providers are installed.
        self._model = onnx_asr.load_model(self.model_name, providers=["CPUExecutionProvider"])
        return self._model

    def transcribe(self, wav_path: Path) -> STTResult:
        if not self.is_available():
            return STTResult(transcript="", segments=[], provider=self.name, available=False,
                              detail=f"onnx-asr not installed/importable: {self._import_error}. "
                                     f"Install with: pip install onnx-asr[cpu,hub]")
        try:
            model = self._load_model()
        except Exception as e:
            return STTResult(transcript="", segments=[], provider=self.name, available=False,
                              detail=f"Failed to load local Parakeet model ({self.model_name}): {e}. "
                                     f"First run downloads the ONNX checkpoint from Hugging Face "
                                     f"(needs network once); after that it's cached and fully offline.")
        try:
            transcript = model.recognize(str(wav_path))
            transcript = (transcript or "").strip()
            if not transcript:
                return STTResult(transcript="", segments=[], provider=self.name, available=False,
                                  detail="Parakeet (local) returned an empty transcript.")
            # Plain text only — see HONESTY NOTE above re: word-level granularity.
            return STTResult(transcript=transcript, segments=[], provider=self.name, available=True)
        except Exception as e:
            return STTResult(transcript="", segments=[], provider=self.name, available=False,
                              detail=f"Parakeet local inference failed: {e}")


# ── Candidate: Moonshine — local, CPU, offline (moonshine_voice==0.1.5) ───────
#
# `pip install moonshine-voice` (already installed per the report above).
# Real API (confirmed against the actually-installed package, NOT
# useful-moonshine / moonshine_onnx, which don't apply here):
#
#   from moonshine_voice import Transcriber, get_model_path, load_wav_file
#   model_path = get_model_path()
#   transcriber = Transcriber(model_path)
#   audio, sample_rate = load_wav_file(wav_path)
#   transcript = transcriber.transcribe(audio, sample_rate=sample_rate)
#   text = " ".join(line.text for line in transcript.lines)
#
# No API key, no network — genuinely offline once the model is cached
# locally (get_model_path() downloads/locates it).
#
# ROBUSTNESS NOTE: get_model_path()/Transcriber() signatures have shifted
# slightly across moonshine-voice releases (some versions want a language
# code, some don't; some need an explicit model_arch, some auto-detect).
# The calls below try the no-argument / positional-only shape first — the
# one that matches the attributes you confirmed are exposed in 0.1.5 — and
# fall back to a couple of other plausible shapes before giving up with a
# clear, actionable error rather than a bare TypeError.

class MoonshineSTTProvider(STTProvider):
    name = "moonshine"

    def __init__(self):
        self._import_error = None
        self._transcriber = None   # loaded lazily, once, reused for every sample
        self._load_detail = None

    def is_available(self) -> bool:
        try:
            import moonshine_voice  # noqa: F401
            return True
        except Exception as e:
            self._import_error = str(e)
            return False

    def _get_model_path(self, get_model_path):
        """get_model_path() signature has varied across moonshine-voice
        releases; try the shapes most consistent with a 0.1.x install."""
        attempts = [
            ((), {}),
            (("en",), {}),
            ((), {"language": "en"}),
        ]
        last_err = None
        for args, kwargs in attempts:
            try:
                return get_model_path(*args, **kwargs)
            except TypeError as e:
                last_err = e
                continue
        raise last_err

    def _build_transcriber(self, Transcriber, model_path):
        last_err = None
        for args, kwargs in ((model_path,), {}), ((), {"model_path": model_path}):
            try:
                return Transcriber(*args, **kwargs)
            except TypeError as e:
                last_err = e
                continue
        raise last_err

    def _load_transcriber(self):
        if self._transcriber is not None:
            return self._transcriber
        from moonshine_voice import Transcriber, get_model_path
        model_path = self._get_model_path(get_model_path)
        self._transcriber = self._build_transcriber(Transcriber, model_path)
        return self._transcriber

    def transcribe(self, wav_path: Path) -> STTResult:
        if not self.is_available():
            return STTResult(
                transcript="", segments=[], provider=self.name, available=False,
                detail=f"moonshine_voice not installed/importable: {self._import_error}. "
                       f"Install with: pip install moonshine-voice",
            )
        try:
            transcriber = self._load_transcriber()
        except Exception as e:
            return STTResult(transcript="", segments=[], provider=self.name, available=False,
                              detail=f"Failed to initialize moonshine_voice Transcriber: {e}. "
                                     f"get_model_path()/Transcriber() signatures vary by moonshine-voice "
                                     f"version — run "
                                     f"`python3 -c \"import inspect,moonshine_voice as m; "
                                     f"print(inspect.signature(m.get_model_path)); "
                                     f"print(inspect.signature(m.Transcriber.__init__))\"` "
                                     f"and adjust the two _*() helpers above to match.")
        try:
            from moonshine_voice import load_wav_file
            audio, sample_rate = load_wav_file(str(wav_path))
            transcript = transcriber.transcribe(audio, sample_rate=sample_rate)
            text = " ".join(line.text for line in getattr(transcript, "lines", [])).strip()
            if not text:
                return STTResult(transcript="", segments=[], provider=self.name, available=False,
                                  detail="Moonshine (local) returned an empty transcript.")
            # moonshine_voice's TranscriptLine carries start_time but no
            # per-word timestamps/confidence — segments=[] is honest here,
            # same reasoning as SaarasSTTProvider's segments=[].
            return STTResult(transcript=text, segments=[], provider=self.name, available=True)
        except Exception as e:
            return STTResult(transcript="", segments=[], provider=self.name, available=False,
                              detail=f"Moonshine inference failed: {e}")


def timed_transcribe(provider: STTProvider, wav_path: Path) -> TimedSTTResult:
    """Wrap any STTProvider.transcribe() call with a wall-clock timer and
    detect timestamp/confidence granularity straight from the returned
    segments — no per-provider special-casing needed here."""
    t0 = time.perf_counter()
    result = provider.transcribe(wav_path)
    latency = time.perf_counter() - t0

    has_words = any(seg.get("words") for seg in (result.segments or []))
    word_ts = has_words and any("start" in w or "end" in w for seg in result.segments for w in seg.get("words", []))
    word_conf = has_words and any("probability" in w for seg in result.segments for w in seg.get("words", []))

    return TimedSTTResult(result=result, latency_seconds=latency,
                           word_level_timestamps=bool(word_ts), word_level_confidence=bool(word_conf))