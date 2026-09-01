"""
Small, dependency-free audio helpers shared by app.py and the STT providers.

wav_duration_seconds() reads a WAV file's actual frame count / sample rate
via the stdlib `wave` module — no ffprobe, no numpy. This is the
authoritative duration source for /assess, /assess/stage, and
/debug/analyze-audio (see save_and_convert() + score_free_speech() in
app.py): the browser-reported `duration` form field is an unreliable
wall-clock value, so every route computes this from the actual converted
WAV instead and only falls back to the client value if the WAV is somehow
unreadable.

Note: stt_provider.SaarasSTTProvider has its own private copy of this exact
logic (`_wav_duration_seconds`), kept local there deliberately to avoid a
circular import between stt_provider.py and app.py. If this function's
behavior ever changes, that copy needs to change too.
"""
from __future__ import annotations

import wave
from pathlib import Path


def wav_duration_seconds(path) -> float:
    """Return a WAV file's duration in seconds, or 0.0 if the file is
    missing, unreadable, not a valid WAV, or has zero frames/sample rate.
    Never raises."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return 0.0
    if rate <= 0 or frames <= 0:
        return 0.0
    return frames / rate
