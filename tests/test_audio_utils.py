"""
Tests for audio_utils.wav_duration_seconds().

Builds real (tiny) WAV files with the stdlib `wave` module at a known
sample rate/frame count, so the expected duration is exact — no fixtures,
no external audio files needed.
"""
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_utils import wav_duration_seconds


def _write_wav(path, seconds: float, rate: int = 16000, channels: int = 1, sampwidth: int = 2):
    n_frames = round(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(rate)
        wf.writeframes(b"\x00" * n_frames * sampwidth * channels)
    return n_frames


def test_known_duration_16k_mono(tmp_path):
    path = tmp_path / "a.wav"
    _write_wav(path, seconds=2.94, rate=16000)
    d = wav_duration_seconds(path)
    assert abs(d - 2.94) < 0.001  # exact: 47040 frames / 16000 = 2.94


def test_known_duration_different_rate(tmp_path):
    path = tmp_path / "b.wav"
    _write_wav(path, seconds=1.5, rate=48000)
    d = wav_duration_seconds(path)
    assert abs(d - 1.5) < 0.0001


def test_zero_length_wav_returns_zero(tmp_path):
    path = tmp_path / "empty.wav"
    _write_wav(path, seconds=0.0, rate=16000)
    assert wav_duration_seconds(path) == 0.0


def test_missing_file_returns_zero(tmp_path):
    assert wav_duration_seconds(tmp_path / "does_not_exist.wav") == 0.0


def test_not_a_wav_file_returns_zero(tmp_path):
    path = tmp_path / "not_audio.txt"
    path.write_text("this is not a wav file")
    assert wav_duration_seconds(path) == 0.0


def test_matches_the_actual_debug_recording_case():
    """Regression anchor for the 2026-08-30 debug session: client reported
    6.0596s, the real converted WAV was 2.94s. This is exactly the kind of
    file wav_duration_seconds() must report correctly regardless of what
    any client claims."""
    path_str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "recordings", "debug_20260830_143103.wav",
    )
    if not os.path.exists(path_str):
        return  # sample recording not present in this checkout — skip silently
    d = wav_duration_seconds(path_str)
    assert 2.9 <= d <= 3.0
