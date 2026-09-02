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

from audio_utils import wav_duration_seconds, analyze_pauses


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


# ── analyze_pauses() ────────────────────────────────────────────────────────

def _seg(words):
    """Build a single Whisper-style segment dict from (word, start, end) tuples."""
    return {"words": [{"word": w, "start": s, "end": e} for (w, s, e) in words]}


def test_no_pauses_below_short_threshold():
    """Natural word-to-word gaps under SHORT_PAUSE_SECONDS must not be
    counted as pauses at all — otherwise every recording would register
    dozens of 'pauses' that are just normal articulation gaps."""
    segs = [_seg([("hi", 0.0, 0.2), ("there", 0.25, 0.5), ("friend", 0.55, 0.9)])]
    r = analyze_pauses(segs, duration=1.0)
    assert r["available"] is True
    assert r["pause_count"] == 0
    assert r["long_pause_count"] == 0


def test_counts_pause_between_short_and_long_threshold():
    segs = [_seg([("one", 0.0, 0.2), ("two", 0.7, 0.9)])]  # 0.5s gap: pause, not "long"
    r = analyze_pauses(segs, duration=1.0)
    assert r["pause_count"] == 1
    assert r["long_pause_count"] == 0


def test_counts_long_hesitation_pause():
    segs = [_seg([("one", 0.0, 0.2), ("two", 2.0, 2.2)])]  # 1.8s gap: long pause
    r = analyze_pauses(segs, duration=3.0)
    assert r["pause_count"] == 1
    assert r["long_pause_count"] == 1
    assert r["avg_pause_ms"] == 1800.0


def test_unavailable_when_no_timestamped_words():
    """Fewer than two timestamped words (e.g. an STT provider with no
    word-level timing, or empty segments) must report available=False
    rather than fabricating zeros that look like 'measured, zero pauses'."""
    assert analyze_pauses([], duration=5.0)["available"] is False
    assert analyze_pauses([{"words": []}], duration=5.0)["available"] is False
    assert analyze_pauses([_seg([("only", 0.0, 0.2)])], duration=5.0)["available"] is False


def test_words_out_of_order_are_sorted_before_gap_calculation():
    """Segments/words aren't guaranteed to already be in time order in every
    caller's data — analyze_pauses must sort by start time itself rather
    than trusting input order, or gaps could come out negative/nonsensical."""
    segs = [_seg([("second", 2.0, 2.2), ("first", 0.0, 0.2)])]
    r = analyze_pauses(segs, duration=3.0)
    assert r["pause_count"] == 1
    assert r["avg_pause_ms"] > 0
