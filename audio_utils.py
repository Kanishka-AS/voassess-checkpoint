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

import shutil
import tempfile
import wave
from pathlib import Path
from typing import List, Dict, Any, Optional


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


# ── WAV chunking for STT providers with a per-request duration cap ────────
#
# Used by stt_provider.SaarasSTTProvider for audio longer than Saaras's
# synchronous REST cap (see SAARAS_REST_MAX_SECONDS there). Pure stdlib
# `wave` — no ffmpeg subprocess, no numpy: each chunk is a raw copy of a
# contiguous block of frames from the source file, written back out with
# the exact same channel count / sample width / frame rate, so there is no
# re-encoding step that could corrupt audio or drift the sample rate.
# Memory cost is one chunk's worth of frames at a time (tens of KB for a
# ~25s 16kHz mono chunk), not the whole file.

def split_wav_into_chunks(wav_path, chunk_seconds: float,
                           min_last_chunk_seconds: float = 1.0) -> List[Path]:
    """Split a WAV file into sequential, non-overlapping chunks of at most
    `chunk_seconds` each, in original order, preserving sample rate/channels/
    sample width exactly.

    If the source is already <= chunk_seconds long, returns [Path(wav_path)]
    unchanged — no copy, no temp dir — so callers can call this
    unconditionally without paying a split cost for short audio.

    Otherwise writes chunk files into a fresh temp directory (one per call)
    and returns their paths in order. A too-short trailing chunk (shorter
    than `min_last_chunk_seconds`) is merged into the previous chunk instead
    of being sent on its own, so a provider never receives a near-empty
    final request.

    Callers own cleanup of the temp directory this creates — see
    cleanup_chunk_files() below. Never raises for a missing/corrupt file;
    returns [Path(wav_path)] in that case, same as the "short enough"
    passthrough, so the caller's own transcribe attempt on it produces the
    real error.
    """
    wav_path = Path(wav_path)
    try:
        with wave.open(str(wav_path), "rb") as src:
            n_channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            framerate = src.getframerate()
            total_frames = src.getnframes()
            if framerate <= 0 or total_frames <= 0:
                return [wav_path]
            duration = total_frames / framerate
            if duration <= chunk_seconds:
                return [wav_path]

            frames_per_chunk = max(1, int(chunk_seconds * framerate))
            min_last_frames = int(min_last_chunk_seconds * framerate)

            # Frame counts per chunk, then merge a too-short trailing chunk
            # into the one before it.
            boundaries = []
            remaining = total_frames
            while remaining > 0:
                take = min(frames_per_chunk, remaining)
                boundaries.append(take)
                remaining -= take
            if len(boundaries) >= 2 and boundaries[-1] < min_last_frames:
                boundaries[-2] += boundaries[-1]
                boundaries.pop()

            out_dir = Path(tempfile.mkdtemp(prefix="sttchunks_"))
            chunk_paths = []
            src.rewind()
            for idx, n_frames in enumerate(boundaries):
                data = src.readframes(n_frames)
                chunk_path = out_dir / f"chunk_{idx:04d}.wav"
                with wave.open(str(chunk_path), "wb") as dst:
                    dst.setnchannels(n_channels)
                    dst.setsampwidth(sampwidth)
                    dst.setframerate(framerate)
                    dst.writeframes(data)
                chunk_paths.append(chunk_path)
            return chunk_paths
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return [wav_path]


def cleanup_chunk_files(chunk_paths: List[Path], original_path) -> None:
    """Delete the temp directory created by split_wav_into_chunks(), if any.
    No-op (never touches the caller's original file) when chunk_paths is
    just the unchanged passthrough of `original_path` — the "short enough,
    no chunking happened" case. Never raises."""
    if not chunk_paths:
        return
    if len(chunk_paths) == 1 and Path(chunk_paths[0]) == Path(original_path):
        return
    chunk_dir = Path(chunk_paths[0]).parent
    shutil.rmtree(chunk_dir, ignore_errors=True)


# ── Pause / hesitation analysis from word-level timestamps ────────────────
#
# Whisper is already called with word_timestamps=True (see stt_provider.py),
# so every word in `segments[i]["words"]` carries real "start"/"end" audio
# offsets. Before this addition, nothing in the scoring pipeline read those
# fields for anything other than display (debug word-timing tables) and a
# text-based filler cross-check — the actual audio gaps between words were
# computed by Whisper and then thrown away. "Hesitations / pauses" (as
# distinct from filler *words*) was therefore only ever measured via a
# text-level immediate-word-repetition regex ("I I", "the the"), which
# misses the far more common disfluency pattern in real learner speech: a
# long silent gap while the speaker searches for a word, with no repeated
# or filled token at all. This function fixes that gap for free — it's pure
# arithmetic over data STT already produced, no extra model call, no extra
# CPU/RAM.
SHORT_PAUSE_SECONDS = 0.35   # below this: normal word/breath boundary, not counted
LONG_PAUSE_SECONDS = 1.2     # at/above this: a hesitation-level pause


def analyze_pauses(segments: list, duration: float) -> dict:
    """Real, audio-grounded pause statistics from Whisper's per-word
    start/end timestamps. Returns available=False (rather than fabricating
    zeros) when no provider supplied word-level timing for this request
    (e.g. Saaras STT — see stt_provider.py) so callers can tell "no pauses
    measured" apart from "no timing data to measure from"."""
    spans = []
    for seg in segments or []:
        for w in seg.get("words", []) or []:
            s, e = w.get("start"), w.get("end")
            if s is not None and e is not None and e >= s:
                spans.append((s, e))
    if len(spans) < 2:
        return {
            "available": False, "pause_count": 0, "long_pause_count": 0,
            "avg_pause_ms": 0.0, "total_pause_seconds": 0.0,
            "pause_rate_per_min": 0.0,
            "filler_gaps": [],  # NEW: for audio-based filler detection
        }
    spans.sort()
    gaps = [max(0.0, s2 - e1) for (_, e1), (s2, _) in zip(spans, spans[1:])]
    pauses = [g for g in gaps if g >= SHORT_PAUSE_SECONDS]
    long_pauses = [g for g in pauses if g >= LONG_PAUSE_SECONDS]
    total_pause = sum(pauses)
    
    # NEW: Identify gaps that are likely filler pauses (shorter than regular pauses)
    filler_gaps = [g for g in gaps if 0.08 < g < 0.35]
    
    return {
        "available": True,
        "pause_count": len(pauses),
        "long_pause_count": len(long_pauses),
        "avg_pause_ms": round((total_pause / len(pauses)) * 1000, 0) if pauses else 0.0,
        "total_pause_seconds": round(total_pause, 2),
        "pause_rate_per_min": round(len(pauses) / max(duration, 1) * 60, 1),
        "filler_gaps": filler_gaps,  # NEW
    }


# ── Audio-based filler detection ────────────────────────────────────────────
#
# Whisper often drops fillers like "um", "uh", "er" from the transcript
# because it treats them as disfluencies. This function detects fillers
# directly from the audio using energy-based analysis, independent of
# Whisper's transcript.
#
# The approach is simple and CPU-cheap:
# 1. Split audio into short windows (100ms)
# 2. Compute RMS energy for each window
# 3. Find low-energy segments (gaps between words)
# 4. Flag short (80-400ms) low-energy segments as potential fillers


def detect_fillers_from_audio(wav_path: Path, sample_rate: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Detect filler sounds (um, uh, er) directly from audio.
    
    Uses a simple energy-based approach:
    1. Split audio into short windows
    2. Detect low-energy segments (between words)
    3. Flag short, low-energy segments as potential fillers
    
    This is a simplified approach. For production, use Silero VAD + filler model.
    
    Args:
        wav_path: Path to WAV file
        sample_rate: Override sample rate (auto-detect if None)
    
    Returns:
        List of dicts: [{'start': float, 'end': float, 'duration': float, 'type': str, 'confidence': float}]
    """
    try:
        import numpy as np
    except ImportError:
        return []
    
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            sr = wf.getframerate()
            channels = wf.getnchannels()
    except Exception as e:
        return []
    
    # If stereo, convert to mono
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    
    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Window size: 100ms
    window_size = int(sr * 0.1)
    hop_size = int(sr * 0.05)  # 50ms hop
    
    # Compute RMS energy per window
    energies = []
    timestamps = []
    for i in range(0, len(audio) - window_size, hop_size):
        chunk = audio[i:i+window_size]
        rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
        energies.append(rms)
        timestamps.append(i / sr)
    
    if not energies:
        return []
    
    # Find energy threshold (dynamic)
    energy_mean = np.mean(energies)
    energy_std = np.std(energies)
    threshold = energy_mean + 0.5 * energy_std
    
    # Find low-energy segments (potential fillers)
    fillers = []
    in_low_energy = False
    start_time = 0
    min_gap_duration = 0.08  # 80ms minimum
    max_gap_duration = 0.4   # 400ms maximum
    
    for i, (rms, ts) in enumerate(zip(energies, timestamps)):
        is_low = rms < threshold
        
        if is_low and not in_low_energy:
            in_low_energy = True
            start_time = ts
        elif not is_low and in_low_energy:
            in_low_energy = False
            end_time = ts
            duration = end_time - start_time
            
            # Typical filler duration: 80ms - 400ms
            if min_gap_duration < duration < max_gap_duration:
                # Confidence based on duration and energy level
                confidence = 0.5 + (duration - min_gap_duration) / (max_gap_duration - min_gap_duration) * 0.4
                confidence = min(0.9, confidence)
                
                fillers.append({
                    'start': round(start_time, 3),
                    'end': round(end_time, 3),
                    'duration': round(duration, 3),
                    'type': 'filled_pause',
                    'confidence': round(confidence, 2),
                    'source': 'audio'
                })
    
    # Merge overlapping fillers
    if len(fillers) > 1:
        merged = []
        current = fillers[0]
        for f in fillers[1:]:
            if f['start'] - current['end'] < 0.05:  # Overlap or very close
                current['end'] = max(current['end'], f['end'])
                current['duration'] = round(current['end'] - current['start'], 3)
                current['confidence'] = max(current['confidence'], f['confidence'])
            else:
                merged.append(current)
                current = f
        merged.append(current)
        fillers = merged
    
    return fillers


def detect_fillers_from_audio_vad(wav_path: Path) -> List[Dict[str, Any]]:
    """
    Detect fillers using Silero VAD (if available).
    
    This is a more accurate approach that requires silero-vad package.
    
    Install: pip install silero-vad
    """
    try:
        import torch
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    except ImportError:
        # Fallback to energy-based detection
        return detect_fillers_from_audio(wav_path)
    
    try:
        # Load VAD model
        model = load_silero_vad()
        
        # Read audio
        audio = read_audio(str(wav_path))
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
        
        # Find gaps between speech segments (potential fillers)
        fillers = []
        if len(speech_timestamps) < 2:
            return []
        
        for i in range(len(speech_timestamps) - 1):
            end_current = speech_timestamps[i]['end'] / 16000
            start_next = speech_timestamps[i+1]['start'] / 16000
            gap = start_next - end_current
            
            # Typical filler duration: 80ms - 500ms
            if 0.08 < gap < 0.5:
                fillers.append({
                    'start': round(end_current, 3),
                    'end': round(start_next, 3),
                    'duration': round(gap, 3),
                    'type': 'filled_pause',
                    'confidence': round(0.7 + (gap - 0.08) / 0.42 * 0.25, 2),
                    'source': 'vad'
                })
        
        return fillers
    except Exception as e:
        # Fallback to energy-based detection
        return detect_fillers_from_audio(wav_path)