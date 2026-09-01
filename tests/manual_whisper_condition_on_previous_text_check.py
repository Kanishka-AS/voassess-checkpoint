"""
MANUAL, not part of the pytest suite — run this yourself where the real
Whisper "base" model is available (it isn't in the sandbox this change was
developed in: no network access to the model-weights host there).

Compares Whisper's output on the exact same WAV
(data/recordings/debug_20260830_143103.wav — the file behind the "How are
you?" / 1.04s-only transcription from the 2026-08-30 debug session) with
condition_on_previous_text=True (Whisper's default, i.e. the OLD behavior)
vs. condition_on_previous_text=False (the change just made to
transcribe_wav() in app.py).

Usage:
    python3 tests/manual_whisper_condition_on_previous_text_check.py

Reads whatever WAV you pass as argv[1], defaulting to the known problem
recording if you don't pass one — handy for testing a *new* recording too.
"""
import sys
from pathlib import Path

import whisper

WAV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).parent.parent / "data" / "recordings" / "debug_20260830_143103.wav"
)


def run(condition_on_previous_text: bool):
    model = whisper.load_model("base")
    result = model.transcribe(
        str(WAV_PATH), language="en", word_timestamps=True,
        condition_on_previous_text=condition_on_previous_text,
    )
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append((w.get("word", ""), w.get("start"), w.get("end"), w.get("probability")))
    return result["text"].strip(), words


def main():
    if not WAV_PATH.exists():
        print(f"WAV not found: {WAV_PATH}")
        sys.exit(1)

    print(f"File: {WAV_PATH}\n")

    for label, flag in [("OLD (condition_on_previous_text=True, the default)", True),
                         ("NEW (condition_on_previous_text=False)", False)]:
        text, words = run(flag)
        last_end = words[-1][2] if words else None
        print(f"--- {label} ---")
        print(f"Transcript: {text!r}")
        print(f"Word count: {len(words)}   Last word ends at: {last_end}")
        for w, start, end, prob in words:
            print(f"    {w!r:>12}  {start:>6.3f}-{end:<6.3f}  p={prob:.2f}" if prob is not None
                  else f"    {w!r:>12}  {start:>6.3f}-{end:<6.3f}  p=?")
        print()


if __name__ == "__main__":
    main()
