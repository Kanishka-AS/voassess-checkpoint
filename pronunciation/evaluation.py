"""
evaluation.py
-------------
Run available PronunciationProvider(s) over test audio and produce JSON
results, so different acoustic-model backends can eventually be compared
on identical inputs.

Currently only AllosaurusProvider is real. The provider registry below is
written so a future Wav2Vec2Provider / L2Wav2VecProvider only needs to be
added to PROVIDERS once implemented -- nothing else in this file changes.

Usage:
    python3 -m pronunciation.evaluation --real         # MediaRepeat clips
    python3 -m pronunciation.evaluation --synthetic     # espeak-ng minimal pairs
    python3 -m pronunciation.evaluation --real --synthetic --out results.json
"""

import argparse
import json
import os
import time

from .allosaurus_provider import AllosaurusProvider

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_REPEAT_DIR = os.path.join(BASE_DIR, "MediaRepeat")

# Real, human-recorded evaluation set. Sentences copied verbatim from
# app.py's ASSESSMENT_MANIFEST["media_repeat"] -- do not edit independently
# of that file or they will drift out of sync.
REAL_TEST_SET = [
    {"audio": "clip-1786358471410-1.wav", "text": "I like apples."},
    {"audio": "clip-1786358735764-2.wav", "text": "I sleep for seven hours every day."},
    {"audio": "clip-1786358849827-3.wav", "text": "Climate change is one of the biggest problems in the world today."},
]

# Providers currently available. Add Wav2Vec2Provider() / L2Wav2VecProvider()
# here once they are actually implemented and testable -- not before.
def _build_providers():
    providers = {}
    try:
        providers["allosaurus"] = AllosaurusProvider()
    except Exception as e:
        print(f"[warn] AllosaurusProvider failed to initialize: {e}")
    return providers


def run_real(providers, out_records):
    for item in REAL_TEST_SET:
        audio_path = os.path.join(MEDIA_REPEAT_DIR, item["audio"])
        if not os.path.exists(audio_path):
            print(f"[warn] missing audio file: {audio_path}")
            continue
        for pname, provider in providers.items():
            t0 = time.time()
            result = provider.analyze(audio_path, item["text"])
            elapsed = time.time() - t0
            record = result.to_dict()
            record["_meta"] = {
                "source": "real",
                "inference_seconds": round(elapsed, 3),
            }
            out_records.append(record)
            print(f"[real] {pname} :: '{item['text']}' ({elapsed:.2f}s)")


def run_synthetic(providers, out_records):
    """Generate espeak-ng minimal-pair audio on the fly and run providers
    on it. Every record from this path is tagged source='synthetic' so it
    is never confused with real-speech evidence downstream."""
    import subprocess
    import tempfile

    pairs = [
        ("think", "θ"), ("tink", "θ"),
        ("this", "ð"), ("dis", "ð"),
        ("right", "ɹ"), ("light", "ɹ"),
        ("vine", "v"), ("wine", "v"),
        ("she", "ʃ"), ("see", "ʃ"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for word, _ in pairs:
            wav_path = os.path.join(tmp, f"{word}.wav")
            subprocess.run(
                ["espeak-ng", "-v", "en-us", "-w", wav_path, word],
                check=True, capture_output=True,
            )
            for pname, provider in providers.items():
                t0 = time.time()
                result = provider.analyze(wav_path, word)
                elapsed = time.time() - t0
                record = result.to_dict()
                record["_meta"] = {
                    "source": "synthetic (espeak-ng)",
                    "inference_seconds": round(elapsed, 3),
                }
                out_records.append(record)
                print(f"[synthetic] {pname} :: '{word}' ({elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run on real MediaRepeat recordings")
    parser.add_argument("--synthetic", action="store_true", help="Run on espeak-ng minimal pairs")
    parser.add_argument("--out", type=str, default=None, help="Write JSON results to this path")
    args = parser.parse_args()

    if not args.real and not args.synthetic:
        args.real = True  # default

    providers = _build_providers()
    if not providers:
        print("[error] no providers available, nothing to run.")
        return

    for pname, provider in providers.items():
        print(f"[info] {pname} resource_info: {provider.resource_info()}")

    records = []
    if args.real:
        run_real(providers, records)
    if args.synthetic:
        run_synthetic(providers, records)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[info] wrote {len(records)} records to {args.out}")
    else:
        print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
