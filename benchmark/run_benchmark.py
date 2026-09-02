"""
STT provider benchmark for assessment purposes — Whisper vs Sarvam Saaras
vs NVIDIA Parakeet vs Moonshine.

Run from the `assessment/` repo root (needs stt_provider.py, audio_utils.py
importable — this script adds the parent dir to sys.path):

    cd benchmark
    python3 run_benchmark.py                          # all 4 providers
    python3 run_benchmark.py --providers saaras,parakeet,moonshine
    python3 run_benchmark.py --providers parakeet      # just one

Does NOT touch app.py, scoring formulas, or any production file. Writes
benchmark_results.json + prints a markdown comparison table.

Static (not measured — see STATIC_CAPABILITIES) columns are architectural
facts about each provider that don't depend on any one audio sample:
CPU-only usability, offline capability, RAM ballpark. Measured columns
(latency, retention, timestamps/confidence *actually returned*) only
appear for providers that actually ran in this environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # same .env app.py uses, for SARVAM_API_KEY
except ImportError:
    pass

from samples_manifest import SAMPLES
from candidate_providers import ParakeetSTTProvider, MoonshineSTTProvider, timed_transcribe
from stt_provider import WhisperSTTProvider, SaarasSTTProvider
from analyze import analyze_sample

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Architectural facts, not measured here — cited from each provider's own
# docs/module docstrings already in this repo (Saaras) or public vendor
# docs (Whisper/Parakeet/Moonshine). Kept separate from measured data so
# the two are never confused in the output.
STATIC_CAPABILITIES = {
    "whisper": {"cpu_only": True, "offline": True, "ram_ballpark": "~1-2 GB (base/small, CPU)"},
    "saaras": {"cpu_only": True, "offline": False, "ram_ballpark": "N/A (hosted API)"},
    "parakeet": {"cpu_only": True, "offline": True,
                 "ram_ballpark": "~2-3 GB (0.6B params, local ONNX via onnx-asr, CPU)"},
    "moonshine": {"cpu_only": True, "offline": True, "ram_ballpark": "~200-400 MB (tiny/base, CPU/edge-oriented)"},
}


def build_whisper_provider():
    try:
        import whisper
        model = whisper.load_model("tiny.en")
        return WhisperSTTProvider(model)
    except Exception as e:
        return _unavailable("whisper", e)


def _unavailable(name, err):
    class _Unavailable:
        def is_available(self_): return False
        def transcribe(self_, wav_path):
            from stt_provider import STTResult
            return STTResult(transcript="", segments=[], provider=name, available=False,
                              detail=f"Could not initialize {name}: {err}")
    _Unavailable.name = name
    return _Unavailable()


ALL_PROVIDER_NAMES = ("whisper", "saaras", "parakeet", "moonshine")

# Lazy builders — only construct the providers actually requested, so e.g.
# `--providers saaras,parakeet,moonshine` never touches whisper.load_model()
# (slow, and irrelevant if you've already ruled Whisper out).
PROVIDER_BUILDERS = {
    "whisper": build_whisper_provider,
    "saaras": SaarasSTTProvider,       # reads SARVAM_API_KEY from env, same as production
    "parakeet": ParakeetSTTProvider,   # local ONNX model (onnx-asr), no API key
    "moonshine": MoonshineSTTProvider,
}


def build_providers(names):
    return {name: PROVIDER_BUILDERS[name]() for name in names}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--providers", type=str, default=",".join(ALL_PROVIDER_NAMES),
                    help=f"Comma-separated subset of {ALL_PROVIDER_NAMES}. Default: all.")
    return p.parse_args()


def main():
    args = parse_args()
    requested = [n.strip() for n in args.providers.split(",") if n.strip()]
    unknown = [n for n in requested if n not in ALL_PROVIDER_NAMES]
    if unknown:
        raise SystemExit(f"Unknown provider(s) {unknown}. Valid: {ALL_PROVIDER_NAMES}")

    providers = build_providers(requested)
    results = []

    for sample_name, gt in SAMPLES.items():
        wav_path = SAMPLES_DIR / f"{sample_name}.wav"
        if not wav_path.exists():
            print(f"SKIP {sample_name}: {wav_path} not found")
            continue
        for pname, provider in providers.items():
            timed = timed_transcribe(provider, wav_path)
            analysis = analyze_sample(sample_name, gt, pname, timed)
            results.append(analysis)
            status = "OK" if analysis.available else f"UNAVAILABLE ({analysis.detail})"
            print(f"[{sample_name}] {pname}: {status}")

    out_path = Path(__file__).resolve().parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print(f"\nWrote {out_path}")

    print_summary_table(results, providers)


def print_summary_table(results, providers):
    print("\n## Comparison\n")
    header = ("Provider | Ran? | Avg latency (s) | Filler retention | Discourse-filler "
              "retention | Repetition retention | False-start retention | Word timestamps | "
              "Word confidence | CPU-only | Offline | RAM ballpark")
    print(header)
    print("|".join(["---"] * header.count("|") + ["---"]))

    for pname in providers:
        rows = [r for r in results if r.provider == pname]
        ran = any(r.available for r in rows)
        lat_vals = [r.latency_seconds for r in rows if r.available]
        avg_lat = round(sum(lat_vals) / len(lat_vals), 3) if lat_vals else "n/a"

        def agg(field):
            ratios = [getattr(r, field)["ratio"] for r in rows
                      if r.available and getattr(r, field)["applicable"]]
            return f"{round(sum(ratios) / len(ratios), 2)}" if ratios else "n/a"

        static = STATIC_CAPABILITIES.get(pname, {})
        print(f"{pname} | {'yes' if ran else 'no'} | {avg_lat} | "
              f"{agg('filler_retention')} | {agg('discourse_filler_retention')} | "
              f"{agg('repetition_retention')} | {agg('false_start_retention')} | "
              f"{any(r.has_word_timestamps for r in rows)} | "
              f"{any(r.has_word_confidence for r in rows)} | "
              f"{static.get('cpu_only', '?')} | {static.get('offline', '?')} | "
              f"{static.get('ram_ballpark', '?')}")


if __name__ == "__main__":
    main()