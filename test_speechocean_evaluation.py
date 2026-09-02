#!/usr/bin/env python3
"""
SpeechOcean762 benchmark evaluation.
-------------------------------------

This is a BENCHMARK/EVALUATION script, not a pytest test suite and not part
of the production assessment. It answers one question: "does the existing
pronunciation-assessment pipeline (pronunciation_provider.py, as wired into
app.score_free_speech()) produce scores that agree with SpeechOcean762's
human-annotated pronunciation-accuracy labels?"

It does NOT change any scoring formula, does NOT touch the production
/assess routes, and does NOT feed benchmark data into a user's real
assessment. It just runs the existing pipeline, exactly as the app runs it
(save_and_convert is skipped because these files are already WAV; the same
transcribe -> score_free_speech call the app makes is used unchanged), on
real local SpeechOcean762 audio and compares the result to the dataset's
own scores.json labels.

────────────────────────────────────────────────────────────────────────────
Dataset layout expected (the official jimbozhang/speechocean762 layout):

    <data_dir>/
        scores.json          # utt-id -> {accuracy, completeness, fluency,
                              #             prosodic, total, text, words:[...]}
        train/wav.scp        # utt-id -> relative WAV path (Kaldi style)
        test/wav.scp
        WAVE/SPEAKER00xx/*.WAV

If wav.scp can't be found/parsed, this script falls back to searching
WAVE/**/<utt-id>.WAV directly. Nothing is copied, uploaded, or committed —
this script only ever reads from <data_dir>.

────────────────────────────────────────────────────────────────────────────
Label conversion (documented, not hidden):

SpeechOcean762's "accuracy" is a sentence-level, human-rated pronunciation-
accuracy score on a 0-10 scale. This is the closest available ground truth
for what our pipeline's pronunciation.score is trying to measure (how
correctly the words were pronounced) -- it is NOT "total" (which also
blends in prosody/fluency) and NOT "fluency" (a different aspect
entirely). We linearly rescale it to 0-100 to match our score's scale:

    reference_score_0_100 = accuracy_0_10 * 10.0

"fluency" and "total" are recorded per-sample for context but are not used
as the primary comparison target.

────────────────────────────────────────────────────────────────────────────
Usage:
    python tests/test_speechocean_evaluation.py --limit 100
    python tests/test_speechocean_evaluation.py --limit 100 --split test
    python tests/test_speechocean_evaluation.py --full            # entire split
    python tests/test_speechocean_evaluation.py --data-dir /path/to/speechocean762

Results are written to:
    evaluation_results/speechocean762_latest.json
    evaluation_results/speechocean762_latest.csv

Nothing here is required for the application to run, and it is never
imported by app.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Backend imports — the SAME pipeline app.py's /assess route uses ─────────
# score_free_speech() is the real, shared five-metric pipeline (see its
# docstring in app.py). We call it exactly as the app does; we do not
# reimplement pace/filler/grammar/pronunciation scoring here.
import app  # noqa: E402  (loads the Whisper model + provider registries once)
from app import score_free_speech, DEFAULT_PRONUNCIATION_PROVIDER, transcribe_wav  # noqa: E402
from audio_utils import wav_duration_seconds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results"

DATASET_NAME = "speechocean762"


# ============================================================================
# 1. LOCATE THE DATASET
# ============================================================================

def _looks_like_speechocean_root(path: Path) -> bool:
    return (path / "scores.json").is_file() and (path / "WAVE").is_dir()


def find_speechocean_dataset(explicit: Optional[str]) -> Optional[Path]:
    """Search common local locations for a real speechocean762 checkout.
    Returns the directory containing scores.json + WAVE/, or None."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_dir = os.environ.get("SPEECHOCEAN_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    for base in (BASE_DIR, BASE_DIR.parent, Path.home(), Path.cwd()):
        for name in ("speechocean_data", "speechocean762", "SpeechOcean762",
                     "speechocean-762", "speech_ocean762"):
            candidates.append(base / name)

    # Some downloads nest an extra folder, e.g. speechocean_data/speechocean762/
    expanded: List[Path] = []
    for c in candidates:
        expanded.append(c)
        if c.is_dir():
            for child in c.iterdir():
                if child.is_dir():
                    expanded.append(child)

    for c in expanded:
        if c.is_dir() and _looks_like_speechocean_root(c):
            return c

    # Last resort: bounded shallow search for a directory literally named
    # "WAVE" sitting next to scores.json, under a few common roots.
    for base in (BASE_DIR, Path.home(), Path.cwd()):
        if not base.is_dir():
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                if depth > 3:
                    dirnames[:] = []
                    continue
                if "scores.json" in filenames and "WAVE" in dirnames:
                    return Path(dirpath)
        except (PermissionError, OSError):
            continue
    return None


# ============================================================================
# 2. LOAD LABELS + AUDIO INDEX
# ============================================================================

def load_scores(data_dir: Path) -> Dict[str, dict]:
    with open(data_dir / "scores.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_wav_index(data_dir: Path, split: str) -> Dict[str, Path]:
    """Parse Kaldi-style wav.scp files (`<utt-id> <path>` per line) for the
    requested split ("train", "test", or "both"). Falls back to nothing if
    wav.scp is missing/unparseable — callers then fall back to a WAVE/
    glob per utt-id."""
    splits = ["train", "test"] if split == "both" else [split]
    index: Dict[str, Path] = {}
    for s in splits:
        scp = data_dir / s / "wav.scp"
        if not scp.is_file():
            continue
        try:
            with open(scp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) != 2:
                        continue
                    utt_id, raw_path = parts
                    p = Path(raw_path)
                    if not p.is_absolute():
                        p = (data_dir / raw_path).resolve()
                    index[utt_id] = p
        except OSError:
            continue
    return index


def resolve_audio_path(utt_id: str, wav_index: Dict[str, Path], data_dir: Path) -> Optional[Path]:
    p = wav_index.get(utt_id)
    if p and p.is_file():
        return p
    # Fallback: WAVE/SPEAKER****/<utt_id>.WAV (case-insensitive extension)
    for match in data_dir.glob(f"WAVE/*/{utt_id}.*"):
        if match.is_file():
            return match
    return None


# ============================================================================
# 3. SAMPLE SELECTION (deterministic, representative)
# ============================================================================

def select_samples(utt_ids: List[str], limit: Optional[int]) -> List[str]:
    """Evenly-spaced selection over the *sorted* utt-id list. Deterministic
    (no RNG, so reruns with the same --limit are reproducible) and
    representative in the sense that it spans the whole sorted range
    (different speakers/sentences) rather than just taking the first N,
    which would bias toward one or two speakers given how utt-ids are
    grouped by speaker in this dataset."""
    ids = sorted(utt_ids)
    if not limit or limit >= len(ids):
        return ids
    if limit <= 0:
        return []
    step = len(ids) / float(limit)
    return [ids[min(int(i * step), len(ids) - 1)] for i in range(limit)]


# ============================================================================
# 4. RUN ONE SAMPLE THROUGH THE EXISTING PIPELINE
# ============================================================================

def process_sample(utt_id: str, audio_path: Path, label: dict,
                    pronunciation_provider: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "sample_id": utt_id,
        "audio_file": str(audio_path),
        "reference_transcript": label.get("text", ""),
        "reference_accuracy_raw": label.get("accuracy"),
        "reference_fluency_raw": label.get("fluency"),
        "reference_total_raw": label.get("total"),
        "reference_score": None,
        "whisper_transcript": None,
        "predicted_score": None,
        "predicted_provider": None,
        "pronunciation_available": False,
        "pronunciation_issues": [],
        "absolute_error": None,
        "status": "failed",
        "error": None,
    }
    try:
        accuracy_raw = label.get("accuracy")
        if accuracy_raw is None:
            record["error"] = "No 'accuracy' label in scores.json for this utt-id."
            return record
        reference_score = float(accuracy_raw) * 10.0
        record["reference_score"] = round(reference_score, 2)

        duration = wav_duration_seconds(audio_path)
        transcript, segments = transcribe_wav(audio_path)
        record["whisper_transcript"] = transcript
        if duration <= 0:
            record["error"] = "Could not read WAV duration."
            return record

        result = score_free_speech(
            transcript, segments, duration, audio_path,
            pronunciation_provider=pronunciation_provider,
        )
        pron = result.get("pronunciation", {})
        record["predicted_score"] = pron.get("score")
        record["predicted_provider"] = pron.get("provider")
        record["pronunciation_available"] = bool(pron.get("available"))
        record["pronunciation_issues"] = pron.get("issues", [])

        if record["predicted_score"] is not None:
            record["absolute_error"] = round(
                abs(record["predicted_score"] - reference_score), 2
            )
        record["status"] = "ok"
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc(limit=6)
    return record


# ============================================================================
# 5. METRICS
# ============================================================================

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom == 0:
        return None  # one series is constant — correlation is undefined, not zero
    return num / denom


def histogram(xs: List[float], edges: List[float]) -> Dict[str, int]:
    labels = [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]
    counts = {label: 0 for label in labels}
    for x in xs:
        for i in range(len(edges) - 1):
            hi_inclusive = i == len(edges) - 2
            if edges[i] <= x < edges[i + 1] or (hi_inclusive and x == edges[i + 1]):
                counts[labels[i]] += 1
                break
    return counts


SCORE_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ERROR_EDGES = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [s for s in samples if s["status"] == "ok" and s["predicted_score"] is not None]
    refs = [s["reference_score"] for s in ok]
    preds = [s["predicted_score"] for s in ok]
    errors = [s["absolute_error"] for s in ok]

    if not ok:
        return {
            "n_scored": 0,
            "mae": None, "rmse": None, "pearson_correlation": None,
            "mean_signed_error": None,
            "reference_score_stats": None, "predicted_score_stats": None,
            "distributions": None,
        }

    mae = _mean(errors)
    rmse = math.sqrt(_mean([e ** 2 for e in errors]))
    signed = [p - r for p, r in zip(preds, refs)]
    mean_signed_error = _mean(signed)
    r = pearson_correlation(refs, preds)

    return {
        "n_scored": len(ok),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "pearson_correlation": round(r, 3) if r is not None else None,
        "mean_signed_error": round(mean_signed_error, 2),
        "reference_score_stats": {
            "mean": round(_mean(refs), 2), "std": round(_std(refs), 2),
            "min": round(min(refs), 2), "max": round(max(refs), 2),
        },
        "predicted_score_stats": {
            "mean": round(_mean(preds), 2), "std": round(_std(preds), 2),
            "min": round(min(preds), 2), "max": round(max(preds), 2),
        },
        "distributions": {
            "reference_score_histogram": histogram(refs, SCORE_EDGES),
            "predicted_score_histogram": histogram(preds, SCORE_EDGES),
            "absolute_error_histogram": histogram(errors, ERROR_EDGES),
        },
    }


def analyze_failure_patterns(samples: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
    """Objective, threshold-based pattern detection over the SCORED samples
    only. Thresholds are fixed here (not tuned against this run's output)."""
    patterns: List[str] = []
    ok = [s for s in samples if s["status"] == "ok" and s["predicted_score"] is not None]
    if len(ok) < 5:
        patterns.append(
            f"Only {len(ok)} sample(s) produced a usable score — too few to "
            "characterize failure patterns reliably."
        )
        return patterns

    mse = metrics.get("mean_signed_error")
    if mse is not None:
        if mse >= 8:
            patterns.append(
                f"Systematic over-scoring: predicted scores average {mse:+.1f} "
                "points above the human reference (positive bias)."
            )
        elif mse <= -8:
            patterns.append(
                f"Systematic under-scoring: predicted scores average {mse:+.1f} "
                "points below the human reference (negative bias)."
            )

    ref_std = (metrics.get("reference_score_stats") or {}).get("std")
    pred_std = (metrics.get("predicted_score_stats") or {}).get("std")
    if ref_std and pred_std is not None and ref_std > 0 and pred_std < 0.5 * ref_std:
        patterns.append(
            f"Compressed prediction range: predicted-score std ({pred_std:.1f}) is "
            f"less than half the reference std ({ref_std:.1f}) — the pipeline is not "
            "discriminating between clearly good and clearly poor pronunciation as "
            "much as the human raters did."
        )

    low = [s for s in ok if s["reference_score"] < 40]
    high = [s for s in ok if s["reference_score"] > 80]
    if low and high:
        low_pred_mean = _mean([s["predicted_score"] for s in low])
        high_pred_mean = _mean([s["predicted_score"] for s in high])
        low_ref_mean = _mean([s["reference_score"] for s in low])
        high_ref_mean = _mean([s["reference_score"] for s in high])
        pred_gap = high_pred_mean - low_pred_mean
        ref_gap = high_ref_mean - low_ref_mean
        if ref_gap > 20 and pred_gap < 0.5 * ref_gap:
            patterns.append(
                f"Poor separation between low-accuracy (ref<40, n={len(low)}) and "
                f"high-accuracy (ref>80, n={len(high)}) samples: human raters separated "
                f"them by {ref_gap:.1f} points on average, the pipeline only by "
                f"{pred_gap:.1f} points."
            )

    failed = [s for s in samples if s["status"] != "ok"]
    if samples and len(failed) / len(samples) > 0.1:
        patterns.append(
            f"{len(failed)}/{len(samples)} samples ({len(failed)/len(samples):.0%}) "
            "failed to process — check the 'error' field on individual samples."
        )

    if not patterns:
        patterns.append("No obvious systematic failure pattern detected against the "
                         "fixed thresholds used above.")
    return patterns


def determine_verdict(metrics: Dict[str, Any], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = metrics.get("n_scored") or 0
    r = metrics.get("pearson_correlation")
    mae = metrics.get("mae")
    fail_rate = 1 - (n / len(samples)) if samples else 1.0

    if n < 10 or r is None or mae is None:
        label = "Not reliable enough yet"
        explanation = (
            f"Only {n} of {len(samples)} sample(s) produced a comparable score "
            "(need at least 10, and a defined correlation, to draw any conclusion)."
        )
    elif fail_rate > 0.2:
        label = "Not reliable enough yet"
        explanation = (
            f"{fail_rate:.0%} of samples failed to process — the pipeline itself "
            "isn't reliable enough on this data to trust the metrics that did compute."
        )
    elif r >= 0.7 and mae <= 10:
        label = "Strong agreement"
        explanation = f"Correlation r={r:.2f} and MAE={mae:.1f} both indicate the pipeline's pronunciation score tracks the human accuracy rating closely."
    elif r >= 0.5 and mae <= 20:
        label = "Moderate agreement"
        explanation = f"Correlation r={r:.2f} and MAE={mae:.1f} indicate a real but imperfect relationship with the human accuracy rating."
    elif r >= 0.3:
        label = "Weak agreement"
        explanation = f"Correlation r={r:.2f} is positive but weak (MAE={mae:.1f}) — the score has some signal but disagrees with human raters often enough that it shouldn't be trusted for fine-grained decisions."
    else:
        label = "Not reliable enough yet"
        explanation = f"Correlation r={r:.2f} and MAE={mae:.1f} do not show a meaningful relationship between the pipeline's pronunciation score and the human accuracy rating on this sample."

    return {
        "label": label,
        "explanation": explanation,
        "failure_patterns": analyze_failure_patterns(samples, metrics),
    }


# ============================================================================
# 6. MAIN
# ============================================================================

def run(args) -> int:
    data_dir = find_speechocean_dataset(args.data_dir)
    if data_dir is None:
        print("BLOCKER: could not locate a local speechocean762 dataset "
              "(looked for a directory containing scores.json + WAVE/, "
              "including ./speechocean_data, ~/speechocean_data, "
              "~/speechocean762, and $SPEECHOCEAN_DATA_DIR). "
              "Pass --data-dir /path/to/speechocean762 explicitly, or set "
              "SPEECHOCEAN_DATA_DIR. Refusing to fabricate results.")
        return 2
    print(f"Found dataset at: {data_dir}")

    try:
        scores = load_scores(data_dir)
    except Exception as e:
        print(f"BLOCKER: could not read/parse {data_dir / 'scores.json'}: {e}")
        return 2
    print(f"Loaded {len(scores)} ground-truth label entries from scores.json")

    wav_index = load_wav_index(data_dir, args.split)
    print(f"Indexed {len(wav_index)} audio paths from wav.scp for split={args.split!r}")

    # Build the candidate set: utt-ids that have BOTH a label and a real,
    # resolvable audio file. This is computed before sampling so --limit
    # samples from what's actually usable, not from ids that will just fail.
    candidate_ids = []
    resolved_paths: Dict[str, Path] = {}
    for utt_id in scores.keys():
        # If we have a wav.scp for this split, restrict to ids listed in it
        # (keeps train/test separation meaningful); otherwise fall back to
        # resolving directly from WAVE/ for every labeled id.
        if wav_index and utt_id not in wav_index:
            continue
        p = resolve_audio_path(utt_id, wav_index, data_dir)
        if p is not None:
            candidate_ids.append(utt_id)
            resolved_paths[utt_id] = p

    print(f"{len(candidate_ids)} utt-ids have both a label and a resolvable audio file")
    if not candidate_ids:
        print("BLOCKER: no utt-id has both a scores.json label and a locatable "
              "WAV file. Check --split and the dataset layout. Refusing to "
              "fabricate results.")
        return 2

    limit = None if args.full else args.limit
    selected_ids = select_samples(candidate_ids, limit)
    print(f"Selected {len(selected_ids)} sample(s) to evaluate "
          f"({'full split' if limit is None else f'limit={limit}'})")

    samples: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, utt_id in enumerate(selected_ids, 1):
        audio_path = resolved_paths[utt_id]
        label = scores[utt_id]
        rec = process_sample(utt_id, audio_path, label, args.pronunciation_provider)
        samples.append(rec)
        status = "OK " if rec["status"] == "ok" else "FAIL"
        pred = rec["predicted_score"]
        ref = rec["reference_score"]
        print(f"[{i}/{len(selected_ids)}] {status} {utt_id}  ref={ref}  pred={pred}"
              + (f"  err={rec['error']}" if rec["error"] else ""))
    elapsed = time.time() - t0

    metrics = compute_metrics(samples)
    verdict = determine_verdict(metrics, samples)

    successful = sum(1 for s in samples if s["status"] == "ok")
    failed = len(samples) - successful

    output = {
        "test_metadata": {
            "dataset": DATASET_NAME,
            "dataset_path": str(data_dir),
            "split": args.split,
            "pronunciation_provider_requested": args.pronunciation_provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "note": (
                "Benchmark/evaluation layer only. Does not modify production "
                "scoring, is not required for the app to run, and is never "
                "imported by app.py."
            ),
        },
        "counts": {
            "labels_in_scores_json": len(scores),
            "utt_ids_with_audio_and_label": len(candidate_ids),
            "samples_selected": len(selected_ids),
            "samples_successful": successful,
            "samples_failed": failed,
        },
        "metrics": metrics,
        "verdict": verdict,
        "samples": samples,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    json_path = RESULTS_DIR / "speechocean762_latest.json"
    csv_path = RESULTS_DIR / "speechocean762_latest.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "reference_score", "predicted_score", "absolute_error",
                    "status", "error", "reference_transcript", "whisper_transcript"])
        for s in samples:
            w.writerow([s["sample_id"], s["reference_score"], s["predicted_score"],
                        s["absolute_error"], s["status"], s["error"] or "",
                        s["reference_transcript"], s["whisper_transcript"] or ""])

    print("\n" + "=" * 70)
    print(f"Dataset:    {data_dir}")
    print(f"Split:      {args.split}")
    print(f"Selected:   {len(selected_ids)}   Successful: {successful}   Failed: {failed}")
    if metrics["n_scored"]:
        print(f"MAE:        {metrics['mae']}")
        print(f"RMSE:       {metrics['rmse']}")
        print(f"Pearson r:  {metrics['pearson_correlation']}")
    else:
        print("No samples produced a comparable score — see verdict below.")
    print(f"Verdict:    {verdict['label']}")
    print(f"  {verdict['explanation']}")
    for p in verdict["failure_patterns"]:
        print(f"  - {p}")
    print(f"\nResults written to:\n  {json_path}\n  {csv_path}")
    print("=" * 70)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SpeechOcean762 local evaluation of the "
                                             "existing pronunciation-assessment pipeline.")
    p.add_argument("--data-dir", default=None,
                    help="Path to a local speechocean762 checkout (contains scores.json, WAVE/). "
                         "If omitted, common local locations are searched.")
    p.add_argument("--split", default="test", choices=["train", "test", "both"],
                    help="Which wav.scp split to draw utt-ids from (default: test, "
                         "the standard held-out evaluation split).")
    p.add_argument("--limit", type=int, default=100,
                    help="Number of samples to evaluate (default: 100). Ignored if --full is set.")
    p.add_argument("--full", action="store_true",
                    help="Evaluate every candidate utt-id in the split instead of --limit.")
    p.add_argument("--pronunciation-provider", default=DEFAULT_PRONUNCIATION_PROVIDER,
                    help=f"Which pronunciation provider to evaluate (default: {DEFAULT_PRONUNCIATION_PROVIDER}).")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    ns = parser.parse_args()
    sys.exit(run(ns))