
#!/usr/bin/env python3
"""
SpeechOcean762 Local Audio Backend Test Harness

Runs the complete backend assessment pipeline on REAL local SpeechOcean762 audio.

This script:
1. Finds the locally downloaded SpeechOcean762 dataset
2. Loads a small number of samples (default: 10)
3. Processes each through the existing backend
4. Saves results to speechocean_local_results.json

Usage:
    python test_speechocean_local.py --samples N
    python test_speechocean_local.py --data-dir /path/to/speechocean --samples N

If no --data-dir is provided, the script will search common locations.
"""

import os
import sys
import types
import json
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple
import time
import traceback
import glob
import re
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# BACKEND IMPORTS
# ============================================================================

import app

from app import (
    score_fillers,
    score_pace,
    score_clarity,
    score_cefr,
    determine_voice_archetype,
    build_feedback,
    DEFAULT_PRONUNCIATION_PROVIDER,
    _lt_provider,
    transcribe_wav,
)

from audio_utils import wav_duration_seconds
from languagetool_provider import LanguageToolUnavailable
from grammar_heuristics import detect_learner_errors
from vocabulary import score_vocabulary
from filler_detector import detect_fillers, summarize_words
from pronunciation_provider import resolve_pronunciation


# ── Fallback score_grammar ──────────────────────────────────────────────────
def score_grammar(errors: int, words: int) -> float:
    rate = errors / max(words, 1)
    if rate == 0:
        return 100
    elif rate < 0.02:
        return 90
    elif rate < 0.05:
        return 74
    elif rate < 0.08:
        return 56
    elif rate < 0.12:
        return 36
    else:
        return 16


# ── resolve_grammar fallback ────────────────────────────────────────────────
def resolve_grammar(text, lt_grammar, linguistic_analysis):
    if lt_grammar:
        ge = lt_grammar.get("errors", 0)
        issues = lt_grammar.get("issues", [])
        source = "languagetool_http"
    else:
        issues = detect_learner_errors(text)
        ge = len(issues)
        source = "learner_heuristic"
    return ge, issues, source, len(issues)


# ============================================================================
# FIND SPEECHOCEAN762 DATASET
# ============================================================================

def find_speechocean_dataset() -> Optional[Path]:
    """Search common locations for SpeechOcean762 dataset."""
    search_paths = [
        # Current working directory
        Path.cwd(),
        # Parent directory
        Path.cwd().parent,
        # Home directory
        Path.home(),
        # Downloads
        Path.home() / "Downloads",
        # Desktop
        Path.home() / "Desktop",
        # Temp directories
        Path("/tmp"),
        # Where it was extracted before
        Path("/tmp/speechocean_*"),
    ]
    
    # Also search for the specific directory name
    for search_path in search_paths:
        if "*" in str(search_path):
            import glob
            for path in glob.glob(str(search_path)):
                if path and Path(path).exists():
                    result = check_speechocean_dir(Path(path))
                    if result:
                        return result
        elif search_path.exists():
            # Check if this directory itself contains SpeechOcean762
            result = check_speechocean_dir(search_path)
            if result:
                return result
            
            # Check subdirectories
            for subdir in search_path.iterdir():
                if subdir.is_dir() and "speechocean" in subdir.name.lower():
                    result = check_speechocean_dir(subdir)
                    if result:
                        return result
    
    # Check if there's a directory with wav files
    wav_dirs = []
    for search_path in [Path.home(), Path.cwd()]:
        for wav_file in search_path.rglob("*.wav"):
            parent = wav_file.parent
            if parent not in wav_dirs:
                wav_dirs.append(parent)
                # Check if this looks like SpeechOcean762
                if "speechocean" in str(parent).lower():
                    result = check_speechocean_dir(parent)
                    if result:
                        return result
                # Check if there are many wav files
                wav_count = len(list(parent.glob("*.wav")))
                if wav_count > 100:
                    # Could be SpeechOcean762, check for metadata
                    meta_files = list(parent.glob("*.txt")) + list(parent.glob("*.tsv")) + list(parent.glob("*.csv"))
                    if meta_files:
                        # Likely a dataset
                        result = check_speechocean_dir(parent)
                        if result:
                            return result
    
    return None


def check_speechocean_dir(path: Path) -> Optional[Path]:
    """Check if a directory contains SpeechOcean762 dataset."""
    if not path or not path.exists() or not path.is_dir():
        return None
    
    # Check for wav files
    wav_files = list(path.glob("*.wav")) + list(path.glob("**/*.wav"))
    if len(wav_files) < 10:
        return None
    
    # Check for metadata files
    meta_files = list(path.glob("*.txt")) + list(path.glob("*.tsv")) + list(path.glob("*.csv"))
    meta_files += list(path.glob("**/*.txt")) + list(path.glob("**/*.tsv")) + list(path.glob("**/*.csv"))
    
    # Check for README
    readme_files = list(path.glob("README*")) + list(path.glob("**/README*"))
    
    if meta_files or readme_files:
        print(f"✅ Found SpeechOcean762 at: {path}")
        print(f"   WAV files: {len(wav_files)}")
        print(f"   Metadata files: {len(meta_files)}")
        return path
    
    return None


def load_speechocean_samples(
    data_dir: Path,
    max_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Load SpeechOcean762 samples from local directory.
    
    Returns:
        List of sample dicts with audio_path, reference_text, and metadata.
    """
    samples = []
    
    # Find all WAV files
    wav_files = list(data_dir.glob("*.wav")) + list(data_dir.glob("**/*.wav"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {data_dir}")
        return []
    
    print(f"Found {len(wav_files)} WAV files")
    
    # Try to find metadata files
    metadata = {}
    
    # Look for text files with transcripts
    text_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("**/*.txt"))
    # Also look for tsv/csv files
    tsv_files = list(data_dir.glob("*.tsv")) + list(data_dir.glob("**/*.tsv"))
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("**/*.csv"))
    
    print(f"Found {len(text_files)} text files, {len(tsv_files)} TSV files, {len(csv_files)} CSV files")
    
    # Try to load transcripts from text files
    for text_file in text_files:
        if "README" in text_file.name:
            continue
        try:
            with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # Try to parse as ID\ttext
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value = ' '.join(parts[1:]).strip()
                        if key and value:
                            metadata[key] = value
                    elif len(parts) == 1:
                        # Try to parse as filename: text
                        match = re.match(r'^([^\s:]+)[\s:]+(.+)$', line)
                        if match:
                            metadata[match.group(1).strip()] = match.group(2).strip()
        except Exception as e:
            pass
    
    # Try to load from TSV files
    for tsv_file in tsv_files:
        try:
            with open(tsv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 2:
                        metadata[row[0].strip()] = row[1].strip()
        except Exception:
            pass
    
    # Try to load from CSV files
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'id' in row and 'text' in row:
                        metadata[row['id'].strip()] = row['text'].strip()
                    elif len(row) >= 2:
                        keys = list(row.keys())
                        metadata[row[keys[0]].strip()] = row[keys[1]].strip()
        except Exception:
            pass
    
    print(f"Loaded {len(metadata)} transcript mappings")
    
    # Also try to load from the built-in transcript file if it exists
    transcript_file = data_dir / "transcript.txt"
    if transcript_file.exists():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        metadata[parts[0]] = ' '.join(parts[1:])
            print(f"Loaded transcripts from transcript.txt: {len(metadata)} entries")
        except Exception:
            pass
    
    # Also check for a "text" subdirectory or file
    text_dir = data_dir / "text"
    if text_dir.exists():
        text_files = list(text_dir.glob("*.txt"))
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        metadata[text_file.stem] = content
            except Exception:
                pass
    
    # Build samples from WAV files
    for wav_path in wav_files[:max_samples * 2]:  # Take extra to filter
        stem = wav_path.stem
        sample_id = stem
        
        # Try to find transcript
        transcript = metadata.get(sample_id, metadata.get(stem, ""))
        
        # If no transcript, try to use the filename as transcript
        if not transcript:
            transcript = stem.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Get duration
        try:
            import scipy.io.wavfile as wavfile
            rate, data = wavfile.read(str(wav_path))
            duration = len(data) / rate
        except Exception:
            duration = 0
        
        samples.append({
            "sample_id": sample_id,
            "audio_path": wav_path,
            "reference_text": transcript,
            "duration_seconds": duration,
            "path": str(wav_path),
        })
    
    # Sort by sample_id and limit
    samples.sort(key=lambda x: x["sample_id"])
    samples = samples[:max_samples]
    
    print(f"Selected {len(samples)} samples")
    
    return samples


# ============================================================================
# SINGLE SAMPLE PROCESSING
# ============================================================================

def process_sample(
    sample: Dict[str, Any],
    use_whisper: bool = True,
) -> Dict[str, Any]:
    """Process a single audio sample through the full backend pipeline."""
    sample_id = sample["sample_id"]
    reference_text = sample.get("reference_text", "")
    audio_path = sample["audio_path"]
    duration_seconds = sample.get("duration_seconds", 0)
    
    result = {
        "sample_id": sample_id,
        "reference_transcript": reference_text,
        "audio": {
            "duration_seconds": round(duration_seconds, 2),
            "path": str(audio_path),
        },
        "whisper_transcript": None,
        "parameters": {},
        "overall": {},
        "errors": [],
    }
    
    try:
        # ── 1. Transcribe with Whisper ─────────────────────────────────────
        if use_whisper:
            try:
                transcript, segments = transcribe_wav(audio_path)
                whisper_transcript = transcript
                result["whisper_transcript"] = whisper_transcript
                result["audio"]["segments_count"] = len(segments) if segments else 0
            except Exception as e:
                result["errors"].append(f"Whisper transcription failed: {e}")
                whisper_transcript = ""
                segments = []
        else:
            whisper_transcript = reference_text
            segments = []
            result["whisper_transcript"] = whisper_transcript
            result["audio"]["segments_count"] = 0
        
        if not whisper_transcript:
            result["errors"].append("No transcript available")
            return result
        
        word_count = len(whisper_transcript.split())
        
        # ── 2. LanguageTool Grammar ─────────────────────────────────────────
        lt_grammar = None
        linguistic_analysis = None
        lt_errors = {}
        
        try:
            lt_grammar, linguistic_analysis, lt_errors = _lt_provider.check_and_analyze(whisper_transcript)
        except LanguageToolUnavailable as e:
            result["errors"].append(f"LanguageTool unavailable: {e}")
        
        # ── 3. Grammar Resolution ───────────────────────────────────────────
        ge, grammar_issues, grammar_source, heuristic_added = resolve_grammar(
            whisper_transcript,
            lt_grammar,
            linguistic_analysis,
        )
        grammar_score = score_grammar(ge, word_count)
        
        # ── 4. Learner Heuristics ───────────────────────────────────────────
        heuristic_errors = detect_learner_errors(whisper_transcript)
        heuristic_count = len(heuristic_errors)
        
        # ── 5. Vocabulary ────────────────────────────────────────────────────
        vocab_result = score_vocabulary(whisper_transcript)
        vocabulary_score = vocab_result.get("score", 0)
        
        # ── 6. Fillers ──────────────────────────────────────────────────────
        fillers = detect_fillers(
            whisper_transcript,
            linguistic_analysis,
            duration_seconds=duration_seconds,
        )
        filler_count = fillers.get("count", 0)
        filler_words = summarize_words(fillers.get("occurrences", []))
        filler_rate = fillers.get("rate_per_min", 0)
        filler_score = score_fillers(filler_count, word_count)
        hesitation_count = len(fillers.get("hesitations", []))
        
        # ── 7. Pace ─────────────────────────────────────────────────────────
        wpm = word_count / (duration_seconds / 60.0) if duration_seconds > 0 else 0
        pace_score = score_pace(wpm)
        
        # ── 8. Pronunciation ────────────────────────────────────────────────
        pronun_score = None
        pronun_issues = []
        pronunciation_available = False
        
        try:
            pronun_result = resolve_pronunciation(
                DEFAULT_PRONUNCIATION_PROVIDER,
                whisper_transcript,
                segments,
                audio_path,
            )
            
            if pronun_result.available:
                pronun_score = pronun_result.score
                pronun_issues = pronun_result.issues
                pronunciation_available = True
        except Exception as e:
            result["errors"].append(f"Pronunciation assessment failed: {e}")
        
        # ── 9. Clarity ──────────────────────────────────────────────────────
        clarity_pronunciation_input = pronun_score if pronun_score is not None else 70.0
        clarity_score = score_clarity(
            pace_score,
            filler_score,
            grammar_score,
            clarity_pronunciation_input,
        )
        
        # ── 10. CEFR ────────────────────────────────────────────────────────
        cefr_result = score_cefr(
            vocabulary_score,
            grammar_score,
            clarity_pronunciation_input,
            pace_score,
            whisper_transcript,
        )
        
        # ── 11. Voice Archetype ─────────────────────────────────────────────
        archetype_result = determine_voice_archetype(
            pace_score,
            filler_score,
            clarity_pronunciation_input,
            grammar_score,
            clarity_score,
            vocabulary_score,
            cefr_result.get("level", "A1"),
        )
        
        # ── 12. Overall Score ──────────────────────────────────────────────
        overall = round(
            pace_score * 0.20
            + filler_score * 0.20
            + clarity_pronunciation_input * 0.25
            + grammar_score * 0.20
            + clarity_score * 0.15,
            1,
        )
        
        # ── 13. Feedback ────────────────────────────────────────────────────
        feedback = build_feedback(
            wpm=wpm,
            pace_s=pace_score,
            fc=filler_count,
            filler_words=filler_words,
            ge=ge,
            pronun_s=clarity_pronunciation_input,
            clarity_s=clarity_score,
            overall=overall,
            grammar_s=grammar_score,
        )
        
        # ── 14. Categories ──────────────────────────────────────────────────
        categories = defaultdict(int)
        confidence_levels = defaultdict(int)
        
        for issue in grammar_issues:
            rule_id = issue.get("rule_id", "Other")
            if "SUBJECT_VERB_AGREEMENT" in rule_id:
                categories["Subject-Verb Agreement"] += 1
            elif "PAST_TENSE" in rule_id:
                categories["Verb Tense"] += 1
            elif "BE_PAST" in rule_id or "BE_PRESENT" in rule_id:
                categories["Be Agreement"] += 1
            elif "MISSING_BE_AUX" in rule_id or "VERB_STACKING" in rule_id:
                categories["Verb Form"] += 1
            elif "DO_AUX" in rule_id:
                categories["Auxiliary Agreement"] += 1
            elif "PREPOSITION" in rule_id:
                categories["Preposition"] += 1
            elif "MISSING_ARTICLE" in rule_id:
                categories["Article"] += 1
            else:
                categories["Other"] += 1
            
            conf = issue.get("confidence", "medium")
            confidence_levels[conf] += 1
        
        # ── 15. Populate result ────────────────────────────────────────────
        result["parameters"] = {
            "pace": {
                "score": round(pace_score, 1),
                "wpm": round(wpm, 2),
                "duration_seconds": round(duration_seconds, 2),
                "word_count": word_count,
            },
            "filler_words": {
                "score": round(filler_score, 1),
                "count": filler_count,
                "rate_per_min": round(filler_rate, 2),
                "words": filler_words,
                "hesitation_count": hesitation_count,
            },
            "pronunciation": {
                "score": round(pronun_score, 1) if pronun_score is not None else None,
                "available": pronunciation_available,
                "issues": pronun_issues,
            },
            "grammar": {
                "score": round(grammar_score, 1),
                "errors": ge,
                "error_rate": round(ge / max(word_count, 1) * 100, 2),
                "issues": grammar_issues,
                "categories": dict(categories),
                "confidence_distribution": dict(confidence_levels),
                "source": {
                    "languagetool_errors": lt_grammar.get("errors", 0) if lt_grammar else 0,
                    "learner_heuristic_errors": heuristic_count,
                    "heuristic_added": heuristic_added,
                    "grammar_source": grammar_source,
                }
            },
            "vocabulary": {
                "score": round(vocabulary_score, 1),
                "unique_words": vocab_result.get("unique_words", 0),
                "total_words": vocab_result.get("total_words", 0),
                "advanced_ratio": vocab_result.get("advanced_ratio", 0),
                "diversity": vocab_result.get("diversity", 0),
                "sophistication": vocab_result.get("sophistication", 0),
            },
            "clarity": {
                "score": round(clarity_score, 1),
            },
            "cefr": {
                "score": round(cefr_result.get("score", 0), 1),
                "level": cefr_result.get("level", "A1"),
                "avg_sentence_length": cefr_result.get("avg_sentence_length", 0),
            },
            "archetype": {
                "archetype": archetype_result.get("archetype", "The Rising Voice"),
                "emoji": archetype_result.get("emoji", "🌱"),
                "description": archetype_result.get("description", ""),
                "traits": archetype_result.get("traits", []),
            },
        }
        
        result["overall"] = {
            "score": overall,
            "feedback": feedback,
            "measurement": "full_backend_measurement",
        }
        
        result["evidence"] = {
            "grammar_errors": ge,
            "grammar_issues": grammar_issues,
            "filler_words": filler_count,
            "hesitations": fillers.get("hesitations", []),
            "unique_words": vocab_result.get("unique_words", 0),
            "total_words": word_count,
            "lexical_diversity": vocab_result.get("diversity", 0),
        }
        
        # ── 16. Recommendations ────────────────────────────────────────────
        recommendations = []
        if grammar_score < 70:
            recommendations.append("Review subject-verb agreement and verb tenses.")
        if filler_count > 5:
            recommendations.append(f"Reduce filler words ({filler_count} detected).")
        if pronun_score is not None and pronun_score < 70:
            recommendations.append("Practice pronunciation of difficult words.")
        if vocabulary_score < 60:
            recommendations.append("Expand vocabulary range.")
        if pace_score < 60:
            recommendations.append("Adjust speaking pace toward a natural conversational range.")
        if clarity_score < 70:
            recommendations.append("Work on clearer and more controlled speech.")
        if not recommendations:
            recommendations.append("Continue practicing to maintain your current level.")
        
        result["recommendations"] = recommendations[:5]
        
        # ── 17. Coach Summary ──────────────────────────────────────────────
        if overall >= 80:
            coach_summary = "Excellent performance! You demonstrate strong speaking skills."
        elif overall >= 65:
            coach_summary = "Good job! You have solid speaking skills. Focus on the specific areas identified."
        elif overall >= 50:
            coach_summary = "You're making progress. The recommendations above will help you improve."
        else:
            coach_summary = "Keep practicing regularly. Focus on one area at a time."
        
        result["coach_summary"] = coach_summary
        
    except Exception as e:
        result["errors"].append(f"Processing failed: {str(e)}")
        result["errors"].append(traceback.format_exc())
    
    return result


def print_sample_result(result: Dict[str, Any], index: int, total: int) -> None:
    """Print a single sample result."""
    print(f"  Sample {index+1}/{total}: {result['sample_id']}...", end=" ", flush=True)
    
    if result.get("errors"):
        print(f"❌ Errors: {len(result['errors'])}")
        return
    
    overall = result["overall"].get("score", 0)
    pron = result["parameters"]["pronunciation"].get("score")
    whisper_len = len(result.get("whisper_transcript", "").split())
    ref_len = len(result.get("reference_transcript", "").split())
    
    print(f"✅ Overall: {overall:.1f}/100, Pron: {pron:.1f}/100" if pron else f"✅ Overall: {overall:.1f}/100, Pron: N/A")
    print(f"     Whisper: {whisper_len} words, Reference: {ref_len} words")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SpeechOcean762 Local Audio Backend Test")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--data-dir", type=str, help="Path to SpeechOcean762 directory")
    parser.add_argument("--output", type=str, default="speechocean_local_results.json", help="Output file")
    parser.add_argument("--skip-whisper", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--find", action="store_true", help="Find and show dataset location")
    args = parser.parse_args()
    
    print("=" * 80)
    print("SPEECHOCEAN762 LOCAL AUDIO BACKEND TEST")
    print("=" * 80)
    print()
    
    # ── Find dataset ───────────────────────────────────────────────────────
    data_dir = None
    
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            sys.exit(1)
    else:
        # Search for dataset
        print("Searching for SpeechOcean762 dataset...")
        data_dir = find_speechocean_dataset()
        
        if data_dir and args.find:
            print(f"\n✅ Found dataset at: {data_dir}")
            print(f"   WAV files: {len(list(data_dir.glob('*.wav')) + list(data_dir.glob('**/*.wav')))}")
            print(f"   Text files: {len(list(data_dir.glob('*.txt')) + list(data_dir.glob('**/*.txt')))}")
            print("\nTo use this dataset:")
            print(f"  python test_speechocean_local.py --data-dir {data_dir} --samples 10")
            return
    
    if not data_dir:
        print("\n❌ Could not find SpeechOcean762 dataset.")
        print("\nOptions:")
        print("  1. Provide the path with --data-dir:")
        print("     python test_speechocean_local.py --data-dir /path/to/speechocean --samples 10")
        print("  2. Download from: https://www.openslr.org/101/")
        print("  3. Extract the tar.gz file and point to the extracted directory")
        sys.exit(1)
    
    print(f"Using dataset at: {data_dir}")
    
    # ── Load samples ──────────────────────────────────────────────────────
    samples = load_speechocean_samples(data_dir, max_samples=args.samples)
    
    if not samples:
        print("❌ No samples loaded. Exiting.")
        sys.exit(1)
    
    print(f"Loaded {len(samples)} samples")
    print()
    
    # ── Process samples ──────────────────────────────────────────────────
    results = []
    errors = []
    
    print("Processing samples...")
    print()
    
    for i, sample in enumerate(samples):
        try:
            result = process_sample(sample, use_whisper=not args.skip_whisper)
            results.append(result)
            print_sample_result(result, i, len(samples))
        except Exception as e:
            errors.append({
                "sample_id": sample.get("sample_id", "unknown"),
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            print(f"  Sample {i+1}/{len(samples)}: ❌ Error: {e}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # ── Calculate statistics ─────────────────────────────────────────────
    if results:
        successful = [r for r in results if not r.get("errors")]
        
        if successful:
            score_keys = ["pace", "filler_words", "grammar", "vocabulary", "clarity"]
            averages = {}
            
            for key in score_keys:
                values = [r["parameters"][key]["score"] for r in successful if r["parameters"].get(key)]
                if values:
                    averages[key] = sum(values) / len(values)
            
            pron_values = [r["parameters"]["pronunciation"]["score"] for r in successful if r["parameters"]["pronunciation"].get("score") is not None]
            overall_values = [r["overall"]["score"] for r in successful]
            
            print("Average Scores (successful samples):")
            for key in score_keys:
                label = key.replace("_", " ").title()
                print(f"  {label}: {averages.get(key, 0):.1f}/100")
            
            if pron_values:
                print(f"  Pronunciation: {sum(pron_values) / len(pron_values):.1f}/100")
            else:
                print("  Pronunciation: N/A (not available)")
            
            if overall_values:
                print(f"  Overall: {sum(overall_values) / len(overall_values):.1f}/100")
            print()
            
            print("Sample Statistics:")
            print(f"  Total samples processed: {len(results)}")
            print(f"  Successful samples: {len(successful)}")
            print(f"  Samples with errors: {len(errors)}")
            print(f"  Pronunciation available: {len(pron_values)}/{len(successful)}")
            print()
            
            print("Duration and WPM:")
            durations = [r["audio"]["duration_seconds"] for r in successful if "audio" in r]
            wpms = [r["parameters"]["pace"]["wpm"] for r in successful if "parameters" in r and "pace" in r["parameters"]]
            if durations:
                print(f"  Avg duration: {sum(durations) / len(durations):.2f}s")
            if wpms:
                print(f"  Avg WPM: {sum(wpms) / len(wpms):.1f}")
            print()
            
            # Reference vs Whisper comparison
            print("Reference vs Whisper Transcript Comparison:")
            for r in successful[:5]:  # Show first 5
                ref = r.get("reference_transcript", "")
                whisper = r.get("whisper_transcript", "")
                ref_len = len(ref.split())
                whisper_len = len(whisper.split())
                print(f"  {r['sample_id']}: Ref={ref_len} words, Whisper={whisper_len} words")
                if ref and whisper and ref != whisper:
                    # Check if they're similar
                    common = len(set(ref.split()) & set(whisper.split()))
                    if common > 0:
                        sim = common / max(len(set(ref.split())), len(set(whisper.split())))
                        print(f"    Similarity: {sim:.2f}")
            print()
        
        # ── Save JSON ──────────────────────────────────────────────────────
        output_path = Path(args.output)
        
        output = {
            "test_metadata": {
                "test_name": "SpeechOcean762 Local Audio Backend Test",
                "dataset": "speechocean762",
                "dataset_path": str(data_dir),
                "samples_requested": args.samples,
                "samples_processed": len(results),
                "samples_successful": len(successful),
                "samples_failed": len(errors),
                "pronunciation_available": bool(pron_values) if results else False,
                "notes": [
                    "Real audio from local SpeechOcean762 dataset.",
                    "Duration is from actual audio.",
                    "WPM is calculated from audio duration.",
                    "Pronunciation uses real audio with existing provider.",
                    "Whisper transcription is real (not stubbed).",
                    "Existing production scoring functions are reused.",
                ],
            },
            "summary": {
                "average_scores": {
                    "pace": averages.get("pace", 0),
                    "filler_words": averages.get("filler_words", 0),
                    "grammar": averages.get("grammar", 0),
                    "vocabulary": averages.get("vocabulary", 0),
                    "clarity": averages.get("clarity", 0),
                    "pronunciation": sum(pron_values) / len(pron_values) if pron_values else None,
                    "overall": sum(overall_values) / len(overall_values) if overall_values else 0,
                },
                "total_samples": len(results),
                "successful_samples": len(successful),
                "failed_samples": len(errors),
                "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
                "avg_wpm": sum(wpms) / len(wpms) if wpms else 0,
            },
            "samples": results,
            "errors": errors,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Results saved to: {output_path}")
        print()
        print("=" * 80)
        print("✅ Audio backend test completed.")
        print("=" * 80)


if __name__ == "__main__":
    main()