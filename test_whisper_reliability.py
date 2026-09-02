#!/usr/bin/env python3
"""
Test Whisper reliability check on SpeechOcean762 samples.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import app
from pronunciation_provider import is_whisper_reliable

# Load the benchmark results - try multiple paths
paths = [
    Path('../evaluation_results/speechocean762_latest.json'),
    Path('./evaluation_results/speechocean762_latest.json'),
    Path('/home/kanish/voassess-main (1)/evaluation_results/speechocean762_latest.json'),
]

json_path = None
for p in paths:
    if p.exists():
        json_path = p
        break

if json_path is None:
    print("❌ Could not find speechocean762_latest.json")
    print("Please run the evaluation first:")
    print("  python test_speechocean_evaluation.py --data-dir ./speechocean_data/speechocean762 --limit 50")
    sys.exit(1)

print(f"Loading: {json_path}")
with open(json_path, 'r') as f:
    data = json.load(f)

samples = data['samples']

print("=" * 70)
print("WHISPER RELIABILITY CHECK - SPEECHOCEAN762 SAMPLES")
print("=" * 70)
print()

reliable_samples = []
unreliable_samples = []

for sample in samples:
    # Get the audio path and transcribe
    audio_path = Path(sample['audio_file'])
    try:
        transcript, segments = app.transcribe_wav(audio_path)
        reliability = is_whisper_reliable(segments)
        
        sample['reliability'] = reliability
        sample['whisper_transcript_actual'] = transcript
        
        if reliability['reliable']:
            reliable_samples.append(sample)
        else:
            unreliable_samples.append(sample)
            
        status = "✅ RELIABLE" if reliability['reliable'] else "❌ UNRELIABLE"
        print(f"{status} {sample['sample_id']}: prob={reliability['avg_word_prob']:.2f} - {reliability['reason']}")
        
    except Exception as e:
        print(f"❌ ERROR {sample['sample_id']}: {e}")
        unreliable_samples.append(sample)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total samples: {len(samples)}")
print(f"Reliable: {len(reliable_samples)} ({len(reliable_samples)/len(samples)*100:.1f}%)")
print(f"Unreliable: {len(unreliable_samples)} ({len(unreliable_samples)/len(samples)*100:.1f}%)")

# Show examples of unreliable samples
print()
print("Unreliable sample examples (5):")
for sample in unreliable_samples[:5]:
    print(f"  {sample['sample_id']}: prob={sample['reliability']['avg_word_prob']:.2f}")
    print(f"    Reason: {sample['reliability']['reason']}")
    print(f"    Transcript: {sample.get('whisper_transcript_actual', '')[:60]}...")
    
# Show examples of reliable samples
print()
print("Reliable sample examples (5):")
for sample in reliable_samples[:5]:
    print(f"  {sample['sample_id']}: prob={sample['reliability']['avg_word_prob']:.2f}")
    print(f"    Transcript: {sample.get('whisper_transcript_actual', '')[:60]}...")