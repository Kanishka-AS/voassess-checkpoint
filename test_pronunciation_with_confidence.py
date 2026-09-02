#!/usr/bin/env python3
"""
Test pronunciation scoring with confidence filter.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import app
from pronunciation_provider import is_whisper_reliable, _score_pronunciation_from_segments

def score_pronunciation_with_confidence(segments, threshold=0.5):
    """Score pronunciation only if Whisper was reliable."""
    reliability = is_whisper_reliable(segments, threshold=threshold)
    
    if not reliability['reliable']:
        return {
            "score": None,
            "available": False,
            "reason": reliability['reason'],
            "avg_word_prob": reliability['avg_word_prob']
        }
    
    # Score normally
    score = _score_pronunciation_from_segments(segments)
    return {
        "score": score,
        "available": True,
        "reason": None,
        "avg_word_prob": reliability['avg_word_prob']
    }

# Load benchmark data
with open('./evaluation_results/speechocean762_latest.json', 'r') as f:
    data = json.load(f)

samples = data['samples']

print("=" * 70)
print("PRONUNCIATION SCORING WITH CONFIDENCE FILTER")
print("=" * 70)
print()

results = []
for sample in samples[:20]:  # Test first 20 samples
    sample_id = sample['sample_id']
    audio_path = Path(sample['audio_file'])
    reference_score = sample['reference_score']
    
    try:
        transcript, segments = app.transcribe_wav(audio_path)
        score_result = score_pronunciation_with_confidence(segments)
        
        results.append({
            'sample_id': sample_id,
            'reference_score': reference_score,
            'old_score': sample['predicted_score'],
            'new_score': score_result['score'],
            'available': score_result['available'],
            'reason': score_result.get('reason'),
            'avg_word_prob': score_result.get('avg_word_prob'),
            'transcript': transcript
        })
        
        status = "✅ SCORED" if score_result['available'] else "❌ SKIPPED"
        print(f"{status} {sample_id}: ref={reference_score} old={sample['predicted_score']} new={score_result['score']}")
        if not score_result['available']:
            print(f"    Reason: {score_result['reason']}")
            
    except Exception as e:
        print(f"❌ ERROR {sample_id}: {e}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

scored = [r for r in results if r['available']]
skipped = [r for r in results if not r['available']]

print(f"Scored: {len(scored)}")
print(f"Skipped: {len(skipped)}")

if scored:
    old_scores = [r['old_score'] for r in scored if r['old_score'] is not None]
    new_scores = [r['new_score'] for r in scored if r['new_score'] is not None]
    ref_scores = [r['reference_score'] for r in scored]
    
    print()
    print("Comparison (scored samples only):")
    print(f"  Reference mean: {sum(ref_scores)/len(ref_scores):.2f}")
    print(f"  Old score mean: {sum(old_scores)/len(old_scores):.2f}")
    print(f"  New score mean: {sum(new_scores)/len(new_scores):.2f}")
    
    # Show sample details
    print()
    print("Sample details (scored):")
    for r in scored[:5]:
        print(f"  {r['sample_id']}: ref={r['reference_score']} old={r['old_score']} new={r['new_score']}")
        print(f"    Transcript: {r['transcript'][:60]}...")