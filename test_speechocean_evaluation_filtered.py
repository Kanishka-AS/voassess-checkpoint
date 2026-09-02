
import sys
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import app
from pronunciation_provider import is_whisper_reliable, _score_pronunciation_from_segments
from audio_utils import wav_duration_seconds

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def std(xs):
    if len(xs) < 2:
        return 0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def pearson_correlation(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom == 0:
        return None
    return num / denom

# Find the JSON file
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
print(f"Testing {len(samples)} samples with reliability filter...")
print()

results = []
scored_samples = []

for i, sample in enumerate(samples):
    sample_id = sample['sample_id']
    audio_path = Path(sample['audio_file'])
    reference_score = sample['reference_score']
    reference_accuracy = sample.get('reference_accuracy_raw')
    old_score = sample['predicted_score']
    
    try:
        transcript, segments = app.transcribe_wav(audio_path)
        
        # Get reliability
        reliability = is_whisper_reliable(segments, threshold=0.5)
        
        if reliability['reliable']:
            # Score normally
            new_score = _score_pronunciation_from_segments(segments)
            scored_samples.append({
                'sample_id': sample_id,
                'reference_score': reference_score,
                'old_score': old_score,
                'new_score': new_score,
                'transcript': transcript,
                'reference_transcript': sample.get('reference_transcript', ''),
                'avg_word_prob': reliability['avg_word_prob']
            })
            status = "✅ SCORED"
        else:
            status = "❌ SKIPPED"
        
        print(f"{status} {sample_id}: ref={reference_score} old={old_score} new={new_score if reliability['reliable'] else 'N/A'}")
        
    except Exception as e:
        print(f"❌ ERROR {sample_id}: {e}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

total = len(samples)
scored_count = len(scored_samples)
skipped_count = total - scored_count

print(f"Total samples: {total}")
print(f"Scored (reliable): {scored_count} ({scored_count/total*100:.1f}%)")
print(f"Skipped (unreliable): {skipped_count} ({skipped_count/total*100:.1f}%)")

if scored_samples:
    refs = [s['reference_score'] for s in scored_samples]
    old_scores = [s['old_score'] for s in scored_samples if s['old_score'] is not None]
    new_scores = [s['new_score'] for s in scored_samples if s['new_score'] is not None]
    
    # Calculate metrics
    old_errors = [abs(old - ref) for old, ref in zip(old_scores, refs)]
    new_errors = [abs(new - ref) for new, ref in zip(new_scores, refs)]
    
    old_signed = [old - ref for old, ref in zip(old_scores, refs)]
    new_signed = [new - ref for new, ref in zip(new_scores, refs)]
    
    old_r = pearson_correlation(refs, old_scores)
    new_r = pearson_correlation(refs, new_scores)
    
    print()
    print("Metrics (scored samples only):")
    print(f"  Count: {scored_count}")
    print(f"  Reference mean: {mean(refs):.2f}")
    print(f"  Old score mean: {mean(old_scores):.2f}")
    print(f"  New score mean: {mean(new_scores):.2f}")
    print(f"  Old MAE: {mean(old_errors):.2f}")
    print(f"  New MAE: {mean(new_errors):.2f}")
    print(f"  Old RMSE: {math.sqrt(mean([e**2 for e in old_errors])):.2f}")
    print(f"  New RMSE: {math.sqrt(mean([e**2 for e in new_errors])):.2f}")
    print(f"  Old Pearson r: {old_r:.3f}" if old_r else "  Old Pearson r: N/A")
    print(f"  New Pearson r: {new_r:.3f}" if new_r else "  New Pearson r: N/A")
    print(f"  Old mean signed error: {mean(old_signed):.2f}")
    print(f"  New mean signed error: {mean(new_signed):.2f}")
    
    # Improvement
    improvement = mean(old_errors) - mean(new_errors)
    print(f"  MAE Improvement: {improvement:.2f} ({improvement/mean(old_errors)*100:.1f}%)")
    
    # Sample comparison
    print()
    print("=" * 70)
    print("SAMPLE COMPARISON (10 representative samples)")
    print("=" * 70)
    print()
    
    # Sort by sample_id
    for s in scored_samples[:10]:
        old_error = abs(s['old_score'] - s['reference_score']) if s['old_score'] is not None else None
        new_error = abs(s['new_score'] - s['reference_score'])
        improvement_score = (old_error - new_error) if old_error is not None else None
        
        print(f"{s['sample_id']}:")
        print(f"  Reference: {s['reference_score']}")
        print(f"  Old score: {s['old_score']} (error: {old_error})")
        print(f"  New score: {s['new_score']} (error: {new_error:.1f})")
        if improvement_score is not None:
            print(f"  Improvement: {improvement_score:.1f} points")
        print(f"  Avg word prob: {s['avg_word_prob']:.3f}")
        print(f"  Reference text: {s['reference_transcript'][:50]}...")
        print(f"  Whisper text: {s['transcript'][:50]}...")
        print()

# Save results
output = {
    "summary": {
        "total": total,
        "scored": scored_count,
        "skipped": skipped_count,
        "reference_mean": mean(refs) if scored_samples else None,
        "old_score_mean": mean(old_scores) if scored_samples else None,
        "new_score_mean": mean(new_scores) if scored_samples else None,
        "old_mae": mean(old_errors) if scored_samples else None,
        "new_mae": mean(new_errors) if scored_samples else None,
        "old_pearson_r": old_r if scored_samples else None,
        "new_pearson_r": new_r if scored_samples else None,
        "old_mean_signed_error": mean(old_signed) if scored_samples else None,
        "new_mean_signed_error": mean(new_signed) if scored_samples else None,
        "mae_improvement": improvement if scored_samples else None,
    },
    "scored_samples": scored_samples
}

output_path = Path('./evaluation_results/speechocean762_filtered_results.json')
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")