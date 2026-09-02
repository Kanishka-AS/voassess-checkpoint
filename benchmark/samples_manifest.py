"""
Ground-truth verbatim scripts for the STT assessment-benchmark samples.

These are SYNTHETIC (espeak-ng TTS) samples generated in this sandbox
because the repo ships no real recorded audio and no dataset — see
BENCHMARK_REPORT.md, "What was actually run here" for why. They are useful
for a structural/plumbing check of the harness and providers, but they are
NOT a substitute for running this same harness against real learner
recordings (accented, hesitant, imperfect audio) — TTS speech is clean and
robotic and doesn't stress a model's acoustic robustness the way a real
learner's voice does. Re-run against real samples before trusting the
numbers for a provider decision.

Each entry tags exactly which tokens are the "phenomena of interest" so the
analyzer can check retention without needing a second STT/NLP pass.
"""

SAMPLES = {
    "sample_filler_heavy": {
        "text": "So, um, I think, uh, the best way to, like, solve this problem is, you know, to break it down.",
        "fillers": ["um", "uh"],                 # filled pauses
        "discourse_fillers": ["like", "you know"],  # discourse/hedge fillers
        "repetitions": [],
        "false_starts": [],
        "notes": "Dense fillers: um, uh, like, you know",
    },
    "sample_repetition": {
        "text": "I I really think that the the meeting went well yesterday.",
        "fillers": [],
        "discourse_fillers": [],
        "repetitions": ["i i", "the the"],
        "false_starts": [],
        "notes": "Immediate word repetitions: I I / the the",
    },
    "sample_false_start": {
        "text": "We should go to the- I mean, we should go to the store tomorrow instead.",
        "fillers": [],
        "discourse_fillers": ["i mean"],
        "repetitions": [],
        "false_starts": ["go to the-"],
        "notes": "Self-correction / false start mid-sentence",
    },
    "sample_clean": {
        "text": "The weather today is quite pleasant and I plan to go for a walk in the park.",
        "fillers": [],
        "discourse_fillers": [],
        "repetitions": [],
        "false_starts": [],
        "notes": "Clean baseline, no disfluencies",
    },
}
