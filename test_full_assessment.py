#python
#!/usr/bin/env python3
"""
Full Assessment Backend Test Harness

Runs realistic speaking-assessment transcripts through the EXISTING backend
assessment functions and produces structured, Debug-UI-ready JSON.

IMPORTANT:
- This is a THIN HARNESS.
- Existing production scoring functions are reused.
- No frontend/UI code is modified.
- Text-only tests cannot produce real pronunciation measurements.
- WPM is estimated from an explicit test duration rather than pretending
  that every transcript was spoken at exactly 150 WPM.

Usage:
    python test_full_assessment.py

Output:
    full_assessment_results.json
"""

import os
import sys
import types
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any


# ============================================================================
# WHISPER STUB
# ============================================================================

# Some backend imports expect whisper to exist.
# The test harness does not need Whisper because transcripts are supplied
# directly.
if "whisper" not in sys.modules:
    _stub = types.ModuleType("whisper")

    class _StubModel:
        def transcribe(self, *args, **kwargs):
            return {
                "text": "stub transcript",
                "segments": [{"avg_logprob": -0.2}]
            }

    _stub.load_model = lambda name: _StubModel()
    sys.modules["whisper"] = _stub


# ============================================================================
# BACKEND IMPORTS
# ============================================================================

import app

from app import (
    score_grammar,
    score_fillers,
    score_pace,
    score_clarity,
    score_cefr,
    determine_voice_archetype,
    build_feedback,
    DEFAULT_PRONUNCIATION_PROVIDER,
    resolve_grammar,
    _lt_provider,
)

from languagetool_provider import LanguageToolUnavailable

from grammar_heuristics import detect_learner_errors
from grammar_pos_rules import detect_pos_aware_errors

from vocabulary import score_vocabulary

from filler_detector import (
    detect_fillers,
    summarize_words,
)

from pronunciation_provider import resolve_pronunciation


# ============================================================================
# TEST TRANSCRIPTS
# ============================================================================

# Explicit duration_seconds is supplied because these are text-only tests.
#
# In a real audio assessment, duration MUST come from the actual audio.
#
# These durations intentionally vary so pacing is actually exercised.

TRANSCRIPTS = [

    {
        "name": "01_clean_self_introduction",
        "category": "clean",
        "duration_seconds": 27.0,
        "text": """
Hello, my name is John. I am a student and I study computer science.
I like programming and I enjoy learning new technologies.
My favorite subject is artificial intelligence.
I think it combines programming, mathematics, and problem-solving.
I want to become a software engineer.
In my free time I read books and play sports.
I believe that hard work and dedication are important for success.
""".strip()
    },

    {
        "name": "02_learner_typical_day",
        "category": "learner",
        "duration_seconds": 39.0,
        "text": """
Every day I wake up at 7 am. I have breakfast and then I go to college.
I study for three hours every day. I use different colored pens for making notes.
Yesterday I come to college late because I miss the bus.
My teacher explain the lesson again for me.
I have lunch at 1 pm with my friends.
In the evening I do my homework and review my notes.
I go to bed at 11 pm every night.
""".strip()
    },

    {
        "name": "03_learner_problem_solving",
        "category": "learner",
        "duration_seconds": 30.0,
        "text": """
Yesterday I face a difficult problem in my project.
I try to solve it but I cannot find the solution.
My friend suggest me to ask the teacher for help.
The teacher explain the problem very clearly.
I understand the solution after his explanation.
I am happy that I complete the project on time.
This experience teach me that asking for help is important.
""".strip()
    },

    {
        "name": "04_learner_memorable_trip",
        "category": "learner",
        "duration_seconds": 31.0,
        "text": """
Last summer I go to Goa with my family.
We stay in a nice hotel near the beach.
I enjoy swimming in the sea and playing in the sand.
My sister make a sand castle.
We eat delicious seafood every day.
I buy many souvenirs for my friends.
This was one of the best trips I have ever taken.
""".strip()
    },

    {
        "name": "05_learner_strengths_weaknesses",
        "category": "learner",
        "duration_seconds": 28.0,
        "text": """
I think my biggest strength is my ability to learn new things quickly.
I have good communication skills and I enjoy working in teams.
My weakness is that I sometimes have difficulty with time management.
Yesterday I have a problem with a deadline.
I need to work on improving my organization.
I know that practice makes perfect.
I am trying to become better every day.
""".strip()
    },

    {
        "name": "06_learner_technology",
        "category": "learner",
        "duration_seconds": 24.0,
        "text": """
Technology is changing the world in many ways.
I use my smartphone for many things like communication and entertainment.
Social media helps me stay connected with my friends.
I think that technology has both advantages and disadvantages.
On one hand it makes our lives easier.
On the other hand it can be distracting.
We need to use technology responsibly.
""".strip()
    },

    {
        "name": "07_learner_person_i_admire",
        "category": "learner",
        "duration_seconds": 25.0,
        "text": """
The person I admire most is my mother.
She is a teacher and she works very hard.
She always helps me with my studies.
My mother teaches me many important values.
She is kind, patient, and understanding.
I want to be like her when I grow up.
I am grateful for everything she has done for me.
""".strip()
    },

    {
        "name": "08_learner_missed_appointment",
        "category": "learner",
        "duration_seconds": 27.0,
        "text": """
I am sorry that I missed the appointment yesterday.
I have a good reason for being late.
My car broke down on the way to the office.
I call the mechanic and he comes to help me.
It takes two hours to fix the car.
I try to call you but your phone was switched off.
I hope we can reschedule the meeting for next week.
""".strip()
    },

    {
        "name": "09_learner_ai_education_opinion",
        "category": "learner",
        "duration_seconds": 29.0,
        "text": """
I believe that AI will transform education in the future.
It can personalize learning for each student.
AI teachers can provide instant feedback and support.
However I also think that human teachers are irreplaceable.
They provide emotional support and motivation.
A combination of AI and human teaching is the best approach.
This will create a more effective learning environment.
""".strip()
    },

    {
        "name": "10_natural_fillers_and_speech",
        "category": "natural_speech",
        "duration_seconds": 37.0,
        "text": """
Um, I think that, you know, the most important thing is to practice every day.
Like, when I was learning English, I would, um, try to speak with native speakers.
You know, it really helped me improve my fluency.
And, uh, I also watched a lot of movies and TV shows in English.
Basically, I found that immersion is the key to language learning.
So, yeah, I would definitely recommend that approach to others.
I mean, it worked really well for me.
""".strip()
    },

    {
        "name": "11_clean_weekend",
        "category": "clean",
        "duration_seconds": 30.0,
        "text": """
I had a wonderful weekend.
On Saturday I went to the park with my friends.
We played football and had a picnic.
The weather was perfect.
On Sunday I stayed at home and read a book.
I also watched a movie.
I spent quality time with my family in the evening.
We cooked dinner together and shared stories.
It was a relaxing and enjoyable weekend.
I feel refreshed and ready for the new week.
""".strip()
    },

    {
        "name": "12_beginner_natural_speech",
        "category": "beginner",
        "duration_seconds": 25.0,
        "text": """
Hi, my name is Maria. I come from Spain.
I am student at university. I study English.
I have problem with speaking. I feel nervous.
My teacher help me with pronunciation.
Yesterday I practice speaking for one hour.
I think I make good progress.
I know playing games can help learn English.
""".strip()
    },
]


# ============================================================================
# BASIC METRICS
# ============================================================================

def count_words(text: str) -> int:
    """
    Count words using the same simple whitespace-based convention used
    by the original test harness.
    """
    return len(text.split())


def calculate_wpm(word_count: int, duration_seconds: float) -> float:
    """
    Calculate words per minute from actual/supplied duration.
    """
    if duration_seconds <= 0:
        return 0.0

    return word_count / (duration_seconds / 60.0)


# ============================================================================
# GRAMMAR CATEGORY MAPPING
# ============================================================================

def categorize_grammar_issue(issue: Dict[str, Any]) -> str:
    rule_id = issue.get("rule_id", "Other")

    if "SUBJECT_VERB_AGREEMENT" in rule_id:
        return "Subject-Verb Agreement"

    if "PAST_TENSE" in rule_id:
        return "Verb Tense"

    if "BE_PAST" in rule_id or "BE_PRESENT" in rule_id:
        return "Be Agreement"

    if "MISSING_BE_AUX" in rule_id or "VERB_STACKING" in rule_id:
        return "Verb Form"

    if "DO_AUX" in rule_id:
        return "Auxiliary Agreement"

    if "PREPOSITION" in rule_id:
        return "Preposition"

    if "MISSING_ARTICLE" in rule_id:
        return "Article"

    return "Other"


# ============================================================================
# SINGLE ASSESSMENT
# ============================================================================

def run_full_assessment(
    text: str,
    name: str = "unnamed",
    duration_seconds: float = 0.0,
) -> Dict[str, Any]:

    word_count = count_words(text)

    # ------------------------------------------------------------------------
    # Duration / WPM
    # ------------------------------------------------------------------------

    estimated_wpm = calculate_wpm(
        word_count,
        duration_seconds
    )

    # ------------------------------------------------------------------------
    # Result skeleton
    # ------------------------------------------------------------------------

    result = {
        "assessment_id": name,

        "input": {
            "text": text,
            "word_count": word_count,
            "duration_seconds": duration_seconds,
            "duration_source": "test_fixture",
            "wpm": round(estimated_wpm, 2),
        },

        "derived_metrics": {},

        "parameters": {},

        "overall": {},

        "recommendations": [],

        "coach_summary": "",

        "evidence": {},
    }

    # ========================================================================
    # 1. LANGUAGE TOOL
    # ========================================================================

    lt_grammar = None
    linguistic_analysis = None
    lt_errors = {}

    try:
        lt_grammar, linguistic_analysis, lt_errors = (
            _lt_provider.check_and_analyze(text)
        )

    except LanguageToolUnavailable:
        pass

    # ========================================================================
    # 2. GRAMMAR RESOLUTION
    # ========================================================================

    ge, grammar_issues, grammar_source, heuristic_added = resolve_grammar(
        text,
        lt_grammar,
        linguistic_analysis,
    )

    grammar_score = score_grammar(
        ge,
        word_count,
    )

    # ========================================================================
    # 3. LEARNER HEURISTICS
    # ========================================================================

    heuristic_errors = detect_learner_errors(text)

    heuristic_count = len(heuristic_errors)

    # ========================================================================
    # 4. POS-AWARE GRAMMAR
    # ========================================================================

    pos_aware_errors = []

    if linguistic_analysis:
        pos_aware_errors = detect_pos_aware_errors(
            text,
            linguistic_analysis,
        )

    pos_aware_count = len(pos_aware_errors)

    # ========================================================================
    # 5. VOCABULARY
    # ========================================================================

    vocab_result = score_vocabulary(text)

    vocabulary_score = vocab_result.get(
        "score",
        0,
    )

    # ========================================================================
    # 6. FILLERS
    # ========================================================================

    fillers = detect_fillers(
        text,
        linguistic_analysis,
        duration_seconds=duration_seconds,
    )

    filler_count = fillers.get(
        "count",
        0,
    )

    filler_words = summarize_words(
        fillers.get("occurrences", [])
    )

    filler_rate = fillers.get(
        "rate_per_min",
        0,
    )

    filler_score = score_fillers(
        filler_count,
        word_count,
    )

    hesitation_count = len(
        fillers.get("hesitations", [])
    )

    # ========================================================================
    # 7. PACE
    # ========================================================================

    pace_score = score_pace(
        estimated_wpm
    )

    # ========================================================================
    # 8. PRONUNCIATION
    # ========================================================================

    # IMPORTANT:
    # This test has transcript text only.
    #
    # Do NOT pretend that pronunciation=72 is a real measurement.
    #
    # We attempt the existing provider, but if real audio evidence is not
    # available, mark the parameter unavailable.

    pronun_score = None
    pronun_issues = []
    pronunciation_available = False

    try:
        pronun_result = resolve_pronunciation(
            DEFAULT_PRONUNCIATION_PROVIDER,
            text,
            [],
            None,
        )

        if pronun_result.available:
            pronun_score = pronun_result.score
            pronun_issues = pronun_result.issues
            pronunciation_available = True

    except Exception:
        pass

    # ========================================================================
    # 9. CLARITY
    # ========================================================================

    # The production clarity scorer currently expects pronunciation.
    #
    # When pronunciation is unavailable, use a neutral backend value only
    # for exercising the clarity function. This is explicitly marked as
    # test-only and must not be confused with real pronunciation evidence.

    clarity_pronunciation_input = (
        pronun_score
        if pronun_score is not None
        else 70.0
    )

    clarity_score = score_clarity(
        pace_score,
        filler_score,
        grammar_score,
        clarity_pronunciation_input,
    )

    # ========================================================================
    # 10. CEFR
    # ========================================================================

    cefr_result = score_cefr(
        vocabulary_score,
        grammar_score,
        clarity_pronunciation_input,
        pace_score,
        text,
    )

    # ========================================================================
    # 11. VOICE ARCHETYPE
    # ========================================================================

    archetype_result = determine_voice_archetype(
        pace_score,
        filler_score,
        clarity_pronunciation_input,
        grammar_score,
        clarity_score,
        vocabulary_score,
        cefr_result.get("level", "A1"),
    )

    # ========================================================================
    # 12. OVERALL SCORE
    # ========================================================================

    # Preserve the existing production weighting.
    #
    # If pronunciation is unavailable, do NOT silently pretend that the
    # 70.0 value is a measured pronunciation result.
    #
    # This test harness therefore calculates a "text-only overall" score
    # using the neutral test value and explicitly labels it.

    overall = round(
        pace_score * 0.20
        + filler_score * 0.20
        + clarity_pronunciation_input * 0.25
        + grammar_score * 0.20
        + clarity_score * 0.15,
        1,
    )

    # ========================================================================
    # 13. FEEDBACK
    # ========================================================================

    feedback = build_feedback(
        wpm=estimated_wpm,
        pace_s=pace_score,
        fc=filler_count,
        filler_words=filler_words,
        ge=ge,
        pronun_s=clarity_pronunciation_input,
        clarity_s=clarity_score,
        overall=overall,
        grammar_s=grammar_score,
    )

    # ========================================================================
    # 14. DERIVED METRICS
    # ========================================================================

    result["derived_metrics"] = {

        "word_count": word_count,

        "duration_seconds": round(
            duration_seconds,
            2,
        ),

        "wpm": round(
            estimated_wpm,
            2,
        ),

        "grammar_error_rate": round(
            ge / max(word_count, 1) * 100,
            2,
        ),

        "filler_rate_per_min": round(
            filler_rate,
            2,
        ),

        "hesitation_count": hesitation_count,

        "unique_words": vocab_result.get(
            "unique_words",
            0,
        ),

        "lexical_diversity": vocab_result.get(
            "diversity",
            0,
        ),

        "vocabulary_sophistication": vocab_result.get(
            "sophistication",
            0,
        ),

        "advanced_ratio": vocab_result.get(
            "advanced_ratio",
            0,
        ),

        "average_sentence_length": cefr_result.get(
            "avg_sentence_length",
            0,
        ),
    }

    # ========================================================================
    # 15. GRAMMAR BREAKDOWN
    # ========================================================================

    categories = defaultdict(int)
    confidence_levels = defaultdict(int)

    for issue in grammar_issues:

        category = categorize_grammar_issue(
            issue
        )

        categories[category] += 1

        confidence = issue.get(
            "confidence",
            "medium",
        )

        confidence_levels[confidence] += 1

    # ========================================================================
    # 16. PARAMETERS
    # ========================================================================

    result["parameters"] = {

        "pace": {
            "score": round(pace_score, 1),
            "wpm": round(estimated_wpm, 2),
            "duration_seconds": round(
                duration_seconds,
                2,
            ),
        },

        "filler_words": {
            "score": round(filler_score, 1),
            "count": filler_count,
            "rate_per_min": round(
                filler_rate,
                2,
            ),
            "words": filler_words,
            "hesitation_count": hesitation_count,
        },

        "pronunciation": {
            "score": (
                round(pronun_score, 1)
                if pronun_score is not None
                else None
            ),
            "available": pronunciation_available,
            "issues": pronun_issues,
            "measurement": (
                "real_backend_provider"
                if pronunciation_available
                else "unavailable_without_audio"
            ),
        },

        "grammar": {

            "score": round(
                grammar_score,
                1,
            ),

            "errors": ge,

            "error_rate": round(
                ge / max(word_count, 1) * 100,
                2,
            ),

            "issues": grammar_issues,

            "categories": dict(
                categories
            ),

            "confidence_distribution": dict(
                confidence_levels
            ),

            "source": {

                "languagetool_errors": (
                    lt_grammar.get(
                        "errors",
                        0,
                    )
                    if lt_grammar
                    else 0
                ),

                "learner_heuristic_errors": heuristic_count,

                "pos_aware_errors": pos_aware_count,

                "heuristic_added": heuristic_added,

                "grammar_source": grammar_source,
            },
        },

        "vocabulary": {

            "score": round(
                vocabulary_score,
                1,
            ),

            "unique_words": vocab_result.get(
                "unique_words",
                0,
            ),

            "total_words": vocab_result.get(
                "total_words",
                0,
            ),

            "advanced_ratio": vocab_result.get(
                "advanced_ratio",
                0,
            ),

            "diversity": vocab_result.get(
                "diversity",
                0,
            ),

            "sophistication": vocab_result.get(
                "sophistication",
                0,
            ),
        },

        "clarity": {

            "score": round(
                clarity_score,
                1,
            ),

            "note": (
                "Uses existing backend clarity scorer. "
                "Pronunciation component is neutral-test input "
                "when audio is unavailable."
            ),
        },

        "cefr": {

            "score": round(
                cefr_result.get(
                    "score",
                    0,
                ),
                1,
            ),

            "level": cefr_result.get(
                "level",
                "A1",
            ),

            "avg_sentence_length": cefr_result.get(
                "avg_sentence_length",
                0,
            ),
        },

        "archetype": {

            "archetype": archetype_result.get(
                "archetype",
                "The Rising Voice",
            ),

            "emoji": archetype_result.get(
                "emoji",
                "🌱",
            ),

            "description": archetype_result.get(
                "description",
                "",
            ),

            "traits": archetype_result.get(
                "traits",
                [],
            ),
        },
    }

    # ========================================================================
    # 17. OVERALL
    # ========================================================================

    result["overall"] = {

        "score": overall,

        "feedback": feedback,

        "measurement": (
            "text_only_test"
            if not pronunciation_available
            else "full_backend_measurement"
        ),
    }

    # ========================================================================
    # 18. EVIDENCE
    # ========================================================================

    result["evidence"] = {

        "grammar_errors": ge,

        "grammar_issues": grammar_issues,

        "learner_heuristic_errors": heuristic_errors,

        "pos_aware_errors": pos_aware_errors,

        "languagetool": lt_errors,

        "filler_words": filler_count,

        "filler_occurrences": fillers.get(
            "occurrences",
            [],
        ),

        "hesitations": fillers.get(
            "hesitations",
            [],
        ),

        "unique_words": vocab_result.get(
            "unique_words",
            0,
        ),

        "total_words": word_count,

        "lexical_diversity": vocab_result.get(
            "diversity",
            0,
        ),
    }

    # ========================================================================
    # 19. RECOMMENDATIONS
    # ========================================================================

    recommendations = []

    if grammar_score < 70:
        recommendations.append(
            "Review subject-verb agreement and verb tenses."
        )

    if filler_count > 5:
        recommendations.append(
            f"Reduce filler words ({filler_count} detected)."
        )

    if (
        pronunciation_available
        and pronun_score is not None
        and pronun_score < 70
    ):
        recommendations.append(
            "Practice pronunciation of difficult words."
        )

    if vocabulary_score < 60:
        recommendations.append(
            "Expand vocabulary range."
        )

    if pace_score < 60:
        recommendations.append(
            "Adjust speaking pace toward a natural conversational range."
        )

    if clarity_score < 70:
        recommendations.append(
            "Work on clearer and more controlled speech."
        )

    if not recommendations:
        recommendations.append(
            "Continue practicing to maintain your current level."
        )

    result["recommendations"] = recommendations[:5]

    # ========================================================================
    # 20. COACH SUMMARY
    # ========================================================================

    if overall >= 80:

        coach_summary = (
            "Excellent performance! You demonstrate strong speaking skills."
        )

    elif overall >= 65:

        coach_summary = (
            "Good job! You have solid speaking skills. "
            "Focus on the specific areas identified."
        )

    elif overall >= 50:

        coach_summary = (
            "You're making progress. "
            "The recommendations above will help you improve."
        )

    else:

        coach_summary = (
            "Keep practicing regularly. "
            "Focus on one area at a time."
        )

    result["coach_summary"] = coach_summary

    return result


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def print_assessment(
    result: Dict[str, Any]
) -> None:

    name = result["assessment_id"]

    params = result["parameters"]

    overall = result["overall"]

    print("=" * 70)
    print(f"ASSESSMENT: {name}")
    print("=" * 70)

    print()

    print(
        f"Words: {result['input']['word_count']}"
    )

    print(
        f"Duration: "
        f"{result['input']['duration_seconds']:.1f}s"
    )

    print(
        f"WPM: "
        f"{result['input']['wpm']:.1f}"
    )

    print()

    print("## SCORES")

    print(
        f"  Pace: "
        f"{params['pace']['score']:.1f}/100 "
        f"({params['pace']['wpm']:.1f} WPM)"
    )

    print(
        f"  Filler Words: "
        f"{params['filler_words']['score']:.1f}/100 "
        f"({params['filler_words']['count']} fillers)"
    )

    pronunciation_score = params["pronunciation"]["score"]

    if pronunciation_score is None:

        print(
            "  Pronunciation: "
            "N/A (audio required)"
        )

    else:

        print(
            f"  Pronunciation: "
            f"{pronunciation_score:.1f}/100"
        )

    print(
        f"  Grammar: "
        f"{params['grammar']['score']:.1f}/100 "
        f"({params['grammar']['errors']} errors)"
    )

    print(
        f"  Vocabulary: "
        f"{params['vocabulary']['score']:.1f}/100"
    )

    print(
        f"  Clarity: "
        f"{params['clarity']['score']:.1f}/100"
    )

    print()

    print("## GRAMMAR DETAILS")

    grammar_source = params["grammar"]["source"]

    print(
        f"  LanguageTool errors: "
        f"{grammar_source['languagetool_errors']}"
    )

    print(
        f"  Learner heuristic errors: "
        f"{grammar_source['learner_heuristic_errors']}"
    )

    print(
        f"  POS-aware errors: "
        f"{grammar_source['pos_aware_errors']}"
    )

    print(
        f"  Heuristic added: "
        f"{grammar_source['heuristic_added']}"
    )

    if params["grammar"]["categories"]:

        print("  Categories:")

        for category, count in (
            params["grammar"]["categories"].items()
        ):

            print(
                f"    {category}: {count}"
            )

    print()

    print(
        f"## CEFR: "
        f"{params['cefr']['level']} "
        f"({params['cefr']['score']:.1f}/100)"
    )

    print()

    print(
        f"## ARCHETYPE: "
        f"{params['archetype']['emoji']} "
        f"{params['archetype']['archetype']}"
    )

    print()

    print(
        f"## OVERALL: "
        f"{overall['score']:.1f}/100"
    )

    print(
        f"Measurement: "
        f"{overall['measurement']}"
    )

    print()

    print("## RECOMMENDATIONS")

    for recommendation in result.get(
        "recommendations",
        []
    ):

        print(
            f"  • {recommendation}"
        )

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("FULL ASSESSMENT BACKEND TEST")
    print("=" * 80)

    print()

    print(
        f"Testing {len(TRANSCRIPTS)} transcripts..."
    )

    print()

    results = []

    for transcript in TRANSCRIPTS:

        result = run_full_assessment(
            text=transcript["text"],
            name=transcript["name"],
            duration_seconds=transcript[
                "duration_seconds"
            ],
        )

        results.append(result)

        print_assessment(result)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()

    score_keys = [
        "pace",
        "filler_words",
        "grammar",
        "vocabulary",
        "clarity",
    ]

    averages = {}

    for key in score_keys:

        values = [
            r["parameters"][key]["score"]
            for r in results
        ]

        averages[key] = (
            sum(values) / len(values)
            if values
            else 0
        )

    pronunciation_values = [
        r["parameters"]["pronunciation"]["score"]
        for r in results
        if r["parameters"]["pronunciation"]["score"]
        is not None
    ]

    overall_values = [
        r["overall"]["score"]
        for r in results
    ]

    print("Average Scores:")

    for key in score_keys:

        label = key.replace(
            "_",
            " "
        ).title()

        print(
            f"  {label}: "
            f"{averages[key]:.1f}/100"
        )

    if pronunciation_values:

        print(
            f"  Pronunciation: "
            f"{sum(pronunciation_values) / len(pronunciation_values):.1f}/100"
        )

    else:

        print(
            "  Pronunciation: "
            "N/A (audio required)"
        )

    print(
        f"  Overall: "
        f"{sum(overall_values) / len(overall_values):.1f}/100"
    )

    total_words = sum(
        r["input"]["word_count"]
        for r in results
    )

    total_grammar_errors = sum(
        r["parameters"]["grammar"]["errors"]
        for r in results
    )

    total_fillers = sum(
        r["parameters"]["filler_words"]["count"]
        for r in results
    )

    total_duration = sum(
        r["input"]["duration_seconds"]
        for r in results
    )

    print()

    print(
        f"Total words: "
        f"{total_words}"
    )

    print(
        f"Total duration: "
        f"{total_duration:.1f}s"
    )

    print(
        f"Total grammar errors: "
        f"{total_grammar_errors}"
    )

    print(
        f"Total fillers: "
        f"{total_fillers}"
    )

    print()

    # ========================================================================
    # SAVE JSON
    # ========================================================================

    output_path = (
        Path(__file__).parent
        / "full_assessment_results.json"
    )

    output = {

        "test_metadata": {

            "test_name":
                "Full Assessment Backend Test",

            "transcript_count":
                len(results),

            "pronunciation_available":
                bool(pronunciation_values),

            "notes": [
                "Transcript-only backend validation.",
                "Duration is supplied by test fixtures.",
                "Pronunciation requires real audio.",
                "Existing production scoring functions are reused.",
                "No frontend code is involved.",
            ],
        },

        "summary": {

            "average_scores": {
                **{
                    key: round(
                        value,
                        2,
                    )
                    for key, value
                    in averages.items()
                },

                "pronunciation": (
                    round(
                        sum(pronunciation_values)
                        / len(pronunciation_values),
                        2,
                    )
                    if pronunciation_values
                    else None
                ),

                "overall": round(
                    sum(overall_values)
                    / len(overall_values),
                    2,
                ),
            },

            "total_words": total_words,

            "total_duration_seconds":
                round(
                    total_duration,
                    2,
                ),

            "total_grammar_errors":
                total_grammar_errors,

            "total_fillers":
                total_fillers,
        },

        "assessments": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print()

    print("=" * 80)
    print("✅ Backend assessment test completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()

