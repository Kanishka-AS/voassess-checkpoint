# tests/test_vocabulary.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from vocabulary import (
    mattr,
    extract_vocabulary_features,
    score_vocabulary,
    score_vocabulary_from_features,
    is_function_word,
    word_sophistication,
    _is_likely_proper_noun_or_gibberish,
    _tokenize_with_case,
)


# ── Proper noun / gibberish detection ──────────────────────────────────
def test_proper_noun_detection():
    # Person names should be detected as proper nouns
    assert _is_likely_proper_noun_or_gibberish("kanish", "Kanish")
    assert _is_likely_proper_noun_or_gibberish("john", "John")
    assert _is_likely_proper_noun_or_gibberish("sarah", "Sarah")
    
    # Place names should be detected as proper nouns
    assert _is_likely_proper_noun_or_gibberish("chennai", "Chennai")
    assert _is_likely_proper_noun_or_gibberish("london", "London")
    assert _is_likely_proper_noun_or_gibberish("mumbai", "Mumbai")
    
    # Brands should be detected as proper nouns
    assert _is_likely_proper_noun_or_gibberish("google", "Google")
    assert _is_likely_proper_noun_or_gibberish("microsoft", "Microsoft")
    
    # Common words should NOT be detected as proper nouns
    assert not _is_likely_proper_noun_or_gibberish("student", "student")
    assert not _is_likely_proper_noun_or_gibberish("technology", "technology")
    # "industrialization" must NOT be filtered
    assert not _is_likely_proper_noun_or_gibberish("industrialization", "industrialization")
    assert not _is_likely_proper_noun_or_gibberish("beneficial", "beneficial")
    assert not _is_likely_proper_noun_or_gibberish("perspicacious", "perspicacious")
    
    # Gibberish should be detected
    assert _is_likely_proper_noun_or_gibberish("asdkjqwezxxxxxx", "asdkjqwezxxxxxx")
    assert _is_likely_proper_noun_or_gibberish("aaaaaa", "aaaaaa")


def test_sophistication_with_proper_nouns():
    """Proper nouns should not get sophistication credit."""
    from vocabulary import word_sophistication_with_proper_noun_filter
    
    # These should get 0.0 because they're proper nouns
    assert word_sophistication_with_proper_noun_filter("kanish", "Kanish") == 0.0
    assert word_sophistication_with_proper_noun_filter("chennai", "Chennai") == 0.0
    assert word_sophistication_with_proper_noun_filter("google", "Google") == 0.0
    assert word_sophistication_with_proper_noun_filter("nasa", "NASA") == 0.0
    
    # These should get normal sophistication scores (> 0)
    student_score = word_sophistication_with_proper_noun_filter("student", "student")
    beneficial_score = word_sophistication_with_proper_noun_filter("beneficial", "beneficial")
    industrialization_score = word_sophistication_with_proper_noun_filter("industrialization", "industrialization")
    perspicacious_score = word_sophistication_with_proper_noun_filter("perspicacious", "perspicacious")
    
    assert student_score > 0
    assert beneficial_score > 0
    assert industrialization_score > 0
    assert perspicacious_score > 0
    
    # industrialization should have a higher sophistication than student
    assert industrialization_score > student_score


def test_advanced_ratio_with_proper_nouns():
    """Proper nouns should not count towards advanced_ratio."""
    # Proper noun should not count as advanced
    result = score_vocabulary("My name is Kanish. I am a student.")
    # Content words: kanish(proper noun), student(common)
    # advanced_ratio = 0/2 = 0.0%
    assert result["advanced_ratio"] == 0.0
    
    # Place name should not count as advanced
    result = score_vocabulary("I visited Chennai last week.")
    # Content words: visited(common), chennai(proper noun), week(common)
    # advanced_ratio = 0/3 = 0.0%
    assert result["advanced_ratio"] == 0.0


def test_advanced_ratio_with_legitimate_rare_words():
    """Legitimate rare English words should count as advanced."""
    result = score_vocabulary("The industrialization of society transformed economic structures.")
    # Content words: industrialization(advanced), society(common), transformed(common),
    #                economic(common), structures(common)
    # advanced_ratio = 1/5 = 20.0%
    assert result["advanced_ratio"] == 20.0


def test_gibberish_not_counted_as_advanced():
    """Gibberish words should not count as advanced."""
    result = score_vocabulary("I used asdkjqwezxxxxxx technology.")
    # Content words: used(common), asdkjqwezxxxxxx(gibberish), technology(common)
    # advanced_ratio = 0/3 = 0.0%
    assert result["advanced_ratio"] == 0.0


def test_comparison_with_meaningful_words():
    """Sentences with legitimate rare words should score higher."""
    proper_result = score_vocabulary("My name is Kanish.")
    rare_result = score_vocabulary("The industrialization is remarkable.")
    
    # The rare word sentence should have higher sophistication
    assert proper_result["sophistication"] < rare_result["sophistication"]


def test_sentence_initial_capitalization():
    """Sentence-initial capitalization should not affect scoring."""
    result = score_vocabulary("The weather today is nice.")
    assert result["score"] > 0
    
    result = score_vocabulary("This is a good example.")
    assert result["score"] > 0


def test_mixed_case_words():
    """Words with mixed case should be handled properly."""
    # "NASA" is in the known proper nouns list
    result = score_vocabulary("NASA launched a rocket.")
    # "NASA" is a proper noun → excluded
    # "launched" and "rocket" are common
    # advanced_ratio = 0/2 = 0.0%
    assert result["advanced_ratio"] == 0.0


def test_unknown_word_fallback():
    """Unknown words should NOT count as advanced."""
    result = score_vocabulary("This is a misspeled word.")
    # Content words: misspeled(unknown), word(common)
    # "misspeled" is unknown → not advanced (z=0)
    # advanced_ratio = 0/2 = 0.0%
    assert result["advanced_ratio"] == 0.0


# ── MATTR ────────────────────────────────────────────────────────────────
def test_mattr_empty():
    assert mattr([], 20) == 0.0


def test_mattr_shorter_than_window_equals_raw_ttr():
    tokens = ["good", "great", "good"]
    assert mattr(tokens, 20) == pytest.approx(2 / 3)


def test_mattr_all_unique_is_one():
    tokens = ["the", "weather", "today", "is", "quite", "unpredictable"]
    assert mattr(tokens, 4) == pytest.approx(1.0)


def test_mattr_lower_for_repetitive_text():
    repetitive = "i like this i like this i like this".split()
    diverse = "the weather today is quite unpredictable and bright".split()
    assert mattr(repetitive, 5) < mattr(diverse, 5)


def test_mattr_window_larger_than_text_no_crash():
    assert mattr(["one", "two"], 999) == 1.0


# ── Function word detection ────────────────────────────────────────────────
def test_function_words_detected():
    for w in ["the", "a", "is", "to", "and", "i", "dont"]:
        assert is_function_word(w)


def test_content_words_not_flagged_as_function():
    for w in ["cricket", "beautiful", "fascinating"]:
        assert not is_function_word(w)


def test_contraction_normalization_matches_function_word():
    assert is_function_word("don't")


# ── wordfreq sophistication mapping ─────────────────────────────────────────
def test_common_word_low_sophistication():
    assert word_sophistication("the") < 0.2
    assert word_sophistication("good") < 0.3


def test_rare_real_word_higher_sophistication_than_common():
    assert word_sophistication("beneficial") > word_sophistication("good")
    assert word_sophistication("significant") > word_sophistication("good")


def test_unknown_word_not_treated_as_maximally_sophisticated_bonus():
    """Unknown words should be treated neutrally (0.0), not as rare."""
    gibberish_score = word_sophistication("asdkjqwezxxxxxx")
    rare_known_score = word_sophistication("perspicacious")
    assert gibberish_score == 0.0
    assert rare_known_score > 0.0
    assert gibberish_score < rare_known_score


# ── Repetition ───────────────────────────────────────────────────────────
def test_excessive_content_word_repetition_penalized():
    repetitive = "I like cricket I like cricket I like cricket I like cricket"
    varied = "I enjoy cricket football tennis swimming running cycling badminton"
    rep = score_vocabulary(repetitive)
    var = score_vocabulary(varied)
    assert rep["repetition_penalty"] > 0
    assert rep["score"] < var["score"]


def test_function_word_repetition_not_penalized():
    features = extract_vocabulary_features(
        "the the the a a a to to to of of of and and and but but but"
    )
    assert features["repetition_ratio"] == 0.0


def test_content_word_repetition_measured_correctly():
    features = extract_vocabulary_features("dog dog dog cat")
    assert features["repetition_ratio"] == pytest.approx(2 / 4)


# ── Response length / evidence dampening ────────────────────────────────────
def test_very_short_response_near_neutral_not_extreme():
    result = score_vocabulary("Good.")
    assert 30 <= result["score"] <= 70
    assert result["confidence"] < 0.3


def test_confidence_increases_with_content_word_count():
    short = score_vocabulary("I like cricket.")
    longer = score_vocabulary(
        "I find this experience fascinating and extremely rewarding "
        "because it challenges my understanding of the subject."
    )
    assert longer["confidence"] > short["confidence"]


def test_text_shorter_than_mattr_window_does_not_crash():
    result = score_vocabulary("Hello world")
    assert result["total_words"] == 2
    assert 0 <= result["score"] <= 100


# ── Sophistication does not blow up on one rare word ──────────────────────
def test_single_rare_word_does_not_dominate_score():
    normal = score_vocabulary(
        "I went to the market and bought some fruit and vegetables for dinner"
    )
    with_one_rare_word = score_vocabulary(
        "I went to the market and bought some fruit and perspicacious vegetables"
    )
    # one rare word inserted shouldn't cause a runaway jump in score
    assert with_one_rare_word["score"] - normal["score"] < 25


# ── Vocabulary/sophistication ordering sanity ──────────────────────────────
def test_more_sophisticated_vocabulary_scores_higher_at_similar_length():
    plain = score_vocabulary("I like this. I really like this very much.")
    sophisticated = score_vocabulary(
        "I find this experience fascinating and extremely rewarding overall."
    )
    assert sophisticated["score"] >= plain["score"]


# ── Normalization: punctuation / capitalization ─────────────────────────────
def test_punctuation_and_capitalization_normalized():
    a = score_vocabulary("I LOVE cricket!!! It's amazing, truly amazing.")
    b = score_vocabulary("i love cricket it's amazing truly amazing")
    assert a["total_words"] == b["total_words"]
    assert a["unique_words"] == b["unique_words"]


# ── Empty / edge cases ─────────────────────────────────────────────────────
def test_empty_transcript():
    result = score_vocabulary("")
    assert result == {
        "score": 0.0, "unique_words": 0, "total_words": 0, "advanced_ratio": 0.0,
        "diversity": 0.0, "sophistication": 0.0, "variety": 0.0,
        "repetition_penalty": 0.0, "confidence": 0.0,
    }


def test_whitespace_only_transcript():
    result = score_vocabulary("   \n\t  ")
    assert result["total_words"] == 0
    assert result["score"] == 0.0


def test_punctuation_only_transcript():
    result = score_vocabulary("... !!! ???")
    assert result["total_words"] == 0


# ── extract/score separation ────────────────────────────────────────────────
def test_extract_then_score_matches_score_vocabulary():
    text = "I really enjoy exploring new ideas and interesting challenges"
    features = extract_vocabulary_features(text)
    scored_separately = score_vocabulary_from_features(features)
    scored_directly = score_vocabulary(text)
    assert scored_separately == scored_directly


# ── Backward-compatible output shape ────────────────────────────────────────
def test_output_has_legacy_keys_frontend_depends_on():
    result = score_vocabulary("I went to Chennai yesterday to watch cricket.")
    for key in ("score", "unique_words", "total_words", "advanced_ratio"):
        assert key in result