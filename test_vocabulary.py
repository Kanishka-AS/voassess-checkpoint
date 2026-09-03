"""
Vocabulary scoring — regression / behavior-documentation tests.

These tests pin down the CURRENT (Phase 2, vocabulary.py) behavior as a
baseline. They are not a claim that every behavior here is methodologically
ideal — see the audit notes above each class for what is intentionally
still open (flagged for future work, not changed in this pass).

Run: pytest test_vocabulary.py -v
"""

import pytest
from vocabulary import (
    extract_vocabulary_features,
    score_vocabulary,
    score_vocabulary_from_features,
    is_function_word,
    word_sophistication,
    _is_likely_proper_noun_or_gibberish,
    _tokenize,
    mattr,
)


# ── Short-response behavior ────────────────────────────────────────────────
class TestShortResponses:
    """Confidence should scale down toward the neutral baseline (50) as
    content-word evidence shrinks, and a response should never crash or
    produce an out-of-range score regardless of length."""

    @pytest.mark.parametrize("text", [
        "Yes.",                                              # ~1 word
        "I like it a lot.",                                  # ~5 words
        "I think school is important for everyone today.",   # ~10 words
        "My favorite hobby is reading because it helps me relax and learn new things every day.",  # ~20 words
    ])
    def test_short_responses_stay_in_range_and_lean_neutral(self, text):
        result = score_vocabulary(text)
        assert 0.0 <= result["score"] <= 100.0
        # Short samples should have visibly reduced confidence.
        assert result["confidence"] < 1.0

    def test_empty_transcript_scores_zero_not_neutral(self):
        result = score_vocabulary("")
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0

    def test_confidence_increases_with_content_word_count(self):
        short = score_vocabulary("I like dogs.")
        longer = score_vocabulary(
            "I genuinely enjoy exploring unfamiliar neighborhoods, sampling regional "
            "cuisine, and photographing interesting architecture whenever I travel."
        )
        assert longer["confidence"] > short["confidence"]


# ── Repetition ──────────────────────────────────────────────────────────
class TestRepetition:
    """Current implementation: repetition is a capped multiplicative
    discount (max 25%) applied uniformly to any repeated content word,
    regardless of *why* it repeats. This class documents that current
    behavior; distinguishing topical vs. disfluent repetition (audit
    issue D) is NOT implemented and is flagged, not silently assumed."""

    def test_natural_topic_repetition_is_penalized_same_as_any_repetition(self):
        # "school" repeated because the whole answer is legitimately about school.
        topical = ("School is important. I go to school every day because school "
                   "helps me learn new things and meet my friends at school.")
        features = extract_vocabulary_features(topical)
        assert features["repetition_ratio"] > 0

    def test_no_repetition_when_all_content_words_distinct(self):
        text = "The teacher explained fractions clearly using colorful diagrams yesterday."
        features = extract_vocabulary_features(text)
        assert features["repetition_ratio"] == 0.0

    def test_repetition_penalty_is_capped(self):
        # Extreme repetition should not drive the multiplicative discount past the cap.
        text = "dog dog dog dog dog dog dog dog dog dog"
        result = score_vocabulary(text)
        assert result["repetition_penalty"] <= 25.0


# ── Sophistication / proper-noun & gibberish filtering ─────────────────
class TestSophistication:
    def test_person_name_not_flagged_as_proper_noun_or_advanced(self):
        assert _is_likely_proper_noun_or_gibberish("kanish") is True
        features = extract_vocabulary_features("My name is Kanish and I am a student.")
        assert features["advanced_ratio"] == 0.0

    def test_place_name_excluded(self):
        assert _is_likely_proper_noun_or_gibberish("chennai") is True

    def test_brand_excluded(self):
        assert _is_likely_proper_noun_or_gibberish("google") is True

    def test_acronym_excluded_via_known_list(self):
        assert _is_likely_proper_noun_or_gibberish("nasa") is True

    def test_unknown_allcaps_acronym_excluded_via_case_heuristic(self):
        # Not in the known list, but ALL-CAPS 3+ letters → treated as acronym.
        assert _is_likely_proper_noun_or_gibberish("xyz", original_word="XYZ") is True

    def test_gibberish_repeated_char_excluded(self):
        assert _is_likely_proper_noun_or_gibberish("asdkjqwezxxxxxx") is True

    def test_legitimate_rare_word_counts_as_advanced(self):
        features = extract_vocabulary_features(
            "The industrialization of society transformed economic structures."
        )
        assert features["advanced_ratio"] > 0.0

    def test_technical_term_not_automatically_excluded(self):
        # Audit issue: technical vocabulary should NOT be blanket-excluded just for
        # being technical — only known-proper-noun/acronym/gibberish patterns are.
        assert _is_likely_proper_noun_or_gibberish("photosynthesis") is False
        z = word_sophistication("photosynthesis")
        assert z > 0.0

    def test_unknown_word_zipf_zero_not_sophisticated(self):
        assert word_sophistication("zzzqqqxxxnotaword") == 0.0


# ── Morphology (documents current surface-form behavior; lemmatization
#    is NOT implemented — flagged, not silently claimed) ────────────────
class TestMorphology:
    def test_inflected_forms_counted_as_distinct_surface_words(self):
        text = "I run every morning. Yesterday I ran, and my sister runs too, while running beside me."
        features = extract_vocabulary_features(text)
        tokens = _tokenize(text)
        for form in ("run", "ran", "runs", "running"):
            assert form in tokens
        # Current implementation has no lemma-collapsing step — each surface
        # form is counted separately toward unique_words.
        assert features["unique_words"] >= 4


# ── Tokenization edge cases ─────────────────────────────────────────────
class TestTokenization:
    def test_contractions_kept_as_single_token(self):
        tokens = _tokenize("I don't think that's right.")
        assert "don't" in tokens
        assert "that's" in tokens

    def test_hyphenated_compound_is_currently_split(self):
        # Audit issue #9: hyphens are not in the tokenizer regex, so
        # "well-known" becomes two separate tokens. Documented here as a
        # known, not-yet-fixed gap rather than assumed-correct behavior.
        tokens = _tokenize("It is a well-known fact.")
        assert "well-known" not in tokens
        assert "well" in tokens and "known" in tokens

    def test_numbers_are_currently_ignored(self):
        # Audit issue #10: the tokenizer regex is letters/apostrophes only,
        # so digits contribute nothing to word counts either way.
        tokens = _tokenize("I have 42 apples and 3 oranges.")
        assert all(not any(ch.isdigit() for ch in t) for t in tokens)

    def test_punctuation_does_not_leak_into_tokens(self):
        tokens = _tokenize("Well, actually... I think, yes!")
        assert all(t.isalpha() or "'" in t for t in tokens)


# ── Content vs. function words ──────────────────────────────────────────
class TestContentFunctionWords:
    def test_ordinary_function_words_excluded_from_content(self):
        for w in ("the", "a", "of", "and", "is", "to"):
            assert is_function_word(w) is True

    def test_discourse_markers_treated_as_function_words(self):
        # Current FUNCTION_WORDS list folds in a handful of filler/discourse
        # tokens (um, uh, well, okay, yeah). "actually", "basically", and
        # "really" are NOT in that list today — documented, not assumed.
        for w in ("um", "uh", "well", "okay", "yeah"):
            assert is_function_word(w) is True
        for w in ("actually", "basically", "really", "like"):
            assert is_function_word(w) is False

    def test_ordinary_content_word_not_excluded(self):
        assert is_function_word("elephant") is False


# ── MATTR sanity ─────────────────────────────────────────────────────────
class TestMATTR:
    def test_mattr_of_all_distinct_words_is_one(self):
        tokens = ["one", "two", "three", "four", "five"]
        assert mattr(tokens, window=5) == 1.0

    def test_mattr_falls_back_to_ttr_when_shorter_than_window(self):
        tokens = ["a", "a", "b"]
        assert mattr(tokens, window=20) == pytest.approx(2 / 3)

    def test_mattr_empty_is_zero(self):
        assert mattr([], window=20) == 0.0


# ── End-to-end score sanity across the length spectrum ──────────────────
class TestScoreSanity:
    def test_score_is_deterministic(self):
        text = "The committee deliberated extensively before reaching a consensus."
        assert score_vocabulary(text) == score_vocabulary(text)

    def test_richer_vocabulary_scores_at_least_as_high_as_repetitive_simple_text(self):
        simple = "I like it. It is good. I like it a lot. It is very good."
        rich = ("The intricate architecture of the ancient cathedral fascinated "
                "every visitor who wandered through its labyrinthine corridors.")
        assert score_vocabulary(rich)["score"] >= score_vocabulary(simple)["score"]
