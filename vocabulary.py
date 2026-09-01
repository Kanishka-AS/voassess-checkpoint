"""
VOCAssess — Vocabulary Assessment (Phase 2)
=============================================

Replaces the old "raw TTR + hardcoded 250-word list + length-penalty fudge
factor" vocabulary proxy with a small, linguistically-motivated pipeline:

    extract_vocabulary_features(text)   →  raw measurements
    score_vocabulary_from_features(...) →  0-100 score + breakdown
    score_vocabulary(text)              →  the two combined (drop-in
                                             replacement for the old
                                             app.py:score_vocabulary)

Four dimensions are measured, each already length-robust on its own:

  1. Lexical diversity     — MATTR (Moving-Average TTR) over ALL words.
  2. Lexical sophistication — mean wordfreq Zipf frequency of unique
                               content words, mapped onto a capped 0-100
                               scale (rarer → higher, but capped so one
                               freak rare word can't dominate).
  3. Vocabulary variety     — MATTR restricted to content words only,
                               i.e. how varied the *meaningful* vocabulary
                               is, independent of function-word padding.
  4. Repetition             — fraction of content-word tokens that are
                               repeats of an already-used content word;
                               applied as a capped multiplicative discount,
                               not a subtracted "fudge factor".

A response's overall word count then gates how much these measured
dimensions are trusted (`evidence_factor`): short answers are pulled
toward a neutral baseline instead of hitting 0 or 100 on noise.

This is NOT a CEFR classifier and makes no claim to be one — it is a
lexical-sophistication/diversity signal only. See DESIGN.md.
"""

from __future__ import annotations

import re
from collections import Counter

from wordfreq import zipf_frequency

# ── Word normalization ────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(transcript: str) -> list[str]:
    return _WORD_RE.findall(transcript.lower())


def _tokenize_with_case(transcript: str) -> list[tuple[str, str]]:
    """Return (lowercase, original_case) pairs for each word."""
    return [(m.group(0).lower(), m.group(0)) for m in _WORD_RE.finditer(transcript)]


# ── Function words ────────────────────────────────────────────────────────
FUNCTION_WORDS = {
    "a","an","the","and","but","or","nor","so","yet","for","if","because","although",
    "though","while","as","than","that","which","who","whom","whose","whichever",
    "i","me","my","mine","myself","you","your","yours","yourself","yourselves",
    "he","him","his","himself","she","her","hers","herself","it","its","itself",
    "we","us","our","ours","ourselves","they","them","their","theirs","themselves",
    "this","these","those","there","here",
    "am","is","are","was","were","be","been","being",
    "have","has","had","having",
    "do","does","did","doing","done",
    "will","would","shall","should","can","could","may","might","must","ought",
    "to","of","in","on","at","by","with","from","into","onto","over","under",
    "about","against","between","among","through","during","before","after",
    "above","below","up","down","out","off","again","further","once",
    "not","no","nor","only","own","same","too","very","just","also",
    "what","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","most",
    "im","ive","id","ill","youre","youve","youd","youll","hes","shes","its",
    "were","theyre","theyve","theyd","theyll","weve","wed","well",
    "dont","doesnt","didnt","isnt","arent","wasnt","werent","havent","hasnt",
    "hadnt","wont","wouldnt","cant","couldnt","shouldnt","mightnt","mustnt",
    "um","uh","er","ah","okay","ok","yeah","yes","well","right","gonna","wanna","gotta",
}


def _strip_apostrophe(word: str) -> str:
    return word.replace("'", "")


def is_function_word(word: str) -> bool:
    w = _strip_apostrophe(word)
    return word in FUNCTION_WORDS or w in FUNCTION_WORDS


# ── Proper noun / gibberish detection ──────────────────────────────────
# A compact list of common proper nouns and acronyms that wordfreq knows
# but should be filtered from vocabulary scoring.
_KNOWN_PROPER_NOUNS = {
    # Person names (common English + Indian)
    "john", "jane", "mike", "sarah", "david", "mary", "james", "robert",
    "michael", "william", "joseph", "thomas", "charles", "christopher",
    "daniel", "matthew", "anthony", "donald", "mark", "paul", "steven",
    "andrew", "kenneth", "joshua", "kevin", "brian", "george", "timothy",
    "ronald", "edward", "jason", "jeffrey", "ryan", "jacob", "gary",
    "nicholas", "eric", "jonathan", "stephen", "larry", "justin", "scott",
    "brandon", "benjamin", "samuel", "gregory", "alexander", "patrick",
    "frank", "raymond", "jack", "dennis", "jerry", "tyler", "aaron",
    "jose", "adam", "henry", "nathan", "carl", "bryan", "noah", "logan",
    "caleb", "isaac", "luke", "susan", "jessica", "jennifer", "amanda",
    "ashley", "stephanie", "melissa", "nicole", "elizabeth", "heather",
    "tiffany", "michelle", "amber", "megan", "rachel", "kimberly",
    "christina", "lauren", "katherine", "andrea", "shannon", "angela",
    "samantha", "emily", "alexis", "kathryn", "maria", "brittany",
    "vanessa", "rebecca", "laura", "danielle", "jasmine", "morgan",
    "kristen", "kelly", "molly", "katie", "anna",
    "kanish", "anand", "arjun", "krish", "rahul", "priya", "neha",
    "sanjay", "raj", "vikram", "amit", "nisha", "deepak", "suresh",
    "gopal", "ram", "shyam", "sita", "gita", "radha", "mohan",
    # Place names
    "chennai", "mumbai", "delhi", "kolkata", "bangalore", "hyderabad",
    "london", "paris", "tokyo", "beijing", "moscow", "sydney", "toronto",
    "chicago", "los angeles", "san francisco",
    # Brands
    "google", "microsoft", "apple", "facebook", "amazon", "netflix",
    "tesla", "spacex", "uber", "lyft", "airbnb", "spotify", "zoom",
    "slack", "github", "linkedin", "twitter", "instagram", "tiktok",
    # Acronyms (all caps)
    "nasa", "fbi", "cia", "nsa", "who", "unicef", "unesco", "nato",
}

# Gibberish detection patterns - ONLY clear gibberish, not legitimate words
_GIBBERISH_PATTERNS = [
    # Same character repeated 4+ times in a row
    re.compile(r'^(.)\1{3,}$'),
    # Same character repeated 3+ times with optional prefix/suffix
    re.compile(r'^[a-z]*(.)\1{3,}[a-z]*$'),
    # No vowels at all (only consonants) - this catches strings like "xyzxyzxyz"
    re.compile(r'^[bcdfghjklmnpqrstvwxyz]{8,}$'),
]


def _is_likely_proper_noun_or_gibberish(word: str, original_word: str = None) -> bool:
    """
    Determine if a word is likely a proper noun or gibberish.
    
    Uses only:
    1. Known proper noun list (case-insensitive)
    2. Clear gibberish patterns
    3. All-caps acronyms (3+ uppercase letters)
    
    Does NOT use capitalization to classify words. This ensures legitimate
    content words like "industrialization" are never misclassified.
    """
    if original_word is None:
        original_word = word
    
    # Check for gibberish patterns
    for pattern in _GIBBERISH_PATTERNS:
        if pattern.match(word):
            return True
    
    # Check against known proper nouns (case-insensitive)
    if word in _KNOWN_PROPER_NOUNS:
        return True
    
    # All-caps acronym with 3+ letters (e.g., NASA, FBI)
    if original_word.isupper() and len(original_word) >= 3:
        return True
    
    return False


def _get_original_case(word: str, tokens_with_case: list) -> str:
    """Get the original case of a word from the token list."""
    for w, ow in tokens_with_case:
        if w == word:
            return ow
    return word


# ── MATTR ────────────────────────────────────────────────────────────────
MATTR_WINDOW = 20
CONTENT_MATTR_WINDOW = 12


def mattr(tokens: list[str], window: int) -> float:
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    ratios = []
    for i in range(n - window + 1):
        w = tokens[i:i + window]
        ratios.append(len(set(w)) / window)
    return sum(ratios) / len(ratios)


# ── Lexical sophistication (wordfreq) ──────────────────────────────────
SOPH_ZIPF_FLOOR = 6.0
SOPH_ZIPF_CEILING = 2.0

# Threshold for "advanced" vocabulary: words with Zipf frequency below this
# value are considered advanced. Words with Zipf >= this value are considered
# common/ordinary.
# 
# Zipf scale reference:
#   6.0+   : Very common (the, be, to, of, and)
#   5.0-6.0: Common everyday words (good, time, student)
#   4.0-5.0: Moderately common (transformed, visited, rocket)
#   3.0-4.0: Less common but legitimate (industrialization, beneficial)
#   2.0-3.0: Rare (perspicacious, ubiquitous)
#   0.0    : Unknown / not in corpus
#
# The threshold of 4.0 means:
#   - Words with zipf < 4.0 are counted as "advanced"
#   - Words with zipf >= 4.0 are considered "common"
#   - Words with zipf == 0.0 are unknown and NOT counted as advanced
ADVANCED_ZIPF_THRESHOLD = 4.0


def word_sophistication(word: str) -> float:
    """
    0.0 (very common) .. 1.0 (rare/sophisticated), capped both ends.
    
    Unknown words (zipf == 0) return 0.0 — they are not sophisticated.
    """
    z = zipf_frequency(word, "en")
    
    # Unknown words are NOT sophisticated
    if z <= 0.0:
        return 0.0
    
    z = max(SOPH_ZIPF_CEILING, min(SOPH_ZIPF_FLOOR, z))
    return (SOPH_ZIPF_FLOOR - z) / (SOPH_ZIPF_FLOOR - SOPH_ZIPF_CEILING)


def word_sophistication_with_proper_noun_filter(word: str, original_word: str) -> float:
    """Same as word_sophistication, but filters out proper nouns and gibberish."""
    if _is_likely_proper_noun_or_gibberish(word, original_word):
        return 0.0
    return word_sophistication(word)


# ── Repetition ──────────────────────────────────────────────────────────
REPETITION_RATIO_AT_FULL_PENALTY = 0.35
REPETITION_MAX_PENALTY_FRACTION = 0.25

# ── Evidence / short-response dampening ──────────────────────────────────
EVIDENCE_TARGET_CONTENT_WORDS = 12
NEUTRAL_BASELINE_SCORE = 50.0

# ── Composite weights ──────────────────────────────────────────────────
WEIGHT_DIVERSITY = 0.35
WEIGHT_SOPHISTICATION = 0.40
WEIGHT_VARIETY = 0.25


def extract_vocabulary_features(transcript: str) -> dict:
    """Pure measurement step — no scoring decisions here."""
    tokens_with_case = _tokenize_with_case(transcript)
    tokens = [w for w, _ in tokens_with_case]
    total_words = len(tokens)

    if total_words == 0:
        return {
            "total_words": 0, "unique_words": 0,
            "content_words": [], "unique_content_words": 0,
            "diversity_mattr": 0.0, "variety_mattr": 0.0,
            "sophistication_raw": 0.0, "repetition_ratio": 0.0,
            "advanced_ratio": 0.0,
        }

    unique_words = set(tokens)
    content_words = [w for w in tokens if not is_function_word(w)]
    unique_content_words = sorted(set(content_words))

    diversity_mattr = mattr(tokens, MATTR_WINDOW)
    variety_mattr = mattr(content_words, CONTENT_MATTR_WINDOW) if content_words else 0.0

    if unique_content_words:
        soph_values = []
        advanced_count = 0
        for w in unique_content_words:
            # Find the original case for this word
            orig_w = _get_original_case(w, tokens_with_case)
            
            # Get sophistication, filtering out proper nouns/gibberish
            soph = word_sophistication_with_proper_noun_filter(w, orig_w)
            soph_values.append(soph)
            
            # Only count as advanced if:
            # 1. It's NOT a proper noun or gibberish
            # 2. It's a KNOWN word (zipf > 0) with zipf < threshold
            if not _is_likely_proper_noun_or_gibberish(w, orig_w):
                z = zipf_frequency(w, "en")
                # CRITICAL: z == 0 means unknown → NOT advanced
                if z > 0.0 and z < ADVANCED_ZIPF_THRESHOLD:
                    advanced_count += 1
        
        sophistication_raw = sum(soph_values) / len(soph_values) if soph_values else 0.0
        advanced_ratio = advanced_count / len(unique_content_words) if unique_content_words else 0.0
    else:
        sophistication_raw = 0.0
        advanced_ratio = 0.0

    if content_words:
        counts = Counter(content_words)
        excess = sum(c - 1 for c in counts.values() if c > 1)
        repetition_ratio = excess / len(content_words)
    else:
        repetition_ratio = 0.0

    return {
        "total_words": total_words,
        "unique_words": len(unique_words),
        "content_words": content_words,
        "unique_content_words": len(unique_content_words),
        "diversity_mattr": diversity_mattr,
        "variety_mattr": variety_mattr,
        "sophistication_raw": sophistication_raw,
        "repetition_ratio": repetition_ratio,
        "advanced_ratio": advanced_ratio,
    }


def score_vocabulary_from_features(features: dict) -> dict:
    """Turn raw measurements into an interpretable 0-100 score."""
    total_words = features["total_words"]
    if total_words == 0:
        return {
            "score": 0.0,
            "unique_words": 0, "total_words": 0,
            "advanced_ratio": 0.0,
            "diversity": 0.0, "sophistication": 0.0, "variety": 0.0,
            "repetition_penalty": 0.0, "confidence": 0.0,
        }

    diversity = features["diversity_mattr"] * 100
    sophistication = features["sophistication_raw"] * 100
    variety = features["variety_mattr"] * 100

    measured = (diversity * WEIGHT_DIVERSITY +
                sophistication * WEIGHT_SOPHISTICATION +
                variety * WEIGHT_VARIETY)

    repetition_penalty_fraction = min(
        features["repetition_ratio"] / REPETITION_RATIO_AT_FULL_PENALTY, 1.0
    ) * REPETITION_MAX_PENALTY_FRACTION
    measured *= (1.0 - repetition_penalty_fraction)

    content_word_count = len(features["content_words"])
    confidence = min(content_word_count / EVIDENCE_TARGET_CONTENT_WORDS, 1.0)
    final_score = confidence * measured + (1 - confidence) * NEUTRAL_BASELINE_SCORE

    return {
        "score": round(final_score, 1),
        "unique_words": features["unique_words"],
        "total_words": total_words,
        "advanced_ratio": round(features["advanced_ratio"] * 100, 1),
        "diversity": round(diversity, 1),
        "sophistication": round(sophistication, 1),
        "variety": round(variety, 1),
        "repetition_penalty": round(repetition_penalty_fraction * 100, 1),
        "confidence": round(confidence, 2),
    }


def score_vocabulary(transcript: str) -> dict:
    """Drop-in replacement for the old app.py:score_vocabulary."""
    features = extract_vocabulary_features(transcript)
    return score_vocabulary_from_features(features)