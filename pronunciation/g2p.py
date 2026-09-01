"""
g2p.py
------
Grapheme-to-phoneme conversion: English text -> a flat list of IPA phone
segments, suitable for comparison against acoustic-model phone inventories.

Pipeline: text -> eng_to_ipa (dictionary-based G2P) -> ligature normalization
-> panphon segmentation.

VERIFIED BUG (reproduced directly in this environment, panphon 0.22.2,
eng_to_ipa 0.0.2):

    eng_to_ipa represents the affricates /dʒ/ and /tʃ/ as single ligature
    codepoints ʤ (U+02A4) and ʧ (U+02A7):

        judge -> "ʤəʤ"
        chair -> "ʧɛr"
        watch -> "wɔʧ"

    panphon's FeatureTable.ipa_segs() does not recognize these ligatures
    as valid segments and SILENTLY DROPS them (does not raise/warn):

        ipa_segs("ʤəʤ") -> ['ə']            # both ʤ silently dropped
        ipa_segs("ʧɛr") -> ['ɛ', 'r']        # ʧ silently dropped
        ipa_segs("wɔʧ") -> ['w', 'ɔ']        # ʧ silently dropped

    panphon's segs_safe() does NOT drop them, but also does not decompose
    them into two segments -- it just keeps the ligature as one opaque
    segment, which still won't match Allosaurus's inventory (which has
    'dʒ' and 'tʃ' as its multi-character units, not the ligature glyphs).

    FIX: normalize ʤ -> dʒ and ʧ -> tʃ in the raw IPA string BEFORE
    segmentation. This is applied unconditionally in `g2p_word` below.

        normalize("ʤəʤ") -> "dʒədʒ" -> segs_safe -> ['d','ʒ','ə','d','ʒ']
        normalize("ʧɛr") -> "tʃɛr"  -> segs_safe -> ['t','ʃ','ɛ','r']
        normalize("wɔʧ") -> "wɔtʃ" -> segs_safe -> ['w','ɔ','t','ʃ']

No other remapping (e.g. r -> ɹ) is applied here. That claim was NOT
verified against the Allosaurus uni2005 inventory (which contains both
r and ɹ as distinct symbols) and must not be assumed elsewhere either.

SECOND BUG FOUND DURING ACTUAL TESTING (not in the original handoff,
discovered by running this module end-to-end via evaluation.py):

    eng_to_ipa includes the IPA primary-stress mark ˈ (U+02C8) in its
    output, e.g. "apples" -> "ˈæpəlz". panphon's segs_safe() does not
    drop unrecognized single characters the way ipa_segs() drops
    ligatures -- it passes ˈ through as its own "segment":

        segs_safe("ˈæpəlz") -> ['ˈ', 'æ', 'p', 'ə', 'l', 'z']

    ˈ is not a phone and has no entry in Allosaurus's (or any acoustic
    model's) phone inventory, so leaving it in expected_phonemes would
    make every downstream evidence lookup for it report
    "not in this model's phone inventory". FIX: after segmentation, drop
    any segment that panphon's FeatureTable.seg_known() reports as not a
    recognized phone (this catches ˈ, ˌ, and similar suprasegmental marks
    without needing to hand-maintain a strip list).
"""

from typing import List
import re

import eng_to_ipa as ipa
import panphon

_ft = panphon.FeatureTable()

# The only normalization we have direct, reproduced evidence for.
_LIGATURE_MAP = {
    "ʤ": "dʒ",
    "ʧ": "tʃ",
}


def normalize_ligatures(ipa_str: str) -> str:
    """Replace affricate ligature codepoints with their two-character
    decomposed form, so panphon segmentation does not silently drop them."""
    for lig, decomposed in _LIGATURE_MAP.items():
        ipa_str = ipa_str.replace(lig, decomposed)
    return ipa_str


def g2p_word(word: str) -> List[str]:
    """Convert a single English word to a list of IPA phone segments."""
    raw = ipa.convert(word)
    if not raw or raw.startswith(word):
        # eng_to_ipa returns the original word (sometimes wrapped) if it's
        # not in its dictionary -- treat as an explicit failure rather than
        # silently feeding garbage into panphon.
        return []
    normalized = normalize_ligatures(raw)
    segs = _ft.segs_safe(normalized)
    # Drop suprasegmentals (stress marks etc.) that panphon does not
    # recognize as phones -- see module docstring, "SECOND BUG FOUND".
    segs = [s for s in segs if _ft.seg_known(s)]
    return segs


def g2p_sentence(text: str) -> List[str]:
    """Convert a sentence to a flat list of IPA phone segments across all
    words (word boundaries are not preserved in the output -- see
    g2p_sentence_with_words if word alignment is needed)."""
    phones: List[str] = []
    for word in _tokenize(text):
        phones.extend(g2p_word(word))
    return phones


def g2p_sentence_with_words(text: str):
    """Like g2p_sentence, but returns [(word, [phones]), ...] so callers
    can reason about which phones belong to which word."""
    out = []
    for word in _tokenize(text):
        out.append((word, g2p_word(word)))
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


if __name__ == "__main__":
    # Quick sanity check -- run with: python3 -m pronunciation.g2p
    tests = ["judge", "chair", "watch", "think", "this", "right", "vine", "she"]
    for w in tests:
        raw = ipa.convert(w)
        fixed = g2p_word(w)
        print(f"{w:8s} raw_ipa={raw!r:15s} segs_safe(normalized)={fixed}")
