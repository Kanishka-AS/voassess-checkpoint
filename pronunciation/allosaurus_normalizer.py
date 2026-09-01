"""
allosaurus_normalizer.py
-------------------------
Model-specific normalization layer between generic G2P output and the
Allosaurus uni2005 phone vocabulary.

WHY THIS EXISTS (see results.json from the first real-audio run):

    1. g2p.py emits "r" for English orthographic 'r'. Allosaurus's uni2005
       phone inventory contains BOTH "r" and "ɹ" as distinct symbols (229
       phones total; verified directly from phone.txt, not assumed -- see
       allosaurus_provider.py's __init__). Allosaurus's greedy decode of
       three real MediaRepeat recordings produced "ɹ" for English 'r' in
       every case (e.g. "for" -> f...ɹ, "hours" -> ...ɹ̩, "every" -> ...ɹ̩,
       "world" -> w...ɹ̩), never plain "r". Searching the posterior column
       for "r" against that audio gives max_posterior ~= 0.02, with the
       model's actual top-1 hypothesis at that frame being <blank> or
       something else. That is a symbol-vocabulary mismatch, not evidence
       of a mispronunciation -- the model was almost certainly right that
       an approximant occurred there, it just doesn't call it "r".

    2. g2p.py (deliberately, see its own docstring) decomposes affricates
       into two segments: /tʃ/ -> ['t','ʃ'], /dʒ/ -> ['d','ʒ']. Allosaurus's
       phone.txt contains "tʃ" and "dʒ" as single atomic multi-character
       phone units (verified directly, not assumed). Greedy decode of
       "change" and "climate...change" produced the atomic units "tʃ" and
       "dʒ" directly, never a "t"-then-"ʃ" or "d"-then-"ʒ" pair. Comparing
       against the decomposed generic phones therefore asks Allosaurus for
       posteriors of symbols that are not even in its output alphabet's
       intended use for this sound.

This module fixes ONLY these two verified mismatches. It does not attempt
to guess other correspondences by symbol similarity (e.g. it does NOT map
"o" -> "oʊ", "e" -> "eɪ", etc, even though such diphthongization is
plausible -- that has not been individually verified against phone.txt +
real decode behavior the way the two rules above have, so it is left
alone rather than "fixed" on a hunch. Extend this module only after doing
the same kind of verification: (a) confirm the target symbol actually
exists in the inventory, (b) confirm the model's own greedy decode
actually prefers that symbol for the sound in question).

ARCHITECTURE (per the requested design):

    text -> G2P -> generic IPA phones -> AllosaurusPhoneNormalizer
         -> Allosaurus phone vocabulary -> acoustic evidence

g2p.py stays 100% generic (no Allosaurus-specific knowledge). All
model-specific vocabulary decisions live here, keyed off the model's
*actual* inventory (passed in, inspected -- never hardcoded as "assume
uni2005 always has X").
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set


# ---------------------------------------------------------------------------
# Explicit, documented mapping rules. Each rule is only applied if its
# target symbol is actually present in the inventory handed to the
# normalizer at construction time -- inventory membership is CHECKED, not
# assumed, every time.
# ---------------------------------------------------------------------------

# Rule A: bare rhotic. English orthographic 'r' as produced by g2p.py's
# dictionary-based G2P is conventionally the alveolar/postalveolar
# approximant, which Allosaurus's inventory represents as "ɹ" rather than
# the trill/tap symbol "r". Only applied if "ɹ" exists in the inventory;
# if it does not, "r" is left as "r" rather than guessing.
RHOTIC_MAP = {"r": "ɹ"}

# Rule B: affricate re-atomization. g2p.py intentionally decomposes /tʃ/
# and /dʒ/ into two segments each (see g2p.py docstring). Allosaurus's
# inventory instead has these as single atomic multi-character phone
# units. Only applied if the atomic symbol exists in the inventory; the
# rule consumes two ADJACENT generic phones and emits one model phone.
AFFRICATE_MERGES = [
    (("t", "ʃ"), "tʃ"),
    (("d", "ʒ"), "dʒ"),
]


@dataclass
class NormalizedPhone:
    """One unit of the normalized (model-vocabulary) expected sequence,
    carrying a pointer back to the generic phone(s) it came from so
    downstream reporting can show both sides of the mapping."""

    generic: List[str]      # the original G2P phone(s) this came from, e.g. ["t","ʃ"] or ["r"]
    model: str              # the phone symbol actually looked up in the acoustic model
    rule: Optional[str] = None   # which rule fired, or None if passed through unchanged
    in_inventory: bool = True    # whether `model` is actually in the model's phone set


class AllosaurusPhoneNormalizer:
    """Maps a generic IPA phone sequence onto a specific Allosaurus model's
    phone vocabulary, using only mapping rules that have been individually
    verified (see module docstring). Nothing here is Allosaurus-version-
    agnostic magic: the inventory is inspected at construction time and
    every rule is gated on the target symbol actually being present.
    """

    def __init__(self, inventory: Iterable[str]):
        # Inspect the actual inventory -- do not assume any symbol exists.
        self.inventory: Set[str] = set(inventory)

        # Resolve which rules are actually usable against this inventory,
        # once, so callers/reports can see exactly what will happen before
        # any phones are processed.
        self.active_rhotic_map = {
            src: dst for src, dst in RHOTIC_MAP.items() if dst in self.inventory
        }
        self.active_affricate_merges = [
            (pair, dst) for pair, dst in AFFRICATE_MERGES if dst in self.inventory
        ]

        self.skipped_rules = []
        for src, dst in RHOTIC_MAP.items():
            if dst not in self.inventory:
                self.skipped_rules.append(
                    f"rhotic map {src!r}->{dst!r} skipped: {dst!r} not in inventory"
                )
        for pair, dst in AFFRICATE_MERGES:
            if dst not in self.inventory:
                self.skipped_rules.append(
                    f"affricate merge {pair!r}->{dst!r} skipped: {dst!r} not in inventory"
                )

    def report(self) -> dict:
        """What this normalizer will actually do against the inventory it
        was built with -- for logging/debugging, not for evidence math."""
        return {
            "inventory_size": len(self.inventory),
            "active_rhotic_map": dict(self.active_rhotic_map),
            "active_affricate_merges": [
                {"generic": list(pair), "model": dst}
                for pair, dst in self.active_affricate_merges
            ],
            "skipped_rules": list(self.skipped_rules),
        }

    def normalize_sequence(self, phones: List[str]) -> List[NormalizedPhone]:
        """Convert a flat generic-IPA phone list into a list of
        NormalizedPhone entries in the model's vocabulary. Affricate merges
        consume two adjacent input phones; everything else is 1-to-1
        (either remapped, e.g. r->ɹ, or passed through unchanged)."""
        out: List[NormalizedPhone] = []
        i = 0
        n = len(phones)
        while i < n:
            merged = False
            for (a, b), dst in self.active_affricate_merges:
                if phones[i] == a and i + 1 < n and phones[i + 1] == b:
                    out.append(
                        NormalizedPhone(
                            generic=[a, b],
                            model=dst,
                            rule=f"affricate_merge({a}+{b}->{dst})",
                            in_inventory=dst in self.inventory,
                        )
                    )
                    i += 2
                    merged = True
                    break
            if merged:
                continue

            phone = phones[i]
            if phone in self.active_rhotic_map:
                dst = self.active_rhotic_map[phone]
                out.append(
                    NormalizedPhone(
                        generic=[phone],
                        model=dst,
                        rule=f"rhotic_map({phone}->{dst})",
                        in_inventory=dst in self.inventory,
                    )
                )
            else:
                out.append(
                    NormalizedPhone(
                        generic=[phone],
                        model=phone,
                        rule=None,
                        in_inventory=phone in self.inventory,
                    )
                )
            i += 1
        return out

    def normalize_single(self, phone: str) -> NormalizedPhone:
        """Normalize one already-atomic phone symbol (used by the manual
        evaluation mode, where the caller supplies the exact expected
        phone directly -- e.g. 'tʃ' as one token, not 't','ʃ'). Applies the
        rhotic map only; affricate merging does not apply to a single
        already-atomic token by definition."""
        if phone in self.active_rhotic_map:
            dst = self.active_rhotic_map[phone]
            return NormalizedPhone(
                generic=[phone], model=dst, rule=f"rhotic_map({phone}->{dst})",
                in_inventory=dst in self.inventory,
            )
        return NormalizedPhone(
            generic=[phone], model=phone, rule=None, in_inventory=phone in self.inventory
        )
