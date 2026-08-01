"""Grounding that checks what a claim asserts, not just which words it uses.

Bag-of-words overlap answers "did this sentence reuse the source's vocabulary".
That is not the question. A model generating from retrieved context always
reuses the vocabulary in front of it, which is exactly why the dangerous
hallucination scores well:

    source:   "Refunds are processed within 10 business days of the return being received."
    response: "Refunds are not processed within 10 business days of the return being received."
    overlap:  0.90, reported as grounded

The fix is not a better similarity metric. It is checking the things that
carry the meaning, and treating a mismatch in any of them as disqualifying no
matter how high the overlap:

**Polarity.** A claim and its source must agree about whether something
happens. One inserted "not" inverts a policy while changing overlap by a
rounding error.

**Quantities.** Numbers and their units are the load-bearing part of most
support answers. "$49 per month" and "$49 per year" differ by 12x and by one
token. Every number in a claim has to appear in the source it matches.

**Conditions.** "within 10 days of the return being received" and "of the
order being placed" start the clock at different events. Prepositional
conditions are compared as units rather than dissolved into a token bag.

Lexical overlap still runs, as the floor. A claim has to pass all four to be
called grounded, so the checks compose as a conjunction: overlap establishes
topical relevance, the rest establish that the claim says the same thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

_WORD = re.compile(r"[a-z0-9$]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "for", "with", "as", "by", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "will", "be", "has", "have",
    "your", "our", "their", "its",
}

# Explicit negation, plus the contracted forms that a naive tokenizer splits
# into harmless-looking fragments.
_NEGATIONS = re.compile(
    r"\b(not|never|no|none|cannot|can't|won't|doesn't|don't|isn't|aren't|"
    r"wasn't|weren't|shouldn't|wouldn't|couldn't|without|neither|nor|"
    r"excluded|excludes|denied|denies|unavailable|ineligible)\b",
    re.IGNORECASE,
)

# A number together with whatever unit follows it, because the unit is half
# the claim: "49 month" and "49 year" must not compare equal. The optional
# "per / a / each" is essential: without it "$49 per year" and "$49 per month"
# both reduce to a bare 49 and compare equal, which is a 12x pricing error the
# checker would wave through.
_QUANTITY = re.compile(
    r"(\$?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:per\s+|a\s+|each\s+|/\s*)?"
    r"(%|percent|dollars?|usd|business\s+days?|days?|daily|hours?|hourly|"
    r"minutes?|weeks?|weekly|months?|monthly|years?|yearly|annually|"
    r"times?|items?|units?)?",
    re.IGNORECASE,
)

# Units that mean the same thing written either way, so a genuine paraphrase
# is not punished for saying "monthly" instead of "per month".
_UNIT_SYNONYMS = {
    "monthly": "month", "months": "month",
    "yearly": "year", "annually": "year", "years": "year",
    "daily": "day", "days": "day",
    "business days": "business day", "business day": "business day",
    "hours": "hour", "hourly": "hour", "weeks": "week", "weekly": "week",
    "minutes": "minute", "dollars": "dollar", "usd": "dollar",
}

# Conditions: "after X", "of Y", "once Z". These say when a claim applies, and
# swapping one for another changes the claim entirely.
_CONDITION = re.compile(
    r"\b(after|once|of|upon|following|when|before|until|from)\s+"
    r"((?:the\s+)?(?:\w+\s+){0,4}\w+)",
    re.IGNORECASE,
)


def _tokenize(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _sentences(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


def _normalize_unit(unit: Optional[str]) -> str:
    if not unit:
        return ""
    key = re.sub(r"\s+", " ", unit.lower().strip())
    return _UNIT_SYNONYMS.get(key, key)


def _quantities(text: str) -> Set[str]:
    """Extract ``value:unit`` pairs, normalized so paraphrases compare equal."""
    found = set()
    lowered = text.lower()
    for match in _QUANTITY.finditer(lowered):
        raw_value, raw_unit = match.group(1), match.group(2)
        value = raw_value.replace(",", "").lstrip("$")
        # "$49" and "49 dollars" are the same quantity.
        unit = _normalize_unit(raw_unit)
        if raw_value.startswith("$") and not unit:
            unit = "dollar"
        found.add(f"{value}:{unit}")

    for phrase, quantity in _LEXICAL_QUANTITIES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            found.add(quantity)
    return found


# Words that assert a quantity without writing a digit. "Free" is a price
# claim of zero, and a source that states a price contradicts it even though
# the two share no numerals.
_LEXICAL_QUANTITIES = {
    "free": "0:dollar",
    "complimentary": "0:dollar",
    "no charge": "0:dollar",
    "unlimited": "unlimited:count",
    "instantly": "0:day",
    "immediately": "0:day",
    "same-day": "0:day",
    "same day": "0:day",
}


def _is_negated(text: str) -> bool:
    return bool(_NEGATIONS.search(text))


def _stem(word: str) -> str:
    """Crude suffix stripping, so "receive", "received", and "receives" match.

    A real stemmer would be better, but this stays dependency-free and the
    failure mode is symmetric: both the claim and the source get the same
    treatment, so a genuine paraphrase survives and a changed noun does not.
    """
    for suffix in ("ation", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    # Also drop a trailing "e", or "receive" and "received" stem differently
    # ("receive" matches no suffix above, "received" reduces to "receiv") and
    # the paraphrase this exists to accept would still be rejected.
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _conditions(text: str) -> Set[frozenset]:
    """Condition phrases, as stemmed content-word sets.

    Compared as sets of stems rather than as strings, because "once we receive
    your return" and "of the return being received" state the same condition
    with different grammar. Word order and inflection are noise here; which
    entity the condition is about is the signal.
    """
    found = set()
    for match in _CONDITION.finditer(text.lower()):
        words = {
            _stem(w) for w in _WORD.findall(match.group(2))
            if w not in _STOP and w not in {"being", "be"}
        }
        if words:
            found.add(frozenset(words))
    return found


@dataclass
class ClaimCheck:
    """One sentence, and why it was or was not considered grounded."""

    sentence: str
    grounded: bool
    best_overlap: float
    #: Which check rejected it. Empty when the claim is grounded.
    failed: str = ""
    detail: str = ""

    def __str__(self) -> str:
        if self.grounded:
            return f"grounded (overlap {self.best_overlap:.2f}): {self.sentence[:60]}"
        return f"UNGROUNDED [{self.failed}] {self.detail}: {self.sentence[:60]}"


@dataclass
class GroundingReport:
    claims: List[ClaimCheck] = field(default_factory=list)

    @property
    def grounded_fraction(self) -> float:
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.grounded) / len(self.claims)

    @property
    def ungrounded(self) -> List[ClaimCheck]:
        return [c for c in self.claims if not c.grounded]

    def to_dict(self) -> dict:
        return {
            "grounded_fraction": round(self.grounded_fraction, 4),
            "claims": [
                {
                    "sentence": c.sentence,
                    "grounded": c.grounded,
                    "best_overlap": c.best_overlap,
                    "failed": c.failed,
                    "detail": c.detail,
                }
                for c in self.claims
            ],
        }


def _check_against_source(sentence: str, source: str, min_overlap: float) -> ClaimCheck:
    """Test one claim against one source, returning the first failure."""
    claim_tokens, source_tokens = _tokenize(sentence), _tokenize(source)
    overlap = (
        len(claim_tokens & source_tokens) / len(claim_tokens) if claim_tokens else 0.0
    )

    if overlap < min_overlap:
        return ClaimCheck(sentence, False, round(overlap, 4), "overlap",
                          f"only {overlap:.0%} of the claim appears in this source")

    # Polarity. Checked before anything else: a negated claim against an
    # unnegated source is a contradiction however well the words line up.
    if _is_negated(sentence) != _is_negated(source):
        return ClaimCheck(sentence, False, round(overlap, 4), "polarity",
                          "the claim and the source disagree about negation")

    # Quantities. Every number in the claim must appear in the source, with a
    # compatible unit. Extra numbers in the source are fine; the source is
    # allowed to say more than the claim.
    claim_quantities = _quantities(sentence)
    if claim_quantities:
        source_quantities = _quantities(source)
        missing = claim_quantities - source_quantities
        # A bare number is satisfied by the same number with any unit, since
        # "ships in 5 days" against "5 business days" is the same claim.
        unmatched = {
            q for q in missing
            if not any(s.split(":")[0] == q.split(":")[0] and not q.split(":")[1]
                       for s in source_quantities)
        }
        if unmatched:
            return ClaimCheck(sentence, False, round(overlap, 4), "quantity",
                              f"{sorted(unmatched)} not supported by this source")

    # Conditions. A claim may state fewer conditions than the source, but the
    # ones it does state have to be the source's.
    claim_conditions = _conditions(sentence)
    if claim_conditions:
        source_conditions = _conditions(source)
        source_stems = {_stem(w) for w in source_tokens}
        # A condition is supported when the source states an overlapping one,
        # or when all of its content words appear anywhere in the source.
        # Requiring an exact phrase match would flag every paraphrase.
        unsupported = [
            condition for condition in claim_conditions
            if not any(condition & source_condition for source_condition in source_conditions)
            and not condition.issubset(source_stems)
        ]
        if unsupported:
            missing = sorted(" ".join(sorted(c)) for c in unsupported)
            return ClaimCheck(sentence, False, round(overlap, 4), "condition",
                              f"condition {missing} is not in this source")

    return ClaimCheck(sentence, True, round(overlap, 4))


def check_grounding(
    response: str,
    sources: List[str],
    min_overlap: float = 0.5,
    min_claim_tokens: int = 3,
) -> GroundingReport:
    """Check each claim in ``response`` against ``sources``.

    A claim is grounded when at least one source supports it on every check.
    When none do, the failure reported is the one from the closest source,
    since that is the most informative thing to show a reviewer.
    """
    report = GroundingReport()

    for sentence in _sentences(response):
        # Redaction placeholders are not factual claims to verify.
        if "redacted" in sentence.lower():
            continue
        if len(_tokenize(sentence)) < min_claim_tokens:
            continue

        best_failure: Optional[ClaimCheck] = None
        grounded = False
        for source in sources:
            check = _check_against_source(sentence, source, min_overlap)
            if check.grounded:
                report.claims.append(check)
                grounded = True
                break
            if best_failure is None or check.best_overlap > best_failure.best_overlap:
                best_failure = check

        if not grounded:
            report.claims.append(
                best_failure
                or ClaimCheck(sentence, False, 0.0, "no_sources", "no sources provided")
            )

    return report
