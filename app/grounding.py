"""Citation/grounding validation: does each claim in an LLM response actually
have support in the provided source documents?

Splits the response into sentences and checks each one's lexical overlap
against the source set. A sentence with no meaningful overlap against any
source is flagged as ungrounded -- a cheap, explainable first line of defense
against hallucination before it reaches a user. It won't catch subtle
factual drift, but it reliably catches the common case: the model asserting
something the retrieved sources never said.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "for", "with", "as", "by", "it", "this", "that",
    "i", "you", "he", "she", "they", "we", "will", "be", "has", "have",
}


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


@dataclass
class ClaimCheck:
    sentence: str
    grounded: bool
    best_overlap: float


@dataclass
class GroundingReport:
    claims: list[ClaimCheck]

    @property
    def grounded_fraction(self) -> float:
        if not self.claims:
            return 1.0
        return sum(1 for c in self.claims if c.grounded) / len(self.claims)

    @property
    def ungrounded(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.grounded]


def check_grounding(response: str, sources: list[str], min_overlap: float = 0.5,
                    min_claim_tokens: int = 3) -> GroundingReport:
    source_tokens = [_tokenize(s) for s in sources]
    claims = []
    for sentence in _sentences(response):
        # a sentence that's just a redaction placeholder ("Contact
        # [REDACTED_EMAIL].") isn't a factual claim to verify against sources
        if "redacted" in sentence.lower():
            continue
        s_tokens = _tokenize(sentence)
        # short/non-substantive fragments aren't factual claims either
        if len(s_tokens) < min_claim_tokens:
            continue
        best = 0.0
        for src_tokens in source_tokens:
            if not src_tokens:
                continue
            overlap = len(s_tokens & src_tokens) / len(s_tokens)
            best = max(best, overlap)
        claims.append(ClaimCheck(sentence, best >= min_overlap, round(best, 4)))
    return GroundingReport(claims)
