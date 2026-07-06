"""PII detection and redaction.

Regex-based detection for structured PII (emails, phone numbers, SSNs, credit
cards, IP addresses) covers the majority of what actually leaks in LLM output
and needs no model or dependency. Pass a real NER model (spaCy, a hosted PII
API, etc.) via the same ``PIIDetector`` interface to also catch unstructured
PII like names -- regex fundamentally can't do that reliably.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class PIIMatch:
    kind: str
    text: str
    start: int
    end: int


_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
# order matters: credit_card's broad digit pattern would otherwise swallow
# phone numbers and SSNs if checked first
_ORDER = ["email", "ssn", "phone", "credit_card", "ip_address"]

_MASKS = {
    "email": "[REDACTED_EMAIL]",
    "ssn": "[REDACTED_SSN]",
    "credit_card": "[REDACTED_CARD]",
    "phone": "[REDACTED_PHONE]",
    "ip_address": "[REDACTED_IP]",
}


class PIIDetector(Protocol):
    def detect(self, text: str) -> list[PIIMatch]:
        ...


class RegexPIIDetector:
    def detect(self, text: str) -> list[PIIMatch]:
        found: list[PIIMatch] = []
        claimed = [False] * len(text)
        for kind in _ORDER:
            for m in _PATTERNS[kind].finditer(text):
                if any(claimed[m.start():m.end()]):
                    continue  # already matched by a higher-priority pattern
                found.append(PIIMatch(kind, m.group(), m.start(), m.end()))
                for i in range(m.start(), m.end()):
                    claimed[i] = True
        found.sort(key=lambda p: p.start)
        return found


@dataclass
class RedactionResult:
    redacted_text: str
    matches: list[PIIMatch]


def redact(text: str, detector: PIIDetector) -> RedactionResult:
    matches = detector.detect(text)
    # replace from the end so earlier offsets stay valid
    redacted = text
    for m in sorted(matches, key=lambda p: p.start, reverse=True):
        redacted = redacted[: m.start] + _MASKS.get(m.kind, "[REDACTED]") + redacted[m.end:]
    return RedactionResult(redacted, matches)
