"""PII detection that validates instead of just pattern-matching.

The original detector scored 100% precision and recall, on eight cases written
in textbook formatting. Measured against how people actually type, it fails in
both directions at once:

*Misses.* "jane dot doe at example dot com", "123 45 6789", "+44 20 7946 0958".
Obfuscated contact details are not an exotic edge case; they are what you get
from pasted text, scraped content, and anyone trying not to be scraped.

*False positives.* A version string "1.2.3.4" is not an IP address. An order
number "4111 1111 1111 1112" is not a card. A measurement "100-200-3000" is
not a phone number. A redactor that mangles product identifiers gets turned
off, and then it protects nothing at all.

The difference here is validation. A credit card is not sixteen digits; it is
sixteen digits that pass the Luhn checksum and start with a real issuer
prefix. An SSN is not three-two-four digits; the SSA never issues 000, 666, or
900-999 in the area field. Checking those rules costs nothing and removes most
of the false positives, which is what makes the extra recall affordable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class PIIMatch:
    kind: str
    text: str
    start: int
    end: int
    #: Why this matched, for auditability. A redaction you cannot explain is
    #: one a reviewer cannot sign off on.
    reason: str = ""


class PIIDetector(Protocol):
    def detect(self, text: str) -> List[PIIMatch]:
        ...


# --- validators -------------------------------------------------------------

def luhn_valid(digits: str) -> bool:
    """The checksum every real card number satisfies.

    This is the single highest-value check available: it rejects roughly 90% of
    random digit strings, which is what most 16-digit false positives are.
    """
    numbers = [int(d) for d in digits if d.isdigit()]
    if len(numbers) < 12:
        return False
    total = 0
    for index, digit in enumerate(reversed(numbers)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


#: Issuer prefixes and their valid lengths. A number passing Luhn but starting
#: with 9 is not a card anyone issued.
_CARD_ISSUERS = (
    (re.compile(r"^4"), {13, 16, 19}),                       # Visa
    (re.compile(r"^5[1-5]"), {16}),                          # Mastercard
    (re.compile(r"^2(2[2-9]|[3-6]|7[01]|720)"), {16}),       # Mastercard 2-series
    (re.compile(r"^3[47]"), {15}),                           # Amex
    (re.compile(r"^3(0[0-5]|[68])"), {14, 16}),              # Diners
    (re.compile(r"^6(011|5|4[4-9])"), {16, 19}),             # Discover
)


def valid_card(digits: str) -> bool:
    if not luhn_valid(digits):
        return False
    return any(
        pattern.match(digits) and len(digits) in lengths
        for pattern, lengths in _CARD_ISSUERS
    )


def valid_ssn(area: str, group: str, serial: str) -> bool:
    """SSA issuance rules, which rule out most SSN-shaped strings.

    000, 666, and 900-999 are never issued as area numbers, and neither the
    group nor the serial is ever all zeroes.
    """
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


def valid_ipv4(octets: List[str]) -> bool:
    """Four octets, each 0-255, none zero-padded.

    The padding rule is what separates an address from a version string:
    "10.0.19041.1" fails on range, and "01.2.3.4" fails on the leading zero.
    """
    if len(octets) != 4:
        return False
    for octet in octets:
        if len(octet) > 1 and octet.startswith("0"):
            return False
        if not 0 <= int(octet) <= 255:
            return False
    return True


# --- patterns ---------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Obfuscated email: "name [at] domain.com", "name (at) domain dot com".
_EMAIL_OBFUSCATED = re.compile(
    r"\b[A-Za-z0-9._%+-]+\s*[\[\(\{]?\s*(?:at|@)\s*[\]\)\}]?\s*"
    r"[A-Za-z0-9.-]+\s*(?:[\[\(\{]?\s*dot\s*[\]\)\}]?\s*[A-Za-z0-9-]+)*"
    r"\s*(?:\.[A-Za-z]{2,}|\s+dot\s+[A-Za-z]{2,})\b",
    re.IGNORECASE,
)

# Fully spelled out: "jane dot doe at example dot com".
_EMAIL_SPELLED = re.compile(
    r"\b[A-Za-z0-9._%+-]+(?:\s+dot\s+[A-Za-z0-9-]+)+\s+at\s+"
    r"[A-Za-z0-9-]+(?:\s+dot\s+[A-Za-z0-9-]+)+\b",
    re.IGNORECASE,
)

_SSN = re.compile(r"\b(\d{3})[-\s](\d{2})[-\s](\d{4})\b")
_SSN_BARE = re.compile(r"\b(\d{3})(\d{2})(\d{4})\b")

_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# North American, plus an international form with a country code.
_PHONE_NA = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)
_PHONE_INTL = re.compile(r"\+\d{1,3}(?:[-.\s]\d{1,4}){2,5}\b")

_IPV4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")

# Context words that make a bare digit run much more likely to be sensitive.
_SSN_CONTEXT = re.compile(
    r"\b(ssn|social security|social|tax id|tin)\b[:\s]*$", re.IGNORECASE
)

# Context that makes a dotted quad a build identifier rather than an address.
# "1.2.3.4" is a perfectly valid IPv4 and a perfectly ordinary version string;
# only the surrounding words distinguish them.
_VERSION_CONTEXT = re.compile(
    r"\b(version|v|build|release|rev|revision|patch|firmware|sdk|schema)\b"
    r"[\s:]*$",
    re.IGNORECASE,
)


def is_nanp_number(digits: str) -> bool:
    """North American Numbering Plan rules for a 10 or 11 digit string.

    Area code and exchange both start 2-9, which is what makes "100-200-3000"
    a measurement rather than a phone number. Without this rule any three
    dash-separated digit groups look like a phone.
    """
    national = digits[-10:]
    if len(national) != 10:
        return False
    if len(digits) == 11 and not digits.startswith("1"):
        return False
    area, exchange = national[:3], national[3:6]
    return area[0] in "23456789" and exchange[0] in "23456789"

_MASKS = {
    "email": "[REDACTED_EMAIL]",
    "ssn": "[REDACTED_SSN]",
    "credit_card": "[REDACTED_CARD]",
    "phone": "[REDACTED_PHONE]",
    "ip_address": "[REDACTED_IP]",
}


class ValidatingPIIDetector:
    """Detects PII, then checks it is actually PII before reporting it.

    Order matters. Emails are claimed first because they contain characters the
    other patterns would carve up; cards before phones because a 16-digit run
    contains phone-shaped substrings.
    """

    def detect(self, text: str) -> List[PIIMatch]:
        found: List[PIIMatch] = []
        claimed = [False] * len(text)

        def claim(start: int, end: int) -> bool:
            if any(claimed[start:end]):
                return False
            for index in range(start, end):
                claimed[index] = True
            return True

        def add(kind: str, match: re.Match, reason: str) -> None:
            if claim(match.start(), match.end()):
                found.append(
                    PIIMatch(kind, match.group(), match.start(), match.end(), reason)
                )

        for pattern, reason in (
            (_EMAIL, "standard address"),
            (_EMAIL_SPELLED, "spelled-out address"),
            (_EMAIL_OBFUSCATED, "obfuscated address"),
        ):
            for match in pattern.finditer(text):
                add("email", match, reason)

        for match in _SSN.finditer(text):
            if valid_ssn(match.group(1), match.group(2), match.group(3)):
                add("ssn", match, "valid SSA area, group, and serial")

        # A bare nine-digit run is only an SSN when something nearby says so;
        # otherwise it is an order number.
        for match in _SSN_BARE.finditer(text):
            if not valid_ssn(match.group(1), match.group(2), match.group(3)):
                continue
            if _SSN_CONTEXT.search(text[: match.start()]):
                add("ssn", match, "nine digits in an SSN context")

        for match in _CARD.finditer(text):
            digits = re.sub(r"\D", "", match.group())
            if valid_card(digits):
                add("credit_card", match, "passes Luhn and a known issuer prefix")

        for pattern, reason in (
            (_PHONE_INTL, "international format"),
            (_PHONE_NA, "North American format"),
        ):
            for match in pattern.finditer(text):
                digits = re.sub(r"\D", "", match.group())
                # A North American number is 10 digits, or 11 with the country
                # code. Anything else is a measurement or an identifier.
                if pattern is _PHONE_NA and not is_nanp_number(digits):
                    continue
                add("phone", match, reason)

        for match in _IPV4.finditer(text):
            if not valid_ipv4([match.group(i) for i in range(1, 5)]):
                continue
            if _VERSION_CONTEXT.search(text[: match.start()]):
                continue  # "Version 1.2.3.4" is a build number
            add("ip_address", match, "four octets in range, no zero padding")

        found.sort(key=lambda m: m.start)
        return found


@dataclass
class RedactionResult:
    redacted_text: str
    matches: List[PIIMatch]

    @property
    def kinds(self) -> set:
        return {m.kind for m in self.matches}


def redact(text: str, detector: Optional[PIIDetector] = None) -> RedactionResult:
    """Replace every detected span with a typed placeholder."""
    detector = detector or ValidatingPIIDetector()
    matches = detector.detect(text)
    redacted = text
    # Replace from the end so earlier offsets stay valid.
    for match in sorted(matches, key=lambda m: m.start, reverse=True):
        mask = _MASKS.get(match.kind, "[REDACTED]")
        redacted = redacted[: match.start] + mask + redacted[match.end :]
    return RedactionResult(redacted, matches)
