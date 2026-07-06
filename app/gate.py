"""The guardrail gate: orchestrates rate limiting, PII redaction, and
grounding validation into a single pass/block/warn decision on outbound LLM
text before it reaches a user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .grounding import GroundingReport, check_grounding
from .pii import PIIDetector, PIIMatch, RegexPIIDetector, redact
from .ratelimit import TokenBucketLimiter


@dataclass
class GateResult:
    allowed: bool
    redacted_text: str
    pii_found: list[PIIMatch]
    grounding: Optional[GroundingReport]
    warnings: list[str] = field(default_factory=list)
    rate_limited: bool = False


class GuardrailGate:
    def __init__(self, pii_detector: Optional[PIIDetector] = None,
                 rate_limiter: Optional[TokenBucketLimiter] = None,
                 min_grounding_fraction: float = 0.6) -> None:
        self.pii_detector = pii_detector or RegexPIIDetector()
        self.rate_limiter = rate_limiter or TokenBucketLimiter()
        self.min_grounding_fraction = min_grounding_fraction

    def check(self, client_id: str, text: str, sources: Optional[list[str]] = None) -> GateResult:
        if not self.rate_limiter.allow(client_id):
            return GateResult(allowed=False, redacted_text="", pii_found=[],
                              grounding=None, warnings=["rate_limited"], rate_limited=True)

        redaction = redact(text, self.pii_detector)
        warnings = []
        if redaction.matches:
            kinds = sorted({m.kind for m in redaction.matches})
            warnings.append(f"pii_redacted:{','.join(kinds)}")

        grounding_report = None
        if sources is not None:
            grounding_report = check_grounding(redaction.redacted_text, sources)
            if grounding_report.grounded_fraction < self.min_grounding_fraction:
                warnings.append("low_grounding")

        allowed = True
        if grounding_report is not None and grounding_report.grounded_fraction < self.min_grounding_fraction:
            allowed = False

        return GateResult(
            allowed=allowed,
            redacted_text=redaction.redacted_text,
            pii_found=redaction.matches,
            grounding=grounding_report,
            warnings=warnings,
        )
