"""The guardrail gate: orchestrates rate limiting, PII redaction, and
grounding validation into a single pass/block/warn decision on outbound LLM
text before it reaches a user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .grounding import check_grounding as check_grounding_v1
from .grounding_v2 import GroundingReport, check_grounding as check_grounding_v2
from .pii import RegexPIIDetector
from .pii_v2 import PIIDetector, PIIMatch, ValidatingPIIDetector, redact
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
                 min_grounding_fraction: float = 0.6,
                 version: str = "v2") -> None:
        # v2 by default. v1's failure modes are silent in both directions: it
        # passes hallucinations that reuse the source's words, and it redacts
        # order numbers that merely look like cards. Selecting it is only
        # useful for reproducing the comparison in the benchmark.
        self.version = version
        if pii_detector is not None:
            self.pii_detector = pii_detector
        else:
            self.pii_detector = (
                RegexPIIDetector() if version == "v1" else ValidatingPIIDetector()
            )
        self.check_grounding = (
            check_grounding_v1 if version == "v1" else check_grounding_v2
        )
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
            grounding_report = self.check_grounding(redaction.redacted_text, sources)
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
