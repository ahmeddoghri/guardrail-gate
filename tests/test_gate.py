from app.eval import SOURCE_DOCS, eval_grounding, eval_pii
from app.gate import GuardrailGate
from app.grounding import check_grounding
from app.pii import RegexPIIDetector, redact
from app.ratelimit import TokenBucketLimiter


def test_detects_email():
    matches = RegexPIIDetector().detect("Reach me at jane.doe@example.com please.")
    assert any(m.kind == "email" for m in matches)


def test_detects_ssn_and_not_phone():
    matches = RegexPIIDetector().detect("SSN 123-45-6789 on file.")
    kinds = {m.kind for m in matches}
    assert kinds == {"ssn"}


def test_redaction_masks_all_matches():
    result = redact("Email me at a@b.com or call 415-555-0134.", RegexPIIDetector())
    assert "a@b.com" not in result.redacted_text
    assert "415-555-0134" not in result.redacted_text
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text


def test_grounded_claim_passes():
    report = check_grounding("Your order ships within 3 to 5 business days.", SOURCE_DOCS)
    assert report.grounded_fraction == 1.0


def test_hallucinated_claim_flagged():
    report = check_grounding("Your order ships within 24 hours guaranteed, no exceptions.", SOURCE_DOCS)
    assert report.grounded_fraction < 0.6
    assert report.ungrounded


def test_rate_limiter_blocks_after_capacity_exhausted():
    t = [0.0]
    limiter = TokenBucketLimiter(capacity=3, refill_per_second=0.0, clock=lambda: t[0])
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False  # capacity exhausted, no refill


def test_rate_limiter_refills_over_time():
    t = [0.0]
    limiter = TokenBucketLimiter(capacity=1, refill_per_second=1.0, clock=lambda: t[0])
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-b") is False
    t[0] += 2.0
    assert limiter.allow("client-b") is True


def test_gate_blocks_hallucinated_response():
    gate = GuardrailGate()
    result = gate.check("client-1", "Your order ships within 24 hours guaranteed.", sources=SOURCE_DOCS)
    assert not result.allowed
    assert "low_grounding" in result.warnings


def test_gate_allows_grounded_response_and_redacts_pii():
    gate = GuardrailGate()
    result = gate.check(
        "client-1",
        "Contact jane.doe@example.com. Your order ships within 3 to 5 business days.",
        sources=SOURCE_DOCS,
    )
    assert result.allowed
    assert any(w.startswith("pii_redacted") for w in result.warnings)
    assert "jane.doe@example.com" not in result.redacted_text


def test_gate_without_sources_skips_grounding_check():
    gate = GuardrailGate()
    result = gate.check("client-1", "Just a plain response with no sources provided.")
    assert result.allowed
    assert result.grounding is None


def test_benchmarks_meet_a_reasonable_bar():
    pii = eval_pii()
    grounding = eval_grounding()
    assert pii["precision"] >= 0.9
    assert pii["recall"] >= 0.9
    assert grounding["accuracy"] >= 0.8
