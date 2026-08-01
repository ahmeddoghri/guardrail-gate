"""Tests for the hard half: obfuscated PII, decoys, and semantic hallucination."""

from __future__ import annotations

import pytest

from app.adversarial import (
    GROUNDING_CASES,
    PII_CASES,
    SOURCE_DOCS,
    grounding_split,
    pii_split,
)
from app.advbench import build_report, score_grounding, score_pii
from app.gate import GuardrailGate
from app.grounding import check_grounding as grounding_v1
from app.grounding_v2 import (
    _conditions,
    _is_negated,
    _quantities,
    _stem,
    check_grounding as grounding_v2,
)
from app.pii import RegexPIIDetector
from app.pii_v2 import (
    ValidatingPIIDetector,
    is_nanp_number,
    luhn_valid,
    redact,
    valid_card,
    valid_ipv4,
    valid_ssn,
)

MIN_OVERLAP = 0.3


def v2_grounded(response: str) -> bool:
    return grounding_v2(response, SOURCE_DOCS, min_overlap=MIN_OVERLAP).grounded_fraction >= 0.6


def v1_grounded(response: str) -> bool:
    return grounding_v1(response, SOURCE_DOCS, min_overlap=MIN_OVERLAP).grounded_fraction >= 0.6


# --- the finding: overlap cannot see meaning --------------------------------

def test_v1_passes_a_negated_claim():
    """One inserted 'not' inverts the policy and barely moves the overlap."""
    assert v1_grounded(
        "Refunds are not processed within 10 business days of the return being received."
    )


def test_v2_catches_a_negated_claim():
    assert not v2_grounded(
        "Refunds are not processed within 10 business days of the return being received."
    )


def test_v1_passes_a_unit_substitution():
    """per month -> per year is a 12x pricing error at 0.89 overlap."""
    assert v1_grounded("The premium plan costs $49 per year and includes priority support.")


def test_v2_catches_a_unit_substitution():
    assert not v2_grounded(
        "The premium plan costs $49 per year and includes priority support."
    )


def test_v1_passes_a_digit_substitution():
    assert v1_grounded(
        "The product ships within 30 to 50 business days after order confirmation."
    )


def test_v2_catches_a_digit_substitution():
    assert not v2_grounded(
        "The product ships within 30 to 50 business days after order confirmation."
    )


def test_v2_catches_a_swapped_condition():
    """'of the order being placed' starts the clock at a different event."""
    assert not v2_grounded(
        "Refunds are processed within 10 business days of the order being placed."
    )


def test_v2_still_accepts_a_genuine_paraphrase():
    """The check is worthless if it flags every rewording."""
    assert v2_grounded("Shipping takes 3 to 5 business days once your order is confirmed.")
    assert v2_grounded("You will be charged $49 monthly for the premium plan.")


def test_v2_beats_v1_on_semantic_hallucinations():
    v1 = score_grounding(grounding_v1, grounding_split("semantic"))
    v2 = score_grounding(grounding_v2, grounding_split("semantic"))
    assert v1["missed_hallucinations"] > v2["missed_hallucinations"]
    assert v2["missed_hallucinations"] == 0


def test_v2_does_not_regress_on_the_easy_half():
    assert score_grounding(grounding_v2, grounding_split("lexical"))["accuracy"] >= (
        score_grounding(grounding_v1, grounding_split("lexical"))["accuracy"]
    )


def test_v2_raises_no_false_alarms():
    assert score_grounding(grounding_v2, GROUNDING_CASES)["false_alarms"] == 0


# --- grounding internals ----------------------------------------------------

def test_negation_detects_contractions():
    assert _is_negated("we don't ship there")
    assert _is_negated("that isn't included")
    assert not _is_negated("we ship there")


def test_quantities_keep_their_units():
    assert "49:month" in _quantities("$49 per month")
    assert "49:year" in _quantities("$49 per year")
    assert _quantities("$49 per month") != _quantities("$49 per year")


def test_per_is_not_swallowed():
    """Without handling 'per', both prices reduce to a bare 49 and compare equal."""
    assert _quantities("costs $49 per year") != _quantities("costs $49 per month")


def test_unit_synonyms_normalize():
    assert _quantities("$49 monthly") == _quantities("$49 per month")


def test_free_is_a_quantity_claim():
    """'Free' contradicts a stated price despite sharing no numerals."""
    assert "0:dollar" in _quantities("the plan is completely free")


def test_stemming_survives_inflection():
    assert _stem("received") == _stem("receives") == _stem("receive")


def test_conditions_compare_as_stemmed_sets():
    a = _conditions("once we receive your return")
    b = _conditions("of the return being received")
    assert any(x & y for x in a for y in b)


def test_ungrounded_claims_say_which_check_failed():
    """A block a reviewer cannot explain is a block they will override."""
    report = grounding_v2(
        "Refunds are not processed within 10 business days of the return being received.",
        SOURCE_DOCS, min_overlap=MIN_OVERLAP,
    )
    assert report.ungrounded
    assert report.ungrounded[0].failed == "polarity"


def test_empty_response_is_vacuously_grounded():
    assert grounding_v2("", SOURCE_DOCS).grounded_fraction == 1.0


def test_no_sources_flags_everything():
    report = grounding_v2("The plan costs $49 per month.", [])
    assert report.grounded_fraction == 0.0


def test_redaction_placeholders_are_not_claims():
    report = grounding_v2("Contact [REDACTED_EMAIL].", SOURCE_DOCS)
    assert report.claims == []


# --- PII validators ---------------------------------------------------------

def test_luhn_accepts_real_card_numbers():
    for number in ("4111111111111111", "5555555555554444", "378282246310005"):
        assert luhn_valid(number)


def test_luhn_rejects_a_transposed_digit():
    assert not luhn_valid("4111111111111112")


def test_card_requires_a_real_issuer_prefix():
    """Passing Luhn is not enough; 9-series cards are not issued."""
    assert valid_card("4111111111111111")
    assert not valid_card("1234567890123456")


def test_ssn_rejects_unissued_area_numbers():
    assert valid_ssn("078", "05", "1120")
    assert not valid_ssn("000", "05", "1120")
    assert not valid_ssn("666", "05", "1120")
    assert not valid_ssn("900", "05", "1120")


def test_ipv4_rejects_out_of_range_and_padded_octets():
    assert valid_ipv4(["192", "168", "1", "42"])
    assert not valid_ipv4(["999", "888", "777", "666"])
    assert not valid_ipv4(["01", "2", "3", "4"])


def test_nanp_rejects_impossible_area_codes():
    """This is what keeps '100-200-3000 units' from being a phone number."""
    assert is_nanp_number("4155550134")
    assert not is_nanp_number("1002003000")


# --- PII detection ----------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("Email me at jane.doe [at] example.com", "email"),
    ("reach me: jane dot doe at example dot com", "email"),
    ("My social is 123 45 6789", "ssn"),
    ("Ring me on +44 20 7946 0958", "phone"),
    ("card 4111111111111111 on file", "credit_card"),
])
def test_obfuscated_pii_is_detected(text, kind):
    assert kind in {m.kind for m in ValidatingPIIDetector().detect(text)}


@pytest.mark.parametrize("text", [
    "Version 1.2.3.4 was released yesterday.",
    "Order 4111 1111 1111 1112 shipped today.",
    "We measured between 100-200-3000 units.",
    "Build 10.0.19041.1 is current.",
    "The SKU is 999-99-9999 in our catalog.",
])
def test_decoys_are_not_redacted(text):
    """A redactor that mangles order numbers gets switched off."""
    assert ValidatingPIIDetector().detect(text) == []


def test_v2_beats_v1_on_obfuscated_pii():
    v1 = score_pii(RegexPIIDetector(), pii_split("obfuscated"))
    v2 = score_pii(ValidatingPIIDetector(), pii_split("obfuscated"))
    assert v2["recall"] > v1["recall"]
    assert v2["false_negatives"] == 0


def test_v2_eliminates_decoy_false_positives():
    v1 = score_pii(RegexPIIDetector(), pii_split("decoy"))
    v2 = score_pii(ValidatingPIIDetector(), pii_split("decoy"))
    assert v1["false_positives"] > 0
    assert v2["false_positives"] == 0


def test_v2_does_not_regress_on_clean_pii():
    assert score_pii(ValidatingPIIDetector(), pii_split("clean"))["f1"] == 1.0


def test_matches_carry_a_reason():
    matches = ValidatingPIIDetector().detect("card 4111111111111111 on file")
    assert matches[0].reason


def test_redaction_masks_by_kind():
    result = redact("write to jane@example.com or call 415-555-0134")
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "jane@example.com" not in result.redacted_text


def test_redaction_preserves_surrounding_text():
    result = redact("Contact jane@example.com today please")
    assert result.redacted_text.startswith("Contact ")
    assert result.redacted_text.endswith(" today please")


def test_overlapping_matches_are_claimed_once():
    matches = ValidatingPIIDetector().detect("card 4111 1111 1111 1111 here")
    assert len(matches) == 1
    assert matches[0].kind == "credit_card"


# --- the gate ---------------------------------------------------------------

def test_gate_defaults_to_v2():
    gate = GuardrailGate()
    assert gate.version == "v2"


def test_gate_blocks_a_semantic_hallucination():
    """End to end: the case v1 waved through."""
    gate = GuardrailGate()
    result = gate.check(
        "client-a",
        "The premium plan costs $49 per year and includes priority support.",
        sources=SOURCE_DOCS,
    )
    assert not result.allowed
    assert "low_grounding" in result.warnings


def test_gate_allows_a_grounded_answer():
    gate = GuardrailGate()
    result = gate.check(
        "client-b", "Your order ships within 3 to 5 business days.", sources=SOURCE_DOCS
    )
    assert result.allowed


def test_gate_does_not_redact_an_order_number():
    gate = GuardrailGate()
    result = gate.check("client-c", "Order 4111 1111 1111 1112 shipped today.")
    assert "REDACTED" not in result.redacted_text


def test_gate_redacts_obfuscated_contact_details():
    gate = GuardrailGate()
    result = gate.check("client-d", "reach me: jane dot doe at example dot com")
    assert "REDACTED" in result.redacted_text


def test_v1_gate_still_selectable_for_comparison():
    gate = GuardrailGate(version="v1")
    assert gate.version == "v1"
    result = gate.check(
        "client-e",
        "The premium plan costs $49 per year and includes priority support.",
        sources=SOURCE_DOCS,
    )
    assert result.allowed  # the regression, reproducible on demand


# --- the benchmark ----------------------------------------------------------

def test_benchmark_covers_both_versions_and_all_splits():
    report = build_report()
    assert set(report["pii"]) == {"v1 regex", "v2 validating"}
    assert set(report["grounding"]["v2 semantic"]) == {"lexical", "semantic", "all"}


def test_benchmark_is_reproducible():
    assert build_report() == build_report()


def test_benchmark_shows_v2_missing_no_hallucinations():
    report = build_report()
    assert report["grounding"]["v2 semantic"]["all"]["missed_hallucinations"] == 0
    assert report["grounding"]["v1 overlap"]["all"]["missed_hallucinations"] > 0


def test_corpus_has_both_difficulties():
    assert grounding_split("lexical") and grounding_split("semantic")
    assert pii_split("clean") and pii_split("obfuscated") and pii_split("decoy")


def test_every_pii_case_declares_expectations():
    for case in PII_CASES:
        assert isinstance(case.expected, set)
