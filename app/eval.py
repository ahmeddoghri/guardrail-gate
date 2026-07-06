"""Does the gate actually catch what it's supposed to?

Two labeled benchmarks bundled together:

1. PII redaction: a set of synthetic support-chat lines, each labeled with the
   PII spans it contains (or none). We report precision/recall on span
   detection.
2. Grounding: a set of (response, sources, is_hallucinated) triples -- some
   responses are fully supported by their sources, some assert things the
   sources never said. We report how well the grounding gate separates them.

    python -m app.eval
"""
from __future__ import annotations


from .gate import GuardrailGate
from .grounding import check_grounding
from .pii import RegexPIIDetector

# --- PII benchmark -----------------------------------------------------------

PII_CASES = [
    ("Reach me at jane.doe@example.com if you have questions.", {"email"}),
    ("My SSN is 123-45-6789, please update your records.", {"ssn"}),
    ("Card number 4111 1111 1111 1111 expires next month.", {"credit_card"}),
    ("Call me at 415-555-0134 tomorrow morning.", {"phone"}),
    ("The server IP is 192.168.1.42 if you need to check logs.", {"ip_address"}),
    ("Thanks for the update, that all makes sense to me.", set()),
    ("Our office is at 100 Main Street, open 9 to 5.", set()),
    ("You can email support@company.com or call 212-555-9876.", {"email", "phone"}),
]


def eval_pii() -> dict:
    detector = RegexPIIDetector()
    tp = fp = fn = 0
    for text, expected_kinds in PII_CASES:
        found_kinds = {m.kind for m in detector.detect(text)}
        tp += len(found_kinds & expected_kinds)
        fp += len(found_kinds - expected_kinds)
        fn += len(expected_kinds - found_kinds)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "n": len(PII_CASES)}


# --- grounding benchmark -----------------------------------------------------

SOURCE_DOCS = [
    "The product ships within 3 to 5 business days after order confirmation.",
    "Refunds are processed within 10 business days of the return being received.",
    "The premium plan costs $49 per month and includes priority support.",
]

GROUNDING_CASES = [
    ("Your order ships within 3 to 5 business days.", SOURCE_DOCS, True),
    ("The premium plan is $49 a month with priority support included.", SOURCE_DOCS, True),
    ("Refunds take 10 business days once we receive your return.", SOURCE_DOCS, True),
    ("Your order will ship within 24 hours guaranteed.", SOURCE_DOCS, False),  # hallucinated
    ("The premium plan is completely free for the first year.", SOURCE_DOCS, False),  # hallucinated
    ("We offer same-day refunds with no questions asked.", SOURCE_DOCS, False),  # hallucinated
]


def eval_grounding(min_overlap: float = 0.3, flag_threshold: float = 0.6) -> dict:
    correct = 0
    for response, sources, is_grounded in GROUNDING_CASES:
        report = check_grounding(response, sources, min_overlap=min_overlap)
        predicted_grounded = report.grounded_fraction >= flag_threshold
        if predicted_grounded == is_grounded:
            correct += 1
    return {"accuracy": round(correct / len(GROUNDING_CASES), 4), "n": len(GROUNDING_CASES)}


def main() -> None:
    pii = eval_pii()
    grounding = eval_grounding()
    print("--- PII redaction benchmark ---")
    print(f"precision={pii['precision']:.0%}  recall={pii['recall']:.0%}  (n={pii['n']})")
    print()
    print("--- grounding / hallucination-flagging benchmark ---")
    print(f"accuracy={grounding['accuracy']:.0%}  (n={grounding['n']})")
    print()

    gate = GuardrailGate()
    print("--- end-to-end gate example ---")
    result = gate.check(
        "demo-client",
        "Contact jane.doe@example.com. Your order ships within 24 hours guaranteed.",
        sources=SOURCE_DOCS,
    )
    print(f"allowed={result.allowed}  warnings={result.warnings}")
    print(f"redacted_text={result.redacted_text!r}")


if __name__ == "__main__":
    main()
