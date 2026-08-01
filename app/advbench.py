"""Scoring both guardrail versions on the easy and the hard half.

The original benchmarks reported 100% PII precision/recall and 83% grounding
accuracy. Both were honest measurements of the easy half: PII in textbook
formatting, and hallucinations that use words the sources never did.

This runs the same checks against cases built to be hard, and reports the
splits separately, because a single blended number lets a guardrail hide its
real failures behind its easy wins.

    python -m app.advbench
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .adversarial import (
    GROUNDING_CASES,
    PII_CASES,
    SOURCE_DOCS,
    grounding_split,
    pii_split,
)
from .grounding import check_grounding as grounding_v1
from .grounding_v2 import check_grounding as grounding_v2
from .pii import RegexPIIDetector
from .pii_v2 import ValidatingPIIDetector

GROUNDING_THRESHOLD = 0.6
MIN_OVERLAP = 0.3


def score_pii(detector, cases) -> Dict:
    true_positive = false_positive = false_negative = 0
    errors: List[str] = []
    for case in cases:
        found = {match.kind for match in detector.detect(case.text)}
        true_positive += len(found & case.expected)
        false_positive += len(found - case.expected)
        false_negative += len(case.expected - found)
        if found != case.expected:
            errors.append(
                f"expected {sorted(case.expected) or '[]'}, "
                f"got {sorted(found) or '[]'}: {case.text[:52]}"
            )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "n": len(cases),
        "errors": errors,
    }


def score_grounding(checker, cases) -> Dict:
    correct = 0
    missed_hallucinations = 0
    false_alarms = 0
    errors: List[str] = []
    for case in cases:
        report = checker(case.response, SOURCE_DOCS, min_overlap=MIN_OVERLAP)
        predicted = report.grounded_fraction >= GROUNDING_THRESHOLD
        if predicted == case.grounded:
            correct += 1
        else:
            # Letting a hallucination through is the failure that reaches a
            # user. Flagging a good answer is only annoying.
            if case.grounded:
                false_alarms += 1
            else:
                missed_hallucinations += 1
            errors.append(
                f"expected grounded={case.grounded}, got {predicted}: "
                f"{case.response[:56]}"
            )
    return {
        "accuracy": round(correct / len(cases), 4) if cases else 1.0,
        "missed_hallucinations": missed_hallucinations,
        "false_alarms": false_alarms,
        "n": len(cases),
        "errors": errors,
    }


def build_report() -> Dict:
    detectors = {"v1 regex": RegexPIIDetector(), "v2 validating": ValidatingPIIDetector()}
    checkers = {"v1 overlap": grounding_v1, "v2 semantic": grounding_v2}

    report: Dict = {"pii": {}, "grounding": {}}
    for name, detector in detectors.items():
        report["pii"][name] = {
            split: score_pii(detector, pii_split(split))
            for split in ("clean", "obfuscated", "decoy")
        }
        report["pii"][name]["all"] = score_pii(detector, PII_CASES)

    for name, checker in checkers.items():
        report["grounding"][name] = {
            split: score_grounding(checker, grounding_split(split))
            for split in ("lexical", "semantic")
        }
        report["grounding"][name]["all"] = score_grounding(checker, GROUNDING_CASES)
    return report


def format_report(report: Dict) -> str:
    lines = [
        "PII detection",
        "=" * 72,
        f"{'detector':<16}{'split':<13}{'precision':>11}{'recall':>9}{'FP':>5}{'FN':>5}",
        "-" * 72,
    ]
    for name, splits in report["pii"].items():
        for split in ("clean", "obfuscated", "decoy", "all"):
            row = splits[split]
            label = name if split == "clean" else ""
            lines.append(
                f"{label:<16}{split:<13}{row['precision']:>10.0%}"
                f"{row['recall']:>9.0%}{row['false_positives']:>5}"
                f"{row['false_negatives']:>5}"
            )
        lines.append("")

    lines += [
        "Grounding / hallucination flagging",
        "=" * 72,
        f"{'checker':<16}{'split':<13}{'accuracy':>11}{'missed':>9}{'false alarms':>14}",
        "-" * 72,
    ]
    for name, splits in report["grounding"].items():
        for split in ("lexical", "semantic", "all"):
            row = splits[split]
            label = name if split == "lexical" else ""
            lines.append(
                f"{label:<16}{split:<13}{row['accuracy']:>10.0%}"
                f"{row['missed_hallucinations']:>9}{row['false_alarms']:>14}"
            )
        lines.append("")

    lines += [
        "obfuscated = PII as people actually type it; decoy = looks like PII, is not",
        "semantic   = hallucinations that reuse the source's own vocabulary",
        "missed     = hallucinations passed to the user, the failure that matters",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    report = build_report()
    print(format_report(report))

    if args.show_errors:
        for section in ("pii", "grounding"):
            for name, splits in report[section].items():
                errors = splits["all"]["errors"]
                if errors:
                    print(f"\n{section} / {name}:")
                    for error in errors:
                        print(f"  {error}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
