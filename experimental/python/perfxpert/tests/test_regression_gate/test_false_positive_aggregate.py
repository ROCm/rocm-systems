"""Aggregate: false-positive rate ≤ 5% on proven optimizations corpus.

Spec §7 exit criterion 4 + risk R17. 10 seeds → at most 0 false-positive
rejections (5% of 10 = 0.5, floors to 0).
"""

import json
from pathlib import Path

import pytest

from tests.test_regression_gate.proven_optimization_runner import (
    ProvenOptimizationRunner,
)


FALSE_POSITIVE_THRESHOLD = 0.05

OUTPUT_DIR = Path(__file__).parent / "_runner_outputs"


@pytest.mark.regression_gate
def test_runner_can_load_all_10_seeds() -> None:
    runner = ProvenOptimizationRunner()
    cases = runner.load_seed_cases()
    if len(cases) == 0:
        pytest.skip("proven_optimizations fixture not present")
    assert len(cases) >= 10, f"Expected ≥ 10 seed cases, got {len(cases)}"


@pytest.mark.regression_gate
def test_false_positive_rate_at_or_below_5pct() -> None:
    runner = ProvenOptimizationRunner()
    cases = runner.load_seed_cases()
    if len(cases) == 0:
        pytest.skip("proven_optimizations fixture not present")

    rejected = []
    accepted = []
    per_case_outcomes = []

    for case in cases:
        verdict = runner.run_on_case(case)
        per_case_outcomes.append(
            {
                "case_id": case.case_id,
                "status": verdict.status,
                "failing_gate": verdict.failing_gate,
                "detail": verdict.detail,
            }
        )
        if verdict.status == "pass":
            accepted.append(case.case_id)
        else:
            rejected.append((case.case_id, verdict.failing_gate, verdict.detail))

    fp_rate = len(rejected) / len(cases)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "_aggregate.json").write_text(
        json.dumps(
            {
                "total": len(cases),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "false_positive_rate": round(fp_rate, 4),
                "threshold": FALSE_POSITIVE_THRESHOLD,
                "per_case": per_case_outcomes,
            },
            indent=2,
        )
    )

    assert fp_rate <= FALSE_POSITIVE_THRESHOLD, (
        f"Regression-gate false-positive rate {fp_rate:.1%} "
        f"exceeds {FALSE_POSITIVE_THRESHOLD:.0%}. Rejected: {rejected}\n"
        f"Spec §7 criterion 4 + R17 — audit gate BLOCKED."
    )
