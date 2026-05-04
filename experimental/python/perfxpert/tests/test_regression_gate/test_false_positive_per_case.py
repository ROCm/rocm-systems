"""Per-case tests — one test per proven-optimization seed.

Separate from the aggregate so CI failures show WHICH case regressed the gate.
"""

import pytest

from tests.test_regression_gate.proven_optimization_runner import (
    ProvenOptimizationRunner,
    ProvenOptimizationCase,
)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        runner = ProvenOptimizationRunner()
        cases = runner.load_seed_cases()
        if not cases:
            # No fixtures; parametrize with a skip marker
            metafunc.parametrize("case", [], ids=[])
        else:
            metafunc.parametrize("case", cases, ids=[c.case_id for c in cases])


@pytest.mark.regression_gate
def test_each_proven_case_passes_gate_cascade(
    proven_runner: ProvenOptimizationRunner, case: ProvenOptimizationCase
) -> None:
    if case is None:
        pytest.skip("proven_optimizations fixture not present")

    verdict = proven_runner.run_on_case(case)
    assert verdict.status == "pass", (
        f"Proven optimization {case.case_id!r} was REJECTED by the gate "
        f"(gate={verdict.failing_gate}, detail={verdict.detail!r}).\n"
        f"This is a false positive — the gate blocks a known-good technique."
    )
