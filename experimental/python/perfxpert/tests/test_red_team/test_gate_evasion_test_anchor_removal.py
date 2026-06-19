"""Attack #6 — patch that DELETES a passing test to make it 'pass'.

GATE 5 (anchors.check) must detect that a previously-passing test is no longer
in the test suite and REJECT.
"""

from pathlib import Path
import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_anchors_reject_test_removal(outcomes_dir, tmp_path: Path) -> None:
    test_suite_before = tmp_path / "tests"
    test_suite_before.mkdir()
    (test_suite_before / "test_correctness.py").write_text(
        "def test_baseline(): assert 1 + 1 == 2\n"
    )
    # Simulate a patch whose effect is to remove test_correctness.py
    anchor_baseline = {"test_baseline": "passed"}
    anchor_new: dict = {}  # test was deleted

    gate_input = GateInput(
        kernel_name="any",
        claimed_speedup=1.1,
        arch="gfx942",
        baseline_runtime_ns=1_000_000,
        achieved_runtime_ns=900_000,
        patch_sha="testremoval01",
        test_anchor_baseline=anchor_baseline,
        test_anchor_new=anchor_new,
    )

    verdict = run_gate_cascade(gate_input, stop_at="anchors")
    assert verdict.status == "reject"
    assert verdict.failing_gate == "anchors"

    record_outcome(
        outcomes_dir,
        attack_id="test_anchor_removal",
        status="defeated",
        details={"removed_tests": ["test_baseline"]},
    )
