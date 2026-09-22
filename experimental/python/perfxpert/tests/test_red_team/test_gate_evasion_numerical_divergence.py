"""Attack #3 — numerical divergence: patch compiles and runs but outputs diverge
beyond np.allclose tolerance.

Exercises GATE 3 (patch.verify_output).
"""

import pytest
import numpy as np

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_bitwise_rejects_numerical_divergence(outcomes_dir) -> None:
    # Simulate verify_output receiving baseline + new array pair that diverge
    baseline = np.linspace(0, 1, 1024, dtype=np.float32)
    new = baseline.copy()
    new[10] += 1e-1  # way beyond allclose rtol=1e-5

    gate_input = GateInput(
        kernel_name="divergent_kernel",
        claimed_speedup=1.15,
        arch="gfx942",
        baseline_runtime_ns=1_000_000,
        achieved_runtime_ns=870_000,
        patch_sha="divergent01",
        verify_output_baseline=baseline,
        verify_output_new=new,
    )

    verdict = run_gate_cascade(gate_input, stop_at="bitwise")
    assert verdict.status == "reject"
    assert verdict.failing_gate == "bitwise"

    record_outcome(
        outcomes_dir,
        attack_id="numerical_divergence",
        status="defeated",
        details={
            "max_abs_diff": float(np.max(np.abs(baseline - new))),
            "verdict_status": verdict.status,
        },
    )
