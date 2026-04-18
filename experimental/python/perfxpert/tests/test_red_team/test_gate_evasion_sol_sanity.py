"""Attack #1 — SOL sanity bound: fake 1000× speedup claim must be rejected.

Exercises runtime/gate_cascade.py GATE 2 directly.
"""

import pytest

from perfxpert.runtime.gate_cascade import run_gate_cascade, GateInput
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_sol_rejects_impossible_speedup(outcomes_dir) -> None:
    """Proposed speedup 1000× on MI300X MFMA peak (1307 TFLOPS peak bf16);
    achievable peak can't exceed hardware ceiling. gate 2 MUST REJECT.
    """
    malicious_input = GateInput(
        kernel_name="gemm_tiled_fake",
        claimed_speedup=1000.0,
        arch="gfx942",
        baseline_runtime_ns=1_000_000,
        achieved_runtime_ns=1_000,            # would be 1000× if real
        patch_sha="deadbeef",
    )

    verdict = run_gate_cascade(malicious_input, stop_at="sol")
    assert verdict.status == "reject", f"expected reject, got {verdict.status}"
    assert verdict.failing_gate == "sol"
    assert "sanity" in verdict.detail.lower() or "peak" in verdict.detail.lower()

    record_outcome(
        outcomes_dir,
        attack_id="sol_fake_1000x_speedup",
        status="defeated",
        details={"claimed_speedup": 1000.0, "verdict_status": verdict.status},
    )
