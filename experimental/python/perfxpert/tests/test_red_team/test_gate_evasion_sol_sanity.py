"""Attack #1 — SOL sanity bound: fake 1000× speedup claim must be rejected.

Exercises runtime/gate_cascade.py GATE 2 directly.

Both tests exercise the UN-MOCKED _run_sol_gate integration path (no mocking of
gate internals).  run_gate_cascade() mocks _run_sol_gate at the module level, so
the second test calls evaluate() with compile/bitwise/regression mocked but sol
left to run for real — this is the path that was silently broken before the fix.
"""

import pytest
from unittest.mock import MagicMock

from perfxpert.runtime import gate_cascade
from perfxpert.runtime.gate_cascade import run_gate_cascade, GateInput, _run_sol_gate
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


@pytest.mark.red_team
def test_sol_gate_unmocked_rejects_impossible_speedup_ratio(outcomes_dir) -> None:
    """Red-team: call _run_sol_gate DIRECTLY (fully un-mocked) with claimed_speedup=1000×.

    Before the fix, _run_sol_gate raised TypeError (wrong kwargs) which was swallowed
    upstream, silently producing a false 'gate passed' verdict. This test would have
    FAILED (TypeError propagated as 'ok'=True) before the fix.
    """
    r = _run_sol_gate(claimed_speedup=1000.0, gfx_id="gfx942")
    assert r["ok"] is False, (
        "Un-mocked _run_sol_gate must REJECT claimed_speedup=1000× "
        f"(anti-Sakana hard cap is {gate_cascade.SOL_MAX_REASONABLE_SPEEDUP}×). Got: {r}"
    )

    record_outcome(
        outcomes_dir,
        attack_id="sol_gate_unmocked_1000x",
        status="defeated",
        details={"claimed_speedup": 1000.0, "ok": r["ok"], "reason": r.get("reason", "")},
    )


@pytest.mark.red_team
def test_sol_gate_unmocked_rejects_flops_exceeding_peak(outcomes_dir) -> None:
    """Red-team: supply achieved_flops_per_sec that exceeds MI300X fp64 hardware peak.

    This exercises the tier-2 absolute-FLOPS path of _run_sol_gate (un-mocked).
    A modest speedup ratio (3×) hides the exploit; only the FLOPS check catches it.
    """
    # MI300X fp64 peak = 81.7 TFLOPS; 500 TFLOPS is impossible
    r = _run_sol_gate(
        claimed_speedup=3.0,        # ratio looks innocent
        gfx_id="gfx942",
        achieved_flops_per_sec=500e12,   # absolute FLOPS exceed hardware peak
        kernel_type="fp64",
    )
    assert r["ok"] is False, (
        "500 TFLOPS fp64 exceeds MI300X peak (81.7 TFLOPS) — must reject. Got: {r}"
    )

    record_outcome(
        outcomes_dir,
        attack_id="sol_gate_unmocked_500tflops_fp64",
        status="defeated",
        details={
            "claimed_speedup": 3.0,
            "achieved_flops_per_sec": 500e12,
            "ok": r["ok"],
            "reason": r.get("reason", ""),
        },
    )
