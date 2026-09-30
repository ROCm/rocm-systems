"""Attack #4 — silent regression on a hot kernel > 10% individually while
total wall-clock improves (optimizer traded a small kernel's win for a big
kernel's regression).

Exercises GATE 4 per-kernel regression branch.
"""

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade, KernelRuntime
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_regression_rejects_silent_10pct_hot_kernel(outcomes_dir) -> None:
    # Before: hot_kernel = 5 ms (50% of total 10 ms)
    # After: hot_kernel = 5.7 ms (14% slower than baseline individually)
    # But total improved because small_kernel went 5 ms → 3 ms
    baseline_runs = [
        KernelRuntime(kernel_name="hot_kernel", total_runtime_ns=5_000_000, share=0.50),
        KernelRuntime(kernel_name="small_kernel", total_runtime_ns=5_000_000, share=0.50),
    ]
    new_runs = [
        KernelRuntime(kernel_name="hot_kernel", total_runtime_ns=5_700_000, share=0.655),
        KernelRuntime(kernel_name="small_kernel", total_runtime_ns=3_000_000, share=0.345),
    ]

    gate_input = GateInput(
        kernel_name="workload",
        claimed_speedup=10.0 / 8.7,
        arch="gfx942",
        baseline_runtime_ns=10_000_000,
        achieved_runtime_ns=8_700_000,
        patch_sha="silentreg01",
        baseline_kernel_runtimes=baseline_runs,
        new_kernel_runtimes=new_runs,
    )

    verdict = run_gate_cascade(gate_input, stop_at="regression")
    assert verdict.status == "regressed"
    assert verdict.failing_gate == "regression"
    # FMEA fix: must surface the per-kernel delta
    assert verdict.per_kernel_deltas, "regression gate must expose per-kernel deltas"
    hot = [d for d in verdict.per_kernel_deltas if d["kernel_name"] == "hot_kernel"]
    assert hot, "hot_kernel delta missing from verdict"
    assert hot[0]["delta_pct"] > 0.10

    record_outcome(
        outcomes_dir,
        attack_id="silent_10pct_regression",
        status="defeated",
        details={
            "hot_kernel_delta_pct": hot[0]["delta_pct"],
            "verdict_status": verdict.status,
        },
    )
