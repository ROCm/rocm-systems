"""Attack #5 — silent tail regression: many small kernels each < 10% regress.
Weighted-geomean branch of GATE 4 must catch it.

Design-review FMEA fix: hot kernels = top-K covering 80% cumulative runtime
∪ any kernel ≥ 3% individually. Weighted-geomean across hot kernels flags this.
"""

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade, KernelRuntime
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_regression_catches_silent_tail_via_weighted_geomean(outcomes_dir) -> None:
    # 10 kernels at 8% each (all hot by ≥3% individually, cumulative 80%).
    # Each regresses 6% — individually invisible to a 10% per-kernel rule.
    baseline_runs = [
        KernelRuntime(kernel_name=f"k{i}", total_runtime_ns=800_000, share=0.08)
        for i in range(10)
    ]
    # Plus one non-hot kernel at 20%
    baseline_runs.append(
        KernelRuntime(kernel_name="cold", total_runtime_ns=2_000_000, share=0.20)
    )
    # After: each hot kernel 7% slower; cold kernel 10% faster
    # 7% per-kernel regresses to ~5.6% weighted-geomean (exceeds 5% threshold)
    new_runs = [
        KernelRuntime(kernel_name=f"k{i}", total_runtime_ns=856_000, share=0.08)
        for i in range(10)
    ]
    new_runs.append(
        KernelRuntime(kernel_name="cold", total_runtime_ns=1_800_000, share=0.18)
    )

    gate_input = GateInput(
        kernel_name="workload",
        claimed_speedup=1.005,
        arch="gfx942",
        baseline_runtime_ns=10_000_000,
        achieved_runtime_ns=10_360_000,  # actually regressed slightly (7% on hot kernels)
        patch_sha="tailreg01",
        baseline_kernel_runtimes=baseline_runs,
        new_kernel_runtimes=new_runs,
    )

    verdict = run_gate_cascade(gate_input, stop_at="regression")
    assert verdict.status == "regressed"
    assert verdict.failing_gate == "regression"
    # Verify the weighted geomean signal specifically
    assert "weighted" in verdict.detail.lower() or "geomean" in verdict.detail.lower()

    record_outcome(
        outcomes_dir,
        attack_id="silent_tail_regression_weighted_geomean",
        status="defeated",
        details={"verdict_detail": verdict.detail},
    )
