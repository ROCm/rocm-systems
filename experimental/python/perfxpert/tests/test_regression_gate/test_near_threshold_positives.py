"""Additional R17 coverage: near-threshold positive-example tests.

The 10 seed cases are at the "measured_impact_min" end of their ranges. The
regression gate's NOISE_THRESHOLD is 3%. We want to confirm that:
  * A 3.5% improvement (just above noise) still PASSES (not false-rejected)
  * A 5% improvement with one 8% kernel regression still PASSES (not blocked)
    because 8% < 10% per-kernel rule AND weighted-geomean doesn't tank
"""

import pytest

from perfxpert.runtime.gate_cascade import GateInput, run_gate_cascade
from perfxpert.tools.regression import KernelRuntime


@pytest.mark.regression_gate
def test_3_5_pct_improvement_above_noise_passes() -> None:
    """Small but valid improvement just above noise threshold."""
    baseline_runs = [KernelRuntime(kernel_name="only", total_runtime_ns=10_000_000, share=1.0)]
    new_runs = [KernelRuntime(kernel_name="only", total_runtime_ns=9_650_000, share=1.0)]

    gate_input = GateInput(
        kernel_name="workload",
        claimed_speedup=10.0 / 9.65,
        arch="gfx942",
        baseline_runtime_ns=10_000_000,
        achieved_runtime_ns=9_650_000,
        patch_sha="near_thresh_01",
        baseline_kernel_runtimes=baseline_runs,
        new_kernel_runtimes=new_runs,
        skip_compile=True, skip_bitwise=True, skip_anchors=True,
    )
    verdict = run_gate_cascade(gate_input)
    assert verdict.status == "pass", (
        f"3.5% improvement above noise was FALSE-rejected: {verdict.detail}"
    )


@pytest.mark.regression_gate
def test_5_pct_win_with_sub_10_pct_kernel_regression_passes() -> None:
    """5% total improvement; one non-hot kernel regressed 8% (< 10% rule)."""
    baseline_runs = [
        KernelRuntime(kernel_name="hot", total_runtime_ns=8_000_000, share=0.80),
        KernelRuntime(kernel_name="tail", total_runtime_ns=2_000_000, share=0.20),
    ]
    new_runs = [
        KernelRuntime(kernel_name="hot", total_runtime_ns=7_300_000, share=0.77),
        KernelRuntime(kernel_name="tail", total_runtime_ns=2_160_000, share=0.23),  # +8%
    ]

    gate_input = GateInput(
        kernel_name="workload",
        claimed_speedup=10.0 / 9.46,
        arch="gfx942",
        baseline_runtime_ns=10_000_000,
        achieved_runtime_ns=9_460_000,
        patch_sha="near_thresh_02",
        baseline_kernel_runtimes=baseline_runs,
        new_kernel_runtimes=new_runs,
        skip_compile=True, skip_bitwise=True, skip_anchors=True,
    )
    verdict = run_gate_cascade(gate_input)
    assert verdict.status == "pass", (
        f"5% win with 8% non-hot kernel regression was FALSE-rejected: {verdict.detail}"
    )
