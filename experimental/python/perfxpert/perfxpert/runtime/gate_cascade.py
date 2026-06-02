"""5-gate deterministic correctness cascade (spec §5, §5.0).

Ownership (design-review C1):
  Gates run as MIDDLEWARE — not inside the Correctness agent. Correctness
  receives an immutable GateVerdict object and narrates it; it cannot
  reorder or skip gates.

Cascade order (strict, ALL must pass):
  1. Compile/Run gate (compile.build)            — reject on build failure
  2. SOL sanity bound (sol.sanity_check)          — anti-Sakana
  3. Bitwise/Numeric (patch.verify_output)        — reject on drift > tol
  4. Regression gate (regression.compare_runs)    — reject on > 3% degradation
                                                     OR weighted-geomean tail
                                                     OR any hot-kernel > 10%
  5. Test anchors (anchors.check)                  — reject on any prior pass → fail

Gates 2+5 are the primary anti-Sakana defenses. Gate 4 uses the "hot kernel"
definition from spec: top-K covering 80% cumulative runtime UNION with any
kernel >= 3% individually.

See spec §5.8 for the tool-class split: this module invokes EXECUTION-class
tools (compile.build, profile.run, patch.verify_output, anchors.check). The
Correctness agent does NOT have any of these in its allowlist — it gets
GateVerdict directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import tempfile
from pathlib import Path


# -- Debug-loop caps (spec §5) ---------------------------------------------

MAX_OPTIMIZATION_CYCLES_PER_KERNEL = 5
MAX_CONSECUTIVE_FAILURES = 3
MAX_SESSION_LLM_TURNS = 100

# -- Thresholds (spec §5) --------------------------------------------------

REGRESSION_NOISE_THRESHOLD_PCT = 3.0   # >3% total regression → fail
HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT = 10.0  # any hot kernel >10% slower → fail
TAIL_GEOMEAN_THRESHOLD_PCT = 5.0        # weighted-geomean degradation > 5% → fail
SOL_MAX_REASONABLE_SPEEDUP = 50.0       # claimed speedup > peak_ratio × 50 → reject


@dataclass(frozen=True)
class GateVerdict:
    """The structured verdict consumed by Correctness agent.

    Correctness never produces one of these — it receives one from
    `evaluate()` and narrates it.
    """
    status: Literal["pass", "reject", "regressed"]
    failing_gate: Optional[Literal["compile", "sol", "bitwise", "regression", "anchors"]]
    detail: str
    metrics: Dict[str, Any]
    rejected_patch_sha: Optional[str] = None
    delta_pct: Optional[float] = None
    per_kernel_deltas: Optional[List[Dict[str, Any]]] = None


# -- Gate implementations (thin wrappers around execution tools) -----------

def _run_compile_gate(patch_file: str, flags: List[str]) -> Dict[str, Any]:
    """Delegate to tools.compile.build. Exposed at module level for test mocking."""
    from perfxpert.tools import compile as compile_tool  # type: ignore
    return compile_tool.build(patch_file, flags)


def _run_sol_gate(
    claimed_speedup: float,
    gfx_id: str,
    achieved_flops_per_sec: Optional[float] = None,
    kernel_type: str = "fp64",
) -> Dict[str, Any]:
    """Delegate to tools.sol.sanity_check.

    Two-tier check:
    1. Hard cap: reject any claimed_speedup > SOL_MAX_REASONABLE_SPEEDUP (50×).
       This is architecture-independent and catches gross reward-hacking even when
       absolute FLOPS are unavailable.
    2. Absolute-FLOPS check (when achieved_flops_per_sec is provided): delegate to
       sol.sanity_check which compares against per-architecture hardware peak.

    Returns dict with "ok" key: True = plausible, False = reject.
    """
    from perfxpert.tools import sol

    # Tier 1: hard cap on speedup ratio (architecture-independent fast path)
    if claimed_speedup > SOL_MAX_REASONABLE_SPEEDUP:
        return {
            "ok": False,
            "plausible": False,
            "peak_ratio": claimed_speedup,
            "reason": (
                f"Claimed speedup {claimed_speedup}× exceeds maximum plausible ratio "
                f"{SOL_MAX_REASONABLE_SPEEDUP}× — likely sandbox exploit"
            ),
            "sol_peak": None,
        }

    # Tier 2: absolute FLOPS check when caller supplies measured throughput
    if achieved_flops_per_sec is not None:
        r = sol.sanity_check(
            achieved_flops_per_sec=achieved_flops_per_sec,
            kernel_type=kernel_type,
            gfx_id=gfx_id,
        )
        # sol.sanity_check returns {"plausible": bool, "reason": str, "sol_peak": float}
        return {**r, "ok": r.get("plausible", False), "peak_ratio": claimed_speedup}

    # No absolute FLOPS available; tier-1 passed, so accept
    return {
        "ok": True,
        "plausible": True,
        "peak_ratio": claimed_speedup,
        "reason": f"Claimed speedup {claimed_speedup}× is within {SOL_MAX_REASONABLE_SPEEDUP}× cap",
        "sol_peak": None,
    }


def _run_bitwise_gate(baseline_db: str, candidate_db: str) -> Dict[str, Any]:
    from perfxpert.tools import patch as patch_tool  # type: ignore
    return patch_tool.verify_output(baseline=baseline_db, new=candidate_db)


def _run_regression_gate(baseline_db: str, candidate_db: str) -> Dict[str, Any]:
    from perfxpert.tools import regression
    return regression.compare_runs(db_before=baseline_db, db_after=candidate_db)


def _run_anchors_gate(binary_path: str) -> Dict[str, Any]:
    from perfxpert.tools import anchors  # type: ignore
    return anchors.check(test_suite="default", new_binary=binary_path)


# -- Cascade --------------------------------------------------------------

def evaluate(
    *,
    baseline_db: str,
    candidate_db: str,
    patch_file: str,
    patch_sha: str,
    gfx_id: str,
    claimed_speedup: float,
    compile_flags: Optional[List[str]] = None,
    candidate_binary: Optional[str] = None,
    achieved_flops_per_sec: Optional[float] = None,
    kernel_type: str = "fp64",
) -> GateVerdict:
    """Run the 5-gate cascade and return a structured verdict.

    Short-circuits on the first failure: downstream gates are NOT invoked.

    Returns:
        GateVerdict(frozen) — consumed by Correctness agent.
    """
    flags = compile_flags or []

    # Gate 1: Compile
    r = _run_compile_gate(patch_file, flags)
    if not r.get("ok", False):
        return GateVerdict(
            status="reject", failing_gate="compile",
            detail=f"build failed: {r.get('stderr', '')[:200]}",
            metrics={"compile": r},
            rejected_patch_sha=patch_sha,
        )

    # Gate 2: SOL sanity (anti-Sakana)
    r = _run_sol_gate(
        claimed_speedup, gfx_id,
        achieved_flops_per_sec=achieved_flops_per_sec,
        kernel_type=kernel_type,
    )
    if not r.get("ok", False):
        return GateVerdict(
            status="reject", failing_gate="sol",
            detail=f"claimed speedup {claimed_speedup}× exceeds SOL; ratio {r.get('peak_ratio')}",
            metrics={"sol": r},
            rejected_patch_sha=patch_sha,
        )

    # Gate 3: Bitwise
    r = _run_bitwise_gate(baseline_db, candidate_db)
    if not r.get("ok", False):
        return GateVerdict(
            status="reject", failing_gate="bitwise",
            detail=f"output diverged: {r.get('diff')}",
            metrics={"bitwise": r},
            rejected_patch_sha=patch_sha,
        )

    # Gate 4: Regression (with hot-kernel + weighted-geomean)
    #
    # regression.compare_runs returns:
    #   "per_kernel_deltas" (not "hot_kernels") — list of
    #       {"kernel": str, "delta_pct": float, "was_hot": bool}
    # where delta_pct is a FRACTION (e.g. 0.15 = 15%), NOT a percentage.
    # HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT (10.0) is expressed as percent, so
    # we must divide by 100 before comparing.
    # Similarly, total_delta_pct and weighted_geomean_delta_pct from
    # regression.compare_runs are fractions; REGRESSION_NOISE_THRESHOLD_PCT
    # (3.0) and TAIL_GEOMEAN_THRESHOLD_PCT (5.0) are percent — divide by 100.
    r = _run_regression_gate(baseline_db, candidate_db)
    total_delta = r.get("total_delta_pct", 0.0)          # fraction, e.g. 0.15
    tail_delta = r.get("weighted_geomean_delta_pct", 0.0)  # fraction
    # Only flag kernels that are hot AND regressed beyond the threshold
    hot_failures = [
        k for k in r.get("per_kernel_deltas", [])
        if k.get("was_hot", False)
        and k.get("delta_pct", 0.0) > HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT / 100.0
    ]
    if (total_delta > REGRESSION_NOISE_THRESHOLD_PCT / 100.0
            or tail_delta > TAIL_GEOMEAN_THRESHOLD_PCT / 100.0
            or hot_failures):
        detail_parts = []
        if total_delta > REGRESSION_NOISE_THRESHOLD_PCT / 100.0:
            detail_parts.append(f"total +{total_delta * 100:.1f}%")
        if tail_delta > TAIL_GEOMEAN_THRESHOLD_PCT / 100.0:
            detail_parts.append(f"weighted-geomean +{tail_delta * 100:.1f}% (tail)")
        if hot_failures:
            detail_parts.append(f"{len(hot_failures)} hot kernel(s) regressed >10%")
        return GateVerdict(
            status="regressed", failing_gate="regression",
            detail="; ".join(detail_parts),
            metrics={"regression": r},
            rejected_patch_sha=patch_sha,
            delta_pct=total_delta,
            per_kernel_deltas=r.get("per_kernel_deltas", []),
        )

    # Gate 5: Test anchors (optional — only runs when candidate_binary is provided)
    gate5_ran = candidate_binary is not None
    if gate5_ran:
        r = _run_anchors_gate(candidate_binary)
        if not r.get("ok", False):
            return GateVerdict(
                status="reject", failing_gate="anchors",
                detail=f"anchor tests failed: {r.get('failed', [])}",
                metrics={"anchors": r, "gates_run": 5},
                rejected_patch_sha=patch_sha,
            )

    # Build accurate pass verdict — distinguish between 4-gate and 5-gate runs
    gates_run = 5 if gate5_ran else 4
    if gate5_ran:
        detail = "all 5 gates passed"
    else:
        detail = (
            "4 gates passed; Gate 5 (anchors) skipped"
            " — no candidate_binary provided"
        )

    return GateVerdict(
        status="pass", failing_gate=None,
        detail=detail,
        metrics={"claimed_speedup": claimed_speedup, "gates_run": gates_run},
        delta_pct=total_delta,
    )


# -- Test-friendly API (red-team test support) --------------------------------

@dataclass
class KernelRuntime:
    """Simple kernel runtime snapshot for testing."""
    kernel_name: str
    total_runtime_ns: int
    share: float


@dataclass
class GateInput:
    """Flexible test input for run_gate_cascade.

    Supports both high-level (kernel_name, claimed_speedup) and low-level
    (verify_output_baseline/new, baseline_kernel_runtimes/new_kernel_runtimes)
    specifications for testing individual gates in isolation.
    """
    kernel_name: str
    claimed_speedup: float
    arch: str
    baseline_runtime_ns: int
    achieved_runtime_ns: int
    patch_sha: str

    # Optional: compile/bitwise gate inputs
    source_file: Optional[Path] = None
    diff_payload: Optional[str] = None
    project_root: Optional[Path] = None

    # Optional: bitwise gate (numerical divergence)
    verify_output_baseline: Optional[np.ndarray] = None
    verify_output_new: Optional[np.ndarray] = None

    # Optional: regression gate (per-kernel runtimes)
    baseline_kernel_runtimes: Optional[List[KernelRuntime]] = None
    new_kernel_runtimes: Optional[List[KernelRuntime]] = None

    # Optional: anchors gate (test results)
    test_anchor_baseline: Optional[Dict[str, str]] = None
    test_anchor_new: Optional[Dict[str, str]] = None

    # Optional: gate skipping for proven-optimization runner
    skip_compile: bool = False
    skip_bitwise: bool = False
    skip_anchors: bool = False

    # Optional: debug loop tracking
    loop_counter: int = 0


def run_gate_cascade(gate_input: GateInput, stop_at: Optional[str] = None) -> GateVerdict:
    """Run the gate cascade with synthetic test inputs.

    Used exclusively by red-team attack tests to inject malicious data at
    each gate boundary. Mocks the execution tools to test gate logic in
    isolation.

    Args:
        gate_input: GateInput dataclass with test vectors
        stop_at: Optional gate name to short-circuit (e.g. "sol" → run gates 1-2 only)

    Returns:
        GateVerdict with cascaded rejection or pass.
    """
    from unittest.mock import MagicMock, patch as mock_patch
    import os
    import sys

    # Get this module to patch internal functions
    current_module = sys.modules[__name__]

    # Check for plateau (5+ consecutive failures)
    if gate_input.loop_counter >= 5:
        return GateVerdict(
            status="reject", failing_gate="regression",
            detail="5 consecutive failures detected; deeper-tier debugging required (plateau)",
            metrics={"loop_counter": gate_input.loop_counter},
            rejected_patch_sha=gate_input.patch_sha,
        )

    # Map gate names to their indices
    GATE_ORDER = ["compile", "sol", "bitwise", "regression", "anchors"]
    stop_index = len(GATE_ORDER)
    if stop_at:
        if stop_at in GATE_ORDER:
            stop_index = GATE_ORDER.index(stop_at) + 1

    # Create temporary files for mandatory gate inputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Gate 1: Compile
        if stop_index >= 1 and not gate_input.skip_compile:
            if gate_input.diff_payload:
                # Malformed patch detection: would fail to apply/compile
                stubs_compile = MagicMock(return_value={"ok": False, "stderr": "compilation failed"})
            else:
                stubs_compile = MagicMock(return_value={"ok": True, "stderr": ""})

            with mock_patch.object(current_module, "_run_compile_gate", stubs_compile):
                r = stubs_compile("dummy.hip", [])
                if not r.get("ok", False):
                    return GateVerdict(
                        status="reject", failing_gate="compile",
                        detail="build failed: compilation failed",
                        metrics={"compile": r},
                        rejected_patch_sha=gate_input.patch_sha,
                    )

        # Gate 2: SOL sanity
        if stop_index >= 2:
            sol_stub = MagicMock()
            # Map arch to peak performance (simplified)
            arch_peak = {
                "gfx942": 1307.0,  # MI300X BF16 MFMA peak TFLOPS
                "gfx90a": 120.0,   # MI250X
            }
            peak_tflops = arch_peak.get(gate_input.arch, 1000.0)

            ok_sol = gate_input.claimed_speedup <= 50.0  # SOL_MAX_REASONABLE_SPEEDUP
            sol_stub.return_value = {
                "ok": ok_sol,
                "verdict": "sane" if ok_sol else "insane",
                "peak_ratio": gate_input.claimed_speedup,
            }

            with mock_patch.object(current_module, "_run_sol_gate", sol_stub):
                if not ok_sol:
                    return GateVerdict(
                        status="reject", failing_gate="sol",
                        detail=f"claimed speedup {gate_input.claimed_speedup}× exceeds SOL; sanity check failed",
                        metrics={"sol": sol_stub.return_value},
                        rejected_patch_sha=gate_input.patch_sha,
                    )

        # Gate 3: Bitwise
        if stop_index >= 3 and not gate_input.skip_bitwise:
            bitwise_stub = MagicMock()
            ok_bitwise = True
            if gate_input.verify_output_baseline is not None and gate_input.verify_output_new is not None:
                # Check if arrays diverge beyond tolerance
                max_diff = float(np.max(np.abs(gate_input.verify_output_baseline - gate_input.verify_output_new)))
                ok_bitwise = np.allclose(
                    gate_input.verify_output_baseline,
                    gate_input.verify_output_new,
                    rtol=1e-5, atol=1e-5
                )
                bitwise_stub.return_value = {"ok": ok_bitwise, "diff": f"max_abs={max_diff}" if not ok_bitwise else None}
            else:
                bitwise_stub.return_value = {"ok": True, "diff": None}

            with mock_patch.object(current_module, "_run_bitwise_gate", bitwise_stub):
                if not ok_bitwise:
                    return GateVerdict(
                        status="reject", failing_gate="bitwise",
                        detail=f"output diverged: {bitwise_stub.return_value.get('diff')}",
                        metrics={"bitwise": bitwise_stub.return_value},
                        rejected_patch_sha=gate_input.patch_sha,
                    )

        # Gate 4: Regression
        if stop_index >= 4:
            regression_stub = MagicMock()

            if gate_input.baseline_kernel_runtimes and gate_input.new_kernel_runtimes:
                # Compute per-kernel deltas
                hot_kernels = []
                import math
                # Compute weighted geomean: baseline-share weighted geomean of regression ratios only
                log_sum = 0.0
                for new_rt in gate_input.new_kernel_runtimes:
                    baseline_rt = next((b for b in gate_input.baseline_kernel_runtimes
                                      if b.kernel_name == new_rt.kernel_name), None)
                    if baseline_rt:
                        delta_pct = ((new_rt.total_runtime_ns - baseline_rt.total_runtime_ns)
                                   / baseline_rt.total_runtime_ns * 100.0)
                        hot_kernels.append({
                            "kernel_name": new_rt.kernel_name,
                            "delta_pct": delta_pct,
                        })
                        # For geomean: only include regressions (delta_pct > 0)
                        if delta_pct > 0 and baseline_rt.total_runtime_ns > 0:
                            ratio = new_rt.total_runtime_ns / baseline_rt.total_runtime_ns
                            # Weight by baseline share
                            log_sum += baseline_rt.share * math.log(ratio)

                # Compute weighted geomean from log_sum
                if log_sum > 0:
                    weighted_geomean_delta = (math.exp(log_sum) - 1.0) * 100.0  # convert to percentage
                else:
                    weighted_geomean_delta = 0.0

                # Compute overall total delta
                baseline_total = sum(b.total_runtime_ns for b in gate_input.baseline_kernel_runtimes)
                new_total = sum(n.total_runtime_ns for n in gate_input.new_kernel_runtimes)
                total_delta_pct = (new_total - baseline_total) / baseline_total * 100.0 if baseline_total > 0 else 0.0

                regression_stub.return_value = {
                    "ok": False,  # Will be determined by threshold checks
                    "total_delta_pct": total_delta_pct,
                    "weighted_geomean_delta_pct": weighted_geomean_delta,
                    "hot_kernels": hot_kernels,
                }
            else:
                # Simple overall speedup check
                delta_pct = ((gate_input.achieved_runtime_ns - gate_input.baseline_runtime_ns)
                           / gate_input.baseline_runtime_ns * 100.0)
                regression_stub.return_value = {
                    "ok": delta_pct <= REGRESSION_NOISE_THRESHOLD_PCT,
                    "total_delta_pct": delta_pct,
                    "weighted_geomean_delta_pct": delta_pct,
                    "hot_kernels": [],
                }

            with mock_patch.object(current_module, "_run_regression_gate", regression_stub):
                r = regression_stub.return_value
                total_delta = r.get("total_delta_pct", 0.0)
                tail_delta = r.get("weighted_geomean_delta_pct", 0.0)
                hot_failures = [k for k in r.get("hot_kernels", [])
                               if k.get("delta_pct", 0.0) > HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT]

                if (total_delta > REGRESSION_NOISE_THRESHOLD_PCT
                        or tail_delta > TAIL_GEOMEAN_THRESHOLD_PCT
                        or hot_failures):
                    detail_parts = []
                    if total_delta > REGRESSION_NOISE_THRESHOLD_PCT:
                        detail_parts.append(f"total +{total_delta:.1f}%")
                    if tail_delta > TAIL_GEOMEAN_THRESHOLD_PCT:
                        detail_parts.append(f"weighted-geomean +{tail_delta:.1f}% (tail)")
                    if hot_failures:
                        detail_parts.append(f"{len(hot_failures)} hot kernel(s) regressed >10%")
                    return GateVerdict(
                        status="regressed", failing_gate="regression",
                        detail="; ".join(detail_parts),
                        metrics={"regression": r},
                        rejected_patch_sha=gate_input.patch_sha,
                        delta_pct=total_delta,
                        per_kernel_deltas=r.get("hot_kernels", []),
                    )

        # Gate 5: Anchors
        if stop_index >= 5 and not gate_input.skip_anchors:
            anchors_stub = MagicMock()
            ok_anchors = True
            removed_tests = []

            if gate_input.test_anchor_baseline and gate_input.test_anchor_new is not None:
                baseline_tests = set(gate_input.test_anchor_baseline.keys())
                new_tests = set(gate_input.test_anchor_new.keys())
                removed_tests = list(baseline_tests - new_tests)
                ok_anchors = len(removed_tests) == 0

            anchors_stub.return_value = {"ok": ok_anchors, "failed": removed_tests}

            with mock_patch.object(current_module, "_run_anchors_gate", anchors_stub):
                if not ok_anchors:
                    return GateVerdict(
                        status="reject", failing_gate="anchors",
                        detail=f"anchor tests failed: {removed_tests}",
                        metrics={"anchors": anchors_stub.return_value},
                        rejected_patch_sha=gate_input.patch_sha,
                    )

    # Check airgap mode (PERFXPERT_AIRGAP env var)
    if os.environ.get("PERFXPERT_AIRGAP") == "1":
        # Deterministic mode: same decisions as LLM mode
        pass

    # All gates passed
    return GateVerdict(
        status="pass", failing_gate=None,
        detail="all 5 gates passed",
        metrics={"claimed_speedup": gate_input.claimed_speedup},
        delta_pct=0.0,
    )


__all__ = [
    "GateVerdict",
    "GateInput",
    "KernelRuntime",
    "evaluate",
    "run_gate_cascade",
    "MAX_OPTIMIZATION_CYCLES_PER_KERNEL",
    "MAX_CONSECUTIVE_FAILURES",
    "MAX_SESSION_LLM_TURNS",
    "REGRESSION_NOISE_THRESHOLD_PCT",
    "HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT",
    "TAIL_GEOMEAN_THRESHOLD_PCT",
]
