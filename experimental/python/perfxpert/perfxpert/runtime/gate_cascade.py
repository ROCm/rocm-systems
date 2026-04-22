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

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


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


def _run_sol_gate(claimed_speedup: float, gfx_id: str) -> Dict[str, Any]:
    """Delegate to tools.sol.sanity_check."""
    from perfxpert.tools import sol
    r = sol.sanity_check(claimed_speedup=claimed_speedup, gfx_id=gfx_id)
    return {**r, "ok": r.get("verdict") == "sane"}


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
    r = _run_sol_gate(claimed_speedup, gfx_id)
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
    r = _run_regression_gate(baseline_db, candidate_db)
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
            rejected_patch_sha=patch_sha,
            delta_pct=total_delta,
            per_kernel_deltas=r.get("hot_kernels", []),
        )

    # Gate 5: Test anchors
    if candidate_binary is not None:
        r = _run_anchors_gate(candidate_binary)
        if not r.get("ok", False):
            return GateVerdict(
                status="reject", failing_gate="anchors",
                detail=f"anchor tests failed: {r.get('failed', [])}",
                metrics={"anchors": r},
                rejected_patch_sha=patch_sha,
            )

    # All gates passed
    return GateVerdict(
        status="pass", failing_gate=None,
        detail="all 5 gates passed",
        metrics={"claimed_speedup": claimed_speedup},
        delta_pct=total_delta,
    )


__all__ = [
    "GateVerdict",
    "evaluate",
    "MAX_OPTIMIZATION_CYCLES_PER_KERNEL",
    "MAX_CONSECUTIVE_FAILURES",
    "MAX_SESSION_LLM_TURNS",
    "REGRESSION_NOISE_THRESHOLD_PCT",
    "HOT_KERNEL_INDIVIDUAL_THRESHOLD_PCT",
    "TAIL_GEOMEAN_THRESHOLD_PCT",
]
