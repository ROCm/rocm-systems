###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""trace_diff — baseline / new rocpd database diff as a READ_ONLY MCP tool.

Confluence row #7 (Trace Diff). Composes the existing Gate-4 primitives:

  * ``regression.compare_runs``               -> verdict + per-kernel deltas
  * ``regression.extract_kernel_runtimes_from_db`` -> baseline/new runtimes
  * ``regression.identify_hot_kernels``       -> Gate-4 "hot" kernel definition

Returns a shape tailored for the Webview / JSON diff report (row-at-a-time
delta bars + sortable speedup table) rather than the Gate-4 verdict shape
(which is optimized for the correctness cascade).

Shared brain across MCP / CLI / gate: every caller (``perfxpert diff``,
``perfxpert ci``, MCP clients, the ``--baseline`` splice in ``perfxpert
analyze``, and the new ``trace_diff_regression`` rule in
``runtime.gate_cascade``) invokes the same :func:`diff_runs`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class


_SCHEMA_VERSION = "0.3.1"


def _resolve_verdict_threshold() -> float:
    """Resolve the verdict threshold (% wall_delta_pct).

    Mirrors ``perfxpert.cli.ci_cmd.resolve_ci_threshold`` — the same
    ``PERFXPERT_CI_REGRESSION_THRESHOLD`` env var drives both the CI
    gate and the ``trace_diff.verdict`` field so users get consistent
    semantics across ``perfxpert ci``, ``perfxpert diff``, and
    ``perfxpert analyze --baseline``. Defaults to 5.0.
    """
    import os

    env = os.environ.get("PERFXPERT_CI_REGRESSION_THRESHOLD")
    if env:
        try:
            return float(env)
        except ValueError:
            return 5.0
    return 5.0


def _classify_verdict(
    wall_delta_pct: float,
    threshold_pct: float,
    regressions: List[Dict[str, Any]],
    improvements: List[Dict[str, Any]],
) -> str:
    """Bucket the diff into regressed / improved / neutral.

    Per-kernel regressions win over a neutral wall-time delta so callers
    do not silently miss a real hot-kernel slowdown when aggregate wall
    time is damped by unrelated improvements.
    """
    try:
        pct = float(wall_delta_pct)
    except (TypeError, ValueError):
        pct = 0.0
    thr = abs(float(threshold_pct))
    if regressions:
        return "regressed"
    if pct > thr:
        return "regressed"
    if pct < -thr:
        return "improved"
    if improvements:
        return "improved"
    return "neutral"


def _safe_pct(delta: float, baseline: float) -> float:
    """Percentage change vs baseline, guarded against div-by-zero."""
    if baseline <= 0:
        return 0.0
    return (delta / baseline) * 100.0


def _airgap_narrative(result: Dict[str, Any]) -> str:
    """Deterministic narrative when no LLM is attached.

    Structure: one-line headline + up to 3 regression bullets + up to 3
    improvement bullets. The CLI / webview formatter substitutes this
    when the user has not opted into an LLM provider.
    """
    wall_pct = result.get("wall_delta_pct", 0.0)
    regressions = result.get("primary_regressions", []) or []
    improvements = result.get("primary_improvements", []) or []

    if regressions:
        if wall_pct > 0.5:
            headline = (
                f"Runtime regressed by {wall_pct:+.2f}% "
                f"({len(regressions)} kernel regression(s) detected)."
            )
        else:
            headline = (
                f"Kernel-level regressions detected while wall time stayed within "
                f"noise ({wall_pct:+.2f}%)."
            )
    elif wall_pct < -0.5:
        headline = (
            f"Runtime improved by {wall_pct:+.2f}% "
            f"({len(improvements)} kernel improvement(s) detected)."
        )
    elif improvements:
        headline = (
            f"Kernel-level improvements detected while wall time stayed within "
            f"noise ({wall_pct:+.2f}%)."
        )
    elif wall_pct > 0.5:
        headline = (
            f"Runtime regressed by {wall_pct:+.2f}% "
            f"({len(regressions)} kernel regression(s) detected)."
        )
    else:
        headline = (
            f"Runtime within noise ({wall_pct:+.2f}%); "
            f"{len(regressions)} regression(s), {len(improvements)} improvement(s)."
        )

    lines = [headline]
    if regressions:
        lines.append("Primary regressions:")
        for r in regressions[:3]:
            lines.append(
                f"  - {r['name']}: {r['delta_pct']:+.1f}% "
                f"({r['baseline_ns']:,} ns → {r['new_ns']:,} ns)"
            )
    if improvements:
        lines.append("Primary improvements:")
        for r in improvements[:3]:
            lines.append(
                f"  - {r['name']}: {r['delta_pct']:+.1f}% "
                f"({r['baseline_ns']:,} ns → {r['new_ns']:,} ns)"
            )
    return "\n".join(lines)


@tool_class(ToolClass.READ_ONLY)
def diff_runs(
    baseline_db: str,
    new_db: str,
    top_kernels: int = 20,
) -> Dict[str, Any]:
    """Produce a structured diff between two rocpd databases.

    Parameters
    ----------
    baseline_db:
        Path to the baseline ``.db`` (before optimization / before fix).
    new_db:
        Path to the new ``.db`` (after change).
    top_kernels:
        How many kernels to include in the ``per_kernel`` list (sorted by
        baseline runtime, descending). The ``primary_regressions`` /
        ``primary_improvements`` lists are filtered separately.

    Returns
    -------
    dict with shape::

        {
          "schema_version": "0.3.1",
          "baseline_db": str, "new_db": str,
          "wall_delta_ns": int, "wall_delta_pct": float,
          "per_kernel": [{
              "name": str,
              "baseline_ns": int, "new_ns": int,
              "delta_ns": int, "delta_pct": float,
              "regressed": bool, "was_hot": bool,
          }, ...],
          "primary_regressions": [...]   # per_kernel entries, delta_pct > 3%
          "primary_improvements": [...]  # per_kernel entries, delta_pct < -3%
          "narrative": str,              # deterministic; caller may overwrite
        }
    """
    from perfxpert.tools import regression as _reg

    # Baseline + new runtime tables — reuse the existing extractor so this
    # module is a thin composition, not a duplicate query.
    baseline_runtimes = _reg.extract_kernel_runtimes_from_db(baseline_db)
    new_runtimes = _reg.extract_kernel_runtimes_from_db(new_db)
    baseline_map = {r.kernel_name: int(r.total_runtime_ns) for r in baseline_runtimes}
    new_map = {r.kernel_name: int(r.total_runtime_ns) for r in new_runtimes}

    hot_set = {k["name"] for k in _reg.identify_hot_kernels(baseline_db)}

    # Wall-time delta — sum of all kernel durations; robust against the
    # case where one run has extra / missing kernels.
    wall_before = sum(baseline_map.values())
    wall_after = sum(new_map.values())
    wall_delta_ns = wall_after - wall_before
    wall_delta_pct = _safe_pct(wall_delta_ns, wall_before)

    # Per-kernel delta — union of names, sorted by baseline runtime desc.
    union = sorted(
        set(baseline_map) | set(new_map),
        key=lambda n: baseline_map.get(n, 0),
        reverse=True,
    )
    all_kernel_deltas: List[Dict[str, Any]] = []
    for name in union:
        b = baseline_map.get(name, 0)
        a = new_map.get(name, 0)
        delta_ns = a - b
        delta_pct = _safe_pct(delta_ns, b) if b > 0 else (0.0 if a == 0 else 100.0)
        all_kernel_deltas.append(
            {
                "name": name,
                "baseline_ns": int(b),
                "new_ns": int(a),
                "delta_ns": int(delta_ns),
                "delta_pct": round(float(delta_pct), 4),
                "regressed": delta_pct > 3.0,
                "was_hot": name in hot_set,
            }
        )
    per_kernel = all_kernel_deltas[: max(int(top_kernels), 0)]

    # Primary regressions / improvements — thresholds mirror Gate-4 noise
    # tolerance (3%). These are derived from the full kernel union, not
    # the display-limited ``per_kernel`` slice, so callers cannot lose
    # true regressions merely by asking for a short top-K table.
    regressions = sorted(
        (k for k in all_kernel_deltas if k["delta_pct"] > 3.0),
        key=lambda k: k["delta_pct"],
        reverse=True,
    )
    improvements = sorted(
        (k for k in all_kernel_deltas if k["delta_pct"] < -3.0),
        key=lambda k: k["delta_pct"],
    )

    # Verdict field — bucket the wall delta against the CI threshold so
    # every downstream consumer (perfxpert analyze --baseline JSON output,
    # perfxpert diff JSON output, MCP trace_diff_diff_runs) gets a single,
    # consistent pass/fail signal. Threshold is resolved from
    # ``PERFXPERT_CI_REGRESSION_THRESHOLD`` (default 5.0) to match
    # ``perfxpert ci``.
    _verdict_threshold = _resolve_verdict_threshold()
    _verdict = _classify_verdict(
        wall_delta_pct,
        _verdict_threshold,
        regressions,
        improvements,
    )

    result: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "baseline_db": baseline_db,
        "new_db": new_db,
        "wall_delta_ns": int(wall_delta_ns),
        "wall_delta_pct": round(float(wall_delta_pct), 4),
        "verdict": _verdict,
        "verdict_threshold_pct": round(float(_verdict_threshold), 4),
        "per_kernel": per_kernel,
        "primary_regressions": regressions,
        "primary_improvements": improvements,
        "narrative": "",
    }
    # Fill in deterministic narrative; LLM-aware callers can overwrite
    # before rendering.
    result["narrative"] = _airgap_narrative(result)
    return result


__all__ = ["diff_runs"]
