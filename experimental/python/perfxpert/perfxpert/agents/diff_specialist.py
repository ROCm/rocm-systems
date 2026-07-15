"""Trace-Diff Specialist (Layer 2).

Compares a baseline rocprofiler-sdk database against a new one and
synthesises a structured verdict (improved / regressed / neutral) plus a
short narrative. Delegates the arithmetic to
:func:`perfxpert.tools.trace_diff.diff_runs`; this agent owns the
ranking / classification / narrative-synthesis layer.

Tool allowlist (3 of 5 cap used):
  trace_diff.diff_runs, regression.compare_runs, roofline.classify

Layer-2 rule: no Layer-2 → Layer-2 handoffs; returns to Recommendation.
Air-gap fallback: sort regressions/improvements by |delta_pct|,
synthesise the deterministic narrative template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import regression, roofline, trace_diff


_FENCE_PATH = Path(__file__).parent / "fence" / "diff_specialist.md"


def build_diff_specialist() -> Agent:
    tools = [
        ToolBinding(name="trace_diff.diff_runs", fn=trace_diff.diff_runs),
        ToolBinding(name="regression.compare_runs", fn=regression.compare_runs),
        ToolBinding(name="roofline.classify", fn=roofline.classify),
    ]
    return Agent(
        name="TraceDiffSpecialist",
        layer=2,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.DiffSpecialistInput,
        output_schema=schemas.DiffSpecialistOutput,
        tools=tools,
        allowed_handoffs=[],
        token_budget=3072,
    )


def _classify_verdict(diff_result: Dict[str, Any]) -> str:
    """Map the trace_diff dict to a three-way verdict string."""
    wall = float(diff_result.get("wall_delta_pct", 0.0) or 0.0)
    regressions = diff_result.get("primary_regressions", []) or []
    if regressions:
        return "regressed"
    if wall > 0.5:
        return "regressed"
    if wall < -0.5:
        return "improved"
    return "neutral"


def _airgap_narrative(diff_result: Dict[str, Any]) -> str:
    """Deterministic one-line summary used under ``PERFXPERT_AIRGAP=1``."""
    wall = float(diff_result.get("wall_delta_pct", 0.0) or 0.0)
    regressions = diff_result.get("primary_regressions", []) or []
    improvements = diff_result.get("primary_improvements", []) or []
    n_reg = len(regressions)
    n_imp = len(improvements)
    if regressions:
        top = regressions[0]
        top_part = (
            f" Top regression: {top.get('name', '<unknown>')} "
            f"({float(top.get('delta_pct', 0.0)):+.1f}%)."
        )
    elif improvements:
        top = improvements[0]
        top_part = (
            f" Top improvement: {top.get('name', '<unknown>')} "
            f"({float(top.get('delta_pct', 0.0)):+.1f}%)."
        )
    else:
        top_part = ""
    return (
        f"The new run finished in {wall:+.1f}% wall-time vs baseline. "
        f"{n_reg} kernels regressed, {n_imp} improved.{top_part}"
    )


def run_diff_specialist(
    payload: schemas.DiffSpecialistInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
    progress_callback: Optional[Any] = None,
) -> schemas.DiffSpecialistOutput:
    """Execute the diff specialist."""
    del progress_callback
    diff_result = trace_diff.diff_runs(
        baseline_db=payload.baseline_db,
        new_db=payload.new_db,
        top_kernels=payload.top_kernels,
    )

    regressions: List[Dict[str, Any]] = list(
        diff_result.get("primary_regressions", []) or []
    )
    improvements: List[Dict[str, Any]] = list(
        diff_result.get("primary_improvements", []) or []
    )
    verdict = _classify_verdict(diff_result)
    wall_delta_pct = float(diff_result.get("wall_delta_pct", 0.0) or 0.0)

    agent = build_diff_specialist()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "diff_result": diff_result},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        narrative = _airgap_narrative(diff_result)
        confidence = 0.7
    else:
        so = raw.get("structured_output") or {}
        narrative = so.get("narrative") or _airgap_narrative(diff_result)
        confidence = float(so.get("confidence", 0.7))
        llm_verdict = so.get("verdict")
        if llm_verdict in {"improved", "regressed", "neutral"}:
            verdict = llm_verdict
        regressions = list(so.get("regressions", regressions))
        improvements = list(so.get("improvements", improvements))

    return schemas.DiffSpecialistOutput(
        wall_delta_pct=wall_delta_pct,
        kernel_deltas={
            "regressions": regressions,
            "improvements": improvements,
        },
        verdict=verdict,  # type: ignore[arg-type]
        narrative=narrative,
        confidence=max(0.0, min(1.0, confidence)),
    )


__all__ = ["build_diff_specialist", "run_diff_specialist"]
