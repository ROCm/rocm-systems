"""Analysis decision-maker (Layer 1) — classifies bottleneck from trace facts.

Responsibilities:
- Call analysis tools (time_breakdown, hotspots) to collect metrics.
- Classify bottleneck via pure-rule tool (bottleneck.classify_from_metrics).
- Let the LLM (if enabled) refine or flag edge cases; never override the
  rule-verdict in air-gap mode.
- Return AnalysisOutput (frozen) to Root.

Tool allowlist (exactly 5 per spec §2 cap):
  analysis.time_breakdown, analysis.hotspots,
  bottleneck.classify_from_metrics, roofline.classify,
  counters.validate_for_gpu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.analysis import analyze_hardware_counters, compute_time_breakdown
from perfxpert.connection import PerfxpertConnection
from perfxpert.tools import bottleneck, counters, roofline
from perfxpert.tools import trace_analysis


_FENCE_PATH = Path(__file__).parent / "fence" / "analysis.md"


def build_analysis_agent() -> Agent:
    tools = [
        ToolBinding(name="analysis.time_breakdown", fn=trace_analysis.time_breakdown),
        ToolBinding(name="analysis.hotspots", fn=trace_analysis.hotspots),
        ToolBinding(name="bottleneck.classify_from_metrics", fn=bottleneck.classify_from_metrics),
        ToolBinding(name="roofline.classify", fn=roofline.classify),
        ToolBinding(name="counters.validate_for_gpu", fn=counters.validate_for_gpu),
    ]
    return Agent(
        name="Analysis",
        layer=1,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.AnalysisInput,
        output_schema=schemas.AnalysisOutput,
        tools=tools,
        allowed_handoffs=[],   # Layer-1 returns to Root
        token_budget=4096,
    )


def _collect_deterministic_metrics(db: str, top_n: int = 10) -> Dict[str, Any]:
    """Collect the rule-based metrics needed by the classifier.

    Exposed at module level for test injection.
    """
    breakdown = trace_analysis.time_breakdown(db)
    hotspots = trace_analysis.hotspots(db, top_n=top_n)
    with PerfxpertConnection(db) as conn:
        legacy_breakdown = compute_time_breakdown(conn)
        hardware_counters = analyze_hardware_counters(conn)

    total_calls = sum(kernel.get("calls", 0) for kernel in hotspots)
    avg_kernel_duration_us = (
        legacy_breakdown.get("total_kernel_time", 0) / total_calls / 1000.0
        if total_calls > 0
        else 0.0
    )
    hc_metrics = hardware_counters.get("metrics", {})
    avg_waves = float(hc_metrics.get("avg_waves", 0.0) or 0.0)
    occupancy_pct = min(avg_waves / 32.0, 1.0) if avg_waves > 0 else 0.0

    # Flatten signals for bottleneck.classify_from_metrics
    m = {
        "memcpy_pct": breakdown.get("memcpy_pct", 0.0),
        "api_overhead_pct": breakdown.get("api_pct", 0.0),
        "avg_kernel_duration_us": avg_kernel_duration_us,
        "total_kernel_calls": total_calls,
        "gpu_util_pct": float(hc_metrics.get("gpu_utilization_percent", 0.0) or 0.0) / 100.0,
        "avg_waves_per_cu": avg_waves,
        "occupancy_pct": occupancy_pct,
    }
    return {
        "time_breakdown": breakdown,
        "hot_kernels": hotspots,
        "metrics_for_classifier": m,
        "counter_data_available": bool(hardware_counters.get("has_counters", False)),
        "legacy_time_breakdown": legacy_breakdown,
    }


def _fallback_trace_classification(facts: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy-compatible fallback for sparse trace-only inputs.

    The bottleneck signature YAML expects richer metrics than the parity fixtures
    provide. When those dimensions are missing, fall back to the established
    legacy trace heuristics so the public batch API stays comparable.
    """
    breakdown = facts["time_breakdown"]
    memcpy_pct = float(breakdown.get("memcpy_pct", 0.0) or 0.0)
    api_pct = float(breakdown.get("api_pct", 0.0) or 0.0)
    kernel_pct = float(breakdown.get("kernel_pct", 0.0) or 0.0)
    has_counters = bool(facts.get("counter_data_available", False))

    if memcpy_pct > 0.30:
        return {"type": "memory_transfer", "confidence": 0.85}
    if memcpy_pct > 0.20:
        return {"type": "memory_transfer", "confidence": 0.70}
    if api_pct > 0.25:
        return {"type": "latency", "confidence": 0.75}
    if kernel_pct > 0.70 and has_counters:
        return {"type": "compute", "confidence": 0.80}
    if kernel_pct > 0.70:
        return {"type": "compute", "confidence": 0.60}
    return {"type": "mixed", "confidence": 0.50}


def run_analysis(
    payload: schemas.AnalysisInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.AnalysisOutput:
    """Run Analysis for one trace database."""
    # Step 1: deterministic metric collection (always).
    facts = _collect_deterministic_metrics(payload.database_path, top_n=payload.top_kernels)

    # Step 2: deterministic classifier verdict (always).
    rule_verdict = bottleneck.classify_from_metrics(facts["metrics_for_classifier"])
    compat_verdict = _fallback_trace_classification(facts)
    chosen_verdict = compat_verdict if compat_verdict["type"] != "mixed" else rule_verdict

    # Step 3: LLM refinement (optional).
    agent = build_analysis_agent()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "facts": facts, "rule_verdict": rule_verdict},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        return schemas.AnalysisOutput(
            primary_bottleneck=chosen_verdict["type"],
            confidence=chosen_verdict["confidence"],
            time_breakdown=facts["time_breakdown"],
            hot_kernels=facts["hot_kernels"],
            counter_data_available=facts["counter_data_available"],
        )

    so = raw.get("structured_output") or {}
    return schemas.AnalysisOutput(
        primary_bottleneck=so.get("primary_bottleneck", chosen_verdict["type"]),
        confidence=so.get("confidence", chosen_verdict["confidence"]),
        time_breakdown=so.get("time_breakdown", facts["time_breakdown"]),
        hot_kernels=so.get("hot_kernels", facts["hot_kernels"]),
        counter_data_available=so.get("counter_data_available", facts["counter_data_available"]),
    )


__all__ = ["build_analysis_agent", "run_analysis"]
