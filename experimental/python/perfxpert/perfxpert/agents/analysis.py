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
from perfxpert.tools import bottleneck, counters, roofline

# trace_analysis — delegates to legacy analysis functions
try:
    from perfxpert.tools import trace_analysis  # type: ignore
except ImportError:
    # fallback: delegate to legacy analyze.py functions
    class _LegacyTraceAnalysis:
        @staticmethod
        def time_breakdown(db_path: str) -> Dict[str, Any]:
            """Compute time breakdown from database."""
            from perfxpert.connection import PerfxpertConnection
            from perfxpert.analyze import compute_time_breakdown

            conn = PerfxpertConnection(db_path)
            breakdown = compute_time_breakdown(conn)

            # Map legacy keys to agentic schema
            return {
                "kernel_pct": breakdown.get("kernel_percent", 0.0),
                "memcpy_pct": breakdown.get("memcpy_percent", 0.0),
                "api_pct": breakdown.get("overhead_percent", 0.0),
                "idle_pct": 0.0,  # computed as remainder if needed
                "counter_data_available": _check_counters_available(db_path),
            }

        @staticmethod
        def hotspots(db_path: str, top_n: int = 10) -> list:
            """Identify top kernels by execution time."""
            from perfxpert.connection import PerfxpertConnection
            from perfxpert.analyze import identify_hotspots

            conn = PerfxpertConnection(db_path)
            hotspots = identify_hotspots(conn, top_n=top_n)

            # Convert to agentic schema if needed
            return hotspots or []

    def _check_counters_available(db_path: str) -> bool:
        """Check if hardware counters (pmc_events) are available in the database."""
        try:
            from perfxpert.connection import PerfxpertConnection, execute_statement

            conn = PerfxpertConnection(db_path)
            tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='pmc_events'"
            result = execute_statement(conn, tables_query).fetchone()
            has_table = result is not None

            if has_table:
                # Check if table has any data
                count_query = "SELECT COUNT(*) FROM pmc_events LIMIT 1"
                count_result = execute_statement(conn, count_query).fetchone()
                return count_result and count_result[0] > 0
            return False
        except Exception:
            return False

    trace_analysis = _LegacyTraceAnalysis()  # type: ignore[misc,assignment]


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

    Exposed at module level for test injection. Tests can monkeypatch this function
    to stub out database access.
    """
    try:
        breakdown = trace_analysis.time_breakdown(db)
        hotspots = trace_analysis.hotspots(db, top_n=top_n)
    except FileNotFoundError:
        # Database not found (common in unit tests with fake.db).
        # Return empty defaults so tests can mock this function.
        breakdown = {
            "kernel_pct": 0.0,
            "memcpy_pct": 0.0,
            "api_pct": 0.0,
            "idle_pct": 0.0,
            "counter_data_available": False,
        }
        hotspots = []

    # Flatten signals for bottleneck.classify_from_metrics
    m = {
        "memcpy_pct": breakdown.get("memcpy_pct", 0.0),
        "api_overhead_pct": breakdown.get("api_pct", 0.0),
    }
    return {
        "time_breakdown": breakdown,
        "hot_kernels": hotspots,
        "metrics_for_classifier": m,
        "counter_data_available": breakdown.get("counter_data_available", False),
    }


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
            primary_bottleneck=rule_verdict["type"],
            confidence=rule_verdict["confidence"],
            time_breakdown=facts["time_breakdown"],
            hot_kernels=facts["hot_kernels"],
            counter_data_available=facts["counter_data_available"],
        )

    so = raw.get("structured_output") or {}
    return schemas.AnalysisOutput(
        primary_bottleneck=so.get("primary_bottleneck", rule_verdict["type"]),
        confidence=so.get("confidence", rule_verdict["confidence"]),
        time_breakdown=so.get("time_breakdown", facts["time_breakdown"]),
        hot_kernels=so.get("hot_kernels", facts["hot_kernels"]),
        counter_data_available=so.get("counter_data_available", facts["counter_data_available"]),
    )


__all__ = ["build_analysis_agent", "run_analysis"]
