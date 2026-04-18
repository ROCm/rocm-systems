"""tracelens — TraceLens-derived metric classification.

Reads knowledge/tracelens_metrics.yaml thresholds.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def classify_overhead(idle_pct: float, wasted_pct: float) -> Dict[str, Any]:
    """Classify TraceLens idle_pct + wasted_pct against thresholds."""
    thresholds = load_yaml("tracelens_metrics")["thresholds"]

    idle_severity = "high" if idle_pct >= thresholds["idle_high"] else "low"
    wasted_severity = "high" if wasted_pct >= thresholds["wasted_high"] else "low"

    actions: List[str] = []
    if idle_severity == "high":
        actions.append("Batch kernels or add parallel streams")
    if wasted_severity == "high":
        actions.append("Consider kernel fusion")

    return {
        "idle_severity": idle_severity,
        "wasted_severity": wasted_severity,
        "idle_pct": idle_pct,
        "wasted_pct": wasted_pct,
        "recommended_actions": actions,
    }


@tool_class(ToolClass.READ_ONLY)
def lookup_metrics() -> Dict[str, Any]:
    """Return the full TraceLens metric catalog."""
    return load_yaml("tracelens_metrics")
