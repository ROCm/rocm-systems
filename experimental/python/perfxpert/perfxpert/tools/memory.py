"""memory — cache hit rate classification.

Reads knowledge/memory_hierarchy.yaml thresholds.

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def classify_cache_performance(l1_hit_rate: float, l2_hit_rate: float) -> Dict[str, Any]:
    """Classify L1/L2 cache hit rates into severity buckets.

    Returns:
        {l1: {severity, threshold_used}, l2: {...}, overall_severity}
    """
    thresholds = load_yaml("memory_hierarchy")["cache_hit_rate_thresholds"]

    def _severity(rate: float, good: float, warn: float, critical: float = None) -> str:
        if rate >= good:
            return "good"
        if rate >= warn:
            return "warn"
        return "critical"

    l1 = _severity(l1_hit_rate, thresholds["l1_good"], thresholds["l1_warn"])
    l2 = _severity(l2_hit_rate, thresholds["l2_good"], thresholds["l2_warn"])

    overall = "critical" if "critical" in (l1, l2) else ("warn" if "warn" in (l1, l2) else "good")

    return {
        "l1": {"severity": l1, "hit_rate": l1_hit_rate},
        "l2": {"severity": l2, "hit_rate": l2_hit_rate},
        "overall_severity": overall,
    }
