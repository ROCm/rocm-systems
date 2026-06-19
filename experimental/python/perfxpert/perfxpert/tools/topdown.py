"""topdown — classify execution-time-breakdown red flags.

Reads knowledge/top_down_analysis.yaml thresholds.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def classify_overhead(
    memcpy_pct: float, api_pct: float, idle_pct: float
) -> List[Dict[str, Any]]:
    """Identify red flags in time breakdown.

    Returns a list of {red_flag_name, priority, message, current_value} for
    every flag whose threshold is breached.
    """
    config = load_yaml("top_down_analysis")
    red_flags = config["red_flags"]

    current_values = {
        "memcpy_pct": memcpy_pct,
        "api_overhead_pct": api_pct,
        "gpu_idle_pct": idle_pct,
    }

    triggered = []
    for name, flag in red_flags.items():
        metric = flag["metric"]
        value = current_values.get(metric, 0.0)
        if value >= flag["threshold"]:
            triggered.append({
                "red_flag_name": name,
                "metric": metric,
                "value": value,
                "threshold": flag["threshold"],
                "priority": flag["priority"],
                "message": flag["message"],
            })
    return triggered
