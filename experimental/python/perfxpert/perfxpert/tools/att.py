"""att — ATT stall reason + ratio classification.

Reads knowledge/pc_sampling_stall_reasons.yaml + att_output_format.yaml.

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def classify_stall_reason(stall_code: str) -> Dict[str, Any]:
    """Return root_cause + mitigation for a stall reason code."""
    reasons = load_yaml("pc_sampling_stall_reasons")
    for entry in reasons:
        if entry["code"] == stall_code:
            return entry
    raise KeyError(f"Unknown stall code {stall_code!r}")


@tool_class(ToolClass.READ_ONLY)
def classify_stall_ratio(stall_ratio: float) -> Dict[str, Any]:
    """Classify stall_ratio into severity bucket."""
    thresholds = load_yaml("att_output_format")["stall_ratio_thresholds"]
    if stall_ratio >= thresholds["critical"]:
        severity = "critical"
    elif stall_ratio >= thresholds["high"]:
        severity = "high"
    elif stall_ratio >= thresholds["medium"]:
        severity = "medium"
    else:
        severity = "low"
    return {"severity": severity, "stall_ratio": stall_ratio}
