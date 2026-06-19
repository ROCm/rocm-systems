"""workflow — profiling workflow state machine.

Given current Tier + goal, return the next profiling step.

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class


_WORKFLOW = {
    (0, "collect"):        {"next_tier": 1, "action": "rocprofv3 --sys-trace"},
    (1, "classify"):       {"next_tier": 1, "action": "analyze trace"},
    (1, "measure"):        {"next_tier": 2, "action": "rocprofv3 --pmc SQ_WAVES GRBM_COUNT"},
    (2, "deep"):           {"next_tier": 3, "action": "rocprofv3 --att --att-library-path /opt/rocm/lib"},
    (2, "pc_sampling"):    {"next_tier": 3, "action": "rocprofv3 --pc-sampling"},
}


@tool_class(ToolClass.READ_ONLY)
def next_step(tier: int, goal: str) -> Dict[str, Any]:
    """Return the next profiling step given current tier + goal.

    Args:
        tier: 0-3 (source / trace / counters / ATT).
        goal: "collect", "classify", "measure", "deep", "pc_sampling".

    Returns:
        {"next_tier": int, "action": str}

    Raises:
        KeyError: if (tier, goal) pair unknown.
    """
    key = (tier, goal)
    if key not in _WORKFLOW:
        known = sorted(_WORKFLOW.keys())
        raise KeyError(f"Unknown (tier, goal) {key}; known: {known}")
    return dict(_WORKFLOW[key])
