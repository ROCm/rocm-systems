"""profiling — cost-ordered cascade tool for choosing next data collection.

Given what we already have (fingerprint) + what we want (goal), return the
cheapest profiling command that fills the gap.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, Optional, Set

from perfxpert.tools._class import ToolClass, tool_class


# Cost order: read-from-db (free) < source-scan (cheap) < rocprofv3 --sys-trace
# < rocprofv3 --pmc < --pc-sampling < --att < rocprof-compute
_LADDER = [
    {"cost_s": 0, "name": "read_existing_db", "covers": {"--sys-trace", "--hip-trace"}},
    {"cost_s": 10, "name": "source_scan", "covers": {"tier0"}},
    {"cost_s": 30, "name": "rocprofv3 --sys-trace", "covers": {"--sys-trace", "--hip-trace", "--kernel-trace"}},
    {"cost_s": 90, "name": "rocprofv3 --pmc SQ_WAVES GRBM_COUNT", "covers": {"pmc:SQ_WAVES", "pmc:GRBM_COUNT"}},
    {"cost_s": 120, "name": "rocprofv3 --pmc FETCH_SIZE", "covers": {"pmc:FETCH_SIZE"}},
    {"cost_s": 120, "name": "rocprofv3 --pmc WRITE_SIZE", "covers": {"pmc:WRITE_SIZE"}},
    {"cost_s": 180, "name": "rocprofv3 --pc-sampling", "covers": {"pc_sampling"}},
    {"cost_s": 240, "name": "rocprofv3 --att", "covers": {"att"}},
    {"cost_s": 300, "name": "rocprof-compute", "covers": {"roofline", "deep_analysis"}},
]


@tool_class(ToolClass.READ_ONLY)
def fill_gap(current_fingerprint: Set[str], goal: str) -> Optional[Dict[str, Any]]:
    """Return the cheapest next action that adds information beyond current_fingerprint.

    Args:
        current_fingerprint: set of canonical tokens we already have
        (e.g., frozenset({"--sys-trace", "pmc:SQ_WAVES"}))
        goal: target signal ("compute_bound_diagnosis", "memory_bandwidth",
              "stall_analysis", ...)

    Returns:
        {cost_s, name, adds: set} for the chosen step; None if all relevant steps done.
    """
    cur = set(current_fingerprint or [])
    for step in _LADDER:
        adds = step["covers"] - cur
        if adds:
            return {"cost_s": step["cost_s"], "name": step["name"], "adds": adds}
    return None
