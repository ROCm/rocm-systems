"""plateau — detect when optimization iterations have converged.

Spec §5: if % change in total_runtime_ns < 2% for 2+ consecutive iterations,
the optimization loop should switch to a deeper-tier strategy.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class


_PLATEAU_THRESHOLD_PCT = 0.02
_PLATEAU_MIN_ITERATIONS = 2


@tool_class(ToolClass.READ_ONLY)
def check(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check if recent history shows plateau (< 2% change for 2+ iterations)."""
    if len(history) < _PLATEAU_MIN_ITERATIONS + 1:
        return {"plateau_detected": False, "reason": "insufficient history"}

    # Check last N iterations: each < threshold vs the previous
    recent = history[-(_PLATEAU_MIN_ITERATIONS + 1):]
    deltas = []
    for i in range(1, len(recent)):
        prev = recent[i - 1]["total_runtime_ns"]
        curr = recent[i]["total_runtime_ns"]
        if prev == 0:
            continue
        deltas.append(abs(curr - prev) / prev)

    if all(d < _PLATEAU_THRESHOLD_PCT for d in deltas):
        return {
            "plateau_detected": True,
            "reason": f"last {_PLATEAU_MIN_ITERATIONS} deltas all < {_PLATEAU_THRESHOLD_PCT:.0%}",
            "recent_deltas": deltas,
        }
    return {"plateau_detected": False, "recent_deltas": deltas}
