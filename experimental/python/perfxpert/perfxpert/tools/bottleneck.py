"""bottleneck — classification + prioritization tools.

Pure-rule bottleneck classifier. Deterministic: same metrics in → same
classification out. No LLM calls.

Replaces the fence text "Common Bottleneck Types and Signatures" with
executable rules. Agents call this instead of reasoning-over-prose.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _signature_match(signature: List[Dict[str, Any]], metrics: Dict[str, Any]) -> float:
    """Compute fraction of signature rules that pass against metrics.

    Returns 0.0–1.0. Rules referencing missing metrics count as false.
    """
    if not signature:
        return 0.0
    passed = 0
    for rule in signature:
        metric = rule["metric"]
        op = _OPS[rule["op"]]
        threshold = rule["threshold"]
        value = metrics.get(metric)
        if value is None:
            continue  # missing metric = can't satisfy rule
        if op(value, threshold):
            passed += 1
    return passed / len(signature)


@tool_class(ToolClass.READ_ONLY)
def classify_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Classify bottleneck type from profiling metrics.

    Rule-based: checks each bottleneck type's signatures against metrics,
    returns the best-matching type + confidence score.

    Args:
        metrics: dict with keys like valu_util_pct, memcpy_pct, etc.
                 Missing keys default to "rule not applicable" (not "rule failed").

    Returns:
        {"type": str, "confidence": float, "reasoning": str, "all_scores": dict}

    Example:
        >>> classify_from_metrics({"valu_util_pct": 0.85, "arithmetic_intensity_above_ridge": 1})
        {"type": "compute", "confidence": 0.67, ...}
    """
    types = load_yaml("bottleneck_types")
    scores = {}
    for entry in types:
        if entry["name"] == "mixed":
            continue  # "mixed" is the fallback, not a direct match
        scores[entry["name"]] = _signature_match(entry["signatures"], metrics)

    best = max(scores, key=scores.get)
    best_score = scores[best]

    # If no type scores above 0.5, classify as mixed
    if best_score < 0.5:
        return {
            "type": "mixed",
            "confidence": 0.5,
            "reasoning": "No single bottleneck signature dominates; triage needed",
            "all_scores": scores,
        }

    return {
        "type": best,
        "confidence": best_score,
        "reasoning": f"Signature match {best_score:.2f} for {best}",
        "all_scores": scores,
    }


@tool_class(ToolClass.READ_ONLY)
def lookup_signatures(bottleneck_type: str) -> Dict[str, Any]:
    """Retrieve the signature definition for a named bottleneck type.

    Args:
        bottleneck_type: one of "compute", "memory_transfer", "latency",
                         "api_overhead", "mixed".

    Returns:
        The matching entry from bottleneck_types.yaml.

    Raises:
        KeyError: if bottleneck_type is not recognized.
    """
    types = load_yaml("bottleneck_types")
    for entry in types:
        if entry["name"] == bottleneck_type:
            return entry
    known = ", ".join(e["name"] for e in types)
    raise KeyError(f"Unknown bottleneck type {bottleneck_type!r}; known: {known}")


@tool_class(ToolClass.READ_ONLY)
def prioritize_by_amdahl(execution_time_pct: float) -> Dict[str, Any]:
    """Assign Amdahl-law priority based on kernel's share of total runtime.

    Args:
        execution_time_pct: kernel runtime / total runtime, in 0.0-1.0.

    Returns:
        {"priority": "high"|"medium"|"low", "rationale": str}

    Rationale: optimizing a kernel with < 5% share yields at most 5% speedup
    (Amdahl ceiling). Optimizing > 10% is high-value.
    """
    thresholds = load_yaml("amdahl_thresholds")
    if execution_time_pct >= thresholds["high_threshold"]:
        return {
            "priority": "high",
            "rationale": f"Kernel is {execution_time_pct:.1%} of total runtime (≥ {thresholds['high_threshold']:.0%})",
        }
    if execution_time_pct >= thresholds["low_threshold"]:
        return {
            "priority": "medium",
            "rationale": f"Kernel is {execution_time_pct:.1%} of total runtime",
        }
    return {
        "priority": "low",
        "rationale": f"Kernel is {execution_time_pct:.1%} of total runtime (< {thresholds['low_threshold']:.0%}); Amdahl ceiling limits gain",
    }
