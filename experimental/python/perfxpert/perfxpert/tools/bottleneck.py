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
    """Compute evidence-weighted confidence that metrics match a signature.

    Returns 0.0–1.0.  Rules referencing missing metrics count as "neutral"
    (not counted toward denominator), allowing partial evidence to still
    produce a valid score.  However, partial evidence is penalised relative
    to full evidence via an evidence-strength factor (finding #26).

    Formula:
        match_ratio    = passed / evaluated          (fraction of rules that fire)
        evidence_factor = evaluated / len(signature)  (fraction of rules testable)
        score           = match_ratio × evidence_factor

    Examples with a 5-rule signature:
        5/5 evaluated, all pass  → 1.0 × 1.0 = 1.0   (certain)
        2/5 evaluated, all pass  → 1.0 × 0.4 = 0.4   (insufficient evidence)
        1/5 evaluated, all pass  → 1.0 × 0.2 = 0.2   (single weak signal)
        5/5 evaluated, 3 pass    → 0.6 × 1.0 = 0.6   (partial match, full evidence)

    This is monotonically increasing in both match ratio and evidence strength,
    and removes the anomaly where a single-rule match returned confidence 1.0
    equal to a full 5-of-5 match.
    """
    if not signature:
        return 0.0

    passed = 0
    evaluated = 0  # rules where we had data to evaluate

    for rule in signature:
        metric = rule["metric"]
        op = _OPS[rule["op"]]
        threshold = rule["threshold"]
        value = metrics.get(metric)

        if value is None:
            continue  # missing metric = neutral (don't count toward denominator)

        evaluated += 1
        if op(value, threshold):
            passed += 1

    # If no rules could be evaluated, signature doesn't apply (0 confidence)
    if evaluated == 0:
        return 0.0

    # Evidence-weighted confidence (finding #26 fix):
    # match_ratio × evidence_factor ensures sparse evidence scores lower
    # than full evidence even when all evaluated rules match.
    match_ratio = passed / evaluated
    evidence_factor = evaluated / len(signature)
    return match_ratio * evidence_factor


@tool_class(ToolClass.READ_ONLY)
def classify_from_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Classify bottleneck type from profiling metrics.

    Rule-based: checks each bottleneck type's signatures against metrics,
    returns the best-matching type + confidence score.

    Args:
        metrics: dict with keys like valu_util_pct, memcpy_pct, etc.
                 Missing keys (None or absent) default to "rule not applicable"
                 (neutral — not counted in denominator).

    Returns:
        {"type": str, "confidence": float, "reasoning": str, "all_scores": dict}

    Special return when all metrics are None/missing:
        {"type": "data_insufficient", "confidence": 0.0, "reasoning": "..."}

    Example:
        >>> classify_from_metrics({"valu_util_pct": 0.85, "arithmetic_intensity_above_ridge": 1})
        {"type": "compute", "confidence": 0.67, ...}
    """
    types = load_yaml("bottleneck_types")
    scores = {}
    any_evaluated = False
    for entry in types:
        if entry["name"] == "mixed":
            continue  # "mixed" is the fallback, not a direct match
        score = _signature_match(entry["signatures"], metrics)
        scores[entry["name"]] = score
        # Track whether at least one rule was evaluated (i.e. had non-None metric)
        for rule in entry["signatures"]:
            if metrics.get(rule["metric"]) is not None:
                any_evaluated = True

    # If no signature could be evaluated at all — every metric is None/absent —
    # return data_insufficient rather than the silent mixed-at-0.5 fallback.
    if not any_evaluated:
        return {
            "type": "data_insufficient",
            "confidence": 0.0,
            "reasoning": (
                "No hardware counter data available — all classifier inputs are None. "
                "Re-capture the trace with hardware counters enabled to get a meaningful "
                "bottleneck classification."
            ),
            "all_scores": scores,
        }

    best = max(scores, key=scores.get)
    best_score = scores[best]

    # If no type scores above 0.5 (but we did have some data), classify as mixed
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
