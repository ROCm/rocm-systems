"""_predict_attach — helper that annotates specialist techniques with
change-impact predictions (via ``predict_impact.predict_change_impact``
when available) without making that tool a hard dependency.

Introduced alongside the Phase-10 advanced-specialist work. Keeps the
compute / memory / latency specialist modules importable even when the
optional ``predict_impact`` tool is not installed — the attachment step
is a no-op in that case.
"""

from __future__ import annotations

from typing import Any, Dict, List


def attach_predictions_to_techniques(
    techniques: List[Dict[str, Any]], payload: Any
) -> List[Dict[str, Any]]:
    """Best-effort: decorate each technique dict with a speedup bracket.

    Args:
        techniques: List of technique dicts as returned by the
            deterministic-ranking path or by the LLM structured output.
        payload: The specialist's input schema instance. We only read
            ``gfx_id`` + ``hot_kernels`` when the predict-impact tool is
            present, so the shape is intentionally duck-typed.

    Returns:
        The same list with a ``predicted_speedup`` dict added to each
        entry (``{"lo": float, "hi": float, "confidence": float}``) —
        or the input unchanged when the predict-impact tool is absent.
    """
    try:
        from perfxpert.tools import predict_impact  # type: ignore
    except ImportError:
        return techniques

    gfx_id = getattr(payload, "gfx_id", "")
    out: List[Dict[str, Any]] = []
    for t in techniques:
        enriched = dict(t)
        try:
            pred = predict_impact.predict_change_impact(
                technique=t.get("name", ""),
                gfx_id=gfx_id,
                hot_kernels=getattr(payload, "hot_kernels", []),
            )
            enriched["predicted_speedup"] = pred
        except Exception:
            # Never let prediction failure break the specialist.
            pass
        out.append(enriched)
    return out


__all__ = ["attach_predictions_to_techniques"]
