"""Shared helper — attach predict_change_impact output to specialist techniques.

Each Layer-2 specialist calls this once per run, AFTER the Agent tool loop
has returned and BEFORE handing the output back to Recommendation. For
every technique emitted whose ``name`` (or ``id``) matches an entry in
``knowledge/change_impact_models.yaml`` the technique dict is augmented
with:

    technique["predicted_impact_range"] = [lo, hi]       # conservative
    technique["confidence"]              = float
    technique["source_citation"]         = str
    technique["predicted_rationale"]     = str
    technique["prediction_id"]           = str
    technique["roofline_delta"]          = dict

Techniques that do NOT map onto a catalog entry are left untouched —
formatters only render the Predicted line when the key exists.

Hard rules (spec §6) are enforced by ``predict_change_impact`` itself;
this helper is a thin iterator that skips emission when the predictor
returns zero-confidence or a null range.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_CATALOG_IDS_CACHE: Optional[set] = None


def _catalog_ids() -> set:
    global _CATALOG_IDS_CACHE
    if _CATALOG_IDS_CACHE is None:
        try:
            from perfxpert.tools import predict_impact  # type: ignore
            entries = predict_impact.list_supported_changes()
            _CATALOG_IDS_CACHE = {e.get("id") for e in entries if e.get("id")}
        except Exception:
            _CATALOG_IDS_CACHE = set()
    return _CATALOG_IDS_CACHE


def _technique_change_type(technique: Dict[str, Any]) -> Optional[str]:
    """Extract a change_type id from a technique dict.

    The specialists emit dicts shaped either ``{"id": ..., "name": ...}``
    or ``{"name": ...}``. We check ``id`` first, then ``name`` after
    normalizing to lowercase-with-underscores. The returned value is only
    returned when the id is actually in the catalog.
    """
    for key in ("id", "name"):
        raw = technique.get(key)
        if not isinstance(raw, str):
            continue
        normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in _catalog_ids():
            return normalized
    return None


def attach_predictions_to_techniques(
    techniques: List[Dict[str, Any]],
    payload: Any,
) -> List[Dict[str, Any]]:
    """Augment every technique dict with predicted_impact_range + confidence.

    Args:
        techniques: list of technique dicts emitted by a specialist.
        payload: the specialist's input schema (frozen pydantic model).
            Used to derive ``kernel_name`` (first hot kernel) and
            ``baseline_db`` (from ``hot_kernels[0]`` or an env hint).

    Returns:
        A NEW list — techniques are copied (shallow) so frozen callers
        upstream never mutate a shared dict.
    """
    if not techniques:
        return list(techniques or [])
    try:
        from perfxpert.tools import predict_impact  # type: ignore
    except ImportError:
        return list(techniques)

    # Derive baseline_db + kernel_name from payload. Hot-kernels carry
    # both; we pick the first as representative. Specialists that don't
    # expose these fields short-circuit to "no prediction attached".
    hot_kernels = list(getattr(payload, "hot_kernels", []) or [])
    if not hot_kernels:
        return list(techniques)
    kernel_name = str(hot_kernels[0].get("name") or "")
    if not kernel_name:
        return list(techniques)
    baseline_db = str(
        hot_kernels[0].get("baseline_db")
        or getattr(payload, "baseline_db", None)
        or getattr(payload, "database_path", None)
        or ""
    )
    kernel_time_pct = hot_kernels[0].get("pct") or hot_kernels[0].get("pct_total")

    out: List[Dict[str, Any]] = []
    for technique in techniques:
        copy = dict(technique)
        change_type = _technique_change_type(copy)
        if change_type is None:
            out.append(copy)
            continue
        change_params: Dict[str, Any] = {}
        if isinstance(kernel_time_pct, (int, float)):
            change_params["kernel_time_pct"] = float(kernel_time_pct)
        # Allow specialists running over DBs without counters to still
        # surface the prediction — pass counter_data_available=True when
        # any counter_data field is non-empty upstream.
        counter_data = getattr(payload, "counter_data", None) or {}
        if counter_data:
            change_params["counter_data_available"] = True
        try:
            prediction = predict_impact.predict_change_impact(
                baseline_db=baseline_db,
                kernel_name=kernel_name,
                change_type=change_type,
                change_params=change_params,
            )
        except Exception:
            out.append(copy)
            continue
        rng = prediction.get("predicted_speedup_range")
        if rng is None or prediction.get("confidence", 0.0) <= 0.0:
            out.append(copy)
            continue
        copy["predicted_impact_range"] = list(rng)
        copy["confidence"] = float(prediction["confidence"])
        copy["source_citation"] = prediction.get("source_citation", "")
        copy["predicted_rationale"] = prediction.get("rationale", "")
        copy["prediction_id"] = prediction.get("prediction_id", "")
        copy["roofline_delta"] = prediction.get("roofline_delta") or {}
        out.append(copy)
    return out


__all__ = ["attach_predictions_to_techniques"]
