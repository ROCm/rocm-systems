"""predict_impact — Change-Impact Prediction (Phase 10, MVP).

Three READ_ONLY MCP tools:

  - predict_change_impact(baseline_db, kernel_name, change_type, change_params)
    Returns a speedup bracket + rationale + roofline delta for a specific
    optimization technique applied to a specific hot kernel.

  - list_supported_changes()
    Returns the set of change_type ids the predictor understands
    (seeded from knowledge/change_impact_models.yaml).

  - explain_prediction(prediction_id)
    Re-hydrates an exact in-process prediction by its id, then falls back
    to a no-create current-project retained-observation lookup.

Hard rules (spec §6):

  - Amdahl guard: kernel <5% total runtime => zero-confidence no-op.
  - Tier-2 gate: missing counter data => zero-confidence, ``rationale``
    asks caller to re-run with ``--pmc basic``.
  - Unknown technique => ``predicted_speedup_range`` is ``None``.
  - Conservative bracket: ``hi`` field is multiplied by 0.85 (undersell).
  - Every prediction cites the ``source_citation`` field from the seed
    entry in ``rationale``.

Tool class: READ_ONLY. No filesystem writes, no network access.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert import __version__ as _PERFXPERT_VERSION
from perfxpert.knowledge import load_yaml
from perfxpert.retention.identity import (
    PREDICTOR_SCHEMA_VERSION,
    catalog_entry_hash,
    semantic_prediction_id_from_context,
)
from perfxpert.retention.scope import resolve_scope_context
from perfxpert.tools._class import ToolClass, tool_class


# ---------------------------------------------------------------------------
# In-process prediction cache. Durable writes live outside this READ_ONLY tool.
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
_PREDICTION_STORE: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}
_PREDICTION_IDENTITIES: Dict[str, Dict[str, Any]] = {}


def _cache_prediction(
    prediction: Dict[str, Any],
    *,
    effective_params: Dict[str, Any],
    catalog_entry: Optional[Dict[str, Any]],
) -> str:
    """Cache a prediction under its source-independent semantic identity."""

    identity_context = {
        "effective_params": dict(effective_params),
        "catalog_hash": catalog_entry_hash(catalog_entry),
        "producer_version": _PERFXPERT_VERSION,
        "predictor_version": PREDICTOR_SCHEMA_VERSION,
    }
    pid = semantic_prediction_id_from_context(
        prediction,
        **identity_context,
    )
    cached = dict(prediction)
    cached["prediction_id"] = pid
    scope_id = resolve_scope_context().scope_id
    source_key = str(prediction.get("baseline_db") or "")
    with _STORE_LOCK:
        previous_context = _PREDICTION_IDENTITIES.get(pid)
        if previous_context is not None and previous_context != identity_context:
            raise RuntimeError(f"prediction identity collision for {pid!r}")
        _PREDICTION_IDENTITIES[pid] = identity_context
        _PREDICTION_STORE.setdefault((scope_id, pid), {})[source_key] = cached
    return pid


def _prediction_identity_context_for_retention(
    prediction_id: str,
) -> Optional[Dict[str, Any]]:
    """Return source-free identity metadata for explicit recorders."""

    with _STORE_LOCK:
        context = _PREDICTION_IDENTITIES.get(prediction_id)
        return dict(context) if context is not None else None


def _reset_store_for_tests() -> None:
    """Private helper — clears the prediction store. Tests only."""
    with _STORE_LOCK:
        _PREDICTION_STORE.clear()
        _PREDICTION_IDENTITIES.clear()


# ---------------------------------------------------------------------------
# Knowledge loading
# ---------------------------------------------------------------------------


def _load_models() -> List[Dict[str, Any]]:
    """Load change_impact_models.yaml. Returns [] on any failure."""
    try:
        models = load_yaml("change_impact_models")
    except FileNotFoundError:
        return []
    return list(models or [])


def _find_model(change_type: str) -> Optional[Dict[str, Any]]:
    for entry in _load_models():
        if entry.get("id") == change_type:
            return entry
    return None


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------


_AMDAHL_DEFAULTS = {"low_threshold": 0.05, "high_threshold": 0.10}


def _amdahl_low_threshold() -> float:
    try:
        thresholds = load_yaml("amdahl_thresholds")
    except FileNotFoundError:
        return _AMDAHL_DEFAULTS["low_threshold"]
    return float(thresholds.get("low_threshold", _AMDAHL_DEFAULTS["low_threshold"]))


def _kernel_time_pct(baseline_db: str, kernel_name: str) -> Optional[float]:
    """Return target kernel's share of total kernel runtime.

    Returns ``None`` when the DB cannot be opened (e.g. missing file or
    the rocpd schema is absent). Callers treat ``None`` as "unknown" —
    they still respect the Amdahl guard when provided via change_params.
    """
    try:
        from perfxpert.tools import regression as _reg
        runtimes = _reg.extract_kernel_runtimes_from_db(baseline_db)
    except Exception:
        return None
    if not runtimes:
        return None
    total = sum(int(r.total_runtime_ns) for r in runtimes) or 0
    if total == 0:
        return None
    match = next(
        (int(r.total_runtime_ns) for r in runtimes if r.kernel_name == kernel_name),
        0,
    )
    return match / total if total else None


_COUNTER_TABLE_CANDIDATES = (
    "pmc_events",
    "rocpd_counter_values",
    "counter_values",
    "COUNTER",
)


def _readonly_sqlite_uri(db_path: str) -> str:
    """Build a read-only SQLite file URI with path metacharacters escaped."""
    if not db_path:
        raise ValueError("db_path must not be empty")
    return f"{Path(db_path).expanduser().resolve(strict=False).as_uri()}?mode=ro"


def _baseline_has_counters(baseline_db: str) -> bool:
    """Return True iff the rocpd DB has any counter rows.

    Tier-2 is gated on the presence of counter data. The specific table
    name has changed across rocprofiler-sdk releases, so we probe a small
    allowlist + fall back to ``False`` on any exception.
    """
    try:
        with sqlite3.connect(_readonly_sqlite_uri(baseline_db), uri=True) as conn:
            cur = conn.cursor()
            for tbl in _COUNTER_TABLE_CANDIDATES:
                try:
                    cur.execute(f"SELECT 1 FROM {tbl} LIMIT 1")
                    row = cur.fetchone()
                    if row is not None:
                        return True
                except sqlite3.Error:
                    continue
    except (ValueError, sqlite3.Error):
        return False
    return False


# ---------------------------------------------------------------------------
# Public READ_ONLY tools
# ---------------------------------------------------------------------------


_CONSERVATIVE_HI_FACTOR = 0.85


@tool_class(ToolClass.READ_ONLY)
def predict_change_impact(
    baseline_db: str,
    kernel_name: str,
    change_type: str,
    change_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Predict a speedup bracket for a single optimization technique.

    Args:
        baseline_db: Path to the baseline rocprofiler-sdk ``.db``. Used
            to derive ``kernel_time_pct`` for the Amdahl guard and to
            verify counters are present (tier-2 gate).
        kernel_name: Name of the hot kernel the technique would target.
        change_type: Id of an entry in
            ``knowledge/change_impact_models.yaml`` (e.g. ``"vgpr_reduction"``).
        change_params: Caller-provided overrides. Honored keys:
            - ``kernel_time_pct`` (float): bypass DB probe (useful for
              tests and offline callers).
            - ``counter_data_available`` (bool): bypass DB probe of the
              counter tier gate.

    Returns:
        dict with keys::

            {
              "predicted_speedup_range": [lo, hi] | None,
              "confidence":              float in [0,1],
              "rationale":               str,
              "roofline_delta":          {"before": ..., "after": ...},
              "assumptions":             List[str],
              "source_citation":         str,
              "prediction_id":           str,
              "change_type":             str,
              "kernel_name":             str,
              "baseline_db":             str,
            }

        When a hard-rule gate fires the ``confidence`` is ``0.0`` and
        ``predicted_speedup_range`` is ``None``. The rationale string
        explains which gate fired so the caller can act on it.
    """
    params = dict(change_params or {})
    assumptions: List[str] = []
    roofline_delta: Dict[str, Any] = {"before": None, "after": None}

    # ---- Unknown technique gate --------------------------------------
    model = _find_model(change_type)
    if model is None:
        prediction = {
            "predicted_speedup_range": None,
            "confidence": 0.0,
            "rationale": (
                f"change_type {change_type!r} is not in the change-impact catalog — "
                "run `perfxpert diff` after applying the change to measure the "
                "actual speedup."
            ),
            "roofline_delta": roofline_delta,
            "assumptions": [],
            "source_citation": "",
            "change_type": change_type,
            "kernel_name": kernel_name,
            "baseline_db": baseline_db,
        }
        prediction["prediction_id"] = _cache_prediction(
            prediction,
            effective_params={},
            catalog_entry=None,
        )
        return prediction

    catalog_lo = float(model["speedup_bounds"]["lo"])
    catalog_hi = float(model["speedup_bounds"]["hi"])
    model_confidence = float(model.get("confidence", 0.5))
    citation = str(model.get("source_citation", ""))

    # ---- Amdahl guard -------------------------------------------------
    low_threshold = _amdahl_low_threshold()
    override_pct = params.get("kernel_time_pct")
    if isinstance(override_pct, (int, float)):
        ktp: Optional[float] = float(override_pct)
    else:
        ktp = _kernel_time_pct(baseline_db, kernel_name)

    if ktp is not None and ktp < low_threshold:
        prediction = {
            "predicted_speedup_range": None,
            "confidence": 0.0,
            "rationale": (
                f"kernel {kernel_name!r} contributes {ktp * 100:.1f}% of runtime "
                f"(< {low_threshold * 100:.0f}%); Amdahl guard — optimizing will "
                "not move the wall clock. See "
                f"{citation} for the technique itself."
            ),
            "roofline_delta": roofline_delta,
            "assumptions": [f"kernel_time_pct={ktp:.4f}"],
            "source_citation": citation,
            "change_type": change_type,
            "kernel_name": kernel_name,
            "baseline_db": baseline_db,
        }
        prediction["prediction_id"] = _cache_prediction(
            prediction,
            effective_params={"kernel_time_pct": ktp},
            catalog_entry=model,
        )
        return prediction

    # ---- Tier-2 counter gate -----------------------------------------
    override_counters = params.get("counter_data_available")
    if isinstance(override_counters, bool):
        has_counters = override_counters
    else:
        has_counters = _baseline_has_counters(baseline_db)

    if not has_counters:
        prediction = {
            "predicted_speedup_range": None,
            "confidence": 0.0,
            "rationale": (
                "needs counter data — add `--pmc basic` to the baseline run "
                "(HBM bandwidth + occupancy counters are required to predict "
                f"the {change_type} impact). See {citation}."
            ),
            "roofline_delta": roofline_delta,
            "assumptions": [],
            "source_citation": citation,
            "change_type": change_type,
            "kernel_name": kernel_name,
            "baseline_db": baseline_db,
        }
        prediction["prediction_id"] = _cache_prediction(
            prediction,
            effective_params={
                "kernel_time_pct": ktp,
                "counter_data_available": False,
            },
            catalog_entry=model,
        )
        return prediction

    # ---- Build roofline delta (best-effort, optional) ----------------
    ai_before = params.get("arithmetic_intensity_before")
    ai_after = params.get("arithmetic_intensity_after")
    if isinstance(ai_before, (int, float)):
        roofline_delta["before"] = {"ai": float(ai_before)}
    if isinstance(ai_after, (int, float)):
        roofline_delta["after"] = {"ai": float(ai_after)}

    # ---- Happy path — emit bracket -----------------------------------
    conservative_hi = round(catalog_hi * _CONSERVATIVE_HI_FACTOR, 3)
    lo = round(catalog_lo, 3)
    if conservative_hi < lo:
        # Defensive — catalog malformed; fall back to catalog lo on both ends.
        conservative_hi = lo
    assumptions.append(
        f"catalog bounds = [{catalog_lo:.2f}, {catalog_hi:.2f}] "
        f"(conservative hi = hi × {_CONSERVATIVE_HI_FACTOR})"
    )
    if ktp is not None:
        assumptions.append(f"kernel_time_pct={ktp:.4f}")

    rationale = (
        f"Technique {change_type!r} applied to {kernel_name!r}: expected "
        f"{lo:.2f}-{conservative_hi:.2f}x speedup (conservative bracket). "
        f"Source: {citation}."
    )
    prediction = {
        "predicted_speedup_range": [lo, conservative_hi],
        "confidence": model_confidence,
        "rationale": rationale,
        "roofline_delta": roofline_delta,
        "assumptions": assumptions,
        "source_citation": citation,
        "change_type": change_type,
        "kernel_name": kernel_name,
        "baseline_db": baseline_db,
    }
    effective_params: Dict[str, Any] = {
        "kernel_time_pct": ktp,
        "counter_data_available": True,
    }
    if isinstance(ai_before, (int, float)):
        effective_params["arithmetic_intensity_before"] = float(ai_before)
    if isinstance(ai_after, (int, float)):
        effective_params["arithmetic_intensity_after"] = float(ai_after)
    prediction["prediction_id"] = _cache_prediction(
        prediction,
        effective_params=effective_params,
        catalog_entry=model,
    )
    return prediction


@tool_class(ToolClass.READ_ONLY)
def list_supported_changes() -> List[Dict[str, Any]]:
    """Enumerate every change_type in knowledge/change_impact_models.yaml.

    Returns:
        List of dicts shaped as::

            {
              "id":               str,
              "applies_to":       dict,
              "required_metrics": List[str],
            }
    """
    return [
        {
            "id": m["id"],
            "applies_to": dict(m.get("applies_to") or {}),
            "required_metrics": list(m.get("required_metrics") or []),
        }
        for m in _load_models()
    ]


@tool_class(ToolClass.READ_ONLY)
def explain_prediction(prediction_id: str) -> Dict[str, Any]:
    """Re-hydrate a prediction previously returned by ``predict_change_impact``.

    The in-process cache preserves the exact prediction result. On a cache
    miss, a no-create read of the current project's retained-observation store
    provides a durable fallback when the prediction was explicitly persisted.

    Raises:
        KeyError: when the id is unknown in both the process cache and current
            project scope.
    """
    scope_id = resolve_scope_context().scope_id
    with _STORE_LOCK:
        cached_by_source = _PREDICTION_STORE.get((scope_id, prediction_id))
        if cached_by_source:
            cached = dict(next(reversed(cached_by_source.values())))
            if len(cached_by_source) > 1:
                cached["baseline_db"] = ""
                cached["provenance_redacted"] = True
                cached["provenance_ambiguous"] = True
            return cached

    from perfxpert.retention.recorder import load_persisted_prediction

    retained = load_persisted_prediction(prediction_id)
    if retained is None:
        raise KeyError(
            f"prediction_id {prediction_id!r} not found in the process cache "
            "or current project retention scope"
        )
    with _STORE_LOCK:
        source_key = str(retained.get("baseline_db") or "")
        _PREDICTION_STORE.setdefault((scope_id, prediction_id), {})[
            source_key
        ] = dict(retained)
    return dict(retained)


__all__ = [
    "predict_change_impact",
    "list_supported_changes",
    "explain_prediction",
]
