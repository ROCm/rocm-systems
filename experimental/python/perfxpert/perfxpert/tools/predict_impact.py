"""predict_impact — Change-Impact Prediction (Phase 10, MVP).

Three READ_ONLY MCP tools:

  - predict_change_impact(baseline_db, kernel_name, change_type, change_params)
    Returns a speedup bracket + rationale + roofline delta for a specific
    optimization technique applied to a specific hot kernel.

  - list_supported_changes()
    Returns the set of change_type ids the predictor understands
    (seeded from knowledge/change_impact_models.yaml).

  - explain_prediction(prediction_id)
    Re-hydrates an in-process prediction by its id. Persistence is in-
    process only for the MVP; Phase 11 will wire up a persistent event
    store.

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

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


# ---------------------------------------------------------------------------
# In-process prediction store (Phase 11 will make this durable).
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
_PREDICTION_STORE: Dict[str, Dict[str, Any]] = {}


def _persist_prediction(prediction: Dict[str, Any]) -> str:
    """Stash prediction under a deterministic id, return the id.

    The id is a SHA-1 of the canonical payload so identical calls reuse
    the same key. This is cheap, deterministic, and cross-platform stable.
    """
    canonical = json.dumps(
        {
            "change_type": prediction.get("change_type"),
            "kernel_name": prediction.get("kernel_name"),
            "baseline_db": prediction.get("baseline_db"),
        },
        sort_keys=True,
    )
    pid = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    with _STORE_LOCK:
        _PREDICTION_STORE[pid] = prediction
    return pid


def _reset_store_for_tests() -> None:
    """Private helper — clears the prediction store. Tests only."""
    with _STORE_LOCK:
        _PREDICTION_STORE.clear()


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


_COUNTER_TABLE_CANDIDATES = ("rocpd_counter_values", "counter_values", "COUNTER")


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
        prediction["prediction_id"] = _persist_prediction(prediction)
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
        prediction["prediction_id"] = _persist_prediction(prediction)
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
        prediction["prediction_id"] = _persist_prediction(prediction)
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
    prediction["prediction_id"] = _persist_prediction(prediction)
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

    Phase 10 stores predictions in-process only — the ``prediction_id``
    is only valid for the lifetime of the current Python process. Phase 11
    will wire this to a durable event store.

    Raises:
        KeyError: when the id is unknown (typical cause: cross-process
        call, or caller misplaced the id).
    """
    with _STORE_LOCK:
        if prediction_id not in _PREDICTION_STORE:
            raise KeyError(
                f"prediction_id {prediction_id!r} not found — "
                "predictions persist in-process only in Phase 10."
            )
        return dict(_PREDICTION_STORE[prediction_id])


__all__ = [
    "predict_change_impact",
    "list_supported_changes",
    "explain_prediction",
]
