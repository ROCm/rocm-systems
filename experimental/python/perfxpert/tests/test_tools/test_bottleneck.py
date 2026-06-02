"""Tests for perfxpert.tools.bottleneck."""

import pytest

from perfxpert.tools import bottleneck
from perfxpert.tools._class import ToolClass


# -- classify_from_metrics ---------------------------------------------------

def test_classify_compute_bound_kernel():
    metrics = {
        "valu_util_pct": 0.85,
        "mfma_util_pct": 0.60,
        "arithmetic_intensity_above_ridge": 1,
        "memcpy_pct": 0.05,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "compute"
    assert result["confidence"] >= 0.7


def test_classify_memory_bound_kernel():
    metrics = {
        "valu_util_pct": 0.35,
        "memcpy_pct": 0.45,
        "hbm_bw_utilization": 0.18,
        "arithmetic_intensity_below_ridge": 1,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "memory_transfer"


def test_classify_latency_bound_kernel():
    metrics = {
        "avg_waves_per_cu": 8,
        "gpu_util_pct": 0.40,
        "occupancy_pct": 0.35,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "latency"


def test_classify_api_overhead_dominated():
    metrics = {
        "api_overhead_pct": 0.30,
        "avg_kernel_duration_us": 4,
        "total_kernel_calls": 5000,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "api_overhead"


def test_classify_mixed_when_no_dominant():
    metrics = {
        "valu_util_pct": 0.40,
        "memcpy_pct": 0.10,
        "gpu_util_pct": 0.65,
        "api_overhead_pct": 0.05,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "mixed"


def test_data_insufficient_when_no_counters():
    """Empty metrics dict (all None / missing) must return data_insufficient, not mixed@0.5."""
    # All hardware-counter keys set to None — exactly what _collect_deterministic_metrics
    # produces when counter_data_available=False.
    metrics = {
        "valu_util_pct": None,
        "mfma_util_pct": None,
        "arithmetic_intensity_above_ridge": None,
        "arithmetic_intensity_below_ridge": None,
        "occupancy_pct": None,
        "avg_waves_per_cu": None,
        "gpu_util_pct": None,
        "hbm_bw_utilization": None,
        "no_dominant_bottleneck": None,
        "total_kernel_calls": None,
        "avg_kernel_duration_us": None,
    }
    result = bottleneck.classify_from_metrics(metrics)
    assert result["type"] == "data_insufficient", (
        f"Expected 'data_insufficient' but got '{result['type']}'; "
        "classifier must not produce silent mixed@0.5 when flying blind."
    )
    assert result["confidence"] == 0.0
    assert "data_insufficient" in result["type"]


def test_data_insufficient_empty_dict():
    """Completely empty dict must also return data_insufficient."""
    result = bottleneck.classify_from_metrics({})
    assert result["type"] == "data_insufficient"
    assert result["confidence"] == 0.0


def test_mixed_returned_when_data_available_but_no_dominant():
    """With some data present but no dominant bottleneck, mixed is still returned (not data_insufficient)."""
    metrics = {
        "valu_util_pct": 0.40,      # present but fails compute threshold
        "memcpy_pct": 0.10,          # present but fails memory threshold
        "gpu_util_pct": 0.65,        # present but fails latency threshold
        "api_overhead_pct": 0.05,    # present but fails api threshold
    }
    result = bottleneck.classify_from_metrics(metrics)
    # Must be mixed, not data_insufficient, because metrics ARE available
    assert result["type"] == "mixed"
    assert result["confidence"] == 0.5


def test_classification_is_deterministic():
    metrics = {"valu_util_pct": 0.85, "mfma_util_pct": 0.60, "arithmetic_intensity_above_ridge": 1}
    r1 = bottleneck.classify_from_metrics(metrics)
    r2 = bottleneck.classify_from_metrics(metrics)
    assert r1 == r2


def test_classify_is_read_only_class():
    assert bottleneck.classify_from_metrics.__tool_class__ == ToolClass.READ_ONLY


# -- lookup_signatures ------------------------------------------------------

def test_lookup_signatures_returns_entry_for_compute():
    sig = bottleneck.lookup_signatures("compute")
    assert sig["name"] == "compute"
    assert len(sig["signatures"]) >= 1


def test_lookup_signatures_unknown_raises():
    with pytest.raises(KeyError):
        bottleneck.lookup_signatures("nonexistent")


# -- prioritize_by_amdahl ---------------------------------------------------

def test_amdahl_above_threshold_high_priority():
    result = bottleneck.prioritize_by_amdahl(execution_time_pct=0.35)
    assert result["priority"] == "high"


def test_amdahl_middle_medium_priority():
    result = bottleneck.prioritize_by_amdahl(execution_time_pct=0.07)
    assert result["priority"] == "medium"


def test_amdahl_below_threshold_low_priority():
    result = bottleneck.prioritize_by_amdahl(execution_time_pct=0.03)
    assert result["priority"] == "low"


# -- Trace-only memcpy path (FINDING #22) ------------------------------------

def test_classify_trace_only_memcpy_above_threshold_returns_non_data_insufficient():
    """Tier 1 trace only — only memcpy_pct extracted (no PMC counters).

    When memcpy_pct alone is high (> 0.20 fraction, i.e. 20%), the classifier
    has real evidence and must NOT return 'data_insufficient'.  With the
    evidence-strength weighting fix (finding #26), a single rule from a 3-rule
    signature scores 1/3 of max confidence = 0.333, which is below the 0.5
    threshold — so the classifier correctly returns 'mixed' rather than
    'memory_transfer'.  This is the correct behavior: single-rule evidence is
    too sparse to make a confident memory_transfer claim.

    NOTE: the agentic path (run_analysis) currently forces data_insufficient
    for all no-PMC traces by design. This test validates the pure rule-based
    classify_from_metrics() tool — which has no such override — so that the
    tool itself remains correct and does not return data_insufficient (it
    returns mixed, indicating uncertainty).
    """
    # User has Tier 1 trace only. Only memcpy_pct extracted. No PMC counters.
    result = bottleneck.classify_from_metrics({"memcpy_pct": 0.25})
    # Must NOT be data_insufficient — we have real evidence (memcpy is high).
    # With evidence weighting, 1/3 rules → confidence 0.33 → returns 'mixed'.
    assert result["type"] != "data_insufficient", (
        f"Got 'data_insufficient' for memcpy_pct=0.25 (25% > 20% threshold). "
        "The classifier must not report data_insufficient when concrete evidence "
        "is present — it should return 'mixed' (insufficient evidence for a "
        "confident classification)."
    )
    # After finding #26 fix: single-rule evidence returns 'mixed' (not memory_transfer).
    # This is correct — 1/3 rules evaluated means we cannot confidently classify.
    assert result["type"] == "mixed", (
        f"Expected 'mixed' for single-rule memcpy_pct=0.25 "
        f"(evidence-weighted confidence 0.33 < 0.5 threshold), got {result['type']!r}."
    )


# -- Finding #26: Partial-PMC confidence weighting (FIXED) -------------------

def test_partial_pmc_rules_confidence_lower_than_full_pmc():
    """2-of-3 compute rules matching must return strictly lower confidence than 3-of-3.

    Finding #26 FIX (evidence-strength × match-ratio):
      partial (2/3 rules): confidence = (2/2) × (2/3) = 0.667
      full    (3/3 rules): confidence = (3/3) × (3/3) = 1.0

    This replaces the old normalize-by-evaluated formula that returned 1.0 for
    both cases.
    """
    # 2 of 3 compute rules evaluated (valu_util + ai_above_ridge, mfma absent)
    partial = bottleneck.classify_from_metrics({
        "valu_util_pct": 0.85,
        "arithmetic_intensity_above_ridge": True,
        # mfma_util_pct absent → not evaluated
    })

    # 3 of 3 compute rules evaluated and all match
    full = bottleneck.classify_from_metrics({
        "valu_util_pct": 0.85,
        "mfma_util_pct": 0.70,
        "arithmetic_intensity_above_ridge": True,
    })

    # Both should classify as compute
    assert partial["type"] == "compute", (
        f"partial metrics should still classify as compute; got {partial['type']!r}"
    )
    assert full["type"] == "compute", (
        f"full metrics should classify as compute; got {full['type']!r}"
    )

    # Full evidence must be strictly higher than partial evidence.
    assert full["confidence"] > partial["confidence"], (
        f"Full evidence (3/3 rules, confidence={full['confidence']:.3f}) must be "
        f"strictly higher than partial evidence (2/3 rules, confidence={partial['confidence']:.3f}). "
        "Finding #26 fix: confidence = match_ratio × evidence_factor."
    )

    # Exact expected values (3-rule signature):
    # partial: 2/2 match × 2/3 evidence = 0.667
    # full:    3/3 match × 3/3 evidence = 1.0
    assert abs(partial["confidence"] - 2 / 3) < 0.01, (
        f"partial confidence expected ≈ 0.667, got {partial['confidence']:.3f}"
    )
    assert abs(full["confidence"] - 1.0) < 0.01, (
        f"full confidence expected 1.0, got {full['confidence']:.3f}"
    )


def test_single_rule_match_returns_mixed_not_full_confidence():
    """Single-rule match must return confidence ≤ 0.5 (→ 'mixed'), not 1.0.

    Finding #26: before fix, a single memcpy_pct rule from a 3-rule signature
    returned confidence 1/1 = 1.0 → classified as memory_transfer with full
    confidence.  After fix: 1/3 evidence_factor × 1.0 match_ratio = 0.333 < 0.5
    → returns 'mixed'.  A single weak signal cannot justify a confident bottleneck
    claim.
    """
    # Only memcpy_pct provided — 1 of 3 memory_transfer rules evaluable.
    result = bottleneck.classify_from_metrics({"memcpy_pct": 0.25})

    # Must return 'mixed': insufficient evidence for a confident classification.
    assert result["type"] == "mixed", (
        f"Expected 'mixed' (confidence below 0.5 threshold), got {result['type']!r} "
        f"with confidence={result['confidence']:.3f}. "
        "Single-rule match must not produce a confident bottleneck classification."
    )
    # Confidence must be below the 0.5 classification threshold.
    assert result["confidence"] <= 0.5, (
        f"Single-rule match confidence={result['confidence']:.3f} must be ≤ 0.5. "
        "Finding #26 fix: evidence-strength factor penalises sparse evidence."
    )
