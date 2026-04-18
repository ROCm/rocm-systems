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

def test_classify_trace_only_memcpy_above_threshold_returns_memory_transfer():
    """Tier 1 trace only — only memcpy_pct extracted (no PMC counters).

    When memcpy_pct alone is high (> 0.20 fraction, i.e. 20%), the classifier
    has real evidence and must return 'memory_transfer', NOT 'data_insufficient'.
    This guards against silently degrading Tier-1-only traces.

    NOTE: the agentic path (run_analysis) currently forces data_insufficient
    for all no-PMC traces as a Phase 6 design decision. This test validates
    the pure rule-based classify_from_metrics() tool — which has no such
    override — so that the tool itself remains correct and Tier-1 signal
    is not lost at the tool level.
    """
    # User has Tier 1 trace only. Only memcpy_pct extracted. No PMC counters.
    result = bottleneck.classify_from_metrics({"memcpy_pct": 0.25})
    assert result["type"] == "memory_transfer", (
        f"Expected 'memory_transfer' for memcpy_pct=0.25 (25% > 20% threshold), "
        f"got {result['type']!r}. The classifier must not report data_insufficient "
        "when concrete memcpy evidence is present."
    )
    # Not data_insufficient — we have real evidence (memcpy is high)
    assert result["type"] != "data_insufficient"
    assert result["confidence"] >= 0.5
