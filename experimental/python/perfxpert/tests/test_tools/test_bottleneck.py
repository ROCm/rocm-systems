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
