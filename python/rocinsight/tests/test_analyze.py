#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
Tests for the AI analysis module (analyze.py).

Covers:
  - Public API exports and imports
  - generate_recommendations: all 6 Tier-1 rules + 2 Tier-2 rules + boundaries
  - _build_summary: all bottleneck classification branches
  - _build_hw_counters_json: with/without counters
  - _build_warnings_json: both cases
  - _build_recommendations_json: stable IDs, duplicate dedup, unknown category
  - _format_as_json: correct value mapping, idle time, Tier 2, bandwidth conversion
  - format_analysis_output: text, json, and markdown formats
"""

import json
import sys
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _empty_breakdown(**overrides):
    """Return a time_breakdown dict with all fields zeroed unless overridden."""
    base = {
        "total_runtime": 0,
        "total_kernel_time": 0,
        "total_memcpy_time": 0,
        "kernel_percent": 0.0,
        "memcpy_percent": 0.0,
        "overhead_percent": 0.0,
    }
    base.update(overrides)
    return base


def _make_hotspot(
    name="k",
    calls=10,
    total=1_000_000,
    pct=10.0,
    avg=100_000,
    min_d=90_000,
    max_d=110_000,
):
    return {
        "name": name,
        "calls": calls,
        "total_duration": total,
        "avg_duration": avg,
        "min_duration": min_d,
        "max_duration": max_d,
        "percent_of_total": pct,
    }


def _hw_counters(avg_waves=None, gpu_util=None):
    """Build a hardware_counters dict for Tier 2 tests."""
    metrics = {}
    if avg_waves is not None:
        metrics["avg_waves"] = avg_waves
        metrics["max_waves"] = avg_waves * 2
        metrics["min_waves"] = avg_waves / 2
    if gpu_util is not None:
        metrics["gpu_utilization_percent"] = gpu_util
    return {"has_counters": True, "metrics": metrics, "counters": {}, "per_kernel": {}}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_analyze_module_import():
    """Verify analyze module can be imported."""
    from rocinsight import analyze

    assert hasattr(analyze, "compute_time_breakdown")
    assert hasattr(analyze, "identify_hotspots")
    assert hasattr(analyze, "analyze_memory_copies")
    assert hasattr(analyze, "generate_recommendations")
    assert hasattr(analyze, "format_analysis_output")
    assert hasattr(analyze, "add_args")
    assert hasattr(analyze, "execute")
    assert hasattr(analyze, "main")


def test_analyze_module_has_all():
    """Verify analyze module exports expected functions."""
    from rocinsight import analyze

    expected_exports = [
        "compute_time_breakdown",
        "identify_hotspots",
        "analyze_memory_copies",
        "generate_recommendations",
        "format_analysis_output",
        "analyze_performance",
        "add_args",
        "execute",
        "main",
    ]
    for export in expected_exports:
        assert export in analyze.__all__, f"Missing export: {export}"


# ---------------------------------------------------------------------------
# generate_recommendations – Tier 1 rules
# ---------------------------------------------------------------------------


def test_rule1_high_memcpy_fires():
    """Rule 1: memcpy_percent > 20 triggers 'Memory Transfer' HIGH recommendation."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(memcpy_percent=25), [], {})
    matches = [r for r in recs if r["category"] == "Memory Transfer"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "HIGH"
    assert "25.0%" in matches[0]["issue"]


def test_rule1_memcpy_boundary_does_not_fire():
    """Rule 1: memcpy_percent exactly 20 does NOT trigger (threshold is >20)."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(memcpy_percent=20), [], {})
    assert not any(r["category"] == "Memory Transfer" for r in recs)


def test_rule2_api_overhead_fires():
    """Rule 2: overhead_percent > 15 triggers 'API Overhead' MEDIUM recommendation."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(overhead_percent=20), [], {})
    matches = [r for r in recs if r["category"] == "API Overhead"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "MEDIUM"
    assert "20.0%" in matches[0]["issue"]


def test_rule2_overhead_boundary_does_not_fire():
    """Rule 2: overhead_percent exactly 15 does NOT trigger (threshold is >15)."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(overhead_percent=15), [], {})
    assert not any(r["category"] == "API Overhead" for r in recs)


def test_rule3_dominant_kernel_fires():
    """Rule 3: single kernel > 50% triggers 'Compute Bottleneck' HIGH recommendation."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="dominant_kernel", pct=60.0)]
    recs = generate_recommendations(_empty_breakdown(), hotspots, {})
    matches = [r for r in recs if r["category"] == "Compute Bottleneck"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "HIGH"
    assert "dominant_kernel" in matches[0]["issue"]


def test_rule3_dominant_kernel_boundary_does_not_fire():
    """Rule 3: top kernel exactly 50% does NOT trigger (threshold is >50)."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(pct=50.0)]
    recs = generate_recommendations(_empty_breakdown(), hotspots, {})
    assert not any(r["category"] == "Compute Bottleneck" for r in recs)


def test_rule3_uses_hotspot_name_in_commands():
    """Rule 3: the kernel name appears in the rocprofv3 command's full_command."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="my_matmul", pct=75.0)]
    recs = generate_recommendations(_empty_breakdown(), hotspots, {})
    matches = [r for r in recs if r["category"] == "Compute Bottleneck"]
    assert matches
    cmds = matches[0].get("commands", [])
    assert any("my_matmul" in c.get("full_command", "") for c in cmds)


def test_rule3_counter_aware_high_util():
    """Rule 3 + counters: GPU util > 90% → 'compute-bound' suggestion, not 'collect counters'."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="my_kernel", pct=75.0)]
    hw = {
        "has_counters": True,
        "metrics": {"gpu_utilization_percent": 95.0, "avg_waves": 480.0},
    }
    recs = generate_recommendations(_empty_breakdown(), hotspots, {}, hardware_counters=hw)
    matches = [r for r in recs if r["category"] == "Compute Bottleneck"]
    assert len(matches) == 1
    assert "compute-bound" in matches[0]["suggestion"]
    assert "95.0%" in matches[0]["suggestion"]
    # Should NOT say "collect hardware counters" since they're already collected
    assert "collect hardware counters" not in matches[0]["suggestion"].lower()


def test_rule3_counter_aware_low_util():
    """Rule 3 + counters: GPU util < 70% → 'significant room for improvement'."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="my_kernel", pct=75.0)]
    hw = {
        "has_counters": True,
        "metrics": {"gpu_utilization_percent": 45.0, "avg_waves": 100.0},
    }
    recs = generate_recommendations(_empty_breakdown(), hotspots, {}, hardware_counters=hw)
    matches = [r for r in recs if r["category"] == "Compute Bottleneck"]
    assert len(matches) == 1
    assert "significant room" in matches[0]["suggestion"].lower()


def test_rule3_no_counters_asks_to_collect():
    """Rule 3 without counters: falls back to 'collect hardware counters'."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="my_kernel", pct=75.0)]
    recs = generate_recommendations(_empty_breakdown(), hotspots, {})
    matches = [r for r in recs if r["category"] == "Compute Bottleneck"]
    assert len(matches) == 1
    assert "hardware counters" in matches[0]["suggestion"].lower()


def test_rule4_many_small_kernels_fires():
    """Rule 4: >1000 total calls with avg <10μs triggers 'Launch Overhead'."""
    from rocinsight.analyze import generate_recommendations

    # 10 kernels × 200 calls = 2000 launches; 2e10 ns / 2000 = 1e7 ns = 10ms >> 10μs...
    # Need avg < 10μs = 10_000 ns, so total_kernel_time < 2000 * 10_000 = 20_000_000
    td = _empty_breakdown(total_kernel_time=10_000_000)  # avg = 5μs
    hotspots = [_make_hotspot(name=f"k{i}", calls=200) for i in range(10)]
    recs = generate_recommendations(td, hotspots, {})
    matches = [r for r in recs if r["category"] == "Launch Overhead"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "MEDIUM"
    assert "2000" in matches[0]["issue"]


def test_rule4_many_calls_but_large_kernels_does_not_fire():
    """Rule 4: >1000 calls but avg >= 10μs does NOT trigger."""
    from rocinsight.analyze import generate_recommendations

    # 2000 calls but avg = 50ms >> 10μs
    td = _empty_breakdown(total_kernel_time=100_000_000_000)
    hotspots = [_make_hotspot(name=f"k{i}", calls=200) for i in range(10)]
    recs = generate_recommendations(td, hotspots, {})
    assert not any(r["category"] == "Launch Overhead" for r in recs)


def test_rule4_few_calls_does_not_fire():
    """Rule 4: <= 1000 total calls does NOT trigger even if each is short."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_kernel_time=1_000_000)
    hotspots = [_make_hotspot(calls=100)]  # only 100 calls
    recs = generate_recommendations(td, hotspots, {})
    assert not any(r["category"] == "Launch Overhead" for r in recs)


def test_rule5_low_bandwidth_fires():
    """Rule 5: bandwidth < 10 GB/s triggers 'Memory Bandwidth' MEDIUM recommendation."""
    from rocinsight.analyze import generate_recommendations

    mem = {"Host-to-Device": {"bandwidth_bytes_per_sec": 5e9, "avg_bytes": 1024}}
    recs = generate_recommendations(_empty_breakdown(), [], mem)
    matches = [r for r in recs if r["category"] == "Memory Bandwidth"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "MEDIUM"
    assert "Host-to-Device" in matches[0]["issue"]
    assert "5.00 GB/s" in matches[0]["issue"]


def test_rule5_high_bandwidth_does_not_fire():
    """Rule 5: bandwidth >= 10 GB/s does NOT trigger."""
    from rocinsight.analyze import generate_recommendations

    mem = {"Host-to-Device": {"bandwidth_bytes_per_sec": 50e9, "avg_bytes": 1024}}
    recs = generate_recommendations(_empty_breakdown(), [], mem)
    assert not any(r["category"] == "Memory Bandwidth" for r in recs)


def test_rule5_zero_bandwidth_does_not_fire():
    """Rule 5: bandwidth == 0 does NOT trigger (guard: bandwidth_gbps > 0)."""
    from rocinsight.analyze import generate_recommendations

    mem = {"Host-to-Device": {"bandwidth_bytes_per_sec": 0, "avg_bytes": 0}}
    recs = generate_recommendations(_empty_breakdown(), [], mem)
    assert not any(r["category"] == "Memory Bandwidth" for r in recs)


def test_rule5_multiple_directions():
    """Rule 5: each low-bandwidth direction generates its own recommendation."""
    from rocinsight.analyze import generate_recommendations

    mem = {
        "Host-to-Device": {"bandwidth_bytes_per_sec": 2e9, "avg_bytes": 512},
        "Device-to-Host": {"bandwidth_bytes_per_sec": 3e9, "avg_bytes": 512},
    }
    recs = generate_recommendations(_empty_breakdown(), [], mem)
    bw_recs = [r for r in recs if r["category"] == "Memory Bandwidth"]
    assert len(bw_recs) == 2
    directions = {r["issue"].split()[0] for r in bw_recs}
    assert "Host-to-Device" in directions
    assert "Device-to-Host" in directions


def test_rule6_default_info_fires_when_no_rules_trigger():
    """Rule 6: INFO/Performance recommendation emitted when no rules fire."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(), [], {})
    assert len(recs) == 1
    assert recs[0]["priority"] == "INFO"
    assert recs[0]["category"] == "Performance"
    assert len(recs[0].get("commands", [])) > 0


def test_rule6_default_suppressed_when_any_rule_fires():
    """Rule 6: default INFO NOT emitted when at least one other rule fires."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(memcpy_percent=25), [], {})
    assert not any(r["priority"] == "INFO" for r in recs)


def test_multiple_rules_fire_simultaneously():
    """Multiple Tier-1 rules can fire at once; all appear in recommendations."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(memcpy_percent=30, overhead_percent=20)
    recs = generate_recommendations(td, [], {})
    categories = {r["category"] for r in recs}
    assert "Memory Transfer" in categories
    assert "API Overhead" in categories


# ---------------------------------------------------------------------------
# generate_recommendations – Tier 2 rules
# ---------------------------------------------------------------------------


def test_tier2_low_occupancy_fires():
    """Tier 2: avg_waves > 0 and < 16 triggers 'Low Occupancy' HIGH."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(
        _empty_breakdown(), [], {}, _hw_counters(avg_waves=8.0)
    )
    matches = [r for r in recs if r["category"] == "Low Occupancy"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "HIGH"
    assert "8.0" in matches[0]["issue"]


def test_tier2_low_occupancy_boundary_does_not_fire():
    """Tier 2: avg_waves exactly 16 does NOT trigger (threshold is < 16)."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(
        _empty_breakdown(), [], {}, _hw_counters(avg_waves=16.0)
    )
    assert not any(r["category"] == "Low Occupancy" for r in recs)


def test_tier2_zero_waves_does_not_fire():
    """Tier 2: avg_waves == 0 does NOT trigger (guard: avg_waves > 0)."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(_empty_breakdown(), [], {}, _hw_counters(avg_waves=0))
    assert not any(r["category"] == "Low Occupancy" for r in recs)


def test_tier2_low_gpu_utilization_fires():
    """Tier 2: gpu_utilization_percent > 0 and < 70 triggers 'GPU Utilization' MEDIUM."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(
        _empty_breakdown(), [], {}, _hw_counters(gpu_util=50.0)
    )
    matches = [r for r in recs if r["category"] == "GPU Utilization"]
    assert len(matches) == 1
    assert matches[0]["priority"] == "MEDIUM"
    assert "50.0%" in matches[0]["issue"]


def test_tier2_gpu_utilization_boundary_does_not_fire():
    """Tier 2: gpu_utilization exactly 70% does NOT trigger (threshold is < 70)."""
    from rocinsight.analyze import generate_recommendations

    recs = generate_recommendations(
        _empty_breakdown(), [], {}, _hw_counters(gpu_util=70.0)
    )
    assert not any(r["category"] == "GPU Utilization" for r in recs)


def test_tier2_not_activated_when_no_counters():
    """Tier 2 rules do NOT fire when has_counters=False."""
    from rocinsight.analyze import generate_recommendations

    hw = {"has_counters": False}
    recs = generate_recommendations(_empty_breakdown(), [], {}, hardware_counters=hw)
    assert not any(r["category"] in ("Low Occupancy", "GPU Utilization") for r in recs)


def test_tier2_commands_use_valid_tools():
    """Tier 2 recommendations include commands with valid tool names."""
    from rocinsight.analyze import generate_recommendations

    VALID_TOOLS = {"rocprofv3", "rocprof-sys", "rocprof-compute"}
    recs = generate_recommendations(
        _empty_breakdown(),
        [],
        {},
        hardware_counters=_hw_counters(avg_waves=4.0, gpu_util=40.0),
    )
    for rec in recs:
        for cmd in rec.get("commands", []):
            assert cmd["tool"] in VALID_TOOLS, f"Invalid tool: {cmd['tool']!r}"


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_recommendation_structure():
    """Test that recommendations have the expected structure."""
    from rocinsight.analyze import generate_recommendations

    recommendations = generate_recommendations(_empty_breakdown(), [], {})
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    rec = recommendations[0]
    for field in ("priority", "category", "issue", "suggestion"):
        assert field in rec
    assert rec["priority"] in ["HIGH", "MEDIUM", "LOW", "INFO"]


def test_high_memcpy_recommendation():
    """Test that high memory copy overhead triggers recommendation."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(memcpy_percent=35)
    recs = generate_recommendations(td, [], {})
    memcpy_recs = [r for r in recs if "Memory Transfer" in r.get("category", "")]
    assert len(memcpy_recs) > 0
    assert memcpy_recs[0]["priority"] == "HIGH"


def test_hotspot_recommendation():
    """Test that dominant kernel triggers recommendation."""
    from rocinsight.analyze import generate_recommendations

    hotspots = [_make_hotspot(name="test_kernel", pct=60)]
    recs = generate_recommendations(_empty_breakdown(), hotspots, {})
    compute_recs = [r for r in recs if "Compute Bottleneck" in r.get("category", "")]
    assert len(compute_recs) > 0
    assert "test_kernel" in compute_recs[0]["issue"]


# ---------------------------------------------------------------------------
# _build_summary – all bottleneck classification branches
# ---------------------------------------------------------------------------


def test_summary_memory_transfer_high_confidence():
    """memcpy_pct > 30 → memory_transfer with confidence 0.85."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 35, "kernel_percent": 50, "overhead_percent": 15}, [], False
    )
    assert result["primary_bottleneck"] == "memory_transfer"
    assert result["confidence"] == 0.85


def test_summary_memory_transfer_medium_confidence():
    """memcpy_pct 20-30 → memory_transfer with confidence 0.70."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 25, "kernel_percent": 60, "overhead_percent": 15}, [], False
    )
    assert result["primary_bottleneck"] == "memory_transfer"
    assert result["confidence"] == 0.70


def test_summary_latency_bottleneck():
    """overhead_pct > 25 (memcpy < 20) → latency with confidence 0.75."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 10, "kernel_percent": 60, "overhead_percent": 30}, [], False
    )
    assert result["primary_bottleneck"] == "latency"
    assert result["confidence"] == 0.75


def test_summary_compute_with_counters():
    """kernel_pct > 70 + has_counters=True → compute with confidence 0.80."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 5, "kernel_percent": 80, "overhead_percent": 5}, [], True
    )
    assert result["primary_bottleneck"] == "compute"
    assert result["confidence"] == 0.80


def test_summary_compute_without_counters():
    """kernel_pct > 70 + has_counters=False → compute with confidence 0.60."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 5, "kernel_percent": 80, "overhead_percent": 5}, [], False
    )
    assert result["primary_bottleneck"] == "compute"
    assert result["confidence"] == 0.60


def test_summary_mixed_bottleneck():
    """Low percentages all round → mixed with confidence 0.50."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 10, "kernel_percent": 50, "overhead_percent": 10}, [], False
    )
    assert result["primary_bottleneck"] == "mixed"
    assert result["confidence"] == 0.50


def test_summary_top_kernel_in_findings():
    """Top kernel name from hotspots[0] appears in key_findings."""
    from rocinsight.analyze import _build_summary

    hotspots = [_make_hotspot(name="gemm_kernel")]
    result = _build_summary(
        {"memcpy_percent": 5, "kernel_percent": 80, "overhead_percent": 5},
        hotspots,
        False,
    )
    assert any("gemm_kernel" in f for f in result["key_findings"])


def test_summary_empty_hotspots_shows_na():
    """Empty hotspots → top kernel reported as 'N/A' in key_findings."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 5, "kernel_percent": 80, "overhead_percent": 5}, [], False
    )
    assert any("N/A" in f for f in result["key_findings"])


def test_summary_counters_finding_present():
    """has_counters=True adds counter-data finding; False adds Tier 1 note."""
    from rocinsight.analyze import _build_summary

    bd = {"memcpy_percent": 5, "kernel_percent": 50, "overhead_percent": 5}
    with_hw = _build_summary(bd, [], True)
    without_hw = _build_summary(bd, [], False)
    assert any("Hardware counter" in f for f in with_hw["key_findings"])
    assert any("Tier 1" in f for f in without_hw["key_findings"])


def test_summary_has_required_keys():
    """Summary dict contains all required schema keys."""
    from rocinsight.analyze import _build_summary

    result = _build_summary(
        {"memcpy_percent": 10, "kernel_percent": 60, "overhead_percent": 10}, [], False
    )
    for key in (
        "overall_assessment",
        "primary_bottleneck",
        "confidence",
        "key_findings",
    ):
        assert key in result, f"Missing key: {key!r}"
    assert isinstance(result["key_findings"], list)
    assert isinstance(result["confidence"], float)


# ---------------------------------------------------------------------------
# _build_hw_counters_json
# ---------------------------------------------------------------------------


def test_hw_counters_no_counters_structure():
    """has_counters=False returns the correct minimal structure."""
    from rocinsight.analyze import _build_hw_counters_json

    result = _build_hw_counters_json({"has_counters": False})
    assert result == {"has_counters": False, "metrics": None, "counters": None}


def test_hw_counters_empty_dict():
    """Empty dict (no has_counters key) treated as no counters."""
    from rocinsight.analyze import _build_hw_counters_json

    result = _build_hw_counters_json({})
    assert result["has_counters"] is False


def test_hw_counters_with_metrics():
    """has_counters=True maps all metric fields correctly."""
    from rocinsight.analyze import _build_hw_counters_json

    hw = {
        "has_counters": True,
        "metrics": {
            "gpu_utilization_percent": 75.5,
            "avg_waves": 32.0,
            "max_waves": 64.0,
            "min_waves": 8.0,
        },
        "counters": {},
    }
    result = _build_hw_counters_json(hw)
    assert result["has_counters"] is True
    m = result["metrics"]
    assert m["gpu_utilization_pct"] == 75.5
    assert m["avg_waves"] == 32.0
    assert m["max_waves"] == 64.0
    assert m["min_waves"] == 8.0


def test_hw_counters_with_counter_data():
    """Counter stats are mapped with correct types."""
    from rocinsight.analyze import _build_hw_counters_json

    hw = {
        "has_counters": True,
        "metrics": {},
        "counters": {
            "GRBM_COUNT": {
                "sample_count": 100,
                "avg_value": 1000.0,
                "min_value": 900.0,
                "max_value": 1100.0,
                "total_value": 100_000.0,
            }
        },
    }
    result = _build_hw_counters_json(hw)
    ctr = result["counters"]["GRBM_COUNT"]
    assert ctr["sample_count"] == 100
    assert isinstance(ctr["sample_count"], int)
    assert ctr["avg_value"] == 1000.0
    assert isinstance(ctr["avg_value"], float)


# ---------------------------------------------------------------------------
# _build_warnings_json
# ---------------------------------------------------------------------------


def test_warnings_no_counters_emits_warning():
    """has_counters=False → one warning with 'warning' severity."""
    from rocinsight.analyze import _build_warnings_json

    warnings = _build_warnings_json(has_counters=False)
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "warning"
    assert "Tier 1" in warnings[0]["message"]
    assert "recommendation" in warnings[0]


def test_warnings_with_counters_is_empty():
    """has_counters=True → empty warnings list."""
    from rocinsight.analyze import _build_warnings_json

    assert _build_warnings_json(has_counters=True) == []


# ---------------------------------------------------------------------------
# _build_recommendations_json – stable IDs, dedup, unknown category
# ---------------------------------------------------------------------------


def _simple_rec(category, priority="INFO"):
    return {"category": category, "priority": priority, "issue": "x", "suggestion": "y"}


def test_recs_json_stable_ids_for_known_categories():
    """Known categories get their stable ROCPD-*-001 IDs."""
    from rocinsight.analyze import _build_recommendations_json

    expected = {
        "Low Occupancy": "ROCPD-OCCUPANCY-001",
        "GPU Utilization": "ROCPD-UTILIZATION-001",
        "Memory Transfer": "ROCPD-MEMCPY-001",
        "API Overhead": "ROCPD-API-001",
        "Compute Bottleneck": "ROCPD-COMPUTE-001",
        "Launch Overhead": "ROCPD-LAUNCH-001",
        "Memory Bandwidth": "ROCPD-MEMBW-001",
        "Performance": "ROCPD-INFO-001",
    }
    recs = [_simple_rec(cat) for cat in expected]
    out = _build_recommendations_json(recs)
    by_cat = {r["category"]: r["id"] for r in out}
    for cat, expected_id in expected.items():
        assert (
            by_cat[cat] == expected_id
        ), f"{cat}: expected {expected_id}, got {by_cat[cat]}"


def test_recs_json_duplicate_category_gets_incremented_id():
    """Two recs with the same category → IDs end in 001 and 002."""
    from rocinsight.analyze import _build_recommendations_json

    recs = [_simple_rec("Memory Transfer"), _simple_rec("Memory Transfer")]
    out = _build_recommendations_json(recs)
    assert out[0]["id"] == "ROCPD-MEMCPY-001"
    assert out[1]["id"] == "ROCPD-MEMCPY-002"


def test_recs_json_unknown_category_generates_id():
    """Unknown category generates a ROCPD-...-001 style ID from the name."""
    from rocinsight.analyze import _build_recommendations_json

    out = _build_recommendations_json([_simple_rec("Custom Analysis")])
    assert out[0]["id"].startswith("ROCPD-")
    assert out[0]["id"].endswith("-001")


def test_recs_json_preserves_all_fields():
    """_build_recommendations_json preserves all expected fields."""
    from rocinsight.analyze import _build_recommendations_json

    rec = {
        "category": "Performance",
        "priority": "INFO",
        "issue": "test issue",
        "suggestion": "test suggestion",
        "actions": ["do this"],
        "estimated_impact": "5%",
        "commands": [
            {
                "tool": "rocprofv3",
                "full_command": "rocprofv3 -- ./app",
                "description": "d",
                "flags": [],
                "args": [],
            }
        ],
    }
    out = _build_recommendations_json([rec])
    assert out[0]["priority"] == "INFO"
    assert out[0]["issue"] == "test issue"
    assert out[0]["actions"] == ["do this"]
    assert len(out[0]["commands"]) == 1


def test_recs_json_empty_input_returns_empty():
    """Empty input list returns empty output list."""
    from rocinsight.analyze import _build_recommendations_json

    assert _build_recommendations_json([]) == []


# ---------------------------------------------------------------------------
# _format_as_json – value mapping correctness
# ---------------------------------------------------------------------------


def test_format_json_time_breakdown_values():
    """_format_as_json maps time_breakdown keys correctly into execution_breakdown."""
    from rocinsight.analyze import _format_as_json

    td = {
        "total_runtime": 1_000_000_000,
        "total_kernel_time": 800_000_000,
        "total_memcpy_time": 100_000_000,
        "kernel_percent": 80.0,
        "memcpy_percent": 10.0,
        "overhead_percent": 5.0,
    }
    doc = json.loads(_format_as_json(td, [], {}, []))
    eb = doc["execution_breakdown"]
    assert eb["total_runtime_ns"] == 1_000_000_000
    assert eb["kernel_time_ns"] == 800_000_000
    assert eb["memcpy_time_ns"] == 100_000_000
    assert eb["kernel_time_pct"] == 80.0
    assert eb["memcpy_time_pct"] == 10.0
    assert eb["api_overhead_pct"] == 5.0


def test_format_json_idle_time_calculation():
    """Idle time = total − kernel − memcpy − api_overhead, clamped to 0."""
    from rocinsight.analyze import _format_as_json

    td = {
        "total_runtime": 1_000_000_000,  # 1 s
        "total_kernel_time": 600_000_000,  # 600 ms
        "total_memcpy_time": 200_000_000,  # 200 ms
        "kernel_percent": 60.0,
        "memcpy_percent": 20.0,
        "overhead_percent": 10.0,  # 100 ms
    }
    doc = json.loads(_format_as_json(td, [], {}, []))
    eb = doc["execution_breakdown"]
    # api_overhead_ns = 10% of 1_000_000_000 = 100_000_000
    assert eb["api_overhead_ns"] == 100_000_000
    # idle = 1_000_000_000 - 600_000_000 - 200_000_000 - 100_000_000 = 100_000_000
    assert eb["idle_time_ns"] == 100_000_000


def test_format_json_idle_time_clamped_to_zero():
    """Idle time never goes negative (clamped to 0)."""
    from rocinsight.analyze import _format_as_json

    # kernel + memcpy already exceed total_runtime
    td = {
        "total_runtime": 100_000_000,
        "total_kernel_time": 80_000_000,
        "total_memcpy_time": 30_000_000,  # overflows
        "kernel_percent": 80.0,
        "memcpy_percent": 30.0,
        "overhead_percent": 5.0,
    }
    doc = json.loads(_format_as_json(td, [], {}, []))
    assert doc["execution_breakdown"]["idle_time_ns"] >= 0


def test_format_json_hotspot_field_mapping():
    """Hotspot fields are mapped with correct names and types."""
    from rocinsight.analyze import _format_as_json

    hotspots = [
        _make_hotspot(
            name="conv_fwd",
            calls=5,
            total=400_000_000,
            avg=80_000_000,
            min_d=60_000_000,
            max_d=100_000_000,
            pct=40.0,
        ),
    ]
    doc = json.loads(_format_as_json(_empty_breakdown(), hotspots, {}, []))
    hs = doc["hotspots"][0]
    assert hs["rank"] == 1
    assert hs["name"] == "conv_fwd"
    assert hs["calls"] == 5
    assert hs["total_duration_ns"] == 400_000_000
    assert hs["avg_duration_ns"] == 80_000_000.0
    assert hs["min_duration_ns"] == 60_000_000
    assert hs["max_duration_ns"] == 100_000_000
    assert hs["pct_of_total"] == 40.0


def test_format_json_hotspot_rank_increments():
    """Multiple hotspots get ranks 1, 2, 3 in order."""
    from rocinsight.analyze import _format_as_json

    hotspots = [_make_hotspot(name=f"k{i}") for i in range(3)]
    doc = json.loads(_format_as_json(_empty_breakdown(), hotspots, {}, []))
    ranks = [h["rank"] for h in doc["hotspots"]]
    assert ranks == [1, 2, 3]


def test_format_json_memory_bandwidth_gbps_conversion():
    """bandwidth_bytes_per_sec is correctly converted to bandwidth_gbps."""
    from rocinsight.analyze import _format_as_json

    mem = {
        "Host-to-Device": {
            "count": 10,
            "total_bytes": 0,
            "total_duration": 0,
            "avg_bytes": 0,
            "avg_duration": 0,
            "bandwidth_bytes_per_sec": 50e9,  # 50 GB/s
        }
    }
    doc = json.loads(_format_as_json(_empty_breakdown(), [], mem, []))
    bw = doc["memory_analysis"]["Host-to-Device"]["bandwidth_gbps"]
    assert abs(bw - 50.0) < 0.001


def test_format_json_analysis_tier_with_counters():
    """analysis_tier=2 and hardware_counters.has_counters=True when counters present."""
    from rocinsight.analyze import _format_as_json

    hw = {"has_counters": True, "metrics": {}, "counters": {}}
    doc = json.loads(
        _format_as_json(_empty_breakdown(), [], {}, [], hardware_counters=hw)
    )
    assert doc["profiling_info"]["analysis_tier"] == 2
    assert doc["hardware_counters"]["has_counters"] is True


def test_format_json_analysis_tier_without_counters():
    """analysis_tier=1 and hardware_counters.has_counters=False when no counters."""
    from rocinsight.analyze import _format_as_json

    doc = json.loads(_format_as_json(_empty_breakdown(), [], {}, []))
    assert doc["profiling_info"]["analysis_tier"] == 1
    assert doc["hardware_counters"]["has_counters"] is False


def test_format_json_database_path_in_metadata():
    """database_file in metadata reflects the database_path argument."""
    from rocinsight.analyze import _format_as_json

    doc = json.loads(
        _format_as_json(_empty_breakdown(), [], {}, [], database_path="/data/trace.db")
    )
    assert doc["metadata"]["database_file"] == "/data/trace.db"


def test_format_json_schema_version():
    """JSON output always carries schema_version = '0.1.0'."""
    from rocinsight.analyze import _format_as_json

    doc = json.loads(_format_as_json(_empty_breakdown(), [], {}, []))
    assert doc["schema_version"] == "0.1.0"


def test_format_json_analysis_version_in_metadata():
    """metadata.analysis_version = '0.1.0'."""
    from rocinsight.analyze import _format_as_json

    doc = json.loads(_format_as_json(_empty_breakdown(), [], {}, []))
    assert doc["metadata"]["analysis_version"] == "0.1.0"


# ---------------------------------------------------------------------------
# format_analysis_output – text, json, markdown
# ---------------------------------------------------------------------------


def _full_sample_data():
    td = {
        "total_runtime": 1_200_000_000,
        "total_kernel_time": 1_000_000_000,
        "total_memcpy_time": 200_000_000,
        "kernel_percent": 83.3,
        "memcpy_percent": 16.7,
        "overhead_percent": 0.0,
    }
    hotspots = [_make_hotspot(name="kernel_1", calls=100, total=500_000_000, pct=50.0)]
    memory_analysis = {
        "Host-to-Device": {
            "count": 10,
            "total_bytes": 1_048_576,
            "total_duration": 100_000_000,
            "avg_bytes": 104_857,
            "avg_duration": 10_000_000,
            "bandwidth_bytes_per_sec": 10_485_760,
        }
    }
    recommendations = [
        {
            "priority": "INFO",
            "category": "Test",
            "issue": "Test issue",
            "suggestion": "Test suggestion",
            "actions": ["Action 1"],
            "estimated_impact": "5%",
            "commands": [],
        }
    ]
    return td, hotspots, memory_analysis, recommendations


def test_format_output_text():
    """Text format contains all expected section headers and data."""
    from rocinsight.analyze import format_analysis_output

    td, hs, mem, recs = _full_sample_data()
    out = format_analysis_output(
        td, hs, mem, recs, output_format="text", database_path="/test/db.db"
    )
    assert isinstance(out, str)
    assert "ROCPD AI PERFORMANCE ANALYSIS" in out
    assert "TIME BREAKDOWN" in out
    assert "HOTSPOTS" in out
    assert "MEMORY COPY ANALYSIS" in out
    assert "RECOMMENDATIONS" in out
    assert "kernel_1" in out
    assert "Host-to-Device" in out


def test_format_output_text_empty_data():
    """Text format with all-zero data still produces valid output."""
    from rocinsight.analyze import format_analysis_output

    out = format_analysis_output(_empty_breakdown(), [], {}, [], output_format="text")
    assert isinstance(out, str)
    assert "ROCPD AI PERFORMANCE ANALYSIS" in out


def test_format_output_json():
    """JSON format returns valid parseable JSON with required top-level keys."""
    from rocinsight.analyze import format_analysis_output

    td, hs, mem, recs = _full_sample_data()
    out = format_analysis_output(td, hs, mem, recs, output_format="json")
    doc = json.loads(out)
    for key in (
        "schema_version",
        "metadata",
        "hotspots",
        "recommendations",
        "execution_breakdown",
        "hardware_counters",
    ):
        assert key in doc, f"Missing key: {key!r}"


def test_format_output_markdown():
    """Markdown format returns well-structured markdown document."""
    from rocinsight.analyze import format_analysis_output

    td, hs, mem, recs = _full_sample_data()
    out = format_analysis_output(
        td, hs, mem, recs, output_format="markdown", database_path="/test/db.db"
    )
    assert isinstance(out, str)
    assert out.startswith("# ROCInsight AI Performance Analysis")
    assert "## Time Breakdown" in out
    assert "## Top Kernel Hotspots" in out
    assert "## Memory Copy Analysis" in out
    assert "## Recommendations" in out
    assert "kernel_1" in out
    assert "Host-to-Device" in out


def test_format_output_markdown_no_hotspots():
    """Markdown format omits hotspot section when list is empty."""
    from rocinsight.analyze import format_analysis_output

    td, _, mem, recs = _full_sample_data()
    out = format_analysis_output(td, [], mem, recs, output_format="markdown")
    assert "## Top Kernel Hotspots" not in out


def test_format_output_markdown_no_memory():
    """Markdown format omits memory section when analysis is empty."""
    from rocinsight.analyze import format_analysis_output

    td, hs, _, recs = _full_sample_data()
    out = format_analysis_output(td, hs, {}, recs, output_format="markdown")
    assert "## Memory Copy Analysis" not in out


def test_format_output_markdown_with_hardware_counters():
    """Markdown format includes Tier 2 section when hardware counters present."""
    from rocinsight.analyze import format_analysis_output

    td, hs, mem, recs = _full_sample_data()
    hw = {
        "has_counters": True,
        "metrics": {
            "gpu_utilization_percent": 65.0,
            "avg_waves": 24.0,
            "max_waves": 48.0,
        },
        "counters": {},
    }
    out = format_analysis_output(
        td, hs, mem, recs, hardware_counters=hw, output_format="markdown"
    )
    assert "## Hardware Counters (Tier 2)" in out
    assert "65.0%" in out


def test_format_output_unknown_format_falls_back_to_text():
    """Unrecognized format falls back to text output."""
    from rocinsight.analyze import format_analysis_output

    out = format_analysis_output(_empty_breakdown(), [], {}, [], output_format="xml")
    assert "ROCPD AI PERFORMANCE ANALYSIS" in out


# ---------------------------------------------------------------------------
# _filter_rec_commands: PMC counter filtering
# ---------------------------------------------------------------------------


def _pmc_cmd(
    counters="GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES", extra_flags=None, extra_args=None
):
    """Build a minimal rocprofv3 recommendation command with a --pmc arg."""
    flags = ["--sys-trace"] + (extra_flags or [])
    args = [
        {"name": "--pmc", "value": counters},
        {"name": "-d", "value": "./output"},
        {"name": "-o", "value": "profile"},
    ] + (extra_args or [])
    return {
        "tool": "rocprofv3",
        "description": "Collect hardware counters",
        "flags": flags,
        "args": args,
        "full_command": (
            f"rocprofv3 --sys-trace --pmc {counters} -d ./output -o profile -- ./app"
        ),
    }


def test_filter_pmc_all_counters_already_collected_drops_command():
    """When every --pmc counter is already in pmc_events, the command is dropped."""
    from rocinsight.analyze import _filter_rec_commands

    already = frozenset(
        {"--sys-trace", "pmc:GRBM_COUNT", "pmc:GRBM_GUI_ACTIVE", "pmc:SQ_WAVES"}
    )
    result = _filter_rec_commands([_pmc_cmd()], already)
    assert result == [], "Command with all counters already collected should be dropped"


def test_filter_pmc_partial_counters_already_collected_updates_arg():
    """When some --pmc counters are already collected, only new ones remain."""
    from rocinsight.analyze import _filter_rec_commands

    # GRBM_COUNT already collected; GRBM_GUI_ACTIVE and SQ_WAVES are new
    already = frozenset({"--sys-trace", "pmc:GRBM_COUNT"})
    result = _filter_rec_commands([_pmc_cmd()], already)
    assert len(result) == 1
    pmc_arg = next(a for a in result[0]["args"] if a.get("name") == "--pmc")
    remaining = set(pmc_arg["value"].split())
    assert remaining == {"GRBM_GUI_ACTIVE", "SQ_WAVES"}
    assert "GRBM_COUNT" not in pmc_arg["value"]


def test_filter_pmc_partial_updates_full_command():
    """full_command reflects the reduced counter list after partial stripping."""
    from rocinsight.analyze import _filter_rec_commands

    already = frozenset({"--sys-trace", "pmc:GRBM_COUNT"})
    result = _filter_rec_commands([_pmc_cmd()], already)
    assert len(result) == 1
    assert "GRBM_COUNT" not in result[0]["full_command"]
    assert "GRBM_GUI_ACTIVE" in result[0]["full_command"]
    assert "SQ_WAVES" in result[0]["full_command"]


def test_filter_pmc_no_counters_collected_keeps_command_unchanged():
    """When already_collected is empty, the command is returned unchanged."""
    from rocinsight.analyze import _filter_rec_commands

    already = frozenset()
    cmd = _pmc_cmd()
    result = _filter_rec_commands([cmd], already)
    assert len(result) == 1
    assert result[0] is cmd  # exact same object, no copy


def test_filter_pmc_description_note_added():
    """A note listing removed PMC counters is appended to description."""
    from rocinsight.analyze import _filter_rec_commands

    already = frozenset({"--sys-trace", "pmc:GRBM_COUNT"})
    result = _filter_rec_commands([_pmc_cmd()], already)
    assert len(result) == 1
    assert "GRBM_COUNT" in result[0]["description"]
    assert "Already collected" in result[0]["description"]


def test_filter_pmc_kernel_names_alone_not_meaningful():
    """--kernel-names is a scope filter; command with only scope+output args is dropped."""
    from rocinsight.analyze import _filter_rec_commands

    cmd = _pmc_cmd(
        counters="GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES",
        extra_args=[{"name": "--kernel-names", "value": "my_kernel"}],
    )
    cmd["full_command"] = (
        "rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"
        ' --kernel-names "my_kernel" -d ./output -o profile -- ./app'
    )
    # All three counters already collected + sys-trace → nothing new
    already = frozenset(
        {"--sys-trace", "pmc:GRBM_COUNT", "pmc:GRBM_GUI_ACTIVE", "pmc:SQ_WAVES"}
    )
    result = _filter_rec_commands([cmd], already)
    assert result == [], "Command with only scope+output args remaining should be dropped"


def test_filter_pmc_rocprof_compute_always_kept():
    """rocprof-compute commands are never dropped, even when counters are collected."""
    from rocinsight.analyze import _filter_rec_commands

    compute_cmd = {
        "tool": "rocprof-compute",
        "description": "Roofline model analysis",
        "flags": [],
        "args": [{"name": "profile", "value": None}],
        "full_command": "rocprof-compute profile -- ./app",
    }
    already = frozenset(
        {"--sys-trace", "pmc:GRBM_COUNT", "pmc:GRBM_GUI_ACTIVE", "pmc:SQ_WAVES"}
    )
    result = _filter_rec_commands([compute_cmd], already)
    assert len(result) == 1
    assert result[0] is compute_cmd


# ---------------------------------------------------------------------------
# Tier 3 ATT: analyze_thread_trace + generate_recommendations ATT rules
# ---------------------------------------------------------------------------


def _att_csv_content(rows):
    """Build CSV text for a stats_*.csv file given a list of row dicts."""
    header = "Instruction ID,Hitcount,Latency (cycles),Stall cycles,Source line"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['pc']},{r['hitcount']},{r['latency']},{r['stall']},{r.get('src', '')}"
        )
    return "\n".join(lines) + "\n"


def test_att_missing_directory():
    """analyze_thread_trace returns has_att_data=False when directory is missing."""
    from rocinsight.analyze import analyze_thread_trace

    result = analyze_thread_trace("/nonexistent_dir_xyzzy")
    assert result["has_att_data"] is False
    assert "not found" in result["reason"].lower()
    assert result["kernels"] == []


def test_att_empty_directory():
    """analyze_thread_trace returns has_att_data=False when no CSVs are present."""
    import tempfile

    from rocinsight.analyze import analyze_thread_trace

    with tempfile.TemporaryDirectory() as d:
        result = analyze_thread_trace(d)
    assert result["has_att_data"] is False
    assert "stats_*.csv" in result["reason"]


def test_att_single_kernel_high_stall():
    """analyze_thread_trace parses a CSV and identifies high VMEM stall."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace

    rows = [
        {
            "pc": "0x0000",
            "hitcount": 8192,
            "latency": 200,
            "stall": 180,
            "src": "k.hip:10",
        },
        {"pc": "0x0004", "hitcount": 8192, "latency": 40, "stall": 5, "src": "k.hip:11"},
    ]
    with tempfile.TemporaryDirectory() as d:
        csv_path = pathlib.Path(d) / "stats_my_kernel.csv"
        csv_path.write_text(_att_csv_content(rows))
        result = analyze_thread_trace(d)

    assert result["has_att_data"] is True
    assert len(result["kernels"]) == 1
    k = result["kernels"][0]
    assert k["name"] == "my_kernel"
    assert k["stall_category"] == "att_vmem_latency"  # 0x0000 → VMEM by default
    assert k["avg_stall_ratio"] > 0.0
    # Top instruction is the worst (180/200 = 0.90)
    top = k["top_stalling_instructions"][0]
    assert top["pc_offset"] == "0x0000"
    assert abs(top["stall_ratio"] - 0.90) < 0.01
    assert top["weighted_stall"] == 180 * 8192


def test_att_stall_ratio_threshold_for_recommendations():
    """generate_recommendations emits HIGH rec when stall_ratio >= 0.60 and hitcount >= 6400."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace, generate_recommendations

    rows = [
        # stall_ratio = 160/200 = 0.80 → HIGH; hitcount = 8192 > 6400
        {
            "pc": "0x0000",
            "hitcount": 8192,
            "latency": 200,
            "stall": 160,
            "src": "k.hip:5",
        },
    ]
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_hot_kernel.csv").write_text(_att_csv_content(rows))
        att = analyze_thread_trace(d)

    recs = generate_recommendations(
        _empty_breakdown(kernel_percent=80),
        [],
        {},
        att_analysis=att,
    )
    att_recs = [r for r in recs if "ATT" in r["category"]]
    assert len(att_recs) == 1
    assert att_recs[0]["priority"] == "HIGH"
    assert "hot_kernel" in att_recs[0]["issue"]
    assert "0x0000" in att_recs[0]["issue"]
    assert "k.hip:5" in att_recs[0]["issue"]
    # Recommendation must include an ATT re-collection command
    cmds = att_recs[0]["commands"]
    assert any(c.get("tool") == "rocprofv3" for c in cmds)


def test_att_stall_ratio_medium_threshold():
    """generate_recommendations emits MEDIUM rec when 0.40 <= stall_ratio < 0.60."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace, generate_recommendations

    rows = [
        # stall_ratio = 100/200 = 0.50 → MEDIUM
        {"pc": "0x0000", "hitcount": 8192, "latency": 200, "stall": 100},
    ]
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_medium_kernel.csv").write_text(_att_csv_content(rows))
        att = analyze_thread_trace(d)

    recs = generate_recommendations(
        _empty_breakdown(kernel_percent=80),
        [],
        {},
        att_analysis=att,
    )
    att_recs = [r for r in recs if "ATT" in r["category"]]
    assert len(att_recs) == 1
    assert att_recs[0]["priority"] == "MEDIUM"


def test_att_below_hitcount_threshold_no_rec():
    """generate_recommendations does NOT emit rec when hitcount < 6400 (statistically unreliable)."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace, generate_recommendations

    rows = [
        # hitcount = 64 < _ATT_MIN_HITCOUNT (6400); stall_ratio = 0.95 → would be HIGH otherwise
        {"pc": "0x0000", "hitcount": 64, "latency": 100, "stall": 95},
    ]
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_tiny_kernel.csv").write_text(_att_csv_content(rows))
        att = analyze_thread_trace(d)

    recs = generate_recommendations(
        _empty_breakdown(kernel_percent=80),
        [],
        {},
        att_analysis=att,
    )
    att_recs = [r for r in recs if "ATT" in r["category"]]
    # Below hitcount threshold: no HIGH/MEDIUM rec, but an INFO rec is emitted
    # confirming ATT ran and found no significant stalls.
    assert len(att_recs) == 1
    assert att_recs[0]["priority"] == "INFO"


def test_att_below_stall_ratio_threshold_no_rec():
    """generate_recommendations does NOT emit rec when stall_ratio < 0.40."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace, generate_recommendations

    rows = [
        # stall_ratio = 30/200 = 0.15 → below threshold
        {"pc": "0x0000", "hitcount": 8192, "latency": 200, "stall": 30},
    ]
    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_compute_kernel.csv").write_text(_att_csv_content(rows))
        att = analyze_thread_trace(d)

    recs = generate_recommendations(
        _empty_breakdown(kernel_percent=80),
        [],
        {},
        att_analysis=att,
    )
    att_recs = [r for r in recs if "ATT" in r["category"]]
    # Below stall_ratio threshold: no HIGH/MEDIUM rec, but an INFO rec is emitted
    # confirming ATT ran and found no significant stalls.
    assert len(att_recs) == 1
    assert att_recs[0]["priority"] == "INFO"


def test_att_multiple_kernels_sorted_by_weighted_stall():
    """analyze_thread_trace sorts kernels by total_weighted_stall descending."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace

    with tempfile.TemporaryDirectory() as d:
        # Kernel A: small weighted stall
        (pathlib.Path(d) / "stats_kernel_a.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 1000, "latency": 100, "stall": 50}]
            )
        )
        # Kernel B: large weighted stall
        (pathlib.Path(d) / "stats_kernel_b.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 100000, "latency": 200, "stall": 190}]
            )
        )
        result = analyze_thread_trace(d)

    assert result["has_att_data"] is True
    assert len(result["kernels"]) == 2
    # Kernel B should be first (larger weighted stall)
    assert result["kernels"][0]["name"] == "kernel_b"
    assert result["kernels"][1]["name"] == "kernel_a"


def test_att_summary_counts():
    """analyze_thread_trace summary counts high-stall kernels correctly."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace

    with tempfile.TemporaryDirectory() as d:
        # High stall (stall_ratio=0.90, hitcount=8192)
        (pathlib.Path(d) / "stats_kernel_high.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 8192, "latency": 100, "stall": 90}]
            )
        )
        # Low stall (stall_ratio=0.10, hitcount=8192)
        (pathlib.Path(d) / "stats_kernel_low.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 8192, "latency": 100, "stall": 10}]
            )
        )
        result = analyze_thread_trace(d)

    assert result["summary"]["kernel_count"] == 2
    assert result["summary"]["high_stall_kernels"] == 1


def test_att_json_output_includes_att_trace_field():
    """_format_as_json includes att_trace key and bumps schema_version to 0.4.0."""
    import json
    import pathlib
    import tempfile

    from rocinsight.analyze import _format_as_json, analyze_thread_trace

    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_k.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 8192, "latency": 100, "stall": 80}]
            )
        )
        att = analyze_thread_trace(d)

    output = _format_as_json(
        _empty_breakdown(),
        [],
        {},
        [],
        att_analysis=att,
    )
    doc = json.loads(output)
    assert (
        "att_trace" in doc
    ), "att_trace key should appear in JSON when ATT data is present"
    assert doc["schema_version"] == "0.4.0"
    assert doc["profiling_info"]["analysis_tier"] == 3


def test_att_json_output_no_att_trace_without_data():
    """_format_as_json does NOT include att_trace when att_analysis is None."""
    import json

    from rocinsight.analyze import _format_as_json

    output = _format_as_json(_empty_breakdown(), [], {}, [], att_analysis=None)
    doc = json.loads(output)
    assert "att_trace" not in doc
    assert doc["schema_version"] == "0.1.0"


def test_att_text_output_shows_att_section():
    """format_analysis_output text format shows the ATT section when ATT data present."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace, format_analysis_output

    with tempfile.TemporaryDirectory() as d:
        (pathlib.Path(d) / "stats_stall_kernel.csv").write_text(
            _att_csv_content(
                [
                    {
                        "pc": "0x0000",
                        "hitcount": 8192,
                        "latency": 200,
                        "stall": 180,
                        "src": "my.hip:42",
                    }
                ]
            )
        )
        att = analyze_thread_trace(d)

    output = format_analysis_output(
        _empty_breakdown(kernel_percent=90),
        [],
        {},
        [],
        att_analysis=att,
    )
    assert "TIER 3" in output
    assert "ATT" in output
    assert "stall_kernel" in output


def test_att_malformed_csv_skipped_gracefully():
    """analyze_thread_trace skips malformed CSVs without crashing."""
    import pathlib
    import tempfile

    from rocinsight.analyze import analyze_thread_trace

    with tempfile.TemporaryDirectory() as d:
        # Completely invalid CSV content
        (pathlib.Path(d) / "stats_bad_kernel.csv").write_text("not,valid,csv\nfoo,bar")
        # Valid CSV alongside it
        (pathlib.Path(d) / "stats_good_kernel.csv").write_text(
            _att_csv_content(
                [{"pc": "0x0000", "hitcount": 8192, "latency": 100, "stall": 80}]
            )
        )
        result = analyze_thread_trace(d)

    assert result["has_att_data"] is True
    # Only the good kernel should appear (bad one has no parseable rows after header)
    kernel_names = [k["name"] for k in result["kernels"]]
    assert "good_kernel" in kernel_names


# ---------------------------------------------------------------------------
# generate_recommendations – Rule 2a: Init-overhead guard
# ---------------------------------------------------------------------------


def test_init_overhead_short_run_low_kernel_pct():
    """Rule 2a: short run (<1s) with <5% kernel → init overhead, not API overhead."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_runtime=500_000_000, overhead_percent=95.0, kernel_percent=2.0)
    recs = generate_recommendations(td, [], {})
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    api_recs = [r for r in recs if r["category"] == "API Overhead"]
    assert len(init_recs) == 1
    assert len(api_recs) == 0  # suppressed by init-overhead guard


def test_init_overhead_long_run_not_triggered():
    """Rule 2a: long run (>1s) with low kernel% does NOT trigger init guard."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_runtime=10_000_000_000, overhead_percent=85.0, kernel_percent=2.0)
    recs = generate_recommendations(td, [], {})
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    assert len(init_recs) == 0  # not init-dominated — it's a real overhead problem


def test_init_overhead_short_run_normal_kernel_pct():
    """Rule 2a: short run but >5% kernel → NOT init overhead."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_runtime=800_000_000, overhead_percent=50.0, kernel_percent=40.0)
    recs = generate_recommendations(td, [], {})
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    assert len(init_recs) == 0


def test_init_overhead_zero_runtime_guard():
    """Rule 2a: total_runtime=0 does NOT trigger (division by zero guard)."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_runtime=0, kernel_percent=0.0)
    recs = generate_recommendations(td, [], {})
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    assert len(init_recs) == 0


def test_init_overhead_suppresses_api_overhead_rule():
    """When init-overhead fires, API overhead rule is suppressed even at 90%+ overhead."""
    from rocinsight.analyze import generate_recommendations

    td = _empty_breakdown(total_runtime=300_000_000, overhead_percent=92.0, kernel_percent=1.0)
    recs = generate_recommendations(td, [], {})
    api_recs = [r for r in recs if r["category"] == "API Overhead"]
    assert len(api_recs) == 0  # suppressed


# ---------------------------------------------------------------------------
# generate_recommendations – Rule 2a: Init-overhead suppressed by prior recs
# ---------------------------------------------------------------------------


def test_init_overhead_not_emitted_when_other_recs_exist():
    """Init-overhead INFO should not fire when higher-priority recs already exist.

    When _is_init_overhead is True but memcpy_percent > 20 fires first (Rule 1),
    the 'not recommendations' guard prevents the init-overhead INFO rec from
    being emitted.
    """
    from rocinsight.analyze import generate_recommendations

    # Short run + low kernel% → _is_init_overhead = True
    # But memcpy_percent=30 → Memory Transfer HIGH fires first (Rule 1)
    td = _empty_breakdown(
        total_runtime=500_000_000,
        kernel_percent=2.0,
        memcpy_percent=30.0,
        overhead_percent=68.0,
    )
    recs = generate_recommendations(td, [], {})
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    assert len(init_recs) == 0, "Init-overhead should be suppressed when memcpy HIGH already fired"
    # Verify memcpy HIGH DID fire
    memcpy_recs = [r for r in recs if r["category"] == "Memory Transfer"]
    assert len(memcpy_recs) == 1


def test_init_overhead_not_emitted_when_low_occupancy_rec_exists():
    """Init-overhead INFO should not fire when Tier 2 low occupancy HIGH rec exists.

    Tier 2 rules (low occupancy, GPU utilization) run before Rule 2a, so if
    they fire, the 'not recommendations' guard suppresses init-overhead.
    """
    from rocinsight.analyze import generate_recommendations

    # Short run + low kernel% → _is_init_overhead = True
    # But hw counters with low waves → Tier 2 LOW OCCUPANCY fires first
    td = _empty_breakdown(
        total_runtime=500_000_000,
        kernel_percent=2.0,
        memcpy_percent=0.0,
        overhead_percent=98.0,
    )
    hw = _hw_counters(avg_waves=5, gpu_util=80.0)
    recs = generate_recommendations(td, [], {}, hardware_counters=hw)
    init_recs = [r for r in recs if r["category"] == "Runtime Initialization"]
    assert len(init_recs) == 0, "Init-overhead suppressed when Low Occupancy already fired"
    # Verify the Tier 2 rec DID fire
    occ_recs = [r for r in recs if r["category"] == "Low Occupancy"]
    assert len(occ_recs) == 1


# ---------------------------------------------------------------------------
# Editor validation in _apply_code_change_interactive
# ---------------------------------------------------------------------------


def test_editor_not_found_skips_subprocess():
    """When shutil.which returns None for $EDITOR, subprocess.run must NOT be called."""
    from unittest.mock import patch, MagicMock
    from rocinsight.analyze import _apply_code_change_interactive

    rec = {
        "category": "Compute Bottleneck",
        "issue": "Kernel is slow",
        "suggestion": "Optimize it",
        "actions": ["Do something"],
        "estimated_impact": "20%",
    }
    colors = {"C": "", "G": "", "Y": "", "R": "", "DIM": "", "N": ""}

    with (
        patch("shutil.which", return_value=None) as mock_which,
        patch("subprocess.run") as mock_run,
        patch("os.environ.get", side_effect=lambda k, d="": "fake-editor" if k == "EDITOR" else d),
        patch("builtins.input", return_value="y"),
        patch("glob.glob", return_value=["/tmp/src/app.hip"]),
        patch("os.path.isfile", return_value=True),
    ):
        _apply_code_change_interactive(
            rec=rec,
            source_dir="/tmp/src",
            llm_provider=None,
            llm_api_key=None,
            llm_model=None,
            colors=colors,
        )
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use --noconftest to avoid loading conftest.py which requires rocprofiler_sdk module
    exit_code = pytest.main(["--noconftest", "-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
