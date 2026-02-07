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
Tests for the AI analysis module.
"""

import sys
import pytest


def test_analyze_module_import():
    """Verify analyze module can be imported."""
    try:
        from rocpd import analyze

        assert hasattr(analyze, "compute_time_breakdown")
        assert hasattr(analyze, "identify_hotspots")
        assert hasattr(analyze, "analyze_memory_copies")
        assert hasattr(analyze, "generate_recommendations")
        assert hasattr(analyze, "format_analysis_output")
        assert hasattr(analyze, "add_args")
        assert hasattr(analyze, "execute")
        assert hasattr(analyze, "main")
    except ImportError as e:
        pytest.fail(f"Failed to import analyze module: {e}")


def test_analyze_module_has_all():
    """Verify analyze module exports expected functions."""
    from rocpd import analyze

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


def test_recommendation_structure():
    """Test that recommendations have the expected structure."""
    from rocpd.analyze import generate_recommendations

    # Test with empty data
    time_breakdown = {
        "kernel_percent": 0,
        "memcpy_percent": 0,
        "overhead_percent": 0,
    }
    hotspots = []
    memory_analysis = {}

    recommendations = generate_recommendations(time_breakdown, hotspots, memory_analysis)

    assert isinstance(recommendations, list)
    assert len(recommendations) > 0, "Should have at least one recommendation"

    # Check structure of first recommendation
    rec = recommendations[0]
    assert "priority" in rec
    assert "category" in rec
    assert "issue" in rec
    assert "suggestion" in rec

    # Verify valid priority
    assert rec["priority"] in ["HIGH", "MEDIUM", "LOW", "INFO"]


def test_high_memcpy_recommendation():
    """Test that high memory copy overhead triggers recommendation."""
    from rocpd.analyze import generate_recommendations

    time_breakdown = {
        "kernel_percent": 50,
        "memcpy_percent": 35,  # >20%, should trigger recommendation
        "overhead_percent": 15,
    }
    hotspots = []
    memory_analysis = {}

    recommendations = generate_recommendations(time_breakdown, hotspots, memory_analysis)

    # Should have recommendation about memory transfers
    memcpy_recs = [r for r in recommendations if "Memory Transfer" in r.get("category", "")]
    assert len(memcpy_recs) > 0, "Should have memory transfer recommendation"
    assert memcpy_recs[0]["priority"] == "HIGH"


def test_hotspot_recommendation():
    """Test that dominant kernel triggers recommendation."""
    from rocpd.analyze import generate_recommendations

    time_breakdown = {
        "kernel_percent": 80,
        "memcpy_percent": 10,
        "overhead_percent": 10,
    }

    # Single kernel dominates
    hotspots = [
        {
            "name": "test_kernel",
            "calls": 100,
            "total_duration": 1000000,
            "percent_of_total": 60,  # >50%, should trigger
        }
    ]
    memory_analysis = {}

    recommendations = generate_recommendations(time_breakdown, hotspots, memory_analysis)

    # Should have recommendation about compute bottleneck
    compute_recs = [r for r in recommendations if "Compute Bottleneck" in r.get("category", "")]
    assert len(compute_recs) > 0, "Should have compute bottleneck recommendation"
    assert "test_kernel" in compute_recs[0]["issue"]


def test_format_output_text():
    """Test text output formatting."""
    from rocpd.analyze import format_analysis_output

    time_breakdown = {
        "total_kernel_time": 1000000000,  # 1000 ms
        "total_memcpy_time": 200000000,  # 200 ms
        "total_runtime": 1200000000,  # 1200 ms
        "kernel_percent": 83.3,
        "memcpy_percent": 16.7,
        "overhead_percent": 0,
    }

    hotspots = [
        {
            "name": "kernel_1",
            "calls": 100,
            "total_duration": 500000000,
            "avg_duration": 5000000,
            "min_duration": 4000000,
            "max_duration": 6000000,
            "percent_of_total": 50,
        }
    ]

    memory_analysis = {
        "Host-to-Device": {
            "count": 10,
            "total_bytes": 1048576,  # 1 MB
            "total_duration": 100000000,  # 100 ms
            "avg_bytes": 104857,
            "avg_duration": 10000000,
            "bandwidth_bytes_per_sec": 10485760,  # ~10 MB/s
        }
    }

    recommendations = [
        {
            "priority": "INFO",
            "category": "Test",
            "issue": "Test issue",
            "suggestion": "Test suggestion",
            "actions": ["Action 1", "Action 2"],
            "next_steps": "Test command",
        }
    ]

    output = format_analysis_output(
        time_breakdown,
        hotspots,
        memory_analysis,
        recommendations,
        database_path="/test/db.db",
        output_format="text",
    )

    assert isinstance(output, str)
    assert len(output) > 0
    assert "ROCPD AI PERFORMANCE ANALYSIS" in output
    assert "TIME BREAKDOWN" in output
    assert "HOTSPOTS" in output
    assert "MEMORY COPY ANALYSIS" in output
    assert "RECOMMENDATIONS" in output
    assert "kernel_1" in output
    assert "Host-to-Device" in output


def test_format_output_with_no_data():
    """Test output formatting with empty data."""
    from rocpd.analyze import format_analysis_output

    time_breakdown = {
        "total_kernel_time": 0,
        "total_memcpy_time": 0,
        "total_runtime": 0,
        "kernel_percent": 0,
        "memcpy_percent": 0,
        "overhead_percent": 0,
    }

    hotspots = []
    memory_analysis = {}
    recommendations = []

    output = format_analysis_output(
        time_breakdown,
        hotspots,
        memory_analysis,
        recommendations,
        database_path="",
        output_format="text",
    )

    assert isinstance(output, str)
    assert "ROCPD AI PERFORMANCE ANALYSIS" in output


if __name__ == "__main__":
    # Use --noconftest to avoid loading conftest.py which requires rocprofiler_sdk module
    exit_code = pytest.main(["--noconftest", "-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
