#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Tests that ATT analysis data is passed through to_json() and to_webview().
"""

import json

import pytest


def _make_result_with_att():
    """Build an AnalysisResult with att_analysis in _raw."""
    from rocinsight.ai_analysis.api import (
        AnalysisResult,
        AnalysisMetadata,
        ProfilingInfo,
        AnalysisSummary,
        ExecutionBreakdown,
        RecommendationSet,
    )

    result = AnalysisResult(
        metadata=AnalysisMetadata(
            rocpd_version="6.3.0",
            database_file="test.db",
            analysis_timestamp="2025-01-01T00:00:00",
        ),
        profiling_info=ProfilingInfo(
            total_duration_ns=1_000_000,
            profiling_mode="sys_trace_only",
            analysis_tier=3,
        ),
        summary=AnalysisSummary(
            overall_assessment="ATT test analysis",
            primary_bottleneck="compute",
            confidence=0.8,
            key_findings=["Kernel time: 80.0%"],
        ),
        execution_breakdown=ExecutionBreakdown(
            kernel_time_ns=800_000,
            kernel_time_pct=80.0,
            memcpy_time_ns=0,
            memcpy_time_pct=0.0,
        ),
        recommendations=RecommendationSet(),
    )
    result._raw = {
        "time_breakdown": {
            "total_kernel_time": 800_000,
            "total_memcpy_time": 0,
            "total_runtime": 1_000_000,
            "kernel_percent": 80.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 20.0,
        },
        "hotspots": [
            {
                "name": "test_kernel",
                "calls": 10,
                "total_duration": 800_000,
                "avg_duration": 80_000,
                "min_duration": 75_000,
                "max_duration": 90_000,
                "percent_of_total": 80.0,
            }
        ],
        "memory_analysis": {},
        "recommendations_raw": [],
        "hardware_counters": {"has_counters": False},
        "database_path": "test.db",
        "att_analysis": {
            "has_att_data": True,
            "kernels": [
                {
                    "name": "test_kernel",
                    "top_stall": "vmem_latency",
                    "stall_ratio": 0.65,
                    "hitcount": 12800,
                }
            ],
        },
    }
    return result


class TestAttSerialization:
    def test_to_json_includes_att_trace(self):
        """to_json() should include att_trace when _raw has att_analysis."""
        result = _make_result_with_att()
        json_str = result.to_json()
        doc = json.loads(json_str)
        assert "att_trace" in doc, "att_trace key missing from JSON output"
        assert doc["att_trace"]["has_att_data"] is True

    def test_to_json_att_trace_absent_when_no_att(self):
        """to_json() should not include att_trace when _raw has no att_analysis."""
        result = _make_result_with_att()
        del result._raw["att_analysis"]
        json_str = result.to_json()
        doc = json.loads(json_str)
        # att_trace should either be absent or None
        assert doc.get("att_trace") is None or "att_trace" not in doc

    def test_to_webview_runs_with_att_data(self):
        """to_webview() should not raise when _raw has att_analysis."""
        result = _make_result_with_att()
        html = result.to_webview()
        # The att_analysis kwarg is passed through; webview should render
        assert isinstance(html, str)
        assert len(html) > 0
