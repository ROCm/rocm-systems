#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Tests that _convert_result_to_llm_format uses the field names expected
by LLMAnalyzer._sanitize_data() and _format_data_for_llm().
"""

import pytest


def _make_result_with_raw():
    """Build an AnalysisResult with hotspot and memory_analysis data in _raw."""
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
            analysis_tier=1,
        ),
        summary=AnalysisSummary(
            overall_assessment="Test analysis",
            primary_bottleneck="unknown",
            confidence=0.5,
            key_findings=["Kernel time: 80.0%"],
        ),
        execution_breakdown=ExecutionBreakdown(
            kernel_time_ns=800_000,
            kernel_time_pct=80.0,
            memcpy_time_ns=100_000,
            memcpy_time_pct=10.0,
        ),
        recommendations=RecommendationSet(),
    )
    result._raw = {
        "time_breakdown": {
            "total_kernel_time": 800_000,
            "total_memcpy_time": 100_000,
            "total_runtime": 1_000_000,
            "kernel_percent": 80.0,
            "memcpy_percent": 10.0,
            "overhead_percent": 10.0,
        },
        "hotspots": [
            {
                "name": "test_kernel",
                "calls": 42,
                "total_duration": 800_000,
                "avg_duration": 19047,
                "min_duration": 10_000,
                "max_duration": 30_000,
                "percent_of_total": 80.0,
            }
        ],
        "memory_analysis": {
            "Host-to-Device": {"count": 5, "total_bytes": 1024, "avg_duration": 200},
            "Device-to-Host": {"count": 3, "total_bytes": 512, "avg_duration": 150},
            "Device-to-Device": {"count": 1, "total_bytes": 256, "avg_duration": 100},
        },
        "recommendations_raw": [],
        "hardware_counters": {"has_counters": False},
        "database_path": "test.db",
    }
    return result


class TestLlmPayloadFieldNames:
    def test_kernel_dispatch_count_field(self):
        """Kernel dicts should use 'dispatch_count' not 'calls'."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_result_with_raw()
        payload = _convert_result_to_llm_format(result)
        kernel = payload["kernels"][0]
        assert "dispatch_count" in kernel, "Expected 'dispatch_count' field"
        assert "calls" not in kernel, "Old 'calls' field should not be present"
        assert kernel["dispatch_count"] == 42

    def test_kernel_pct_total_time_field(self):
        """Kernel dicts should use 'pct_total_time' not 'percent_of_total'."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_result_with_raw()
        payload = _convert_result_to_llm_format(result)
        kernel = payload["kernels"][0]
        assert "pct_total_time" in kernel, "Expected 'pct_total_time' field"
        assert "percent_of_total" not in kernel, "Old 'percent_of_total' field should not be present"
        assert kernel["pct_total_time"] == 80.0

    def test_memory_direction_h2d(self):
        """Memory ops should use 'h2d' not 'Host-to-Device'."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_result_with_raw()
        payload = _convert_result_to_llm_format(result)
        mem = payload["memory_ops"]
        assert "h2d" in mem, "Expected 'h2d' key in memory_ops"
        assert "Host-to-Device" not in mem, "Raw 'Host-to-Device' key should be normalised"
        assert mem["h2d"]["count"] == 5

    def test_memory_direction_d2h(self):
        """Memory ops should use 'd2h' not 'Device-to-Host'."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_result_with_raw()
        payload = _convert_result_to_llm_format(result)
        mem = payload["memory_ops"]
        assert "d2h" in mem, "Expected 'd2h' key in memory_ops"
        assert "Device-to-Host" not in mem

    def test_memory_direction_d2d(self):
        """Memory ops should use 'd2d' not 'Device-to-Device'."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_result_with_raw()
        payload = _convert_result_to_llm_format(result)
        mem = payload["memory_ops"]
        assert "d2d" in mem, "Expected 'd2d' key in memory_ops"
        assert "Device-to-Device" not in mem

    def test_memory_dir_map_exists(self):
        """_MEMORY_DIR_MAP should be importable and contain expected entries."""
        from rocinsight.ai_analysis.api import _MEMORY_DIR_MAP

        assert _MEMORY_DIR_MAP["Host-to-Device"] == "h2d"
        assert _MEMORY_DIR_MAP["Device-to-Host"] == "d2h"
        assert _MEMORY_DIR_MAP["Device-to-Device"] == "d2d"
