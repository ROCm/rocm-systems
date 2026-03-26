#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Standalone unit tests for the rocpd ai_analysis module.

These tests do NOT require a real GPU trace database.
They DO require the rocpd package to be importable (needs the built libpyrocpd
C extension). Run with the system-installed rocpd path first, then the source
path for the edited Python modules:

    ROCINSIGHT_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    ROCINSIGHT_SRC=<repo>/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${ROCINSIGHT_SYS}:${ROCINSIGHT_SRC}" pytest --noconftest test_api_standalone.py -v

IMPORTANT: ROCINSIGHT_SYS must come BEFORE ROCINSIGHT_SRC in PYTHONPATH to avoid a
circular import of libpyrocpd.
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal AnalysisResult without touching a real DB
# ---------------------------------------------------------------------------


def _make_minimal_result():
    """Build an AnalysisResult with empty/zero payloads for serialization tests."""
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
            memcpy_time_ns=0,
            memcpy_time_pct=0.0,
        ),
        recommendations=RecommendationSet(),
    )
    return result


def _attach_raw(
    result,
    *,
    time_breakdown=None,
    hotspots=None,
    memory_analysis=None,
    recommendations_raw=None,
    hardware_counters=None,
    database_path="test.db",
):
    """Attach a _raw dict to an AnalysisResult for to_json()/to_webview() tests."""
    result._raw = {
        "time_breakdown": time_breakdown
        or {
            "total_kernel_time": 800_000,
            "total_memcpy_time": 0,
            "total_runtime": 1_000_000,
            "kernel_percent": 80.0,
            "memcpy_percent": 0.0,
            "overhead_percent": 20.0,
        },
        "hotspots": hotspots
        or [
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
        "memory_analysis": memory_analysis or {},
        "recommendations_raw": recommendations_raw or [],
        "hardware_counters": hardware_counters or {"has_counters": False},
        "database_path": database_path,
    }
    return result


# ===========================================================================
# Tests: OutputFormat enum (AIA-003)
# ===========================================================================


class TestOutputFormat:
    def test_has_python_object(self):
        from rocinsight.ai_analysis.api import OutputFormat

        assert OutputFormat.PYTHON_OBJECT.value == "python_object"

    def test_has_json(self):
        from rocinsight.ai_analysis.api import OutputFormat

        assert OutputFormat.JSON.value == "json"

    def test_has_text(self):
        from rocinsight.ai_analysis.api import OutputFormat

        assert OutputFormat.TEXT.value == "text"

    def test_has_markdown(self):
        from rocinsight.ai_analysis.api import OutputFormat

        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_has_webview(self):
        """AIA-003: WEBVIEW must be present in OutputFormat."""
        from rocinsight.ai_analysis.api import OutputFormat

        assert OutputFormat.WEBVIEW.value == "webview"

    def test_five_members(self):
        from rocinsight.ai_analysis.api import OutputFormat

        assert len(list(OutputFormat)) == 5


# ===========================================================================
# Tests: Exceptions (AIA-008, AIA-010, AIA-011)
# ===========================================================================


class TestExceptions:
    def test_missing_data_error_optional_list(self):
        """AIA-010: missing_tables should be Optional[List[str]]."""
        from rocinsight.ai_analysis.exceptions import MissingDataError

        # Both None and a list should work
        err_no_list = MissingDataError("msg")
        assert err_no_list.missing_tables == []
        err_with_list = MissingDataError("msg", ["kernels"])
        assert err_with_list.missing_tables == ["kernels"]

    def test_unsupported_gpu_error_optional_str(self):
        """AIA-010: gpu_arch should be Optional[str]."""
        from rocinsight.ai_analysis.exceptions import UnsupportedGPUError

        err_no_arch = UnsupportedGPUError("msg")
        assert err_no_arch.gpu_arch is None
        err_with_arch = UnsupportedGPUError("msg", "gfx906")
        assert err_with_arch.gpu_arch == "gfx906"

    def test_reference_guide_not_found_shows_all_paths(self):
        """AIA-008: ReferenceGuideNotFoundError must list all attempted paths."""
        from rocinsight.ai_analysis.exceptions import ReferenceGuideNotFoundError

        paths = ["/path/one/guide.md", "/path/two/guide.md", "/path/three/guide.md"]
        err = ReferenceGuideNotFoundError(paths)
        msg = str(err)
        for p in paths:
            assert p in msg, f"Path '{p}' not found in error message"
        assert err.attempted_paths == paths

    def test_reference_guide_exported_from_init(self):
        """AIA-011: ReferenceGuideNotFoundError must be importable from rocinsight.ai_analysis."""
        from rocinsight.ai_analysis import ReferenceGuideNotFoundError

        assert ReferenceGuideNotFoundError is not None

    def test_all_exceptions_exported(self):
        """Verify all documented exceptions are accessible from the public API."""
        import rocinsight.ai_analysis as m

        for name in [
            "AnalysisError",
            "DatabaseNotFoundError",
            "DatabaseCorruptedError",
            "MissingDataError",
            "UnsupportedGPUError",
            "LLMAuthenticationError",
            "LLMRateLimitError",
            "ReferenceGuideNotFoundError",
        ]:
            assert hasattr(m, name), f"{name} not exported from rocinsight.ai_analysis"


# ===========================================================================
# Tests: validate_database (AIA-013)
# ===========================================================================


class TestValidateDatabase:
    def test_raises_for_missing_file(self):
        """validate_database() must raise DatabaseNotFoundError for missing file."""
        from rocinsight.ai_analysis import validate_database, DatabaseNotFoundError

        with pytest.raises(DatabaseNotFoundError):
            validate_database(Path("/nonexistent/path/to/trace.db"))


# ===========================================================================
# Tests: AnalysisResult serialization (AIA-004)
# ===========================================================================


class TestAnalysisResultSerialization:
    def test_to_dict_returns_dict(self):
        result = _make_minimal_result()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "metadata" in d
        assert "recommendations" in d

    def test_to_json_raises_without_raw(self):
        """to_json() raises RuntimeError when _raw is not populated."""
        result = _make_minimal_result()
        # _raw is not attached — raises RuntimeError (correct behavior: caller must use analyze_database())
        with pytest.raises(RuntimeError, match="Raw analysis data not available"):
            result.to_json()

    def test_to_json_with_raw_returns_schema_conformant_json(self):
        """AIA-004: to_json() with _raw must include schema_version."""
        result = _attach_raw(_make_minimal_result())
        j = result.to_json()
        parsed = json.loads(j)
        # schema-conformant output includes schema_version
        assert "schema_version" in parsed, "JSON output missing schema_version field"
        assert parsed["schema_version"] == "0.1.0"

    def test_to_webview_raises_without_raw(self):
        """to_webview() must raise RuntimeError if _raw is not attached."""
        result = _make_minimal_result()
        with pytest.raises(RuntimeError, match="analyze_database"):
            result.to_webview()

    def test_to_webview_with_raw_returns_html(self):
        """AIA-004: to_webview() with _raw must return HTML string."""
        result = _attach_raw(_make_minimal_result())
        html = result.to_webview()
        assert isinstance(html, str)
        assert "<!DOCTYPE" in html or "<html" in html
        assert len(html) > 1000  # must be a real HTML document


# ===========================================================================
# Tests: _convert_result_to_llm_format (AIA-006)
# ===========================================================================


class TestConvertResultToLlmFormat:
    def test_returns_real_kernel_data(self):
        """AIA-006: kernels list must not be empty when hotspots are present."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _attach_raw(
            _make_minimal_result(),
            hotspots=[
                {
                    "name": "conv2d",
                    "calls": 5,
                    "total_duration": 500_000,
                    "avg_duration": 100_000,
                    "percent_of_total": 50.0,
                }
            ],
        )
        llm_data = _convert_result_to_llm_format(result)
        assert len(llm_data["kernels"]) == 1
        assert llm_data["kernels"][0]["name"] == "conv2d"

    def test_returns_empty_kernels_without_raw(self):
        """Without _raw, kernels defaults to empty list (graceful degradation)."""
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_minimal_result()
        llm_data = _convert_result_to_llm_format(result)
        assert llm_data["kernels"] == []

    def test_has_execution_breakdown(self):
        from rocinsight.ai_analysis.api import _convert_result_to_llm_format

        result = _make_minimal_result()
        llm_data = _convert_result_to_llm_format(result)
        assert "execution_breakdown" in llm_data
        assert "kernel_time_pct" in llm_data["execution_breakdown"]


# ===========================================================================
# Tests: _build_analysis_result key mapping (AIA-002)
# ===========================================================================


class TestBuildAnalysisResultKeyMapping:
    """Verify that recommendation keys from generate_recommendations() are mapped correctly."""

    def _make_raw_rec(self, priority="HIGH"):
        return {
            "priority": priority,
            "category": "Low Occupancy",
            "issue": "Average wave occupancy is very low",
            "suggestion": "Increase occupancy by reducing VGPR usage",
            "estimated_impact": "15-20% performance improvement",
            "actions": ["Compile with -O3", "Reduce local arrays"],
            "commands": [],
        }

    def test_high_priority_bucketing(self):
        from rocinsight.ai_analysis.api import _build_analysis_result

        result = _build_analysis_result(
            time_breakdown={
                "total_kernel_time": 0,
                "total_memcpy_time": 0,
                "total_runtime": 0,
                "kernel_percent": 0.0,
                "memcpy_percent": 0.0,
                "overhead_percent": 0.0,
            },
            hotspots=[],
            memory_analysis={},
            recommendations=[self._make_raw_rec("HIGH")],
            hardware_counters={"has_counters": False},
            database_path=Path("test.db"),
            custom_prompt=None,
        )
        assert len(result.recommendations.high_priority) == 1
        rec = result.recommendations.high_priority[0]
        assert rec.title == "Average wave occupancy is very low"
        assert rec.description == "Increase occupancy by reducing VGPR usage"
        assert rec.estimated_impact == "15-20% performance improvement"
        assert rec.next_steps == ["Compile with -O3", "Reduce local arrays"]
        assert rec.priority == "high"  # normalized to lowercase

    def test_medium_priority_bucketing(self):
        from rocinsight.ai_analysis.api import _build_analysis_result

        result = _build_analysis_result(
            time_breakdown={
                "total_kernel_time": 0,
                "total_memcpy_time": 0,
                "total_runtime": 0,
                "kernel_percent": 0.0,
                "memcpy_percent": 0.0,
                "overhead_percent": 0.0,
            },
            hotspots=[],
            memory_analysis={},
            recommendations=[self._make_raw_rec("MEDIUM")],
            hardware_counters={"has_counters": False},
            database_path=Path("test.db"),
            custom_prompt=None,
        )
        assert len(result.recommendations.medium_priority) == 1

    def test_info_bucketed_as_medium(self):
        """INFO priority should be placed in medium_priority bucket."""
        from rocinsight.ai_analysis.api import _build_analysis_result

        result = _build_analysis_result(
            time_breakdown={
                "total_kernel_time": 0,
                "total_memcpy_time": 0,
                "total_runtime": 0,
                "kernel_percent": 0.0,
                "memcpy_percent": 0.0,
                "overhead_percent": 0.0,
            },
            hotspots=[],
            memory_analysis={},
            recommendations=[self._make_raw_rec("INFO")],
            hardware_counters={"has_counters": False},
            database_path=Path("test.db"),
            custom_prompt=None,
        )
        assert len(result.recommendations.medium_priority) == 1
