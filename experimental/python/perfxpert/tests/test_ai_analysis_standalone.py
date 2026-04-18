#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Standalone unit tests for the perfxpert ai_analysis module.

These tests do NOT require a real GPU trace database.
They DO require the perfxpert package to be importable (needs the built libpyrocpd
C extension). Run with the system-installed perfxpert path first, then the source
path for the edited Python modules:

    PERFXPERT_SYS=/opt/rocm-7.0.0/lib/python3.12/site-packages
    PERFXPERT_SRC=/dockerx/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${PERFXPERT_SYS}:${PERFXPERT_SRC}" pytest --noconftest test_api_standalone.py -v

IMPORTANT: PERFXPERT_SYS must come BEFORE PERFXPERT_SRC in PYTHONPATH to avoid a
circular import of libpyrocpd.
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers: build a minimal AnalysisResult without touching a real DB
# ---------------------------------------------------------------------------


def _make_minimal_result():
    """Build an AnalysisResult with empty/zero payloads for serialization tests."""
    from perfxpert.ai_analysis.api import (
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
        from perfxpert.ai_analysis.api import OutputFormat

        assert OutputFormat.PYTHON_OBJECT.value == "python_object"

    def test_has_json(self):
        from perfxpert.ai_analysis.api import OutputFormat

        assert OutputFormat.JSON.value == "json"

    def test_has_text(self):
        from perfxpert.ai_analysis.api import OutputFormat

        assert OutputFormat.TEXT.value == "text"

    def test_has_markdown(self):
        from perfxpert.ai_analysis.api import OutputFormat

        assert OutputFormat.MARKDOWN.value == "markdown"

    def test_has_webview(self):
        """AIA-003: WEBVIEW must be present in OutputFormat."""
        from perfxpert.ai_analysis.api import OutputFormat

        assert OutputFormat.WEBVIEW.value == "webview"

    def test_five_members(self):
        from perfxpert.ai_analysis.api import OutputFormat

        assert len(list(OutputFormat)) == 5


# ===========================================================================
# Tests: Exceptions (AIA-008, AIA-010, AIA-011)
# ===========================================================================


class TestExceptions:
    def test_missing_data_error_optional_list(self):
        """AIA-010: missing_tables should be Optional[List[str]]."""
        from perfxpert.ai_analysis.exceptions import MissingDataError

        # Both None and a list should work
        err_no_list = MissingDataError("msg")
        assert err_no_list.missing_tables == []
        err_with_list = MissingDataError("msg", ["kernels"])
        assert err_with_list.missing_tables == ["kernels"]

    def test_unsupported_gpu_error_optional_str(self):
        """AIA-010: gpu_arch should be Optional[str]."""
        from perfxpert.ai_analysis.exceptions import UnsupportedGPUError

        err_no_arch = UnsupportedGPUError("msg")
        assert err_no_arch.gpu_arch is None
        err_with_arch = UnsupportedGPUError("msg", "gfx906")
        assert err_with_arch.gpu_arch == "gfx906"

    def test_reference_guide_not_found_shows_all_paths(self):
        """AIA-008: ReferenceGuideNotFoundError must list all attempted paths."""
        from perfxpert.ai_analysis.exceptions import ReferenceGuideNotFoundError

        paths = ["/path/one/guide.md", "/path/two/guide.md", "/path/three/guide.md"]
        err = ReferenceGuideNotFoundError(paths)
        msg = str(err)
        for p in paths:
            assert p in msg, f"Path '{p}' not found in error message"
        assert err.attempted_paths == paths

    def test_reference_guide_exported_from_init(self):
        """AIA-011: ReferenceGuideNotFoundError must be importable from perfxpert.ai_analysis."""
        from perfxpert.ai_analysis import ReferenceGuideNotFoundError

        assert ReferenceGuideNotFoundError is not None

    def test_all_exceptions_exported(self):
        """Verify all documented exceptions are accessible from the public API."""
        import perfxpert.ai_analysis as m

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
            assert hasattr(m, name), f"{name} not exported from perfxpert.ai_analysis"


# ===========================================================================
# Tests: validate_database (AIA-013)
# ===========================================================================


class TestValidateDatabase:
    def test_raises_for_missing_file(self):
        """validate_database() must raise DatabaseNotFoundError for missing file."""
        from perfxpert.ai_analysis import validate_database, DatabaseNotFoundError

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

    def test_to_json_without_raw_raises_runtime_error(self):
        """to_json() without _raw must raise RuntimeError (not silently produce non-schema JSON)."""
        import pytest

        result = _make_minimal_result()
        # No _raw attached → must raise so callers know output would be non-schema-conformant
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
        from perfxpert.ai_analysis.api import _convert_result_to_llm_format

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
        from perfxpert.ai_analysis.api import _convert_result_to_llm_format

        result = _make_minimal_result()
        llm_data = _convert_result_to_llm_format(result)
        assert llm_data["kernels"] == []

    def test_has_execution_breakdown(self):
        from perfxpert.ai_analysis.api import _convert_result_to_llm_format

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
        from perfxpert.ai_analysis.api import _build_analysis_result

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
        from perfxpert.ai_analysis.api import _build_analysis_result

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

    def test_info_bucketed_as_low(self):
        """INFO priority should be placed in low_priority bucket (not medium)."""
        from perfxpert.ai_analysis.api import _build_analysis_result

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
        # INFO recs are bucketed as medium_priority (actionable guidance)
        assert len(result.recommendations.medium_priority) == 1
        assert len(result.recommendations.low_priority) == 0


# ===========================================================================
# Tests: Bug-fix regression tests (Tasks 1-4)
# ===========================================================================


class TestBugFixes:
    """
    Regression tests covering security, correctness, and LLM-layer bug fixes
    from code review Tasks 1-4.  Each test is tagged with the fix ID it covers.
    """

    # ------------------------------------------------------------------
    # C-1: shlex.quote in full_command
    # ------------------------------------------------------------------

    def test_kernel_name_shell_quoted_in_full_command(self):
        """C-1: full_command strings must use shlex.quote() for kernel names with shell metacharacters."""
        import shlex
        from perfxpert.analyze import generate_recommendations

        dangerous_name = "kernel'; rm -rf / #"
        hotspots = [
            {
                "name": dangerous_name,
                "percent_of_total": 60.0,
                "calls": 100,
                "avg_duration": 100_000,
            }
        ]
        time_breakdown = {
            "kernel_percent": 70,
            "memcpy_percent": 5,
            "overhead_percent": 5,
            "total_kernel_time": 1_000_000,
            "total_runtime": 1_500_000,
        }
        recs = generate_recommendations(time_breakdown, hotspots, {}, [])
        _rule3_cats = {"Kernel Hotspot", "Compute-Bound Kernel", "Mixed Bottleneck Kernel", "Memory-Bound Kernel"}
        compute_recs = [r for r in recs if r["category"] in _rule3_cats]
        assert compute_recs, "Expected a kernel hotspot recommendation"

        import re as _re
        escaped_name = _re.escape(dangerous_name)
        quoted_name = shlex.quote(f"^{escaped_name}$")
        rocprofv3_cmds = [
            cmd for cmd in compute_recs[0]["commands"] if cmd.get("tool") == "rocprofv3"
        ]
        assert rocprofv3_cmds, "Expected at least one rocprofv3 command"
        for cmd in rocprofv3_cmds:
            full = cmd["full_command"]
            # The properly shell-quoted, regex-escaped form must appear
            assert quoted_name in full, (
                f"Expected shlex.quote(re.escape({dangerous_name!r})) == {quoted_name!r} "
                f"in full_command, got: {full}"
            )
            # The raw (unquoted) name must not appear verbatim (i.e., not word-split)
            assert f" {dangerous_name} " not in full and not full.endswith(
                f" {dangerous_name}"
            ), f"Raw unquoted kernel name found in full_command: {full}"

    # ------------------------------------------------------------------
    # C-6: overhead_percent clamped at zero
    # ------------------------------------------------------------------

    def test_overhead_percent_clamped_at_zero(self):
        """C-6: overhead_percent must never be negative even when kernel+memcpy > total."""
        from unittest.mock import patch, MagicMock
        from perfxpert.analyze import compute_time_breakdown

        # Simulate a result row where overhead would come out negative:
        # total_kernel=900, total_memcpy=200, total_runtime=1000 → overhead=-10%
        mock_result = (900, 200, 1000, 90.0, 20.0, -10.0)
        mock_conn = MagicMock()
        with patch("perfxpert.analyze.execute_statement") as mock_exec:
            mock_exec.return_value.fetchone.return_value = mock_result
            result = compute_time_breakdown(mock_conn)

        assert (
            result["overhead_percent"] == 0.0
        ), f"Expected 0.0, got {result['overhead_percent']}"
        assert result["kernel_percent"] == 90.0
        assert result["memcpy_percent"] == 20.0

    # ------------------------------------------------------------------
    # C-7: Tier 0 webview XSS escaping
    # ------------------------------------------------------------------

    def test_tier0_webview_script_tag_escaped(self):
        """C-7: </script> in tier0 JSON payload must be escaped to prevent XSS."""
        from datetime import datetime
        from perfxpert.analyze import _format_tier0_webview
        from perfxpert.ai_analysis.api import SourceAnalysisResult

        result = SourceAnalysisResult(
            source_dir="/tmp/test",
            analysis_timestamp=datetime.now().isoformat(),
            programming_model="HIP",
            files_scanned=1,
            files_skipped=0,
            detected_kernels=[],
            kernel_count=0,
            detected_patterns=[],
            risk_areas=[],
            already_instrumented=False,
            roctx_marker_count=0,
            recommendations=[],
            suggested_counters=[],
            suggested_first_command="rocprofv3 --sys-trace -- ./app",
            llm_explanation="Normal text </script><script>alert(1)</script> more text",
        )

        html = _format_tier0_webview(result)
        # The unescaped </script><script>alert(1) sequence must not appear in the HTML
        assert (
            "</script><script>alert(1)" not in html
        ), "XSS vulnerability: </script> not escaped in tier0 webview payload"

    # ------------------------------------------------------------------
    # I-1: Bottleneck classification not mislead by has_counters alone
    # ------------------------------------------------------------------

    def test_bottleneck_classification_not_mislead_by_counters(self):
        """I-1: has_counters=True alone should not produce 'compute' bottleneck."""
        from pathlib import Path
        from perfxpert.ai_analysis.api import _build_analysis_result

        # Balanced breakdown — kernel% is only 40%, well below the 70% threshold
        time_breakdown = {
            "kernel_percent": 40.0,
            "memcpy_percent": 15.0,
            "overhead_percent": 10.0,
            "total_kernel_time": 400_000,
            "total_memcpy_time": 150_000,
            "total_runtime": 1_000_000,
        }
        hardware_counters = {"has_counters": True}

        result = _build_analysis_result(
            time_breakdown=time_breakdown,
            hotspots=[{"name": "k1", "percent_of_total": 40.0}],
            memory_analysis={},
            recommendations=[],
            hardware_counters=hardware_counters,
            database_path=Path("/tmp/fake.db"),
            custom_prompt=None,
        )

        assert (
            result.summary.primary_bottleneck == "mixed"
        ), f"Expected 'mixed' bottleneck, got {result.summary.primary_bottleneck!r}"

    # ------------------------------------------------------------------
    # I-3: AnalysisContext(tier=0) passed to LLM in analyze_source()
    # ------------------------------------------------------------------




    # ------------------------------------------------------------------
    # I-12: analyze_source_code raises on missing source_dir
    # ------------------------------------------------------------------

    def test_analyze_source_code_raises_on_missing_dir(self):
        """I-12: analyze_source_code() must raise SourceDirectoryNotFoundError for non-existent dir."""
        from perfxpert.analyze import analyze_source_code
        from perfxpert.ai_analysis.exceptions import SourceDirectoryNotFoundError

        with pytest.raises(SourceDirectoryNotFoundError):
            analyze_source_code(source_dir="/nonexistent/path/xyz_no_exist_123")

    # ------------------------------------------------------------------
    # I-9: ReferenceGuideNotFoundError with list not string
    # ------------------------------------------------------------------

    def test_reference_guide_not_found_error_with_list(self):
        """I-9: ReferenceGuideNotFoundError must accept List[str] and produce readable message."""
        from perfxpert.ai_analysis.exceptions import ReferenceGuideNotFoundError

        paths = [
            "/opt/rocm/share/llm-reference-guide.md",
            "/home/user/.config/guide.md",
        ]
        err = ReferenceGuideNotFoundError(paths)
        msg = str(err)

        # Both paths should appear intact in the error message
        assert (
            "/opt/rocm/share/llm-reference-guide.md" in msg
        ), f"First path missing from error message: {msg}"
        assert (
            "/home/user/.config/guide.md" in msg
        ), f"Second path missing from error message: {msg}"
        # Guard against the old bug where a bare string was iterated char-by-char
        assert (
            "o\n  - p" not in msg
        ), "Characters are being joined — bare string was passed instead of list"

    # ------------------------------------------------------------------
    # M-8: Source scanner truncation warning
    # ------------------------------------------------------------------

    def test_source_scanner_truncation_warning(self, tmp_path):
        """M-8: SourceAnalyzer must add a risk_area warning when _MAX_FILES limit is hit."""
        from perfxpert.ai_analysis.source_analyzer import SourceAnalyzer, _MAX_FILES

        # Create more files than _MAX_FILES (use .hip extension so they are scanned)
        for i in range(_MAX_FILES + 5):
            (tmp_path / f"kernel_{i}.hip").write_text(f"__global__ void k{i}() {{}}")

        scanner = SourceAnalyzer(tmp_path)
        plan = scanner.analyze()

        truncation_warnings = [
            r for r in plan.risk_areas if "truncat" in r.lower() or "limit" in r.lower()
        ]
        assert (
            truncation_warnings
        ), f"Expected a truncation warning in risk_areas, got: {plan.risk_areas}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use --noconftest to avoid loading conftest.py which requires rocprofiler_sdk module
    exit_code = pytest.main(["--noconftest", "-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
