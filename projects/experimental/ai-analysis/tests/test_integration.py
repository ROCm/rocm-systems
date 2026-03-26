#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Integration tests for the rocm_ai_analysis module.

These tests cover bug-fix regressions and extended thinking support that are
unique to the integration test suite (not duplicated in test_api_standalone.py).
"""

import pytest
import sys
from pathlib import Path

import pytest


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
        from rocm_ai_analysis.analyze_engine import generate_recommendations

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
        compute_recs = [r for r in recs if r["category"] == "Compute Bottleneck"]
        assert compute_recs, "Expected a compute bottleneck recommendation"

        quoted_name = shlex.quote(dangerous_name)
        # The kernel name is scoped via rocprof-compute (rocprofv3 collects general
        # PMC counters without kernel filtering, so the name only appears in the
        # rocprof-compute command where shlex.quote is applied).
        kernel_cmds = [
            cmd
            for cmd in compute_recs[0]["commands"]
            if cmd.get("tool") == "rocprof-compute"
        ]
        assert kernel_cmds, "Expected at least one rocprof-compute command"
        for cmd in kernel_cmds:
            full = cmd["full_command"]
            # The properly shell-quoted form of the kernel name must appear
            assert quoted_name in full, (
                f"Expected shlex.quote({dangerous_name!r}) == {quoted_name!r} "
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
        from rocm_ai_analysis.analyze_engine import compute_time_breakdown

        # Simulate a result row where overhead would come out negative:
        # total_kernel=900, total_memcpy=200, total_runtime=1000 → overhead=-10%
        mock_result = (900, 200, 1000, 90.0, 20.0, -10.0)
        mock_conn = MagicMock()
        with patch("rocm_ai_analysis.analyze_engine.execute_statement") as mock_exec:
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
        from rocm_ai_analysis.analyze_engine import _format_tier0_webview
        from rocm_ai_analysis.api import SourceAnalysisResult

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
        from rocm_ai_analysis.api import _build_analysis_result

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

    def test_analyze_source_passes_analysis_context_to_llm(self, tmp_path):
        """I-3: analyze_source() must pass AnalysisContext(tier=0) to analyze_source_with_llm."""
        from unittest.mock import patch, MagicMock
        from rocm_ai_analysis.api import analyze_source
        from rocm_ai_analysis.llm_analyzer import AnalysisContext

        # Create a minimal hip file so SourceAnalyzer has something to scan
        (tmp_path / "test.hip").write_text("__global__ void myKernel() {}")

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_source_with_llm.return_value = "LLM result"

        with patch("rocm_ai_analysis.api.LLMAnalyzer", return_value=mock_analyzer):
            analyze_source(
                tmp_path, enable_llm=True, llm_provider="anthropic", llm_api_key="fake"
            )

        assert (
            mock_analyzer.analyze_source_with_llm.called
        ), "analyze_source_with_llm was not called"
        call_kwargs = mock_analyzer.analyze_source_with_llm.call_args
        # Accept both positional and keyword arg style
        kwargs = call_kwargs[1] if call_kwargs[1] else {}
        context = kwargs.get("context")
        if context is None and call_kwargs[0]:
            # Unlikely but check positional args too
            for arg in call_kwargs[0]:
                if isinstance(arg, AnalysisContext):
                    context = arg
                    break

        assert (
            context is not None
        ), "context= argument not passed to analyze_source_with_llm"
        assert isinstance(
            context, AnalysisContext
        ), f"Expected AnalysisContext, got {type(context)}"
        assert context.tier == 0, f"Expected tier=0, got {context.tier}"

    # ------------------------------------------------------------------
    # I-4: LLMAnalyzer construction without API key does not raise
    # ------------------------------------------------------------------

    def test_llm_analyzer_construction_without_api_key_does_not_raise(self):
        """I-4: LLMAnalyzer() must not raise LLMAuthenticationError at construction time."""
        import os
        from unittest.mock import patch
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer
        from rocm_ai_analysis.exceptions import LLMAuthenticationError

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                LLMAnalyzer(provider="anthropic")
            except LLMAuthenticationError:
                pytest.fail(
                    "LLMAnalyzer raised LLMAuthenticationError at construction time; "
                    "authentication should be deferred until the first API call"
                )

    # ------------------------------------------------------------------
    # I-5: self.model honored in LLMAnalyzer
    # ------------------------------------------------------------------

    def test_llm_analyzer_model_parameter_honored(self):
        """I-5: LLMAnalyzer(model='my-model') must use that model in the API call."""
        pytest.importorskip("anthropic")
        from unittest.mock import patch, MagicMock
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        custom_model = "claude-haiku-4-5-20251001"
        analyzer = LLMAnalyzer(
            provider="anthropic", api_key="sk-test", model=custom_model
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            analyzer._call_anthropic("sys", "user")

        assert mock_client.messages.create.called, "messages.create was not called"
        used_model = mock_client.messages.create.call_args[1].get("model")
        assert (
            used_model == custom_model
        ), f"Expected model {custom_model!r}, got {used_model!r}"

    # ------------------------------------------------------------------
    # P-2: Timeout added to LLM calls
    # ------------------------------------------------------------------

    def test_llm_calls_have_timeout(self):
        """P-2: All Anthropic LLM API calls must include a timeout parameter."""
        pytest.importorskip("anthropic")
        from unittest.mock import patch, MagicMock
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider="anthropic", api_key="sk-test")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="ok")]
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            analyzer._call_anthropic("sys", "user")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert (
            "timeout" in call_kwargs
        ), "timeout parameter missing from Anthropic API call"
        assert (
            call_kwargs["timeout"] == 120
        ), f"Expected timeout=120, got {call_kwargs['timeout']}"

    # ------------------------------------------------------------------
    # I-12: analyze_source_code raises on missing source_dir
    # ------------------------------------------------------------------

    def test_analyze_source_code_raises_on_missing_dir(self):
        """I-12: analyze_source_code() must raise SourceDirectoryNotFoundError for non-existent dir."""
        from rocm_ai_analysis.analyze_engine import analyze_source_code
        from rocm_ai_analysis.exceptions import SourceDirectoryNotFoundError

        with pytest.raises(SourceDirectoryNotFoundError):
            analyze_source_code(source_dir="/nonexistent/path/xyz_no_exist_123")

    # ------------------------------------------------------------------
    # I-9: ReferenceGuideNotFoundError with list not string
    # ------------------------------------------------------------------

    def test_reference_guide_not_found_error_with_list(self):
        """I-9: ReferenceGuideNotFoundError must accept List[str] and produce readable message."""
        from rocm_ai_analysis.exceptions import ReferenceGuideNotFoundError

        paths = [
            "share/rocprofiler-sdk/llm-reference-guide.md",
            "~/.config/rocpd/guide.md",
        ]
        err = ReferenceGuideNotFoundError(paths)
        msg = str(err)

        # Both paths should appear intact in the error message
        assert (
            "share/rocprofiler-sdk/llm-reference-guide.md" in msg
        ), f"First path missing from error message: {msg}"
        assert (
            "~/.config/rocpd/guide.md" in msg
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
        from rocm_ai_analysis.source_analyzer import SourceAnalyzer, _MAX_FILES

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


# ===========================================================================
# Tests: Extended thinking / --llm-thinking flag (Task 22)
# ===========================================================================


class TestLLMThinking:
    """Tests for extended thinking support via thinking_budget_tokens."""

    def test_llm_thinking_parameter_stored(self):
        """thinking_budget_tokens passed to __init__ must be stored on the instance."""
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(provider="anthropic", thinking_budget_tokens=8000)
        assert (
            analyzer.thinking_budget_tokens == 8000
        ), f"Expected thinking_budget_tokens=8000, got {analyzer.thinking_budget_tokens!r}"

    def test_llm_thinking_defaults_to_none(self):
        """When thinking_budget_tokens is not supplied, the attribute must be None."""
        import os
        from unittest.mock import patch
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        # Ensure env var is absent so it doesn't override the default
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROCPD_LLM_THINKING", None)
            analyzer = LLMAnalyzer(provider="anthropic")

        assert (
            analyzer.thinking_budget_tokens is None
        ), f"Expected thinking_budget_tokens=None, got {analyzer.thinking_budget_tokens!r}"

    def test_llm_thinking_openai_raises(self):
        """analyze_with_llm() must raise ValueError when provider=openai and thinking is set."""
        from unittest.mock import patch, MagicMock
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        analyzer = LLMAnalyzer(
            provider="openai",
            api_key="sk-test",
            thinking_budget_tokens=8000,
        )

        # analyze_with_llm() should raise before any API call is made
        with pytest.raises(
            ValueError,
            match="Extended thinking is only supported with the Anthropic provider",
        ):
            # Patch openai to avoid ImportError; the ValueError should fire before the actual call
            with patch.dict("sys.modules", {"openai": MagicMock()}):
                analyzer.analyze_with_llm(
                    {"has_counters": False, "has_pc_sampling": False},
                    custom_prompt=None,
                )

    def test_llm_thinking_env_var(self):
        """ROCPD_LLM_THINKING env var must set thinking_budget_tokens on construction."""
        import os
        from unittest.mock import patch
        from rocm_ai_analysis.llm_analyzer import LLMAnalyzer

        with patch.dict(os.environ, {"ROCPD_LLM_THINKING": "5000"}):
            analyzer = LLMAnalyzer(provider="anthropic")

        assert analyzer.thinking_budget_tokens == 5000, (
            f"Expected thinking_budget_tokens=5000 from env var, "
            f"got {analyzer.thinking_budget_tokens!r}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
