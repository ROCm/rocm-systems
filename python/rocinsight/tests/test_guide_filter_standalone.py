#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
Standalone unit tests for LLM reference guide context-aware filtering.

These tests do NOT require a GPU trace database or real LLM credentials.
Run with:
    ROCINSIGHT_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    PYTHONPATH="${ROCINSIGHT_SYS}" pytest --noconftest test_guide_filter_standalone.py -v
"""

import sys

import pytest

# ---------------------------------------------------------------------------
# Group A: AnalysisContext defaults and construction (5 tests)
# ---------------------------------------------------------------------------


class TestAnalysisContextDefaults:

    def test_default_tier_is_1(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        ctx = AnalysisContext()
        assert ctx.tier == 1

    def test_default_has_counters_false(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        ctx = AnalysisContext()
        assert ctx.has_counters is False

    def test_default_nullable_fields_are_none(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        ctx = AnalysisContext()
        assert ctx.bottleneck_type is None
        assert ctx.gpu_arch is None
        assert ctx.custom_prompt is None

    def test_explicit_values_preserved(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        ctx = AnalysisContext(
            tier=2,
            has_counters=True,
            bottleneck_type="compute",
            gpu_arch="gfx942",
            custom_prompt="why is my kernel slow?",
        )
        assert ctx.tier == 2
        assert ctx.has_counters is True
        assert ctx.bottleneck_type == "compute"
        assert ctx.gpu_arch == "gfx942"
        assert ctx.custom_prompt == "why is my kernel slow?"

    def test_dataclass_equality(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        a = AnalysisContext(tier=1, has_counters=False)
        b = AnalysisContext(tier=1, has_counters=False)
        assert a == b


# ---------------------------------------------------------------------------
# Group B: _select_tags logic (14 tests)
# ---------------------------------------------------------------------------


class TestSelectTags:

    def _tags(self, **kwargs):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext, _select_tags

        return _select_tags(AnalysisContext(**kwargs))

    def test_tier1_no_counters_gives_always_and_tier1_only(self):
        tags = self._tags(tier=1, has_counters=False)
        assert tags == {"always", "tier1"}

    def test_tier2_value_adds_tier2_even_without_flag(self):
        tags = self._tags(tier=2, has_counters=False)
        assert "tier2" in tags
        assert "tier1" in tags

    def test_has_counters_true_adds_tier2_regardless_of_tier_field(self):
        tags = self._tags(tier=1, has_counters=True)
        assert "tier2" in tags

    def test_tier0_gives_always_source_compiler_not_tier1_or_tier2(self):
        tags = self._tags(tier=0)
        assert "always" in tags
        assert "source" in tags
        assert "compiler" in tags
        assert "tier1" not in tags
        assert "tier2" not in tags

    def test_bottleneck_compute_adds_compiler(self):
        tags = self._tags(tier=1, bottleneck_type="compute")
        assert "compiler" in tags

    def test_bottleneck_memory_adds_compiler(self):
        tags = self._tags(tier=1, bottleneck_type="memory")
        assert "compiler" in tags

    def test_bottleneck_latency_does_not_add_compiler(self):
        tags = self._tags(tier=2, has_counters=True, bottleneck_type="latency")
        assert "compiler" not in tags

    def test_bottleneck_mixed_does_not_add_compiler(self):
        tags = self._tags(tier=2, has_counters=True, bottleneck_type="mixed")
        assert "compiler" not in tags

    def test_custom_prompt_compiler_keyword_adds_compiler(self):
        tags = self._tags(tier=1, custom_prompt="check compiler flags")
        assert "compiler" in tags

    def test_custom_prompt_build_keyword_adds_compiler(self):
        tags = self._tags(tier=1, custom_prompt="build options to try")
        assert "compiler" in tags

    def test_custom_prompt_memory_keyword_does_not_add_compiler(self):
        tags = self._tags(tier=1, custom_prompt="memory bottleneck analysis")
        assert "compiler" not in tags

    def test_custom_prompt_none_does_not_add_compiler(self):
        tags = self._tags(tier=1, custom_prompt=None)
        assert "compiler" not in tags

    def test_full_tier2_compute_bottleneck_has_all_tags(self):
        tags = self._tags(tier=2, has_counters=True, bottleneck_type="compute")
        assert tags == {"always", "tier1", "tier2", "compiler"}

    def test_full_tier2_latency_bottleneck_has_no_compiler(self):
        tags = self._tags(tier=2, has_counters=True, bottleneck_type="latency")
        assert tags == {"always", "tier1", "tier2"}


# ---------------------------------------------------------------------------
# Group C: _filter_guide section parsing (12 tests)
# ---------------------------------------------------------------------------


class TestFilterGuide:

    def _filter(self, guide, tags):
        from rocinsight.ai_analysis.llm_analyzer import _filter_guide

        return _filter_guide(guide, tags)

    def _make_guide(self, *sections):
        """Build a mini guide string from (title, tag_or_None, content) tuples."""
        parts = ["# LLM Reference Guide\n\nIntro block with no tag.\n"]
        for title, tag, content in sections:
            tag_line = f"<!-- rocinsight-context: {tag} -->\n" if tag else ""
            parts.append(f"## {title}\n{tag_line}{content}\n")
        return "\n".join(parts)

    def test_always_tagged_section_included_when_always_in_tags(self):
        guide = self._make_guide(("Critical", "always", "critical content"))
        result = self._filter(guide, {"always"})
        assert "critical content" in result

    def test_tier2_section_excluded_when_only_tier1_in_tags(self):
        guide = self._make_guide(
            ("HW Counters", "tier2", "counter content"),
            ("Workflow", "tier1", "workflow content"),
        )
        result = self._filter(guide, {"always", "tier1"})
        assert "counter content" not in result
        assert "workflow content" in result

    def test_tier2_section_included_when_tier2_in_tags(self):
        guide = self._make_guide(("HW Counters", "tier2", "counter content"))
        result = self._filter(guide, {"always", "tier1", "tier2"})
        assert "counter content" in result

    def test_section_with_no_tag_always_included(self):
        guide = self._make_guide(("Untagged Section", None, "untagged content"))
        result = self._filter(guide, {"always"})
        assert "untagged content" in result

    def test_section_with_multiple_tags_included_on_any_match(self):
        guide = (
            "# Guide\n\n## Multi\n<!-- rocinsight-context: tier1, tier2 -->\nmulti content\n"
        )
        result = self._filter(guide, {"always", "tier2"})
        assert "multi content" in result

    def test_empty_guide_returns_empty_string(self):
        result = self._filter("", {"always"})
        assert result == ""

    def test_guide_with_zero_tagged_sections_returns_full_content(self):
        guide = self._make_guide(
            ("Alpha", None, "alpha content"),
            ("Beta", None, "beta content"),
        )
        result = self._filter(guide, {"always"})
        assert "alpha content" in result
        assert "beta content" in result

    def test_tag_comment_with_extra_whitespace_parsed_correctly(self):
        guide = (
            "# Guide\n\n## Section\n<!--  rocinsight-context:  tier2  -->\nspaced content\n"
        )
        result = self._filter(guide, {"tier2"})
        assert "spaced content" in result

    def test_unknown_tag_excludes_section(self):
        guide = self._make_guide(("Future", "future_tag", "future content"))
        result = self._filter(guide, {"always", "tier1", "tier2"})
        assert "future content" not in result

    def test_tag_comment_on_line2_still_found(self):
        guide = (
            "# Guide\n\n## Section\n\n<!-- rocinsight-context: tier1 -->\nline2 tag content\n"
        )
        result = self._filter(guide, {"tier1"})
        assert "line2 tag content" in result

    def test_tag_comment_beyond_scan_window_treated_as_no_tag(self):
        # Tag comment on line 5 (beyond first-3-line scan) → treated as no tag → included
        guide = (
            "# Guide\n\n## Section\nline1\nline2\nline3\nline4\n"
            "<!-- rocinsight-context: tier2 -->\nlate tag content\n"
        )
        result = self._filter(guide, {"always"})
        assert "late tag content" in result

    def test_multiple_sections_ordering_preserved(self):
        guide = self._make_guide(
            ("First", "always", "first content"),
            ("Second", "tier2", "second content"),
            ("Third", "always", "third content"),
        )
        result = self._filter(guide, {"always"})
        assert result.index("first content") < result.index("third content")
        assert "second content" not in result


# ---------------------------------------------------------------------------
# Group D: _build_system_prompt integration (4 tests)
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:

    def _make_analyzer(self):
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer
        from unittest.mock import patch

        with patch.object(
            LLMAnalyzer,
            "_load_reference_guide",
            return_value=(
                "# Guide\n\n## Always Section\n<!-- rocinsight-context: always -->\nalways content\n\n"
                "## Tier2 Section\n<!-- rocinsight-context: tier2 -->\ntier2 content\n\n"
                "## Compiler Section\n<!-- rocinsight-context: compiler -->\ncompiler content\n"
            ),
        ):
            return LLMAnalyzer(provider="anthropic", api_key="fake-key")

    def test_context_none_returns_full_guide(self):
        analyzer = self._make_analyzer()
        prompt = analyzer._build_system_prompt(context=None)
        assert "always content" in prompt
        assert "tier2 content" in prompt
        assert "compiler content" in prompt

    def test_tier1_context_excludes_tier2_and_compiler(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=1, has_counters=False)
        prompt = analyzer._build_system_prompt(context=ctx)
        assert "always content" in prompt
        assert "tier2 content" not in prompt
        assert "compiler content" not in prompt

    def test_tier2_context_includes_tier2_excludes_compiler(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=2, has_counters=True, bottleneck_type="latency")
        prompt = analyzer._build_system_prompt(context=ctx)
        assert "tier2 content" in prompt
        assert "compiler content" not in prompt

    def test_returned_prompt_is_always_non_empty(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=1)
        prompt = analyzer._build_system_prompt(context=ctx)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# Group D continued: context propagation through public methods (3 tests)
# ---------------------------------------------------------------------------


class TestAnalyzeWithLLMContextParam:

    def _make_analyzer_capturing_prompt(self):
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer
        from unittest.mock import patch

        captured = {}

        with patch.object(
            LLMAnalyzer,
            "_load_reference_guide",
            return_value=(
                "# Guide\n\n## Always\n<!-- rocinsight-context: always -->\nalways text\n\n"
                "## Tier2\n<!-- rocinsight-context: tier2 -->\ntier2 text\n"
            ),
        ):
            analyzer = LLMAnalyzer(provider="anthropic", api_key="fake")

        def fake_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return "fake llm response"

        analyzer._call_anthropic = fake_call
        return analyzer, captured

    def test_analyze_with_llm_context_filters_guide(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext

        analyzer, captured = self._make_analyzer_capturing_prompt()
        ctx = AnalysisContext(tier=1, has_counters=False)
        analyzer.analyze_with_llm(analysis_data={}, context=ctx)
        assert "tier2 text" not in captured["system_prompt"]
        assert "always text" in captured["system_prompt"]

    def test_analyze_with_llm_no_context_uses_full_guide(self):
        analyzer, captured = self._make_analyzer_capturing_prompt()
        analyzer.analyze_with_llm(analysis_data={})
        assert "tier2 text" in captured["system_prompt"]

    def test_analyze_source_with_llm_context_filters_guide(self):
        from rocinsight.ai_analysis.llm_analyzer import AnalysisContext, LLMAnalyzer
        from unittest.mock import patch
        from rocinsight.ai_analysis.api import SourceAnalysisResult

        captured = {}

        with patch.object(
            LLMAnalyzer,
            "_load_reference_guide",
            return_value=(
                "# Guide\n\n## Always\n<!-- rocinsight-context: always -->\nalways text\n\n"
                "## Compiler\n<!-- rocinsight-context: compiler -->\ncompiler text\n"
            ),
        ):
            analyzer = LLMAnalyzer(provider="anthropic", api_key="fake")

        def fake_call(system_prompt, user_prompt, **kwargs):
            captured["system_prompt"] = system_prompt
            return "fake source response"

        analyzer._call_anthropic = fake_call

        ctx = AnalysisContext(tier=0)  # Tier 0 → compiler tag active
        minimal_result = SourceAnalysisResult(
            source_dir="/tmp",
            analysis_timestamp="2026-01-01T00:00:00",
            programming_model="HIP",
            files_scanned=0,
            files_skipped=0,
            detected_kernels=[],
            kernel_count=0,
            detected_patterns=[],
            risk_areas=[],
            already_instrumented=False,
            roctx_marker_count=0,
            recommendations=[],
            suggested_counters=[],
            suggested_first_command="",
        )
        analyzer.analyze_source_with_llm(minimal_result, context=ctx)
        assert "compiler text" in captured["system_prompt"]


# ---------------------------------------------------------------------------
# Group F: public API export (2 tests)
# ---------------------------------------------------------------------------


class TestPublicExport:

    def test_analysis_context_importable_from_package(self):
        from rocinsight.ai_analysis import AnalysisContext

        ctx = AnalysisContext(tier=2)
        assert ctx.tier == 2

    def test_analysis_context_in_all(self):
        import rocinsight.ai_analysis as pkg

        assert "AnalysisContext" in pkg.__all__


# ---------------------------------------------------------------------------
# Group E: end-to-end with real guide file (6 tests)
# ---------------------------------------------------------------------------


class TestEndToEndWithRealGuide:
    """
    Load the actual llm-reference-guide.md and verify filtering behaviour.
    These tests do NOT call any external LLM API.
    """

    def _build_prompt(self, **ctx_kwargs):
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer, AnalysisContext
        from unittest.mock import patch

        guide = (
            LLMAnalyzer.__module__
            and __import__(
                "rocinsight.ai_analysis.llm_analyzer", fromlist=["get_reference_guide_path"]
            )
            .get_reference_guide_path()
            .read_text()
        )
        with patch.object(LLMAnalyzer, "_load_reference_guide", return_value=guide):
            analyzer = LLMAnalyzer(provider="anthropic", api_key="fake")
        ctx = AnalysisContext(**ctx_kwargs)
        return analyzer._build_system_prompt(context=ctx)

    def test_tier1_excludes_compiler_section(self):
        prompt = self._build_prompt(tier=1, has_counters=False)
        assert "Compiler Optimization Flags" not in prompt

    def test_tier2_latency_excludes_compiler_section(self):
        prompt = self._build_prompt(tier=2, has_counters=True, bottleneck_type="latency")
        assert "Compiler Optimization Flags" not in prompt

    def test_tier0_includes_compiler_section(self):
        prompt = self._build_prompt(tier=0)
        assert "Compiler Optimization Flags" in prompt

    def test_bottleneck_compute_includes_compiler_section(self):
        prompt = self._build_prompt(tier=2, has_counters=True, bottleneck_type="compute")
        assert "Compiler Optimization Flags" in prompt

    def test_critical_requirements_always_present(self):
        for tier in (0, 1, 2):
            prompt = self._build_prompt(tier=tier)
            assert "CRITICAL REQUIREMENTS" in prompt, f"Missing in tier {tier}"

    def test_always_tagged_sections_present_in_every_tier(self):
        always_markers = [
            "Your Role",
            "Output Format Requirements",
            "What NOT to Do",
            "Summary",
        ]
        for tier in (0, 1, 2):
            prompt = self._build_prompt(tier=tier)
            for marker in always_markers:
                assert marker in prompt, f"'{marker}' missing for tier {tier}"


# ---------------------------------------------------------------------------
# Group F: guide file integrity (2 tests)
# ---------------------------------------------------------------------------


class TestGuideIntegrity:
    """Validate that the real llm-reference-guide.md is correctly tagged."""

    KNOWN_TAGS = {"always", "tier1", "tier2", "tier3", "compiler", "source", "tracelens_metrics"}
    # The intro block (before the first ## section) is intentionally untagged
    UNTAGGED_ALLOWED_PREFIXES = ("LLM Reference Guide",)

    @classmethod
    def _sections(cls):
        """Return list of (title, tag_or_None) for every ## section."""
        import re
        from rocinsight.ai_analysis.llm_analyzer import get_reference_guide_path

        text = get_reference_guide_path().read_text()
        tag_re = re.compile(r"<!--\s*rocinsight-context:\s*([^-]+?)\s*-->")
        results = []
        for raw in re.split(r"\n(?=## )", text):
            if not raw.startswith("## "):
                continue
            title = raw.splitlines()[0][3:].strip()
            head = "\n".join(raw.splitlines()[:3])
            match = tag_re.search(head)
            tag = match.group(1).strip() if match else None
            results.append((title, tag))
        return results

    def test_every_section_has_a_tag(self):
        """No ## section should be accidentally left without a rocinsight-context tag."""
        untagged = [
            title
            for title, tag in self._sections()
            if tag is None
            and not any(title.startswith(p) for p in self.UNTAGGED_ALLOWED_PREFIXES)
        ]
        assert untagged == [], f"Sections missing rocinsight-context tag: {untagged}"

    def test_all_tags_are_from_known_vocabulary(self):
        """Catch typos in tag names e.g. 'tier_2' instead of 'tier2'."""
        bad = []
        for title, tag in self._sections():
            if tag is None:
                continue
            for t in (t.strip() for t in tag.split(",")):
                if t not in self.KNOWN_TAGS:
                    bad.append((title, t))
        assert bad == [], f"Unknown tags found: {bad}"


# ---------------------------------------------------------------------------
# Group G: Path sanitization / redaction (6 tests)
# ---------------------------------------------------------------------------


class TestPathSanitization:
    """Tests for _redact_paths() in llm_analyzer.py."""

    def _redact(self, value: str) -> str:
        from rocinsight.ai_analysis.llm_analyzer import _redact_paths

        return _redact_paths(value)

    def test_absolute_home_path_redacted(self):
        """Absolute /home/user/... path should be replaced with [REDACTED]."""
        result = self._redact("kernel at /home/user/secret.py line 42")
        assert "/home/user/secret.py" not in result
        assert "[REDACTED]" in result

    def test_absolute_opt_path_redacted(self):
        """Absolute /opt/rocm/... path should be replaced with [REDACTED]."""
        result = self._redact("loaded from /opt/rocm/lib/foo.so")
        assert "/opt/rocm/lib/foo.so" not in result
        assert "[REDACTED]" in result

    def test_relative_path_traversal_redacted(self):
        """Relative path traversal ../../etc/passwd should be redacted."""
        result = self._redact("reading ../../etc/passwd")
        assert "../../etc/passwd" not in result
        assert "[REDACTED_PATH]" in result

    def test_normal_text_unchanged(self):
        """Text without paths should remain unchanged."""
        text = "GPU utilization is 85.3% with 480 waves"
        result = self._redact(text)
        assert result == text

    def test_mixed_paths_both_redacted(self):
        """Both absolute and relative paths in the same string should be redacted."""
        text = "kernel at /opt/rocm/lib/foo.so and ../../bar/baz.cpp"
        result = self._redact(text)
        assert "/opt/rocm/lib/foo.so" not in result
        assert "../../bar/baz.cpp" not in result
        assert "[REDACTED]" in result
        assert "[REDACTED_PATH]" in result

    def test_tmp_path_redacted(self):
        """/tmp/ paths should also be redacted."""
        result = self._redact("output saved to /tmp/rocpd_trace/run_001/results.db")
        assert "/tmp/rocpd_trace" not in result
        assert "[REDACTED]" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
