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
    ROCPD_SYS=/opt/rocm-7.0.0/lib/python3.12/site-packages
    PYTHONPATH="${ROCPD_SYS}" pytest --noconftest test_guide_filter_standalone.py -v
"""

import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Group A: AnalysisContext defaults and construction (5 tests)
# ---------------------------------------------------------------------------

class TestAnalysisContextDefaults:

    def test_default_tier_is_1(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        ctx = AnalysisContext()
        assert ctx.tier == 1

    def test_default_has_counters_false(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        ctx = AnalysisContext()
        assert ctx.has_counters is False

    def test_default_nullable_fields_are_none(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        ctx = AnalysisContext()
        assert ctx.bottleneck_type is None
        assert ctx.gpu_arch is None
        assert ctx.custom_prompt is None

    def test_explicit_values_preserved(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
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
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        a = AnalysisContext(tier=1, has_counters=False)
        b = AnalysisContext(tier=1, has_counters=False)
        assert a == b


# ---------------------------------------------------------------------------
# Group B: _select_tags logic (14 tests)
# ---------------------------------------------------------------------------

class TestSelectTags:

    def _tags(self, **kwargs):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext, _select_tags
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
        from rocpd.ai_analysis.llm_analyzer import _filter_guide
        return _filter_guide(guide, tags)

    def _make_guide(self, *sections):
        """Build a mini guide string from (title, tag_or_None, content) tuples."""
        parts = ["# LLM Reference Guide\n\nIntro block with no tag.\n"]
        for title, tag, content in sections:
            tag_line = f"<!-- rocpd-context: {tag} -->\n" if tag else ""
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
        guide = "# Guide\n\n## Multi\n<!-- rocpd-context: tier1, tier2 -->\nmulti content\n"
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
        guide = "# Guide\n\n## Section\n<!--  rocpd-context:  tier2  -->\nspaced content\n"
        result = self._filter(guide, {"tier2"})
        assert "spaced content" in result

    def test_unknown_tag_excludes_section(self):
        guide = self._make_guide(("Future", "future_tag", "future content"))
        result = self._filter(guide, {"always", "tier1", "tier2"})
        assert "future content" not in result

    def test_tag_comment_on_line2_still_found(self):
        guide = "# Guide\n\n## Section\n\n<!-- rocpd-context: tier1 -->\nline2 tag content\n"
        result = self._filter(guide, {"tier1"})
        assert "line2 tag content" in result

    def test_tag_comment_beyond_scan_window_treated_as_no_tag(self):
        # Tag comment on line 5 (beyond first-3-line scan) → treated as no tag → included
        guide = (
            "# Guide\n\n## Section\nline1\nline2\nline3\nline4\n"
            "<!-- rocpd-context: tier2 -->\nlate tag content\n"
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
        from rocpd.ai_analysis.llm_analyzer import LLMAnalyzer
        from unittest.mock import patch
        with patch.object(LLMAnalyzer, "_load_reference_guide", return_value=(
            "# Guide\n\n## Always Section\n<!-- rocpd-context: always -->\nalways content\n\n"
            "## Tier2 Section\n<!-- rocpd-context: tier2 -->\ntier2 content\n\n"
            "## Compiler Section\n<!-- rocpd-context: compiler -->\ncompiler content\n"
        )):
            return LLMAnalyzer(provider="anthropic", api_key="fake-key")

    def test_context_none_returns_full_guide(self):
        analyzer = self._make_analyzer()
        prompt = analyzer._build_system_prompt(context=None)
        assert "always content" in prompt
        assert "tier2 content" in prompt
        assert "compiler content" in prompt

    def test_tier1_context_excludes_tier2_and_compiler(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=1, has_counters=False)
        prompt = analyzer._build_system_prompt(context=ctx)
        assert "always content" in prompt
        assert "tier2 content" not in prompt
        assert "compiler content" not in prompt

    def test_tier2_context_includes_tier2_excludes_compiler(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=2, has_counters=True, bottleneck_type="latency")
        prompt = analyzer._build_system_prompt(context=ctx)
        assert "tier2 content" in prompt
        assert "compiler content" not in prompt

    def test_returned_prompt_is_always_non_empty(self):
        from rocpd.ai_analysis.llm_analyzer import AnalysisContext
        analyzer = self._make_analyzer()
        ctx = AnalysisContext(tier=1)
        prompt = analyzer._build_system_prompt(context=ctx)
        assert len(prompt) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
