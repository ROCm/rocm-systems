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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
