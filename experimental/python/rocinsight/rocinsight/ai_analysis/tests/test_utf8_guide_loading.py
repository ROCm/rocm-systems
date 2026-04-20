"""Tests for UTF-8 reference guide loading (Windows compatibility)."""
import pytest
from unittest.mock import patch


class TestUTF8GuideLoading:
    def _make_utf8_guide(self, tmp_path):
        content = "# Guide\n\nHardware Counter Limits \u2014 MUST NOT EXCEED\n\u2192 split into\n\u2705 SAFE\n\u274c UNSAFE\n"
        p = tmp_path / "test-guide.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_load_reference_guide_public(self, tmp_path):
        guide_path = self._make_utf8_guide(tmp_path)
        with patch("rocinsight.ai_analysis.llm_analyzer.get_reference_guide_path", return_value=guide_path):
            from rocinsight.ai_analysis.llm_analyzer import load_reference_guide
            text = load_reference_guide()
        assert "\u2014" in text
        assert "\u2192" in text

    def test_load_reference_guide_instance(self, tmp_path):
        guide_path = self._make_utf8_guide(tmp_path)
        from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer
        analyzer = LLMAnalyzer.__new__(LLMAnalyzer)
        analyzer.reference_guide_path = guide_path
        text = analyzer._load_reference_guide()
        assert "\u2014" in text
        assert "\u2705" in text
