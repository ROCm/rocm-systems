#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from format_yaml import normalize_equation  # noqa: E402


class TestCriticalCorrectness:
    """Tests for bugs that would corrupt equation semantics."""

    def test_peak_empirical_preserved(self):
        """Peak (Empirical) is a column header - space must be preserved."""
        assert normalize_equation("Peak (Empirical)") == "Peak (Empirical)"

    def test_unary_negation_preserves_semantics(self):
        """-(a + b) must stay -(a + b), not become -a + b which changes the result."""
        assert normalize_equation("-(a + b)") == "-(a + b)"
        assert normalize_equation("-(a * b)") == "-(a * b)"
        assert normalize_equation("-(a - b)") == "-(a - b)"

    def test_unknown_character_fallback(self):
        """Unknown characters cause fallback to original."""
        assert normalize_equation("a @ b") == "a @ b"
        assert normalize_equation("x & y") == "x & y"

    def test_incomplete_parse_fallback(self):
        """Trailing tokens should cause fallback, not partial parsing."""
        assert normalize_equation("a + b extra") == "a + b extra"


class TestConstantFactoring:
    """Main normalization: factor constants out of aggregations."""

    @pytest.mark.parametrize("func", ["SUM", "AVG", "MIN", "MAX"])
    def test_constant_factored_out(self, func):
        """SUM(x * 128) becomes 128 * SUM(x)."""
        assert normalize_equation(f"{func}(x * 128)") == f"128 * {func}(x)"
        assert normalize_equation(f"{func}(64 * x)") == f"64 * {func}(x)"

    def test_real_world_cache_bw_equation(self):
        """Actual equation from gfx115x configs."""
        input_eq = (
            "SUM(TCP_TOTAL_CACHE_ACCESSES_sum * 128) / "
            "SUM(End_Timestamp - Start_Timestamp)"
        )
        expected = (
            "128 * SUM(TCP_TOTAL_CACHE_ACCESSES_sum) / "
            "SUM(End_Timestamp - Start_Timestamp)"
        )
        assert normalize_equation(input_eq) == expected

    def test_nested_aggregations(self):
        """Each nested aggregation factors independently."""
        result = normalize_equation("SUM(x * 2) / SUM(y * 3)")
        assert result == "2 * SUM(x) / (3 * SUM(y))"

    def test_no_factoring_for_non_aggregation(self):
        """Only SUM/AVG/MIN/MAX factor, not other functions."""
        assert normalize_equation("SQRT(x * 2)") == "SQRT(x * 2)"

    def test_no_factoring_for_addition(self):
        """Only multiplication factors, not addition."""
        assert normalize_equation("SUM(x + 128)") == "SUM(x + 128)"


class TestParenthesesMinimization:
    """Remove unnecessary parens while preserving semantics."""

    def test_unnecessary_parens_removed(self):
        """Same precedence ops don't need parens."""
        assert normalize_equation("(a + b) + c") == "a + b + c"
        assert normalize_equation("(a * b) * c") == "a * b * c"

    def test_necessary_parens_preserved(self):
        """Right-associative ops need parens."""
        assert normalize_equation("a - (b - c)") == "a - (b - c)"
        assert normalize_equation("a / (b / c)") == "a / (b / c)"

    def test_precedence_parens_preserved(self):
        """Precedence parens must stay."""
        assert normalize_equation("(a + b) * c") == "(a + b) * c"
        assert normalize_equation("a * (b + c)") == "a * (b + c)"


class TestRealWorldEquations:
    """Actual equations from YAML configs."""

    def test_variable_substitution(self):
        """$variables like $wave_size work correctly."""
        eq = (
            "SUM($wave_size * SQ_INSTS_VALU_sum) / SUM(End_Timestamp - Start_Timestamp)"
        )
        result = normalize_equation(eq)
        assert "$wave_size" in result
        assert "SUM" in result

    def test_peak_value_equation(self):
        """Peak values like $L2Bw_empirical_peak * 1e9."""
        eq = "$L2Bw_empirical_peak * 1e9"
        assert normalize_equation(eq) == eq

    def test_scientific_notation(self):
        """Scientific notation works."""
        assert normalize_equation("x / 1e9") == "x / 1e9"

    def test_literal_values(self):
        """Literals are never parsed."""
        for literal in ["N/A", "None", "null", "true", "false"]:
            assert normalize_equation(literal) == literal


class TestErrorHandling:
    """Graceful error handling and fallback."""

    def test_unmatched_parentheses_fallback(self):
        """Unmatched parens return original."""
        assert normalize_equation("a + (b") == "a + (b"
        assert normalize_equation("a + b)") == "a + b)"

    def test_whitespace_normalization(self):
        """Whitespace gets normalized."""
        assert normalize_equation("a+b") == "a + b"
        assert normalize_equation("a  +  b") == "a + b"

    def test_empty_and_simple_cases(self):
        """Simple cases work."""
        assert normalize_equation("42") == "42"
        assert normalize_equation("x") == "x"
        assert normalize_equation("SUM()") == "SUM()"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
