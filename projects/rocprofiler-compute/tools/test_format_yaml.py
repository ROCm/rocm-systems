#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for YAML metric equation formatter."""

import textwrap

import pytest
from format_yaml import check_file, normalize_equation

# A panel with one equation that needs formatting and one equation with a
# dangling operator
PANEL_WITH_BROKEN_EQUATION = textwrap.dedent("""\
    Panel Config:
      id: 400
      data source:
      - metric_table:
          id: 401
          metric:
            L2 Hit:
              value: SUM(TCC_HIT_sum * 2)
            AI LDS:
              value: SUM(SQ_INSTS_VALU_MFMA_MOPS_F64 * 512 + ) / SUM(SQ_LDS_IDX_ACTIVE)
""")

# A per-channel table and a simple-box table. Each has a metric with a dangling
# operator, and a header with display text under the equation keys.
PANEL_WITH_STD_DEV_AND_EXPR = textwrap.dedent("""\
    Panel Config:
      id: 1800
      data source:
      - metric_table:
          id: 1801
          header:
            metric: Metric
            avg: Avg
            std dev: Std Dev
          metric:
            L2 Cache Hit Rate:
              avg: 100 * SUM(TCC_HIT_sum) / SUM(TCC_HIT_sum + TCC_MISS_sum)
              std dev: 100 * STD(TCC_HIT_sum / ) )
      - metric_table:
          id: 1802
          header:
            metric: Metric
            expr: Expression
          metric:
            Stalled on L2 Data:
              expr: 100 * TCP_PENDING_STALL_CYCLES_sum /
""")

# Panel collectables. Each top-level key names a formula, and one formula needs
# formatting.
COLLECTABLES = textwrap.dedent("""\
    # Named formulas that metric formulas use as $name.
    GRBM_GUI_ACTIVE_PER_XCD: GRBM_GUI_ACTIVE / $num_xcd
    hbm_write_bytes: ((TCC_EA0_WRREQ_sum-TCC_EA0_WRREQ_64B_sum)*32)
""")


def write_config(tmp_path, relative_path, text):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestCriticalCorrectness:
    """Rewrites that would change the value of an equation."""

    def test_unary_negation_preserves_semantics(self):
        """A minus sign keeps the parentheses of its operand: -(a + b) is not -a + b."""
        for equation in ["-(a + b)", "-(a * b)", "-(a - b)", "a - -b", "-SUM(x) / y"]:
            assert normalize_equation(equation) == equation


class TestConstantFactoring:
    """Constants move out of the aggregation functions."""

    @pytest.mark.parametrize("func", ["SUM", "AVG", "MIN", "MAX"])
    def test_constant_factored_out(self, func):
        """SUM(x * 128) becomes 128 * SUM(x)."""
        assert normalize_equation(f"{func}(x * 128)") == f"128 * {func}(x)"
        assert normalize_equation(f"{func}(64 * x)") == f"64 * {func}(x)"

    def test_real_world_cache_bw_equation(self):
        """An equation from the gfx115x configs."""
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
        """Each aggregation gives up its own constant."""
        result = normalize_equation("SUM(x * 2) / SUM(y * 3)")
        assert result == "2 * SUM(x) / (3 * SUM(y))"

    def test_every_constant_factored_out(self):
        """Both constants come out in one run, so a second run changes nothing."""
        assert normalize_equation("SUM(x * 4 * 2)") == "2 * (4 * SUM(x))"
        assert normalize_equation("2 * (4 * SUM(x))") == "2 * (4 * SUM(x))"

    def test_no_factoring_for_non_aggregation(self):
        """Only SUM, AVG, MIN and MAX give up their constants."""
        assert normalize_equation("SQRT(x * 2)") == "SQRT(x * 2)"

    def test_no_factoring_for_addition(self):
        """Only a product gives up its constant, not a sum."""
        assert normalize_equation("SUM(x + 128)") == "SUM(x + 128)"

    def test_no_factoring_that_changes_the_result(self):
        """A call with two arguments keeps its constant, and so does a negative
        constant, which would swap MIN and MAX."""
        assert normalize_equation("MAX(2 * x, 1)") == "MAX(2 * x, 1)"
        assert normalize_equation("MIN(-2 * x)") == "MIN(-2 * x)"


class TestParenthesesMinimization:
    """Only the parentheses that the order of evaluation needs are kept."""

    def test_unnecessary_parens_removed(self):
        """A left operand of the same precedence needs no parentheses."""
        assert normalize_equation("(a + b) + c") == "a + b + c"
        assert normalize_equation("(a * b) * c") == "a * b * c"

    def test_necessary_parens_preserved(self):
        """A right operand of the same precedence keeps its parentheses, so the
        order of evaluation stays the same."""
        assert normalize_equation("a - (b - c)") == "a - (b - c)"
        assert normalize_equation("a / (b / c)") == "a / (b / c)"
        assert normalize_equation("a + (b + c)") == "a + (b + c)"

    def test_precedence_parens_preserved(self):
        """A sum inside a product keeps its parentheses."""
        assert normalize_equation("(a + b) * c") == "(a + b) * c"
        assert normalize_equation("a * (b + c)") == "a * (b + c)"


class TestRealWorldEquations:
    """Equations in the forms the YAML files use."""

    def test_variable_substitution(self):
        """A $variable is a name like a counter, not a constant to factor out."""
        eq = (
            "SUM($wave_size * SQ_INSTS_VALU_sum) / SUM(End_Timestamp - Start_Timestamp)"
        )
        assert normalize_equation(eq) == eq

    def test_peak_value_equation(self):
        """A peak such as $L2Bw_empirical_peak * 1e9 stays as written."""
        eq = "$L2Bw_empirical_peak * 1e9"
        assert normalize_equation(eq) == eq

    def test_scientific_notation(self):
        """A number in scientific notation stays as written."""
        assert normalize_equation("x / 1e9") == "x / 1e9"

    def test_channel_templates_left_as_written(self):
        """A per-channel template such as TCC_ATOMIC[::_1] stays as written."""
        equation = "SUM(TO_INT(TCC_ATOMIC[::_1])) / SUM($denom)"
        assert normalize_equation(equation) == equation


class TestErrorHandling:
    """Equations that do not parse raise ValueError."""

    @pytest.mark.parametrize(
        "equation",
        [
            "a @ b",  # a character that no equation uses
            "a + b extra",  # a name after the end of the equation
            "a + (b",  # an unclosed parenthesis
            "a + b)",  # a closing parenthesis without its opening one
            "a +",  # an operator without its right operand
            "SUM()",  # a call without an argument
            "TCC_HIT[0]",  # a single channel, which no equation names
        ],
    )
    def test_unparseable_equation_raises(self, equation):
        with pytest.raises(ValueError):
            normalize_equation(equation)

    def test_whitespace_normalization(self):
        """Each operator has one space on each side."""
        assert normalize_equation("a+b") == "a + b"
        assert normalize_equation("a  +  b") == "a + b"

    def test_empty_and_simple_cases(self):
        """A single number or name stays as written."""
        assert normalize_equation("42") == "42"
        assert normalize_equation("x") == "x"


def test_unparseable_equation_reported(tmp_path):
    """A broken equation is reported by key; the other equations are still fixed."""
    panel = write_config(
        tmp_path,
        "analysis_configs/gfx942/0400_roofline.yaml",
        PANEL_WITH_BROKEN_EQUATION,
    )

    is_valid, issues = check_file(panel, fix=True)

    assert not is_valid
    assert issues[0].startswith(
        "  Panel Config.data source[0].metric_table.metric.AI LDS.value: "
        "cannot parse equation: "
    )
    content = panel.read_text(encoding="utf-8")
    assert "value: 2 * SUM(TCC_HIT_sum)" in content
    assert "MOPS_F64 * 512 + ) /" in content


def test_std_dev_and_expr_checked_but_not_headers(tmp_path):
    """std dev and simple-box expr values are equations; header text is not."""
    panel = write_config(
        tmp_path,
        "analysis_configs/gfx942/1800_l2_cache_per_channel.yaml",
        PANEL_WITH_STD_DEV_AND_EXPR,
    )
    _, issues = check_file(panel)
    assert [issue.split(": cannot parse")[0] for issue in issues] == [
        "  Panel Config.data source[0].metric_table.metric.L2 Cache Hit Rate.std dev",
        "  Panel Config.data source[1].metric_table.metric.Stalled on L2 Data.expr",
    ]


def test_unformatted_library_formula_reported(tmp_path):
    """A metric library formula is checked like a panel equation."""
    path = write_config(
        tmp_path,
        "metric_library/gfx908/l2.yaml",
        "l2.ai:\n  formula: ( SUM( (a * 2) ) / SUM(b) )\n",
    )
    is_valid, issues = check_file(path)
    assert not is_valid
    assert "l2.ai.formula: equation needs normalization" in issues[0]


def test_panel_collectables_fixed_in_place(tmp_path):
    """Each top-level value of a panel collectables file is an equation."""
    path = write_config(
        tmp_path, "analysis_configs/collectables/gfx942_collectables.yaml", COLLECTABLES
    )
    check_file(path, fix=True)
    assert path.read_text(encoding="utf-8") == COLLECTABLES.replace(
        "((TCC_EA0_WRREQ_sum-TCC_EA0_WRREQ_64B_sum)*32)",
        "(TCC_EA0_WRREQ_sum - TCC_EA0_WRREQ_64B_sum) * 32",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
