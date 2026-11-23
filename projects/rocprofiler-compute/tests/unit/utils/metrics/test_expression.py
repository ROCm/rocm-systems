# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/expression.py."""

import ast
from unittest.mock import patch

import pytest

from utils.metrics.expression import (
    CodeTransformer,
    InvalidExpressionError,
    build_eval_string,
    columns_read,
    normalize_equation,
    update_denominator_string,
    update_normal_unit_string,
)
from utils.utils_counter_defs import UNIT_COUNTER

# =============================================================================
# Tests for utils.metrics.expression
# =============================================================================


class TestExpression:
    """Tests for utils.metrics.expression."""

    def test_build_eval_string_returns_empty_for_empty_equation(self):
        """build_eval_string returns the empty string when given an empty equation."""
        assert build_eval_string("") == ""

    def test_update_denominator_string_returns_empty_for_empty_equation(self):
        """update_denominator_string returns the empty string when input is empty."""
        assert update_denominator_string("", "per_wave") == ""

    def test_visit_call_raises_for_unknown_function(self):
        """CodeTransformer rejects unsupported function calls."""
        tree = ast.parse("UNKNOWN_FUNC(5)")
        transformer = CodeTransformer()
        with pytest.raises(InvalidExpressionError, match="Unsupported call"):
            transformer.visit(tree)

    def test_visit_call_translates_supported_function_to_helper_name(self):
        """CodeTransformer.visit_Call rewrites MIN to the to_min helper name."""
        transformer = CodeTransformer()
        supported_call = ast.Call(
            func=ast.Name(id="MIN", ctx=ast.Load()),
            args=[ast.Constant(value=5)],
            keywords=[],
        )
        result = transformer.visit_Call(supported_call)
        assert result.func.id == "to_min", f"Expected 'to_min', got: {result.func.id}"

    def test_update_denominator_string_substitutes_denom_for_per_wave(self):
        """update_denominator_string replaces $denom."""
        result = update_denominator_string("SUM(SQ_WAVES) / SUM($denom)", "per_wave")
        assert "$denom" not in result
        assert "SQ_WAVES" in result

    def test_update_denominator_string_substitutes_denom_for_per_cycle(self):
        """update_denominator_string injects $GRBM_GUI_ACTIVE_PER_XCD for per_cycle."""
        result = update_denominator_string("SUM(DATA) / SUM($denom)", "per_cycle")
        assert "$GRBM_GUI_ACTIVE_PER_XCD" in result

    def test_update_denominator_string_substitutes_denom_for_per_second(self):
        """update_denominator_string substitutes the timestamp delta for per_second."""
        result = update_denominator_string("SUM(DATA) / SUM($denom)", "per_second")
        assert "End_Timestamp - Start_Timestamp" in result

    def test_update_denominator_string_substitutes_denom_for_per_kernel(self):
        """update_denominator_string substitutes UNIT_COUNTER for per_kernel."""
        result = update_denominator_string("SUM($denom)", "per_kernel")
        assert result == f"SUM({UNIT_COUNTER})"

    def test_update_denominator_string_keeps_denom_for_unsupported_unit(self):
        """update_denominator_string leaves $denom in place for unknown units."""
        result = update_denominator_string(
            "SUM(DATA) / SUM($denom)", "unsupported_unit"
        )
        assert "$denom" in result

    @pytest.mark.parametrize(
        "equation, normal_unit, expected",
        [
            ("(Prefix + $normUnit)", "per_wave", "Prefix per wave"),
            ("GB/s", "per_kernel", "GB/s"),
            ("Conflicts per Access", "per_kernel", "Conflicts per Access"),
        ],
    )
    def test_update_normal_unit_string(self, equation, normal_unit, expected):
        """Substitutes $normUnit and leaves case intact elsewhere.

        Regression for .capitalize() mangling "GB/s" into "Gb/s".
        """
        assert update_normal_unit_string(equation, normal_unit) == expected

    @pytest.mark.parametrize(
        "formula",
        [
            (
                "100 * SUM(SQ_ACTIVE_INST_SCA)"
                " / SUM(ammolite__GRBM_GUI_ACTIVE_PER_XCD"
                " * ammolite__cu_per_gpu)"
            ),
            (
                "TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31"
                " if (TCC_EA_RDREQ_31 != 0) else TCC_EA_RDREQ_31"
            ),
            "ROUND(AVG(4 * SQ_BUSY_CU_CYCLES), 0)",
        ],
    )
    def test_code_transformer_accepts_legitimate_formulas(
        self,
        formula: str,
    ) -> None:
        """CodeTransformer.generic_visit passes for real metric formulas."""
        tree = ast.parse(formula)
        transformer = CodeTransformer()
        transformer.visit(tree)

    @pytest.mark.parametrize(
        "formula, blocked_node",
        [
            ('"".__class__', "Attribute"),
            ("(lambda: 1)()", "Lambda"),
            ("[x for x in (1,)]", "ListComp"),
            ("(x for x in (1,))", "GeneratorExp"),
            ("{x: 1 for x in (1,)}", "DictComp"),
            ("{x for x in (1,)}", "SetComp"),
        ],
    )
    def test_code_transformer_rejects_unsafe_node_types(
        self,
        formula: str,
        blocked_node: str,
    ) -> None:
        """CodeTransformer.generic_visit raises for disallowed AST node types."""
        tree = ast.parse(formula)
        transformer = CodeTransformer()
        with pytest.raises(InvalidExpressionError, match=blocked_node):
            transformer.visit(tree)

    def test_build_eval_string_warns_on_unsafe_formula(self) -> None:
        """build_eval_string warns and returns empty for unsafe formulas."""
        with patch("utils.metrics.expression.console_warning") as mock_warning:
            result = build_eval_string('"".__class__')
        assert result == ""
        assert mock_warning.called


def test_columns_read_follows_built_in_variables():
    """Columns read through nested built-in variables count too."""
    build_in_vars = {
        "numActiveCUs": "SQ_BUSY_CU_CYCLES / $GRBM_GUI_ACTIVE_PER_XCD",
        "GRBM_GUI_ACTIVE_PER_XCD": "GRBM_GUI_ACTIVE / $num_xcd",
    }

    assert columns_read(
        "raw_pmc_df['TCC_HIT[0]'] / ammolite__numActiveCUs", build_in_vars
    ) == {"TCC_HIT[0]", "SQ_BUSY_CU_CYCLES", "GRBM_GUI_ACTIVE"}


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
            "a ** b",  # Python syntax outside the equation grammar
            "a if b else c",
            "a < b",
            "$x(1)",  # a call of a variable
            "a$b",  # a $ inside a name
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
        """A single number or name stays as written, also None, which a panel
        writes for a column without a value."""
        assert normalize_equation("42") == "42"
        assert normalize_equation("x") == "x"
        assert normalize_equation("None") == "None"
