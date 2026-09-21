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
    gen_counter_list,
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

    def test_gen_counter_list_with_none_returns_empty(self):
        """gen_counter_list returns (False, []) when given None."""
        visited, counters = gen_counter_list(None)
        assert not visited
        assert counters == []

    def test_gen_counter_list_with_non_string_returns_empty(self):
        """gen_counter_list returns (False, []) when given a non-string input."""
        visited, counters = gen_counter_list(123)
        assert not visited
        assert counters == []

    def test_gen_counter_list_extracts_counters_from_aggregation(self):
        """gen_counter_list extracts every counter referenced in an AVG expression."""
        visited, counters = gen_counter_list("AVG(SQ_WAVES + TCC_HIT)")
        assert visited
        assert "SQ_WAVES" in counters
        assert "TCC_HIT" in counters

    def test_gen_counter_list_handles_timestamp_expression(self):
        """gen_counter_list visits timestamp-only expressions successfully."""
        visited, _ = gen_counter_list("Start_Timestamp + End_Timestamp")
        assert visited

    def test_gen_counter_list_with_invalid_syntax_returns_unvisited(self):
        """gen_counter_list returns visited=False when the equation is unparseable."""
        visited, _ = gen_counter_list("INVALID SYNTAX !!!")
        assert not visited

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
