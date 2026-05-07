# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.expression."""

import ast

import pytest

from utils.metrics.expression import (
    CodeTransformer,
    build_eval_string,
    gen_counter_list,
    update_denominator_string,
    update_normal_unit_string,
)
from utils.utils_common import calc_builtin_var


class TestExpression:
    """Tests for utils.metrics.expression."""

    def test_build_eval_string_returns_empty_for_empty_equation(self):
        """build_eval_string returns the empty string when given an empty equation."""
        assert build_eval_string("") == ""

    def test_update_denominator_string_returns_empty_for_empty_equation(self):
        """update_denominator_string returns the empty string when input is empty."""
        assert update_denominator_string("", "per_wave") == ""

    def test_calc_builtin_var_exits_for_unsupported_var(self):
        """calc_builtin_var exits when given an unsupported variable name."""
        sys_info = {"total_l2_chan": 32}
        with pytest.raises(SystemExit):
            calc_builtin_var("$unsupported_var", sys_info)

    def test_visit_call_raises_for_unknown_function(self):
        """CodeTransformer.visit_Call raises for unknown function names."""
        transformer = CodeTransformer()
        unknown_call = ast.Call(
            func=ast.Name(id="ammolite__UNKNOWN", ctx=ast.Load()),
            args=[ast.Constant(value=5)],
            keywords=[],
        )
        with pytest.raises(Exception, match="Unknown call"):
            transformer.visit_Call(unknown_call)

    def test_visit_call_translates_supported_function_to_helper_name(self):
        """CodeTransformer.visit_Call rewrites MIN to the to_min helper name."""
        transformer = CodeTransformer()
        supported_call = ast.Call(
            func=ast.Name(id="MIN", ctx=ast.Load()),
            args=[ast.Constant(value=5) if hasattr(ast, "Constant") else ast.Num(n=5)],
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

    def test_update_denominator_string_keeps_denom_for_unsupported_unit(self):
        """update_denominator_string leaves $denom in place for unknown units."""
        result = update_denominator_string(
            "SUM(DATA) / SUM($denom)", "unsupported_unit"
        )
        assert "$denom" in result

    def test_update_normal_unit_string_capitalizes_per_wave(self):
        """update_normal_unit_string substitutes 'per wave' and capitalizes."""
        result = update_normal_unit_string("(Prefix + $normUnit)", "per_wave")
        assert "per wave" in result.lower()
        assert result[0].isupper()
