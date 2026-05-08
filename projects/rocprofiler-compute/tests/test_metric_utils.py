# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.* modules."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.aggregation import (
    to_concat,
    to_int,
    to_max,
    to_median,
    to_min,
    to_mod,
    to_quantile,
    to_round,
    to_std,
)
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import (
    CodeTransformer,
    build_eval_string,
    gen_counter_list,
    update_denominator_string,
    update_normal_unit_string,
)
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.utils_common import calc_builtin_var

# =============================================================================
# Tests for utils.metrics.aggregation
# =============================================================================


class TestAggregation:
    """Tests for utils.metrics.aggregation."""

    def test_to_min_with_all_none_raises(self):
        """to_min raises TypeError when every argument is None."""
        with pytest.raises(TypeError):
            to_min(None, None)

    def test_to_min_with_partial_none_raises(self):
        """to_min raises TypeError when any argument is None."""
        with pytest.raises(TypeError):
            to_min(None, 5)

    def test_to_min_returns_minimum_value(self):
        """to_min returns the smallest value among its scalar arguments."""
        assert to_min(7, 3, 9, 1) == 1, "to_min should return the smallest value"

    def test_to_max_with_all_none_raises(self):
        """to_max raises TypeError when every argument is None."""
        with pytest.raises(TypeError):
            to_max(None, None)

    def test_to_max_with_partial_none_raises(self):
        """to_max raises TypeError when any argument is None."""
        with pytest.raises(TypeError):
            to_max(None, 5)

    def test_to_max_returns_maximum_value(self):
        """to_max returns the largest value among its scalar arguments."""
        assert to_max(7, 3, 9, 1) == 9, "to_max should return the largest value"

    def test_to_median_returns_nan_for_none(self):
        """to_median returns np.nan when the input is None."""
        assert np.isnan(to_median(None)), "to_median should return np.nan for None"

    def test_to_median_raises_for_unsupported_type(self):
        """to_median raises for non-Series, non-None inputs."""
        with pytest.raises(Exception, match="unsupported type"):
            to_median("invalid_string")

    def test_to_std_raises_for_unsupported_type(self):
        """to_std raises for non-Series inputs."""
        with pytest.raises(Exception, match="unsupported type"):
            to_std("invalid_string")

    def test_to_int_returns_nan_for_none(self):
        """to_int returns np.nan when the input is None."""
        assert np.isnan(to_int(None)), "to_int should return np.nan for None"

    def test_to_int_raises_for_unsupported_list_type(self):
        """to_int raises when given a list, which is not a supported input."""
        with pytest.raises(Exception, match="unsupported type"):
            to_int(["list", "not", "supported"])

    def test_to_quantile_returns_nan_for_none(self):
        """to_quantile returns np.nan when the input is None."""
        assert np.isnan(to_quantile(None, 0.5)), (
            "to_quantile should return np.nan for None"
        )

    def test_to_quantile_raises_for_unsupported_type(self):
        """to_quantile raises for non-Series, non-None inputs."""
        with pytest.raises(Exception, match="unsupported type"):
            to_quantile("invalid_string", 0.5)

    def test_to_concat_concatenates_strings(self):
        """to_concat joins two string arguments without a separator."""
        assert to_concat("hello", "world") == "helloworld", (
            "to_concat should join strings without a separator"
        )

    def test_to_concat_converts_numbers_to_strings(self):
        """to_concat coerces numeric arguments to strings before joining."""
        assert to_concat(123, 456) == "123456", (
            "to_concat should coerce numeric arguments to strings"
        )

    def test_to_round_rounds_series_values(self):
        """to_round rounds every element of a Series to the requested precision."""
        series = pd.Series([1.234, 2.567, 3.890])
        result = to_round(series, 2)
        expected = pd.Series([1.23, 2.57, 3.89])
        pd.testing.assert_series_equal(result, expected)

    def test_to_round_rounds_scalar_value(self):
        """to_round rounds a scalar value to the requested precision."""
        assert to_round(3.14159, 2) == 3.14, (
            "to_round should round a scalar value to the requested precision"
        )

    def test_to_mod_returns_series_modulo(self):
        """to_mod applies modulo element-wise to a Series."""
        series = pd.Series([10, 15, 20])
        result = to_mod(series, 3)
        expected = pd.Series([1, 0, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_to_mod_returns_scalar_modulo(self):
        """to_mod returns the modulo of two scalar arguments."""
        assert to_mod(10, 3) == 1, (
            "to_mod should return the modulo of two scalar arguments"
        )


# =============================================================================
# Tests for utils.metrics.expression
# =============================================================================


class TestExpression:
    """Tests for utils.metrics.expression."""

    def test_build_eval_string_raises_when_coll_level_is_none(self):
        """build_eval_string raises when coll_level is None."""
        with pytest.raises(Exception, match="coll_level can not be None"):
            build_eval_string("AVG(SQ_WAVES)", None, config={})

    def test_build_eval_string_returns_empty_for_empty_equation(self):
        """build_eval_string returns the empty string when given an empty equation."""
        assert build_eval_string("", "pmc_perf", config={}) == ""

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


# =============================================================================
# Tests for utils.metrics.evaluation_pipeline
# =============================================================================


class TestEvaluationPipeline:
    """Tests for utils.metrics.evaluation_pipeline."""

    def _build_eval_metric_inputs(self, metric_fields=None):
        """Build the dfs/sys_info/raw_pmc_df fixture used by eval_metric tests.

        Args:
            metric_fields: Optional dict mapping metric-field column names to
                cell values.
                Defaults to a fixture with both 'Value' and 'Average=None'.
        """
        if metric_fields is None:
            metric_fields = {
                "Value": "to_sum(raw_pmc_df['pmc_perf']['SQ_WAVES'])",
                "Average": None,
            }

        metric_field_columns = {
            field_name: [field_value]
            for field_name, field_value in metric_fields.items()
        }

        metric_df = pd.DataFrame({
            "Metric_ID": ["1.1.0"],
            "Metric": ["Test Metric"],
            **metric_field_columns,
        }).set_index("Metric_ID")
        dfs = {1: metric_df}
        dfs_type = {1: "metric_table"}
        sys_info = pd.Series({
            "ip_blocks": "standard",
            "gpu_arch": "gfx90a",
            "se_per_gpu": 4,
            "pipes_per_gpu": 4,
            "cu_per_gpu": 64,
            "simd_per_cu": 4,
            "sqc_per_gpu": 16,
            "lds_banks_per_cu": 32,
            "cur_sclk": 1800.0,
            "cur_mclk": 1200.0,
            "max_sclk": 2100.0,
            "max_mclk": 1600.0,
            "max_waves_per_cu": 40,
            "num_hbm_channels": 4,
            "total_l2_chan": 32,
            "num_xcd": 1,
            "wave_size": 64,
        })
        raw_pmc_df = {
            "pmc_perf": pd.DataFrame({
                "SQ_WAVES": [100, 200, 150],
                "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
            })
        }
        return metric_df, dfs, dfs_type, sys_info, raw_pmc_df

    def test_eval_metric_in_debug_mode(self):
        """eval_metric with debug=True invokes debug_row_tracker and writes back."""
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['pmc_perf']['SQ_WAVES'])",
            }
        )
        with (
            patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}),
            patch(
                "utils.metrics.evaluation_pipeline.debug_row_tracker"
            ) as mock_debug_row_tracker,
        ):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=True,
                config={},
            )

        mock_debug_row_tracker.assert_called_once()
        call_args = mock_debug_row_tracker.call_args
        assert call_args.args[0] == "Value"
        assert call_args.kwargs["show_inputs"] is True
        assert metric_df.loc["1.1.0", "Value"] == 450

    def test_eval_metric_computes_value_from_expression(self):
        """eval_metric writes the computed Value back to the metric DataFrame."""
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['pmc_perf']['SQ_WAVES'])",
            }
        )
        with patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
                config={},
            )
        assert metric_df.loc["1.1.0", "Value"] == 450

    def test_eval_metric_normalizes_falsey_average_to_empty_string(self):
        """eval_metric replaces a falsey Average value with the empty string."""
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={"Average": None}
        )
        assert metric_df.loc["1.1.0", "Average"] is None

        with patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
                config={},
            )
        assert metric_df.loc["1.1.0", "Average"] == ""


# =============================================================================
# Tests for utils.metrics.metric_evaluator
# =============================================================================


class TestMetricEvaluator:
    """Tests for utils.metrics.metric_evaluator."""

    def test_eval_expression_returns_na_when_eval_returns_none(self):
        """eval_expression returns 'N/A' when the evaluated expression yields None."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.return_value = None
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_returns_nan(self):
        """eval_expression returns 'N/A' when the evaluated expression yields NaN."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.return_value = np.nan
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_raises_type_error(self):
        """eval_expression returns 'N/A' when eval raises a TypeError."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.side_effect = TypeError("Mock exception")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_raises_name_error_empirical_peak(
        self,
    ):
        """eval_expression returns 'N/A' for empirical_peak NameError lookups."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.side_effect = NameError("empirical_peak")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_raises_key_error(self):
        """eval_expression returns 'N/A' when eval raises a KeyError."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.side_effect = KeyError("Some KeyError")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_raises_attribute_error(self):
        """eval_expression returns 'N/A' for a generic AttributeError."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with (
            patch("builtins.eval") as mock_eval,
            patch("builtins.compile"),
            patch("sys.exit"),
        ):
            mock_eval.side_effect = AttributeError("Some AttributeError")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_eval_expression_returns_na_when_eval_raises_nonetype_attribute_error(
        self,
    ):
        """eval_expression returns 'N/A' for a NoneType.get AttributeError."""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            mock_eval.side_effect = AttributeError(
                "'NoneType' object has no attribute 'get'"
            )
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def _make_evaluator(self, columns, sys_vars=None):
        """Build a MetricEvaluator from the given pmc_perf columns and sys_vars."""
        pmc_perf_df = pd.DataFrame(columns)
        raw_pmc_df = {"pmc_perf": pmc_perf_df}
        return MetricEvaluator(raw_pmc_df, sys_vars or {}, {})

    def _to_eval_str(self, equation):
        """Run a YAML-style equation through build_eval_string for pmc_perf."""
        return build_eval_string(equation, "pmc_perf", config={})

    def test_eval_expression_returns_na_for_division_by_all_zero_series(self):
        """All-zero Series denominator yields inf, mapped to 'N/A'."""
        evaluator = self._make_evaluator({
            "NUMERATOR": [100.0, 200.0, 300.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = self._to_eval_str("MIN(NUMERATOR / DENOMINATOR)")
        assert evaluator.eval_expression(eval_str) == "N/A", (
            f"Expected 'N/A', got: {evaluator.eval_expression(eval_str)}"
        )

    def test_eval_expression_returns_na_for_zero_over_zero_scalar(self):
        """SUM(0) / SUM(0) yields NaN, which eval_expression maps to 'N/A'."""
        evaluator = self._make_evaluator({
            "NUMERATOR": [0.0, 0.0, 0.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = self._to_eval_str("SUM(NUMERATOR) / SUM(DENOMINATOR)")
        assert evaluator.eval_expression(eval_str) == "N/A", (
            f"Expected 'N/A', got: {evaluator.eval_expression(eval_str)}"
        )

    def test_eval_expression_returns_correct_value_for_normal_division(self):
        """SUM(100*BUSY)/SUM(TOTAL) returns the expected float for non-zero data."""
        evaluator = self._make_evaluator({
            "BUSY": [800.0, 600.0, 400.0],
            "TOTAL": [1000.0, 1000.0, 1000.0],
        })
        eval_str = self._to_eval_str("SUM(100 * BUSY) / SUM(TOTAL)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert result == pytest.approx(60.0), (
            "SUM(100*[800,600,400]) / SUM([1000,1000,1000]) should be 60.0, "
            f"got {result}"
        )

    def test_eval_expression_returns_na_for_all_nan_numerator(self):
        """SUM of an all-NaN numerator propagates NaN, mapped to 'N/A'."""
        evaluator = self._make_evaluator({
            "A_sum": [np.nan, np.nan, np.nan],
            "B_sum": [10.0, 20.0, 30.0],
        })
        eval_str = self._to_eval_str("SUM(A_sum) / SUM(B_sum)")
        assert evaluator.eval_expression(eval_str) == "N/A", (
            f"Expected 'N/A', got: {evaluator.eval_expression(eval_str)}"
        )

    def test_eval_expression_returns_na_for_all_nan_denominator(self):
        """SUM of an all-NaN denominator propagates NaN, mapped to 'N/A'."""
        evaluator = self._make_evaluator({
            "A_sum": [100.0, 200.0, 300.0],
            "B_sum": [np.nan, np.nan, np.nan],
        })
        eval_str = self._to_eval_str("SUM(A_sum) / SUM(B_sum)")
        assert evaluator.eval_expression(eval_str) == "N/A", (
            f"Expected 'N/A', got: {evaluator.eval_expression(eval_str)}"
        )

    def test_eval_expression_handles_mixed_nan_and_valid_values(self):
        """SUM skips NaN values, producing a finite result for mixed data."""
        evaluator = self._make_evaluator({
            "X_sum": [100.0, np.nan, 300.0],
            "Y_sum": [10.0, 0.0, 30.0],
        })
        eval_str = self._to_eval_str("SUM(X_sum) / SUM(Y_sum)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert result == pytest.approx(10.0), (
            f"SUM([100,NaN,300]) / SUM([10,0,30]) should be 10.0, got {result}"
        )

    def test_eval_expression_uses_system_variable_as_denominator(self):
        """eval_expression resolves $var from sys_vars and divides correctly."""
        evaluator = self._make_evaluator(
            {"COUNTER": [100.0, 200.0]},
            sys_vars={"ammolite__var": 5},
        )
        eval_str = self._to_eval_str("SUM(COUNTER) / $var")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert result == pytest.approx(60.0), (
            f"SUM([100,200]) / 5 should be 60.0, got {result}"
        )

    def test_eval_expression_aggregates_past_partial_zeros_in_denominator(self):
        """SUM aggregates past zero entries in the denominator without erroring."""
        evaluator = self._make_evaluator({
            "LEVEL": [100.0, 200.0, 300.0],
            "REQ": [10.0, 0.0, 5.0],
        })
        eval_str = self._to_eval_str("SUM(LEVEL) / SUM(REQ)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert result == pytest.approx(40.0), (
            f"SUM([100,200,300]) / SUM([10,0,5]) should be 40.0, got {result}"
        )
