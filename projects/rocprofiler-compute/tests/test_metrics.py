# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.* modules."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd

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

    def test_parser_utility_functions(self):
        """Test parser utility functions edge cases"""
        try:
            result = to_min(None, None)
            assert np.isnan(result), "to_min with all None should return nan"
        except TypeError:
            pass

        try:
            result = to_min(None, 5)
            assert False, "Should have crashed"
        except TypeError:
            pass

        result = to_min(7, 3, 9, 1)
        assert result == 1, "to_min should return minimum value"

        try:
            result = to_max(None, None)
            assert np.isnan(result), "to_max with all None should return nan"
        except TypeError:
            pass

        try:
            result = to_max(None, 5)
            assert False, "Should have crashed"
        except TypeError:
            pass

        result = to_max(7, 3, 9, 1)
        assert result == 9, "to_max should return maximum value"

        result = to_median(None)
        assert np.isnan(result), "to_median should return np.nan for None input"

        try:
            to_median("invalid_string")
            assert False, "to_median should raise exception for invalid type"
        except Exception as e:
            assert "unsupported type" in str(e)

        try:
            to_std("invalid_string")
            assert False, "to_std should raise exception for invalid type"
        except Exception as e:
            assert "unsupported type" in str(e)

        result = to_int(None)
        assert np.isnan(result), "to_int should return np.nan for None input"

        try:
            to_int(["list", "not", "supported"])
            assert False, "to_int should raise exception for invalid type"
        except Exception as e:
            assert "unsupported type" in str(e)

        result = to_quantile(None, 0.5)
        assert np.isnan(result), "to_quantile should return np.nan for None input"

        try:
            to_quantile("invalid_string", 0.5)
            assert False, "to_quantile should raise exception for invalid type"
        except Exception as e:
            assert "unsupported type" in str(e)

        result = to_concat("hello", "world")
        assert result == "helloworld", "to_concat should concatenate strings"

        result = to_concat(123, 456)
        assert result == "123456", "to_concat should convert to strings and concatenate"

        series = pd.Series([1.234, 2.567, 3.890])
        result = to_round(series, 2)
        expected = pd.Series([1.23, 2.57, 3.89])
        pd.testing.assert_series_equal(result, expected)

        result = to_round(3.14159, 2)
        assert result == 3.14, "to_round should round scalar values"

        series = pd.Series([10, 15, 20])
        result = to_mod(series, 3)
        expected = pd.Series([1, 0, 2])
        pd.testing.assert_series_equal(result, expected)

        result = to_mod(10, 3)
        assert result == 1, "to_mod should return modulo for scalars"


# =============================================================================
# Tests for utils.metrics.expression
# =============================================================================


class TestExpression:
    """Tests for utils.metrics.expression."""

    def test_parser_error_handling(self):
        """Test parser error handling paths"""
        try:
            build_eval_string("AVG(SQ_WAVES)", None, config={})
            assert False, "Should have raised exception for None coll_level"
        except Exception as e:
            assert "coll_level can not be None" in str(e)

        assert build_eval_string("", "pmc_perf", config={}) == ""
        assert update_denominator_string("", "per_wave") == ""

        sys_info = {"total_l2_chan": 32}
        try:
            calc_builtin_var("$unsupported_var", sys_info)
            assert False, "Should have raised exception for unsupported var"
        except SystemExit:
            pass

    def test_ast_transformer_edge_cases(self):
        """Simplified test focusing on the actual code paths"""
        transformer = CodeTransformer()

        unknown_call = ast.Call(
            func=ast.Name(id="UNKNOWN_FUNCTION", ctx=ast.Load()),
            args=[ast.Constant(value=5) if hasattr(ast, "Constant") else ast.Num(n=5)],
            keywords=[],
        )

        try:
            result = transformer.visit_Call(unknown_call)
            if hasattr(result.func, "id") and result.func.id == "UNKNOWN_FUNCTION":
                assert False, (
                    "Function name should have been changed or exception raised"
                )
        except Exception as e:
            assert "Unknown call" in str(e), (
                f"Expected 'Unknown call' in error, got: {str(e)}"
            )

        SUPPORTED_CALL = ast.Call(
            func=ast.Name(id="MIN", ctx=ast.Load()),
            args=[ast.Constant(value=5) if hasattr(ast, "Constant") else ast.Num(n=5)],
            keywords=[],
        )

        try:
            result = transformer.visit_Call(SUPPORTED_CALL)
            assert result.func.id == "to_min", (
                f"Expected 'to_min', got: {result.func.id}"
            )
        except Exception as e:
            assert False, f"Supported function call should not raise exception: {e}"

    def test_build_dfs_edge_cases(self):
        """Test build_dfs and gen_counter_list with various configurations"""
        visited, counters = gen_counter_list(None)
        assert not visited
        assert counters == []

        visited, counters = gen_counter_list(123)
        assert not visited
        assert counters == []

        visited, counters = gen_counter_list("AVG(SQ_WAVES + TCC_HIT)")
        assert visited
        assert "SQ_WAVES" in counters
        assert "TCC_HIT" in counters

        visited, counters = gen_counter_list("Start_Timestamp + End_Timestamp")
        assert visited

        visited, counters = gen_counter_list("INVALID SYNTAX !!!")
        assert not visited

    def test_update_functions_coverage(self):
        """Test update_denominator_string and update_norm_unit_string branches"""
        result = update_denominator_string("SUM(SQ_WAVES) / SUM($denom)", "per_wave")
        assert "$denom" not in result
        assert "SQ_WAVES" in result

        result = update_denominator_string("SUM(DATA) / SUM($denom)", "per_cycle")
        assert "$GRBM_GUI_ACTIVE_PER_XCD" in result

        result = update_denominator_string("SUM(DATA) / SUM($denom)", "per_second")
        assert "End_Timestamp - Start_Timestamp" in result

        result = update_denominator_string(
            "SUM(DATA) / SUM($denom)", "unsupported_unit"
        )
        assert "$denom" in result

        result = update_normal_unit_string("(Prefix + $normUnit)", "per_wave")
        assert "per wave" in result.lower()
        assert result[0].isupper()


# =============================================================================
# Tests for utils.metrics.evaluation_pipeline
# =============================================================================


class TestEvaluationPipeline:
    """Tests for utils.metrics.evaluation_pipeline."""

    def test_analyze_with_debug_mode(self):
        """Test analyze to cover debug paths in eval_metric via direct function call."""
        mock_dfs = {
            1: pd.DataFrame({
                "Metric_ID": ["1.1.0"],
                "Metric": ["Test Metric"],
                "Expr": ["AVG(SQ_WAVES)"],
                "coll_level": ["pmc_perf"],
            }).set_index("Metric_ID")
        }

        mock_dfs_type = {1: "metric_table"}

        class MockSysInfo:
            ip_blocks = "standard"
            se_per_gpu = 4
            pipes_per_gpu = 4
            cu_per_gpu = 64
            simd_per_cu = 4
            sqc_per_gpu = 16
            lds_banks_per_cu = 32
            cur_sclk = 1800.0
            cur_mclk = 1200.0
            max_sclk = 2100.0
            max_mclk = 1600.0
            max_waves_per_cu = 40
            num_hbm_channels = 4
            total_l2_chan = 32
            num_xcd = 1
            wave_size = 64

        sys_info = MockSysInfo()

        raw_pmc_df = {
            "pmc_perf": pd.DataFrame({
                "SQ_WAVES": [100, 200, 150],
                "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
                "End_Timestamp": [1000000, 2000000, 1500000],
                "Start_Timestamp": [0, 1000000, 500000],
            })
        }

        try:
            eval_metric(
                mock_dfs, mock_dfs_type, sys_info, raw_pmc_df, debug=True, config={}
            )
        except Exception:
            pass

    def test_eval_metric_writes_back_falsey_supported_fields(self):
        """Test eval_metric normalizes falsey supported fields on the DataFrame."""
        metric_df = pd.DataFrame({
            "Metric_ID": ["1.1.0"],
            "Metric": ["Test Metric"],
            "Value": ["to_sum(raw_pmc_df['pmc_perf']['SQ_WAVES'])"],
            "Average": [None],
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

        assert metric_df.loc["1.1.0", "Value"] == 450
        assert metric_df.loc["1.1.0", "Average"] == ""


# =============================================================================
# Tests for utils.metrics.metric_evaluator
# =============================================================================


class TestMetricEvaluator:
    """Tests for utils.metrics.metric_evaluator."""

    def test_metric_evaluation_no_valid_data(self):
        """Test emetric evaluation with no valid data"""
        metric_evaluator = MetricEvaluator({}, {}, {})
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"):
            # Test when eval returns None
            mock_eval.return_value = None
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            # Test when eval returns NaN
            mock_eval.return_value = np.nan
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            # Test when eval raises an exception
            mock_eval.side_effect = TypeError("Mock exception")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            mock_eval.side_effect = NameError("empirical_peak")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            mock_eval.side_effect = KeyError("Some KeyError")
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            with patch("sys.exit"):
                mock_eval.side_effect = AttributeError("Some AttributeError")
                assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

            mock_eval.side_effect = AttributeError(
                "'NoneType' object has no attribute 'get'"
            )
            assert metric_evaluator.eval_expression("Mock Metric") == "N/A"

    def test_metric_evaluator_division_by_zero(self):
        """Test MetricEvaluator.eval_expression handles division-by-zero cases.

        The evaluator must gracefully handle all denominator-zero and NaN scenarios
        that can arise from real counter data. This test exercises the checks around
        parser.py eval_expression (None, NaN, inf detection).
        """

        # ---------------------------------------------------------------
        # Helper: build a MetricEvaluator with the given pmc_perf columns
        # ---------------------------------------------------------------
        def make_evaluator(columns, sys_vars=None):
            pmc_perf_df = pd.DataFrame(columns)
            raw_pmc_df = {"pmc_perf": pmc_perf_df}
            return MetricEvaluator(raw_pmc_df, sys_vars or {}, {})

        # ---------------------------------------------------------------
        # Helper: transform a YAML-style equation through the full pipeline
        # ---------------------------------------------------------------
        def to_eval_str(equation):
            return build_eval_string(equation, "pmc_perf", config={})

        # ---------------------------------------------------------------
        # 1. Division by all-zero denominator -> inf -> "N/A"
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "NUMERATOR": [100.0, 200.0, 300.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = to_eval_str("MIN(NUMERATOR / DENOMINATOR)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", (
            "Division by all-zero Series should produce inf, caught as N/A"
        )

        # ---------------------------------------------------------------
        # 2. 0/0 scalar division -> NaN -> "N/A"
        #    SUM of all-zero returns 0.0; 0.0 / 0.0 = NaN
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "NUMERATOR": [0.0, 0.0, 0.0],
            "DENOMINATOR": [0.0, 0.0, 0.0],
        })
        eval_str = to_eval_str("SUM(NUMERATOR) / SUM(DENOMINATOR)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", "SUM(0) / SUM(0) should produce NaN, caught as N/A"

        # ---------------------------------------------------------------
        # 3. Normal case: all non-zero -> valid numeric result
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "BUSY": [800.0, 600.0, 400.0],
            "TOTAL": [1000.0, 1000.0, 1000.0],
        })
        eval_str = to_eval_str("SUM(100 * BUSY) / SUM(TOTAL)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert abs(result - 60.0) < 1e-9, (
            f"SUM(100*[800,600,400]) / SUM([1000,1000,1000]) should be 60.0, "
            f"got {result}"
        )

        # ---------------------------------------------------------------
        # 4. All-NaN numerator -> NaN propagation -> "N/A"
        #    SUM of all-NaN returns NaN; NaN / 60.0 = NaN
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "A_sum": [np.nan, np.nan, np.nan],
            "B_sum": [10.0, 20.0, 30.0],
        })
        eval_str = to_eval_str("SUM(A_sum) / SUM(B_sum)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", (
            "SUM(all-NaN) / SUM(valid) should produce NaN, caught as N/A"
        )

        # ---------------------------------------------------------------
        # 5. All-NaN denominator -> NaN propagation -> "N/A"
        #    600.0 / NaN = NaN
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "A_sum": [100.0, 200.0, 300.0],
            "B_sum": [np.nan, np.nan, np.nan],
        })
        eval_str = to_eval_str("SUM(A_sum) / SUM(B_sum)")
        result = evaluator.eval_expression(eval_str)
        assert result == "N/A", (
            "SUM(valid) / SUM(all-NaN) should produce NaN, caught as N/A"
        )

        # ---------------------------------------------------------------
        # 6. Mixed NaN and valid values -> NaN skipped by SUM, valid result
        #    SUM skips NaN: SUM([100, NaN, 300]) = 400, SUM([10, 0, 30]) = 40
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "X_sum": [100.0, np.nan, 300.0],
            "Y_sum": [10.0, 0.0, 30.0],
        })
        eval_str = to_eval_str("SUM(X_sum) / SUM(Y_sum)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert abs(result - 10.0) < 1e-9, (
            f"SUM([100,NaN,300]) / SUM([10,0,30]) should be 10.0, got {result}"
        )

        # ---------------------------------------------------------------
        # 7. System variable as denominator
        # ---------------------------------------------------------------
        evaluator = make_evaluator(
            {"COUNTER": [100.0, 200.0]},
            sys_vars={"ammolite__var": 5},
        )
        eval_str = to_eval_str("SUM(COUNTER) / $var")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert abs(result - 60.0) < 1e-9, (
            f"SUM([100, 200]) / 5 should be 60.0, got {result}"
        )

        # ---------------------------------------------------------------
        # 8. Partial zeros in denominator -> SUM aggregates past them
        # ---------------------------------------------------------------
        evaluator = make_evaluator({
            "LEVEL": [100.0, 200.0, 300.0],
            "REQ": [10.0, 0.0, 5.0],
        })
        eval_str = to_eval_str("SUM(LEVEL) / SUM(REQ)")
        result = evaluator.eval_expression(eval_str)
        # SUM([100,200,300]) / SUM([10,0,5]) = 600 / 15 = 40.0
        assert isinstance(result, float)
        assert abs(result - 40.0) < 1e-9, (
            f"SUM(LEVEL) / SUM(REQ) should be 40.0, got {result}"
        )
