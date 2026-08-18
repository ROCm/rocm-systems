# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/metric_evaluator.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.expression import (
    build_eval_string,
    update_denominator_string,
)
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.utils_analysis import add_unit_counter
from utils.utils_counter_defs import SUPPORTED_DENOM, UNIT_COUNTER

# =============================================================================
# Tests for utils.metrics.metric_evaluator
# =============================================================================


class TestMetricEvaluator:
    """Tests for utils.metrics.metric_evaluator."""

    @pytest.mark.parametrize("expr", ["open('/etc/passwd')", "getattr(1, 'real')"])
    def test_eval_expression_cannot_reach_builtins(self, expr):
        """Builtins other than __import__ are suppressed at the eval site."""
        evaluator = MetricEvaluator(pd.DataFrame(), {}, {})
        with patch("utils.metrics.metric_evaluator.console_warning") as mock_warning:
            result = evaluator.eval_expression(expr)
        assert result == "N/A"
        assert mock_warning.called

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
        with patch("builtins.eval") as mock_eval, patch("builtins.compile"), patch(
            "sys.exit"
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
        """Build a MetricEvaluator from the given raw_pmc columns and sys_vars."""
        raw_pmc_df = pd.DataFrame(columns)
        return MetricEvaluator(raw_pmc_df, sys_vars or {}, {})

    def _to_eval_str(self, equation):
        """Run a YAML-style equation through build_eval_string."""
        return build_eval_string(equation)

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

    def test_eval_expression_returns_na_for_nullified_incomplete_kernel(self):
        """Both numerator and denominator all-NaN yields NaN, mapped to 'N/A'."""
        evaluator = self._make_evaluator({
            "NUMERATOR": [np.nan, np.nan, np.nan],
            "DENOMINATOR": [np.nan, np.nan, np.nan],
        })
        eval_str = self._to_eval_str("SUM(NUMERATOR) / SUM(DENOMINATOR)")
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

    def test_eval_expression_divide_by_zero_silenced_and_logged_at_debug(self):
        """
        Divide-by-zero (x/0 -> inf, 0/0 -> NaN) emits a numpy RuntimeWarning
        that is captured and logged via console_debug. The "evaluated to N/A"
        console_warning must not fire when a RuntimeWarning was caught; the
        function still returns 'N/A' for both cases.
        """
        cases = [
            # x/0 -> inf, taken by the np.isinf branch
            (
                {"NUMERATOR": [100.0, 200.0], "DENOMINATOR": [0.0, 0.0]},
                "SUM(NUMERATOR) / SUM(DENOMINATOR)",
            ),
            # 0/0 -> NaN
            (
                {"NUMERATOR": [0.0, 0.0], "DENOMINATOR": [0.0, 0.0]},
                "SUM(NUMERATOR) / SUM(DENOMINATOR)",
            ),
        ]

        for columns, equation in cases:
            evaluator = self._make_evaluator(columns)
            eval_str = self._to_eval_str(equation)
            with patch(
                "utils.metrics.metric_evaluator.console_warning"
            ) as mock_warning, patch(
                "utils.metrics.metric_evaluator.console_debug"
            ) as mock_debug:
                result = evaluator.eval_expression(eval_str)

            assert result == "N/A", (
                f"Expected 'N/A' for '{equation}' with {columns}, got {result}"
            )
            mock_warning.assert_not_called()
            debug_msgs = [str(call) for call in mock_debug.call_args_list]
            assert any("RuntimeWarning" in m for m in debug_msgs), (
                f"Expected RuntimeWarning in console_debug output for "
                f"'{equation}', got {debug_msgs}"
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

    def test_build_eval_string_rewrites_accum_alias_as_flat_column_lookup(self):
        """`*_ACCUM` aliases become flat ``raw_pmc_df['<alias>']`` lookups."""
        eval_str = self._to_eval_str(
            "SUM(SQ_INST_LEVEL_VMEM_ACCUM) / SUM(SQ_INSTS_VMEM)"
        )
        assert "raw_pmc_df['SQ_INST_LEVEL_VMEM_ACCUM']" in eval_str
        assert "raw_pmc_df['SQ_INSTS_VMEM']" in eval_str

    def test_eval_expression_resolves_accum_alias_column(self):
        """SUM(<alias>_ACCUM) / SUM(...) returns the expected ratio for flat data."""
        evaluator = self._make_evaluator({
            "SQ_INST_LEVEL_VMEM_ACCUM": [100.0, 200.0, 300.0],
            "SQ_INSTS_VMEM": [10.0, 20.0, 30.0],
        })
        eval_str = self._to_eval_str(
            "SUM(SQ_INST_LEVEL_VMEM_ACCUM) / SUM(SQ_INSTS_VMEM)"
        )
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert abs(result - 10.0) < 1e-9, (
            f"SUM([100,200,300]) / SUM([10,20,30]) should be 10.0, got {result}"
        )

    def test_eval_expression_aggregates_per_row_accum_alias_with_min(self):
        """MIN(<alias>_ACCUM / counter) computes the per-row minimum ratio."""
        evaluator = self._make_evaluator({
            "SQ_INST_LEVEL_VMEM_ACCUM": [100.0, 50.0, 300.0],
            "SQ_INSTS_VMEM": [10.0, 25.0, 30.0],
        })
        eval_str = self._to_eval_str("MIN(SQ_INST_LEVEL_VMEM_ACCUM / SQ_INSTS_VMEM)")
        result = evaluator.eval_expression(eval_str)
        assert isinstance(result, float)
        assert abs(result - 2.0) < 1e-9, f"MIN([10, 2, 10]) should be 2.0, got {result}"

    def test_add_unit_counter_adds_column_of_ones(self):
        """add_unit_counter injects UNIT_COUNTER with 1 per dispatch."""
        df = pd.DataFrame({"COUNTER": [10.0, 20.0, 30.0]})
        add_unit_counter(df)
        assert (df[UNIT_COUNTER] == 1).all()
        assert len(df[UNIT_COUNTER]) == 3

    # YAML metric form: SUM(num)/SUM($denom) for Avg, MIN/MAX(num/$denom) bounds.
    _AVG_MIN_MAX_EQUATIONS = [
        pytest.param(
            "SUM(SQ_WAVE_CYCLES) / SUM($denom)",
            "MIN(SQ_WAVE_CYCLES / $denom)",
            "MAX(SQ_WAVE_CYCLES / $denom)",
            id="count",
        ),
        pytest.param(
            "4 * SUM(SQ_ACTIVE_INST_ANY) / SUM($denom)",
            "4 * MIN(SQ_ACTIVE_INST_ANY / $denom)",
            "4 * MAX(SQ_ACTIVE_INST_ANY / $denom)",
            id="scaled",
        ),
        pytest.param(
            "SUM(SQ_WAVE_CYCLES + SQ_ACTIVE_INST_ANY) / SUM($denom)",
            "MIN((SQ_WAVE_CYCLES + SQ_ACTIVE_INST_ANY) / $denom)",
            "MAX((SQ_WAVE_CYCLES + SQ_ACTIVE_INST_ANY) / $denom)",
            id="composite",
        ),
    ]

    def _normalized_evaluator(self):
        """Evaluator over multi-dispatch counters with positive per-dispatch denoms.

        per_cycle's $GRBM_GUI_ACTIVE_PER_XCD built-in is precomputed as a Series,
        the way calc_builtin_vars supplies it in a real run.
        """
        df = pd.DataFrame({
            "SQ_WAVE_CYCLES": [120.0, 300.0, 210.0, 450.0, 180.0],
            "SQ_ACTIVE_INST_ANY": [80.0, 160.0, 300.0, 100.0, 220.0],
            "SQ_WAVES": [10.0, 20.0, 15.0, 25.0, 12.0],
            "GRBM_GUI_ACTIVE": [900.0, 1800.0, 1400.0, 2500.0, 1100.0],
            "Start_Timestamp": [0.0, 500.0, 1000.0, 1500.0, 2000.0],
            "End_Timestamp": [400.0, 1300.0, 1600.0, 2600.0, 2500.0],
        })
        add_unit_counter(df)
        sys_vars = {"ammolite__num_xcd": 2}
        sys_vars["ammolite__GRBM_GUI_ACTIVE_PER_XCD"] = MetricEvaluator(
            df, sys_vars, {}
        ).eval_expression(build_eval_string("(GRBM_GUI_ACTIVE / $num_xcd)"))
        return MetricEvaluator(df, sys_vars, {})

    @pytest.mark.parametrize("normal_unit", list(SUPPORTED_DENOM))
    @pytest.mark.parametrize("avg_eq, min_eq, max_eq", _AVG_MIN_MAX_EQUATIONS)
    def test_pooled_avg_stays_within_min_max(self, normal_unit, avg_eq, min_eq, max_eq):
        """Pooled Avg = SUM(num)/SUM($denom) stays within [Min, Max] for every unit."""
        evaluator = self._normalized_evaluator()

        def evaluate(equation):
            return evaluator.eval_expression(
                build_eval_string(update_denominator_string(equation, normal_unit))
            )

        avg = evaluate(avg_eq)
        minimum = evaluate(min_eq)
        maximum = evaluate(max_eq)
        assert minimum < maximum  # varied data keeps the bound non-trivial
        assert minimum <= avg <= maximum
