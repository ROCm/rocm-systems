# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.metric_evaluator."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator


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
