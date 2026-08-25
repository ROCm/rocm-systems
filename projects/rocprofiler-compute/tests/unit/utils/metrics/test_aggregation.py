# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/aggregation.py."""

import numpy as np
import pandas as pd
import pytest

from utils.metrics.aggregation import (
    calc_pct_of_peak,
    to_bound_ratio,
    to_concat,
    to_int,
    to_max,
    to_median,
    to_min,
    to_mod,
    to_quantile,
    to_round,
    to_std,
    to_sum,
)

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

    def test_to_min_elementwise_between_series(self):
        """to_min returns element-wise minimum as a Series for two Series."""
        left = pd.Series([150.0, 80.0, 100.0])
        right = pd.Series([136.0, 100.0, 90.0])
        result = to_min(left, right)
        pd.testing.assert_series_equal(
            result,
            pd.Series([136.0, 80.0, 90.0]),
        )
        assert to_sum(result) == 306.0

    def test_to_bound_ratio_caps_per_row_percent(self):
        """to_bound_ratio caps inflated counter ratios at 100%."""
        num = pd.Series([152.0, 80.0])
        den = pd.Series([136.0, 100.0])
        result = to_bound_ratio(num, den)
        pd.testing.assert_series_equal(
            result,
            pd.Series([100.0, 80.0]),
        )

    def test_to_bound_ratio_scalar(self):
        """to_bound_ratio caps scalar ratios."""
        assert to_bound_ratio(152.0, 136.0) == 100.0
        assert to_bound_ratio(80.0, 100.0) == 80.0

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

    def test_calc_pct_of_peak_returns_correct_percentage(self):
        """calc_pct_of_peak returns 100.0 * value / peak for valid inputs."""
        assert calc_pct_of_peak(50.0, 200.0) == pytest.approx(25.0)

    def test_calc_pct_of_peak_returns_none_for_zero_peak(self):
        """calc_pct_of_peak returns None when peak is zero."""
        assert calc_pct_of_peak(50.0, 0.0) is None

    def test_calc_pct_of_peak_returns_none_for_nan_value(self):
        """calc_pct_of_peak returns None when value is NaN."""
        assert calc_pct_of_peak(np.nan, 200.0) is None

    def test_calc_pct_of_peak_returns_none_for_nan_peak(self):
        """calc_pct_of_peak returns None when peak is NaN."""
        assert calc_pct_of_peak(50.0, np.nan) is None

    def test_calc_pct_of_peak_returns_none_for_non_numeric(self):
        """calc_pct_of_peak returns None for non-numeric inputs."""
        assert calc_pct_of_peak("N/A", 200.0) is None
