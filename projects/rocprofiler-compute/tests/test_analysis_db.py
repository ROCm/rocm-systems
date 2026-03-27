# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_db.py static methods."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from rocprof_compute_analyze.analysis_db import db_analysis

# =============================================================================
# db_analysis.evaluate() tests
# =============================================================================


def test_evaluate_parse_false_basic_expressions():
    """Test parse=False mode with basic expressions and substitutions."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"numCUs": 64, "clock_speed": 1500}

    # Test raw_pmc_df['pmc_perf'] substitution
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['pmc_perf']['Counter1']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert list(result) == [10, 20, 30]

    # Test ammolite__ substitution for sys_info access
    result = db_analysis.evaluate(
        "test_metric",
        "ammolite__numCUs * 2",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 128

    # Test expression with helper function
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['pmc_perf']['Counter1'])",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 60


def test_evaluate_parse_true_basic_expressions():
    """Test parse=True mode with $ substitution and AST transformation."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [2, 4, 6],
    })
    sys_info = {"numCUs": 64, "multiplier": 2}

    # Test $variable substitution
    result = db_analysis.evaluate(
        "test_metric",
        "$numCUs * $multiplier",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 128

    # Test AST transformation with SUPPORTED_CALL functions (SUM -> to_sum)
    # and bare identifiers (Counter1 -> raw_pmc_df["Counter1"])
    result = db_analysis.evaluate(
        "test_metric",
        "SUM(Counter1)",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 60

    # Test combined $ substitution and column access with AVG
    result = db_analysis.evaluate(
        "test_metric",
        "AVG(Counter1) + $numCUs",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 84  # avg(10,20,30)=20 + 64


def test_evaluate_none_and_na_handling():
    """Test evaluate() handling of None and NA values."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    # Explicit None in expression result returns None without warning
    result = db_analysis.evaluate(
        "test_metric",
        "None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None

    # Scalar NA values (NaN) return None
    pmc_df_nan = pd.DataFrame({"Counter1": [np.nan, np.nan, np.nan]})
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['pmc_perf']['Counter1'])",
        pmc_df_nan,
        sys_info,
        parse=False,
    )
    assert result is None

    # Series with NA values are preserved (not converted to None)
    pmc_df_mixed = pd.DataFrame({"Counter1": [10, np.nan, 30]})
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['pmc_perf']['Counter1']",
        pmc_df_mixed,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 30

    # Exceptions return None gracefully
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['pmc_perf']['NonExistent']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None


def test_evaluate_with_none_in_formula_does_not_nullify_valid_result():
    """
    Test that expressions containing 'None' in formula string
    still return valid results when evaluation produces a value.

    This is a regression test for the bugfix where expressions like
    .where(..., None) were incorrectly returning None even when
    the actual result was valid.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 0, 3],  # Has a zero for conditional
    })
    sys_info = {}

    # Expression with None as fallback in .where() - should return valid result
    # when condition is met for at least some values
    result = db_analysis.evaluate(
        "test_metric",
        "(raw_pmc_df['pmc_perf']['Counter1'] / "
        "raw_pmc_df['pmc_perf']['Counter2'].where("
        "raw_pmc_df['pmc_perf']['Counter2'] != 0, None))",
        pmc_df,
        sys_info,
        parse=False,
    )
    # Result should be a Series, not None
    assert result is not None
    assert isinstance(result, pd.Series)

    # Expression that literally has "None" string but evaluates to a number
    result = db_analysis.evaluate(
        "test_metric",
        "10 if True else None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 10


# =============================================================================
# db_analysis.calc_builtin_vars() tests
# =============================================================================


def test_calc_builtin_vars_processes_per_xcd_first():
    """
    Test that PER_XCD variables are processed before non-PER_XCD variables,
    allowing non-PER_XCD vars to reference PER_XCD vars via $placeholder.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [100, 200],
    })
    sys_info = {"base_value": 10}

    # Mock BUILD_IN_VARS with dependency chain:
    # - PER_XCD_VAR: computed from base_value
    # - DERIVED_VAR: depends on PER_XCD_VAR via $PER_XCD_VAR
    mock_builtin_vars = {
        "PER_XCD_VAR": "$base_value * 2",  # Should be processed first -> 20
        "DERIVED_VAR": "$PER_XCD_VAR + 5",  # Depends on PER_XCD_VAR -> 25
    }

    with patch("rocprof_compute_analyze.analysis_db.BUILD_IN_VARS", mock_builtin_vars):
        result = db_analysis.calc_builtin_vars(pmc_df, sys_info)

    # Verify PER_XCD var was computed
    assert sys_info["PER_XCD_VAR"] == 20

    # Verify DERIVED_VAR used the computed PER_XCD_VAR value
    assert sys_info["DERIVED_VAR"] == 25

    # Verify pmc_df is returned unchanged
    pd.testing.assert_frame_equal(result, pmc_df)


def test_calc_builtin_vars_with_dataframe_expressions():
    """Test builtin vars that operate on DataFrame columns."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
    })
    sys_info = {"multiplier": 2}

    # Use SUPPORTED_CALL function names (SUM -> to_sum via CodeTransformer)
    mock_builtin_vars = {
        "TOTAL_COUNT": "SUM(Counter1)",  # 60
        "SCALED_TOTAL": "$TOTAL_COUNT * $multiplier",  # 120
    }

    with patch("rocprof_compute_analyze.analysis_db.BUILD_IN_VARS", mock_builtin_vars):
        db_analysis.calc_builtin_vars(pmc_df, sys_info)

    assert sys_info["TOTAL_COUNT"] == 60
    assert sys_info["SCALED_TOTAL"] == 120


# =============================================================================
# db_analysis.calc_dataframe_expressions() tests
# =============================================================================


def test_calc_dataframe_expressions_applies_evaluate_to_rows():
    """Test that expressions are evaluated for each row of expression_df."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"scale": 100}

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["sum", "scaled"],
        "value": [
            "to_sum(raw_pmc_df['pmc_perf']['Counter1'])",
            "ammolite__scale * 2",
        ],
    })

    with patch("rocprof_compute_analyze.analysis_db.BUILD_IN_VARS", {}):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.iloc[0] == 60  # sum of Counter1
    assert result.iloc[1] == 200  # 100 * 2


def test_calc_dataframe_expressions_with_builtin_vars():
    """Test that calc_dataframe_expressions calls calc_builtin_vars first."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {"base": 5}

    # Expression references a builtin var that gets computed
    mock_builtin_vars = {
        "COMPUTED_VAR": "$base * 10",  # 50
    }

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["test", "none_result"],
        "value": [
            "ammolite__COMPUTED_VAR + 1",  # Should be 51
            "None",
        ],
    })

    with patch("rocprof_compute_analyze.analysis_db.BUILD_IN_VARS", mock_builtin_vars):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert result.iloc[0] == 51
    # None from evaluate becomes NaN in pandas Series
    assert pd.isna(result.iloc[1])
