# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.expression_evaluator."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from rocprof_compute_analyze.analysis_db import db_analysis
from rocprof_compute_analyze.analysis_db import (
    report_evaluation_diagnostics as db_report_evaluation_diagnostics,
)
from utils.metrics.expression_evaluator import (
    calc_builtin_vars,
    evaluate,
    report_evaluation_diagnostics,
)
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
)


def drain_evaluation_diagnostics():
    """Discard messages left over from earlier evaluate() calls."""
    with patch("utils.metrics.expression_evaluator.console_warning"), patch(
        "utils.metrics.expression_evaluator.console_debug"
    ):
        report_evaluation_diagnostics()


def test_db_analysis_exposes_static_evaluator_aliases():
    assert isinstance(db_analysis.__dict__["evaluate"], staticmethod)
    assert isinstance(db_analysis.__dict__["calc_builtin_vars"], staticmethod)
    assert db_analysis.evaluate is evaluate
    assert db_analysis.calc_builtin_vars is calc_builtin_vars
    assert db_report_evaluation_diagnostics is report_evaluation_diagnostics


# =============================================================================
# evaluate() tests
# =============================================================================


def test_evaluate_variance_warning_gate():
    """Variance correction warns only when explicitly requested."""
    noise_clamp_expression = (
        "to_noise_clamp(to_min(raw_pmc_df['DIFF']), to_max(raw_pmc_df['REF']))"
    )
    pmc_df = pd.DataFrame({"DIFF": [-100.0], "REF": [1000.0]})

    clear_noise_clamp_warnings()
    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        evaluate(
            "direct_test",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=True,
        )
    mock_warning.assert_called_once_with("Variance corrected for metric: direct_test")
    assert get_noise_clamp_warnings()["count"] >= 1

    clear_noise_clamp_warnings()
    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        evaluate(
            "direct_test_off",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=False,
        )
    mock_warning.assert_not_called()
    assert get_noise_clamp_warnings()["count"] >= 1


def test_evaluate_parse_false_basic_expressions():
    """Test parse=False mode with basic expressions and substitutions."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"numCUs": 64, "clock_speed": 1500}

    # Test raw_pmc_df -> pmc_df substitution on flat single-index columns
    result = evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert list(result) == [10, 20, 30]

    # Test ammolite__ substitution for sys_info access
    result = evaluate(
        "test_metric",
        "ammolite__numCUs * 2",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 128

    # Test expression with helper function
    result = evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
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
    result = evaluate(
        "test_metric",
        "$numCUs * $multiplier",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 128

    # Test AST transformation with SUPPORTED_CALL functions (SUM -> to_sum)
    # and bare identifiers (Counter1 -> raw_pmc_df["Counter1"])
    result = evaluate(
        "test_metric",
        "SUM(Counter1)",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 60

    # Test combined $ substitution and column access with AVG
    result = evaluate(
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
    result = evaluate(
        "test_metric",
        "None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None

    # Scalar NA values (NaN) return None
    pmc_df_nan = pd.DataFrame({"Counter1": [np.nan, np.nan, np.nan]})
    result = evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df_nan,
        sys_info,
        parse=False,
    )
    assert result is None

    # Series with NA values are preserved (not converted to None)
    pmc_df_mixed = pd.DataFrame({"Counter1": [10, np.nan, 30]})
    result = evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df_mixed,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 30

    # Exceptions return None gracefully
    result = evaluate(
        "test_metric",
        "raw_pmc_df['NonExistent']",
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
    result = evaluate(
        "test_metric",
        "(raw_pmc_df['Counter1'] / "
        "raw_pmc_df['Counter2'].where("
        "raw_pmc_df['Counter2'] != 0, None))",
        pmc_df,
        sys_info,
        parse=False,
    )
    # Result should be a Series, not None
    assert result is not None
    assert isinstance(result, pd.Series)

    # Expression that literally has "None" string but evaluates to a number
    result = evaluate(
        "test_metric",
        "10 if True else None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 10


def test_evaluate_divide_by_zero_silenced_and_logged_at_debug():
    """
    Divide-by-zero (x/0 -> inf, 0/0 -> NaN) emits a numpy RuntimeWarning
    that is captured and logged via console_debug. The "evaluated to N/A"
    console_warning must not fire when a RuntimeWarning was caught.
    """
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    cases = [
        # x/0 yields scalar inf; evaluate() collapses to None
        "to_sum(raw_pmc_df['Counter1']) / 0",
        # 0/0 yields scalar NaN; evaluate() collapses to None
        "(to_sum(raw_pmc_df['Counter1']) * 0) / 0",
    ]

    for expr in cases:
        with patch(
            "utils.metrics.expression_evaluator.console_warning"
        ) as mock_warning:
            with patch(
                "utils.metrics.expression_evaluator.console_debug"
            ) as mock_debug:
                result = evaluate(
                    "test_metric",
                    expr,
                    pmc_df,
                    sys_info,
                    parse=False,
                )

        assert result is None, f"Expected None for '{expr}', got {result}"

        mock_warning.assert_not_called()
        debug_msgs = [str(call) for call in mock_debug.call_args_list]
        assert any("RuntimeWarning" in m for m in debug_msgs), (
            f"Expected RuntimeWarning in console_debug output for '{expr}', "
            f"got {debug_msgs}"
        )


# =============================================================================
# Evaluation diagnostics tests
# =============================================================================


def test_evaluate_reports_na_at_debug_level():
    """An expression that evaluates to N/A is reported as debug, not warning."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": []})

    evaluate(
        "test_metric",
        "to_max(raw_pmc_df['Counter1'])",
        pmc_df,
        {},
        parse=False,
    )

    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        with patch("utils.metrics.expression_evaluator.console_debug") as mock_debug:
            report_evaluation_diagnostics()

    mock_warning.assert_not_called()
    debug_msgs = [call.args[0] for call in mock_debug.call_args_list]
    assert any("evaluated to N/A" in msg for msg in debug_msgs), (
        f"Expected an N/A message at debug level, got {debug_msgs}"
    )


def test_evaluate_failure_message_names_the_exception_type():
    """A missing counter is reported as a warning naming the exception type."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['TCC_TAG_STALL_sum'])",
        pmc_df,
        {},
        parse=False,
    )

    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        report_evaluation_diagnostics()

    warning_msgs = [call.args[0] for call in mock_warning.call_args_list]
    assert len(warning_msgs) == 1, f"Expected one warning, got {warning_msgs}"
    assert "KeyError: 'TCC_TAG_STALL_sum'" in warning_msgs[0]


def test_evaluate_reports_a_repeated_failure_once():
    """The same failure across kernels is reported once, not once per kernel."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    for _ in range(5):
        evaluate(
            "test_metric",
            "to_sum(raw_pmc_df['Missing_Counter'])",
            pmc_df,
            {},
            parse=False,
        )

    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        report_evaluation_diagnostics()

    assert mock_warning.call_count == 1, (
        f"Expected one warning, got {mock_warning.call_args_list}"
    )


def test_report_evaluation_diagnostics_clears_collected_messages():
    """A second report emits nothing, so a later run starts clean."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Missing_Counter'])",
        pmc_df,
        {},
        parse=False,
    )
    drain_evaluation_diagnostics()

    with patch("utils.metrics.expression_evaluator.console_warning") as mock_warning:
        with patch("utils.metrics.expression_evaluator.console_debug") as mock_debug:
            report_evaluation_diagnostics()

    mock_warning.assert_not_called()
    mock_debug.assert_not_called()


# =============================================================================
# calc_builtin_vars() tests
# =============================================================================


def test_calc_builtin_vars_processes_per_xcd_first():
    """
    Test that PER_XCD variables are processed before non-PER_XCD variables,
    allowing non-PER_XCD vars to reference PER_XCD vars via $placeholder.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [100, 200],
    })
    sys_info = {"base_value": 10, "gpu_arch": "gfx942"}

    # Mock BUILD_IN_VARS with dependency chain:
    # - PER_XCD_VAR: computed from base_value
    # - DERIVED_VAR: depends on PER_XCD_VAR via $PER_XCD_VAR
    mock_builtin_vars = {
        "PER_XCD_VAR": "$base_value * 2",  # Should be processed first -> 20
        "DERIVED_VAR": "$PER_XCD_VAR + 5",  # Depends on PER_XCD_VAR -> 25
    }

    with patch(
        "utils.metrics.expression_evaluator.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "utils.metrics.expression_evaluator.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                calc_builtin_vars(pmc_df, sys_info, ["$PER_XCD_VAR", "$DERIVED_VAR"])

    # Verify PER_XCD var was computed
    assert sys_info["PER_XCD_VAR"] == 20

    # Verify DERIVED_VAR used the computed PER_XCD_VAR value
    assert sys_info["DERIVED_VAR"] == 25


def test_calc_builtin_vars_with_dataframe_expressions():
    """Test builtin vars that operate on DataFrame columns."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
    })
    sys_info = {"multiplier": 2, "gpu_arch": "gfx942"}

    # Use SUPPORTED_CALL function names (SUM -> to_sum via CodeTransformer)
    mock_builtin_vars = {
        "TOTAL_COUNT": "SUM(Counter1)",  # 60
        "SCALED_TOTAL": "$TOTAL_COUNT * $multiplier",  # 120
    }

    with patch(
        "utils.metrics.expression_evaluator.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "utils.metrics.expression_evaluator.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                calc_builtin_vars(pmc_df, sys_info, ["$TOTAL_COUNT", "$SCALED_TOTAL"])

    assert sys_info["TOTAL_COUNT"] == 60
    assert sys_info["SCALED_TOTAL"] == 120


def test_calc_builtin_vars_uses_gfx1250_gui_active_sum():
    """gfx1250 computes per-dispatch GUI-active cycles from the XCD sum."""
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE_sum": [800, 1600]})
    sys_info = {"num_xcd": 8, "gpu_arch": "gfx1250"}

    calc_builtin_vars(
        pmc_df,
        sys_info,
        ["$GRBM_GUI_ACTIVE_PER_XCD"],
    )

    pd.testing.assert_series_equal(
        sys_info["GRBM_GUI_ACTIVE_PER_XCD"],
        pd.Series([100.0, 200.0]),
        check_names=False,
    )
