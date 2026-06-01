# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_db.py static methods."""

from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_db import db_analysis
from utils import schema
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
)


def make_dual_issue_arch_config(metric_name: str, peak_col: str = "Peak"):
    """Build an arch_config with a metric_table carrying one VALU row."""
    metric_df = pd.DataFrame(
        {
            "Metric": [metric_name],
            "Value": ["unused_expression"],
            peak_col: ["unused_peak_expression"],
        },
        index=pd.Index(["1.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {201: metric_df}
    arch_config.dfs_type = {201: "metric_table"}
    return arch_config


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

    # Test raw_pmc_df -> pmc_df substitution on flat single-index columns
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
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
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df_nan,
        sys_info,
        parse=False,
    )
    assert result is None

    # Series with NA values are preserved (not converted to None)
    pmc_df_mixed = pd.DataFrame({"Counter1": [10, np.nan, 30]})
    result = db_analysis.evaluate(
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
    result = db_analysis.evaluate(
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
    result = db_analysis.evaluate(
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
    result = db_analysis.evaluate(
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
        with (
            patch(
                "rocprof_compute_analyze.analysis_db.console_warning"
            ) as mock_warning,
            patch("rocprof_compute_analyze.analysis_db.console_debug") as mock_debug,
        ):
            result = db_analysis.evaluate(
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
    sys_info = {"base_value": 10, "gpu_arch": "gfx942"}

    # Mock BUILD_IN_VARS with dependency chain:
    # - PER_XCD_VAR: computed from base_value
    # - DERIVED_VAR: depends on PER_XCD_VAR via $PER_XCD_VAR
    mock_builtin_vars = {
        "PER_XCD_VAR": "$base_value * 2",  # Should be processed first -> 20
        "DERIVED_VAR": "$PER_XCD_VAR + 5",  # Depends on PER_XCD_VAR -> 25
    }

    with (
        patch(
            "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
            return_value="MI300",
        ),
        patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
        patch(
            "utils.utils_counter_defs.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
    ):
        db_analysis.calc_builtin_vars(
            pmc_df, sys_info, ["$PER_XCD_VAR", "$DERIVED_VAR"]
        )

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

    with (
        patch(
            "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
            return_value="MI300",
        ),
        patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
        patch(
            "utils.utils_counter_defs.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
    ):
        db_analysis.calc_builtin_vars(
            pmc_df, sys_info, ["$TOTAL_COUNT", "$SCALED_TOTAL"]
        )

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
    sys_info = {"scale": 100, "gpu_arch": "gfx942"}

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["sum", "scaled"],
        "value": [
            "to_sum(raw_pmc_df['Counter1'])",
            "ammolite__scale * 2",
        ],
    })

    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.iloc[0] == 60  # sum of Counter1
    assert result.iloc[1] == 200  # 100 * 2


def test_calc_dataframe_expressions_with_builtin_vars():
    """Test that calc_dataframe_expressions calls calc_builtin_vars first."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {"base": 5, "gpu_arch": "gfx942"}

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

    with (
        patch(
            "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
            return_value="MI300",
        ),
        patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
        patch(
            "utils.utils_counter_defs.get_build_in_vars",
            return_value=mock_builtin_vars,
        ),
    ):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert result.iloc[0] == 51
    # None from evaluate becomes NaN in pandas Series
    assert pd.isna(result.iloc[1])


# =============================================================================
# Noise-clamp warning + summary tests
# =============================================================================


def test_calc_expressions_noise_clamp():
    """Variance warnings fire only at workload level, summary once per workload.

    - evaluate(emit_variance_warnings=True) emits the per-metric warning when
      to_noise_clamp advances the global counter; the False kwarg stays silent.
    - calc_expressions emits exactly one variance warning per workload
      (kernel-level pass is silent) and calls print_noise_clamp_summary once.
    """
    workload_path = "/fake/workload"
    noise_clamp_expression = (
        "to_noise_clamp(to_min(raw_pmc_df['DIFF']), to_max(raw_pmc_df['REF']))"
    )
    # Two distinct kernels so groupby yields two kernel-level evaluate calls
    # in addition to one workload-level call. Without the kwarg gate the
    # unguarded code would emit three warnings; with the gate, exactly one.
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b"],
        "DIFF": [-100.0, -100.0],
        "REF": [1000.0, 1000.0],
    })
    expression_template = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["clamped"],
        "value": [noise_clamp_expression],
    })
    sys_info_df = pd.DataFrame([{"placeholder": 1, "gpu_arch": "gfx942"}])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pmc_df}
    analyzer._metric_expression_data_per_workload = {workload_path: expression_template}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._roofline_ceilings_per_workload = {workload_path: {}}
    analyzer._runs = {workload_path: MagicMock(sys_info=sys_info_df)}
    analyzer._arch_configs = MagicMock()

    # Direct evaluate kwarg behavior.
    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=True,
        )
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric: direct_test" in warning_call.args[0]
        ]
        assert len(variance_warning_calls) == 1
        assert get_noise_clamp_warnings()["count"] >= 1

    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test_off",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=False,
        )
        assert get_noise_clamp_warnings()["count"] >= 1
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric:" in warning_call.args[0]
        ]
        assert variance_warning_calls == []

    # calc_expressions per-workload bracket.
    clear_noise_clamp_warnings()
    with (
        patch("rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}),
        patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as console_warning_mock,
        patch(
            "rocprof_compute_analyze.analysis_db.print_noise_clamp_summary"
        ) as print_noise_clamp_summary_mock,
        patch.object(db_analysis, "validate_dual_issue_metrics"),
    ):
        analyzer.calc_expressions()

    variance_warning_calls = [
        warning_call
        for warning_call in console_warning_mock.call_args_list
        if "Variance corrected for metric:" in warning_call.args[0]
    ]
    assert len(variance_warning_calls) == 1
    assert "1.1 - clamped" in variance_warning_calls[0].args[0]
    print_noise_clamp_summary_mock.assert_called_once()
    assert get_noise_clamp_warnings()["count"] >= 1


# =============================================================================
# _derive_pop_values tests
# =============================================================================


class TestDerivePopValues:
    """Tests for db_analysis._derive_pop_values."""

    def _make_values_df(
        self,
        metric_ids: list[str],
        value_names: list[str],
        values: list[float],
        kernel_names: Optional[list[str]] = None,
    ):
        """Build a long-format values DataFrame as produced by calc_expressions."""
        data = {
            "metric_id": metric_ids,
            "value_name": value_names,
            "value": values,
        }
        if kernel_names is not None:
            data["kernel_name"] = kernel_names
        return pd.DataFrame(data)

    def test_pop_true_metric_appends_pct_of_peak_row(self):
        """A pop-enabled metric produces one new Pct of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 200.0],
        )
        new_rows = db_analysis._derive_pop_values({"1.1"}, values_df)
        assert len(new_rows) == 1
        assert new_rows[0]["value_name"] == "Pct of Peak"
        assert new_rows[0]["value"] == pytest.approx(25.0)

    def test_multi_kernel_produces_one_row_per_kernel(self):
        """Calling once per kernel produces one Pct of Peak row per kernel."""
        kernel_a_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[100.0, 200.0],
            kernel_names=["kernel_a", "kernel_a"],
        )
        kernel_b_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[60.0, 300.0],
            kernel_names=["kernel_b", "kernel_b"],
        )
        rows_a = db_analysis._derive_pop_values({"1.1"}, kernel_a_df)
        rows_b = db_analysis._derive_pop_values({"1.1"}, kernel_b_df)
        assert len(rows_a) == 1
        assert rows_a[0]["value"] == pytest.approx(50.0)  # 100/200*100
        assert len(rows_b) == 1
        assert rows_b[0]["value"] == pytest.approx(20.0)  # 60/300*100

    def test_pop_false_metric_produces_no_pct_row(self):
        """A metric not in pop_metric_ids produces no Pct of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 100.0],
        )
        new_rows = db_analysis._derive_pop_values(set(), values_df)
        assert new_rows == []

    def test_incomplete_data_skips_metric(self):
        """A metric missing Peak or Avg/Value must be skipped gracefully."""
        incomplete_cases = [
            # Only "Avg" present -- no "Peak" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Avg"], values=[50.0]
            ),
            # Only "Peak" present -- no "Avg" or "Value" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Peak"], values=[100.0]
            ),
        ]
        for incomplete_values in incomplete_cases:
            new_rows = db_analysis._derive_pop_values({"1.1"}, incomplete_values)
            assert new_rows == []


# =============================================================================
# Dual-issue VALU validation tests
# =============================================================================


def test_validate_dual_issue_metrics_emits_warning_above_peak():
    """Long-format VALU Utilization above peak triggers the dual-issue warning."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU Utilization can go up to 200%" in msg


def test_validate_dual_issue_metrics_silent_below_peak():
    """Below-peak VALU Utilization stays silent."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [80.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()


def test_validate_dual_issue_metrics_uses_peak_empirical_fallback():
    """Peak (Empirical) wins when present; falls back to Peak otherwise."""
    arch_config = make_dual_issue_arch_config(
        "VALU FLOPs (F64)", peak_col="Peak (Empirical)"
    )
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak (Empirical)"],
        "value": [600.0, 400.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU FLOPs can exceed the peak value" in msg


def test_validate_dual_issue_metrics_appends_valu2_suffix_on_gfx950():
    """gfx950 with non-zero SQ_ACTIVE_INST_VALU2 appends the confirmation."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"SQ_ACTIVE_INST_VALU2": [1, 2, 3]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx950"},
            workload_values_df,
            arch_config,
        )

    msg = console_warning_mock.call_args.args[0]
    assert "Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter" in msg


def test_validate_dual_issue_metrics_skips_non_metric_table_dfs():
    """dfs entries whose dfs_type is not metric_table are ignored."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    arch_config.dfs_type = {201: "raw_csv_table"}
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()
