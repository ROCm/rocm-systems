# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_db.py static methods."""

import copy
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import common
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from rocprof_compute_analyze.analysis_db import db_analysis
from utils import analysis_orm as orm
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
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as mock_warning:
            with patch(
                "rocprof_compute_analyze.analysis_db.console_debug"
            ) as mock_debug:
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

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
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

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
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

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                result = db_analysis.calc_dataframe_expressions(
                    pmc_df, sys_info, expression_df
                )

    assert result.iloc[0] == 51
    # None from evaluate becomes NaN in pandas Series
    assert pd.isna(result.iloc[1])


def test_calc_dataframe_expressions_empty_returns_assignable_series():
    """An empty expression_df returns an empty Series, not a DataFrame."""
    expression_df = pd.DataFrame(columns=["metric_id", "value_name", "value"])

    result = db_analysis.calc_dataframe_expressions(
        pd.DataFrame({"Counter1": [1, 2, 3]}),
        {"gpu_arch": "gfx942"},
        expression_df,
    )

    assert isinstance(result, pd.Series)
    assert result.empty
    # Reproduces the call site: assigning the result as a single column must
    # not raise "Columns must be same length as key".
    expression_df["value"] = result


# =============================================================================
# calc_metrics_data tests
# =============================================================================


def test_calc_metrics_data_builds_rows_and_preserves_schema():
    """Metric tables expand into rows with table-level fields resolved once;
    non-metric tables are skipped and the output frames keep their columns."""
    workload_path = "/fake/workload"
    metric_df = pd.DataFrame(
        {
            "Metric": ["Grid Size"],
            "Avg": [" 10 "],
            "Min": [" 5 "],
            "Max": [" 20 "],
            "Unit": ["Work items"],
            "Description": ["Grid size desc"],
        },
        index=pd.Index(["7.1.0"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    # Table 1 has no Metric/Channel column and is skipped; table 701 maps to
    # panel 700 (table_name) and sub-table 701 (sub_table_name).
    arch_config.dfs = {
        1: pd.DataFrame({"from_csv": ["pmc_kernel_top.csv"]}),
        701: metric_df,
    }
    arch_config.panel_configs = {
        700: {
            "id": 700,
            "title": "Wavefront",
            "data source": [
                {"metric_table": {"id": 701, "title": "Wavefront Launch Stats"}}
            ],
        }
    }

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pd.DataFrame({"Counter1": [1]})}
    analyzer._runs = {
        workload_path: MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]))
    }
    analyzer._arch_configs = {"gfx942": arch_config}

    metrics_info, expressions = analyzer.calc_metrics_data()

    info = metrics_info[workload_path]
    assert "pct_of_peak" in info.columns
    assert list(info["metric_id"]) == ["7.1.0"]
    assert list(info["name"]) == ["Grid Size"]
    assert list(info["table_name"]) == ["Wavefront"]
    assert list(info["sub_table_name"]) == ["Wavefront Launch Stats"]
    assert not bool(info["pct_of_peak"].iloc[0])

    exprs = expressions[workload_path]
    assert list(exprs.columns) == ["metric_id", "value_name", "value"]
    # Metric/Unit/Description are non-expression columns, so only Avg/Min/Max
    # expand into expression rows, stripped and in dataframe-column order.
    assert list(exprs["value_name"]) == ["Avg", "Min", "Max"]
    assert list(exprs["value"]) == ["10", "5", "20"]
    assert set(exprs["metric_id"]) == {"7.1.0"}


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
    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as console_warning_mock:
            with patch(
                "rocprof_compute_analyze.analysis_db.print_noise_clamp_summary"
            ) as print_noise_clamp_summary_mock:
                with patch.object(db_analysis, "validate_dual_issue_metrics"):
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
# _derive_pct_of_peak_values tests
# =============================================================================


class TestDerivePctOfPeakValues:
    """Tests for db_analysis._derive_pct_of_peak_values."""

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

    def test_pct_of_peak_true_metric_appends_percent_of_peak_row(self):
        """A pct_of_peak-enabled metric produces one new Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 200.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values({"1.1"}, values_df)
        assert len(new_rows) == 1
        assert new_rows[0]["value_name"] == "Percent of Peak"
        assert new_rows[0]["value"] == pytest.approx(25.0)

    def test_multi_kernel_produces_one_row_per_kernel(self):
        """Calling once per kernel produces one Percent of Peak row per kernel."""
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
        rows_a = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_a_df)
        rows_b = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_b_df)
        assert len(rows_a) == 1
        assert rows_a[0]["value"] == pytest.approx(50.0)  # 100/200*100
        assert len(rows_b) == 1
        assert rows_b[0]["value"] == pytest.approx(20.0)  # 60/300*100

    def test_pct_of_peak_false_metric_produces_no_pct_row(self):
        """A metric not in pct_of_peak_metric_ids produces no Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 100.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values(set(), values_df)
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
            new_rows = db_analysis._derive_pct_of_peak_values(
                {"1.1"}, incomplete_values
            )
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


# =============================================================================
# PC-sampling population
# =============================================================================


def make_pc_sampling_dispatch(dispatch_id, kernel_id):
    """Build one PC-sampling kernel dispatch record."""
    return {
        "start_timestamp": 0,
        "end_timestamp": 0,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": 1},
        },
    }


def make_pc_sampling_tool_data():
    """Two offsets under two kernels sharing one code object, with counts."""
    stall = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT"
    inst_type = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
    return {
        "metadata": {"pid": 42},
        "buffer_records": {
            "pc_sample_host_trap": [],
            "pc_sample_stochastic": [
                {
                    "inst_index": 0,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x10},
                        "dispatch_id": 0,
                        "wave_issued": False,
                        "snapshot": {"stall_reason": stall},
                        "inst_type": inst_type,
                    },
                },
                {
                    "inst_index": 1,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x20},
                        "dispatch_id": 1,
                        "wave_issued": True,
                        "snapshot": {},
                        "inst_type": inst_type,
                    },
                },
            ],
            "kernel_dispatch": [
                make_pc_sampling_dispatch(0, 100),
                make_pc_sampling_dispatch(1, 101),
            ],
        },
        "strings": {
            "pc_sample_instructions": ["v_mov", "v_add"],
            "pc_sample_comments": ["/s/a.cpp:1", "/s/a.cpp:2"],
        },
        "kernel_symbols": [
            {
                "kernel_id": 100,
                "code_object_id": 5,
                "kernel_name": "_Z7vecCopyv.kd",
                "formatted_kernel_name": "vecCopy",
            },
            {
                "kernel_id": 101,
                "code_object_id": 5,
                "kernel_name": "vecAdd.kd",
                "formatted_kernel_name": "vecAdd",
            },
        ],
        "code_objects": [{"code_object_id": 5, "load_base": 0x1000}],
        "agents": [],
    }


def make_colliding_pc_sampling_tool_data(process_id: int, sample_count: int):
    tool_data = make_pc_sampling_tool_data()
    shared_sample = tool_data["buffer_records"]["pc_sample_stochastic"][0]
    tool_data["metadata"]["pid"] = process_id
    tool_data["buffer_records"]["pc_sample_stochastic"] = [
        copy.deepcopy(shared_sample) for _ in range(sample_count)
    ]
    tool_data["buffer_records"]["kernel_dispatch"] = [make_pc_sampling_dispatch(0, 100)]
    tool_data["strings"] = {
        "pc_sample_instructions": ["v_mov"],
        "pc_sample_comments": ["/s/shared.cpp:7"],
    }
    tool_data["kernel_symbols"] = [tool_data["kernel_symbols"][0]]
    return tool_data


def make_pc_sampling_only_database_analyzer(workload_path, tool_data_records):
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
        )
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(tool_data_records)
    }
    analyzer._roofline_data_per_kernel = {}
    analyzer._roofline_data_per_workload = {}
    return analyzer


def make_counter_backed_database_analyzer(
    workload_path,
    filter_blocks,
    tool_data_records,
):
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
        )
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": filter_blocks}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "dispatch_id": 7,
                "pid": None,
                "kernel_name": "vecCopy",
                "gpu_id": 0,
                "start_timestamp": 10,
                "end_timestamp": 20,
            }
        ])
    }
    analyzer._roofline_data_per_kernel = {workload_path: pd.DataFrame()}
    analyzer._roofline_data_per_workload = {}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._kernel_values_data_per_workload = {}
    analyzer._workload_values_data_per_workload = {}
    return analyzer


def run_analysis_with_existing_database(analyzer):
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "create_views"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()


def run_analysis_with_materialized_views(analyzer):
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()


def test_add_pc_sampling_data_no_tool_data_is_noop(db_session):
    """A workload without tool data inserts no rows."""
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {"/fake/workload": []}

    code_object_stores = analyzer.add_pc_sampling_data("/fake/workload", workload, {})
    db_session.commit()

    assert code_object_stores == {}
    assert db_session.query(orm.CodeObjectStore).count() == 0
    assert db_session.query(orm.InstructionLine).count() == 0


def test_add_pc_sampling_data_populates_and_attributes_kernels(db_session):
    """Instruction lines use the PID-scoped kernels and transient store registry."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        (42, "vecCopy"): orm.Kernel(kernel_name="vecCopy", workload=workload),
        (42, "vecAdd"): orm.Kernel(kernel_name="vecAdd", workload=workload),
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path, workload, kernel_objs
    )
    db_session.commit()

    code_object = db_session.query(orm.CodeObjectStore).one()
    assert set(code_object_stores) == {(42, 5)}
    assert code_object_stores[(42, 5)] is code_object
    assert code_object.code_object_uuid is not None
    assert code_object.load_base == 0x1000

    lines = db_session.query(orm.InstructionLine).all()
    kernel_by_offset = {
        line.code_object_offset: line.kernel.kernel_name for line in lines
    }
    assert kernel_by_offset == {0x10: "vecCopy", 0x20: "vecAdd"}

    # The stalled sample carries a stall-reason count; both carry an inst type.
    stalled = next(line for line in lines if line.code_object_offset == 0x10)
    assert stalled.pc_sample_state.stall_count == 1
    assert {
        r.stall_reason_lookup.text for r in stalled.pc_sample_state.stall_reasons
    } == {"WAITCNT"}


def test_add_pc_sampling_data_separates_shared_code_object_ids_across_pids(
    db_session,
):
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        (process_id, kernel_name): orm.Kernel(
            kernel_name=kernel_name,
            workload=workload,
        )
        for process_id in (42, 99)
        for kernel_name in ("vecCopy", "vecAdd")
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    first_tool_data = make_pc_sampling_tool_data()
    second_tool_data = copy.deepcopy(first_tool_data)
    second_tool_data["metadata"]["pid"] = 99
    second_tool_data["code_objects"][0]["load_base"] = 0x3000

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [first_tool_data, second_tool_data]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path, workload, kernel_objs
    )
    db_session.commit()

    assert set(code_object_stores) == {(42, 5), (99, 5)}
    first_store = code_object_stores[(42, 5)]
    second_store = code_object_stores[(99, 5)]
    assert first_store is not second_store
    assert first_store.code_object_uuid != second_store.code_object_uuid
    assert {line.kernel for line in first_store.instruction_lines} == {
        kernel_objs[(42, "vecCopy")],
        kernel_objs[(42, "vecAdd")],
    }
    assert {line.kernel for line in second_store.instruction_lines} == {
        kernel_objs[(99, "vecCopy")],
        kernel_objs[(99, "vecAdd")],
    }
    total_vec_copy_samples = sum(
        line.pc_sample_state.total_count
        for line in db_session.query(orm.InstructionLine).all()
        if line.kernel.kernel_name == "vecCopy"
    )
    assert total_vec_copy_samples == 2


def test_run_analysis_scopes_pc_sampling_uuids_by_process(db_session):
    workload_path = "/fake/workload"
    tool_data_records = [
        make_colliding_pc_sampling_tool_data(42, 1),
        make_colliding_pc_sampling_tool_data(99, 2),
    ]
    tool_data_records[0]["buffer_records"]["kernel_dispatch"].append(
        make_pc_sampling_dispatch(1, 100)
    )
    workload = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
    )
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {workload_path: workload}
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(tool_data_records)
    }
    analyzer._roofline_data_per_kernel = {}
    analyzer._roofline_data_per_workload = {}

    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(orm.Database, "create_views"))
        patch_stack.enter_context(patch.object(orm.Database, "write"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.run_analysis()

    dispatches = (
        db_session
        .query(orm.Dispatch)
        .order_by(orm.Dispatch.pid, orm.Dispatch.dispatch_id)
        .all()
    )
    assert [
        (dispatch.pid, dispatch.dispatch_id, dispatch.kernel.kernel_name)
        for dispatch in dispatches
    ] == [
        (42, 0, "vecCopy"),
        (42, 1, "vecCopy"),
        (99, 0, "vecCopy"),
    ]
    kernel_uuids_by_process_id = {
        process_id: {
            dispatch.kernel_uuid
            for dispatch in dispatches
            if dispatch.pid == process_id
        }
        for process_id in (42, 99)
    }
    assert {
        process_id: len(kernel_uuids)
        for process_id, kernel_uuids in kernel_uuids_by_process_id.items()
    } == {42: 1, 99: 1}
    assert kernel_uuids_by_process_id[42].isdisjoint(kernel_uuids_by_process_id[99])
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 3

    code_object_stores = db_session.query(orm.CodeObjectStore).all()
    assert [store.code_object_id for store in code_object_stores] == [5, 5]
    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.code_object_uuid for line in instruction_lines}) == 2
    assert {
        (line.code_object_offset, line.instruction, line.comment)
        for line in instruction_lines
    } == {(0x10, "v_mov", "/s/shared.cpp:7")}

    process_id_by_kernel_uuid = {
        dispatch.kernel_uuid: dispatch.pid for dispatch in dispatches
    }
    instruction_by_process_id = {
        process_id_by_kernel_uuid[line.kernel_uuid]: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    for process_id, instruction_line in instruction_by_process_id.items():
        assert instruction_line.kernel_uuid in kernel_uuids_by_process_id[process_id]
        assert {
            kernel_dispatch.pid
            for kernel_dispatch in instruction_line.kernel.dispatches
        } == {process_id}
        assert instruction_line.code_object_store.code_object_id == 5


def test_run_analysis_materialized_views_keep_pid_scoped_pc_sampling_origins(
    db_session,
):
    workload_path = "/fake/workload"
    analyzer = make_pc_sampling_only_database_analyzer(
        workload_path,
        [
            make_colliding_pc_sampling_tool_data(42, 1),
            make_colliding_pc_sampling_tool_data(99, 2),
        ],
    )

    run_analysis_with_materialized_views(analyzer)

    dispatches = db_session.query(orm.Dispatch).order_by(orm.Dispatch.pid).all()
    assert [
        (
            dispatch.pid,
            dispatch.dispatch_id,
            dispatch.kernel.kernel_name,
        )
        for dispatch in dispatches
    ] == [
        (42, 0, "vecCopy"),
        (99, 0, "vecCopy"),
    ]
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 2
    assert len({dispatch.kernel_uuid for dispatch in dispatches}) == 2

    code_object_stores = db_session.query(orm.CodeObjectStore).all()
    assert len(code_object_stores) == 2
    assert {store.code_object_id for store in code_object_stores} == {5}
    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.code_object_uuid for line in instruction_lines}) == 2
    assert {line.code_object_offset for line in instruction_lines} == {0x10}
    assert {line.instruction for line in instruction_lines} == {"v_mov"}
    assert {line.comment for line in instruction_lines} == {"/s/shared.cpp:7"}

    process_id_by_kernel_uuid = {
        dispatch.kernel_uuid: dispatch.pid for dispatch in dispatches
    }
    instruction_by_process_id = {
        process_id_by_kernel_uuid[line.kernel_uuid]: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    for process_id, instruction_line in instruction_by_process_id.items():
        dispatch = next(
            dispatch for dispatch in dispatches if dispatch.pid == process_id
        )
        assert instruction_line.kernel_uuid == dispatch.kernel_uuid
        assert (
            instruction_line.code_object_uuid
            == instruction_line.code_object_store.code_object_uuid
        )
        assert instruction_line.code_object_store.code_object_id == 5

    sample_states = db_session.query(orm.PCSampleState).all()
    stall_reasons = db_session.query(orm.PCSampleStallReason).all()
    instruction_samples = db_session.query(orm.InstructionSample).all()
    assert len(sample_states) == 2
    assert {state.total_count for state in sample_states} == {1, 2}
    assert len(stall_reasons) == 2
    assert {reason.count for reason in stall_reasons} == {1, 2}
    assert len(instruction_samples) == 2
    assert {sample.count for sample in instruction_samples} == {1, 2}

    orphan_queries = {
        "instruction_line_to_kernel": """
            SELECT COUNT(*)
            FROM compute_instruction_line AS instruction_line
            LEFT JOIN compute_kernel AS kernel
                ON instruction_line.kernel_uuid = kernel.kernel_uuid
            WHERE kernel.kernel_uuid IS NULL
        """,
        "instruction_line_to_code_object": """
            SELECT COUNT(*)
            FROM compute_instruction_line AS instruction_line
            LEFT JOIN compute_code_object_store AS code_object
                ON instruction_line.code_object_uuid = code_object.code_object_uuid
            WHERE code_object.code_object_uuid IS NULL
        """,
        "sample_state_to_instruction_line": """
            SELECT COUNT(*)
            FROM compute_pc_sample_state AS sample_state
            LEFT JOIN compute_instruction_line AS instruction_line
                ON sample_state.instruction_uuid = instruction_line.instruction_uuid
            WHERE instruction_line.instruction_uuid IS NULL
        """,
        "stall_reason_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_pc_sample_stall_reason AS stall_reason
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON stall_reason.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_pc_sample_stall_reason_lookup AS stall_lookup
                ON stall_reason.pc_sample_stall_reason_lookup_uuid =
                    stall_lookup.pc_sample_stall_reason_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR stall_lookup.pc_sample_stall_reason_lookup_uuid IS NULL
        """,
        "instruction_sample_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_instruction_sample AS instruction_sample
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON instruction_sample.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_instruction_sample_lookup AS sample_lookup
                ON instruction_sample.instruction_sample_lookup_uuid =
                    sample_lookup.instruction_sample_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR sample_lookup.instruction_sample_lookup_uuid IS NULL
        """,
    }
    orphan_counts = {
        relationship: db_session.execute(text(query)).scalar_one()
        for relationship, query in orphan_queries.items()
    }
    assert orphan_counts == dict.fromkeys(orphan_queries, 0)

    pc_sampling_rows = (
        db_session
        .execute(
            text(
                "SELECT kernel_uuid, kernel_name, count "
                "FROM compute_pc_sampling_view ORDER BY kernel_uuid"
            )
        )
        .mappings()
        .all()
    )
    assert [row["kernel_name"] for row in pc_sampling_rows] == [
        "vecCopy",
        "vecCopy",
    ]
    pc_sampling_mapping = {row["kernel_uuid"]: row["count"] for row in pc_sampling_rows}
    expected_pc_sampling_mapping = {
        line.kernel_uuid: line.pc_sample_state.total_count for line in instruction_lines
    }
    assert pc_sampling_mapping == expected_pc_sampling_mapping
    assert sum(pc_sampling_mapping.values()) == 3

    kernel_rows = (
        db_session
        .execute(
            text(
                "SELECT kernel_uuid, kernel_name, dispatch_count "
                "FROM compute_kernel_view ORDER BY kernel_uuid"
            )
        )
        .mappings()
        .all()
    )
    assert [row["kernel_name"] for row in kernel_rows] == [
        "vecCopy",
        "vecCopy",
    ]
    kernel_mapping = {row["kernel_uuid"]: row["dispatch_count"] for row in kernel_rows}
    assert kernel_mapping == dict.fromkeys(expected_pc_sampling_mapping, 1)


def test_run_analysis_exports_existing_csv_surface_for_pid_scoped_pc_sampling(
    tmp_path,
):
    try:
        orm.Database.init(":memory:")
        database_session = orm.Database.get_session()
        assert database_session is not None
        analyzer = make_pc_sampling_only_database_analyzer(
            str(tmp_path),
            [
                make_colliding_pc_sampling_tool_data(42, 1),
                make_colliding_pc_sampling_tool_data(99, 2),
            ],
        )

        run_analysis_with_materialized_views(analyzer)

        expected_pc_sampling_mapping = {
            kernel_uuid: count
            for kernel_uuid, count in database_session.execute(
                text(
                    "SELECT kernel_uuid, count "
                    "FROM compute_pc_sampling_view ORDER BY kernel_uuid"
                )
            )
        }
        expected_kernel_mapping = {
            kernel_uuid: dispatch_count
            for kernel_uuid, dispatch_count in database_session.execute(
                text(
                    "SELECT kernel_uuid, dispatch_count "
                    "FROM compute_kernel_view ORDER BY kernel_uuid"
                )
            )
        }

        csv_directory = tmp_path / "csv"
        orm.Database.write_csv_dir(csv_directory)

        output_filenames = {path.name for path in csv_directory.iterdir()}
        assert output_filenames == {
            "kernel.csv",
            "kernel_metric.csv",
            "workload_metric.csv",
            "pc_sampling.csv",
        }
        assert "dispatch.csv" not in output_filenames

        pc_sampling_frame = pd.read_csv(csv_directory / "pc_sampling.csv")
        kernel_frame = pd.read_csv(csv_directory / "kernel.csv")
        assert "pid" not in pc_sampling_frame.columns
        assert "pid" not in kernel_frame.columns
        assert list(pc_sampling_frame["kernel_name"]) == ["vecCopy", "vecCopy"]
        assert list(kernel_frame["kernel_name"]) == ["vecCopy", "vecCopy"]

        pc_sampling_csv_mapping = {
            row.kernel_uuid: row.count
            for row in pc_sampling_frame.itertuples(index=False)
        }
        assert len(pc_sampling_csv_mapping) == 2
        assert set(pc_sampling_csv_mapping) == set(expected_pc_sampling_mapping)
        assert set(pc_sampling_csv_mapping.values()) == {1, 2}
        assert sum(pc_sampling_csv_mapping.values()) == 3
        assert pc_sampling_csv_mapping == expected_pc_sampling_mapping

        kernel_csv_mapping = {
            row.kernel_uuid: row.dispatch_count
            for row in kernel_frame.itertuples(index=False)
        }
        assert len(kernel_csv_mapping) == 2
        assert set(kernel_csv_mapping) == set(expected_pc_sampling_mapping)
        assert set(kernel_csv_mapping.values()) == {1}
        assert kernel_csv_mapping == expected_kernel_mapping
    finally:
        current_session = orm.Database._session
        if current_session is not None:
            current_session.close()
        current_engine = orm.Database._engine
        if current_engine is not None:
            current_engine.dispose()
        orm.Database._session = None
        orm.Database._engine = None


def test_run_analysis_keeps_mixed_counter_and_pc_sampling_ownership(
    db_session,
    tmp_path,
):
    workload_path = str(tmp_path)
    first_tool_data = make_colliding_pc_sampling_tool_data(42, 1)
    second_tool_data = make_colliding_pc_sampling_tool_data(99, 2)
    second_tool_data["code_objects"][0]["load_base"] = 0x3000
    tool_data_records = [first_tool_data, second_tool_data]

    process_id_by_load_base = {0x1000: 42, 0x3000: 99}
    sample_count_by_process_id = {42: 1, 99: 2}
    for load_base, process_id in process_id_by_load_base.items():
        (tmp_path / f"{process_id}_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "retained ISA",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        tool_data_records,
    )
    analyzer._roofline_data_per_kernel = {
        workload_path: pd.DataFrame([{"kernel_name": "vecCopy", "total_flops": 64.0}])
    }
    analyzer._metrics_info_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "name": "Counter metric",
                "metric_id": "1.1",
                "description": "Counter-derived value",
                "unit": "cycles",
                "table_name": "Counter",
                "sub_table_name": "Counter values",
            }
        ])
    }
    analyzer._kernel_values_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "metric_id": "1.1",
                "kernel_name": "vecCopy",
                "value_name": "avg",
                "value": 12.5,
            }
        ])
    }

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()
    aggregate_kernel = dispatch.kernel
    assert dispatch.pid is None
    assert dispatch.dispatch_id == 7
    assert aggregate_kernel.kernel_name == "vecCopy"

    metric_value = db_session.query(orm.KernelMetricValue).one()
    roofline_data = db_session.query(orm.KernelRooflineData).one()
    assert metric_value.kernel_uuid == aggregate_kernel.kernel_uuid
    assert metric_value.value == 12.5
    assert roofline_data.kernel_uuid == aggregate_kernel.kernel_uuid
    assert roofline_data.total_flops == 64.0

    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 3
    assert {kernel.kernel_name for kernel in kernels} == {"vecCopy"}

    stores = (
        db_session
        .query(orm.CodeObjectStore)
        .order_by(orm.CodeObjectStore.load_base)
        .all()
    )
    assert [(store.code_object_id, store.load_base) for store in stores] == [
        (5, 0x1000),
        (5, 0x3000),
    ]

    chain_by_process_id = {}
    for store in stores:
        instruction_by_offset = {
            line.code_object_offset: line for line in store.instruction_lines
        }
        assert set(instruction_by_offset) == {0x10, 0x30}

        sampled_line = instruction_by_offset[0x10]
        isa_line = instruction_by_offset[0x30]
        assert sampled_line.kernel_uuid == isa_line.kernel_uuid
        assert isa_line.pc_sample_state is None

        process_id = process_id_by_load_base[store.load_base]
        assert (
            sampled_line.pc_sample_state.total_count
            == sample_count_by_process_id[process_id]
        )
        chain_by_process_id[process_id] = (
            sampled_line.kernel_uuid,
            store.code_object_uuid,
            sampled_line.instruction_uuid,
            isa_line.instruction_uuid,
        )

    assert set(chain_by_process_id) == {42, 99}
    (
        kernel_uuid_chain,
        code_object_uuid_chain,
        sampled_instruction_uuid_chain,
        isa_instruction_uuid_chain,
    ) = zip(*chain_by_process_id.values())
    assert len(set(kernel_uuid_chain)) == 2
    assert len(set(code_object_uuid_chain)) == 2
    assert len(set(sampled_instruction_uuid_chain)) == 2
    assert len(set(isa_instruction_uuid_chain)) == 2

    pid_kernel_uuids = set(kernel_uuid_chain)
    assert aggregate_kernel.kernel_uuid not in pid_kernel_uuids
    assert {kernel.kernel_uuid for kernel in kernels} == {
        aggregate_kernel.kernel_uuid,
        *pid_kernel_uuids,
    }
    assert aggregate_kernel.instruction_lines == []

    pid_kernels = [
        kernel for kernel in kernels if kernel.kernel_uuid in pid_kernel_uuids
    ]
    for kernel in pid_kernels:
        assert kernel.dispatches == []
        assert kernel.metric_values == []
        assert kernel.roofline_data_points == []

    sample_states = db_session.query(orm.PCSampleState).all()
    assert len(sample_states) == 2
    assert all(
        state.instruction_line.kernel_uuid in pid_kernel_uuids
        for state in sample_states
    )
    assert db_session.query(orm.PCSampleStallReason).count() == 2
    assert db_session.query(orm.InstructionSample).count() == 2
    assert db_session.query(orm.InstructionLine).count() == 4


def test_run_analysis_does_not_register_filtered_pc_sampling_symbols(
    db_session,
    tmp_path,
):
    workload_path = str(tmp_path)
    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        [make_pc_sampling_tool_data()],
    )

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()
    assert dispatch.pid is None

    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 2
    assert {kernel.kernel_name for kernel in kernels} == {"vecCopy"}

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in instruction_lines] == [0x10]

    sampled_line = instruction_lines[0]
    assert sampled_line.kernel_uuid != dispatch.kernel_uuid
    assert sampled_line.pc_sample_state.total_count == 1
    assert db_session.query(orm.PCSampleState).count() == 1
    assert db_session.query(orm.PCSampleStallReason).count() == 1
    assert db_session.query(orm.InstructionSample).count() == 1


# =============================================================================
# Code-object ISA ingestion (add_code_object_isa)
# =============================================================================


def make_disasm_code_object(code_object_id, instructions, symbol_name="sym"):
    """Build one code_obj_info code object with a single symbol."""
    return {
        "id": code_object_id,
        "symbols": [{"name": symbol_name, "instructions": instructions}],
    }


def test_add_code_object_isa_adds_unsampled_lines(db_session):
    """Un-sampled instructions of a dispatched kernel are added and attributed
    via the mangled-name join; a disassembly offset that matches an already-
    sampled offset inserts no duplicate row; ISA of an un-dispatched symbol is
    not stored at all; a non-surviving pid's disassembly is skipped."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    load_base = 0x1000
    try:
        # 0x1010 -> offset 0x10 is already sampled; 0x1030 -> offset 0x30 is new
        # and joins vecCopy through its mangled ELF name; 0x1040 -> offset 0x40
        # belongs to a symbol never dispatched, so it is dropped entirely.
        code_objects = [
            {
                "id": 5,
                "symbols": [
                    {
                        "name": "_Z7vecCopyv",
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x10,
                                "name": "v_mov",
                                "comment": "",
                            },
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "c",
                            },
                        ],
                    },
                    {
                        "name": "_Z7unknownv",
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x40,
                                "name": "s_endpgm",
                                "comment": "",
                            }
                        ],
                    },
                ],
            }
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )
        # A non-surviving pid (metadata pid is 42) reuses code_object_id 5; its
        # 0x50 offset must never be stored since only pid 42's load_base is known.
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x50,
                                "name": "s_nop",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            (42, "vecCopy"): orm.Kernel(kernel_name="vecCopy", workload=workload),
            (42, "vecAdd"): orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [make_pc_sampling_tool_data()]
        }
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path, workload, kernel_objs
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
        )
        db_session.commit()

        lines = db_session.query(orm.InstructionLine).all()
        by_offset = {line.code_object_offset: line for line in lines}
        # Two sampled offsets + one new un-sampled offset; the un-dispatched
        # symbol's 0x40 line and the non-surviving pid's 0x50 line are dropped,
        # and 0x10 is not duplicated.
        assert set(by_offset) == {0x10, 0x20, 0x30}
        assert set(code_object_stores) == {(42, 5)}
        code_object_store = code_object_stores[(42, 5)]
        assert db_session.query(orm.CodeObjectStore).one() is code_object_store
        assert code_object_store is by_offset[0x10].code_object_store
        # The disassembly-only line joins its kernel and carries no sample state.
        isa_line = by_offset[0x30]
        assert isa_line.kernel.kernel_name == "vecCopy"
        assert isa_line.pc_sample_state is None
        assert isa_line.instruction == "s_nop"
        # The sampled line at 0x10 kept its kernel attribution and sample state.
        assert by_offset[0x10].kernel.kernel_name == "vecCopy"
        assert by_offset[0x10].pc_sample_state is not None
        # Both belong to the same (reused) code object store.
        assert isa_line.code_object_store is code_object_store
        assert {line.code_object_uuid for line in by_offset.values()} == {
            code_object_store.code_object_uuid
        }
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_unsampled_code_objects_by_process(db_session):
    """Matching ISA-only code objects retain distinct process-local UUID chains."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["pc_sample_stochastic"] = []
        # code object 9 has a load_base and a dispatched kernel, but no samples.
        first_tool_data["code_objects"].append({
            "code_object_id": 9,
            "load_base": 0x2000,
        })
        first_tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        first_tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][-1]["load_base"] = 0x4000

        first_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x2000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        second_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x4000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": first_code_objects}), encoding="utf-8"
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({"code_objects": second_code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            (process_id, "helper"): orm.Kernel(
                kernel_name="helper",
                workload=workload,
            )
            for process_id in (42, 99)
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path, workload, kernel_objs
        )
        assert code_object_stores == {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 9), (99, 9)}
        first_store = code_object_stores[(42, 9)]
        second_store = code_object_stores[(99, 9)]
        assert first_store.load_base == 0x2000
        assert second_store.load_base == 0x4000
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = first_store.instruction_lines[0]
        second_line = second_store.instruction_lines[0]
        assert first_line.code_object_offset == second_line.code_object_offset == 0x8
        assert first_line.instruction == second_line.instruction == "s_endpgm"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert first_line.kernel is kernel_objs[(42, "helper")]
        assert second_line.kernel is kernel_objs[(99, "helper")]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_skips_code_object_without_load_base(db_session):
    """A code object with no known load_base cannot be offset-mapped, so it is
    skipped rather than stored with an inconsistent offset."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        tool_data = make_pc_sampling_tool_data()
        tool_data["code_objects"].append({"code_object_id": 9, "load_base": None})
        tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x500, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        helper = orm.Kernel(kernel_name="helper", workload=workload)
        db_session.add(helper)
        kernel_objs = {(42, "helper"): helper}

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {workload_path: [tool_data]}
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path, workload, kernel_objs
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
        )
        db_session.commit()

        # The store exists (its kernel was dispatched) but no ISA line was added.
        store = db_session.query(orm.CodeObjectStore).filter_by(code_object_id=9).one()
        assert code_object_stores[(42, 9)] is store
        assert store.instruction_lines == []
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_duplicate_offsets_by_process(db_session):
    """One process's ISA offset cannot suppress the matching offset in another."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000

        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x1000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x3000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            (process_id, kernel_name): orm.Kernel(
                kernel_name=kernel_name,
                workload=workload,
            )
            for process_id in (42, 99)
            for kernel_name in ("vecCopy", "vecAdd")
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path, workload, kernel_objs
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 5), (99, 5)}
        first_store = code_object_stores[(42, 5)]
        second_store = code_object_stores[(99, 5)]
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = next(
            line
            for line in first_store.instruction_lines
            if line.code_object_offset == 0x30
        )
        second_line = next(
            line
            for line in second_store.instruction_lines
            if line.code_object_offset == 0x30
        )
        assert first_line.instruction == second_line.instruction == "s_shared"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert first_line.code_object_uuid != second_line.code_object_uuid
        assert first_line.kernel is kernel_objs[(42, "vecCopy")]
        assert second_line.kernel is kernel_objs[(99, "vecCopy")]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_requires_process_local_dispatch(db_session):
    """Store only symbols dispatched by the process that loaded the code object."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 100)
        ]

        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000
        second_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 101)
        ]

        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    {
                        "id": 5,
                        "symbols": [
                            {
                                "name": "_Z7vecCopyv",
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x30,
                                        "name": "not_dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                            {
                                "name": "vecAdd",
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x40,
                                        "name": "dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            (99, "vecCopy"): orm.Kernel(kernel_name="vecCopy", workload=workload),
            (99, "vecAdd"): orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        code_object_stores = {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
        )
        db_session.commit()

        assert set(code_object_stores) == {(99, 5)}
        second_store = code_object_stores[(99, 5)]
        assert second_store.code_object_uuid is not None
        stored_isa = {
            (line.instruction, line.kernel.kernel_name)
            for line in second_store.instruction_lines
        }
        assert stored_isa == {("dispatched", "vecAdd")}
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_pc_sampling_data_drops_lines_without_kernel(db_session):
    """Lines whose kernel is absent from kernel_objs (filtered out) are dropped
    along with their sample state and child counts, not attributed to no kernel."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    # Only vecCopy survives filtering; vecAdd's line must be dropped.
    kernel_objs = {
        (42, "vecCopy"): orm.Kernel(kernel_name="vecCopy", workload=workload)
    }
    db_session.add(kernel_objs[(42, "vecCopy")])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    analyzer.add_pc_sampling_data(workload_path, workload, kernel_objs)
    db_session.commit()

    lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in lines] == [0x10]
    retained_line = lines[0]
    assert retained_line.kernel is kernel_objs[(42, "vecCopy")]

    retained_sample_state = db_session.query(orm.PCSampleState).one()
    assert retained_line.pc_sample_state is retained_sample_state

    stall_reason_count = db_session.query(orm.PCSampleStallReason).one()
    assert stall_reason_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.stall_reasons == [stall_reason_count]
    assert stall_reason_count.stall_reason_lookup.text == "WAITCNT"
    assert stall_reason_count.count == 1

    instruction_type_count = db_session.query(orm.InstructionSample).one()
    assert instruction_type_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.instruction_samples == [instruction_type_count]
    assert instruction_type_count.instruction_sample_lookup.text == "VALU"
    assert instruction_type_count.count == 1
