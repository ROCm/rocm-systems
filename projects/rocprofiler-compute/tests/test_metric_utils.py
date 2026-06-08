# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.* modules."""

import ast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.aggregation import (
    calc_pct_of_peak,
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
from utils.metrics.evaluation_pipeline import (
    compute_pct_of_peak,
    eval_metric,
    validate_dual_issue_metrics,
)
from utils.metrics.expression import (
    CodeTransformer,
    build_eval_string,
    gen_counter_list,
    update_denominator_string,
    update_normal_unit_string,
)
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
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


# =============================================================================
# Tests for utils.metrics.expression
# =============================================================================


class TestExpression:
    """Tests for utils.metrics.expression."""

    def test_build_eval_string_returns_empty_for_empty_equation(self):
        """build_eval_string returns the empty string when given an empty equation."""
        assert build_eval_string("") == ""

    def test_update_denominator_string_returns_empty_for_empty_equation(self):
        """update_denominator_string returns the empty string when input is empty."""
        assert update_denominator_string("", "per_wave") == ""

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
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
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
        dfs_expressions = {
            1: [
                v
                for v in metric_fields.values()
                if isinstance(v, str) and v and v != "None"
            ]
        }
        sys_info = pd.Series({
            "ip_blocks": "standard",
            "gpu_arch": "gfx90a",
            "se_per_gpu": 4,
            "sa_per_se": 2,
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
        raw_pmc_df = pd.DataFrame({
            "SQ_WAVES": [100, 200, 150],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })
        return metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df

    def test_eval_metric_in_debug_mode(self):
        """eval_metric with debug=True invokes debug_row_tracker and writes back."""
        fixture = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
            }
        )
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        with (
            patch(
                "utils.metrics.evaluation_pipeline.get_build_in_vars",
                return_value={},
            ),
            patch(
                "utils.metrics.evaluation_pipeline.debug_row_tracker"
            ) as mock_debug_row_tracker,
        ):
            eval_metric(
                dfs,
                dfs_type,
                dfs_expressions,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=True,
            )

        mock_debug_row_tracker.assert_called_once()
        call_args = mock_debug_row_tracker.call_args
        assert call_args.args[0] == "Value"
        assert call_args.kwargs["show_inputs"] is True
        assert metric_df.loc["1.1.0", "Value"] == 450

    def test_eval_metric_computes_value_from_expression(self):
        """eval_metric writes the computed Value back to the metric DataFrame."""
        fixture = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
            }
        )
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        with patch(
            "utils.metrics.evaluation_pipeline.get_build_in_vars", return_value={}
        ):
            eval_metric(
                dfs,
                dfs_type,
                dfs_expressions,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Value"] == 450

    def test_eval_metric_resolves_accum_alias_column_end_to_end(self):
        """eval_metric resolves YAML formulas that reference ACCUM alias columns."""
        fixture = self._build_eval_metric_inputs(
            metric_fields={
                "Value": (
                    "to_sum(raw_pmc_df['SQ_INST_LEVEL_VMEM_ACCUM']) / "
                    "to_sum(raw_pmc_df['SQ_INSTS_VMEM'])"
                ),
            }
        )
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        flat_raw_pmc_df = pd.DataFrame({
            "SQ_INST_LEVEL_VMEM_ACCUM": [100.0, 200.0, 300.0],
            "SQ_INSTS_VMEM": [10.0, 20.0, 30.0],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })
        with patch(
            "utils.metrics.evaluation_pipeline.get_build_in_vars", return_value={}
        ):
            eval_metric(
                dfs,
                dfs_type,
                dfs_expressions,
                sys_info,
                pd.DataFrame(),
                flat_raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Value"] == 10.0

    def test_eval_metric_normalizes_falsey_average_to_empty_string(self):
        """eval_metric replaces a falsey Average value with the empty string."""
        fixture = self._build_eval_metric_inputs(metric_fields={"Average": None})
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        assert metric_df.loc["1.1.0", "Average"] is None

        with patch(
            "utils.metrics.evaluation_pipeline.get_build_in_vars", return_value={}
        ):
            eval_metric(
                dfs,
                dfs_type,
                dfs_expressions,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Average"] == ""

    def test_eval_metric_noise_clamp(self):
        """eval_metric emits per-metric variance warning + summary on clamp."""
        # Negative DIFF over a positive REF crosses the 1% threshold and bumps
        # the noise-clamp counter when to_noise_clamp evaluates the expression.
        fixture = self._build_eval_metric_inputs(
            metric_fields={
                "Value": (
                    "to_noise_clamp("
                    "to_min(raw_pmc_df['DIFF']), "
                    "to_max(raw_pmc_df['REF']))"
                ),
            }
        )
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        raw_pmc_df = pd.DataFrame({
            "GRBM_GUI_ACTIVE": [1000],
            "DIFF": [-100.0],
            "REF": [1000.0],
        })

        clear_noise_clamp_warnings()
        with (
            patch(
                "utils.metrics.evaluation_pipeline.get_build_in_vars", return_value={}
            ),
            patch(
                "utils.metrics.evaluation_pipeline.console_warning"
            ) as mock_console_warning,
            patch(
                "utils.metrics.evaluation_pipeline.print_noise_clamp_summary"
            ) as mock_print_summary,
        ):
            eval_metric(
                dfs,
                dfs_type,
                dfs_expressions,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
            )

        assert get_noise_clamp_warnings()["count"] >= 1
        variance_calls = [
            call_args
            for call_args in mock_console_warning.call_args_list
            if "Variance corrected for metric:" in call_args.args[0]
        ]
        assert len(variance_calls) == 1
        assert "Test Metric" in variance_calls[0].args[0]
        mock_print_summary.assert_called_once()

    def make_dual_issue_dfs(
        self, metric_name: str, value: float, peak: float, peak_col: str = "Peak"
    ):
        """Build the (dfs, dfs_type) fixture used by dual-issue tests."""
        df = pd.DataFrame({
            "Metric": [metric_name],
            "Value": [value],
            peak_col: [peak],
        })
        return {1: df}, {1: "metric_table"}

    def test_validate_dual_issue_metrics_emits_valu_utilization_warning(self):
        """VALU Utilization above peak triggers the dual-issue warning."""
        dfs, dfs_type = self.make_dual_issue_dfs(
            "VALU Utilization", value=150.0, peak=100.0
        )
        sys_info = pd.Series({"gpu_arch": "gfx942"})

        with patch("utils.metrics.common.console_warning") as mock_warning:
            validate_dual_issue_metrics(
                dfs, dfs_type, sys_info, raw_pmc_df=pd.DataFrame()
            )

        mock_warning.assert_called_once()
        msg = mock_warning.call_args.args[0]
        assert "VALU Utilization can go up to 200%" in msg
        assert "SQ_ACTIVE_INST_VALU2" not in msg

    def test_validate_dual_issue_metrics_emits_valu_flops_warning(self):
        """VALU FLOPs (F64) above peak triggers the FLOPs-flavored warning."""
        dfs, dfs_type = self.make_dual_issue_dfs(
            "VALU FLOPs (F64)", value=600.0, peak=400.0
        )
        sys_info = pd.Series({"gpu_arch": "gfx942"})

        with patch("utils.metrics.common.console_warning") as mock_warning:
            validate_dual_issue_metrics(
                dfs, dfs_type, sys_info, raw_pmc_df=pd.DataFrame()
            )

        msg = mock_warning.call_args.args[0]
        assert "VALU FLOPs can exceed the peak value" in msg

    def test_validate_dual_issue_metrics_silent_below_peak(self):
        """Below-peak VALU Utilization stays silent."""
        dfs, dfs_type = self.make_dual_issue_dfs(
            "VALU Utilization", value=80.0, peak=100.0
        )
        sys_info = pd.Series({"gpu_arch": "gfx942"})

        with patch("utils.metrics.common.console_warning") as mock_warning:
            validate_dual_issue_metrics(
                dfs, dfs_type, sys_info, raw_pmc_df=pd.DataFrame()
            )

        mock_warning.assert_not_called()

    def test_validate_dual_issue_metrics_appends_valu2_suffix_on_gfx950(self):
        """gfx950 with non-zero SQ_ACTIVE_INST_VALU2 appends the confirmation."""
        dfs, dfs_type = self.make_dual_issue_dfs(
            "VALU Utilization", value=150.0, peak=100.0
        )
        sys_info = pd.Series({"gpu_arch": "gfx950"})
        raw_pmc_df = pd.DataFrame({"SQ_ACTIVE_INST_VALU2": [1, 2, 3]})

        with patch("utils.metrics.common.console_warning") as mock_warning:
            validate_dual_issue_metrics(dfs, dfs_type, sys_info, raw_pmc_df)

        msg = mock_warning.call_args.args[0]
        assert "Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter" in msg

    def test_validate_dual_issue_metrics_uses_peak_empirical_fallback(self):
        """Peak (Empirical) column is used when present alongside Value."""
        dfs, dfs_type = self.make_dual_issue_dfs(
            "VALU Utilization",
            value=150.0,
            peak=100.0,
            peak_col="Peak (Empirical)",
        )
        sys_info = pd.Series({"gpu_arch": "gfx942"})

        with patch("utils.metrics.common.console_warning") as mock_warning:
            validate_dual_issue_metrics(
                dfs, dfs_type, sys_info, raw_pmc_df=pd.DataFrame()
            )

        mock_warning.assert_called_once()
        msg = mock_warning.call_args.args[0]
        assert "VALU Utilization can go up to 200%" in msg

    def make_pct_of_peak_dfs(
        self,
        pct_of_peak_flags: list,
        avg_values: list,
        peak_values: list,
        value_col: str = "Avg",
        peak_col: str = "Peak",
    ):
        """Build (dfs, dfs_type) fixture for compute_pct_of_peak tests."""
        df = pd.DataFrame({
            "Metric": [f"M{i}" for i in range(len(pct_of_peak_flags))],
            value_col: avg_values,
            peak_col: peak_values,
            "Percent of Peak": pct_of_peak_flags,
        })
        return {1: df}, {1: "metric_table"}

    def test_compute_pct_of_peak_true_writes_correct_value(self):
        """A pct_of_peak=True row writes 100 * value / peak into Percent of Peak."""
        dfs, dfs_type = self.make_pct_of_peak_dfs(
            pct_of_peak_flags=[True], avg_values=[50.0], peak_values=[200.0]
        )
        compute_pct_of_peak(dfs, dfs_type)
        assert dfs[1].loc[0, "Percent of Peak"] == pytest.approx(25.0)

    def test_compute_pct_of_peak_false_writes_empty_string(self):
        """A pct_of_peak=False row gets an empty string in Percent of Peak."""
        dfs, dfs_type = self.make_pct_of_peak_dfs(
            pct_of_peak_flags=[False], avg_values=[50.0], peak_values=[200.0]
        )
        compute_pct_of_peak(dfs, dfs_type)
        assert dfs[1].loc[0, "Percent of Peak"] == ""

    def test_compute_pct_of_peak_zero_peak_writes_empty_string(self):
        """A pct_of_peak=True row with zero peak gets an empty
        string (division undefined)."""
        dfs, dfs_type = self.make_pct_of_peak_dfs(
            pct_of_peak_flags=[True], avg_values=[50.0], peak_values=[0.0]
        )
        compute_pct_of_peak(dfs, dfs_type)
        assert dfs[1].loc[0, "Percent of Peak"] == ""

    def test_compute_pct_of_peak_skips_df_without_pct_of_peak_column(self):
        """A metric_table with no Percent of Peak column is left untouched."""
        df = pd.DataFrame({"Metric": ["M1"], "Avg": [50.0], "Peak": [200.0]})
        dfs, dfs_type = {1: df}, {1: "metric_table"}
        compute_pct_of_peak(dfs, dfs_type)
        assert "Percent of Peak" not in dfs[1].columns

    def test_compute_pct_of_peak_skips_non_metric_table(self):
        """A non-metric_table df is skipped even if it has a Percent of Peak column."""
        df = pd.DataFrame({"Percent of Peak": [True], "Avg": [50.0], "Peak": [200.0]})
        dfs, dfs_type = {1: df}, {1: "raw_csv_table"}
        compute_pct_of_peak(dfs, dfs_type)
        assert bool(dfs[1].loc[0, "Percent of Peak"]) is True

    def test_compute_pct_of_peak_prefers_avg_over_value_column(self):
        """Avg is preferred over Value when both columns are present."""
        df = pd.DataFrame({
            "Metric": ["M1"],
            "Avg": [50.0],
            "Value": [999.0],
            "Peak": [200.0],
            "Percent of Peak": [True],
        })
        dfs, dfs_type = {1: df}, {1: "metric_table"}
        compute_pct_of_peak(dfs, dfs_type)
        assert dfs[1].loc[0, "Percent of Peak"] == pytest.approx(25.0)

    def test_compute_pct_of_peak_prefers_peak_over_peak_empirical(self):
        """When both Peak and Peak (Empirical) are present, Peak is used."""
        df = pd.DataFrame({
            "Metric": ["M1"],
            "Avg": [50.0],
            "Peak": [200.0],
            "Peak (Empirical)": [500.0],
            "Percent of Peak": [True],
        })
        dfs, dfs_type = {1: df}, {1: "metric_table"}
        compute_pct_of_peak(dfs, dfs_type)
        assert dfs[1].loc[0, "Percent of Peak"] == pytest.approx(25.0)

    def test_compute_pct_of_peak_skips_when_no_value_column(self):
        """A metric_table with no recognised value column is left untouched."""
        df = pd.DataFrame({
            "Metric": ["M1"],
            "Peak": [200.0],
            "Percent of Peak": [True],
        })
        dfs, dfs_type = {1: df}, {1: "metric_table"}
        compute_pct_of_peak(dfs, dfs_type)
        assert bool(dfs[1].loc[0, "Percent of Peak"]) is True


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
            with (
                patch("utils.metrics.metric_evaluator.console_warning") as mock_warning,
                patch("utils.metrics.metric_evaluator.console_debug") as mock_debug,
            ):
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


# =============================================================================
# Tests for utils.utils_counter_defs.extract_counters_and_variables
# =============================================================================


class TestExtractCountersAndVariables:
    """Tests for utils.utils_counter_defs.extract_counters_and_variables."""

    def test_returns_hw_counters_and_referenced_builtin_vars(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        text = "$GRBM_GUI_ACTIVE_PER_XCD / SQ_WAVES"
        hw, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_GUI_ACTIVE" in hw
        assert "SQ_WAVES" in hw
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_resolves_builtin_var_dependencies_transitively(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        # numActiveCUs references $GRBM_GUI_ACTIVE_PER_XCD -> GRBM_GUI_ACTIVE
        text = "$numActiveCUs"
        hw, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_GUI_ACTIVE" in hw
        assert "numActiveCUs" in vars_
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_unreferenced_builtin_vars_are_not_returned(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        # SUPPORTED_DENOM["per_cycle"] pulls in $GRBM_GUI_ACTIVE_PER_XCD
        # unconditionally; unrelated built-in vars must not appear.
        text = "SQ_WAVES"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_COUNT_PER_XCD" not in vars_
        assert "GRBM_SPI_BUSY_PER_XCD" not in vars_
        assert "numActiveCUs" not in vars_

    def test_non_builtin_vars_dropped_from_variables_set(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        # $num_xcd is a sys var, not a built-in var; should not appear in vars_
        text = "GRBM_GUI_ACTIVE / $num_xcd"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "num_xcd" not in vars_

    def test_handles_ammolite_prefix(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        # After build_eval_string, $var becomes ammolite__var
        text = "(100 * ammolite__numActiveCUs) / ammolite__cu_per_gpu"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "numActiveCUs" in vars_

    def test_ignores_non_builtin_ammolite(self):
        from utils.utils_counter_defs import extract_counters_and_variables

        # ammolite__cu_per_gpu is a sys var, not a built-in var
        _, vars_ = extract_counters_and_variables("ammolite__cu_per_gpu", "MI200")
        assert "cu_per_gpu" not in vars_
