# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/evaluation_pipeline.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.evaluation_pipeline import (
    calc_builtin_vars,
    check_gui_active_counter,
    compute_pct_of_peak,
    create_empirical_peaks_dict,
    create_sys_vars,
    eval_metric,
    is_pct_of_peak_flagged,
    type_sys_vars,
    validate_dual_issue_metrics,
)
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
)

# =============================================================================
# Tests for utils.metrics.evaluation_pipeline
# =============================================================================


class TestEvaluationPipeline:
    """Tests for utils.metrics.evaluation_pipeline."""

    sys_info_fields = {
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
        "num_memory_channels": 4,
        "total_l2_chan": 32,
        "num_xcd": 1,
        "wave_size": 64,
    }

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
        sys_info = pd.Series(self.sys_info_fields)
        raw_pmc_df = pd.DataFrame({
            "SQ_WAVES": [100, 200, 150],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })
        return metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df

    def test_empirical_peaks_without_roofline_csv_are_nan(self):
        peaks = create_empirical_peaks_dict(pd.DataFrame())
        # Ceilings only the gfx9 and only the gfx11 benchmarks measure.
        for name in ("MFMAF8Flops", "BF16Flops"):
            assert np.isnan(peaks[f"ammolite__{name}_empirical_peak"])
        assert all(np.isnan(value) for value in peaks.values())

    def test_eval_metric_in_debug_mode(self):
        """eval_metric with debug=True invokes debug_row_tracker and writes back."""
        fixture = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
            }
        )
        metric_df, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        with patch(
            "utils.metrics.evaluation_pipeline.get_build_in_vars",
            return_value={},
        ), patch(
            "utils.metrics.evaluation_pipeline.debug_row_tracker"
        ) as mock_debug_row_tracker:
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
        with patch(
            "utils.metrics.evaluation_pipeline.get_build_in_vars", return_value={}
        ), patch(
            "utils.metrics.evaluation_pipeline.console_warning"
        ) as mock_console_warning, patch(
            "utils.metrics.evaluation_pipeline.print_noise_clamp_summary"
        ) as mock_print_summary:
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

    @pytest.mark.parametrize(
        "gui_active_counter",
        ["GRBM_GUI_ACTIVE_sum", "GRBM_GUI_ACTIVE"],
    )
    def test_eval_metric_rejects_zero_gui_active_counter(self, gui_active_counter):
        """The zero guard checks the available GUI-active counter by name."""
        fixture = self._build_eval_metric_inputs()
        _, dfs, dfs_type, dfs_expressions, sys_info, raw_pmc_df = fixture
        raw_pmc_df[gui_active_counter] = 0

        with patch(
            "utils.metrics.evaluation_pipeline.console_warning"
        ) as mock_warning, patch(
            "utils.metrics.evaluation_pipeline.console_error",
            side_effect=SystemExit,
        ) as mock_error:
            with pytest.raises(SystemExit):
                eval_metric(
                    dfs,
                    dfs_type,
                    dfs_expressions,
                    sys_info,
                    pd.DataFrame(),
                    raw_pmc_df,
                    debug=False,
                )

        mock_warning.assert_called_once_with(f"Detected {gui_active_counter} == 0")
        mock_error.assert_called_once_with("Halting execution for warning above.")

    def test_calc_builtin_vars_uses_gfx1250_gui_active_sum(self):
        """gfx1250 computes per-dispatch GUI-active cycles from the XCD sum."""
        raw_pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE_sum": [800, 1600]})
        sys_vars = {"ammolite__num_xcd": 8}

        builtin_vars = calc_builtin_vars(
            raw_pmc_df,
            sys_vars,
            "gfx1250",
            ["$GRBM_GUI_ACTIVE_PER_XCD"],
        )

        pd.testing.assert_series_equal(
            builtin_vars["ammolite__GRBM_GUI_ACTIVE_PER_XCD"],
            pd.Series([100.0, 200.0]),
            check_names=False,
        )

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

    def test_create_sys_vars_maps_required_and_optional_fields(self):
        """Required and present optional fields are prefixed and type-coerced."""
        result = create_sys_vars(pd.Series(self.sys_info_fields))
        assert result["ammolite__total_l2_chan"] == 32
        assert isinstance(result["ammolite__total_l2_chan"], int)
        assert result["ammolite__num_memory_channels"] == 4.0
        assert result["ammolite__num_xcd"] == 1

    @pytest.mark.parametrize(
        "override, missing_field",
        [
            ({"cu_per_gpu": np.nan}, "cu_per_gpu"),
            ({"total_l2_chan": 0}, "total_l2_chan"),
        ],
        ids=["nan", "zero"],
    )
    def test_create_sys_vars_warns_and_zeroes_missing_required(
        self, override, missing_field
    ):
        """Missing or zero required fields warn and fall back to 0."""
        sys_info = pd.Series({**self.sys_info_fields, **override})
        with patch(
            "utils.metrics.evaluation_pipeline.console_warning"
        ) as console_warning_mock:
            result = create_sys_vars(sys_info)
        assert result[f"ammolite__{missing_field}"] == 0
        assert any(
            missing_field in warning_call.args[0]
            for warning_call in console_warning_mock.call_args_list
        )

    def test_create_sys_vars_skips_absent_optional_fields(self):
        """Absent optional fields are omitted; num_xcd defaults to 1 for RDNA."""
        sys_info = pd.Series({
            key: value
            for key, value in self.sys_info_fields.items()
            if key not in {"num_memory_channels", "num_gl1c", "num_xcd"}
        })
        result = create_sys_vars(sys_info)
        assert "ammolite__num_memory_channels" not in result
        assert "ammolite__num_gl1c" not in result
        assert result["ammolite__num_xcd"] == 1

    def test_create_sys_vars_includes_num_gl1c_when_present(self):
        """num_gl1c is mapped when the RDNA sysinfo column is present."""
        sys_info = pd.Series({**self.sys_info_fields, "num_gl1c": 8})
        result = create_sys_vars(sys_info)
        assert result["ammolite__num_gl1c"] == 8


class TestSharedEvaluationHelpers:
    """Helpers the CLI and DB evaluation paths share (FR-12, FR-14, FR-16)."""

    @pytest.mark.parametrize(
        "flag, expected",
        [
            (True, True),
            ("True", True),
            (np.True_, True),
            (False, False),
            ("", False),
            (None, False),
            (np.nan, False),
        ],
        ids=["bool", "string", "numpy-bool", "false", "empty", "none", "nan"],
    )
    def test_is_pct_of_peak_flagged_reads_bool_and_stringified_flag(
        self, flag, expected
    ):
        """build_metric_value_string turns the flag into a string; both forms count."""
        assert is_pct_of_peak_flagged(flag) is expected

    def test_type_sys_vars_casts_specs_correction_strings(self, gfx942_sys_info):
        """A --specs-correction value arrives as a string and is cast to its type."""
        sys_info = {**gfx942_sys_info, "cu_per_gpu": "304", "cur_sclk": "2400"}
        typed = type_sys_vars(sys_info, missing_as_null=True)
        assert typed["cu_per_gpu"] == 304
        assert isinstance(typed["cu_per_gpu"], int)
        assert typed["cur_sclk"] == pytest.approx(2400.0)

    def test_type_sys_vars_missing_required_is_nan_with_warning(self, gfx942_sys_info):
        """Null mode leaves a missing required value NaN and warns once."""
        sys_info = {**gfx942_sys_info, "cu_per_gpu": np.nan}
        with patch(
            "utils.metrics.evaluation_pipeline.console_warning"
        ) as console_warning_mock:
            typed = type_sys_vars(sys_info, missing_as_null=True)
        assert np.isnan(typed["cu_per_gpu"])
        console_warning_mock.assert_called_once()
        assert "cu_per_gpu" in console_warning_mock.call_args.args[0]

    def test_type_sys_vars_non_numeric_string_counts_as_missing(self, gfx942_sys_info):
        """A value that is not a number is treated as missing, not raised."""
        sys_info = {**gfx942_sys_info, "total_l2_chan": "N/A"}
        with patch("utils.metrics.evaluation_pipeline.console_warning"):
            typed = type_sys_vars(sys_info, missing_as_null=True)
        assert np.isnan(typed["total_l2_chan"])

    def test_type_sys_vars_zero_stays_zero(self, gfx942_sys_info):
        """A zero required value stays 0 in null mode and still warns."""
        sys_info = {**gfx942_sys_info, "total_l2_chan": 0}
        with patch(
            "utils.metrics.evaluation_pipeline.console_warning"
        ) as console_warning_mock:
            typed = type_sys_vars(sys_info, missing_as_null=True)
        assert typed["total_l2_chan"] == 0
        assert "total_l2_chan" in console_warning_mock.call_args.args[0]

    def test_type_sys_vars_num_xcd_defaults_to_one(self, gfx942_sys_info):
        """num_xcd is 1 when the arch does not report it."""
        sys_info = {k: v for k, v in gfx942_sys_info.items() if k != "num_xcd"}
        typed = type_sys_vars(sys_info, missing_as_null=True)
        assert typed["num_xcd"] == 1

    def test_create_sys_vars_keeps_zero_fallback_for_missing_required(
        self, gfx942_sys_info
    ):
        """create_sys_vars keeps the CLI's 0 fallback for a missing required value."""
        sys_info = pd.Series({
            **gfx942_sys_info,
            "cu_per_gpu": np.nan,
            "cur_sclk": np.nan,
        })
        with patch("utils.metrics.evaluation_pipeline.console_warning"):
            result = create_sys_vars(sys_info)
        assert result["ammolite__cu_per_gpu"] == 0
        assert isinstance(result["ammolite__cu_per_gpu"], int)
        # A float var keeps its float type when it falls back to 0.
        assert isinstance(result["ammolite__cur_sclk"], float)
        assert result["ammolite__cur_sclk"] == 0.0

    def test_empirical_peaks_keep_every_column_from_the_first_row(self):
        """Every roofline.csv column becomes a peak, taken from the first row."""
        roofline_df = pd.DataFrame({
            "device": [0, 1],
            "MFMAF6F4Flops": [1234.5, 99.0],
            "WMMAF64Flops": [50.0, 7.0],
        })
        peaks = create_empirical_peaks_dict(roofline_df, prefix="")
        assert peaks == {
            "device_empirical_peak": 0,
            "MFMAF6F4Flops_empirical_peak": 1234.5,
            "WMMAF64Flops_empirical_peak": 50.0,
        }

    def test_check_gui_active_counter_exempts_roofline_only_runs(self, gfx942_sys_info):
        """A roofline-only run carries no GPU-activity counters, so it never halts."""
        raw_pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [0, 0]})
        sys_info = pd.Series({**gfx942_sys_info, "ip_blocks": "roofline"})
        with patch(
            "utils.metrics.evaluation_pipeline.console_error"
        ) as console_error_mock:
            check_gui_active_counter(sys_info, raw_pmc_df)
        console_error_mock.assert_not_called()

    def test_check_gui_active_counter_passes_nonzero_activity(self, gfx942_sys_info):
        """Non-zero activity, or no GRBM counter at all, does not halt."""
        with patch(
            "utils.metrics.evaluation_pipeline.console_error"
        ) as console_error_mock:
            check_gui_active_counter(
                gfx942_sys_info, pd.DataFrame({"GRBM_GUI_ACTIVE": [5, 9]})
            )
            check_gui_active_counter(gfx942_sys_info, pd.DataFrame({"SQ_WAVES": [0]}))
        console_error_mock.assert_not_called()
