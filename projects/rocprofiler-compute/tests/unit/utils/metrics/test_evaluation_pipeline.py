# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/evaluation_pipeline.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.evaluation_pipeline import (
    compute_pct_of_peak,
    create_sys_vars,
    eval_metric,
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
