# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.evaluation_pipeline."""

from unittest.mock import patch

import pandas as pd

from utils.metrics.evaluation_pipeline import eval_metric


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
        sys_info = pd.Series({
            "ip_blocks": "standard",
            "gpu_arch": "gfx90a",
            "se_per_gpu": 4,
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
        return metric_df, dfs, dfs_type, sys_info, raw_pmc_df

    def test_eval_metric_in_debug_mode(self):
        """eval_metric with debug=True invokes debug_row_tracker and writes back."""
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
            }
        )
        with (
            patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}),
            patch(
                "utils.metrics.evaluation_pipeline.debug_row_tracker"
            ) as mock_debug_row_tracker,
        ):
            eval_metric(
                dfs,
                dfs_type,
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
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={
                "Value": "to_sum(raw_pmc_df['SQ_WAVES'])",
            }
        )
        with patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Value"] == 450

    def test_eval_metric_resolves_accum_alias_column_end_to_end(self):
        """eval_metric resolves YAML formulas that reference ACCUM alias columns."""
        metric_df, dfs, dfs_type, sys_info, _ = self._build_eval_metric_inputs(
            metric_fields={
                "Value": (
                    "to_sum(raw_pmc_df['SQ_INST_LEVEL_VMEM_ACCUM']) / "
                    "to_sum(raw_pmc_df['SQ_INSTS_VMEM'])"
                ),
            }
        )
        flat_raw_pmc_df = pd.DataFrame({
            "SQ_INST_LEVEL_VMEM_ACCUM": [100.0, 200.0, 300.0],
            "SQ_INSTS_VMEM": [10.0, 20.0, 30.0],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })
        with patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                flat_raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Value"] == 10.0

    def test_eval_metric_normalizes_falsey_average_to_empty_string(self):
        """eval_metric replaces a falsey Average value with the empty string."""
        metric_df, dfs, dfs_type, sys_info, raw_pmc_df = self._build_eval_metric_inputs(
            metric_fields={"Average": None}
        )
        assert metric_df.loc["1.1.0", "Average"] is None

        with patch("utils.metrics.evaluation_pipeline.BUILD_IN_VARS", {}):
            eval_metric(
                dfs,
                dfs_type,
                sys_info,
                pd.DataFrame(),
                raw_pmc_df,
                debug=False,
            )
        assert metric_df.loc["1.1.0", "Average"] == ""
