# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metrics/evaluation_pipeline.py."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from utils.metrics.evaluation_pipeline import (
    check_gui_active_counter,
    create_empirical_peaks_dict,
    is_pct_of_peak_flagged,
    type_sys_vars,
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

    def test_empirical_peaks_without_roofline_csv_are_nan(self):
        peaks = create_empirical_peaks_dict(pd.DataFrame())
        # Ceilings only the gfx9 and only the gfx11 benchmarks measure.
        for name in ("MFMAF8Flops", "BF16Flops"):
            assert np.isnan(peaks[f"ammolite__{name}_empirical_peak"])
        assert all(np.isnan(value) for value in peaks.values())

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
