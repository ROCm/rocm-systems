# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""High-level metric-evaluation pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import console_error, console_warning, demarcate
from utils.metrics.common import ValuDualIssueDetector
from utils.metrics.debug_row_tracker import DebugRowTracker, debug_row_tracker
from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
    print_noise_clamp_summary,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import SUPPORTED_FIELD, calc_builtin_var
from utils.utils_counter_defs import get_build_in_vars


def create_empirical_peaks_dict(empirical_peaks_df: pd.DataFrame) -> dict[str, float]:
    """Create empirical peaks dictionary."""
    empirical_peaks = {}

    if not empirical_peaks_df.empty:
        peak_data_row = empirical_peaks_df.iloc[0]
        for col in empirical_peaks_df.columns:
            empirical_peaks[f"ammolite__{col}_empirical_peak"] = peak_data_row[col]
    else:
        peak_names = [
            "FP16Flops",
            "FP32Flops",
            "FP64Flops",
            "MFMAF64Flops",
            "MFMAF32Flops",
            "MFMAF16Flops",
            "MFMABF16Flops",
            "MFMAF8Flops",
            "MFMAI8Ops",
            "HBMBw",
            "L2Bw",
            "L1Bw",
            "LDSBw",
            "MFMAF6F4Flops",
        ]
        # initialize peaks to NaN
        for peak_name in peak_names:
            empirical_peaks[f"ammolite__{peak_name}_empirical_peak"] = np.nan

    return empirical_peaks


def create_sys_vars(sys_info: pd.Series) -> dict[str, int | float]:
    """Create variables from sys.info."""
    sys_vars_collection = {}

    sys_vars_config = [
        ("se_per_gpu", int, "se_per_gpu"),
        ("pipes_per_gpu", int, "pipes_per_gpu"),
        ("cu_per_gpu", int, "cu_per_gpu"),
        ("simd_per_cu", int, "simd_per_cu"),
        ("sqc_per_gpu", int, "sqc_per_gpu"),
        ("lds_banks_per_cu", int, "lds_banks_per_cu"),
        ("cur_sclk", float, "cur_sclk"),
        ("cur_mclk", float, "cur_mclk"),
        ("max_mclk", float, "max_mclk"),
        ("max_sclk", float, "max_sclk"),
        ("max_waves_per_cu", int, "max_waves_per_cu"),
        ("num_hbm_channels", float, "num_hbm_channels"),
        ("num_xcd", int, "num_xcd"),
        ("wave_size", int, "wave_size"),
    ]

    for var_name, var_type, attr_name in sys_vars_config:
        variable_value = var_type(getattr(sys_info, attr_name))
        if np.isnan(variable_value) or variable_value == 0:
            console_warning(
                f"{attr_name} is not available in sysinfo.csv, please provide the "
                "correct value using --specs-correction"
            )
        sys_vars_collection[f"ammolite__{var_name}"] = variable_value

    # Special case for total_l2_chan
    total_l2_channel_count = calc_builtin_var("$total_l2_chan", sys_info.to_dict())
    if np.isnan(total_l2_channel_count) or total_l2_channel_count == 0:
        console_warning(
            "total_l2_chan is not available in sysinfo.csv, please provide the correct "
            "value using --specs-correction"
        )
    sys_vars_collection["ammolite__total_l2_chan"] = total_l2_channel_count

    return sys_vars_collection


def calc_builtin_vars(
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, int | float],
    gpu_arch: str,
) -> dict[str, Optional[str | float | int]]:
    """Calculate built-in variables."""
    # TODO: fix all $normUnit in Unit column or title
    # build and eval all derived build-in global variables
    builtin_vars_collection = {}
    build_in_vars = get_build_in_vars(mi_gpu_specs.get_gpu_series(gpu_arch))

    # First pass: calculate per-XCD values
    for variable_key, variable_value in build_in_vars.items():
        if "PER_XCD" not in variable_key:
            continue

        eval_string = build_eval_string(variable_value)
        try:
            # Create temporary evaluator for this calculation
            # Pass sys_vars so that $num_xcd and other system variables are available
            temporary_evaluator = MetricEvaluator(raw_pmc_df, sys_vars, {})
            calculation_result = temporary_evaluator.eval_expression(eval_string)
            # Convert "N/A" string to np.nan to maintain numeric type for calculations
            if np.isscalar(calculation_result) and calculation_result == "N/A":
                calculation_result = np.nan
            builtin_vars_collection[f"ammolite__{variable_key}"] = calculation_result
        except (TypeError, NameError, KeyError, AttributeError):
            builtin_vars_collection[f"ammolite__{variable_key}"] = np.nan

    # Second pass: calculate remaining variables that depend on per-XCD values
    for variable_key, variable_value in build_in_vars.items():
        if "PER_XCD" in variable_key:
            continue

        eval_string = build_eval_string(variable_value)
        try:
            # Merge sys_vars with builtin_vars_collection for second pass
            combined_vars = {**sys_vars, **builtin_vars_collection}
            temporary_evaluator = MetricEvaluator(raw_pmc_df, combined_vars, {})
            calculation_result = temporary_evaluator.eval_expression(eval_string)
            # Convert "N/A" string to np.nan to maintain numeric type for calculations
            if np.isscalar(calculation_result) and calculation_result == "N/A":
                calculation_result = np.nan
            builtin_vars_collection[f"ammolite__{variable_key}"] = calculation_result
        except (TypeError, NameError, KeyError, AttributeError):
            builtin_vars_collection[f"ammolite__{variable_key}"] = np.nan

    return builtin_vars_collection


@demarcate
def eval_metric(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    debug: bool,
) -> None:
    """Execute the expr string for each metric in the df."""
    # confirm no illogical counter values (only consider non-roofline runs)
    roof_only_run = sys_info.ip_blocks == "roofline"
    if (
        (not roof_only_run)
        and "GRBM_GUI_ACTIVE" in raw_pmc_df.columns
        and (raw_pmc_df["GRBM_GUI_ACTIVE"] == 0).any()
    ):
        console_warning("Detected GRBM_GUI_ACTIVE == 0")
        console_error("Halting execution for warning above.")

    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    builtin_vars = calc_builtin_vars(raw_pmc_df, sys_vars, sys_info["gpu_arch"])
    sys_vars.update(builtin_vars)

    # Clear any previous noise clamp warnings before this analysis
    clear_noise_clamp_warnings()

    # Create metric evaluator
    metric_evaluator = MetricEvaluator(raw_pmc_df, sys_vars, empirical_peaks)

    exprs_to_eval = []
    debug_tracker = DebugRowTracker() if debug else None

    # Hmmm... apply + lambda should just work
    # df['Value'] = df['Value'].apply(
    #     lambda s: eval(
    #         compile(str(s), '<string>', 'eval')
    #     )
    # )
    for df_id, df in dfs.items():
        if dfs_type[df_id] == "metric_table":
            for row_id, row in df.iterrows():
                for expr in df.columns:
                    if expr in SUPPORTED_FIELD and expr.lower() != "alias":
                        if row[expr]:
                            exprs_to_eval.append((df_id, row_id, expr, row[expr]))

                            if debug:
                                debug_row_tracker(
                                    expr,
                                    row[expr],
                                    metric_evaluator,
                                    raw_pmc_df,
                                    show_inputs=debug_tracker.should_show_inputs(
                                        df_id,
                                        row_id,
                                    ),
                                )
                        else:
                            # If not insert nan, the whole col might be treated
                            # as string but not number if there is NONE
                            df.at[row_id, expr] = ""

    for df_id, row_id, col, expr in exprs_to_eval:
        noise_clamp_count_prev = get_noise_clamp_warnings()["count"]
        eval_result = metric_evaluator.eval_expression(expr)
        noise_clamp_count_new = get_noise_clamp_warnings()["count"]
        if (
            noise_clamp_count_new > noise_clamp_count_prev
            and "Metric" in dfs[df_id].columns
        ):
            metric_name = dfs[df_id].loc[row_id, "Metric"]
            console_warning(
                f"Variance corrected for metric: {row_id} {metric_name} {col}"
            )
        dfs[df_id].loc[row_id, col] = eval_result

    # Print aggregated summary of any noise clamping warnings
    print_noise_clamp_summary()

    # Check for metrics exceeding theoretical peak due to dual-issue
    validate_dual_issue_metrics(dfs, dfs_type, sys_info, raw_pmc_df)


def validate_dual_issue_metrics(
    dfs: dict,
    dfs_type: dict,
    sys_info: pd.Series,
    raw_pmc_df: pd.DataFrame,
) -> None:
    """Warn when VALU metrics exceed peak in the eval_metric results."""
    detector = ValuDualIssueDetector(
        gpu_arch=sys_info.get("gpu_arch", ""),
        raw_pmc_df=raw_pmc_df,
    )

    for df_id, df in dfs.items():
        if dfs_type[df_id] != "metric_table":
            continue
        if "Metric" not in df.columns or "Value" not in df.columns:
            continue
        if "Peak (Empirical)" in df.columns:
            peak_col = "Peak (Empirical)"
        elif "Peak" in df.columns:
            peak_col = "Peak"
        else:
            continue

        for _, row in df.iterrows():
            metric_name = row.get("Metric", "")
            if metric_name not in ValuDualIssueDetector.candidate_metrics:
                continue
            try:
                value = float(row.get("Value", 0))
                peak = float(row.get(peak_col, 0))
            except (ValueError, TypeError):
                # DB cells may be non-numeric (e.g. "N/A"); skip those.
                continue
            detector.check(metric_name, value, peak)
