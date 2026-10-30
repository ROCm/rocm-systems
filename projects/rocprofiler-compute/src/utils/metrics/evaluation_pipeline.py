# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""High-level metric-evaluation pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from utils.logger import console_error, console_warning, demarcate
from utils.metrics.aggregation import calc_pct_of_peak
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
from utils.utils_analysis import PEAK_COL_PREFERENCE, VALUE_COL_PREFERENCE
from utils.utils_common import SUPPORTED_FIELD
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars

# Present for every arch; a missing or zero value warns.
REQUIRED_SYS_VARS: tuple[tuple[str, type], ...] = (
    ("se_per_gpu", int),
    ("pipes_per_gpu", int),
    ("cu_per_gpu", int),
    ("simd_per_cu", int),
    ("sqc_per_gpu", int),
    ("lds_banks_per_cu", int),
    ("cur_sclk", float),
    ("cur_mclk", float),
    ("max_mclk", float),
    ("max_sclk", float),
    ("max_waves_per_cu", int),
    ("wave_size", int),
    ("total_l2_chan", int),
)
# Arch-specific; silently skipped when the value is absent.
OPTIONAL_SYS_VARS: tuple[tuple[str, type], ...] = (
    ("num_memory_channels", float),
    ("num_gl1c", int),
)


def create_empirical_peaks_dict(
    empirical_peaks_df: pd.DataFrame,
    prefix: str = "ammolite__",
) -> dict[str, float]:
    """Create empirical peaks dictionary, one entry per roofline.csv column.

    The peaks come from the first row. The keys are prefix plus the column name
    and "_empirical_peak".
    """
    if empirical_peaks_df.empty:
        # Without roofline.csv, every ceiling is NaN.
        return {
            f"{prefix}{name}_empirical_peak": np.nan
            for name in mi_gpu_specs.get_all_roofline_columns()
        }

    peak_data_row = empirical_peaks_df.iloc[0]
    return {
        f"{prefix}{col}_empirical_peak": peak_data_row[col]
        for col in empirical_peaks_df.columns
    }


def type_sys_vars(
    sys_info: Mapping[str, Any],
    missing_as_null: bool,
) -> dict[str, int | float]:
    """Type the sys.info values that metric expressions read, by bare name.

    A missing required value warns. It becomes 0 for the CLI, and NaN (NULL) when
    missing_as_null is set, so an expression that uses it evaluates to NULL. A zero
    stays 0 and warns too. num_xcd defaults to 1 because RDNA is single-die.
    """
    typed: dict[str, Any] = {}

    for var_name, var_type in REQUIRED_SYS_VARS:
        number = _cast_sys_value(sys_info.get(var_name), var_type)
        if number is None or number == 0:
            console_warning(
                f"{var_name} is not available in sysinfo.csv, please provide the "
                "correct value using --specs-correction"
            )
        if number is None:
            number = np.nan if missing_as_null else var_type(0)
        typed[var_name] = number

    for var_name, var_type in OPTIONAL_SYS_VARS:
        number = _cast_sys_value(sys_info.get(var_name), var_type)
        if number is not None:
            typed[var_name] = number

    # num_xcd is a CDNA-only concept; RDNA is single-die, so default to 1.
    num_xcd = _cast_sys_value(sys_info.get("num_xcd"), int)
    typed["num_xcd"] = 1 if num_xcd is None else num_xcd

    return typed


def create_sys_vars(sys_info: pd.Series) -> dict[str, int | float]:
    """Create CLI variables from sys.info, where a missing required value is 0."""
    return {
        f"ammolite__{var_name}": value
        for var_name, value in type_sys_vars(
            sys_info.to_dict(), missing_as_null=False
        ).items()
    }


def calc_builtin_vars(
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, int | float],
    gpu_arch: str,
    expressions: list[str],
) -> dict[str, Optional[str | float | int]]:
    """Evaluate built-in variables referenced by expressions."""
    # TODO: fix all $normUnit in Unit column or title
    builtin_vars_collection = {}
    gpu_series = mi_gpu_specs.get_gpu_series(gpu_arch)
    _, expression_builtin_vars = extract_counters_and_variables(
        "\n".join(expressions), gpu_series
    )
    build_in_vars = {
        k: v
        for k, v in get_build_in_vars(gpu_series).items()
        if k in expression_builtin_vars
    }

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


def check_gui_active_counter(
    sys_info: pd.Series | Mapping[str, Any],
    raw_pmc_df: pd.DataFrame,
) -> None:
    """Halt the run when a non-roofline workload reports zero GPU activity."""
    # Only non-roofline runs carry the GRBM counters that prove activity.
    if sys_info.get("ip_blocks") == "roofline":
        return
    gui_active_counter = next(
        (
            counter
            for counter in ("GRBM_GUI_ACTIVE_sum", "GRBM_GUI_ACTIVE")
            if counter in raw_pmc_df.columns
        ),
        None,
    )
    if gui_active_counter is not None and (raw_pmc_df[gui_active_counter] == 0).any():
        console_warning(f"Detected {gui_active_counter} == 0")
        console_error("Halting execution for warning above.")


def is_pct_of_peak_flagged(flag: object) -> bool:
    """Read a Percent of Peak flag, whether it is still a bool or already a string.

    The metric config's pct_of_peak is a bool, but build_metric_value_string turns
    it into "True" or "" before evaluation. Only "True" counts as flagged.
    """
    return str(flag).strip() == "True"


@demarcate
def eval_metric(
    dfs: dict,
    dfs_type: dict,
    dfs_expressions: dict[int, list[str]],
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    debug: bool,
) -> None:
    """Execute the expr string for each metric in the df."""
    check_gui_active_counter(sys_info, raw_pmc_df)

    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    expressions = [
        expr
        for df_id in dfs
        if dfs_type.get(df_id) == "metric_table"
        for expr in dfs_expressions.get(df_id, [])
    ]
    builtin_vars = calc_builtin_vars(
        raw_pmc_df, sys_vars, sys_info["gpu_arch"], expressions
    )
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
                    if expr in SUPPORTED_FIELD and expr.lower() not in {
                        "alias",
                        "percent of peak",
                    }:
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

    # Derive Percent of Peak from evaluated Value and Peak columns
    compute_pct_of_peak(dfs, dfs_type)

    # Check for metrics exceeding theoretical peak due to dual-issue
    validate_dual_issue_metrics(dfs, dfs_type, sys_info, raw_pmc_df)


def compute_pct_of_peak(dfs: dict, dfs_type: dict) -> None:
    """Compute and store 100 * value / peak for each row where pct_of_peak is True."""
    pct_of_peak_col = "Percent of Peak"
    for df_id, df in dfs.items():
        if dfs_type[df_id] != "metric_table":
            continue
        if pct_of_peak_col not in df.columns:
            continue

        # Detect value and peak columns using canonical preference order
        value_col = next(
            (col for col in VALUE_COL_PREFERENCE if col in df.columns), None
        )
        peak_col = next((col for col in PEAK_COL_PREFERENCE if col in df.columns), None)
        if not value_col or not peak_col:
            continue

        mask = df[pct_of_peak_col].map(is_pct_of_peak_flagged).astype(bool)
        df[pct_of_peak_col] = ""
        df.loc[mask, pct_of_peak_col] = [
            pct if (pct := calc_pct_of_peak(v, p)) is not None else ""
            for v, p in zip(df.loc[mask, value_col], df.loc[mask, peak_col])
        ]


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


def _cast_sys_value(raw_value: Any, var_type: type) -> Optional[int | float]:  # noqa ANN401
    """Cast one sys.info value to var_type; None when it is missing or not a number."""
    if raw_value is None:
        return None
    number = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(number):
        return None
    return var_type(number)
