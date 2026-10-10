# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""High-level metric-evaluation pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import console_debug, console_error, console_warning, demarcate
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
from utils.metrics.pass_provenance import (
    PassLayout,
    bind_metric_tables_to_passes,
    legacy_pass_merge_enabled,
    ordered_scoped_builtin_bindings,
    pass_scoped_builtins,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_analysis import PEAK_COL_PREFERENCE, VALUE_COL_PREFERENCE
from utils.utils_common import SUPPORTED_FIELD
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars


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
            "BF16Flops",
            "FP32Flops",
            "FP64Flops",
            "I8Ops",
            "I32Ops",
            "I64Ops",
            "MFMAF4Flops",
            "MFMAF6Flops",
            "MFMAF6F4Flops",
            "MFMAF64Flops",
            "MFMAF32Flops",
            "MFMAF16Flops",
            "MFMABF16Flops",
            "MFMAMXF8Flops",
            "MFMAF8Flops",
            "MFMAI8Ops",
            "WMMAF4Flops",
            "WMMAF6Flops",
            "WMMAF6F4Flops",
            "WMMAF64Flops",
            "WMMAF32Flops",
            "WMMAF16Flops",
            "WMMABF16Flops",
            "WMMAMXF8Flops",
            "WMMAF8Flops",
            "WMMAI8Ops",
            "HBMBw",
            "MALLBw",
            "L2Bw",
            "L1Bw",
            "L0Bw",
            "LDSBw",
        ]
        # initialize peaks to NaN
        for peak_name in peak_names:
            empirical_peaks[f"ammolite__{peak_name}_empirical_peak"] = np.nan

    return empirical_peaks


def create_sys_vars(sys_info: pd.Series) -> dict[str, int | float]:
    """Create variables from sys.info."""
    sys_vars_collection = {}
    sys_info_dict = sys_info.to_dict()

    # Present for every arch; warn when missing or zero.
    required_sys_vars = [
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
    ]
    # Arch-specific; silently skipped when the column is absent.
    optional_sys_vars = [
        ("num_memory_channels", float),
        ("num_gl1c", int),
    ]

    for var_name, var_type in required_sys_vars:
        raw_value = sys_info_dict.get(var_name)
        if pd.isna(raw_value) or var_type(raw_value) == 0:
            console_warning(
                f"{var_name} is not available in sysinfo.csv, please provide the "
                "correct value using --specs-correction"
            )
            raw_value = 0
        sys_vars_collection[f"ammolite__{var_name}"] = var_type(raw_value)

    for var_name, var_type in optional_sys_vars:
        raw_value = sys_info_dict.get(var_name)
        if not pd.isna(raw_value):
            sys_vars_collection[f"ammolite__{var_name}"] = var_type(raw_value)

    # num_xcd is a CDNA-only concept; RDNA is single-die, so default to 1.
    raw_num_xcd = sys_info_dict.get("num_xcd")
    sys_vars_collection["ammolite__num_xcd"] = (
        1 if pd.isna(raw_num_xcd) else int(raw_num_xcd)
    )

    return sys_vars_collection


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


def _gui_active_zero_detected(raw_pmc_df: pd.DataFrame) -> Optional[str]:
    """Return a GUI-active column name that contains a zero, if any."""
    candidates = [
        column
        for column in raw_pmc_df.columns
        if column == "GRBM_GUI_ACTIVE"
        or column == "GRBM_GUI_ACTIVE_sum"
        or column.startswith("GRBM_GUI_ACTIVE@pass:")
        or column.startswith("GRBM_GUI_ACTIVE_sum@pass:")
    ]
    for column in candidates:
        if (raw_pmc_df[column] == 0).any():
            return column
    return None


def _calc_pass_scoped_builtins(
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, int | float],
    gpu_arch: str,
    pass_layout: PassLayout,
    used_passes: set[str],
    expressions: list[str],
) -> dict[str, Optional[str | float | int]]:
    """Compute built-ins with duplicated counters bound to each used pass."""
    gpu_series = mi_gpu_specs.get_gpu_series(gpu_arch)
    scoped = pass_scoped_builtins(pass_layout, gpu_series)
    if not scoped or not used_passes:
        return {}

    _, expression_builtin_vars = extract_counters_and_variables(
        "\n".join(expressions), gpu_series
    )
    build_in_vars = {
        key: value
        for key, value in get_build_in_vars(gpu_series).items()
        if key in expression_builtin_vars and key in scoped
    }
    if not build_in_vars:
        return {}

    results: dict[str, Optional[str | float | int]] = {}
    for pass_key in sorted(used_passes, key=lambda key: pass_layout.ordinal(key)):
        ordinal = pass_layout.ordinal(pass_key)
        pass_locals: dict[str, Optional[str | float | int]] = {}
        for variable_key, eval_string in ordered_scoped_builtin_bindings(
            build_in_vars, pass_key, pass_layout, scoped
        ):
            try:
                combined_vars = {**sys_vars, **pass_locals}
                temporary_evaluator = MetricEvaluator(raw_pmc_df, combined_vars, {})
                calculation_result = temporary_evaluator.eval_expression(eval_string)
                if np.isscalar(calculation_result) and calculation_result == "N/A":
                    calculation_result = np.nan
                pass_locals[f"ammolite__{variable_key}__pass{ordinal}"] = (
                    calculation_result
                )
            except (TypeError, NameError, KeyError):
                console_debug(
                    "pass_provenance",
                    f"builtin {variable_key!r} pass{ordinal}: eval failed, using nan",
                )
                pass_locals[f"ammolite__{variable_key}__pass{ordinal}"] = np.nan
        results.update(pass_locals)
    return results


@demarcate
def eval_metric(
    dfs: dict,
    dfs_type: dict,
    dfs_expressions: dict[int, list[str]],
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    debug: bool,
    pass_layout: Optional[PassLayout] = None,
) -> None:
    """Execute the expr string for each metric in the df."""
    # confirm no illogical counter values (only consider non-roofline runs)
    roof_only_run = sys_info.ip_blocks == "roofline"
    zero_gui = None if roof_only_run else _gui_active_zero_detected(raw_pmc_df)
    if zero_gui is not None:
        console_warning(f"Detected {zero_gui} == 0")
        console_error("Halting execution for warning above.")

    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    expressions = [
        expr
        for df_id in dfs
        if dfs_type.get(df_id) == "metric_table"
        for expr in dfs_expressions.get(df_id, [])
    ]

    used_passes: set[str] = set()
    if (
        pass_layout is not None
        and pass_layout.has_duplicates
        and not legacy_pass_merge_enabled()
    ):
        gpu_series = mi_gpu_specs.get_gpu_series(sys_info["gpu_arch"])
        used_passes = bind_metric_tables_to_passes(
            dfs,
            dfs_type,
            pass_layout,
            gpu_series,
            frozenset(SUPPORTED_FIELD),
        )

    builtin_vars = calc_builtin_vars(
        raw_pmc_df, sys_vars, sys_info["gpu_arch"], expressions
    )
    sys_vars.update(builtin_vars)
    if pass_layout is not None and used_passes:
        sys_vars.update(
            _calc_pass_scoped_builtins(
                raw_pmc_df,
                sys_vars,
                sys_info["gpu_arch"],
                pass_layout,
                used_passes,
                expressions,
            )
        )

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

        # astype(bool) handles both Python bool and numpy.bool_ from pandas dtypes
        mask = df[pct_of_peak_col].astype(bool)
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
