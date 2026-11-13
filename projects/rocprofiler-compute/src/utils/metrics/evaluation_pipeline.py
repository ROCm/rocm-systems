# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""High-level metric-evaluation pipeline."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import console_error, console_warning, demarcate
from utils.metrics.aggregation import calc_pct_of_peak
from utils.metrics.collectable import (
    COLLECTABLE_EXPR_CACHE_ATTR,
    CollectPieceLookup,
    apply_composite_metrics,
    avg_column_name,
    begin_collectable_warnings,
    cache_collectable_expressions,
    composite_parent_ids,
    flush_collectable_warnings,
)
from utils.metrics.common import ValuDualIssueDetector
from utils.metrics.debug_row_tracker import DebugRowTracker, debug_row_tracker
from utils.metrics.expression import build_eval_string
from utils.metrics.metric_evaluator import MetricEvaluator, counter_table_index
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
    print_noise_clamp_summary,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_analysis import PEAK_COL_PREFERENCE, VALUE_COL_PREFERENCE
from utils.utils_common import SUPPORTED_FIELD
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars

# Columns of a composite row that combine its parts on each dispatch
_EXTENT_COLUMNS = ("Min", "Max", "Minimum", "Maximum")


def create_empirical_peaks_dict(empirical_peaks_df: pd.DataFrame) -> dict[str, float]:
    """Create empirical peaks dictionary."""
    if empirical_peaks_df.empty:
        # Without roofline.csv, every ceiling is NaN.
        return {
            f"ammolite__{name}_empirical_peak": np.nan
            for name in mi_gpu_specs.get_all_roofline_columns()
        }

    peak_data_row = empirical_peaks_df.iloc[0]
    return {
        f"ammolite__{col}_empirical_peak": peak_data_row[col]
        for col in empirical_peaks_df.columns
    }


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


def pass_evaluators(
    raw_pmc_df: pd.DataFrame,
    pass_tables: Optional[list[pd.DataFrame]],
    sys_vars: dict[str, int | float],
    gpu_arch: str,
    expressions: list[str],
    empirical_peaks: dict[str, float],
) -> list[MetricEvaluator]:
    """Return one evaluator per pass table, then one for raw_pmc_df.

    Each evaluator has the built-in variables of its own table.
    """
    sys_vars = {
        **sys_vars,
        **calc_builtin_vars(raw_pmc_df, sys_vars, gpu_arch, expressions),
    }
    return [
        *(
            MetricEvaluator(
                table,
                {
                    **sys_vars,
                    **calc_builtin_vars(table, sys_vars, gpu_arch, expressions),
                },
                empirical_peaks,
            )
            for table in pass_tables or []
        ),
        MetricEvaluator(raw_pmc_df, sys_vars, empirical_peaks),
    ]


def evaluate_metric_composites(
    dfs: dict,
    dfs_type: dict,
    raw_pmc_df: pd.DataFrame,
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    pass_tables: Optional[list[pd.DataFrame]] = None,
) -> dict[tuple[object, str], object]:
    """Evaluate the COLLECT_SUM and COLLECT_RATIO metrics on copies of the
    metric tables that hold them.

    Returns the value of each (metric_id, column) of those metrics. The
    analysis-db path writes them into its expressions and roofline points.
    """
    copies = {
        table_id: df.copy(deep=True)
        for table_id, df in dfs.items()
        if dfs_type.get(table_id) == "metric_table" and composite_parent_ids(df)
    }
    copy_types = dict.fromkeys(copies, "metric_table")
    cache_collectable_expressions(copies, copy_types)
    expressions = [
        built
        for df in copies.values()
        for built in df.attrs.get(COLLECTABLE_EXPR_CACHE_ATTR, {}).values()
    ]
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    evaluators = pass_evaluators(
        raw_pmc_df,
        pass_tables,
        create_sys_vars(sys_info),
        sys_info["gpu_arch"],
        expressions,
        empirical_peaks,
    )
    build_in_vars = get_build_in_vars(mi_gpu_specs.get_gpu_series(sys_info["gpu_arch"]))
    apply_composite_metrics(copies, copy_types, evaluators, build_in_vars)

    results: dict[tuple[object, str], object] = {}
    for df in copies.values():
        avg_col = avg_column_name(df)
        piece_evaluator = _collect_piece_evaluator(df, evaluators, build_in_vars)
        for metric_id in composite_parent_ids(df):
            results[(metric_id, avg_col)] = df.at[metric_id, avg_col]
            # Min and max combine the parts on each dispatch
            for column in _EXTENT_COLUMNS:
                formula = df.at[metric_id, column] if column in df.columns else ""
                if isinstance(formula, str) and formula:
                    results[(metric_id, column)] = piece_evaluator.eval_expression(
                        formula
                    )
    return results


def overlay_composite_expression_values(
    expression_df: pd.DataFrame,
    filled: dict[tuple[object, str], object],
) -> None:
    """Replace the composite expression cells with their evaluated values."""
    if expression_df.empty or not filled:
        return
    for idx, row in expression_df.iterrows():
        value = filled.get((row.metric_id, row.value_name))
        if value is None:
            continue
        if isinstance(value, str):
            expression_df.at[idx, "value"] = "None" if value in {"", "N/A"} else value
        elif pd.isna(value):
            expression_df.at[idx, "value"] = "None"
        else:
            expression_df.at[idx, "value"] = repr(float(value))


@demarcate
def eval_metric(
    dfs: dict,
    dfs_type: dict,
    dfs_expressions: dict[int, list[str]],
    sys_info: pd.Series,
    empirical_peaks_df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    debug: bool,
    pass_tables: Optional[list[pd.DataFrame]] = None,
) -> None:
    """Execute the expr string for each metric in the df.

    A metric reads the first table in pass_tables that has all of its
    counters. When none has them all, it reads raw_pmc_df. COLLECT_SUM and
    COLLECT_RATIO metrics combine their parts after the other metrics.
    """
    # confirm no illogical counter values
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

    sys_vars = create_sys_vars(sys_info)
    empirical_peaks = create_empirical_peaks_dict(empirical_peaks_df)
    expressions = [
        expr
        for df_id in dfs
        if dfs_type.get(df_id) == "metric_table"
        for expr in dfs_expressions.get(df_id, [])
    ]
    # Keep the _collect rows' expressions; evaluation replaces them with values
    cache_collectable_expressions(dfs, dfs_type)

    # Clear any previous noise clamp warnings before this analysis
    clear_noise_clamp_warnings()

    evaluators = pass_evaluators(
        raw_pmc_df,
        pass_tables,
        sys_vars,
        sys_info["gpu_arch"],
        expressions,
        empirical_peaks,
    )
    tables = [evaluator.raw_pmc_df for evaluator in evaluators]
    build_in_vars = get_build_in_vars(mi_gpu_specs.get_gpu_series(sys_info["gpu_arch"]))

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
            # COLLECT_SUM and COLLECT_RATIO rows read their parts per dispatch
            composite_ids = composite_parent_ids(df)
            piece_evaluator = _collect_piece_evaluator(df, evaluators, build_in_vars)
            for row_id, row in df.iterrows():
                for expr in df.columns:
                    if expr in SUPPORTED_FIELD and expr.lower() not in {
                        "alias",
                        "percent of peak",
                    }:
                        if row[expr]:
                            evaluator = (
                                piece_evaluator
                                if row_id in composite_ids
                                else evaluators[
                                    counter_table_index(
                                        row[expr], tables, build_in_vars
                                    )
                                ]
                            )
                            exprs_to_eval.append((
                                df_id,
                                row_id,
                                expr,
                                row[expr],
                                evaluator,
                            ))

                            if debug:
                                debug_row_tracker(
                                    expr,
                                    row[expr],
                                    evaluator,
                                    evaluator.raw_pmc_df,
                                    show_inputs=debug_tracker.should_show_inputs(
                                        df_id,
                                        row_id,
                                    ),
                                )
                        else:
                            # If not insert nan, the whole col might be treated
                            # as string but not number if there is NONE
                            df.at[row_id, expr] = ""

    # One summary of the collectable parts that were unset or missing
    begin_collectable_warnings()
    try:
        for df_id, row_id, col, expr, evaluator in exprs_to_eval:
            noise_clamp_count_prev = get_noise_clamp_warnings()["count"]
            eval_result = evaluator.eval_expression(expr)
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
        apply_composite_metrics(dfs, dfs_type, evaluators, build_in_vars)
    finally:
        flush_collectable_warnings()

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


def _collect_piece_evaluator(
    metric_table: pd.DataFrame,
    evaluators: list[MetricEvaluator],
    build_in_vars: dict[str, str],
) -> MetricEvaluator:
    """Return an evaluator that resolves the _collect pieces of metric_table.

    It reads the merged table, since a composite row's own formulas name no
    counter.
    """
    merged = evaluators[-1]
    return MetricEvaluator(
        merged.raw_pmc_df,
        merged.sys_vars,
        merged.empirical_peaks,
        collect_piece=CollectPieceLookup(
            metric_table, evaluators, build_in_vars
        ).dispatch_series,
    )
