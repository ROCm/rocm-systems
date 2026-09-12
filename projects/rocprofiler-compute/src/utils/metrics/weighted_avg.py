# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""WEIGHTED_AVG composite metrics (AIPROFCOMP-865 Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.logger import console_warning
from utils.metrics.aggregation import merge_dispatch_weighted_avg
from utils.metrics.expression import parse_weighted_avg_submetrics
from utils.metrics.metric_evaluator import MetricEvaluator
from vendored import yaml

WEIGHTED_AVG_FIELD_NAMES = frozenset({"avg", "average"})
WEIGHTED_AVG_ATTR = "weighted_avg_specs"
WEIGHTED_AVG_SUB_EXPR_ATTR = "weighted_avg_sub_avg_expr"
METRIC_NAME_COLUMN = "Metric"


def _metric_column_name(df: pd.DataFrame) -> str | None:
    for candidate in (METRIC_NAME_COLUMN, "metric"):
        if candidate in df.columns:
            return candidate
    return None


def _avg_column_name(df: pd.DataFrame) -> str | None:
    for candidate in ("Avg", "Average"):
        if candidate in df.columns:
            return candidate
    return None


def _eval_built_expr_on_frame(
    built_expr: str,
    frame: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> float | str:
    if not built_expr:
        return "N/A"
    evaluator = MetricEvaluator(frame, sys_vars, empirical_peaks)
    return evaluator.eval_expression(built_expr)


def _per_dispatch_series(
    built_expr: str,
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> pd.Series:
    if raw_pmc_df.empty:
        return pd.Series(dtype=float)
    dispatch_col = "Dispatch_ID" if "Dispatch_ID" in raw_pmc_df.columns else None
    if dispatch_col is None:
        scalar = _eval_built_expr_on_frame(
            built_expr, raw_pmc_df, sys_vars, empirical_peaks
        )
        if scalar == "N/A" or pd.isna(scalar):
            return pd.Series(dtype=float)
        return pd.Series({0: float(scalar)})

    values: dict[Any, float] = {}
    for dispatch_id, group in raw_pmc_df.groupby(dispatch_col, dropna=False):
        scalar = _eval_built_expr_on_frame(built_expr, group, sys_vars, empirical_peaks)
        if scalar == "N/A" or pd.isna(scalar):
            continue
        values[dispatch_id] = float(scalar)
    return pd.Series(values, dtype=float)


def _weight_counter_per_dispatch(
    counter: str,
    raw_pmc_df: pd.DataFrame,
) -> pd.Series:
    if counter not in raw_pmc_df.columns:
        return pd.Series(dtype=float)
    if "Dispatch_ID" in raw_pmc_df.columns:
        return raw_pmc_df.groupby("Dispatch_ID", dropna=False)[counter].sum()
    total = raw_pmc_df[counter].sum()
    return pd.Series({0: float(total)})


def cache_weighted_avg_sub_expressions(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
) -> None:
    """Snapshot submetric built Avg strings before eval_metric overwrites them."""
    for df_id, df in dfs.items():
        if dfs_type.get(df_id) != "metric_table":
            continue
        if not df.attrs.get(WEIGHTED_AVG_ATTR):
            continue
        name_col = _metric_column_name(df)
        avg_col = _avg_column_name(df)
        if name_col is None or avg_col is None:
            continue
        by_name: dict[str, str] = {}
        for _, row in df.iterrows():
            metric_name = row[name_col]
            if not isinstance(metric_name, str):
                continue
            built = row[avg_col]
            if not isinstance(built, str) or not built:
                continue
            if parse_weighted_avg_submetrics(built) is not None:
                continue
            by_name[metric_name] = built
        df.attrs[WEIGHTED_AVG_SUB_EXPR_ATTR] = by_name


def _lookup_submetric_built_avg(
    df: pd.DataFrame,
    metric_name: str,
    avg_col: str,
) -> str | None:
    cached = df.attrs.get(WEIGHTED_AVG_SUB_EXPR_ATTR, {})
    if isinstance(cached, dict) and metric_name in cached:
        return cached[metric_name]

    name_col = _metric_column_name(df)
    if name_col is None:
        return None
    matches = df[df[name_col] == metric_name]
    if matches.empty:
        return None
    built = matches.iloc[0][avg_col]
    if not isinstance(built, str) or not built:
        return None
    return built


def evaluate_weighted_avg_parent(
    submetric_names: list[str],
    weight_meta: dict[str, Any],
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> float | str:
    avg_col = _avg_column_name(df)
    if avg_col is None:
        return "N/A"

    ratio_series_list: list[pd.Series] = []
    weight_series_list: list[pd.Series] = []

    for sub_name in submetric_names:
        sub_meta = weight_meta.get(sub_name)
        if not isinstance(sub_meta, dict):
            console_warning(
                f"WEIGHTED_AVG: missing _weighted_avg entry for submetric '{sub_name}'."
            )
            return "N/A"
        weight_counter = sub_meta.get("weight_counter")
        if not isinstance(weight_counter, str) or not weight_counter:
            console_warning(
                f"WEIGHTED_AVG: missing weight_counter for submetric '{sub_name}'."
            )
            return "N/A"

        built_avg = _lookup_submetric_built_avg(df, sub_name, avg_col)
        if built_avg is None:
            console_warning(
                f"WEIGHTED_AVG: submetric '{sub_name}' not found in metric table."
            )
            return "N/A"

        ratio_series_list.append(
            _per_dispatch_series(built_avg, raw_pmc_df, sys_vars, empirical_peaks)
        )
        weight_series_list.append(
            _weight_counter_per_dispatch(weight_counter, raw_pmc_df)
        )

    merged = merge_dispatch_weighted_avg(ratio_series_list, weight_series_list)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def apply_weighted_avg_metrics(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> None:
    """Fill parent Avg cells that use WEIGHTED_AVG(...) after submetrics are built."""
    for df_id, df in dfs.items():
        if dfs_type.get(df_id) != "metric_table":
            continue
        specs = df.attrs.get(WEIGHTED_AVG_ATTR)
        if not specs:
            continue
        avg_col = _avg_column_name(df)
        if avg_col is None:
            continue

        subs_by_id = df.attrs.get("weighted_avg_subs", {})
        for metric_id, weight_meta in specs.items():
            if metric_id not in df.index:
                continue
            sub_names = subs_by_id.get(metric_id)
            if not sub_names:
                continue

            result = evaluate_weighted_avg_parent(
                sub_names,
                weight_meta,
                df,
                raw_pmc_df,
                sys_vars,
                empirical_peaks,
            )
            df.at[metric_id, avg_col] = result


def scan_weighted_avg_parents(
    config_arch_path: Path,
) -> list[tuple[str, str, list[str]]]:
    """Return (yaml file, metric key, submetric names) for WEIGHTED_AVG parents."""
    if not config_arch_path.is_dir():
        return []

    found: list[tuple[str, str, list[str]]] = []
    for ypath in sorted(config_arch_path.glob("*.yaml")):
        try:
            with open(ypath, encoding="utf-8") as stream:
                doc = yaml.safe_load(stream)
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        panel_cfg = doc.get("Panel Config")
        if not isinstance(panel_cfg, dict):
            continue
        sources = panel_cfg.get("data source")
        if not isinstance(sources, list):
            continue
        for section in sources:
            if not isinstance(section, dict):
                continue
            metric_table = section.get("metric_table")
            if not isinstance(metric_table, dict):
                continue
            metrics = metric_table.get("metric")
            if not isinstance(metrics, dict):
                continue
            for metric_key, body in metrics.items():
                if not isinstance(body, dict):
                    continue
                avg_formula = body.get("avg")
                if not isinstance(avg_formula, str):
                    continue
                subs = parse_weighted_avg_submetrics(avg_formula)
                if subs and isinstance(body.get("_weighted_avg"), dict):
                    found.append((ypath.name, metric_key, subs))
    return found


def format_weighted_avg_inspector_section(
    config_arch_path: Path,
) -> str:
    """Text block for counter_grouping_inspector (Milestone C hint)."""
    parents = scan_weighted_avg_parents(config_arch_path)
    lines = ["WEIGHTED_AVG parent metrics (analyze-only composites):"]
    if not parents:
        lines.append("  (none in analysis_configs for this arch)")
        return "\n".join(lines) + "\n\n"
    for file_name, metric_key, subs in parents:
        lines.append(f"  - {file_name}: {metric_key} -> submetrics {subs}")
    lines.append("  Verify each submetric id is single-bucket in the plan above.")
    return "\n".join(lines) + "\n\n"
