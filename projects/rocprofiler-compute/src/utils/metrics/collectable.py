# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Collectables and composite metric evaluation graph (Layer 1.5 analyze path)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from utils.logger import console_warning
from utils.metrics.aggregation import (
    merge_dispatch_collect_sum,
    merge_dispatch_weighted_avg,
)
from utils.metrics.expression import (
    is_composite_avg_formula,
)
from utils.metrics.metric_evaluator import MetricEvaluator

METRIC_EVAL_GRAPH_ATTR = "metric_eval_graph"
COLLECTABLE_EXPR_CACHE_ATTR = "collectable_expr_cache"
COLLECTABLE_IDS_ATTR = "collectable_ids"
COLLECT_SUM_SPECS_ATTR = "collect_sum_specs"
WEIGHTED_AVG_ATTR = "weighted_avg_specs"
WEIGHTED_AVG_SUBS_ATTR = "weighted_avg_subs"
METRIC_NAME_COLUMN = "Metric"


class CompositeKind(str, Enum):
    WEIGHTED_AVG = "weighted_avg"
    COLLECT_SUM = "collect_sum"


@dataclass
class CompositeDef:
    metric_id: str
    kind: CompositeKind
    refs: list[str]
    weight_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricEvalGraph:
    """Collectable rows and composite parents for one metric_table."""

    collectable_ids: dict[str, str] = field(default_factory=dict)
    collectable_row_names: set[str] = field(default_factory=set)
    composites: list[CompositeDef] = field(default_factory=list)

    def composite_order(self) -> list[CompositeDef]:
        """Topological order: collectables first, then composites (v1: flat deps)."""
        return list(self.composites)


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


def build_metric_eval_graph(df: pd.DataFrame) -> MetricEvalGraph:
    """Build collectable + composite graph from parser attrs on a metric_table df."""
    graph = MetricEvalGraph()

    collectable_ids_attr = df.attrs.get(COLLECTABLE_IDS_ATTR)
    if isinstance(collectable_ids_attr, dict):
        name_col = _metric_column_name(df)
        if name_col is not None:
            for metric_id, collect_id in collectable_ids_attr.items():
                if not isinstance(collect_id, str) or metric_id not in df.index:
                    continue
                row_name = df.at[metric_id, name_col]
                if isinstance(row_name, str):
                    graph.collectable_ids[row_name] = collect_id
                    graph.collectable_row_names.add(row_name)

    weighted_specs = df.attrs.get(WEIGHTED_AVG_ATTR)
    subs_by_id = df.attrs.get(WEIGHTED_AVG_SUBS_ATTR, {})
    if isinstance(weighted_specs, dict) and isinstance(subs_by_id, dict):
        for metric_id, weight_meta in weighted_specs.items():
            refs = subs_by_id.get(metric_id)
            if not isinstance(refs, list) or metric_id not in df.index:
                continue
            graph.collectable_row_names.update(refs)
            graph.composites.append(
                CompositeDef(
                    metric_id=metric_id,
                    kind=CompositeKind.WEIGHTED_AVG,
                    refs=refs,
                    weight_meta=weight_meta if isinstance(weight_meta, dict) else {},
                )
            )

    collect_sum_specs = df.attrs.get(COLLECT_SUM_SPECS_ATTR)
    if isinstance(collect_sum_specs, dict):
        for metric_id, refs in collect_sum_specs.items():
            if not isinstance(refs, list) or metric_id not in df.index:
                continue
            graph.collectable_row_names.update(refs)
            graph.composites.append(
                CompositeDef(
                    metric_id=metric_id,
                    kind=CompositeKind.COLLECT_SUM,
                    refs=refs,
                )
            )

    df.attrs[METRIC_EVAL_GRAPH_ATTR] = graph
    return graph


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


def per_dispatch_ratio_series(
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


def _weight_counter_per_dispatch(counter: str, raw_pmc_df: pd.DataFrame) -> pd.Series:
    if counter not in raw_pmc_df.columns:
        return pd.Series(dtype=float)
    if "Dispatch_ID" in raw_pmc_df.columns:
        return raw_pmc_df.groupby("Dispatch_ID", dropna=False)[counter].sum()
    total = raw_pmc_df[counter].sum()
    return pd.Series({0: float(total)})


def cache_collectable_expressions(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
) -> None:
    """Snapshot built Avg strings for collectable rows before eval_metric overwrites."""
    for df_id, df in dfs.items():
        if dfs_type.get(df_id) != "metric_table":
            continue
        graph = build_metric_eval_graph(df)
        if not graph.collectable_row_names and not graph.composites:
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
            if metric_name not in graph.collectable_row_names:
                continue
            built = row[avg_col]
            if not isinstance(built, str) or not built:
                continue
            if is_composite_avg_formula(built):
                continue
            by_name[metric_name] = built
        df.attrs[COLLECTABLE_EXPR_CACHE_ATTR] = by_name


def _lookup_collectable_built_avg(
    df: pd.DataFrame,
    metric_name: str,
    avg_col: str,
) -> str | None:
    cached = df.attrs.get(COLLECTABLE_EXPR_CACHE_ATTR, {})
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


def _evaluate_weighted_composite(
    composite: CompositeDef,
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

    for ref_name in composite.refs:
        sub_meta = composite.weight_meta.get(ref_name)
        if not isinstance(sub_meta, dict):
            console_warning(
                f"WEIGHTED_AVG: missing _weighted_avg entry for '{ref_name}'."
            )
            return "N/A"
        weight_counter = sub_meta.get("weight_counter")
        if not isinstance(weight_counter, str) or not weight_counter:
            console_warning(f"WEIGHTED_AVG: missing weight_counter for '{ref_name}'.")
            return "N/A"

        built_avg = _lookup_collectable_built_avg(df, ref_name, avg_col)
        if built_avg is None:
            console_warning(
                f"WEIGHTED_AVG: collectable '{ref_name}' not found in metric table."
            )
            return "N/A"

        ratio_series_list.append(
            per_dispatch_ratio_series(built_avg, raw_pmc_df, sys_vars, empirical_peaks)
        )
        weight_series_list.append(
            _weight_counter_per_dispatch(weight_counter, raw_pmc_df)
        )

    merged = merge_dispatch_weighted_avg(ratio_series_list, weight_series_list)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def _evaluate_collect_sum_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> float | str:
    avg_col = _avg_column_name(df)
    if avg_col is None:
        return "N/A"

    series_list: list[pd.Series] = []
    for ref_name in composite.refs:
        built_avg = _lookup_collectable_built_avg(df, ref_name, avg_col)
        if built_avg is None:
            console_warning(
                f"COLLECT_SUM: collectable '{ref_name}' not found in metric table."
            )
            return "N/A"
        series_list.append(
            per_dispatch_ratio_series(built_avg, raw_pmc_df, sys_vars, empirical_peaks)
        )

    merged = merge_dispatch_collect_sum(series_list)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def apply_composite_metrics(
    dfs: dict[int, pd.DataFrame],
    dfs_type: dict[int, str],
    raw_pmc_df: pd.DataFrame,
    sys_vars: dict[str, Any],
    empirical_peaks: dict[str, Any],
) -> None:
    """Evaluate composite parents after collectable rows (graph order)."""
    for df_id, df in dfs.items():
        if dfs_type.get(df_id) != "metric_table":
            continue
        graph = build_metric_eval_graph(df)
        if not graph.composites:
            continue
        avg_col = _avg_column_name(df)
        if avg_col is None:
            continue

        for composite in graph.composite_order():
            if composite.metric_id not in df.index:
                continue
            if composite.kind is CompositeKind.WEIGHTED_AVG:
                result = _evaluate_weighted_composite(
                    composite, df, raw_pmc_df, sys_vars, empirical_peaks
                )
            elif composite.kind is CompositeKind.COLLECT_SUM:
                result = _evaluate_collect_sum_composite(
                    composite, df, raw_pmc_df, sys_vars, empirical_peaks
                )
            else:
                continue
            df.at[composite.metric_id, avg_col] = result


def collectable_row_names_from_graph(df: pd.DataFrame) -> set[str]:
    return build_metric_eval_graph(df).collectable_row_names
