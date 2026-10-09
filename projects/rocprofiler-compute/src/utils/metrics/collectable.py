# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Collectables and composite metric evaluation graph (Layer 1.5 analyze path)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union

import pandas as pd

from utils.logger import console_warning
from utils.metrics.aggregation import (
    merge_dispatch_collect_ratio,
    merge_dispatch_collect_sum,
    merge_dispatch_weighted_avg,
)
from utils.metrics.expression import (
    is_composite_avg_formula,
)
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.metrics.pass_provenance import resolve_weight_counter_column

if TYPE_CHECKING:
    from utils.metrics.pass_provenance import PassLayout

METRIC_EVAL_GRAPH_ATTR = "metric_eval_graph"
COLLECTABLE_EXPR_CACHE_ATTR = "collectable_expr_cache"
COLLECTABLE_IDS_ATTR = "collectable_ids"
COLLECT_SUM_SPECS_ATTR = "collect_sum_specs"
COLLECT_RATIO_SPECS_ATTR = "collect_ratio_specs"
WEIGHTED_AVG_ATTR = "weighted_avg_specs"
WEIGHTED_AVG_SUBS_ATTR = "weighted_avg_subs"
METRIC_NAME_COLUMN = "Metric"


class CompositeKind(str, Enum):
    WEIGHTED_AVG = "weighted_avg"
    COLLECT_SUM = "collect_sum"
    COLLECT_RATIO = "collect_ratio"


@dataclass
class CompositeDef:
    metric_id: str
    kind: CompositeKind
    refs: List[str]
    weight_meta: Dict[str, Any] = field(default_factory=dict)
    metric_name: str = ""


def _ref_base_name(ref: str) -> str:
    name = ref.strip()
    if name.startswith("-"):
        name = name[1:].strip()
    return name


@dataclass
class MetricEvalGraph:
    """Collectable rows and composite parents for one metric_table."""

    collectable_ids: Dict[str, str] = field(default_factory=dict)
    collectable_row_names: Set[str] = field(default_factory=set)
    composites: List[CompositeDef] = field(default_factory=list)

    def composite_order(self) -> List[CompositeDef]:
        """Parents after the composite rows they reference.

        Order is stable. A cycle is omitted so those parents stay unset.
        """
        by_name: Dict[str, CompositeDef] = {
            composite.metric_name: composite
            for composite in self.composites
            if composite.metric_name
        }
        ordered: List[CompositeDef] = []
        emitted: Set[str] = set()
        pending = list(self.composites)
        while pending:
            ready: List[CompositeDef] = []
            still: List[CompositeDef] = []
            for composite in pending:
                deps = [
                    _ref_base_name(ref)
                    for ref in composite.refs
                    if _ref_base_name(ref) in by_name
                    and _ref_base_name(ref) != composite.metric_name
                ]
                if all(dep in emitted for dep in deps):
                    ready.append(composite)
                else:
                    still.append(composite)
            if not ready:
                names = ", ".join(
                    composite.metric_name or composite.metric_id for composite in still
                )
                console_warning(
                    "metrics",
                    f"Composite cycle in {names}; leaving those metrics unset",
                )
                break
            for composite in ready:
                ordered.append(composite)
                if composite.metric_name:
                    emitted.add(composite.metric_name)
            pending = still
        return ordered


def _metric_column_name(df: pd.DataFrame) -> Optional[str]:
    for candidate in (METRIC_NAME_COLUMN, "metric"):
        if candidate in df.columns:
            return candidate
    return None


def _avg_column_name(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("Avg", "Average", "Value"):
        if candidate in df.columns:
            return candidate
    return None


def _row_metric_name(df: pd.DataFrame, metric_id: object) -> str:
    name_col = _metric_column_name(df)
    if name_col is None or metric_id not in df.index:
        return ""
    row_name = df.at[metric_id, name_col]
    if isinstance(row_name, str):
        return row_name
    return ""


def _resolve_collectable_name(df: pd.DataFrame, ref_name: str) -> str:
    """Resolve a ref to its display name.

    ``_collectable_id`` wins inside this table when the ref matches an id.
    Display keys still resolve for formulas that name the row.
    """
    signed = ref_name.startswith("-")
    base = ref_name[1:].strip() if signed else ref_name
    ids = df.attrs.get(COLLECTABLE_IDS_ATTR, {})
    name_col = _metric_column_name(df)
    if isinstance(ids, dict) and name_col is not None:
        for metric_id, collect_id in ids.items():
            if collect_id != base or metric_id not in df.index:
                continue
            row_name = df.at[metric_id, name_col]
            if isinstance(row_name, str):
                return f"-{row_name}" if signed else row_name
    return ref_name


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
                    # Id is the lookup key within the table when it is present.
                    graph.collectable_ids[collect_id] = row_name
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
                    metric_name=_row_metric_name(df, metric_id),
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
                    metric_name=_row_metric_name(df, metric_id),
                )
            )

    collect_ratio_specs = df.attrs.get(COLLECT_RATIO_SPECS_ATTR)
    if isinstance(collect_ratio_specs, dict):
        for metric_id, parts in collect_ratio_specs.items():
            if not isinstance(parts, dict) or metric_id not in df.index:
                continue
            nums = parts.get("numerator")
            dens = parts.get("denominator")
            if not isinstance(nums, list) or not isinstance(dens, list):
                continue
            graph.collectable_row_names.update(nums)
            graph.collectable_row_names.update(dens)
            graph.composites.append(
                CompositeDef(
                    metric_id=metric_id,
                    kind=CompositeKind.COLLECT_RATIO,
                    refs=list(nums) + list(dens),
                    weight_meta={"numerator": nums, "denominator": dens},
                    metric_name=_row_metric_name(df, metric_id),
                )
            )

    df.attrs[METRIC_EVAL_GRAPH_ATTR] = graph
    return graph


def _eval_built_expr_on_frame(
    built_expr: str,
    frame: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
) -> Union[float, str]:
    if not built_expr:
        return "N/A"
    evaluator = MetricEvaluator(frame, sys_vars, empirical_peaks)
    return evaluator.eval_expression(built_expr)


def per_dispatch_ratio_series(
    built_expr: str,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
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

    values: Dict[Any, float] = {}
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
    dfs: Dict[int, pd.DataFrame],
    dfs_type: Dict[int, str],
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
        by_name: Dict[str, str] = {}
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
) -> Optional[str]:
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


def _collectable_series(
    df: pd.DataFrame,
    ref_name: str,
    avg_col: str,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
    role: str,
) -> Optional[pd.Series]:
    """Per-dispatch values for one collectable ref, honoring a leading minus."""
    negate = ref_name.startswith("-")
    lookup_name = _resolve_collectable_name(df, _ref_base_name(ref_name))
    built = _lookup_collectable_built_avg(df, lookup_name, avg_col)
    if built is None:
        console_warning(f"{role}: collectable '{ref_name}' not found in metric table.")
        return None
    series = per_dispatch_ratio_series(built, raw_pmc_df, sys_vars, empirical_peaks)
    if negate:
        return -series
    return series


def _evaluate_weighted_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
    pass_layout: Optional["PassLayout"] = None,
) -> Union[float, str]:
    avg_col = _avg_column_name(df)
    if avg_col is None:
        return "N/A"

    ratio_series_list: List[pd.Series] = []
    weight_series_list: List[pd.Series] = []

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

        bound_weight = resolve_weight_counter_column(
            weight_counter, ref_name, df, pass_layout
        )
        if not bound_weight:
            console_warning(
                "metrics",
                f"WEIGHTED_AVG: weight {weight_counter!r} for '{ref_name}' "
                "is not on the sub-metric pass; leaving the composite unset",
            )
            return "N/A"
        weight_series = _weight_counter_per_dispatch(bound_weight, raw_pmc_df)
        if weight_series.empty:
            console_warning(
                "metrics",
                f"WEIGHTED_AVG: weight column {bound_weight!r} for '{ref_name}' "
                "is missing; leaving the composite unset",
            )
            return "N/A"
        ratio_series_list.append(
            per_dispatch_ratio_series(built_avg, raw_pmc_df, sys_vars, empirical_peaks)
        )
        weight_series_list.append(weight_series)

    merged = merge_dispatch_weighted_avg(ratio_series_list, weight_series_list)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def _evaluate_collect_sum_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
) -> Union[float, str]:
    avg_col = _avg_column_name(df)
    if avg_col is None:
        return "N/A"

    series_list: List[pd.Series] = []
    for ref_name in composite.refs:
        series = _collectable_series(
            df, ref_name, avg_col, raw_pmc_df, sys_vars, empirical_peaks, "COLLECT_SUM"
        )
        if series is None:
            return "N/A"
        series_list.append(series)

    merged = merge_dispatch_collect_sum(series_list)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def _evaluate_collect_ratio_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
) -> Union[float, str]:
    avg_col = _avg_column_name(df)
    if avg_col is None:
        return "N/A"

    nums = composite.weight_meta.get("numerator")
    dens = composite.weight_meta.get("denominator")
    if not isinstance(nums, list) or not isinstance(dens, list):
        console_warning("COLLECT_RATIO: missing numerator/denominator metadata.")
        return "N/A"

    num_series: List[pd.Series] = []
    den_series: List[pd.Series] = []
    for ref_name in nums:
        series = _collectable_series(
            df,
            ref_name,
            avg_col,
            raw_pmc_df,
            sys_vars,
            empirical_peaks,
            "COLLECT_RATIO numerator",
        )
        if series is None:
            return "N/A"
        num_series.append(series)
    for ref_name in dens:
        series = _collectable_series(
            df,
            ref_name,
            avg_col,
            raw_pmc_df,
            sys_vars,
            empirical_peaks,
            "COLLECT_RATIO denominator",
        )
        if series is None:
            return "N/A"
        den_series.append(series)

    merged = merge_dispatch_collect_ratio(num_series, den_series)
    if pd.isna(merged):
        return "N/A"
    return float(merged)


def apply_composite_metrics(
    dfs: Dict[int, pd.DataFrame],
    dfs_type: Dict[int, str],
    raw_pmc_df: pd.DataFrame,
    sys_vars: Dict[str, Any],
    empirical_peaks: Dict[str, Any],
    pass_layout: Optional["PassLayout"] = None,
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

        ordered = graph.composite_order()
        ordered_ids = {composite.metric_id for composite in ordered}
        for composite in graph.composites:
            in_order = composite.metric_id in ordered_ids
            in_table = composite.metric_id in df.index
            if in_order or not in_table:
                continue
            df.at[composite.metric_id, avg_col] = "N/A"

        for composite in ordered:
            if composite.metric_id not in df.index:
                continue
            if composite.kind is CompositeKind.WEIGHTED_AVG:
                result = _evaluate_weighted_composite(
                    composite,
                    df,
                    raw_pmc_df,
                    sys_vars,
                    empirical_peaks,
                    pass_layout=pass_layout,
                )
            elif composite.kind is CompositeKind.COLLECT_SUM:
                result = _evaluate_collect_sum_composite(
                    composite, df, raw_pmc_df, sys_vars, empirical_peaks
                )
            elif composite.kind is CompositeKind.COLLECT_RATIO:
                result = _evaluate_collect_ratio_composite(
                    composite, df, raw_pmc_df, sys_vars, empirical_peaks
                )
            else:
                continue
            df.at[composite.metric_id, avg_col] = result


def collectable_row_names_from_graph(df: pd.DataFrame) -> Set[str]:
    return build_metric_eval_graph(df).collectable_row_names
