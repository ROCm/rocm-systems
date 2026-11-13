# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Collectables and composite metric evaluation graph (Layer 1.5 analyze path).

Each collectable is evaluated on the one pass that holds its counters.
COLLECT_SUM adds those finished values, including a negative piece.
COLLECT_RATIO divides the pooled numerator sum by the pooled denominator
sum. A piece whose own counters fit no single pass stays unset, and so
does its parent.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from utils.logger import console_warning
from utils.metrics.aggregation import merge_dispatch_collect_ratio
from utils.metrics.expression import (
    is_composite_avg_formula,
)
from utils.metrics.metric_evaluator import MetricEvaluator, counter_table_index
from utils.metrics.noise_clamper import to_noise_clamp

# Names recorded while one analyze run is open, then printed once.
_pending_unset_collectables: Set[str] = set()
_pending_missing_collectables: Set[str] = set()
_collectable_warning_depth = 0

METRIC_EVAL_GRAPH_ATTR = "metric_eval_graph"
COLLECTABLE_EXPR_CACHE_ATTR = "collectable_expr_cache"
COLLECTABLE_IDS_ATTR = "collectable_ids"
COLLECT_SUM_SPECS_ATTR = "collect_sum_specs"
COLLECT_RATIO_SPECS_ATTR = "collect_ratio_specs"
COMPOSITE_FINISH_ATTR = "composite_finish"
METRIC_NAME_COLUMN = "Metric"


class CompositeKind(str, Enum):
    COLLECT_SUM = "collect_sum"
    COLLECT_RATIO = "collect_ratio"


@dataclass
class CompositeDef:
    metric_id: str
    kind: CompositeKind
    refs: List[str]
    weight_meta: Dict[str, Any] = field(default_factory=dict)
    metric_name: str = ""
    finish: Optional[Tuple[str, float]] = None


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


def avg_column_name(df: pd.DataFrame) -> Optional[str]:
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

    finish_by_id = df.attrs.get(COMPOSITE_FINISH_ATTR)
    if isinstance(finish_by_id, dict):
        for composite in graph.composites:
            composite.finish = finish_by_id.get(composite.metric_id)

    df.attrs[METRIC_EVAL_GRAPH_ATTR] = graph
    return graph


def _finish_composite_value(
    value: Union[float, str],
    finish: Optional[Tuple[str, float]],
) -> Union[float, str]:
    """Round or floor a finished parent value, after its pieces combine."""
    if finish is None or not isinstance(value, (int, float)) or math.isnan(value):
        return value
    func, arg = finish
    if func == "ROUND":
        return float(round(value, int(arg)))
    return float(to_noise_clamp(value, arg))


def per_dispatch_series(built_expr: str, evaluator: MetricEvaluator) -> pd.Series:
    """Evaluate built_expr on each dispatch of the evaluator's table.

    Returns the values keyed by Dispatch_ID, without the dispatches that
    evaluate to N/A. A table without Dispatch_ID gives one value for the whole
    table.
    """
    table = evaluator.raw_pmc_df
    if table.empty:
        return pd.Series(dtype=float)
    if "Dispatch_ID" not in table.columns:
        scalar = evaluator.eval_expression(built_expr)
        if scalar == "N/A" or pd.isna(scalar):
            return pd.Series(dtype=float)
        return pd.Series({0: float(scalar)})

    values: Dict[Any, float] = {}
    for dispatch_id, group in table.groupby("Dispatch_ID", dropna=False):
        dispatch_evaluator = MetricEvaluator(
            group,
            _sys_vars_for_rows(evaluator.sys_vars, group.index),
            evaluator.empirical_peaks,
        )
        scalar = dispatch_evaluator.eval_expression(built_expr)
        if scalar == "N/A" or pd.isna(scalar):
            continue
        values[dispatch_id] = float(scalar)
    return pd.Series(values, dtype=float)


def _sys_vars_for_rows(
    sys_vars: Dict[str, Any],
    index: pd.Index,
) -> Dict[str, Any]:
    """Keep Series built-ins on the rows of one dispatch.

    A whole-table Series makes SUM($cycles) the total of every dispatch.
    """
    scoped: Dict[str, Any] = {}
    for key, value in sys_vars.items():
        if isinstance(value, pd.Series):
            scoped[key] = value.reindex(index)
        else:
            scoped[key] = value
    return scoped


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
        avg_col = avg_column_name(df)
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


def begin_collectable_warnings() -> None:
    """Start one analyze run's collectable summary."""
    global _collectable_warning_depth
    if _collectable_warning_depth == 0:
        _pending_unset_collectables.clear()
        _pending_missing_collectables.clear()
    _collectable_warning_depth += 1


def flush_collectable_warnings() -> None:
    """Print one summary for the collectable problems recorded since begin."""
    global _collectable_warning_depth
    if _collectable_warning_depth == 0:
        return
    _collectable_warning_depth -= 1
    if _collectable_warning_depth:
        return
    if _pending_unset_collectables:
        console_warning(
            "metrics",
            f"{len(_pending_unset_collectables)} collectable part(s) were left unset",
        )
    if _pending_missing_collectables:
        console_warning(
            "metrics",
            f"{len(_pending_missing_collectables)} collectable part(s) "
            "were not in the metric table",
        )
    _pending_unset_collectables.clear()
    _pending_missing_collectables.clear()


def _note_collectable_problem(ref_name: str, kind: str) -> None:
    """Record a collectable problem, or print it when no run is open."""
    if _collectable_warning_depth:
        bucket = (
            _pending_unset_collectables
            if kind == "unset"
            else _pending_missing_collectables
        )
        bucket.add(ref_name)
        return
    if kind == "unset":
        console_warning("metrics", f"collectable '{ref_name}' was left unset")
        return
    console_warning(
        "metrics",
        f"collectable '{ref_name}' was not in the metric table",
    )


def _collectable_expr_state(df: pd.DataFrame, metric_name: str, avg_col: str) -> str:
    """``ready``, ``unset`` when the row has no expression, or ``missing``."""
    cached = df.attrs.get(COLLECTABLE_EXPR_CACHE_ATTR, {})
    if isinstance(cached, dict) and metric_name in cached:
        built = cached[metric_name]
        if isinstance(built, str) and built:
            return "ready"
        return "unset"
    name_col = _metric_column_name(df)
    if name_col is None:
        return "missing"
    matches = df[df[name_col] == metric_name]
    if matches.empty:
        return "missing"
    built = matches.iloc[0][avg_col]
    if isinstance(built, str) and built:
        return "ready"
    return "unset"


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


def _collect_piece_dispatch_series(
    df: pd.DataFrame,
    piece_name: str,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> Optional[pd.Series]:
    """Per-dispatch values for one _collect piece, using the COLLECT_SUM lookup.

    The built expression is the same string COLLECT_SUM evaluates. It is run
    per dispatch so a later min or max is the extremum of the combined pieces.
    Returns None when the table has no Dispatch_ID: only scalar piece averages
    exist then, and those averages are not an extent.
    """
    if "Dispatch_ID" not in evaluators[-1].raw_pmc_df.columns:
        return None
    avg_col = avg_column_name(df)
    if avg_col is None:
        return None
    return _collectable_series(df, piece_name, avg_col, evaluators, build_in_vars)


def _collectable_series(
    df: pd.DataFrame,
    ref_name: str,
    avg_col: str,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> Optional[pd.Series]:
    """Per-dispatch values for one collectable ref, honoring a leading minus."""
    negate = ref_name.startswith("-")
    lookup_name = _resolve_collectable_name(df, _ref_base_name(ref_name))
    built = _lookup_collectable_built_avg(df, lookup_name, avg_col)
    if built is None:
        state = _collectable_expr_state(df, lookup_name, avg_col)
        _note_collectable_problem(ref_name, "unset" if state == "unset" else "missing")
        return None
    series = per_dispatch_series(
        built, _piece_evaluator(built, evaluators, build_in_vars)
    )
    if negate:
        return -series
    return series


def _piece_evaluator(
    built: str,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> MetricEvaluator:
    """Return the evaluator of the first pass table that holds every counter of
    built. The merged table's evaluator is last."""
    tables = [evaluator.raw_pmc_df for evaluator in evaluators]
    return evaluators[counter_table_index(built, tables, build_in_vars)]


def _piece_value_on_dispatch_filter(
    df: pd.DataFrame,
    ref_name: str,
    avg_col: str,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
    parent_name: str,
) -> Optional[float]:
    """Evaluate one COLLECT_SUM piece once on the dispatch filter.

    SUM(numerator) / SUM(denominator) stays that pooled ratio. The caller
    adds the piece values.
    """
    negate = ref_name.startswith("-")
    lookup_name = _resolve_collectable_name(df, _ref_base_name(ref_name))
    built = _lookup_collectable_built_avg(df, lookup_name, avg_col)
    if not built:
        state = _collectable_expr_state(df, lookup_name, avg_col)
        _note_collectable_problem(ref_name, "unset" if state == "unset" else "missing")
        return None
    scalar = _piece_evaluator(built, evaluators, build_in_vars).eval_expression(built)
    if isinstance(scalar, (str, bool)):
        value = None
    else:
        try:
            value = float(scalar)
        except (TypeError, ValueError):
            value = None
    if value is None or not math.isfinite(value):
        console_warning(
            "metrics",
            f"COLLECT_SUM: {parent_name} piece '{ref_name}' is missing or its "
            "denominator is zero; leaving the composite unset",
        )
        return None
    if negate:
        return -value
    return value


def _evaluate_collect_sum_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> Union[float, str]:
    avg_col = avg_column_name(df)
    if avg_col is None:
        return "N/A"

    parent_name = composite.metric_name or str(composite.metric_id)
    total = 0.0
    for ref_name in composite.refs:
        value = _piece_value_on_dispatch_filter(
            df,
            ref_name,
            avg_col,
            evaluators,
            build_in_vars,
            parent_name,
        )
        if value is None:
            return "N/A"
        total += value
    return total


def _evaluate_collect_ratio_composite(
    composite: CompositeDef,
    df: pd.DataFrame,
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> Union[float, str]:
    avg_col = avg_column_name(df)
    if avg_col is None:
        return "N/A"

    nums = composite.weight_meta.get("numerator")
    dens = composite.weight_meta.get("denominator")
    if not isinstance(nums, list) or not isinstance(dens, list):
        console_warning(
            "metrics",
            "COLLECT_RATIO: missing numerator/denominator metadata.",
        )
        return "N/A"

    num_series: List[pd.Series] = []
    den_series: List[pd.Series] = []
    for ref_name in nums:
        series = _collectable_series(
            df,
            ref_name,
            avg_col,
            evaluators,
            build_in_vars,
        )
        if series is None:
            return "N/A"
        num_series.append(series)
    for ref_name in dens:
        series = _collectable_series(
            df,
            ref_name,
            avg_col,
            evaluators,
            build_in_vars,
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
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> None:
    """Evaluate composite parents after collectable rows (graph order)."""
    own_batch = _collectable_warning_depth == 0
    if own_batch:
        begin_collectable_warnings()
    try:
        _apply_composite_metrics(dfs, dfs_type, evaluators, build_in_vars)
    finally:
        if own_batch:
            flush_collectable_warnings()


def _apply_composite_metrics(
    dfs: Dict[int, pd.DataFrame],
    dfs_type: Dict[int, str],
    evaluators: List[MetricEvaluator],
    build_in_vars: Dict[str, str],
) -> None:
    for df_id, df in dfs.items():
        if dfs_type.get(df_id) != "metric_table":
            continue
        graph = build_metric_eval_graph(df)
        if not graph.composites:
            continue
        avg_col = avg_column_name(df)
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
            if composite.kind is CompositeKind.COLLECT_SUM:
                result = _evaluate_collect_sum_composite(
                    composite, df, evaluators, build_in_vars
                )
            elif composite.kind is CompositeKind.COLLECT_RATIO:
                result = _evaluate_collect_ratio_composite(
                    composite, df, evaluators, build_in_vars
                )
            else:
                continue
            df.at[composite.metric_id, avg_col] = _finish_composite_value(
                result, composite.finish
            )


def composite_parent_ids(df: pd.DataFrame) -> Set[str]:
    """Metric ids of the COLLECT_SUM and COLLECT_RATIO parents."""
    parent_ids: Set[str] = set()
    for attr in (
        COLLECT_SUM_SPECS_ATTR,
        COLLECT_RATIO_SPECS_ATTR,
    ):
        specs = df.attrs.get(attr)
        if isinstance(specs, dict):
            parent_ids.update(str(metric_id) for metric_id in specs)
    return parent_ids


def collectable_row_names_from_graph(df: pd.DataFrame) -> Set[str]:
    return build_metric_eval_graph(df).collectable_row_names


class CollectPieceLookup:
    """Per-dispatch values of the _collect pieces in one metric table.

    MetricEvaluator calls dispatch_series for each _collect.<piece> that a min
    or max formula reads.
    """

    def __init__(
        self,
        metric_table: pd.DataFrame,
        evaluators: List[MetricEvaluator],
        build_in_vars: Dict[str, str],
    ) -> None:
        self.metric_table = metric_table
        self.evaluators = evaluators
        self.build_in_vars = build_in_vars

    def dispatch_series(self, piece_name: str) -> pd.Series:
        """Return the piece's value on each dispatch. An empty Series leaves the
        min or max that reads it unset."""
        series = _collect_piece_dispatch_series(
            self.metric_table, piece_name, self.evaluators, self.build_in_vars
        )
        return pd.Series(dtype=float) if series is None else series
