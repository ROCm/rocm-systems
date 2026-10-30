# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Evaluate aggregate and per-kernel data for the HTML memory chart.

Two inputs feed the same views. build_chart_views evaluates the counters of
an in-memory workload. build_chart_views_from_db fills the same tables from
the workload's stored metric rows.
"""

import copy
import numbers
from argparse import Namespace
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import pandas as pd

from membw_analysis.engine import evaluate_membw_dfs, load_membw_spec
from membw_analysis.models import MEMBW_TABLE_IDS, MemBwAnalysisResult, TreeSpec
from membw_analysis.summary import (
    ACTIVE_FALLBACK_TEXT,
    active_stall_leaves,
    has_active_nodes,
    status_text,
)
from memory_chart.html.db_rows import MetricValues, WorkloadChartRows
from memory_chart.html.diagram import SlotSpec, diagram_payload, slot_specs
from memory_chart.loader import Layout, Layouts, layout_from_stored
from memory_chart.render import progress_bar
from memory_chart.units import display_unit, format_value, numeric, panel_units
from utils import parser, schema, tty
from utils.metrics.base_units import from_base_unit_value
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import top_level_round_digits
from utils.utils_common import SUPPORTED_FIELD


@dataclass(frozen=True)
class SlotValue:
    """One diagram slot with its terminal-equivalent display value."""

    slot_id: str
    metric: str
    text: str
    numeric: Optional[float]
    bar: Optional[str]
    unit_label: str = ""


@dataclass(frozen=True)
class TableView:
    """One processed panel table with immutable rows and column order."""

    table_id: int
    title: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class StallAnnotation:
    """A chart block's active terminal bottleneck."""

    block_id: str
    label: str
    value: str


@dataclass(frozen=True)
class MembwDetail:
    """Availability, annotations, and guidance for one evaluated view."""

    availability: str
    status: Optional[str]
    annotations: Tuple[StallAnnotation, ...]
    guidance_blocks: Tuple[str, ...]


@dataclass(frozen=True)
class ChartView:
    """All data displayed when aggregate or one kernel is selected."""

    key: str
    slots: Tuple[SlotValue, ...]
    tables: Tuple[TableView, ...]
    membw: Optional[MembwDetail]


@dataclass(frozen=True)
class KernelEntry:
    """A kernel option in the existing duration-sorted top table."""

    index: int
    name: str


@dataclass(frozen=True)
class ChartScope:
    """The GPU, dispatch and kernel filters that scope one workload's chart."""

    gpu_ids: List[Any] = field(default_factory=list)
    dispatch_ids: List[Any] = field(default_factory=list)
    kernel_ids: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class ChartViewSet:
    """Diagram topology and all selectable chart data for one workload."""

    arch: str
    diagram: Dict[str, object]
    aggregate: ChartView
    kernels: Tuple[KernelEntry, ...]
    kernel_views: Tuple[ChartView, ...]
    initial_kernel: Optional[int]


@dataclass(frozen=True)
class _ViewContext:
    """Shared chart templates and architecture data for evaluating each view."""

    configs: Tuple[Tuple[int, str, Dict[str, Any]], ...]
    specs: Tuple[SlotSpec, ...]
    templates: Dict[int, pd.DataFrame]
    table_types: Dict[int, str]
    expressions: Dict[int, List[str]]
    args: Namespace
    tree_spec: Optional[TreeSpec]
    arch: str
    layout: Layout
    units: Dict[str, str]


# Gives the evaluated tables of one view: None for the aggregate, else a kernel index
ViewTables = Callable[[Optional[int]], Dict[int, pd.DataFrame]]

# Columns eval_metric evaluates as expressions; Percent of Peak is derived instead
_EXPRESSION_COLUMNS = frozenset(SUPPORTED_FIELD) - {"Percent of Peak"}


def build_chart_views(
    workload: schema.Workload,
    arch_config: schema.ArchConfig,
    args: Namespace,
) -> ChartViewSet:
    """Evaluate chart templates once for all dispatches and once per kernel."""
    arch = str(workload.sys_info.iloc[0]["gpu_arch"])
    layout = Layouts.for_arch(arch)

    if layout is None:
        raise ValueError(f"No memory chart layout for {arch!r}")

    filtered_pmc = parser.apply_non_kernel_filters(workload)
    kernels = _kernel_entries(
        workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID]["Kernel_Name"]
    )
    context = _make_context(arch, layout, arch_config, args, set(workload.dfs))
    debug = bool(getattr(args, "debug", False))

    def view_tables(index: Optional[int]) -> Dict[int, pd.DataFrame]:
        pmc = (
            filtered_pmc
            if index is None
            else filtered_pmc.loc[filtered_pmc["Kernel_Name"] == kernels[index].name]
        )
        return _evaluated_tables(context, workload, pmc, debug)

    return _assemble_view_set(
        context, kernels, list(workload.filter_kernel_ids), view_tables
    )


def build_chart_views_from_db(
    rows: WorkloadChartRows,
    arch_config: schema.ArchConfig,
    args: Namespace,
    scope: ChartScope,
) -> ChartViewSet:
    """Fill the same chart views from one workload's stored metric rows."""
    if rows.layout_json is None:
        raise ValueError(f"No memory chart layout stored for {rows.arch!r}")
    layout = layout_from_stored(rows.layout_json)
    kernels = _kernel_entries([kernel.name for kernel in rows.kernels])
    context = _make_context(
        rows.arch, layout, arch_config, args, stored_chart_tables(rows, arch_config)
    )

    def view_tables(index: Optional[int]) -> Dict[int, pd.DataFrame]:
        values = rows.aggregate if index is None else rows.kernels[index].values
        return _tables_from_rows(context, values)

    return _assemble_view_set(context, kernels, scope.kernel_ids, view_tables)


def stored_chart_tables(
    rows: WorkloadChartRows, arch_config: schema.ArchConfig
) -> Set[int]:
    """The configured tables that have at least one stored metric row.

    The database has no table key, so a table counts as loaded when one of its
    metric ids has a row, as the counters path counts a table in workload.dfs.
    """
    stored: Set[str] = set(rows.aggregate)
    stored.update(metric_id for kernel in rows.kernels for metric_id in kernel.values)

    return {
        table_id
        for table_id, template in arch_config.dfs.items()
        if any(str(metric_id) in stored for metric_id in template.index)
    }


def chart_metric_ids(arch_config: schema.ArchConfig) -> List[str]:
    """The metric ids of every table the memory chart reads."""
    ids: List[str] = []
    for table_id, _, _ in _available_configs(arch_config, set(arch_config.dfs)):
        ids.extend(str(metric_id) for metric_id in arch_config.dfs[table_id].index)
    return ids


def _kernel_entries(names: Iterable[object]) -> Tuple[KernelEntry, ...]:
    """Number kernels in top-stats order."""
    return tuple(KernelEntry(index, str(name)) for index, name in enumerate(names))


def _make_context(
    arch: str,
    layout: Layout,
    arch_config: schema.ArchConfig,
    args: Namespace,
    present_tables: Container[int],
) -> _ViewContext:
    """Collect the templates, slots and membw spec shared by every view."""
    configs = _available_configs(arch_config, present_tables)
    template_ids = {table_id for table_id, _, _ in configs}

    spec = (
        load_membw_spec(arch)
        if any(table_id in template_ids for table_id in MEMBW_TABLE_IDS)
        else None
    )
    units = panel_units(arch_config.panel_configs[300])

    return _ViewContext(
        configs=configs,
        templates={table_id: arch_config.dfs[table_id] for table_id in template_ids},
        table_types={
            table_id: arch_config.dfs_type[table_id] for table_id in template_ids
        },
        expressions={
            table_id: arch_config.dfs_expressions.get(table_id, [])
            for table_id in template_ids
        },
        args=args,
        tree_spec=spec,
        arch=arch,
        layout=layout,
        units=units,
        specs=slot_specs(layout, units),
    )


def _assemble_view_set(
    context: _ViewContext,
    kernels: Tuple[KernelEntry, ...],
    filter_kernel_ids: List[int],
    view_tables: ViewTables,
) -> ChartViewSet:
    """Build the aggregate and per-kernel views from one table source."""
    aggregate = _view("aggregate", view_tables(None), context)
    kernel_views = tuple(
        _view(f"kernel-{entry.index}", view_tables(entry.index), context)
        for entry in kernels
    )

    return ChartViewSet(
        arch=context.arch,
        diagram=diagram_payload(context.layout, context.units),
        aggregate=aggregate,
        kernels=kernels,
        kernel_views=kernel_views,
        initial_kernel=_initial_kernel(filter_kernel_ids, kernels),
    )


def _evaluated_tables(
    context: _ViewContext,
    workload: schema.Workload,
    pmc: pd.DataFrame,
    debug: bool,
) -> Dict[int, pd.DataFrame]:
    """Evaluate the chart templates on one view's counter frame."""
    dfs = copy.deepcopy(context.templates)
    if dfs:
        eval_metric(
            dfs,
            context.table_types,
            context.expressions,
            workload.sys_info.iloc[0],
            workload.roofline_peaks,
            pmc,
            debug,
        )
    return dfs


def _tables_from_rows(
    context: _ViewContext,
    values: MetricValues,
) -> Dict[int, pd.DataFrame]:
    """Fill copies of the chart templates with one view's stored values.

    Each cell gets what eval_metric would have written: an empty string where
    the template has no expression, and otherwise the stored value converted
    to its config unit and rounded to the digits of its to_round expression.
    """
    dfs = copy.deepcopy(context.templates)
    for table_id, df in dfs.items():
        template = context.templates[table_id]
        for metric_id in df.index:
            row_values = values.get(str(metric_id), {})
            unit = template.at[metric_id, "Unit"] if "Unit" in template else None
            for column in df.columns:
                if column in _EXPRESSION_COLUMNS:
                    expression = template.at[metric_id, column]
                    if not expression:
                        df.at[metric_id, column] = ""
                        continue
                    df.at[metric_id, column] = _stored_cell_value(
                        row_values.get(column), unit, column, expression
                    )
                elif column == "Percent of Peak":
                    # Stored from the unrounded Value, so its last digit can differ
                    # from the terminal's, which derives it from the rounded Value.
                    stored = row_values.get(column)
                    df.at[metric_id, column] = "" if stored is None else stored
    return dfs


def _stored_cell_value(
    stored: object,
    unit: object,
    column: str,
    expression: object,
) -> object:
    """Return one stored value as the terminal computes it for display."""
    value = from_base_unit_value(
        unit if isinstance(unit, str) else None, column, stored
    )
    digits = top_level_round_digits(expression)
    if digits is not None and isinstance(value, numbers.Real):
        return round(value, digits)
    return value


def _view(key: str, dfs: Dict[int, pd.DataFrame], context: _ViewContext) -> ChartView:
    """Evaluate chart tables, slots, and bandwidth details for one data view."""
    membw_result = _evaluate_membw(dfs, context.tree_spec, context.arch)
    tables, metrics = _process_tables(dfs, context.configs, context.args, context.specs)

    return ChartView(
        key=key,
        slots=tuple(
            _slot_value(item, metrics.get(item.metric)) for item in context.specs
        ),
        tables=tables,
        membw=_membw_detail(membw_result, context.layout),
    )


def _available_configs(
    arch_config: schema.ArchConfig, present_tables: Container[int]
) -> Tuple[Tuple[int, str, Dict[str, Any]], ...]:
    """Return configured metric tables available to the HTML memory chart."""
    configs: List[Tuple[int, str, Dict[str, Any]]] = []
    allowed = set(MEMBW_TABLE_IDS)

    for panel_id, panel in arch_config.panel_configs.items():
        if panel_id != 300 and panel_id != 3000:
            continue

        for source in panel["data source"]:
            for table_type, config in source.items():
                table_id = config["id"]
                if not (300 < table_id < 400 or table_id in allowed):
                    continue
                if table_id not in arch_config.dfs or table_id not in present_tables:
                    continue
                if table_type != "metric_table":
                    continue

                configs.append((table_id, table_type, config))

    return tuple(configs)


def _process_tables(
    dfs: Dict[int, pd.DataFrame],
    configs: Tuple[Tuple[int, str, Dict[str, Any]], ...],
    args: Namespace,
    specs: Tuple[SlotSpec, ...],
) -> Tuple[Tuple[TableView, ...], Dict[str, Any]]:
    """Format configured tables and collect displayed chart metric values."""
    runs = {"chart": SimpleNamespace(dfs=dfs)}
    comparable = parser.build_comparable_columns(args.time_unit)
    hidden = tty.resolve_hidden_columns(args)

    tables: List[TableView] = []
    metrics: Dict[str, Any] = {}
    spec_by_metric = {spec.metric: spec for spec in specs}

    for table_id, table_type, config in configs:
        processed = tty.process_table_data(
            args, runs, config, table_type, comparable, hidden
        )

        if processed.empty:
            continue

        if table_id in MEMBW_TABLE_IDS and "Unit" in processed.columns:
            processed = processed.assign(
                Unit=processed["Unit"].replace({"Percent": "%"})
            )

        rows = tuple(tuple(row) for row in processed.itertuples(index=False, name=None))
        if 300 < table_id < 400 and {"Metric", "Value"} <= set(processed.columns):
            rows, table_metrics = _format_chart_metric_rows(processed, spec_by_metric)
            metrics.update(table_metrics)

        tables.append(
            TableView(
                table_id=table_id,
                title=str(config["title"]),
                columns=tuple(str(column) for column in processed.columns),
                rows=rows,
            )
        )

    return tuple(tables), metrics


def _format_chart_metric_rows(
    table: pd.DataFrame, specs: Dict[str, SlotSpec]
) -> Tuple[Tuple[Tuple[Any, ...], ...], Dict[str, Any]]:
    """Format diagram metrics, retaining table units when chart suffixes are absent."""
    metric_index = table.columns.get_loc("Metric")
    value_index = table.columns.get_loc("Value")
    unit_index = table.columns.get_loc("Unit") if "Unit" in table.columns else None

    rows: List[Tuple[Any, ...]] = []
    metrics: Dict[str, Any] = {}

    for source_row in table.itertuples(index=False, name=None):
        row = list(source_row)
        metric = str(row[metric_index])
        raw_value = row[value_index]
        metrics[metric] = raw_value

        spec = specs.get(metric)
        if spec is not None:
            formatted = format_value(raw_value, spec.unit)

            if unit_index is not None:
                suffix = spec.unit.suffix
                row[value_index] = (
                    formatted[: -len(suffix)]
                    if suffix and formatted.endswith(suffix)
                    else formatted
                )
                row[unit_index] = suffix.strip() or row[unit_index]
            else:
                row[value_index] = formatted

        rows.append(tuple(row))

    return tuple(rows), metrics


def _slot_value(spec: SlotSpec, raw: Any) -> SlotValue:  # noqa: ANN401
    """Format one diagram slot and derive its numeric progress value."""
    number = numeric(raw)

    return SlotValue(
        slot_id=spec.slot_id,
        metric=spec.metric,
        text=format_value(raw, spec.unit),
        numeric=number,
        bar=progress_bar(number) if spec.bar else None,
    )


def _evaluate_membw(
    dfs: Dict[int, pd.DataFrame], spec: Optional[TreeSpec], arch: str
) -> Optional[MemBwAnalysisResult]:
    """Evaluate bandwidth bottlenecks when the architecture has a tree spec."""
    if spec is None:
        return None

    return evaluate_membw_dfs(dfs, spec, arch)


def _membw_detail(
    result: Optional[MemBwAnalysisResult], layout: Layout
) -> Optional[MembwDetail]:
    """Build display status, annotations, and guidance from an analysis result."""
    if result is None:
        return None

    annotations = tuple(
        StallAnnotation(
            block_id=block.id,
            label=node.label,
            value=format_value(
                node.supporting[0].value if node.supporting else None,
                display_unit("Percent"),
            ),
        )
        for block in layout.blocks()
        for node in active_stall_leaves(result, block.stall_level)
    )

    active = has_active_nodes(result.nodes)
    status = status_text(result) if not active else None
    if active and not result.guidance_blocks:
        status = ACTIVE_FALLBACK_TEXT

    return MembwDetail(
        availability=result.availability,
        status=status,
        annotations=annotations,
        guidance_blocks=result.guidance_blocks,
    )


def _initial_kernel(
    filter_ids: List[int], kernels: Tuple[KernelEntry, ...]
) -> Optional[int]:
    """Return the uniquely selected kernel index, if the filters identify one."""
    selected = {kernel_id for kernel_id in filter_ids if 0 <= kernel_id < len(kernels)}

    return next(iter(selected)) if len(selected) == 1 else None
