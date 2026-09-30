# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Evaluate aggregate and per-kernel data for the HTML memory chart."""

import copy
from argparse import Namespace
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from membw_analysis.engine import evaluate_membw_dfs, load_membw_spec
from membw_analysis.models import MEMBW_TABLE_IDS, MemBwAnalysisResult, TreeSpec
from membw_analysis.summary import (
    ACTIVE_FALLBACK_TEXT,
    STALL_BLOCK_LEVELS,
    active_stall_leaves,
    has_active_nodes,
    status_text,
)
from memory_chart.html.diagram import SlotSpec, diagram_payload, slot_specs
from memory_chart.loader import load_layout
from memory_chart.mem_chart import (
    format_scientific,
    format_value,
    progress_bar,
    safe_float,
)
from utils import parser, schema, tty
from utils.metrics.evaluation_pipeline import eval_metric


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
    workload: schema.Workload
    configs: Tuple[Tuple[int, str, Dict[str, Any]], ...]
    specs: Tuple[SlotSpec, ...]
    templates: Dict[int, pd.DataFrame]
    table_types: Dict[int, str]
    expressions: Dict[int, List[str]]
    args: Namespace
    tree_spec: Optional[TreeSpec]
    arch: str


def build_chart_views(
    workload: schema.Workload,
    arch_config: schema.ArchConfig,
    args: Namespace,
) -> ChartViewSet:
    """Evaluate chart templates once for all dispatches and once per kernel."""
    arch = str(workload.sys_info.iloc[0]["gpu_arch"])
    layout = load_layout(arch)
    specs = slot_specs(layout)
    configs = _available_configs(workload, arch_config)
    template_ids = {table_id for table_id, _, _ in configs}
    templates = {table_id: arch_config.dfs[table_id] for table_id in template_ids}
    table_types = {
        table_id: arch_config.dfs_type[table_id] for table_id in template_ids
    }
    expressions = {
        table_id: arch_config.dfs_expressions.get(table_id, [])
        for table_id in template_ids
    }
    spec = (
        load_membw_spec(arch)
        if any(table_id in template_ids for table_id in MEMBW_TABLE_IDS)
        else None
    )
    filtered_pmc = parser.apply_non_kernel_filters(workload)
    kernels = tuple(
        KernelEntry(index, str(name))
        for index, name in enumerate(workload.dfs[1]["Kernel_Name"])
    )

    context = _ViewContext(
        workload,
        configs,
        specs,
        templates,
        table_types,
        expressions,
        args,
        spec,
        arch,
    )
    aggregate = _evaluate_view("aggregate", filtered_pmc, context)
    kernel_views = tuple(
        _evaluate_view(
            f"kernel-{entry.index}",
            filtered_pmc.loc[filtered_pmc["Kernel_Name"] == entry.name],
            context,
        )
        for entry in kernels
    )
    return ChartViewSet(
        arch=arch,
        diagram=diagram_payload(layout),
        aggregate=aggregate,
        kernels=kernels,
        kernel_views=kernel_views,
        initial_kernel=_initial_kernel(workload.filter_kernel_ids, kernels),
    )


def _evaluate_view(key: str, pmc: pd.DataFrame, context: _ViewContext) -> ChartView:
    dfs = copy.deepcopy(context.templates)
    if dfs:
        eval_metric(
            dfs,
            context.table_types,
            context.expressions,
            context.workload.sys_info.iloc[0],
            context.workload.roofline_peaks,
            pmc,
            bool(getattr(context.args, "debug", False)),
        )
    membw_result = _evaluate_membw(dfs, context.tree_spec, context.arch)
    tables, metrics = _process_tables(dfs, context.configs, context.args)
    return ChartView(
        key=key,
        slots=tuple(
            _slot_value(item, metrics.get(item.metric)) for item in context.specs
        ),
        tables=tables,
        membw=_membw_detail(membw_result),
    )


def _available_configs(
    workload: schema.Workload, arch_config: schema.ArchConfig
) -> Tuple[Tuple[int, str, Dict[str, Any]], ...]:
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
                if table_id not in arch_config.dfs or table_id not in workload.dfs:
                    continue
                if table_type != "metric_table":
                    continue
                configs.append((table_id, table_type, config))
    return tuple(configs)


def _process_tables(
    dfs: Dict[int, pd.DataFrame],
    configs: Tuple[Tuple[int, str, Dict[str, Any]], ...],
    args: Namespace,
) -> Tuple[Tuple[TableView, ...], Dict[str, Any]]:
    runs = {"chart": SimpleNamespace(dfs=dfs)}
    comparable = parser.build_comparable_columns(args.time_unit)
    hidden = tty.resolve_hidden_columns(args)
    tables: List[TableView] = []
    metrics: Dict[str, Any] = {}
    for table_id, table_type, config in configs:
        processed = tty.process_table_data(
            args, runs, config, table_type, comparable, hidden
        )
        if processed.empty:
            continue
        tables.append(
            TableView(
                table_id=table_id,
                title=str(config["title"]),
                columns=tuple(str(column) for column in processed.columns),
                rows=tuple(tuple(row) for row in processed.itertuples(index=False)),
            )
        )
        if 300 < table_id < 400 and {"Metric", "Value"} <= set(processed.columns):
            metrics.update(zip(processed["Metric"], processed["Value"]))
    return tuple(tables), metrics


def _slot_value(spec: SlotSpec, raw: Any) -> SlotValue:  # noqa: ANN401
    numeric = safe_float(raw)
    unit_label = ""
    if spec.cu_block and spec.metric in {"Scratch Allocation", "LDS Allocation"}:
        numeric = numeric / 1024 if numeric is not None else None
        text = format_value(numeric, " KB")
    elif spec.cu_block and spec.metric == "Wavefront Occupancy":
        text = format_value(raw)
        unit_label = "waves/CU"
    elif spec.on_arrow and spec.unit != "Bytes/s":
        text = format_scientific(raw)
    else:
        text = format_value(raw, spec.unit)
    return SlotValue(
        slot_id=spec.slot_id,
        metric=spec.metric,
        text=text,
        numeric=numeric,
        bar=progress_bar(numeric) if spec.bar else None,
        unit_label=unit_label,
    )


def _evaluate_membw(
    dfs: Dict[int, pd.DataFrame], spec: Optional[TreeSpec], arch: str
) -> Optional[MemBwAnalysisResult]:
    if spec is None:
        return None
    return evaluate_membw_dfs(dfs, spec, arch)


def _membw_detail(result: Optional[MemBwAnalysisResult]) -> Optional[MembwDetail]:
    if result is None:
        return None
    annotations = tuple(
        StallAnnotation(
            block_id=block_id,
            label=f"[!] {node.label}",
            value=format_value(
                node.supporting[0].value if node.supporting else None, "%"
            ),
        )
        for block_id in STALL_BLOCK_LEVELS
        for node in active_stall_leaves(result, block_id)
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
    filter_ids: List[Union[int, str]], kernels: Tuple[KernelEntry, ...]
) -> Optional[int]:
    if not filter_ids:
        return None
    selected: Set[int] = set()
    for kernel_id in filter_ids:
        if isinstance(kernel_id, int) and 0 <= kernel_id < len(kernels):
            selected.add(kernel_id)
        elif isinstance(kernel_id, str):
            selected.update(entry.index for entry in kernels if entry.name == kernel_id)
    return next(iter(selected)) if len(selected) == 1 else None
