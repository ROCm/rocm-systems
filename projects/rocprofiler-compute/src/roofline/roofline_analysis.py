# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Compute roofline results once, persist them, and load renderer inputs."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config
from utils import analysis_orm as orm
from utils import schema
from utils.file_io import kernel_duration_stats
from utils.logger import console_warning
from utils.metrics.aggregation import calc_pct_of_peak
from utils.metrics.expression_evaluator import calc_builtin_vars
from utils.roofline_calc import (
    _METRIC_TO_AI_FIELD,
    CACHE_LEVELS,
    SUPPORTED_DATATYPES,
    PlotPoints,
    RooflineBenchmark,
    datatype_for_compute_column,
    kernel_roof_bounds,
    sanitize_ai_value,
    sanitize_mem_level,
)

_CACHE_COLUMNS = {
    "L0": "l0_cache_data",
    "L1": "l1_cache_data",
    "L2": "l2_cache_data",
    "HBM": "hbm_cache_data",
    "LDS": "lds_cache_data",
}


@dataclass
class RooflineMetricRow:
    """An evaluated metric row, independent of display formatting."""

    metric_id: str
    metric: str
    table_id: int
    unit: str
    value: Optional[float]
    peak: Optional[float]
    percent_of_peak: Optional[float]


@dataclass
class EnvelopeBounds:
    """A datatype's compute cap and the roof at each memory level."""

    limiter: str
    compute_ceiling: Optional[float]
    compute_ceiling_label: Optional[str]
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]]


@dataclass
class KernelRoofline:
    """One kernel's complete roofline metrics, statistics, and envelopes."""

    kernel_name: str
    kernel_rank: int
    dispatch_count: int
    total_duration_ns: float
    percent_runtime: float
    longest_dispatch_ns: float
    metrics: List[RooflineMetricRow]
    level_ai: Dict[str, float]
    performance: float
    envelopes: Dict[str, EnvelopeBounds]


@dataclass
class RooflineResult:
    """All filtered kernels and empirical ceilings for one workload/device."""

    benchmark: RooflineBenchmark
    kernels: List[KernelRoofline]


@dataclass
class RooflineView:
    """Renderer inputs reconstructed exclusively from database query rows."""

    benchmark_peaks: Dict[str, float]
    plot_points: Dict[str, Any]
    limiters: List[Dict[str, Any]]
    kernel_rows: List[Dict[str, Any]]
    metric_rows: List[Dict[str, Any]]

    def tty_tables(
        self,
        arch_config: schema.ArchConfig,
        kernel_order: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Fill architecture templates with queried values, preserving NULLs."""
        kernels = {
            row["kernel_rank"]: row
            for row in self.kernel_rows
            if row["kernel_rank"] is not None
        }
        order = kernel_order or sorted(kernels)
        metrics_by_kernel: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for metric in self.metric_rows:
            metrics_by_kernel.setdefault(metric["kernel_uuid"], {})[
                metric["metric_id"]
            ] = metric
        tables: Dict[int, Dict[str, Any]] = {}
        for rank in order:
            if rank not in kernels:
                continue
            kernel = kernels[rank]
            metrics = metrics_by_kernel.get(kernel["kernel_uuid"], {})
            tables[rank] = {
                "name": kernel["kernel_name"],
                "ai_table": _tty_table(arch_config.dfs.get(401), metrics),
                "calc_table": _tty_table(arch_config.dfs.get(402), metrics),
            }
        return tables


def _tty_table(
    template: Optional[pd.DataFrame],
    metrics: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Copy one display table and replace expression cells with stored values."""
    if template is None:
        return pd.DataFrame()
    table = template.copy().astype(object)
    for metric_id, row in table.iterrows():
        metric = metrics.get(str(metric_id), {})
        if "Value" in table.columns:
            value = metric.get("value")
            table.at[metric_id, "Value"] = "N/A" if value is None else value
        for column in ("Peak (Empirical)", "Peak"):
            if column in table.columns:
                peak = metric.get("peak")
                table.at[metric_id, column] = "N/A" if peak is None else peak
        if "Percent of Peak" in table.columns:
            percentage = metric.get("percent_of_peak")
            table.at[metric_id, "Percent of Peak"] = (
                "" if percentage is None else percentage
            )
    return table


def _optional_number(value: object) -> Optional[float]:
    """Represent invalid evaluations as NULL while retaining finite values."""
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return None
    return numeric if math.isfinite(numeric) else None


def _evaluate_metrics(
    arch_config: schema.ArchConfig,
    pmc_df: pd.DataFrame,
    context: Dict[str, Any],
    evaluate: Callable[..., Any],
) -> List[RooflineMetricRow]:
    """Evaluate both roofline tables with the kernel's built-in variables."""
    tables = [arch_config.dfs.get(table_id, pd.DataFrame()) for table_id in (401, 402)]
    expressions = [
        expression
        for table in tables
        for column in ("Value", "Peak (Empirical)", "Peak")
        if column in table.columns
        for expression in table[column]
        if isinstance(expression, str) and expression and expression != "None"
    ]
    calc_builtin_vars(pmc_df, context, expressions)
    metrics: List[RooflineMetricRow] = []
    for table_id, table in zip((401, 402), tables):
        for metric_id, row in table.iterrows():
            metric = str(row.get("Metric", ""))
            value = _evaluate_cell(evaluate, metric, row.get("Value"), pmc_df, context)
            peak = None
            if table_id == 401:
                peak_expression = row.get("Peak (Empirical)", row.get("Peak"))
                peak = _evaluate_cell(
                    evaluate, metric, peak_expression, pmc_df, context
                )
            percentage = (
                calc_pct_of_peak(value, peak)
                if bool(row.get("Percent of Peak", False))
                else None
            )
            metrics.append(
                RooflineMetricRow(
                    str(metric_id),
                    metric,
                    table_id,
                    str(row.get("Unit", "")),
                    value,
                    peak,
                    percentage,
                )
            )
    return metrics


def _evaluate_cell(
    evaluate: Callable[..., Any],
    metric: str,
    expression: object,
    pmc_df: pd.DataFrame,
    context: Dict[str, Any],
) -> Optional[float]:
    """Evaluate an expression cell or retain an already numeric constant."""
    if isinstance(expression, str):
        return _optional_number(evaluate(metric, expression, pmc_df, context))
    return _optional_number(expression)


def _envelopes(
    level_ai: Dict[str, float],
    performance: float,
    benchmark: RooflineBenchmark,
    gpu_arch: str,
    gpu_model: str,
) -> Dict[str, EnvelopeBounds]:
    """Build FLOP datatype and uncapped memory envelopes on the ALL hierarchy."""
    levels = sanitize_mem_level(["ALL"], gpu_model)
    point_intensities = {
        level: intensity for level, intensity in level_ai.items() if level in levels
    }
    bandwidths = {
        level: bandwidth
        for level, bandwidth in benchmark.bandwidths().items()
        if level in levels
    }
    compute_by_datatype: Dict[str, List[Tuple[str, float]]] = {}
    for column, peak in benchmark.compute_peaks().items():
        mapped = datatype_for_compute_column(
            column, gpu_arch, benchmark.matrix_ops_type
        )
        if mapped is None:
            continue
        datatype, pipe = mapped
        pipe_label = "VALU" if pipe == "VALU" else benchmark.matrix_ops_type
        compute_by_datatype.setdefault(datatype, []).append((
            f"{datatype} {pipe_label}",
            peak,
        ))
    envelopes: Dict[str, EnvelopeBounds] = {}
    datatypes = [
        datatype
        for datatype in SUPPORTED_DATATYPES.get(gpu_arch, {})
        if not datatype.startswith("I")
    ]
    for datatype in [*datatypes, "MEMORY"]:
        peaks = compute_by_datatype.get(datatype, []) if datatype != "MEMORY" else []
        label, ceiling = (
            max(peaks, key=lambda peak: peak[1]) if peaks else ("", math.inf)
        )
        limiter, bounds = kernel_roof_bounds(
            point_intensities, performance, bandwidths, ceiling, label
        )
        envelopes[datatype] = EnvelopeBounds(
            limiter,
            None if math.isinf(ceiling) else ceiling,
            label or None,
            bounds,
        )
    return envelopes


def compute_roofline(
    *,
    sys_info: Mapping[str, Any],
    arch_config: schema.ArchConfig,
    benchmark: RooflineBenchmark,
    pmc_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    evaluate: Callable[..., Any],
) -> RooflineResult:
    """Compute every filtered kernel while ranking against GPU/dispatch stats."""
    points_table = arch_config.dfs.get(402)
    if points_table is None or points_table.empty:
        console_warning(
            "Roofline data is filtered out or not found for "
            f"{sys_info.get('_workload_path', sys_info.get('gpu_arch', ''))}"
        )
        return RooflineResult(benchmark, [])
    if pmc_df.empty:
        return RooflineResult(benchmark, [])
    context = dict(sys_info)
    context.update(benchmark.empirical_peak_vars())
    statistics = kernel_duration_stats(stats_df)
    stats_by_name = {
        row["Kernel_Name"]: (rank, row) for rank, row in statistics.iterrows()
    }
    kernels: List[KernelRoofline] = []
    for name, kernel_pmc in pmc_df.groupby("Kernel_Name"):
        metrics = _evaluate_metrics(arch_config, kernel_pmc, context.copy(), evaluate)
        level_ai = {}
        performance = 0.0
        for metric in metrics:
            if metric.table_id != 402:
                continue
            field = _METRIC_TO_AI_FIELD.get(metric.metric)
            if field:
                level_ai[field[3:].upper()] = sanitize_ai_value(metric.value)
            if metric.metric == "Performance (GFLOPs)":
                performance = sanitize_ai_value(metric.value)
        rank, stats = stats_by_name[name]
        longest = (kernel_pmc.End_Timestamp - kernel_pmc.Start_Timestamp).max()
        kernels.append(
            KernelRoofline(
                str(name),
                int(rank),
                int(stats["Count"]),
                float(stats["Sum(ns)"]),
                float(stats["Percent"]),
                float(longest),
                metrics,
                level_ai,
                performance,
                _envelopes(
                    level_ai,
                    performance,
                    benchmark,
                    sys_info["gpu_arch"],
                    sys_info["gpu_model"],
                ),
            )
        )
    kernels.sort(key=lambda kernel: kernel.longest_dispatch_ns, reverse=True)
    return RooflineResult(benchmark, kernels)


def persist_roofline(
    result: RooflineResult,
    workload_obj: orm.Workload,
    kernel_objs: Dict[str, orm.Kernel],
) -> None:
    """Add empirical ceilings, complete metrics, and per-kernel roof envelopes."""
    session = orm.Database.get_session()
    benchmark = result.benchmark
    gpu_arch = workload_obj.sys_info_extdata["gpu_arch"]
    for level, bandwidth in benchmark.bandwidths().items():
        session.add(
            orm.RooflineBandwidthCeiling(
                workload=workload_obj,
                device_id=benchmark.device_id,
                mem_level=level,
                benchmark_column=f"{level}Bw",
                bandwidth=bandwidth,
            )
        )
    for column, peak in benchmark.compute_peaks().items():
        mapped = datatype_for_compute_column(
            column, gpu_arch, benchmark.matrix_ops_type
        )
        datatype, pipe = mapped if mapped else (None, None)
        session.add(
            orm.RooflineComputeCeiling(
                workload=workload_obj,
                device_id=benchmark.device_id,
                benchmark_column=column,
                datatype=datatype,
                pipe=pipe,
                pipe_label=("VALU" if pipe == "VALU" else benchmark.matrix_ops_type)
                if pipe
                else None,
                peak=peak,
                unit="GFLOP/s" if column.endswith("Flops") else "GOP/s",
            )
        )
    for kernel_result in result.kernels:
        kernel = kernel_objs.get(kernel_result.kernel_name)
        if kernel is None:
            console_warning(
                f"Kernel {kernel_result.kernel_name} from roofline data not found "
                "in dispatch data. Skipping roofline entry."
            )
            continue
        _persist_kernel(kernel_result, kernel)


def _persist_kernel(result: KernelRoofline, kernel: orm.Kernel) -> None:
    """Persist one kernel without a workload-level aggregate."""
    session = orm.Database.get_session()
    session.add(
        orm.KernelRooflineData(
            kernel=kernel,
            total_flops=result.performance,
            kernel_rank=result.kernel_rank,
            dispatch_count=result.dispatch_count,
            total_duration_ns=result.total_duration_ns,
            percent_runtime=result.percent_runtime,
            **{
                column: result.level_ai.get(level)
                for level, column in _CACHE_COLUMNS.items()
            },
        )
    )
    for metric in result.metrics:
        session.add(orm.KernelRooflineMetric(kernel=kernel, **metric.__dict__))
    for envelope, bounds in result.envelopes.items():
        session.add(
            orm.KernelRooflineLimiter(
                kernel=kernel,
                envelope=envelope,
                limiter=bounds.limiter,
                compute_ceiling=bounds.compute_ceiling,
                compute_ceiling_label=bounds.compute_ceiling_label,
            )
        )
        for level, (roof, percentage) in bounds.bounds.items():
            session.add(
                orm.KernelRooflinePoint(
                    kernel=kernel,
                    envelope=envelope,
                    mem_level=level,
                    arithmetic_intensity=result.level_ai[level],
                    performance=result.performance,
                    roof_performance=roof,
                    percent_of_roof=percentage,
                )
            )


def load_roofline_view(
    workload_id: int, device_id: int, time_unit: str
) -> RooflineView:
    """Rebuild plot points and display values using only database queries."""
    ceilings = orm.Database.get_roofline_ceilings(workload_id, device_id)
    rows = orm.Database.get_kernel_roofline_rows(workload_id)
    metrics = orm.Database.get_kernel_roofline_metrics(workload_id)
    peaks = {row["benchmark_column"]: row["value"] for row in ceilings}
    plot = PlotPoints.empty()
    plot.timeUnit = time_unit
    kernels: Dict[int, Dict[str, Any]] = {}
    limiters: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in rows:
        kernels.setdefault(row["kernel_uuid"], row)
        if row["envelope"] is not None:
            limiters[(row["kernel_uuid"], row["envelope"])] = {
                key: row[key]
                for key in (
                    "kernel_uuid",
                    "kernel_name",
                    "envelope",
                    "limiter",
                    "compute_ceiling",
                    "compute_ceiling_label",
                )
            }
    ordered = sorted(
        kernels.values(),
        key=lambda row: (
            row["kernel_rank"] is None,
            row["kernel_rank"] or 0,
            row["kernel_uuid"],
        ),
    )
    for kernel in ordered:
        performance = sanitize_ai_value(kernel.get("total_flops"))
        if performance <= 0:
            continue
        for field in CACHE_LEVELS:
            level = field[3:].upper()
            getattr(plot, field)[0].append(
                sanitize_ai_value(kernel.get(_CACHE_COLUMNS[level]))
            )
            getattr(plot, field)[1].append(performance)
        plot.kernelNames.append(kernel["kernel_name"])
        plot.counts.append(kernel["dispatch_count"])
        plot.totalTime.append(
            kernel["total_duration_ns"] / config.TIME_UNITS[time_unit]
        )
        plot.pctRuntime.append(kernel["percent_runtime"])
    return RooflineView(peaks, plot.__dict__, list(limiters.values()), ordered, metrics)
