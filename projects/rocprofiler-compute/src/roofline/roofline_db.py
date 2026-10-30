# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Render a workload's interactive roofline HTML from the analysis database rows.

The ceilings are the workload's roofline ceiling rows. The kernel points are the
Roofline Plot Points metric rows, stored per kernel in base units. Both feed the
same Plotly builder the CLI roofline uses, so the limiter and percent of roofline
come from those rows through the same rule.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

import utils.analysis_orm as orm
from roofline.roofline_main import Roofline
from utils.logger import console_warning
from utils.metrics.base_units import SCALED_UNITS
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import (
    CACHE_LEVELS,
    METRIC_TO_AI_FIELD,
    PERFORMANCE_METRIC,
    ROOFLINE_DEVICE_ID,
    Ceiling,
    PlotPoints,
    benchmark_from_ceilings,
)
from utils.specs import MachineSpecs

# The sub-table title the Roofline Plot Points metric rows are stored under.
PLOT_POINTS_SUB_TABLE = "Roofline Plot Points"
PLOT_POINTS_VALUE_NAME = "Value"
# Kernel durations are stored in ns, the unit the top-stats table sums in by default.
TIME_UNIT = "ns"
_AI_METRIC_BY_FIELD = {field: metric for metric, field in METRIC_TO_AI_FIELD.items()}
_AI_METRIC_PREFIX = "AI "


def plot_points_from_rows(
    rows: "Sequence[KernelPlotRow]",
    total_duration_ns: Optional[int],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build the roofline plot points from kernel rows, and the notes they leave.

    A kernel whose performance is NULL is left off the plot, and so is a memory
    level whose arithmetic intensity is NULL. Each is named in a note. A level the
    workload does not measure is plotted as zero, as the CLI roofline plots it.
    Returns the plot points as the Plotly builder reads them, and the notes.
    """
    plot_points = PlotPoints.empty()
    plot_points.timeUnit = TIME_UNIT
    plot_points.totalAppTime = (
        float(total_duration_ns) if total_duration_ns is not None else None
    )
    if rows:
        plot_points.totalAppDispatches = float(sum(row.dispatch_count for row in rows))

    notes: List[str] = []
    by_duration = sorted(rows, key=lambda row: (-row.duration_ns, row.kernel_name))
    for row in by_duration:
        performance = _gflops(row.metrics.get(PERFORMANCE_METRIC))
        if performance is None:
            notes.append(
                f"Kernel {row.kernel_name} is left off the plot "
                "because its performance is N/A."
            )
            continue
        if performance <= 0:
            continue

        ai_by_field, missing_levels = _kernel_intensities(row)
        if missing_levels:
            notes.append(
                f"Kernel {row.kernel_name} has N/A arithmetic intensity at "
                f"{', '.join(missing_levels)}, so those points are left off the plot."
            )
        _append_kernel_point(
            plot_points, row, performance, ai_by_field, total_duration_ns
        )
    return plot_points.__dict__, notes


def load_ceilings(session: Session, workload_id: int) -> List[Ceiling]:
    """Read the workload's roofline ceiling rows."""
    rows = (
        session
        .query(orm.RooflineCeiling)
        .filter(orm.RooflineCeiling.workload_id == workload_id)
        .all()
    )
    return [
        Ceiling(
            kind=row.kind,
            memory_level=row.memory_level,
            data_type=row.data_type,
            pipe=row.pipe,
            unit=row.unit,
            value=row.value,
        )
        for row in rows
    ]


def load_kernel_plot_rows(session: Session, workload_id: int) -> List["KernelPlotRow"]:
    """Read every kernel's Roofline Plot Points rows and its dispatch totals."""
    metric_names = [
        name
        for (name,) in session
        .query(orm.MetricDefinition.name)
        .filter(
            orm.MetricDefinition.workload_id == workload_id,
            orm.MetricDefinition.sub_table_name == PLOT_POINTS_SUB_TABLE,
        )
        .distinct()
    ]
    values = {
        (kernel_name, metric_name): value
        for kernel_name, metric_name, value in session
        .query(
            orm.Kernel.kernel_name,
            orm.MetricDefinition.name,
            orm.KernelMetricValue.value,
        )
        .join(
            orm.KernelMetricValue,
            orm.KernelMetricValue.kernel_uuid == orm.Kernel.kernel_uuid,
        )
        .join(
            orm.MetricDefinition,
            orm.KernelMetricValue.metric_uuid == orm.MetricDefinition.metric_uuid,
        )
        .filter(
            orm.Kernel.workload_id == workload_id,
            orm.MetricDefinition.sub_table_name == PLOT_POINTS_SUB_TABLE,
            orm.KernelMetricValue.value_name == PLOT_POINTS_VALUE_NAME,
        )
    }
    dispatch_totals = (
        session
        .query(
            orm.Kernel.kernel_name,
            func.count(orm.Dispatch.dispatch_uuid),
            func.sum(orm.Dispatch.end_timestamp - orm.Dispatch.start_timestamp),
        )
        .join(orm.Dispatch, orm.Dispatch.kernel_uuid == orm.Kernel.kernel_uuid)
        .filter(orm.Kernel.workload_id == workload_id)
        .group_by(orm.Kernel.kernel_name)
        .all()
    )
    return [
        KernelPlotRow(
            kernel_name=kernel_name,
            metrics={name: values.get((kernel_name, name)) for name in metric_names},
            dispatch_count=int(dispatch_count),
            duration_ns=int(duration_ns or 0),
        )
        for kernel_name, dispatch_count, duration_ns in dispatch_totals
    ]


def write_roofline_html(
    session: Session,
    workload_id: int,
    mspec: MachineSpecs,
    output_path: Path,
) -> Optional[Path]:
    """Write one workload's roofline HTML to output_path, from its database rows.

    Returns output_path. Returns None, and writes nothing, when the workload has no
    ceiling rows or no datatype has usable benchmark data. Each note about a kernel
    left off the plot is printed as a warning.
    """
    ceilings = load_ceilings(session, workload_id)
    if not ceilings:
        return None
    workload = session.get(orm.Workload, workload_id)
    ai_data, notes = plot_points_from_rows(
        load_kernel_plot_rows(session, workload_id),
        workload.total_duration_ns,
    )
    for note in notes:
        console_warning("roofline", note)

    roofline = Roofline(
        argparse.Namespace(),
        mspec,
        {
            "device_id": ROOFLINE_DEVICE_ID,
            "matrix_ops_type": mi_gpu_specs.get_roofline_matrix_type(mspec.gpu_arch),
        },
        benchmark=benchmark_from_ceilings(ceilings, mspec.gpu_arch),
    )
    ops_figure, flops_figure, ops_datatypes, flops_datatypes = (
        roofline.construct_plotly_figures(ai_data=ai_data)
    )
    written = roofline.save_html_files(
        ops_figure,
        flops_figure,
        ops_datatypes,
        flops_datatypes,
        output_path=output_path,
    )
    return None if written is None else output_path


def _gflops(flops_per_second: Optional[float]) -> Optional[float]:
    """Convert a stored FLOP/s value to the GFLOP/s the roofline plots."""
    if flops_per_second is None:
        return None
    return flops_per_second / SCALED_UNITS["GFLOP/s"][1]


def _kernel_intensities(row: "KernelPlotRow") -> Tuple[Dict[str, float], List[str]]:
    """Return each memory level's arithmetic intensity, and the levels that are NULL.

    A level the workload has no metric for is not NULL: it reads as zero, as the CLI
    roofline reads a level it does not measure.
    """
    ai_by_field = {field: 0.0 for field in CACHE_LEVELS}
    missing_levels: List[str] = []
    for field in CACHE_LEVELS:
        metric = _AI_METRIC_BY_FIELD[field]
        if metric not in row.metrics:
            continue
        ai_value = row.metrics[metric]
        if ai_value is None:
            missing_levels.append(metric[len(_AI_METRIC_PREFIX) :])
        else:
            ai_by_field[field] = ai_value
    return ai_by_field, missing_levels


def _append_kernel_point(
    plot_points: PlotPoints,
    row: "KernelPlotRow",
    performance: float,
    ai_by_field: Dict[str, float],
    total_duration_ns: Optional[int],
) -> None:
    """Append one kernel to every series of the plot points, in the same order."""
    plot_points.kernelNames.append(row.kernel_name)
    for field in CACHE_LEVELS:
        getattr(plot_points, field)[0].append(ai_by_field[field])
        getattr(plot_points, field)[1].append(performance)
    plot_points.counts.append(float(row.dispatch_count))
    plot_points.kernelTime.append(float(row.duration_ns))
    plot_points.kernelPctRuntime.append(
        100.0 * row.duration_ns / total_duration_ns if total_duration_ns else None
    )


@dataclass(frozen=True)
class KernelPlotRow:
    """One kernel's roofline values as the database stores them.

    metrics maps each Roofline Plot Points metric defined for the workload to its
    stored base-unit value, or to None where the kernel's value is NULL.
    """

    kernel_name: str
    metrics: Dict[str, Optional[float]]
    dispatch_count: int
    duration_ns: int
