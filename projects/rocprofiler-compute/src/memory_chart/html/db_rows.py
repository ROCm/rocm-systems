# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Read one workload's memory chart values from the analysis database."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import object_session

from utils import analysis_orm as orm
from utils.file_io import rank_kernels_by_total_duration

# metric_id -> {value_name: value}, as stored: unrounded, in base units
MetricValues = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class KernelRows:
    """One kernel's stored metric values, in top-stats order."""

    name: str
    values: MetricValues


@dataclass(frozen=True)
class WorkloadChartRows:
    """Everything the memory chart renderer reads from one workload's rows."""

    arch: str
    layout_json: Optional[Dict[str, Any]]
    aggregate: MetricValues
    kernels: Tuple[KernelRows, ...]


def read_workload_chart_rows(
    workload: orm.Workload, metric_ids: Iterable[str]
) -> WorkloadChartRows:
    """Read the rows of the given metric ids for the workload and each kernel.

    The kernels follow the top-stats order (total duration, longest first), so
    their index matches the one the terminal and CSV outputs use. The rows hold
    only the kernels that a -k filter keeps, so a -k run has no unfiltered
    aggregate or kernel list to read; the caller must not render such a run.
    """
    session = object_session(workload)
    ids = list(metric_ids)

    aggregate: MetricValues = {}
    for metric_id, value_name, value in session.execute(
        select(
            orm.MetricDefinition.metric_id,
            orm.WorkloadMetricValue.value_name,
            orm.WorkloadMetricValue.value,
        )
        .join(orm.WorkloadMetricValue.metric)
        .where(
            orm.WorkloadMetricValue.workload_id == workload.workload_id,
            orm.MetricDefinition.metric_id.in_(ids),
        )
    ):
        aggregate.setdefault(metric_id, {})[value_name] = value

    per_kernel: Dict[str, MetricValues] = {}
    for kernel_name, metric_id, value_name, value in session.execute(
        select(
            orm.Kernel.kernel_name,
            orm.MetricDefinition.metric_id,
            orm.KernelMetricValue.value_name,
            orm.KernelMetricValue.value,
        )
        .join(orm.KernelMetricValue.kernel)
        .join(orm.KernelMetricValue.metric)
        .where(
            orm.Kernel.workload_id == workload.workload_id,
            orm.MetricDefinition.metric_id.in_(ids),
        )
    ):
        per_kernel.setdefault(kernel_name, {}).setdefault(metric_id, {})[value_name] = (
            value
        )

    dispatches = pd.DataFrame(
        session.execute(
            select(
                orm.Kernel.kernel_name,
                orm.Dispatch.start_timestamp,
                orm.Dispatch.end_timestamp,
            )
            .join(orm.Dispatch.kernel)
            .where(orm.Kernel.workload_id == workload.workload_id)
        ).all(),
        columns=["Kernel_Name", "Start_Timestamp", "End_Timestamp"],
    )
    ranked = rank_kernels_by_total_duration(dispatches) if not dispatches.empty else []

    return WorkloadChartRows(
        arch=str(workload.sys_info_extdata["gpu_arch"]),
        layout_json=workload.memory_chart_layout_extdata,
        aggregate=aggregate,
        kernels=tuple(KernelRows(name, per_kernel.get(name, {})) for name in ranked),
    )
