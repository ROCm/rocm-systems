# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Rebuild the workload frames that tty.show_all renders from the analysis DB.

The analysis DB stores every value in its base unit and unrounded, with NULL
where a value cannot be calculated. These functions read those rows back into
the frames the terminal summary and report.txt render: the Top Kernels, Dispatch
List and System Info tables, and every metric table of the run's arch config.
Descriptive columns (Metric, Unit, Description and the like) come from the
config template, which the database does not store.

Renderers round the stored values (D1). A metric whose config expression is one
ROUND call is rounded to that call's digits, as the CLI rounds it. Where the
CLI leaves a metric cell blank, the adapter shows N/A for a NULL DB value
(FR-15). A Percent with a NULL or zero total is N/A where the CLI shows NaN.
System Info keeps the CLI's NaN for a NULL field.

The kernel statistics are read from the kernel view, so the views must exist
(Database.create_views) before the runs are built.
"""

import numbers
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

import config
import utils.analysis_orm as orm
from utils import parser, schema
from utils.metrics.aggregation import to_round
from utils.metrics.base_units import from_base_unit_value
from utils.metrics.expression import SUPPORTED_CALL
from utils.roofline_calc import ROOFLINE_POINTS_TABLE_ID, ROOFLINE_RATES_TABLE_ID

KERNEL_TOP_TABLE_ID = parser.PMC_KERNEL_TOP_TABLE_ID
DISPATCH_TABLE_ID = parser.PMC_DISPATCH_INFO_TABLE_ID
SYS_INFO_TABLE_ID = 101

# Shown in place of a NULL value, as the CLI shows a value it could not evaluate.
NULL_TEXT = "N/A"
# How the expression parser writes a config ROUND call, e.g. "to_round(...)".
ROUND_PREFIX = f"{SUPPORTED_CALL['ROUND']}("

KERNEL_STATS_SQL = text(f"""
    SELECT kernel_name, dispatch_count, duration_ns_sum, duration_ns_mean,
        duration_ns_median
    FROM {orm.PREFIX}kernel_view
    WHERE workload_id = :workload_id
    ORDER BY duration_ns_sum DESC, kernel_name
""")


def build_report_runs(
    session: Session,
    workload_ids: Mapping[str, int],
    arch_configs: Mapping[str, schema.ArchConfig],
    time_unit: str,
) -> Dict[str, schema.Workload]:
    """Return one Workload-like object per run path, ready for tty.show_all.

    Args:
        session: The open session of the analysis database.
        workload_ids: Workload row id of each run path.
        arch_configs: The run's arch config of each GPU arch, the templates.
        time_unit: The --time-unit the top stats times are shown in.
    """
    runs: Dict[str, schema.Workload] = {}
    for run_path, workload_id in workload_ids.items():
        workload_row = session.get(orm.Workload, workload_id)
        arch_config = arch_configs[workload_row.sys_info_extdata["gpu_arch"]]
        runs[run_path] = build_report_workload(
            session, workload_row, arch_config, time_unit
        )
    return runs


def build_report_workload(
    session: Session,
    workload_row: orm.Workload,
    arch_config: schema.ArchConfig,
    time_unit: str,
) -> schema.Workload:
    """Return one workload's frames, as the CLI builds them for the same run.

    The differences from the CLI frames are listed in the module docstring. The
    membw result and the roofline HTML path are not rebuilt here, so the
    returned workload leaves them at their defaults.
    """
    workload_id = workload_row.workload_id
    kernel_stats = _read_frame(session, KERNEL_STATS_SQL, {"workload_id": workload_id})
    metric_templates = _metric_templates(arch_config)
    workload_values = _value_lookup(_read_workload_values(session, workload_id))

    dfs: Dict[int, pd.DataFrame] = {
        KERNEL_TOP_TABLE_ID: _top_kernel_frame(
            kernel_stats, workload_row.total_duration_ns, time_unit
        ),
        DISPATCH_TABLE_ID: _dispatch_frame(_read_dispatch_rows(session, workload_id)),
        SYS_INFO_TABLE_ID: _sys_info_frame(workload_row.sys_info_extdata),
    }
    dfs_type = {
        KERNEL_TOP_TABLE_ID: "raw_csv_table",
        DISPATCH_TABLE_ID: "raw_csv_table",
        SYS_INFO_TABLE_ID: "raw_csv_table",
    }
    for table_id, template in metric_templates.items():
        dfs[table_id] = _metric_frame(template, workload_values)
        dfs_type[table_id] = "metric_table"

    roofline_peaks = _roofline_peaks_frame(session, workload_id)
    roofline_metrics = {}
    if not roofline_peaks.empty:
        roofline_metrics = _roofline_metrics(
            session, workload_id, kernel_stats, metric_templates
        )

    return schema.Workload(
        sys_info=pd.DataFrame([workload_row.sys_info_extdata]),
        dfs=dfs,
        dfs_type=dfs_type,
        roofline_peaks=roofline_peaks,
        roofline_metrics=roofline_metrics,
    )


def _metric_templates(arch_config: schema.ArchConfig) -> Dict[int, pd.DataFrame]:
    """The metric tables of the arch config that the run's profile kept."""
    return {
        table_id: arch_config.dfs[table_id]
        for table_id, table_type in arch_config.dfs_type.items()
        if table_type == "metric_table" and table_id in arch_config.dfs
    }


def _read_frame(
    session: Session,
    statement: Any,  # noqa: ANN401
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run one read statement on the session's connection."""
    return pd.read_sql_query(statement, session.connection(), params=params)


def _read_dispatch_rows(session: Session, workload_id: int) -> pd.DataFrame:
    statement = (
        select(orm.Dispatch.dispatch_id, orm.Kernel.kernel_name, orm.Dispatch.gpu_id)
        .select_from(orm.Dispatch)
        .join(orm.Kernel, orm.Dispatch.kernel_uuid == orm.Kernel.kernel_uuid)
        .where(orm.Kernel.workload_id == workload_id)
        .order_by(orm.Dispatch.dispatch_id)
    )
    return _read_frame(session, statement)


def _read_workload_values(session: Session, workload_id: int) -> pd.DataFrame:
    statement = (
        select(
            orm.MetricDefinition.metric_id,
            orm.WorkloadMetricValue.value_name,
            orm.WorkloadMetricValue.value,
        )
        .select_from(orm.WorkloadMetricValue)
        .join(
            orm.MetricDefinition,
            orm.WorkloadMetricValue.metric_uuid == orm.MetricDefinition.metric_uuid,
        )
        .where(orm.MetricDefinition.workload_id == workload_id)
    )
    return _read_frame(session, statement)


def _read_kernel_values(
    session: Session,
    workload_id: int,
    metric_ids: List[str],
) -> pd.DataFrame:
    statement = (
        select(
            orm.Kernel.kernel_name,
            orm.MetricDefinition.metric_id,
            orm.KernelMetricValue.value_name,
            orm.KernelMetricValue.value,
        )
        .select_from(orm.KernelMetricValue)
        .join(orm.Kernel, orm.KernelMetricValue.kernel_uuid == orm.Kernel.kernel_uuid)
        .join(
            orm.MetricDefinition,
            orm.KernelMetricValue.metric_uuid == orm.MetricDefinition.metric_uuid,
        )
        .where(
            orm.MetricDefinition.workload_id == workload_id,
            orm.MetricDefinition.metric_id.in_(metric_ids),
        )
    )
    return _read_frame(session, statement)


def _roofline_peaks_frame(session: Session, workload_id: int) -> pd.DataFrame:
    """Every roofline.csv row of the workload; empty when the roofline is not valid."""
    statement = (
        select(orm.WorkloadBenchmark.benchmark_extdata)
        .where(orm.WorkloadBenchmark.workload_id == workload_id)
        .order_by(orm.WorkloadBenchmark.workload_benchmark_uuid)
    )
    return pd.DataFrame(session.scalars(statement).all())


def _top_kernel_frame(
    kernel_stats: pd.DataFrame,
    total_duration_ns: Optional[int],
    time_unit: str,
) -> pd.DataFrame:
    """The Top Kernels rows in duration order, with times in time_unit.

    Percent is each kernel's share of the filtered workload total, which the
    database stores, so a kernel the -k filter leaves out has no row here. A
    NULL or zero total leaves the share unknown, so Percent is N/A.
    """
    divisor = config.TIME_UNITS[time_unit]
    suffix = f"({time_unit})"
    if total_duration_ns:
        percent = kernel_stats["duration_ns_sum"] / total_duration_ns * 100
    else:
        percent = pd.Series(NULL_TEXT, index=kernel_stats.index, dtype=object)
    return pd.DataFrame({
        "Kernel_Name": kernel_stats["kernel_name"],
        "Count": kernel_stats["dispatch_count"],
        f"Sum{suffix}": kernel_stats["duration_ns_sum"] / divisor,
        f"Mean{suffix}": kernel_stats["duration_ns_mean"] / divisor,
        f"Median{suffix}": kernel_stats["duration_ns_median"] / divisor,
        "Percent": percent,
    }).reset_index(drop=True)


def _dispatch_frame(dispatch_rows: pd.DataFrame) -> pd.DataFrame:
    """The Dispatch List rows of the dispatches that survive the filters."""
    return pd.DataFrame({
        "Dispatch_ID": dispatch_rows["dispatch_id"],
        "Kernel_Name": dispatch_rows["kernel_name"],
        "GPU_ID": dispatch_rows["gpu_id"],
    })


def _sys_info_frame(sys_info: Dict[str, Any]) -> pd.DataFrame:
    """The System Info table: one row per sysinfo field, in one Info column.

    A NULL field is NaN, which the CLI table shows the same way. FR-27 would
    print N/A instead, but NFR-2 lists no exception for System Info, so the
    CLI's form is kept until the HLD decides it.
    """
    values = [np.nan if value is None else value for value in sys_info.values()]
    return pd.DataFrame({
        "Info": pd.Series(values, index=list(sys_info.keys()), dtype=object),
    })


def _value_lookup(values: pd.DataFrame) -> Dict[Tuple[str, str], Any]:
    """Map (metric_id, value_name) to the stored value."""
    return {
        (row.metric_id, row.value_name): row.value
        for row in values.itertuples(index=False)
    }


def _rounding_digits(expression: object) -> Optional[int]:
    """The digits of a config expression that is one ROUND call, or None.

    The CLI rounds such an expression to these digits when it evaluates it. An
    expression with other terms around its ROUND call is not rounded here.
    """
    if not isinstance(expression, str):
        return None
    expression = expression.strip()
    if not expression.startswith(ROUND_PREFIX) or not expression.endswith(")"):
        return None
    arguments = expression[len(ROUND_PREFIX) : -1]
    _, comma, digits = arguments.rpartition(",")
    if not comma or not digits.strip().isdigit() or not _brackets_balance(arguments):
        return None
    return int(digits)


def _brackets_balance(arguments: str) -> bool:
    """True when every bracket opened in arguments is closed within it."""
    depth = 0
    for char in arguments:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _display_value(
    values: Mapping[Tuple[str, str], Any],
    metric_id: str,
    value_name: str,
    unit: Optional[str],
    digits: Optional[int],
) -> Any:  # noqa: ANN401
    """The cell a metric table shows for one value column of one metric.

    A value the database has no row for is blank, as the CLI leaves a column
    that has no expression. A NULL value shows as N/A. A value its config rounds
    is rounded to those digits, in the unit the config labels it with.
    """
    key = (metric_id, value_name)
    if key not in values:
        return ""
    value = values[key]
    if value is None or pd.isna(value):
        return NULL_TEXT
    shown = from_base_unit_value(unit, value_name, value)
    if digits is None or not isinstance(shown, numbers.Real):
        return shown
    return to_round(shown, digits)


def _metric_frame(
    template: pd.DataFrame,
    values: Mapping[Tuple[str, str], Any],
) -> pd.DataFrame:
    """Fill the value columns of a metric table template from stored values.

    The descriptive columns stay as the template has them. Each value column is
    shown in the unit its metric config labels it with, rounded as its config
    expression rounds it.
    """
    frame = template.copy()
    units = (
        template["Unit"].tolist()
        if "Unit" in template.columns
        else [None] * len(template)
    )
    for column in template.columns:
        if column in schema.METRIC_DESCRIPTIVE_COLUMNS:
            continue
        expressions = template[column].tolist()
        frame[column] = [
            _display_value(
                values, metric_id, column, unit, _rounding_digits(expression)
            )
            for metric_id, unit, expression in zip(template.index, units, expressions)
        ]
    return frame


def _roofline_metrics(
    session: Session,
    workload_id: int,
    kernel_stats: pd.DataFrame,
    metric_templates: Mapping[int, pd.DataFrame],
) -> Dict[int, Dict[str, Any]]:
    """Per-kernel rate and plot-point tables, keyed by the kernel's top-stats row."""
    rate_template = metric_templates.get(ROOFLINE_RATES_TABLE_ID, pd.DataFrame())
    points_template = metric_templates.get(ROOFLINE_POINTS_TABLE_ID, pd.DataFrame())
    roofline_metric_ids = [
        *rate_template.index,
        *points_template.index,
    ]
    kernel_values = _read_kernel_values(session, workload_id, roofline_metric_ids)
    values_by_kernel = {
        kernel_name: _value_lookup(group)
        for kernel_name, group in kernel_values.groupby("kernel_name")
    }

    roofline_metrics: Dict[int, Dict[str, Any]] = {}
    for kernel_index, kernel_name in enumerate(kernel_stats["kernel_name"]):
        values = values_by_kernel.get(kernel_name, {})
        roofline_metrics[kernel_index] = {
            "name": kernel_name,
            "ai_table": _metric_frame_or_empty(rate_template, values),
            "calc_table": _metric_frame_or_empty(points_template, values),
        }
    return roofline_metrics


def _metric_frame_or_empty(
    template: pd.DataFrame,
    values: Mapping[Tuple[str, str], Any],
) -> pd.DataFrame:
    if template.empty:
        return pd.DataFrame()
    return _metric_frame(template, values)
