# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Writes the workload folders of an analyze run folder.

Every file is read from the open analysis DB, in base units, with NULL written
as an empty cell. Kernel folders are named by layout_names. Workload
directories are never read.

The DB keeps only the kernels that -k selects, so the kernel index comes from
the top-stats ranking the caller passes in, which lists every kernel.

pc-sampling-summary.csv opens with a '#' note line above its header, so read it
with comment='#' or skip its first line.

The kernel ISA and source files are not written by the pc_sampling exporters.
Those write one file per kernel, code object and process with the instruction
locations of a separate export, and one source file per source path with ISA
references. The one-folder-per-kernel layout uses neither. The column names are
shared through build_header and the exporters' column tuples.
"""

import csv
import itertools
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Row

import utils.analysis_orm as orm
from pc_sampling.pc_sampling_csv_export import build_header
from pc_sampling.per_kernel_isa_export import BASE_COLUMNS, TRAILING_COLUMNS
from pc_sampling.source_snapshot_analysis import (
    SOURCE_FRAME_SEPARATOR,
    UNKNOWN_SOURCE_LINE_TOKEN,
)
from rocprof_compute_analyze.layout_names import (
    demangle_kernel_names,
    kernel_folder_name,
    kernel_function_name,
    kernel_index_width,
)
from rocprof_compute_analyze.output_location import workload_folder_name
from utils.analysis_orm import Database
from utils.logger import console_error
from utils.roofline_calc import ROOFLINE_POINTS_TABLE_ID
from utils.utils_analysis import VALUE_COL_PREFERENCE

KERNELS_FILE_NAME = "kernels.csv"
DISPATCHES_FILE_NAME = "dispatches.csv"
WORKLOAD_METRICS_FILE_NAME = "metrics-summary.csv"
PC_SAMPLING_SUMMARY_FILE_NAME = "pc-sampling-summary.csv"
ROOFLINE_FILE_NAME = "roofline.csv"
ML_API_TRACE_FILE_NAME = "ml-api-trace.csv"
KERNEL_METRICS_FILE_SUFFIX = "-metrics.csv"
PC_SAMPLING_ISA_FILE_SUFFIX = "-pc-sampling-isa.csv"
PC_SAMPLING_SOURCE_FILE_SUFFIX = "-pc-sampling-src.csv"

# The roofline plot points are the metrics of the points table, whose ids start with it.
ROOFLINE_PLOT_POINTS_ID_PREFIX = f"{ROOFLINE_POINTS_TABLE_ID}."
PC_SAMPLING_NARROWING_NOTE = (
    "# PC samples are narrowed by -k only. --gpu-id and --kernel-iteration-range "
    "do not apply to them yet."
)
UNATTRIBUTABLE_INSTRUCTIONS_LABEL = "Unattributable instructions"

KERNELS_HEADER = ["Index", "Kernel_Name", "Folder", "Duration_ns", "Pct"]
DISPATCHES_HEADER = [
    "Dispatch_ID",
    "Kernel_Index",
    "Kernel_Name",
    "GPU_ID",
    "Start_Timestamp",
    "End_Timestamp",
    "Duration_ns",
]
METRIC_IDENTITY_HEADER = ["Metric_ID", "Metric", "Unit", "Table", "Sub_Table"]
ROOFLINE_HEADER = [
    "Kind",
    "Memory_Level",
    "Data_Type",
    "Pipe",
    "Kernel_Index",
    "Kernel_Name",
    "Metric",
    "Unit",
    "Value",
]
POINT_KIND = "point"
SOURCE_IDENTITY_COLUMNS = ("Source file", "Source line number", "Source line")
PC_SAMPLING_MEASURE_COLUMNS = ("Total count", "Active count", "Stall count")
PC_SAMPLING_SUMMARY_LEADING_COLUMNS = ("Index", "Kernel_Name", "Folder")


def write_workload_folders(
    run_folder: Path,
    workload_paths: Sequence[str],
    kernel_rankings: Mapping[str, Sequence[str]],
) -> List[Path]:
    """Write one folder per -p workload, in -p order, into the run folder.

    The folder of the workload at position N, counting from 1, is named by
    workload_folder_name. Each workload's rows are found in the open analysis
    DB by the name pair that build_database stored. kernel_rankings maps each
    workload path to its kernel names in top-stats order, as the analysis
    calc_kernel_rankings gives them, so a kernel's index is its id for -k.

    Returns:
        The workload folders that were written.
    """
    if Database.get_session() is None:
        console_error("analysis", "No active database session for the run folder")
    stall_reasons_by_workload = Database.get_stall_reasons_by_workload()
    workload_folders = []
    for position, workload_path in enumerate(workload_paths, start=1):
        workload_folder = run_folder / workload_folder_name(
            position, Path(workload_path)
        )
        workload_id = _find_workload_id(workload_path)
        workload_folder.mkdir()
        _write_workload_folder(
            workload_id,
            workload_folder,
            stall_reasons_by_workload.get(workload_id, []),
            list(kernel_rankings[workload_path]),
        )
        workload_folders.append(workload_folder)
    return workload_folders


def write_ml_api_trace_csv(workload_folder: Path, trace: pd.DataFrame) -> Path:
    """Write a workload's ML API trace into its folder as ml-api-trace.csv.

    The trace is the joined marker and counter table of the operator analysis.
    Missing cells are written empty.

    Returns:
        The path of the written file.
    """
    trace_path = workload_folder / ML_API_TRACE_FILE_NAME
    trace.to_csv(trace_path, index=False, na_rep="", encoding="utf-8")
    return trace_path


def _find_workload_id(workload_path: str) -> int:
    """Return the id of the analysis DB row that build_database wrote for a workload."""
    # Mirrors the name pair build_database stores for each workload.
    name = workload_path.split("/")[-2]
    sub_name = workload_path.split("/")[-1]
    workload_id = (
        Database
        .get_session()
        .execute(
            select(orm.Workload.workload_id).where(
                orm.Workload.name == name,
                orm.Workload.sub_name == sub_name,
            )
        )
        .scalar_one_or_none()
    )
    if workload_id is None:
        console_error(
            "analysis",
            f"No analysis database rows were stored for workload {workload_path}",
        )
    return workload_id


def _write_workload_folder(
    workload_id: int,
    workload_folder: Path,
    stall_reasons: List[str],
    kernel_ranking: List[str],
) -> None:
    """Write every file of one workload folder."""
    kernels = _kernel_folders(workload_id, kernel_ranking)
    _write_csv(
        workload_folder / KERNELS_FILE_NAME,
        KERNELS_HEADER,
        (
            [
                kernel.index,
                kernel.kernel_name,
                kernel.folder_name,
                kernel.duration_ns,
                kernel.pct,
            ]
            for kernel in kernels
        ),
    )
    _write_dispatches(workload_id, workload_folder, kernels)
    _write_metric_table(
        workload_folder / WORKLOAD_METRICS_FILE_NAME,
        _workload_metric_rows(workload_id),
    )
    value_names = _value_names_of_kernel_metrics(workload_id)
    roofline_points = _write_kernel_folders(workload_folder, kernels, value_names)
    _write_roofline(workload_id, workload_folder, kernels, roofline_points)
    _write_pc_sampling(workload_id, workload_folder, kernels, stall_reasons)


def _kernel_folders(
    workload_id: int, kernel_ranking: List[str]
) -> List["KernelFolder"]:
    """Return the workload's kernels that ran, in top-stats order, with their folders.

    A kernel's index is its place in kernel_ranking, which lists every kernel of
    the workload. A kernel that -k leaves out has no rows here, so the indexes
    of the others still match their ids. A kernel's duration is the sum of its
    dispatches. Pct is that sum over the workload's filtered total duration, in
    percent, and NULL without a total.
    """
    duration_ns = func.sum(orm.Dispatch.end_timestamp - orm.Dispatch.start_timestamp)
    rows = (
        Database
        .get_session()
        .execute(
            select(
                orm.Kernel.kernel_uuid,
                orm.Kernel.kernel_name,
                duration_ns.label("duration_ns"),
                orm.Workload.total_duration_ns,
            )
            .select_from(orm.Dispatch)
            .join(orm.Kernel, orm.Dispatch.kernel_uuid == orm.Kernel.kernel_uuid)
            .join(orm.Workload, orm.Kernel.workload_id == orm.Workload.workload_id)
            .where(orm.Kernel.workload_id == workload_id)
            .group_by(
                orm.Kernel.kernel_uuid,
                orm.Kernel.kernel_name,
                orm.Workload.total_duration_ns,
            )
            .order_by(orm.Kernel.kernel_uuid)
        )
        .all()
    )
    rows_by_name = {row.kernel_name: row for row in rows}
    unranked_names = sorted(set(rows_by_name) - set(kernel_ranking))
    if unranked_names:
        console_error(
            "analysis",
            f"Kernels missing from the top-stats ranking: {unranked_names}",
        )
    demangled_names = demangle_kernel_names(row.kernel_name for row in rows)
    index_width = kernel_index_width(len(kernel_ranking))
    kernels = []
    for index, kernel_name in enumerate(kernel_ranking):
        row = rows_by_name.get(kernel_name)
        if row is None:
            continue
        function_name = kernel_function_name(
            demangled_names.get(row.kernel_name, row.kernel_name)
        )
        kernels.append(
            KernelFolder(
                kernel_uuid=row.kernel_uuid,
                index=index,
                kernel_name=row.kernel_name,
                folder_name=kernel_folder_name(index, index_width, function_name),
                duration_ns=row.duration_ns,
                pct=_percent_of(row.duration_ns, row.total_duration_ns),
            )
        )
    return kernels


def _percent_of(part: Optional[int], whole: Optional[int]) -> Optional[float]:
    """Return part as a percent of whole, or None if either is missing or whole is 0."""
    if part is None or not whole:
        return None
    return part / whole * 100


def _write_dispatches(
    workload_id: int,
    workload_folder: Path,
    kernels: List["KernelFolder"],
) -> None:
    """Write one row per dispatch of the workload, in dispatch order."""
    kernel_by_uuid = {kernel.kernel_uuid: kernel for kernel in kernels}
    dispatch_rows = (
        Database
        .get_session()
        .execute(
            select(
                orm.Dispatch.dispatch_id,
                orm.Dispatch.kernel_uuid,
                orm.Dispatch.gpu_id,
                orm.Dispatch.start_timestamp,
                orm.Dispatch.end_timestamp,
            )
            .select_from(orm.Dispatch)
            .join(orm.Kernel, orm.Dispatch.kernel_uuid == orm.Kernel.kernel_uuid)
            .where(orm.Kernel.workload_id == workload_id)
            .order_by(orm.Dispatch.dispatch_uuid)
        )
        .all()
    )
    _write_csv(
        workload_folder / DISPATCHES_FILE_NAME,
        DISPATCHES_HEADER,
        (
            _dispatch_csv_row(row, kernel_by_uuid[row.kernel_uuid])
            for row in dispatch_rows
        ),
    )


def _dispatch_csv_row(row: Row, kernel: "KernelFolder") -> List[Any]:
    """Return one dispatches.csv row, with the duration in ns."""
    duration_ns = None
    if row.start_timestamp is not None and row.end_timestamp is not None:
        duration_ns = row.end_timestamp - row.start_timestamp
    return [
        row.dispatch_id,
        kernel.index,
        kernel.kernel_name,
        row.gpu_id,
        row.start_timestamp,
        row.end_timestamp,
        duration_ns,
    ]


def _workload_metric_rows(workload_id: int) -> List[Any]:
    """Return the workload's metric value rows, ordered by metric then value."""
    return (
        Database
        .get_session()
        .execute(
            select(
                orm.MetricDefinition.metric_uuid,
                orm.MetricDefinition.metric_id,
                orm.MetricDefinition.name.label("metric_name"),
                orm.MetricDefinition.unit,
                orm.MetricDefinition.table_name,
                orm.MetricDefinition.sub_table_name,
                orm.WorkloadMetricValue.value_name,
                orm.WorkloadMetricValue.value,
            )
            .select_from(orm.WorkloadMetricValue)
            .join(
                orm.MetricDefinition,
                orm.WorkloadMetricValue.metric_uuid == orm.MetricDefinition.metric_uuid,
            )
            .where(orm.WorkloadMetricValue.workload_id == workload_id)
            .order_by(
                orm.MetricDefinition.metric_uuid,
                orm.WorkloadMetricValue.value_uuid,
            )
        )
        .all()
    )


def _value_names_of_kernel_metrics(workload_id: int) -> List[str]:
    """Return the kernel metric value names of a workload, in first-seen order."""
    names = (
        Database
        .get_session()
        .execute(
            select(orm.KernelMetricValue.value_name)
            .select_from(orm.KernelMetricValue)
            .join(
                orm.Kernel, orm.KernelMetricValue.kernel_uuid == orm.Kernel.kernel_uuid
            )
            .where(orm.Kernel.workload_id == workload_id)
            .group_by(orm.KernelMetricValue.value_name)
            .order_by(func.min(orm.KernelMetricValue.value_uuid))
        )
        .scalars()
        .all()
    )
    return list(names)


def _kernel_metric_rows(kernel_uuid: int) -> List[Any]:
    """Return one kernel's metric value rows, ordered by metric then value."""
    return (
        Database
        .get_session()
        .execute(
            select(
                orm.MetricDefinition.metric_uuid,
                orm.MetricDefinition.metric_id,
                orm.MetricDefinition.name.label("metric_name"),
                orm.MetricDefinition.unit,
                orm.MetricDefinition.table_name,
                orm.MetricDefinition.sub_table_name,
                orm.KernelMetricValue.value_name,
                orm.KernelMetricValue.value,
            )
            .select_from(orm.KernelMetricValue)
            .join(
                orm.MetricDefinition,
                orm.KernelMetricValue.metric_uuid == orm.MetricDefinition.metric_uuid,
            )
            .where(orm.KernelMetricValue.kernel_uuid == kernel_uuid)
            .order_by(
                orm.MetricDefinition.metric_uuid,
                orm.KernelMetricValue.value_uuid,
            )
        )
        .all()
    )


def _write_kernel_folders(
    workload_folder: Path,
    kernels: List["KernelFolder"],
    value_names: List[str],
) -> List["RooflinePoint"]:
    """Create each kernel folder and write its metrics file.

    Returns:
        The roofline plot points of every kernel, in top-stats order.
    """
    roofline_points = []
    for kernel in kernels:
        kernel_folder = workload_folder / kernel.folder_name
        kernel_folder.mkdir()
        metric_rows = _kernel_metric_rows(kernel.kernel_uuid)
        _write_metric_table(
            kernel_folder / f"{kernel.folder_name}{KERNEL_METRICS_FILE_SUFFIX}",
            metric_rows,
            value_names,
        )
        roofline_points.extend(_roofline_points(kernel, metric_rows))
    return roofline_points


def _write_metric_table(
    metrics_path: Path,
    metric_rows: Iterable[Any],
    value_names: Optional[List[str]] = None,
) -> None:
    """Write metric value rows as one row per metric, with a column per value name.

    A metric without a value of some name leaves that cell empty. The value
    names are taken from the rows when value_names is not given.
    """
    metric_rows = list(metric_rows)
    if value_names is None:
        value_names = list(dict.fromkeys(row.value_name for row in metric_rows))
    _write_csv(
        metrics_path,
        [*METRIC_IDENTITY_HEADER, *value_names],
        _pivot_metric_rows(metric_rows, value_names),
    )


def _pivot_metric_rows(
    metric_rows: Iterable[Any],
    value_names: List[str],
) -> Iterator[List[Any]]:
    """Yield one row per metric from rows ordered by metric, then value."""
    for _, same_metric_rows in itertools.groupby(
        metric_rows, key=lambda row: row.metric_uuid
    ):
        same_metric_rows = list(same_metric_rows)
        first_row = same_metric_rows[0]
        values_by_name = {row.value_name: row.value for row in same_metric_rows}
        yield [
            first_row.metric_id,
            first_row.metric_name,
            first_row.unit,
            first_row.table_name,
            first_row.sub_table_name,
            *(values_by_name.get(value_name) for value_name in value_names),
        ]


def _roofline_points(
    kernel: "KernelFolder", metric_rows: List[Any]
) -> List["RooflinePoint"]:
    """Return the plot points among one kernel's metric rows."""
    return [
        RooflinePoint(
            kernel_index=kernel.index,
            kernel_name=kernel.kernel_name,
            metric_name=row.metric_name,
            unit=row.unit,
            value=row.value,
        )
        for row in metric_rows
        if row.metric_id.startswith(ROOFLINE_PLOT_POINTS_ID_PREFIX)
        and row.value_name in VALUE_COL_PREFERENCE
    ]


def _write_roofline(
    workload_id: int,
    workload_folder: Path,
    kernels: List["KernelFolder"],
    roofline_points: List["RooflinePoint"],
) -> None:
    """Write roofline.csv with the ceilings and the plot points, when either exists.

    Each ceiling row names its memory level, or its data type and pipe. Each
    point row names its kernel and the plot-point metric.
    """
    ceiling_rows = _ceiling_rows(workload_id)
    if not ceiling_rows and not roofline_points:
        return
    _write_csv(
        workload_folder / ROOFLINE_FILE_NAME,
        ROOFLINE_HEADER,
        [
            *ceiling_rows,
            *(
                [
                    POINT_KIND,
                    None,
                    None,
                    None,
                    point.kernel_index,
                    point.kernel_name,
                    point.metric_name,
                    point.unit,
                    point.value,
                ]
                for point in roofline_points
            ),
        ],
    )


def _ceiling_rows(workload_id: int) -> List[List[Any]]:
    """Return the workload's roofline ceilings as roofline.csv rows."""
    ceilings = (
        Database
        .get_session()
        .execute(
            select(
                orm.RooflineCeiling.kind,
                orm.RooflineCeiling.memory_level,
                orm.RooflineCeiling.data_type,
                orm.RooflineCeiling.pipe,
                orm.RooflineCeiling.unit,
                orm.RooflineCeiling.value,
            )
            .where(orm.RooflineCeiling.workload_id == workload_id)
            .order_by(orm.RooflineCeiling.ceiling_uuid)
        )
        .all()
    )
    return [
        [
            ceiling.kind,
            ceiling.memory_level,
            ceiling.data_type,
            ceiling.pipe,
            None,
            None,
            None,
            ceiling.unit,
            ceiling.value,
        ]
        for ceiling in ceilings
    ]


def _write_pc_sampling(
    workload_id: int,
    workload_folder: Path,
    kernels: List["KernelFolder"],
    stall_reasons: List[str],
) -> None:
    """Write the PC sampling files of the kernels that have samples.

    Each such kernel gets one ISA file and one source file in its folder. The
    ISA file has a row per instruction line of each code object and process
    that ran the kernel, so an offset can repeat. Its Code object id and Pid
    columns tell those rows apart. The source file sums across processes. The
    summary lists the same kernels with their sample totals, and is written
    only when some kernel has samples.
    """
    kernel_by_uuid = {kernel.kernel_uuid: kernel for kernel in kernels}
    source_contents = _source_line_contents(workload_id)
    summary_rows = []
    isa_rows = Database.stream_per_kernel_isa_rows(workload_id, stall_reasons)
    # The statement orders rows by kernel first, so each kernel's rows are contiguous.
    for kernel_uuid, kernel_rows in itertools.groupby(isa_rows, key=lambda row: row[2]):
        kernel = kernel_by_uuid.get(kernel_uuid)
        instruction_lines = [_isa_line(row, len(stall_reasons)) for row in kernel_rows]
        if kernel is None or not any(
            line.count is not None for line in instruction_lines
        ):
            continue
        kernel_folder = workload_folder / kernel.folder_name
        _write_csv(
            kernel_folder / f"{kernel.folder_name}{PC_SAMPLING_ISA_FILE_SUFFIX}",
            build_header(BASE_COLUMNS, stall_reasons, TRAILING_COLUMNS),
            _isa_csv_rows(instruction_lines),
        )
        _write_source_file(
            kernel_folder / f"{kernel.folder_name}{PC_SAMPLING_SOURCE_FILE_SUFFIX}",
            instruction_lines,
            stall_reasons,
            source_contents,
        )
        summary_rows.append((kernel, _sum_measures(instruction_lines, stall_reasons)))
    if summary_rows:
        _write_pc_sampling_summary(workload_folder, summary_rows, stall_reasons)


def _write_pc_sampling_summary(
    workload_folder: Path,
    summary_rows: List[Tuple["KernelFolder", Dict[str, Optional[int]]]],
    stall_reasons: List[str],
) -> None:
    """Write pc-sampling-summary.csv, with the -k only note on a '#' line first."""
    measure_columns = build_header(PC_SAMPLING_MEASURE_COLUMNS, stall_reasons, ())
    _write_csv(
        workload_folder / PC_SAMPLING_SUMMARY_FILE_NAME,
        [*PC_SAMPLING_SUMMARY_LEADING_COLUMNS, *measure_columns],
        (
            [
                kernel.index,
                kernel.kernel_name,
                kernel.folder_name,
                *(sums.get(column) for column in measure_columns),
            ]
            for kernel, sums in sorted(summary_rows, key=lambda row: row[0].index)
        ),
        preamble=PC_SAMPLING_NARROWING_NOTE,
    )


def _write_source_file(
    source_path: Path,
    instruction_lines: List["IsaLine"],
    stall_reasons: List[str],
    source_contents: Dict[Tuple[str, Optional[int]], Optional[str]],
) -> None:
    """Write one kernel's samples per source line, from its instruction lines."""
    measure_columns = build_header(PC_SAMPLING_MEASURE_COLUMNS, stall_reasons, ())
    sums_by_frame = _sum_measures_by_source_frame(instruction_lines, stall_reasons)
    _write_csv(
        source_path,
        [*SOURCE_IDENTITY_COLUMNS, *measure_columns],
        (
            _source_csv_row(
                frame, sums_by_frame[frame], measure_columns, source_contents
            )
            for frame in _sorted_source_frames(sums_by_frame)
        ),
    )


def _source_csv_row(
    frame: Optional[str],
    sums: Dict[str, Optional[int]],
    measure_columns: List[str],
    source_contents: Dict[Tuple[str, Optional[int]], Optional[str]],
) -> List[Any]:
    """Return one source file row. A frame of None is the unattributable row."""
    if frame is None:
        return [
            None,
            None,
            UNATTRIBUTABLE_INSTRUCTIONS_LABEL,
            *(sums.get(column) for column in measure_columns),
        ]
    file_path, line_number = _split_frame(frame)
    return [
        file_path,
        line_number,
        source_contents.get((file_path, line_number)),
        *(sums.get(column) for column in measure_columns),
    ]


def _source_line_contents(
    workload_id: int,
) -> Dict[Tuple[str, Optional[int]], Optional[str]]:
    """Map each captured source line of a workload to its text."""
    rows = (
        Database
        .get_session()
        .execute(
            select(
                orm.SourceFile.file_path,
                orm.SourceLine.line_number,
                orm.SourceLine.content,
            )
            .select_from(orm.SourceLine)
            .join(
                orm.SourceFile,
                orm.SourceLine.source_file_uuid == orm.SourceFile.source_file_uuid,
            )
            .where(orm.SourceFile.workload_id == workload_id)
        )
        .all()
    )
    return {(row.file_path, row.line_number): row.content for row in rows}


def _isa_line(row: Tuple[Any, ...], stall_reason_count: int) -> "IsaLine":
    """Split one stream row of the per-kernel ISA statement into its fields.

    The first six fields name the file, and the instruction uuid follows them.
    """
    (
        _workload_name,
        _workload_sub_name,
        kernel_uuid,
        _kernel_short_name,
        _code_object_id,
        _pid,
        instruction_uuid,
        offset,
        instruction,
        instruction_type,
        count,
        issue_count,
        stall_count,
        wave_occupancy,
        active_thread,
        *rest,
    ) = row
    stall_counts = tuple(rest[:stall_reason_count])
    source, code_object_id, pid = rest[stall_reason_count:]
    return IsaLine(
        kernel_uuid=kernel_uuid,
        instruction_uuid=instruction_uuid,
        offset=offset,
        instruction=instruction,
        instruction_type=instruction_type,
        count=count,
        issue_count=issue_count,
        stall_count=stall_count,
        wave_occupancy=wave_occupancy,
        active_thread=active_thread,
        stall_counts=stall_counts,
        source=source,
        code_object_id=code_object_id,
        pid=pid,
    )


def _isa_csv_rows(instruction_lines: List["IsaLine"]) -> Iterator[List[Any]]:
    """Yield the ISA file rows, numbering the instruction lines from 1."""
    for line_number, line in enumerate(instruction_lines, start=1):
        yield [
            line_number,
            line.offset,
            line.instruction,
            line.instruction_type,
            line.count,
            line.issue_count,
            line.stall_count,
            line.wave_occupancy,
            line.active_thread,
            *line.stall_counts,
            line.source,
            line.code_object_id,
            line.pid,
        ]


def _sum_measures(
    instruction_lines: List["IsaLine"],
    stall_reasons: List[str],
) -> Dict[str, Optional[int]]:
    """Return the sample totals of instruction lines, keyed by column name.

    A column stays None when no line carries its value.
    """
    measure_columns = build_header(PC_SAMPLING_MEASURE_COLUMNS, stall_reasons, ())
    sums: Dict[str, Optional[int]] = {}
    for line in instruction_lines:
        for column, value in zip(measure_columns, _measures_of(line)):
            _add_measure(sums, column, value)
    return sums


def _sum_measures_by_source_frame(
    instruction_lines: List["IsaLine"],
    stall_reasons: List[str],
) -> Dict[Optional[str], Dict[str, Optional[int]]]:
    """Return the sample totals of each source frame, keyed by frame text.

    An instruction counts toward each distinct frame of its inline stack. An
    instruction with no source counts toward the None frame. Instructions that
    no sample landed on are left out, so their lines do not appear.
    """
    measure_columns = build_header(PC_SAMPLING_MEASURE_COLUMNS, stall_reasons, ())
    sums_by_frame: Dict[Optional[str], Dict[str, Optional[int]]] = {}
    for line in instruction_lines:
        if line.count is None:
            continue
        frames = _frames_of(line.source) or [None]
        for frame in frames:
            sums = sums_by_frame.setdefault(frame, {})
            for column, value in zip(measure_columns, _measures_of(line)):
                _add_measure(sums, column, value)
    return sums_by_frame


def _sorted_source_frames(
    sums_by_frame: Dict[Optional[str], Dict[str, Optional[int]]],
) -> List[Optional[str]]:
    """Order the frames by file, then line number with unknown lines first.

    The unattributable frame comes last.
    """
    return sorted(sums_by_frame, key=_source_frame_order)


def _source_frame_order(frame: Optional[str]) -> Tuple[int, str, bool, int]:
    """Return the sort key of a source frame, as _sorted_source_frames uses it."""
    if frame is None:
        return (1, "", False, 0)
    file_path, line_number = _split_frame(frame)
    return (0, file_path, line_number is not None, line_number or 0)


def _measures_of(line: "IsaLine") -> List[Optional[int]]:
    """Return a line's measures in the order of the measure columns."""
    return [line.count, line.issue_count, line.stall_count, *line.stall_counts]


def _add_measure(
    sums: Dict[str, Optional[int]], column: str, value: Optional[int]
) -> None:
    """Add one value to a column total, leaving the total None while it has no value."""
    if value is None:
        sums.setdefault(column, None)
    elif sums.get(column) is None:
        sums[column] = value
    else:
        sums[column] = sums[column] + value


def _frames_of(source: Optional[str]) -> List[str]:
    """Return the distinct frames of an instruction's source chain, innermost first."""
    if not source:
        return []
    return list(dict.fromkeys(source.split(SOURCE_FRAME_SEPARATOR)))


def _split_frame(frame: str) -> Tuple[str, Optional[int]]:
    """Split a frame of the form file:line, where the unknown line is ?."""
    file_path, _, line_token = frame.rpartition(":")
    line_number = None if line_token == UNKNOWN_SOURCE_LINE_TOKEN else int(line_token)
    return file_path, line_number


def _write_csv(
    csv_path: Path,
    header: List[str],
    rows: Iterable[Sequence[Any]],
    preamble: Optional[str] = None,
) -> None:
    """Write a CSV file, with an optional line above the header.

    None is written as an empty cell.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        if preamble is not None:
            csv_file.write(f"{preamble}\n")
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


class KernelFolder(NamedTuple):
    """One kernel of a workload, with the folder its files go in."""

    kernel_uuid: int
    # Place in the top-stats order, counting from 0
    index: int
    kernel_name: str
    folder_name: str
    duration_ns: Optional[int]
    pct: Optional[float]


class RooflinePoint(NamedTuple):
    """One plot point of a kernel on the roofline."""

    kernel_index: int
    kernel_name: str
    metric_name: str
    unit: Optional[str]
    value: Optional[float]


class IsaLine(NamedTuple):
    """One instruction line of a kernel's ISA, with the samples on it."""

    kernel_uuid: int
    instruction_uuid: int
    offset: Optional[int]
    instruction: Optional[str]
    instruction_type: Optional[str]
    count: Optional[int]
    issue_count: Optional[int]
    stall_count: Optional[int]
    wave_occupancy: Optional[float]
    active_thread: Optional[float]
    stall_counts: Tuple[Optional[int], ...]
    source: Optional[str]
    code_object_id: int
    pid: int
