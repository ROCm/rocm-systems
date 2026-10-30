# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for layout_writer.py, on rows stored in the in-memory analysis DB."""

import argparse
import csv
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import common
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from argparser import omniarg_parser
from rocprof_compute_analyze import layout_writer
from rocprof_compute_analyze.analysis_db import db_analysis
from rocprof_compute_analyze.layout_writer import (
    DISPATCHES_FILE_NAME,
    KERNELS_FILE_NAME,
    ML_API_TRACE_FILE_NAME,
    PC_SAMPLING_NARROWING_NOTE,
    PC_SAMPLING_SUMMARY_FILE_NAME,
    ROOFLINE_FILE_NAME,
    WORKLOAD_METRICS_FILE_NAME,
    write_ml_api_trace_csv,
    write_workload_folders,
)
from utils.analysis_orm import (
    CodeObjectStore,
    Database,
    Dispatch,
    InstructionLine,
    InstructionSourceLine,
    Kernel,
    KernelMetricValue,
    KernelSymbol,
    MetricDefinition,
    PCSampleStallReason,
    PCSampleStallReasonLookup,
    PCSampleState,
    RooflineCeiling,
    SourceFile,
    SourceLine,
    Workload,
    WorkloadMetricValue,
)
from utils.specs import generate_machine_specs

WORKLOADS = Path(__file__).resolve().parents[2] / "workloads"
WORKLOAD_PATH = "/data/vcopy/MI300X_A1"
WORKLOAD_FOLDER = "1-MI300X_A1"
VECCOPY_NAME = "vecCopy(double*, double*, double*, int, int) (.kd)"
VECCOPY_FOLDER = "kernel-000-vecCopy"
SCALE_NAME = "scale<float>(float*, float)"
SCALE_FOLDER = "kernel-001-scale"


def make_workload(session: Session, total_duration_ns: Optional[int] = 400) -> Workload:
    """Add the vcopy workload, named the way build_database names it."""
    workload = Workload(
        name="vcopy",
        sub_name="MI300X_A1",
        total_duration_ns=total_duration_ns,
    )
    session.add(workload)
    return workload


def add_kernel(
    session: Session, workload: Workload, name: str, durations: List[int]
) -> Kernel:
    """Add a kernel with one dispatch per duration, each lasting that many ns."""
    kernel = Kernel(kernel_name=name, workload=workload)
    session.add(kernel)
    for dispatch_id, duration in enumerate(durations):
        session.add(
            Dispatch(
                dispatch_id=dispatch_id,
                gpu_id=0,
                start_timestamp=100,
                end_timestamp=100 + duration,
                kernel=kernel,
            )
        )
    return kernel


def add_metric(
    session: Session,
    workload: Workload,
    metric_id: str,
    name: str,
    unit: str,
    table_name: str = "Wavefront",
    sub_table_name: str = "Wavefront stats",
) -> MetricDefinition:
    """Add a metric definition of the workload."""
    metric = MetricDefinition(
        workload=workload,
        metric_id=metric_id,
        name=name,
        unit=unit,
        table_name=table_name,
        sub_table_name=sub_table_name,
    )
    session.add(metric)
    return metric


def add_kernel_value(
    session: Session,
    metric: MetricDefinition,
    kernel: Kernel,
    value_name: str,
    value: Optional[float],
) -> None:
    """Add one value of a metric for a kernel."""
    session.add(
        KernelMetricValue(
            metric=metric,
            kernel=kernel,
            value_name=value_name,
            value=value,
        )
    )


def add_sampled_instruction(
    session: Session,
    kernel_symbol: KernelSymbol,
    offset: int,
    frames: List[Tuple[str, Optional[int]]],
    counts: Tuple[int, int, int],
    stall_reasons: Optional[dict] = None,
) -> None:
    """Add an instruction line with its sample counts and inline stack.

    counts holds the total, issue and stall counts. Frames run innermost first.
    """
    instruction_line = InstructionLine(
        kernel_symbol=kernel_symbol,
        code_object_offset=offset,
        instruction="v_mov_b32",
    )
    add_frames(session, kernel_symbol, instruction_line, frames)
    sample_state = PCSampleState(
        instruction_line=instruction_line,
        total_count=counts[0],
        issue_count=counts[1],
        stall_count=counts[2],
    )
    session.add(sample_state)
    for reason_text, reason_count in (stall_reasons or {}).items():
        session.add(
            PCSampleStallReason(
                pc_sample_state=sample_state,
                stall_reason_lookup=Database.get_or_create_type(
                    PCSampleStallReasonLookup, reason_text
                ),
                count=reason_count,
            )
        )


def add_unsampled_instruction(
    session: Session, kernel_symbol: KernelSymbol, offset: int
) -> None:
    """Add a disassembled instruction that no sample landed on."""
    session.add(
        InstructionLine(
            kernel_symbol=kernel_symbol,
            code_object_offset=offset,
            instruction="s_nop",
        )
    )


def add_frames(
    session: Session,
    kernel_symbol: KernelSymbol,
    instruction_line: InstructionLine,
    frames: List[Tuple[str, Optional[int]]],
) -> None:
    """Attach an inline stack of (file, line) frames to an instruction line."""
    workload = kernel_symbol.code_object_store.workload
    for frame_index, (file_path, line_number) in enumerate(frames):
        source_file = next(
            (
                candidate
                for candidate in workload.source_files
                if candidate.file_path == file_path
            ),
            None,
        )
        if source_file is None:
            source_file = SourceFile(workload=workload, file_path=file_path)
            session.add(source_file)
        source_line = next(
            (
                candidate
                for candidate in source_file.source_lines
                if candidate.line_number == line_number
            ),
            None,
        )
        if source_line is None:
            source_line = SourceLine(
                source_file=source_file,
                line_number=line_number,
                content=None if line_number is None else f"code line {line_number}",
            )
            session.add(source_line)
        session.add(
            InstructionSourceLine(
                instruction_line=instruction_line,
                source_line=source_line,
                frame_index=frame_index,
            )
        )


def add_kernel_symbol(
    session: Session, workload: Workload, kernel: Kernel, pid: int = 1
) -> KernelSymbol:
    """Add a process's code object and the symbol that holds a kernel's code."""
    code_object = CodeObjectStore(
        workload=workload,
        pid=pid,
        code_object_id=0,
        load_base=0,
    )
    kernel_symbol = KernelSymbol(
        code_object_store=code_object,
        kernel=kernel,
        code_object_offset=0,
    )
    session.add(kernel_symbol)
    return kernel_symbol


def duration_ranking(workload_path: str) -> List[str]:
    """Return a workload's kernel names by duration, as a run without -k ranks them."""
    workload_name, workload_sub_name = workload_path.split("/")[-2:]
    names = (
        Database
        .get_session()
        .execute(
            select(Kernel.kernel_name)
            .join(Dispatch, Dispatch.kernel_uuid == Kernel.kernel_uuid)
            .join(Workload, Workload.workload_id == Kernel.workload_id)
            .where(
                Workload.name == workload_name,
                Workload.sub_name == workload_sub_name,
            )
            .group_by(Kernel.kernel_name)
            .order_by(
                func.sum(Dispatch.end_timestamp - Dispatch.start_timestamp).desc(),
                Kernel.kernel_name,
            )
        )
        .scalars()
        .all()
    )
    return list(names)


def write_run(
    tmp_path: Path,
    workload_paths: Optional[List[str]] = None,
    kernel_rankings: Optional[Dict[str, List[str]]] = None,
) -> Path:
    """Commit the DB the way the analysis does, then write the run folder.

    Without kernel_rankings, each workload is ranked by its kernels' durations.
    """
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    Database.commit()
    paths = workload_paths or [WORKLOAD_PATH]
    if kernel_rankings is None:
        kernel_rankings = {path: duration_ranking(path) for path in paths}
    write_workload_folders(run_folder, workload_ids_of(paths), kernel_rankings)
    return run_folder


def workload_ids_of(paths: List[str]) -> Dict[str, int]:
    """Return the analysis DB id of each workload path, as build_database stored it."""
    session = Database.get_session()
    return {
        path: session.execute(
            select(Workload.workload_id).where(
                Workload.name == path.split("/")[-2],
                Workload.sub_name == path.split("/")[-1],
            )
        ).scalar_one()
        for path in paths
    }


def read_csv(csv_path: Path) -> List[List[str]]:
    """Return the rows of a CSV file, header included."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.reader(csv_file))


def parse_analyze_args(workload_path: str) -> argparse.Namespace:
    """Parse an analyze command line for one workload, with the real options."""
    parser = argparse.ArgumentParser(prog="tool")
    omniarg_parser(
        parser,
        Path.cwd(),
        common.SUPPORTED_ARCHS,
        {"ver_pretty": "rocprof-compute (unit test)"},
    )
    return parser.parse_args(["analyze", "--path", workload_path])


def build_real_run(workload_name: str, tmp_path: Path) -> Path:
    """Analyze a checked-in workload in memory and write its run folder.

    The analyzer runs the way the analyze CLI does, with the SoC of the
    workload's sysinfo.csv, and the database is closed afterwards.
    """
    workload_path = str((WORKLOADS / workload_name).resolve())
    args = parse_analyze_args(workload_path)
    sys_info = {
        key: value[0]
        for key, value in pd
        .read_csv(f"{workload_path}/sysinfo.csv")
        .to_dict("list")
        .items()
    }
    machine_specs = generate_machine_specs(args, sys_info)
    arch = machine_specs.gpu_arch
    soc_class = getattr(
        importlib.import_module(f"rocprof_compute_soc.soc_{arch}"), f"{arch}_soc"
    )
    analyzer = db_analysis(args, common.SUPPORTED_ARCHS)
    analyzer.sanitize()
    analyzer.set_soc({arch: soc_class(args, machine_specs)})
    analyzer.pre_processing()
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    try:
        analyzer.build_database(run_folder / "analysis.db")
        write_workload_folders(
            run_folder,
            analyzer.workload_ids(),
            analyzer._kernel_rankings_per_workload,
        )
    finally:
        Database.close()
    return run_folder


def test_writes_the_single_workload_file_set(db_session, tmp_path):
    """A workload without PC sampling or roofline data gets the core files only."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    scale = add_kernel(db_session, workload, SCALE_NAME, [100])
    wave_count = add_metric(db_session, workload, "1.1", "Wavefronts", "Count")
    add_kernel_value(db_session, wave_count, veccopy, "Avg", 8.0)
    add_kernel_value(db_session, wave_count, scale, "Avg", 2.0)
    db_session.add(
        WorkloadMetricValue(
            metric=wave_count,
            workload=workload,
            value_name="Avg",
            value=10.0,
        )
    )

    run_folder = write_run(tmp_path)

    workload_folder = run_folder / WORKLOAD_FOLDER
    assert sorted(path.name for path in workload_folder.iterdir()) == sorted([
        KERNELS_FILE_NAME,
        DISPATCHES_FILE_NAME,
        WORKLOAD_METRICS_FILE_NAME,
        VECCOPY_FOLDER,
        SCALE_FOLDER,
    ])
    assert sorted(
        path.name for path in (workload_folder / VECCOPY_FOLDER).iterdir()
    ) == [f"{VECCOPY_FOLDER}-metrics.csv"]


def test_kernels_csv_lists_top_stats_order_with_pct_of_the_workload(
    db_session, tmp_path
):
    """Kernels are ordered by total duration, and Pct divides by the workload total."""
    workload = make_workload(db_session, total_duration_ns=400)
    add_kernel(db_session, workload, SCALE_NAME, [100])
    add_kernel(db_session, workload, VECCOPY_NAME, [200, 100])

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / KERNELS_FILE_NAME)
    assert rows == [
        ["Index", "Kernel_Name", "Folder", "Duration_ns", "Pct"],
        ["0", VECCOPY_NAME, VECCOPY_FOLDER, "300", "75.0"],
        ["1", SCALE_NAME, SCALE_FOLDER, "100", "25.0"],
    ]


@pytest.mark.parametrize("total_duration_ns", [None, 0])
def test_kernels_csv_leaves_pct_empty_without_a_workload_total(
    db_session, tmp_path, total_duration_ns
):
    """A workload with no total duration, or a total of 0, writes an empty Pct cell."""
    workload = make_workload(db_session, total_duration_ns=total_duration_ns)
    add_kernel(db_session, workload, VECCOPY_NAME, [300])

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / KERNELS_FILE_NAME)
    assert rows[1] == ["0", VECCOPY_NAME, VECCOPY_FOLDER, "300", ""]


def test_kernel_index_is_its_top_stats_rank_when_k_leaves_kernels_out(
    db_session, tmp_path
):
    """With -k, a kept kernel keeps the index of its top-stats rank, not its row."""
    workload = make_workload(db_session)
    add_kernel(db_session, workload, SCALE_NAME, [100])

    run_folder = write_run(
        tmp_path, kernel_rankings={WORKLOAD_PATH: [VECCOPY_NAME, SCALE_NAME]}
    )

    workload_folder = run_folder / WORKLOAD_FOLDER
    kernel_rows = read_csv(workload_folder / KERNELS_FILE_NAME)
    assert kernel_rows[1][:3] == ["1", SCALE_NAME, SCALE_FOLDER]
    dispatch_rows = read_csv(workload_folder / DISPATCHES_FILE_NAME)
    assert dispatch_rows[1][1] == "1"
    assert not (workload_folder / VECCOPY_FOLDER).exists()


def test_kernel_missing_from_the_top_stats_ranking_stops_before_the_write(db_session):
    """A kernel with rows in the DB but no place in the ranking is an error.

    The check runs in the build phase, before any run folder is created.
    """
    workload = make_workload(db_session)
    add_kernel(db_session, workload, VECCOPY_NAME, [300])
    Database.commit()

    with pytest.raises(SystemExit):
        layout_writer.check_workload_folders(
            workload_ids_of([WORKLOAD_PATH]), {WORKLOAD_PATH: [SCALE_NAME]}
        )


def test_mangled_kernel_folder_uses_the_demangled_name_and_kernels_csv_the_raw_one(
    db_session, tmp_path, monkeypatch
):
    """A _Z kernel's folder takes its demangled name; kernels.csv keeps the raw one."""
    mangled_name = "_Z6vecCopyPdS_S_iii"
    monkeypatch.setattr(
        layout_writer,
        "demangle_kernel_names",
        lambda kernel_names: {mangled_name: VECCOPY_NAME},
    )
    workload = make_workload(db_session)
    add_kernel(db_session, workload, mangled_name, [300])

    run_folder = write_run(tmp_path)

    workload_folder = run_folder / WORKLOAD_FOLDER
    kernel_rows = read_csv(workload_folder / KERNELS_FILE_NAME)
    assert kernel_rows[1][:3] == ["0", mangled_name, VECCOPY_FOLDER]
    assert (workload_folder / VECCOPY_FOLDER).is_dir()
    dispatch_rows = read_csv(workload_folder / DISPATCHES_FILE_NAME)
    assert dispatch_rows[1][2] == mangled_name


def test_dispatches_csv_maps_each_dispatch_to_its_kernel_index(db_session, tmp_path):
    """Each dispatch row names its kernel's index and gives its duration in ns."""
    workload = make_workload(db_session)
    add_kernel(db_session, workload, SCALE_NAME, [100])
    add_kernel(db_session, workload, VECCOPY_NAME, [300])

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / DISPATCHES_FILE_NAME)
    assert rows == [
        [
            "Dispatch_ID",
            "Kernel_Index",
            "Kernel_Name",
            "GPU_ID",
            "Start_Timestamp",
            "End_Timestamp",
            "Duration_ns",
        ],
        ["0", "1", SCALE_NAME, "0", "100", "200", "100"],
        ["0", "0", VECCOPY_NAME, "0", "100", "400", "300"],
    ]


def test_dispatch_with_null_timestamps_writes_empty_cells(db_session, tmp_path):
    """A dispatch without timestamps has empty timestamp and duration cells."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    db_session.add(
        Dispatch(
            dispatch_id=1,
            gpu_id=0,
            start_timestamp=None,
            end_timestamp=None,
            kernel=veccopy,
        )
    )

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / DISPATCHES_FILE_NAME)
    assert rows[2] == ["1", "0", VECCOPY_NAME, "0", "", "", ""]


def test_workload_metric_with_a_null_value_writes_an_empty_cell(db_session, tmp_path):
    """A workload metric whose value is NULL leaves its cell empty."""
    workload = make_workload(db_session)
    add_kernel(db_session, workload, VECCOPY_NAME, [300])
    wave_count = add_metric(db_session, workload, "1.1", "Wavefronts", "Count")
    db_session.add(
        WorkloadMetricValue(
            metric=wave_count,
            workload=workload,
            value_name="Avg",
            value=None,
        )
    )

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / WORKLOAD_METRICS_FILE_NAME)
    assert rows == [
        ["Metric_ID", "Metric", "Unit", "Table", "Sub_Table", "Avg"],
        ["1.1", "Wavefronts", "Count", "Wavefront", "Wavefront stats", ""],
    ]


def test_kernel_metrics_pivot_one_column_per_value_name_and_leave_nulls_empty(
    db_session, tmp_path
):
    """A metric's values share a row, and a missing or NULL value is an empty cell."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    fetch_size = add_metric(db_session, workload, "2.1", "Fetch Size", "KB")
    add_kernel_value(db_session, fetch_size, veccopy, "Avg", 1.5)
    add_kernel_value(db_session, fetch_size, veccopy, "Min", None)
    hit_rate = add_metric(db_session, workload, "2.2", "Hit Rate", "Percent")
    add_kernel_value(db_session, hit_rate, veccopy, "Avg", 90.0)

    run_folder = write_run(tmp_path)

    rows = read_csv(
        run_folder / WORKLOAD_FOLDER / VECCOPY_FOLDER / f"{VECCOPY_FOLDER}-metrics.csv"
    )
    assert rows == [
        ["Metric_ID", "Metric", "Unit", "Table", "Sub_Table", "Avg", "Min"],
        ["2.1", "Fetch Size", "KB", "Wavefront", "Wavefront stats", "1.5", ""],
        ["2.2", "Hit Rate", "Percent", "Wavefront", "Wavefront stats", "90.0", ""],
    ]


def test_roofline_csv_holds_ceilings_and_the_kernel_plot_points(db_session, tmp_path):
    """roofline.csv lists each ceiling, then each kernel's plot points."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    db_session.add(
        RooflineCeiling(
            workload=workload,
            kind="bandwidth",
            memory_level="HBM",
            unit="Bytes/s",
            value=1.0e12,
        )
    )
    db_session.add(
        RooflineCeiling(
            workload=workload,
            kind="compute",
            data_type="FP32",
            pipe="VALU",
            unit="FLOP/s",
            value=2.0e13,
        )
    )
    intensity = add_metric(db_session, workload, "402.1", "AI HBM", "FLOP/byte")
    add_kernel_value(db_session, intensity, veccopy, "Value", 2.5)

    run_folder = write_run(tmp_path)

    rows = read_csv(run_folder / WORKLOAD_FOLDER / ROOFLINE_FILE_NAME)
    assert rows[0] == [
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
    assert [row[0] for row in rows[1:]] == ["bandwidth", "compute", "point"]
    assert rows[1][1:4] == ["HBM", "", ""]
    assert rows[2][1:4] == ["", "FP32", "VALU"]
    assert rows[3] == [
        "point",
        "",
        "",
        "",
        "0",
        VECCOPY_NAME,
        "AI HBM",
        "FLOP/byte",
        "2.5",
    ]


def test_roofline_csv_is_absent_without_roofline_data(db_session, tmp_path):
    """No ceilings and no plot points means no roofline.csv."""
    workload = make_workload(db_session)
    add_kernel(db_session, workload, VECCOPY_NAME, [300])

    run_folder = write_run(tmp_path)

    assert not (run_folder / WORKLOAD_FOLDER / ROOFLINE_FILE_NAME).exists()


def test_pc_sampling_files_cover_only_kernels_with_samples(db_session, tmp_path):
    """Sampled kernels get ISA and source files, and the summary lists them."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    add_kernel(db_session, workload, SCALE_NAME, [100])
    veccopy_symbol = add_kernel_symbol(db_session, workload, veccopy)
    add_sampled_instruction(
        db_session,
        veccopy_symbol,
        offset=16,
        frames=[("/src/vcopy.cpp", 4)],
        counts=(3, 1, 2),
        stall_reasons={"WAITCNT": 2},
    )
    add_sampled_instruction(
        db_session,
        veccopy_symbol,
        offset=32,
        frames=[],
        counts=(1, 0, 1),
        stall_reasons={"NONE": 1},
    )
    add_unsampled_instruction(db_session, veccopy_symbol, offset=48)

    run_folder = write_run(tmp_path)

    workload_folder = run_folder / WORKLOAD_FOLDER
    kernel_folder = workload_folder / VECCOPY_FOLDER
    assert (kernel_folder / f"{VECCOPY_FOLDER}-pc-sampling-isa.csv").is_file()
    assert (kernel_folder / f"{VECCOPY_FOLDER}-pc-sampling-src.csv").is_file()
    assert not (
        workload_folder / SCALE_FOLDER / f"{SCALE_FOLDER}-pc-sampling-isa.csv"
    ).exists()

    summary_lines = (
        (workload_folder / PC_SAMPLING_SUMMARY_FILE_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert summary_lines[0] == PC_SAMPLING_NARROWING_NOTE
    summary_rows = list(csv.reader(summary_lines[1:]))
    assert summary_rows[0] == [
        "Index",
        "Kernel_Name",
        "Folder",
        "Total count",
        "Active count",
        "Stall count",
        "Stall NONE",
        "Stall WAITCNT",
    ]
    assert summary_rows[1] == [
        "0",
        VECCOPY_NAME,
        VECCOPY_FOLDER,
        "4",
        "1",
        "3",
        "1",
        "2",
    ]
    assert len(summary_rows) == 2


def test_pc_sampling_isa_file_numbers_lines_and_keeps_unsampled_ones_empty(
    db_session, tmp_path
):
    """The ISA file lists every instruction of the kernel in offset order."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    veccopy_symbol = add_kernel_symbol(db_session, workload, veccopy)
    add_sampled_instruction(
        db_session,
        veccopy_symbol,
        offset=16,
        frames=[("/src/vcopy.cpp", 4)],
        counts=(3, 1, 2),
        stall_reasons={"WAITCNT": 2},
    )
    add_unsampled_instruction(db_session, veccopy_symbol, offset=32)

    run_folder = write_run(tmp_path)

    isa_path = (
        run_folder
        / WORKLOAD_FOLDER
        / VECCOPY_FOLDER
        / f"{VECCOPY_FOLDER}-pc-sampling-isa.csv"
    )
    rows = read_csv(isa_path)
    assert rows[0] == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
        "Instruction type",
        "Total count",
        "Active count",
        "Stall count",
        "Wave occupancy percent",
        "Active thread percent",
        "Stall WAITCNT",
        "Source",
        "Code object id",
        "Pid",
    ]
    assert rows[1] == [
        "1",
        "16",
        "v_mov_b32",
        "",
        "3",
        "1",
        "2",
        "",
        "",
        "2",
        "/src/vcopy.cpp:4",
        "0",
        "1",
    ]
    assert rows[2] == ["2", "32", "s_nop", "", "", "", "", "", "", "", "", "0", "1"]


def test_pc_sampling_files_keep_each_process_row_at_a_repeated_offset(
    db_session, tmp_path
):
    """A kernel run by two processes has both rows at one offset, told apart by Pid."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    for pid in (1, 2):
        kernel_symbol = add_kernel_symbol(db_session, workload, veccopy, pid=pid)
        add_sampled_instruction(
            db_session,
            kernel_symbol,
            offset=16,
            frames=[("/src/vcopy.cpp", 4)],
            counts=(3, 1, 2),
        )

    run_folder = write_run(tmp_path)

    kernel_folder = run_folder / WORKLOAD_FOLDER / VECCOPY_FOLDER
    isa_rows = read_csv(kernel_folder / f"{VECCOPY_FOLDER}-pc-sampling-isa.csv")[1:]
    assert [row[1] for row in isa_rows] == ["16", "16"]
    assert sorted(row[-1] for row in isa_rows) == ["1", "2"]
    source_rows = read_csv(kernel_folder / f"{VECCOPY_FOLDER}-pc-sampling-src.csv")
    assert source_rows[1] == ["/src/vcopy.cpp", "4", "code line 4", "6", "2", "4"]


def test_pc_sampling_source_file_sums_each_line_and_reports_unattributable_samples(
    db_session, tmp_path
):
    """Each source line sums the samples of the instructions in its inline stack."""
    workload = make_workload(db_session)
    veccopy = add_kernel(db_session, workload, VECCOPY_NAME, [300])
    veccopy_symbol = add_kernel_symbol(db_session, workload, veccopy)
    add_sampled_instruction(
        db_session,
        veccopy_symbol,
        offset=16,
        frames=[("/src/vcopy.cpp", 4), ("/src/inline.h", 9)],
        counts=(3, 1, 2),
    )
    add_sampled_instruction(
        db_session,
        veccopy_symbol,
        offset=32,
        frames=[("/src/vcopy.cpp", 4)],
        counts=(2, 2, 0),
    )
    add_sampled_instruction(
        db_session, veccopy_symbol, offset=48, frames=[], counts=(1, 0, 1)
    )

    run_folder = write_run(tmp_path)

    rows = read_csv(
        run_folder
        / WORKLOAD_FOLDER
        / VECCOPY_FOLDER
        / f"{VECCOPY_FOLDER}-pc-sampling-src.csv"
    )
    assert rows == [
        [
            "Source file",
            "Source line number",
            "Source line",
            "Total count",
            "Active count",
            "Stall count",
        ],
        ["/src/inline.h", "9", "code line 9", "3", "1", "2"],
        ["/src/vcopy.cpp", "4", "code line 4", "5", "3", "2"],
        ["", "", "Unattributable instructions", "1", "0", "1"],
    ]


def test_ml_api_trace_csv_writes_the_trace_with_empty_missing_cells(tmp_path):
    """The trace is written as given, with NaN and None cells left empty."""
    trace = pd.DataFrame({
        "Function": ["torch.add", "torch.mul"],
        "Start_Timestamp": [10, 20],
        "File_Name": [None, "model.py"],
        "Line_Number": [np.nan, 42],
    })
    workload_folder = tmp_path / WORKLOAD_FOLDER
    workload_folder.mkdir()

    trace_path = write_ml_api_trace_csv(workload_folder, trace)

    assert trace_path == workload_folder / ML_API_TRACE_FILE_NAME
    assert read_csv(trace_path) == [
        ["Function", "Start_Timestamp", "File_Name", "Line_Number"],
        ["torch.add", "10", "", ""],
        ["torch.mul", "20", "model.py", "42.0"],
    ]


def test_each_workload_gets_a_folder_named_by_its_position(db_session, tmp_path):
    """Two -p workloads write two folders, numbered by their -p position."""
    first = make_workload(db_session)
    add_kernel(db_session, first, VECCOPY_NAME, [300])
    second = Workload(name="vcopy", sub_name="MI350", total_duration_ns=50)
    db_session.add(second)
    add_kernel(db_session, second, SCALE_NAME, [50])

    run_folder = write_run(tmp_path, [WORKLOAD_PATH, "/data/vcopy/MI350"])

    assert (run_folder / WORKLOAD_FOLDER / KERNELS_FILE_NAME).is_file()
    second_rows = read_csv(run_folder / "2-MI350" / KERNELS_FILE_NAME)
    assert second_rows[1][:3] == ["0", SCALE_NAME, "kernel-000-scale"]


@pytest.mark.parametrize(
    ("kernel_count", "first_folder", "last_folder"),
    [
        (1000, "kernel-000-k0", "kernel-999-k999"),
        (1001, "kernel-0000-k0", "kernel-1000-k1000"),
    ],
)
def test_kernel_index_width_grows_past_the_last_three_digit_index(
    db_session, tmp_path, kernel_count, first_folder, last_folder
):
    """With 1000 kernels the last index is 999, and with 1001 it is 1000."""
    workload = make_workload(db_session, total_duration_ns=None)
    for kernel_number in range(kernel_count):
        add_kernel(
            db_session, workload, f"k{kernel_number}", [kernel_count - kernel_number]
        )

    run_folder = write_run(tmp_path)

    kernel_rows = read_csv(run_folder / WORKLOAD_FOLDER / KERNELS_FILE_NAME)[1:]
    assert kernel_rows[0][2] == first_folder
    assert kernel_rows[-1][2] == last_folder
    assert len(kernel_rows) == kernel_count


def test_real_single_workload_writes_the_core_files(tmp_path):
    """A checked-in vcopy workload gets the kernel, dispatch and metric files."""
    run_folder = build_real_run("vcopy/MI300X_A1", tmp_path)

    workload_folder = run_folder / "1-MI300X_A1"
    assert sorted(path.name for path in workload_folder.iterdir()) == [
        "dispatches.csv",
        "kernel-000-vecCopy",
        "kernels.csv",
        "metrics-summary.csv",
    ]
    kernel_rows = read_csv(workload_folder / KERNELS_FILE_NAME)
    assert kernel_rows[1][:3] == [
        "0",
        "vecCopy(double*, double*, double*, int, int) (.kd)",
        "kernel-000-vecCopy",
    ]


def test_real_pc_sampling_workload_writes_the_sampling_files(tmp_path):
    """The checked-in PC-sampling-only MI350 workload gets its sampling files."""
    run_folder = build_real_run("vcopy_pc_sampling_only/MI350", tmp_path)

    workload_folder = run_folder / "1-MI350"
    kernel_folder = workload_folder / "kernel-000-vecCopy"
    assert sorted(path.name for path in kernel_folder.iterdir()) == [
        "kernel-000-vecCopy-metrics.csv",
        "kernel-000-vecCopy-pc-sampling-isa.csv",
        "kernel-000-vecCopy-pc-sampling-src.csv",
    ]
    summary_lines = (
        (workload_folder / PC_SAMPLING_SUMMARY_FILE_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert summary_lines[0] == PC_SAMPLING_NARROWING_NOTE
    summary_rows = list(csv.reader(summary_lines[1:]))
    assert summary_rows[1][:3] == [
        "0",
        "vecCopy(double*, double*, double*, int, int)",
        "kernel-000-vecCopy",
    ]
    assert int(summary_rows[1][3]) > 0


def test_real_roofline_workload_writes_the_ceilings(tmp_path):
    """A checked-in roofline workload gets a roofline.csv with its ceilings."""
    run_folder = build_real_run("mem_levels_HBM/MI200", tmp_path)

    roofline_rows = read_csv(run_folder / "1-MI200" / ROOFLINE_FILE_NAME)
    kinds = {row[0] for row in roofline_rows[1:]}
    assert kinds == {"bandwidth", "compute"}
    assert ["bandwidth", "HBM", "", "", "", "", "", "Bytes/s"] == roofline_rows[1][:8]


def test_ml_api_trace_is_written_beside_the_kernel_files_of_a_workload(
    db_session, tmp_path
):
    """The ML API trace lands in the workload folder with the kernel files."""
    workload = make_workload(db_session)
    add_kernel(db_session, workload, VECCOPY_NAME, [300])
    run_folder = write_run(tmp_path)
    workload_folder = run_folder / WORKLOAD_FOLDER

    write_ml_api_trace_csv(
        workload_folder,
        pd.DataFrame({"Function": ["torch.add"], "Start_Timestamp": [10]}),
    )

    assert read_csv(workload_folder / ML_API_TRACE_FILE_NAME) == [
        ["Function", "Start_Timestamp"],
        ["torch.add", "10"],
    ]
    assert (workload_folder / KERNELS_FILE_NAME).is_file()
