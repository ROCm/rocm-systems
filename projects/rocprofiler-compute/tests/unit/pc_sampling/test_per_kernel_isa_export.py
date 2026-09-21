# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for per_kernel_isa_export.py."""

import csv
from pathlib import Path

from pc_sampling.per_kernel_isa_export import (
    _build_isa_header,
    _resolve_isa_export_path,
    _write_per_kernel_isa_files,
    export_per_kernel_isa_files,
)


def make_isa_row(
    kernel_uuid=1,
    code_object_id=5,
    pid=42,
    offset=0x10,
    instruction="v_mov",
    total_count=3,
    issue_count=2,
    stall_count=1,
    stall_reason_counts=(),
    source="/s/a.cpp:1",
    workload_name="vector_copy",
    workload_sub_name="MI300X_A1",
):
    """Build one row as the per-kernel ISA select returns it."""
    return (
        workload_name,
        workload_sub_name,
        kernel_uuid,
        code_object_id,
        pid,
        offset,
        instruction,
        total_count,
        issue_count,
        stall_count,
        None,
        None,
        *stall_reason_counts,
        source,
        code_object_id,
        pid,
    )


def read_exported_rows(export_path):
    """Return one exported file as a list of string rows."""
    with export_path.open(newline="", encoding="utf-8") as export_file:
        return list(csv.reader(export_file))


def test_resolve_isa_export_path_names_kernel_code_object_and_process():
    """One kernel's ISA is filed by uuid, then by the code object and process."""
    export_path = _resolve_isa_export_path(
        Path("/results/per_kernel_pc_sampling"),
        ("vector_copy", "MI300X_A1", 7, 5, 42),
    )

    assert export_path == Path(
        "/results/per_kernel_pc_sampling/vector_copy/MI300X_A1"
        "/kernel_7/isa_code_object_id_5_pid_42.csv"
    )


def test_build_isa_header_places_stall_reasons_between_counts_and_source():
    """The reasons a workload observed become columns of their own."""
    header = _build_isa_header(["NONE", "WAITCNT"])

    assert header == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
        "Total count",
        "Active count",
        "Stall count",
        "Wave occupancy percent",
        "Active thread percent",
        "Stall NONE",
        "Stall WAITCNT",
        "Source",
        "Code object id",
        "Pid",
    ]


def test_build_isa_header_carries_no_stall_columns_without_reasons():
    """A host_trap workload records no stall reason, so it gets no column."""
    header = _build_isa_header([])

    assert "Stall NONE" not in header
    assert header[-3:] == ["Source", "Code object id", "Pid"]


def test_write_per_kernel_isa_files_opens_one_file_per_grouping_key(tmp_path):
    """A kernel spanning code objects and processes lands in separate files."""
    isa_rows = [
        make_isa_row(kernel_uuid=1, code_object_id=5, pid=42),
        make_isa_row(kernel_uuid=1, code_object_id=6, pid=42),
        make_isa_row(kernel_uuid=1, code_object_id=6, pid=43),
        make_isa_row(kernel_uuid=2, code_object_id=6, pid=43),
    ]

    file_count = _write_per_kernel_isa_files(tmp_path, iter(isa_rows), [])

    assert file_count == 4
    assert sorted(
        str(export_path.relative_to(tmp_path))
        for export_path in tmp_path.rglob("*.csv")
    ) == [
        "vector_copy/MI300X_A1/kernel_1/isa_code_object_id_5_pid_42.csv",
        "vector_copy/MI300X_A1/kernel_1/isa_code_object_id_6_pid_42.csv",
        "vector_copy/MI300X_A1/kernel_1/isa_code_object_id_6_pid_43.csv",
        "vector_copy/MI300X_A1/kernel_2/isa_code_object_id_6_pid_43.csv",
    ]


def test_write_per_kernel_isa_files_numbers_instruction_lines_per_file(tmp_path):
    """Line numbers count the file's own lines, not the kernel's."""
    isa_rows = [
        make_isa_row(code_object_id=5, offset=0x10),
        make_isa_row(code_object_id=5, offset=0x14),
        make_isa_row(code_object_id=6, offset=0x20),
    ]

    _write_per_kernel_isa_files(tmp_path, iter(isa_rows), [])

    workload_directory = tmp_path / "vector_copy" / "MI300X_A1" / "kernel_1"
    first_rows = read_exported_rows(
        workload_directory / "isa_code_object_id_5_pid_42.csv"
    )
    second_rows = read_exported_rows(
        workload_directory / "isa_code_object_id_6_pid_42.csv"
    )

    assert [row[0] for row in first_rows[1:]] == ["1", "2"]
    assert [row[0] for row in second_rows[1:]] == ["1"]


def test_write_per_kernel_isa_files_leaves_unsampled_counts_empty(tmp_path):
    """A disassembled line no sample landed on carries no counts."""
    isa_rows = [
        make_isa_row(
            offset=0x10,
            total_count=None,
            issue_count=None,
            stall_count=None,
            stall_reason_counts=(None,),
            source=None,
        ),
        make_isa_row(offset=0x14, stall_reason_counts=(4,)),
    ]

    _write_per_kernel_isa_files(tmp_path, iter(isa_rows), ["WAITCNT"])

    exported_rows = read_exported_rows(
        tmp_path
        / "vector_copy"
        / "MI300X_A1"
        / "kernel_1"
        / "isa_code_object_id_5_pid_42.csv"
    )
    assert exported_rows[1] == [
        "1",
        "16",
        "v_mov",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "5",
        "42",
    ]
    assert exported_rows[2][3:6] == ["3", "2", "1"]
    assert exported_rows[2][8] == "4"


def test_write_per_kernel_isa_files_writes_nothing_without_rows(tmp_path):
    """A workload with no instruction lines leaves no folder behind."""
    file_count = _write_per_kernel_isa_files(tmp_path / "per_kernel", iter([]), [])

    assert file_count == 0
    assert not (tmp_path / "per_kernel").exists()


def test_export_per_kernel_isa_files_returns_the_folder_it_wrote(tmp_path):
    """The export folder is returned so the source lands beside the ISA."""
    per_kernel_directory = export_per_kernel_isa_files(tmp_path, [])

    assert per_kernel_directory == tmp_path / "per_kernel_pc_sampling"
    assert not per_kernel_directory.exists()
