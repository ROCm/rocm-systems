# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for per_kernel_isa_export.py."""

import csv
from pathlib import Path

from pc_sampling.pc_sampling_csv_export import IsaLineLocation
from pc_sampling.per_kernel_isa_export import (
    WorkloadIsaExport,
    _resolve_isa_export_path,
    _write_per_kernel_isa_files,
    export_per_kernel_isa_files,
)


def make_kernel_directories(root):
    """Return the kernel folders under root, keyed by kernel uuid."""
    return {1: root / "kernel-0", 2: root / "kernel-1"}


def make_isa_row(
    kernel_uuid=1,
    short_name="vecCopy",
    code_object_id=5,
    pid=42,
    instruction_uuid=100,
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
        short_name,
        code_object_id,
        pid,
        instruction_uuid,
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
    """One kernel's ISA is filed by short name and uuid, then by object and process."""
    export_path = _resolve_isa_export_path(
        make_kernel_directories(Path("/results/workload-1")),
        ("vector_copy", "MI300X_A1", 2, "vecCopy_2", 5, 42),
    )

    assert export_path == Path(
        "/results/workload-1/kernel-1/vecCopy_2_uuid_2/isa_code_object_id_5_pid_42.csv"
    )


def test_write_per_kernel_isa_files_places_stall_reasons_between_counts_and_source(
    tmp_path,
):
    """The reasons a workload observed become columns of their own."""
    _write_per_kernel_isa_files(
        iter([make_isa_row(stall_reason_counts=(None, 1))]),
        ["NONE", "WAITCNT"],
        {},
        make_kernel_directories(tmp_path),
    )

    header = read_exported_rows(
        tmp_path / "kernel-0" / "vecCopy_uuid_1" / "isa_code_object_id_5_pid_42.csv"
    )[0]
    assert header == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
        "Instruction type",
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


def test_write_per_kernel_isa_files_opens_one_file_per_grouping_key(tmp_path):
    """A kernel spanning code objects and processes lands in separate files."""
    isa_rows = [
        make_isa_row(kernel_uuid=1, code_object_id=5, pid=42),
        make_isa_row(kernel_uuid=1, code_object_id=6, pid=42),
        make_isa_row(kernel_uuid=1, code_object_id=6, pid=43),
        make_isa_row(kernel_uuid=2, code_object_id=6, pid=43),
    ]

    file_count = _write_per_kernel_isa_files(
        iter(isa_rows), [], {}, make_kernel_directories(tmp_path)
    )

    assert file_count == 4
    assert sorted(
        str(export_path.relative_to(tmp_path))
        for export_path in tmp_path.rglob("*.csv")
    ) == [
        "kernel-0/vecCopy_uuid_1/isa_code_object_id_5_pid_42.csv",
        "kernel-0/vecCopy_uuid_1/isa_code_object_id_6_pid_42.csv",
        "kernel-0/vecCopy_uuid_1/isa_code_object_id_6_pid_43.csv",
        "kernel-1/vecCopy_uuid_2/isa_code_object_id_6_pid_43.csv",
    ]


def test_write_per_kernel_isa_files_numbers_instruction_lines_per_file(tmp_path):
    """Line numbers count the file's own lines, not the kernel's.

    The location of each instruction is recorded with the same numbers.
    """
    isa_rows = [
        make_isa_row(code_object_id=5, instruction_uuid=100, offset=0x10),
        make_isa_row(code_object_id=5, instruction_uuid=101, offset=0x14),
        make_isa_row(code_object_id=6, instruction_uuid=102, offset=0x20),
    ]
    isa_line_locations = {}

    _write_per_kernel_isa_files(
        iter(isa_rows), [], isa_line_locations, make_kernel_directories(tmp_path)
    )

    kernel_directory = tmp_path / "kernel-0" / "vecCopy_uuid_1"
    first_rows = read_exported_rows(
        kernel_directory / "isa_code_object_id_5_pid_42.csv"
    )
    second_rows = read_exported_rows(
        kernel_directory / "isa_code_object_id_6_pid_42.csv"
    )

    assert [row[0] for row in first_rows[1:]] == ["1", "2"]
    assert [row[0] for row in second_rows[1:]] == ["1"]
    assert isa_line_locations == {
        100: IsaLineLocation(Path("vecCopy_uuid_1/isa_code_object_id_5_pid_42.csv"), 1),
        101: IsaLineLocation(Path("vecCopy_uuid_1/isa_code_object_id_5_pid_42.csv"), 2),
        102: IsaLineLocation(Path("vecCopy_uuid_1/isa_code_object_id_6_pid_42.csv"), 1),
    }


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

    _write_per_kernel_isa_files(
        iter(isa_rows), ["WAITCNT"], {}, make_kernel_directories(tmp_path)
    )

    exported_rows = read_exported_rows(
        tmp_path / "kernel-0" / "vecCopy_uuid_1" / "isa_code_object_id_5_pid_42.csv"
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
    file_count = _write_per_kernel_isa_files(
        iter([]), [], {}, make_kernel_directories(tmp_path)
    )

    assert file_count == 0
    assert not list(tmp_path.iterdir())


def test_export_per_kernel_isa_files_returns_each_instructions_location(tmp_path):
    """The returned locations let the source-line export reference the ISA."""
    isa_line_locations = export_per_kernel_isa_files([
        WorkloadIsaExport(
            1,
            [],
            iter([make_isa_row(instruction_uuid=100)]),
            make_kernel_directories(tmp_path),
        )
    ])

    assert isa_line_locations == {
        100: IsaLineLocation(Path("vecCopy_uuid_1/isa_code_object_id_5_pid_42.csv"), 1)
    }
