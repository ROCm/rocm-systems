# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for per_source_line_export.py."""

import csv
import json
from pathlib import Path

import pytest

from pc_sampling.per_source_line_export import (
    WorkloadSourceLineExport,
    _format_isa_references,
    _format_number_ranges,
    _resolve_source_line_export_path,
    _write_per_source_line_files,
    export_per_source_line_files,
)

ISA_REFERENCE_FIELDS = (
    "kernel_uuid",
    "short_name",
    "code_object_id",
    "pid",
    "instruction_line_number",
)
# The one reference that a source line without instructions carries.
NO_ISA_REFERENCE = (None, None, None, None, None)


def make_isa_references_json(*isa_references):
    """Encode references as the per-source-line select returns them."""
    return json.dumps([
        dict(zip(ISA_REFERENCE_FIELDS, isa_reference))
        for isa_reference in isa_references
    ])


def make_source_line_row(
    line_number=1,
    content="int a;",
    total_count=3,
    issue_count=2,
    stall_count=1,
    stall_reason_counts=(),
    isa_references=((7, "vecCopy", 5, 42, 1),),
    file_path="/s/a.cpp",
    workload_name="vector_copy",
    workload_sub_name="MI300X_A1",
):
    """Build one row as the per-source-line select returns it."""
    return (
        workload_name,
        workload_sub_name,
        file_path,
        line_number,
        content,
        total_count,
        issue_count,
        stall_count,
        None,
        None,
        *stall_reason_counts,
        make_isa_references_json(*isa_references),
    )


def read_exported_rows(export_path):
    """Return one exported file as a list of string rows."""
    with export_path.open(newline="", encoding="utf-8") as export_file:
        return list(csv.reader(export_file))


def test_resolve_source_line_export_path_sits_beside_the_source_copy():
    """The file is named after the exported copy of the source file."""
    export_path = _resolve_source_line_export_path(
        Path("/results/per_kernel_pc_sampling"),
        ("vector_copy", "MI300X_A1", "/home/u/app/vcopy.cpp"),
    )

    assert export_path == Path(
        "/results/per_kernel_pc_sampling/vector_copy/MI300X_A1"
        "/source/home/u/app/vcopy.cpp_pc_sampling.csv"
    )


@pytest.mark.parametrize(
    ("ascending_numbers", "expected"),
    [
        ([4], "4"),
        ([1, 2, 3], "1-3"),
        ([1, 2, 5, 7, 8], "1-2,5,7-8"),
    ],
    ids=["single", "one_run", "runs_and_singles"],
)
def test_format_number_ranges(ascending_numbers, expected):
    """Consecutive numbers collapse into one range."""
    assert _format_number_ranges(ascending_numbers) == expected


def test_format_isa_references_lists_one_isa_file_per_line():
    """References are grouped by ISA file, in kernel, code object and pid order."""
    isa_references_json = make_isa_references_json(
        (8, "vecAdd", 5, 42, 3),
        (7, "vecCopy", 5, 43, 2),
        (7, "vecCopy", 5, 42, 4),
        (7, "vecCopy", 5, 42, 2),
        (7, "vecCopy", 5, 42, 3),
    )

    assert _format_isa_references(isa_references_json) == (
        "vecCopy_uuid_7/isa_code_object_id_5_pid_42.csv:2-4\n"
        "vecCopy_uuid_7/isa_code_object_id_5_pid_43.csv:2\n"
        "vecAdd_uuid_8/isa_code_object_id_5_pid_42.csv:3"
    )


def test_format_isa_references_names_a_kernel_without_a_short_name():
    """A kernel without a short name is referenced by the folder its ISA uses."""
    isa_references_json = make_isa_references_json((7, None, 5, 42, 1))

    assert _format_isa_references(isa_references_json) == (
        "kernel_uuid_7/isa_code_object_id_5_pid_42.csv:1"
    )


def test_format_isa_references_is_empty_for_a_line_without_instructions():
    """The all-null reference of a line without instructions is dropped."""
    isa_references_json = make_isa_references_json(NO_ISA_REFERENCE)

    assert _format_isa_references(isa_references_json) == ""


def test_write_per_source_line_files_opens_one_file_per_source_file(tmp_path):
    """Each captured source file gets its own per-source-line file."""
    source_line_rows = [
        make_source_line_row(file_path="/s/a.cpp", line_number=1),
        make_source_line_row(file_path="/s/a.cpp", line_number=2),
        make_source_line_row(file_path="/s/b.hpp", line_number=1),
    ]

    file_count = _write_per_source_line_files(tmp_path, iter(source_line_rows), [])

    assert file_count == 2
    assert sorted(
        str(export_path.relative_to(tmp_path))
        for export_path in tmp_path.rglob("*.csv")
    ) == [
        "vector_copy/MI300X_A1/source/s/a.cpp_pc_sampling.csv",
        "vector_copy/MI300X_A1/source/s/b.hpp_pc_sampling.csv",
    ]


def test_write_per_source_line_files_writes_header_and_rows(tmp_path):
    """Stall reasons sit between the percentages and the ISA references."""
    source_line_rows = [
        make_source_line_row(
            line_number=1,
            content="// header",
            total_count=None,
            issue_count=None,
            stall_count=None,
            stall_reason_counts=(None,),
            isa_references=(NO_ISA_REFERENCE,),
        ),
        make_source_line_row(line_number=2, stall_reason_counts=(4,)),
    ]

    _write_per_source_line_files(tmp_path, iter(source_line_rows), ["WAITCNT"])

    exported_rows = read_exported_rows(
        tmp_path
        / "vector_copy"
        / "MI300X_A1"
        / "source"
        / "s"
        / "a.cpp_pc_sampling.csv"
    )
    assert exported_rows == [
        [
            "Source line number",
            "Source line",
            "Total count",
            "Active count",
            "Stall count",
            "Wave occupancy percent",
            "Active thread percent",
            "Stall WAITCNT",
            "ISA references",
        ],
        ["1", "// header", "", "", "", "", "", "", ""],
        [
            "2",
            "int a;",
            "3",
            "2",
            "1",
            "",
            "",
            "4",
            "vecCopy_uuid_7/isa_code_object_id_5_pid_42.csv:1",
        ],
    ]


def test_write_per_source_line_files_writes_nothing_without_rows(tmp_path):
    """A workload without captured source files leaves no folder behind."""
    file_count = _write_per_source_line_files(tmp_path / "per_kernel", iter([]), [])

    assert file_count == 0
    assert not (tmp_path / "per_kernel").exists()


def test_export_per_source_line_files_writes_each_workloads_files(tmp_path):
    """The rows of each workload land under the folder of that workload."""
    export_per_source_line_files(
        tmp_path,
        [
            WorkloadSourceLineExport(
                1, [], iter([make_source_line_row(workload_name="first")])
            ),
            WorkloadSourceLineExport(
                2, [], iter([make_source_line_row(workload_name="second")])
            ),
        ],
    )

    assert sorted(
        str(export_path.relative_to(tmp_path))
        for export_path in tmp_path.rglob("*.csv")
    ) == [
        "first/MI300X_A1/source/s/a.cpp_pc_sampling.csv",
        "second/MI300X_A1/source/s/a.cpp_pc_sampling.csv",
    ]
