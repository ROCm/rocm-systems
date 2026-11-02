# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Per-kernel ISA export.

A CSV analysis result carries the PC sampling counts of every kernel in one
`pc_sampling_summary.csv`, which is hard to read against a kernel's own
disassembly. This module writes one file per kernel per code object per
process, holding that kernel's instruction lines in offset order with the
counts collected on them.
"""

from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from pc_sampling.pc_sampling_csv_export import (
    IsaLineLocation,
    Row,
    build_header,
    resolve_isa_relative_path,
    write_grouped_csv_files,
)
from utils.analysis_orm import per_kernel_isa_file_key_columns
from utils.logger import console_debug, console_warning

PER_KERNEL_DIRECTORY_NAME = "per_kernel_pc_sampling"

BASE_COLUMNS = (
    "Instruction line number",
    "Code object offset",
    "Instruction line",
    "Instruction type",
    "Total count",
    "Active count",
    "Stall count",
    "Wave occupancy percent",
    "Active thread percent",
)
TRAILING_COLUMNS = (
    "Source",
    "Code object id",
    "Pid",
)

# The leading row columns, in the order analysis_orm selects them:
# (workload name, workload sub-name, kernel uuid, short name, code object id, pid)
FileKey = Tuple[str, str, int, Optional[str], int, int]


class WorkloadIsaExport(NamedTuple):
    """One workload's instruction rows and the stall columns they fill.

    The rows carry the workload's own names, so the export folder cannot drift
    from the database.
    """

    workload_id: int
    stall_reasons: List[str]
    isa_rows: Iterator[Tuple[Any, ...]]


class PerKernelIsaFiles(NamedTuple):
    """The per-kernel ISA files that were written, and where each instruction is."""

    # The folder of the files, which is also where their source is exported
    directory: Path
    # The location of each instruction line, keyed by its instruction uuid
    isa_line_locations: Dict[int, IsaLineLocation]


def export_per_kernel_isa_files(
    csv_result_directory: Path,
    workload_isa_exports: Iterable[WorkloadIsaExport],
) -> PerKernelIsaFiles:
    """Export every workload's per-kernel ISA beneath a CSV result folder."""
    per_kernel_directory = csv_result_directory / PER_KERNEL_DIRECTORY_NAME
    isa_line_locations: Dict[int, IsaLineLocation] = {}

    for workload_isa_export in workload_isa_exports:
        file_count = _write_per_kernel_isa_files(
            per_kernel_directory,
            workload_isa_export.isa_rows,
            workload_isa_export.stall_reasons,
            isa_line_locations,
        )
        console_debug(
            f"Exported {file_count} per-kernel ISA files for workload "
            f"{workload_isa_export.workload_id}."
        )

    if per_kernel_directory.is_dir():
        console_warning(f"Created directory: {per_kernel_directory}")
    return PerKernelIsaFiles(per_kernel_directory, isa_line_locations)


def _resolve_isa_export_path(per_kernel_directory: Path, file_key: FileKey) -> Path:
    """Return the file one kernel's ISA is written to."""
    workload_name, workload_sub_name, *kernel_key = file_key
    return (
        per_kernel_directory
        / workload_name
        / workload_sub_name
        / resolve_isa_relative_path(*kernel_key)
    )


def _write_per_kernel_isa_files(
    per_kernel_directory: Path,
    isa_rows: Iterator[Row],
    stall_reasons: List[str],
    isa_line_locations: Dict[int, IsaLineLocation],
) -> int:
    """Write one workload's grouped ISA rows, and return the number of files."""
    return write_grouped_csv_files(
        isa_rows,
        len(per_kernel_isa_file_key_columns()),
        build_header(BASE_COLUMNS, stall_reasons, TRAILING_COLUMNS),
        partial(_resolve_isa_export_path, per_kernel_directory),
        partial(_number_and_locate_instruction_lines, isa_line_locations),
    )


def _number_and_locate_instruction_lines(
    isa_line_locations: Dict[int, IsaLineLocation],
    file_key: FileKey,
    file_rows: List[Row],
) -> Iterator[List[Any]]:
    """Prefix each instruction row of one ISA file with its instruction line number.

    The instruction line numbers count from 1. Records the location of each
    instruction in isa_line_locations. Each row starts with its instruction
    uuid, which is not written.
    """
    _, _, *kernel_key = file_key
    isa_file = resolve_isa_relative_path(*kernel_key)
    for instruction_line_number, (instruction_uuid, *cells) in enumerate(
        file_rows, start=1
    ):
        isa_line_locations[instruction_uuid] = IsaLineLocation(
            isa_file, instruction_line_number
        )
        yield [instruction_line_number, *cells]
