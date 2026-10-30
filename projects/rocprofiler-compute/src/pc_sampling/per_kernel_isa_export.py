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
from utils.logger import console_debug

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
    """One workload's instruction rows and the stall columns they fill."""

    workload_id: int
    stall_reasons: List[str]
    isa_rows: Iterator[Tuple[Any, ...]]
    # The folder each kernel's files go to, keyed by kernel uuid
    kernel_directories: Dict[int, Path]


def export_per_kernel_isa_files(
    workload_isa_exports: Iterable[WorkloadIsaExport],
) -> Dict[int, IsaLineLocation]:
    """Export each kernel's ISA beneath its kernel directory.

    Returns the location of each instruction line, keyed by its instruction uuid.
    """
    isa_line_locations: Dict[int, IsaLineLocation] = {}

    for workload_isa_export in workload_isa_exports:
        file_count = _write_per_kernel_isa_files(
            workload_isa_export.isa_rows,
            workload_isa_export.stall_reasons,
            isa_line_locations,
            workload_isa_export.kernel_directories,
        )
        console_debug(
            f"Exported {file_count} per-kernel ISA files for workload "
            f"{workload_isa_export.workload_id}."
        )

    return isa_line_locations


def _resolve_isa_export_path(
    kernel_directories: Dict[int, Path],
    file_key: FileKey,
) -> Path:
    """Return the file one kernel's ISA is written to."""
    _, _, *kernel_key = file_key
    kernel_uuid = kernel_key[0]
    return kernel_directories[kernel_uuid] / resolve_isa_relative_path(*kernel_key)


def _write_per_kernel_isa_files(
    isa_rows: Iterator[Row],
    stall_reasons: List[str],
    isa_line_locations: Dict[int, IsaLineLocation],
    kernel_directories: Dict[int, Path],
) -> int:
    """Write one workload's grouped ISA rows, and return the number of files."""
    return write_grouped_csv_files(
        isa_rows,
        len(per_kernel_isa_file_key_columns()),
        build_header(BASE_COLUMNS, stall_reasons, TRAILING_COLUMNS),
        partial(_resolve_isa_export_path, kernel_directories),
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
