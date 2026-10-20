# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Per-source-line PC sampling CSV export.

Writes one CSV file per captured source file, beside its exported copy. Each row
is one line of the source file, with the PC sampling counts of the instructions
compiled from that line and the rows of those instructions in the per-kernel
ISA files.
"""

import itertools
import json
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

from pc_sampling.pc_sampling_csv_export import (
    Row,
    build_header,
    resolve_isa_relative_path,
    write_grouped_csv_files,
)
from pc_sampling.source_snapshot_analysis import (
    SOURCE_EXPORT_DIRECTORY_NAME,
    resolve_export_path,
)
from utils.analysis_orm import per_source_line_file_key_columns
from utils.logger import console_debug

SOURCE_LINE_FILE_SUFFIX = "_pc_sampling.csv"
# Spreadsheets show each ISA file of a cell on its own line.
ISA_REFERENCE_SEPARATOR = "\n"

BASE_COLUMNS = (
    "Source line number",
    "Source line",
    "Total count",
    "Active count",
    "Stall count",
    "Wave occupancy percent",
    "Active thread percent",
)
TRAILING_COLUMNS = ("ISA references",)

# The leading row columns, in the order analysis_orm selects them:
# (workload name, workload sub-name, source file path)
FileKey = Tuple[str, str, str]


class IsaReference(NamedTuple):
    """Identifies the row of one instruction in a per-kernel ISA file."""

    kernel_uuid: int
    short_name: Optional[str]
    code_object_id: int
    pid: int
    instruction_line_number: int


class WorkloadSourceLineExport(NamedTuple):
    """Holds one workload's source line rows and the stall reasons they carry."""

    workload_id: int
    stall_reasons: List[str]
    source_line_rows: Iterator[Row]


def export_per_source_line_files(
    export_directory: Path,
    workload_source_line_exports: Iterable[WorkloadSourceLineExport],
) -> None:
    """Write every workload's per-source-line files beside its source copies."""
    for workload_source_line_export in workload_source_line_exports:
        file_count = _write_per_source_line_files(
            export_directory,
            workload_source_line_export.source_line_rows,
            workload_source_line_export.stall_reasons,
        )
        console_debug(
            f"Exported {file_count} per-source-line files for workload "
            f"{workload_source_line_export.workload_id}."
        )


def _resolve_source_line_export_path(export_directory: Path, file_key: FileKey) -> Path:
    """Return the per-source-line file of one source file.

    The file is named after the exported copy of the source file, so the two
    sit side by side.
    """
    workload_name, workload_sub_name, file_path = file_key
    source_export_path = resolve_export_path(
        export_directory
        / workload_name
        / workload_sub_name
        / SOURCE_EXPORT_DIRECTORY_NAME,
        file_path,
    )
    return source_export_path.with_name(
        f"{source_export_path.name}{SOURCE_LINE_FILE_SUFFIX}"
    )


def _write_per_source_line_files(
    export_directory: Path,
    source_line_rows: Iterator[Row],
    stall_reasons: List[str],
) -> int:
    """Write one workload's grouped source line rows, and return the file count."""
    return write_grouped_csv_files(
        source_line_rows,
        len(per_source_line_file_key_columns()),
        build_header(BASE_COLUMNS, stall_reasons, TRAILING_COLUMNS),
        partial(_resolve_source_line_export_path, export_directory),
        _format_source_line_rows,
    )


def _format_source_line_rows(file_rows: List[Row]) -> Iterator[List[Any]]:
    """Replace the ISA references of each source line row with their text."""
    for *source_line_cells, isa_references_json in file_rows:
        yield [*source_line_cells, _format_isa_references(isa_references_json)]


def _format_isa_references(isa_references_json: str) -> str:
    """Return the ISA references of one source line, one ISA file per line.

    Each line reads "<kernel folder>/<ISA file>:<instruction line numbers>",
    and consecutive instruction line numbers are written as one range.
    """
    # A source line that no instruction came from has one all-null reference.
    isa_references = sorted(
        IsaReference(**isa_reference)
        for isa_reference in json.loads(isa_references_json)
        if isa_reference["kernel_uuid"] is not None
    )
    return ISA_REFERENCE_SEPARATOR.join(
        f"{isa_file.as_posix()}:"
        + _format_number_ranges([
            isa_reference.instruction_line_number for isa_reference in file_references
        ])
        for isa_file, file_references in itertools.groupby(
            isa_references, key=_resolve_isa_reference_file
        )
    )


def _resolve_isa_reference_file(isa_reference: IsaReference) -> Path:
    """Return the per-kernel ISA file that holds one referenced instruction."""
    return resolve_isa_relative_path(
        isa_reference.kernel_uuid,
        isa_reference.short_name,
        isa_reference.code_object_id,
        isa_reference.pid,
    )


def _format_number_ranges(ascending_numbers: List[int]) -> str:
    """Join ascending numbers, and write each consecutive run as "first-last"."""
    ranges = []
    # Within a consecutive run, a number minus its position stays constant.
    for _, run in itertools.groupby(
        enumerate(ascending_numbers),
        key=lambda position_and_number: position_and_number[1] - position_and_number[0],
    ):
        run_numbers = [number for _, number in run]
        ranges.append(
            f"{run_numbers[0]}-{run_numbers[-1]}"
            if len(run_numbers) > 1
            else str(run_numbers[0])
        )
    return ",".join(ranges)
