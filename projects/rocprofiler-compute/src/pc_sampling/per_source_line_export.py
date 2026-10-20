# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Per-source-line export.

The per-kernel ISA files carry PC sampling counts per instruction, which is hard
to read against the source it was compiled from. This module writes one file
per captured source file, beside its exported copy, holding every line of the
file with the counts of the instructions compiled from it.
"""

import csv
import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, List, NamedTuple, Tuple

from pc_sampling.per_kernel_isa_export import (
    STALL_COLUMN_PREFIX,
    resolve_isa_relative_path,
)
from pc_sampling.source_snapshot_analysis import (
    SOURCE_EXPORT_DIRECTORY_NAME,
    resolve_export_path,
)
from utils.analysis_orm import PER_SOURCE_LINE_FILE_KEY_COLUMN_COUNT
from utils.logger import console_debug

SOURCE_LINE_FILE_SUFFIX = "_pc_sampling.csv"
# One ISA file per line of the cell; spreadsheets show each on its own line.
ISA_REFERENCE_SEPARATOR = "\n"
# The ISA file of a reference, in the order analysis_orm selects it:
# (kernel uuid, short name, code object id, pid); the instruction line follows.
ISA_FILE_KEY_LENGTH = 4

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


class WorkloadSourceLineExport(NamedTuple):
    """One workload's source line rows and the stall columns they fill."""

    workload_id: int
    stall_reasons: List[str]
    source_line_rows: Iterator[Tuple[Any, ...]]


def export_per_source_line_files(
    export_directory: Path,
    workload_source_line_exports: Iterable[WorkloadSourceLineExport],
) -> None:
    """Export every workload's per-source-line counts beside its source copies."""
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


def _build_source_line_header(stall_reasons: Iterable[str]) -> List[str]:
    """Return the column names of one workload's source line files."""
    return [
        *BASE_COLUMNS,
        *(f"{STALL_COLUMN_PREFIX}{stall_reason}" for stall_reason in stall_reasons),
        *TRAILING_COLUMNS,
    ]


def _resolve_source_line_export_path(export_directory: Path, file_key: FileKey) -> Path:
    """Return the file one source file's line counts are written to.

    It is named after the exported copy, so the two sit side by side.
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
    source_line_rows: Iterator[Tuple[Any, ...]],
    stall_reasons: List[str],
) -> int:
    """Write one workload's grouped source line rows, returning the file count.

    The rows arrive grouped by file and ordered by line, so a file is written
    whole before the next one starts.
    """
    header = _build_source_line_header(stall_reasons)
    file_count = 0
    for file_key, file_rows in itertools.groupby(
        source_line_rows,
        key=lambda row: row[:PER_SOURCE_LINE_FILE_KEY_COLUMN_COUNT],
    ):
        export_path = _resolve_source_line_export_path(export_directory, file_key)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8") as export_file:
            writer = csv.writer(export_file)
            writer.writerow(header)
            writer.writerows(
                [
                    *row[PER_SOURCE_LINE_FILE_KEY_COLUMN_COUNT:-1],
                    _format_isa_references(row[-1]),
                ]
                for row in file_rows
            )
        file_count += 1
    return file_count


def _format_isa_references(isa_references_json: str) -> str:
    """Name the ISA rows a source line came from, one entry per ISA file.

    An entry reads "<kernel folder>/<ISA file>:<instruction line numbers>", with
    consecutive line numbers written as one range.
    """
    # A line no instruction came from joins a single all-null reference.
    isa_references = sorted(
        tuple(isa_reference)
        for isa_reference in json.loads(isa_references_json)
        if isa_reference[0] is not None
    )
    return ISA_REFERENCE_SEPARATOR.join(
        f"{resolve_isa_relative_path(*isa_file_key).as_posix()}:"
        + _format_number_ranges([
            isa_reference[ISA_FILE_KEY_LENGTH] for isa_reference in file_references
        ])
        for isa_file_key, file_references in itertools.groupby(
            isa_references,
            key=lambda isa_reference: isa_reference[:ISA_FILE_KEY_LENGTH],
        )
    )


def _format_number_ranges(ascending_numbers: List[int]) -> str:
    """Join ascending numbers, writing each consecutive run as "first-last"."""
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
