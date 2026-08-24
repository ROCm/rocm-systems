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

import csv
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from utils.logger import console_debug, console_warning

PER_KERNEL_DIRECTORY_NAME = "per_kernel_pc_sampling"
STALL_COLUMN_PREFIX = "Stall "

# The row columns naming the file, ahead of the columns written into it.
FILE_KEY_COLUMN_COUNT = 6

# Characters a short name can hold that a path component should not. `_` is
# left out of the safe set so a run of them collapses to the one replacing it.
UNSAFE_PATH_CHARACTERS = re.compile(r"[^A-Za-z0-9.-]+")
KERNEL_DESCRIPTOR_SUFFIX = ".kd"
# Generous for a truncated name, and keeps the whole path inside PATH_MAX.
MAX_SHORT_NAME_LENGTH = 64

BASE_COLUMNS = (
    "Instruction line number",
    "Code object offset",
    "Instruction line",
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

# (workload name, workload sub-name, kernel uuid, short name, code object id, pid)
FileKey = tuple[str, str, int, str, int, int]


class WorkloadIsaExport(NamedTuple):
    """One workload's instruction rows and the stall columns they fill.

    The rows carry the workload's own names, so the export folder cannot drift
    from the database.
    """

    workload_id: int
    stall_reasons: list[str]
    isa_rows: Iterator[tuple[Any, ...]]


def export_per_kernel_isa_files(
    csv_result_directory: Path,
    workload_isa_exports: Iterable[WorkloadIsaExport],
) -> Path:
    """Export every workload's per-kernel ISA beneath a CSV result folder.

    Returns the folder the files were written to, which is also where the
    source they point at is exported.
    """
    per_kernel_directory = csv_result_directory / PER_KERNEL_DIRECTORY_NAME

    for workload_isa_export in workload_isa_exports:
        file_count = _write_per_kernel_isa_files(
            per_kernel_directory,
            workload_isa_export.isa_rows,
            workload_isa_export.stall_reasons,
        )
        console_debug(
            f"Exported {file_count} per-kernel ISA files for workload "
            f"{workload_isa_export.workload_id}."
        )

    if per_kernel_directory.is_dir():
        console_warning(f"Created directory: {per_kernel_directory}")
    return per_kernel_directory


def _build_isa_header(stall_reasons: Iterable[str]) -> list[str]:
    """Return the column names of one workload's ISA files.

    A workload's stall reasons vary with how it was sampled, so the columns
    are the ones its samples carry. The prefix keeps a reason such as NONE
    from reading as an unrelated column next to the counts.
    """
    return [
        *BASE_COLUMNS,
        *(f"{STALL_COLUMN_PREFIX}{stall_reason}" for stall_reason in stall_reasons),
        *TRAILING_COLUMNS,
    ]


def _resolve_isa_export_path(per_kernel_directory: Path, file_key: FileKey) -> Path:
    """Return the file one kernel's ISA is written to.

    The folder leads with the kernel's short name so it reads as the kernel it
    holds. Overloads and template instantiations share a short name, so the
    kernel uuid follows it to keep the folder unique and to map back to the
    row ``kernel.csv`` carries for it.
    """
    (
        workload_name,
        workload_sub_name,
        kernel_uuid,
        short_name,
        code_object_id,
        pid,
    ) = file_key
    return (
        per_kernel_directory
        / workload_name
        / workload_sub_name
        / f"{_sanitize_short_name(short_name)}_{kernel_uuid}"
        / f"isa_code_object_id_{code_object_id}_pid_{pid}.csv"
    )


def _sanitize_short_name(short_name: str) -> str:
    """Reduce a kernel's short name to a path component.

    ``truncate_name`` strips a signature down to its identifier, which still
    leaves spellings a path cannot hold, such as ``operator/`` or a lambda.
    """
    identifier = short_name
    if identifier.endswith(KERNEL_DESCRIPTOR_SUFFIX):
        identifier = identifier[: -len(KERNEL_DESCRIPTOR_SUFFIX)]
    identifier = UNSAFE_PATH_CHARACTERS.sub("_", identifier)
    return identifier.strip("_.")[:MAX_SHORT_NAME_LENGTH].strip("_.")


def _write_per_kernel_isa_files(
    per_kernel_directory: Path,
    isa_rows: Iterator[tuple[Any, ...]],
    stall_reasons: list[str],
) -> int:
    """Write one workload's grouped ISA rows, returning the file count.

    The rows arrive grouped and ordered, so a file is opened when the key
    changes and closed when the next one starts.
    """
    header = _build_isa_header(stall_reasons)
    file_count = 0
    for file_key, file_rows in _group_rows_by_file(isa_rows):
        export_path = _resolve_isa_export_path(per_kernel_directory, file_key)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8") as export_file:
            writer = csv.writer(export_file)
            writer.writerow(header)
            writer.writerows(
                [line_number, *row]
                for line_number, row in enumerate(file_rows, start=1)
            )
        file_count += 1
    return file_count


def _group_rows_by_file(
    isa_rows: Iterator[tuple[Any, ...]],
) -> Iterator[tuple[FileKey, list[tuple[Any, ...]]]]:
    """Split the ordered row stream into the rows of one file at a time.

    Only the file being written is held, so the whole result set never is.
    """
    current_key = None
    current_rows: list[tuple[Any, ...]] = []
    for row in isa_rows:
        file_key: FileKey = row[:FILE_KEY_COLUMN_COUNT]
        if file_key != current_key:
            if current_key is not None:
                yield current_key, current_rows
            current_key = file_key
            current_rows = []
        current_rows.append(row[FILE_KEY_COLUMN_COUNT:])

    if current_key is not None:
        yield current_key, current_rows
