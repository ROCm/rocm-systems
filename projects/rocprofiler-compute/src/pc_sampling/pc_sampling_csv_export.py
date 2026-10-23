# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
PC sampling CSV export.

Names the per-kernel ISA files, builds CSV headers with one column per stall
reason, and writes rows that arrive grouped by file into one CSV file per
group.
"""

import csv
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

STALL_COLUMN_PREFIX = "Stall "

# Characters a short name can hold that a path component should not. `_` is
# not in the safe set, so consecutive underscores are collapsed into one.
UNSAFE_PATH_CHARACTERS = re.compile(r"[^A-Za-z0-9.-]+")
KERNEL_DESCRIPTOR_SUFFIX = ".kd"
# Generous for a truncated name, and keeps the whole path inside PATH_MAX.
MAX_SHORT_NAME_LENGTH = 64
# Names the folder of a kernel with no short name
UNNAMED_KERNEL_FOLDER_PREFIX = "kernel"

Row = Tuple[Any, ...]


def build_header(
    leading_columns: Iterable[str],
    stall_reasons: Iterable[str],
    trailing_columns: Iterable[str],
) -> List[str]:
    """Return the column names of a CSV file, with one column per stall reason.

    A workload's stall reasons vary with how it was sampled, so the columns
    are the ones its samples carry. The prefix keeps a reason such as NONE
    from reading as an unrelated column next to the counts.
    """
    return [
        *leading_columns,
        *(f"{STALL_COLUMN_PREFIX}{stall_reason}" for stall_reason in stall_reasons),
        *trailing_columns,
    ]


def resolve_isa_relative_path(
    kernel_uuid: int,
    short_name: Optional[str],
    code_object_id: int,
    pid: int,
) -> Path:
    """Return the path of one kernel's ISA file, relative to its workload's folder.

    The folder leads with the kernel's short name, so it reads as the kernel it
    holds. Two kernels can share a short name, so the kernel uuid follows it to
    keep the folder unique and to map it back to its kernel.csv row.
    """
    folder_prefix = _sanitize_short_name(short_name) or UNNAMED_KERNEL_FOLDER_PREFIX
    return (
        Path(f"{folder_prefix}_uuid_{kernel_uuid}")
        / f"isa_code_object_id_{code_object_id}_pid_{pid}.csv"
    )


def write_grouped_csv_files(
    rows: Iterator[Row],
    file_key_column_count: int,
    header: List[str],
    resolve_export_path: Callable[[Row], Path],
    format_file_rows: Callable[[List[Row]], Iterable[Sequence[Any]]],
) -> int:
    """Write rows that arrive grouped by file, and return the number of files.

    The first file_key_column_count columns of a row name its file.
    resolve_export_path receives those columns, and format_file_rows receives
    the rest of each row of one file.
    """
    file_count = 0
    for file_key, file_rows in _group_rows_by_file(rows, file_key_column_count):
        export_path = resolve_export_path(file_key)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", newline="", encoding="utf-8") as export_file:
            writer = csv.writer(export_file)
            writer.writerow(header)
            writer.writerows(format_file_rows(file_rows))
        file_count += 1
    return file_count


def _sanitize_short_name(short_name: Optional[str]) -> Optional[str]:
    """Reduce a kernel's short name to a path component.

    A short name is the identifier a C++ signature demangles down to, which can
    still hold characters a path cannot, such as the / of operator/.
    """
    if short_name is None:
        return None

    identifier = short_name
    if identifier.endswith(KERNEL_DESCRIPTOR_SUFFIX):
        identifier = identifier[: -len(KERNEL_DESCRIPTOR_SUFFIX)]
    identifier = UNSAFE_PATH_CHARACTERS.sub("_", identifier)
    return identifier.strip("_.")[:MAX_SHORT_NAME_LENGTH].strip("_.")


def _group_rows_by_file(
    rows: Iterator[Row],
    file_key_column_count: int,
) -> Iterator[Tuple[Row, List[Row]]]:
    """Split the ordered row stream into the rows of one file at a time.

    Only the file being written is held, so the whole result set never is.
    """
    current_key = None
    current_rows: List[Row] = []
    for row in rows:
        file_key = row[:file_key_column_count]
        if file_key != current_key:
            if current_key is not None:
                yield current_key, current_rows
            current_key = file_key
            current_rows = []
        current_rows.append(row[file_key_column_count:])

    if current_key is not None:
        yield current_key, current_rows
