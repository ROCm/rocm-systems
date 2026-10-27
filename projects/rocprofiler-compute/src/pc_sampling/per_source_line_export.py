# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Per-source-line PC sampling CSV export.

Writes one CSV file per captured source file, beside the exported copy of that
source file. Each row holds one source line, the PC sampling counts of the
instructions compiled from that source line, and the rows of those instructions
in the per-kernel ISA files.
"""

import itertools
import json
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

from pc_sampling.pc_sampling_csv_export import (
    IsaLineLocation,
    Row,
    build_header,
    write_grouped_csv_files,
)
from pc_sampling.source_snapshot_analysis import (
    SOURCE_EXPORT_DIRECTORY_NAME,
    resolve_export_path,
)
from utils.analysis_orm import per_source_line_file_key_columns
from utils.logger import console_debug

SOURCE_LINE_FILE_SUFFIX = "_pc_sampling.csv"
# Labels the row for instructions without a source line number
UNATTRIBUTABLE_INSTRUCTIONS_LABEL = "Unattributable instructions"
# A spreadsheet shows each ISA file of the cell on a separate line of the cell.
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


class WorkloadSourceLineExport(NamedTuple):
    """Holds one workload's source line rows and the stall reasons they carry."""

    workload_id: int
    stall_reasons: List[str]
    source_line_rows: Iterator[Row]


def export_per_source_line_files(
    export_directory: Path,
    isa_line_locations: Dict[int, IsaLineLocation],
    workload_source_line_exports: Iterable[WorkloadSourceLineExport],
) -> None:
    """Write every workload's per-source-line files beside its source copies.

    isa_line_locations holds where the per-kernel ISA export wrote each
    instruction, keyed by instruction uuid.
    """
    for workload_source_line_export in workload_source_line_exports:
        file_count = _write_per_source_line_files(
            export_directory,
            workload_source_line_export.source_line_rows,
            workload_source_line_export.stall_reasons,
            isa_line_locations,
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
    isa_line_locations: Dict[int, IsaLineLocation],
) -> int:
    """Write one workload's grouped source line rows, and return the file count."""
    return write_grouped_csv_files(
        source_line_rows,
        len(per_source_line_file_key_columns()),
        build_header(BASE_COLUMNS, stall_reasons, TRAILING_COLUMNS),
        partial(_resolve_source_line_export_path, export_directory),
        lambda file_key, file_rows: _format_source_line_rows(
            file_rows, isa_line_locations
        ),
    )


def _format_source_line_rows(
    file_rows: List[Row],
    isa_line_locations: Dict[int, IsaLineLocation],
) -> Iterator[List[Any]]:
    """Replace the instruction uuids of each source line row with ISA references."""
    for (
        line_number,
        content,
        total_count,
        *other_cells,
        instruction_uuids_json,
    ) in file_rows:
        isa_references = _format_isa_references(
            instruction_uuids_json, isa_line_locations
        )
        # The "?" row has no source line number. It keeps only its total count,
        # which is the samples lost to instructions without a source line number.
        if line_number is None:
            yield [
                None,
                UNATTRIBUTABLE_INSTRUCTIONS_LABEL,
                total_count,
                *(None for _ in other_cells),
                isa_references,
            ]
        else:
            yield [line_number, content, total_count, *other_cells, isa_references]


def _format_isa_references(
    instruction_uuids_json: str,
    isa_line_locations: Dict[int, IsaLineLocation],
) -> str:
    """Return the ISA references of one source line as text.

    Each ISA file goes on its own line of text, written as
    "<kernel folder>/<ISA file>:<instruction line numbers>". Consecutive
    instruction line numbers are written as one range.
    """
    locations = sorted(
        isa_line_locations[instruction_uuid]
        for instruction_uuid in json.loads(instruction_uuids_json)
    )
    return ISA_REFERENCE_SEPARATOR.join(
        f"{isa_file.as_posix()}:"
        + _format_number_ranges([
            location.instruction_line_number for location in file_locations
        ])
        for isa_file, file_locations in itertools.groupby(
            locations, key=lambda location: location.isa_file
        )
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
