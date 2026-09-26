# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Source-snapshot analysis utilities.

Each disassembled instruction carries a comment naming the source lines it came
from, as an inline stack of "path:line" frames joined by " -> ", innermost
first. This module parses those comments, reads the lines they name from the
source snapshot, and exports captured source files with CSV analysis results.
"""

import hashlib
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple, Optional

from utils.logger import console_debug

SOURCE_FRAME_SEPARATOR = " -> "
UNKNOWN_SOURCE_LINE_TOKEN = "?"
SOURCE_EXPORT_DIRECTORY_NAME = "source"

SourceFrame = tuple[str, Optional[int]]


class WorkloadSourceSnapshot(NamedTuple):
    """One workload's captured source files and the names it is filed under.

    The names are the ones stored on the workload row, so the export folder
    cannot drift from the database.
    """

    workload_path: Path
    workload_name: str
    workload_sub_name: str
    absolute_source_paths: tuple[str, ...]


def parse_source_frames(source: Optional[str]) -> list[SourceFrame]:
    """Split an instruction comment into its frames, innermost first."""
    if not source:
        return []

    return [parse_source_frame(frame) for frame in source.split(SOURCE_FRAME_SEPARATOR)]


def parse_source_frame(source_frame: str) -> SourceFrame:
    """Split one frame into its path and line number.

    The line number is null for a ":?" frame and for a frame carrying no
    recognizable line token, which keeps its whole text as the path.
    """
    source_path, separator, line_token = source_frame.rpartition(":")
    if not separator or not is_source_line_token(line_token):
        return source_frame, None
    if line_token == UNKNOWN_SOURCE_LINE_TOKEN:
        return source_path, None
    return source_path, int(line_token)


def is_source_line_token(token: str) -> bool:
    """Return whether a token is a source line or an unknown line."""
    return token == UNKNOWN_SOURCE_LINE_TOKEN or (token.isascii() and token.isdigit())


def resolve_snapshot_path(workload_path: Path, absolute_path: str) -> Path:
    """Return where the profiler copied one source file inside a workload.

    The snapshot mirrors absolute paths beneath the workload's "src", so
    /a/b.cpp is copied to <workload>/src/a/b.cpp.
    """
    return workload_path / "src" / Path(absolute_path).relative_to("/")


def read_source_file_digest_and_lines(
    snapshot_path: Path,
) -> tuple[Optional[str], dict[int, str]]:
    """Read one snapshot file's md5 and every line, keyed by line number.

    A missing file yields no digest and no lines. Undecodable bytes are
    replaced so one mis-encoded source file cannot abort an analysis run; the
    digest covers the raw bytes.
    """
    if not snapshot_path.is_file():
        return None, {}

    file_bytes = snapshot_path.read_bytes()
    file_lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    return (
        hashlib.md5(file_bytes).hexdigest(),
        dict(enumerate(file_lines, start=1)),
    )


def resolve_export_path(
    workload_result_directory: Path,
    absolute_path: str,
) -> Path:
    """Return where one source file is exported under a workload's folder.

    The export mirrors the absolute path the database records for the file, so
    a source row locates its copy by dropping the leading separator.
    """
    return workload_result_directory / Path(absolute_path).relative_to("/")


def export_source_snapshot_files(
    workload_source_snapshots: Iterable[WorkloadSourceSnapshot],
    export_directory: Path,
) -> None:
    """Copy the source files a CSV result references into the result folder.

    The source of one workload sits beside the ISA that points at it, under
    that workload's folder.
    """
    for workload_source_snapshot in workload_source_snapshots:
        workload_result_directory = (
            export_directory
            / workload_source_snapshot.workload_name
            / workload_source_snapshot.workload_sub_name
            / SOURCE_EXPORT_DIRECTORY_NAME
        )
        for absolute_path in workload_source_snapshot.absolute_source_paths:
            export_path = resolve_export_path(workload_result_directory, absolute_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                resolve_snapshot_path(
                    workload_source_snapshot.workload_path, absolute_path
                ),
                export_path,
            )

        console_debug(
            f"Exported {len(workload_source_snapshot.absolute_source_paths)} "
            f"source files for workload {workload_source_snapshot.workload_path}."
        )
