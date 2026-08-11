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
from typing import Optional

from utils.logger import console_debug, console_warning

SOURCE_FRAME_SEPARATOR = " -> "
UNKNOWN_SOURCE_LINE_TOKEN = "?"

SourceFrame = tuple[str, Optional[int]]


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


def export_source_snapshot_files(
    workload_paths: Iterable[str],
    csv_result_directory: Path,
) -> None:
    """Export workload source snapshots beneath a CSV result folder."""
    source_result_directory = csv_result_directory / "source"

    for workload_path in workload_paths:
        workload_directory = Path(workload_path)
        source_snapshot_directory = workload_directory / "src"
        source_snapshot_file_pairs = [
            (
                source_snapshot_file,
                Path("/") / source_snapshot_file.relative_to(source_snapshot_directory),
            )
            for source_snapshot_file in _find_source_snapshot_files(
                source_snapshot_directory
            )
        ]

        if not source_snapshot_file_pairs:
            _log_skipped_source_snapshot_export(workload_path)
            continue

        source_files_common_ancestor = _find_common_parent([
            original_source_file
            for _source_snapshot_file, original_source_file in (
                source_snapshot_file_pairs
            )
        ])

        workload_name = workload_directory.parent.name
        workload_sub_name = workload_directory.name
        workload_result_directory = (
            source_result_directory / workload_name / workload_sub_name
        )
        source_snapshot_destinations = [
            (
                source_snapshot_file,
                _make_source_snapshot_destination(
                    original_source_file,
                    source_files_common_ancestor,
                    workload_result_directory,
                ),
            )
            for source_snapshot_file, original_source_file in (
                source_snapshot_file_pairs
            )
        ]
        for source_snapshot_file, destination_file in source_snapshot_destinations:
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_snapshot_file, destination_file)

        console_debug(
            f"Exported {len(source_snapshot_file_pairs)} source snapshot files "
            f"for workload {workload_path}."
        )


def _find_source_snapshot_files(
    source_snapshot_directory: Path,
) -> list[Path]:
    """Return regular files in a source snapshot."""
    return sorted(
        snapshot_path
        for snapshot_path in source_snapshot_directory.rglob("*")
        if snapshot_path.is_file()
    )


def _make_source_snapshot_destination(
    original_source_file: Path,
    source_files_common_ancestor: Path,
    workload_result_directory: Path,
) -> Path:
    """Return the ancestor-relative destination for a snapshot file."""
    return workload_result_directory / original_source_file.relative_to(
        source_files_common_ancestor
    )


def _log_skipped_source_snapshot_export(workload_path: str) -> None:
    """Log that no source snapshot files were exported for a workload."""
    console_warning(f"Source snapshot export skipped for workload {workload_path}.")
    console_debug(f"Exported 0 source snapshot files for workload {workload_path}.")


def _find_common_parent(source_files: list[Path]) -> Path:
    """Return the deepest directory containing every source file."""
    first_parent = source_files[0].parent
    return next(
        ancestor
        for ancestor in (first_parent, *first_parent.parents)
        if all(source_file.is_relative_to(ancestor) for source_file in source_files)
    )
