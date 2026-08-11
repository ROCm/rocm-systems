# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Source-snapshot analysis utilities.

Each disassembled instruction carries a comment naming the source lines it came
from, as an inline stack of "path:line" frames joined by " -> ", innermost
first. This module parses those comments and reads the lines they name out of
the source snapshot the profiler copied into the workload directory.
"""

import hashlib
from pathlib import Path
from typing import Optional

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


def shorten_source_file_paths(absolute_paths: set[str]) -> dict[str, str]:
    """Map each path to the shortest suffix unique within the given paths.

    Absolute capture-host paths are unreadable in analysis output, and their
    common ancestor is "/" as soon as a workload inlines a ROCm header.
    """
    return {
        absolute_path: shortest_unambiguous_suffix(absolute_path, absolute_paths)
        for absolute_path in absolute_paths
    }


def shortest_unambiguous_suffix(source_path: str, absolute_paths: set[str]) -> str:
    """Return the fewest trailing components that identify one file.

    Returns the path unchanged when it is relative, or when no suffix short of
    the whole path is unique, since dropping only the leading slash would read
    as a relative path.
    """
    if not Path(source_path).is_absolute():
        return source_path

    # parts[0] is the root marker, never part of a suffix.
    path_components = Path(source_path).parts[1:]
    for component_count in range(1, len(path_components)):
        candidate = str(Path(*path_components[-component_count:]))
        match_count = sum(
            other_path.endswith(f"/{candidate}") for other_path in absolute_paths
        )
        if match_count == 1:
            return candidate
    return source_path


def resolve_snapshot_path(workload_path: Path, absolute_path: str) -> Path:
    """Return where the profiler copied one source file inside a workload.

    The snapshot mirrors absolute paths beneath the workload's "src", so
    /a/b.cpp is copied to <workload>/src/a/b.cpp.
    """
    return workload_path / "src" / Path(absolute_path).relative_to("/")


def read_source_file_digest_and_lines(
    snapshot_path: Path,
    line_numbers: set[int],
) -> tuple[Optional[str], dict[int, str]]:
    """Read one snapshot file's md5 and the content of the requested lines.

    A missing file yields no digest and no lines, and a line past the end of
    the file is absent from the mapping. Undecodable bytes are replaced so one
    mis-encoded source file cannot abort an analysis run; the digest covers the
    raw bytes.
    """
    if not snapshot_path.is_file():
        return None, {}

    file_bytes = snapshot_path.read_bytes()
    file_lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    requested_lines = {
        line_number: file_lines[line_number - 1]
        for line_number in line_numbers
        if 1 <= line_number <= len(file_lines)
    }
    return hashlib.md5(file_bytes).hexdigest(), requested_lines
