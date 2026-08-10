# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Source-snapshot analysis utilities.

Source snapshots mirror capture-host absolute paths beneath a workload's
``src`` directory. Recover their original common ancestor for downstream
analysis output.
"""

from pathlib import Path
from typing import Optional

SOURCE_FRAME_SEPARATOR = " -> "


def find_source_files_common_ancestor(
    source_snapshot_directory: Path,
) -> Optional[Path]:
    """
    Return the original absolute common parent of snapshot source files.

    Each snapshot path is interpreted relative to ``source_snapshot_directory``
    and restored beneath the filesystem root. A missing or empty snapshot tree
    has no common ancestor.
    """
    original_source_files = [
        Path("/") / snapshot_file.relative_to(source_snapshot_directory)
        for snapshot_file in source_snapshot_directory.rglob("*")
        if snapshot_file.is_file()
    ]
    return _find_common_parent(original_source_files)


def make_source_relative_to_common_ancestor(
    source: Optional[str],
    source_files_common_ancestor: Optional[Path],
) -> Optional[str]:
    """Make absolute source-frame paths relative to their common ancestor."""
    if (
        source is None
        or source_files_common_ancestor is None
        or source_files_common_ancestor == Path("/")
    ):
        return source

    return SOURCE_FRAME_SEPARATOR.join(
        _make_source_frame_path_relative(frame, source_files_common_ancestor)
        for frame in source.split(SOURCE_FRAME_SEPARATOR)
    )


def _find_common_parent(source_files: list[Path]) -> Optional[Path]:
    """Return the deepest directory containing every source file."""
    if not source_files:
        return None

    first_parent = source_files[0].parent
    return next(
        ancestor
        for ancestor in (first_parent, *first_parent.parents)
        if all(source_file.is_relative_to(ancestor) for source_file in source_files)
    )


def _make_source_frame_path_relative(
    source_frame: str,
    source_files_common_ancestor: Path,
) -> str:
    """Make a source-frame path relative when it descends from the ancestor."""
    source_path_text, line_token_suffix = _split_source_frame(source_frame)
    source_path = Path(source_path_text)
    if not source_path.is_absolute() or not source_path.is_relative_to(
        source_files_common_ancestor
    ):
        return source_frame

    relative_source_path = source_path.relative_to(source_files_common_ancestor)
    return f"{relative_source_path}{line_token_suffix}"


def _split_source_frame(source_frame: str) -> tuple[str, str]:
    """Split a source frame into its path and recognized line-token suffix."""
    source_path, separator, line_token = source_frame.rpartition(":")
    if not separator or not _is_source_line_token(line_token):
        return source_frame, ""
    return source_path, f"{separator}{line_token}"


def _is_source_line_token(token: str) -> bool:
    """Return whether a token represents a source line or an unknown line."""
    return token == "?" or (token.isascii() and token.isdigit())
