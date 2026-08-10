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
