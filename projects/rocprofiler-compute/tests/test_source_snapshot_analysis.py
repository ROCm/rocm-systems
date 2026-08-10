# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

from pathlib import Path

from pc_sampling.source_snapshot_analysis import (
    find_source_files_common_ancestor,
)


def create_source_snapshot(source_snapshot_directory, original_source_path):
    """Create an empty snapshot file for an absolute source path."""
    snapshot_path = source_snapshot_directory / original_source_path.relative_to("/")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.touch()


def test_find_source_files_common_ancestor_missing_directory_returns_none(tmp_path):
    """Return None when the source snapshot directory is missing."""
    assert find_source_files_common_ancestor(tmp_path / "missing") is None


def test_find_source_files_common_ancestor_empty_directory_returns_none(tmp_path):
    """Return None when the source snapshot directory is empty."""
    source_snapshot_directory = tmp_path / "src"
    source_snapshot_directory.mkdir()

    assert find_source_files_common_ancestor(source_snapshot_directory) is None


def test_find_source_files_common_ancestor_single_file_returns_parent(tmp_path):
    """Return the parent directory when the snapshot contains one file."""
    source_snapshot_directory = tmp_path / "src"
    create_source_snapshot(
        source_snapshot_directory,
        Path("/home/u/app/kernel.cpp"),
    )

    assert find_source_files_common_ancestor(source_snapshot_directory) == Path(
        "/home/u/app"
    )


def test_find_source_files_common_ancestor_returns_deepest_shared_parent(tmp_path):
    """Return the deepest parent shared by all snapshotted source files."""
    source_snapshot_directory = tmp_path / "src"
    create_source_snapshot(
        source_snapshot_directory,
        Path("/home/u/app/src/kernel.cpp"),
    )
    create_source_snapshot(
        source_snapshot_directory,
        Path("/home/u/app/include/kernel.hpp"),
    )

    assert find_source_files_common_ancestor(source_snapshot_directory) == Path(
        "/home/u/app"
    )


def test_find_source_files_common_ancestor_different_top_level_branches_returns_root(
    tmp_path,
):
    """Return the root for files from different top-level branches."""
    source_snapshot_directory = tmp_path / "src"
    create_source_snapshot(
        source_snapshot_directory,
        Path("/home/u/app/kernel.cpp"),
    )
    create_source_snapshot(
        source_snapshot_directory,
        Path("/opt/rocm/include/runtime.hpp"),
    )

    assert find_source_files_common_ancestor(source_snapshot_directory) == Path("/")


def test_find_source_files_common_ancestor_ignores_empty_directories(tmp_path):
    """Ignore empty directories when finding the common source ancestor."""
    source_snapshot_directory = tmp_path / "src"
    create_source_snapshot(
        source_snapshot_directory,
        Path("/home/u/app/kernel.cpp"),
    )
    (source_snapshot_directory / "opt/empty").mkdir(parents=True)

    assert find_source_files_common_ancestor(source_snapshot_directory) == Path(
        "/home/u/app"
    )
