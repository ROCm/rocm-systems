# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

from pathlib import Path

import pytest

from pc_sampling import source_snapshot_analysis
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


@pytest.mark.parametrize(
    ("source_paths", "empty_directory_paths", "expected_ancestor"),
    [
        pytest.param(
            (),
            (),
            None,
            id="empty_directory",
        ),
        pytest.param(
            (Path("/home/u/app/kernel.cpp"),),
            (),
            Path("/home/u/app"),
            id="single_file",
        ),
        pytest.param(
            (
                Path("/home/u/app/src/kernel.cpp"),
                Path("/home/u/app/include/kernel.hpp"),
            ),
            (),
            Path("/home/u/app"),
            id="deepest_shared_parent",
        ),
        pytest.param(
            (
                Path("/home/u/app/kernel.cpp"),
                Path("/opt/rocm/include/runtime.hpp"),
            ),
            (),
            Path("/"),
            id="different_top_level_branches",
        ),
        pytest.param(
            (Path("/home/u/app/kernel.cpp"),),
            (Path("opt/empty"),),
            Path("/home/u/app"),
            id="ignores_empty_directories",
        ),
    ],
)
def test_find_source_files_common_ancestor(
    tmp_path,
    source_paths,
    empty_directory_paths,
    expected_ancestor,
):
    """Return the common ancestor for representative snapshot layouts."""
    source_snapshot_directory = tmp_path / "src"
    source_snapshot_directory.mkdir()

    for source_path in source_paths:
        create_source_snapshot(source_snapshot_directory, source_path)

    for empty_directory_path in empty_directory_paths:
        (source_snapshot_directory / empty_directory_path).mkdir(parents=True)

    assert (
        find_source_files_common_ancestor(source_snapshot_directory)
        == expected_ancestor
    )


@pytest.mark.parametrize(
    ("source", "common_ancestor", "expected"),
    [
        pytest.param(
            ("/home/u/app/src/kernel.cpp:42 -> /home/u/app/include/kernel.hpp:?"),
            Path("/home/u/app"),
            "src/kernel.cpp:42 -> include/kernel.hpp:?",
            id="multiple_frames",
        ),
        pytest.param(
            "/home/u/app/kernel.cpp",
            Path("/home/u/app"),
            "kernel.cpp",
            id="no_line_token",
        ),
        pytest.param(
            None,
            Path("/home/u/app"),
            None,
            id="null_source",
        ),
        pytest.param(
            "N/A",
            Path("/home/u/app"),
            "N/A",
            id="missing_source_sentinel",
        ),
        pytest.param(
            "/home/u/app/kernel.cpp:42",
            None,
            "/home/u/app/kernel.cpp:42",
            id="missing_ancestor",
        ),
        pytest.param(
            "/home/u/app/kernel.cpp:42",
            Path("/"),
            "/home/u/app/kernel.cpp:42",
            id="filesystem_root_ancestor",
        ),
    ],
)
def test_make_source_relative_to_common_ancestor(
    source,
    common_ancestor,
    expected,
):
    transformed_source = (
        source_snapshot_analysis.make_source_relative_to_common_ancestor(
            source,
            common_ancestor,
        )
    )

    assert transformed_source == expected
