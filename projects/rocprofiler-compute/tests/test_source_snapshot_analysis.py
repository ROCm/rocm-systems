# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

import hashlib
import logging
import os
import stat
from pathlib import Path

import common
import pytest

from pc_sampling import source_snapshot_analysis
from pc_sampling.source_snapshot_analysis import (
    parse_source_frames,
    read_source_file_digest_and_lines,
    resolve_snapshot_path,
)


def create_source_snapshot(
    source_snapshot_directory: Path,
    original_source_path: Path,
    contents: bytes = b"",
) -> Path:
    """Create and return a snapshot file for an absolute source path."""
    snapshot_path = source_snapshot_directory / original_source_path.relative_to("/")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(contents)
    return snapshot_path


@pytest.mark.parametrize(
    ("source", "expected_frames"),
    [
        pytest.param(None, [], id="null_comment"),
        pytest.param("", [], id="empty_comment"),
        pytest.param(
            "/home/u/app/kernel.cpp:42",
            [("/home/u/app/kernel.cpp", 42)],
            id="single_frame",
        ),
        pytest.param(
            "/opt/rocm/hip.h:258 -> /opt/rocm/hip.h:317 -> /home/u/vcopy.cpp:36",
            [
                ("/opt/rocm/hip.h", 258),
                ("/opt/rocm/hip.h", 317),
                ("/home/u/vcopy.cpp", 36),
            ],
            id="inline_chain_innermost_first",
        ),
        pytest.param(
            "/opt/rocm/hip.h:? -> /home/u/vcopy.cpp:36",
            [("/opt/rocm/hip.h", None), ("/home/u/vcopy.cpp", 36)],
            id="dropped_line_number",
        ),
        pytest.param(
            "/home/u/app/kernel.cpp",
            [("/home/u/app/kernel.cpp", None)],
            id="no_line_token",
        ),
        pytest.param("N/A", [("N/A", None)], id="missing_source_sentinel"),
        pytest.param(
            "C:/src/kernel.cpp:9",
            [("C:/src/kernel.cpp", 9)],
            id="colon_inside_path",
        ),
    ],
)
def test_parse_source_frames(source, expected_frames):
    """Split representative instruction comments into ordered frames."""
    assert parse_source_frames(source) == expected_frames


def test_resolve_snapshot_path_mirrors_absolute_path(tmp_path):
    """The snapshot mirrors the capture-host path beneath the workload's src."""
    assert resolve_snapshot_path(tmp_path, "/home/u/app/vcopy.cpp") == (
        tmp_path / "src" / "home" / "u" / "app" / "vcopy.cpp"
    )


def test_read_source_file_digest_and_lines_returns_every_line(tmp_path):
    """Return the file's md5 and all of its lines, numbered from one."""
    source_file = tmp_path / "vcopy.cpp"
    source_file.write_text("first\nsecond\nthird\n", encoding="utf-8")

    digest, lines = read_source_file_digest_and_lines(source_file)

    assert digest == hashlib.md5(source_file.read_bytes()).hexdigest()
    assert lines == {1: "first", 2: "second", 3: "third"}


def test_read_source_file_digest_and_lines_missing_file(tmp_path):
    """A file absent from the snapshot yields no digest and no lines."""
    assert read_source_file_digest_and_lines(tmp_path / "absent.cpp") == (None, {})


def test_read_source_file_digest_and_lines_replaces_undecodable_bytes(tmp_path):
    """One mis-encoded source file must not abort an analysis run."""
    source_file = tmp_path / "latin1.cpp"
    source_file.write_bytes(b"caf\xe9\n")

    digest, lines = read_source_file_digest_and_lines(source_file)

    # The digest covers the raw bytes, so it still identifies the file version.
    assert digest == hashlib.md5(b"caf\xe9\n").hexdigest()
    assert lines == {1: "caf\ufffd"}


@pytest.mark.parametrize(
    (
        "snapshot_contents_by_original_path",
        "empty_directory_paths",
        "expected_exported_files",
    ),
    [
        pytest.param(
            {Path("/home/u/app/kernel.cpp"): b"kernel source\n"},
            (),
            {Path("kernel.cpp"): b"kernel source\n"},
            id="single_file",
        ),
        pytest.param(
            {
                Path("/home/u/app/src/kernel.cpp"): b"kernel source\n",
                Path("/home/u/app/include/kernel.hpp"): b"header source\n",
            },
            (),
            {
                Path("src/kernel.cpp"): b"kernel source\n",
                Path("include/kernel.hpp"): b"header source\n",
            },
            id="deepest_parent",
        ),
        pytest.param(
            {
                Path("/home/u/app/kernel.cpp"): b"kernel source\n",
                Path("/opt/rocm/include/runtime.hpp"): b"runtime header\n",
            },
            (),
            {
                Path("home/u/app/kernel.cpp"): b"kernel source\n",
                Path("opt/rocm/include/runtime.hpp"): b"runtime header\n",
            },
            id="root_ancestor",
        ),
        pytest.param(
            {Path("/home/u/app/kernel.cpp"): b"kernel source\n"},
            (Path("opt/empty"),),
            {Path("kernel.cpp"): b"kernel source\n"},
            id="ignores_empty_directories",
        ),
    ],
)
def test_export_source_snapshot_files_uses_deepest_parent_layout(
    tmp_path,
    snapshot_contents_by_original_path,
    empty_directory_paths,
    expected_exported_files,
):
    """Lay out exports relative to the snapshot's deepest common parent."""
    workload_path = tmp_path / "vector_copy" / "MI300X_A1"
    source_snapshot_directory = workload_path / "src"
    for original_source_path, contents in snapshot_contents_by_original_path.items():
        create_source_snapshot(
            source_snapshot_directory,
            original_source_path,
            contents,
        )

    for empty_directory_path in empty_directory_paths:
        (source_snapshot_directory / empty_directory_path).mkdir(parents=True)

    csv_result_directory = tmp_path / "csv_result"
    source_snapshot_analysis.export_source_snapshot_files(
        workload_paths=[str(workload_path)],
        csv_result_directory=csv_result_directory,
    )

    workload_export_directory = (
        csv_result_directory / "source" / "vector_copy" / "MI300X_A1"
    )
    assert (
        common.read_binary_file_tree(workload_export_directory)
        == expected_exported_files
    )


def test_export_source_snapshot_files_overwrites_and_preserves_file_metadata(
    tmp_path,
):
    """Overwrite an existing export while preserving snapshot metadata."""
    workload_path = tmp_path / "vector_copy" / "MI300X_A1"
    original_source_path = Path("/home/u/app/src/kernel.cpp")
    snapshot_path = create_source_snapshot(
        workload_path / "src",
        original_source_path,
        b"\x00new source contents\xff",
    )
    snapshot_path.chmod(0o640)
    snapshot_timestamp_ns = 1_700_000_000_123_456_789
    os.utime(
        snapshot_path,
        ns=(snapshot_timestamp_ns, snapshot_timestamp_ns),
    )

    csv_result_directory = tmp_path / "csv_result"
    exported_path = (
        csv_result_directory / "source" / "vector_copy" / "MI300X_A1" / "kernel.cpp"
    )
    exported_path.parent.mkdir(parents=True)
    exported_path.write_bytes(b"stale source contents")
    exported_path.chmod(0o600)
    os.utime(exported_path, ns=(1_600_000_000, 1_600_000_000))

    source_snapshot_analysis.export_source_snapshot_files(
        workload_paths=[str(workload_path)],
        csv_result_directory=csv_result_directory,
    )

    snapshot_metadata = snapshot_path.stat()
    exported_metadata = exported_path.stat()
    assert exported_path.read_bytes() == snapshot_path.read_bytes()
    assert exported_metadata.st_mtime_ns == snapshot_metadata.st_mtime_ns
    assert stat.S_IMODE(exported_metadata.st_mode) == stat.S_IMODE(
        snapshot_metadata.st_mode
    )


def test_export_source_snapshot_files_scans_each_workload_once(tmp_path, monkeypatch):
    """Discover each workload's source snapshot exactly once."""
    workload_paths = [
        tmp_path / "first_workload" / "MI300X_A1",
        tmp_path / "second_workload" / "MI300X_A1",
    ]
    for workload_path in workload_paths:
        create_source_snapshot(
            workload_path / "src",
            Path("/home/u/app/kernel.cpp"),
            b"kernel source\n",
        )

    discovered_snapshot_directories = []
    original_snapshot_finder = source_snapshot_analysis._find_source_snapshot_files

    def record_snapshot_discovery(source_snapshot_directory):
        """Record a snapshot directory before delegating to the real finder."""
        discovered_snapshot_directories.append(source_snapshot_directory)
        return original_snapshot_finder(source_snapshot_directory)

    monkeypatch.setattr(
        source_snapshot_analysis,
        "_find_source_snapshot_files",
        record_snapshot_discovery,
    )

    source_snapshot_analysis.export_source_snapshot_files(
        workload_paths=[str(workload_path) for workload_path in workload_paths],
        csv_result_directory=tmp_path / "csv_result",
    )

    assert discovered_snapshot_directories == [
        workload_path / "src" for workload_path in workload_paths
    ]


@pytest.mark.parametrize(
    "create_empty_snapshot_directory",
    (
        pytest.param(False, id="missing_directory"),
        pytest.param(True, id="empty_directory"),
    ),
)
def test_export_source_snapshot_files_warns_and_continues_for_missing_or_empty_workload(
    tmp_path,
    caplog,
    create_empty_snapshot_directory,
):
    """Warn for an empty snapshot and continue exporting later workloads."""
    skipped_workload_path = tmp_path / "skipped_workload" / "MI300X_A1"
    if create_empty_snapshot_directory:
        (skipped_workload_path / "src").mkdir(parents=True)

    populated_workload_path = tmp_path / "populated_workload" / "MI300X_A1"
    create_source_snapshot(
        populated_workload_path / "src",
        Path("/home/u/app/kernel.cpp"),
        b"kernel source\n",
    )
    workload_paths = [str(skipped_workload_path), str(populated_workload_path)]
    csv_result_directory = tmp_path / "csv_result"

    with caplog.at_level(logging.WARNING):
        source_snapshot_analysis.export_source_snapshot_files(
            workload_paths=workload_paths,
            csv_result_directory=csv_result_directory,
        )

    skipped_export_directory = (
        csv_result_directory / "source" / "skipped_workload" / "MI300X_A1"
    )
    populated_export_path = (
        csv_result_directory
        / "source"
        / "populated_workload"
        / "MI300X_A1"
        / "kernel.cpp"
    )
    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ]
    assert not skipped_export_directory.exists()
    assert populated_export_path.read_bytes() == b"kernel source\n"
    assert len(warning_messages) == 1
    assert str(skipped_workload_path) in warning_messages[0]
