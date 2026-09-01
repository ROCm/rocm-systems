# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import common
import pytest

from pc_sampling import source_snapshot_analysis
from pc_sampling.source_snapshot_analysis import (
    WorkloadSourceSnapshot,
    load_source_path_map,
    parse_source_frames,
    read_source_file_digest_and_lines,
    resolve_export_path,
    resolve_snapshot_path,
)


def write_source_path_map(
    workload_path: Path,
    pid: int,
    source_paths: dict[str, str],
) -> Path:
    """Write one process's raw DWARF to canonical path map."""
    map_path = workload_path / "src" / f"{pid}_source_map.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps({"source_paths": source_paths}), encoding="utf-8")
    return map_path


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
    assert parse_source_frames(source, {}) == expected_frames


def test_parse_source_frames_canonicalizes_each_frame():
    """Every frame of a comment names the canonical path of its file."""
    source = "/app/bin/../inc/hip.h:12 -> /app/kernel.cpp:7"
    source_path_map = {"/app/bin/../inc/hip.h": "/app/inc/hip.h"}

    assert parse_source_frames(source, source_path_map) == [
        ("/app/inc/hip.h", 12),
        ("/app/kernel.cpp", 7),
    ]


def test_load_source_path_map_without_snapshot(tmp_path):
    """A workload profiled without a snapshot names no paths."""
    assert load_source_path_map(tmp_path) == {}


def test_load_source_path_map_merges_every_process(tmp_path):
    """Each process of a run records the files it captured."""
    write_source_path_map(tmp_path, 101, {"/app/bin/../kernel.cpp": "/app/kernel.cpp"})
    write_source_path_map(tmp_path, 202, {"/app/link/util.h": "/app/include/util.h"})

    assert load_source_path_map(tmp_path) == {
        "/app/bin/../kernel.cpp": "/app/kernel.cpp",
        "/app/link/util.h": "/app/include/util.h",
    }


def test_load_source_path_map_with_unreadable_map(tmp_path, monkeypatch):
    """An unreadable map leaves its files to be looked up by raw DWARF path."""
    map_path = tmp_path / "src" / "303_source_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("{not json", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(source_snapshot_analysis, "console_warning", warnings.append)

    assert load_source_path_map(tmp_path) == {}
    assert len(warnings) == 1


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


def make_workload_source_snapshot(
    workload_path: Path,
    snapshot_contents_by_absolute_path: dict[str, bytes],
    exported_absolute_paths: Optional[tuple[str, ...]] = None,
) -> WorkloadSourceSnapshot:
    """Populate a workload's snapshot and describe what to export from it."""
    for absolute_path, contents in snapshot_contents_by_absolute_path.items():
        create_source_snapshot(workload_path / "src", Path(absolute_path), contents)

    if exported_absolute_paths is None:
        exported_absolute_paths = tuple(snapshot_contents_by_absolute_path)
    return WorkloadSourceSnapshot(
        workload_path=workload_path,
        workload_name=workload_path.parent.name,
        workload_sub_name=workload_path.name,
        absolute_source_paths=exported_absolute_paths,
    )


def test_resolve_export_path_mirrors_absolute_path(tmp_path):
    """An exported file keeps the absolute path the database records for it."""
    assert resolve_export_path(tmp_path, "/home/u/app/vcopy.cpp") == (
        tmp_path / "home" / "u" / "app" / "vcopy.cpp"
    )


@pytest.mark.parametrize(
    ("snapshot_contents_by_absolute_path", "expected_exported_files"),
    [
        pytest.param(
            {"/home/u/app/kernel.cpp": b"kernel source\n"},
            {Path("home/u/app/kernel.cpp"): b"kernel source\n"},
            id="single_file",
        ),
        pytest.param(
            {
                "/home/u/app/src/kernel.cpp": b"kernel source\n",
                "/home/u/app/include/kernel.hpp": b"header source\n",
            },
            {
                Path("home/u/app/src/kernel.cpp"): b"kernel source\n",
                Path("home/u/app/include/kernel.hpp"): b"header source\n",
            },
            id="shared_ancestor_is_not_stripped",
        ),
        pytest.param(
            {
                "/home/u/app/kernel.cpp": b"kernel source\n",
                "/opt/rocm/include/runtime.hpp": b"runtime header\n",
            },
            {
                Path("home/u/app/kernel.cpp"): b"kernel source\n",
                Path("opt/rocm/include/runtime.hpp"): b"runtime header\n",
            },
            id="unrelated_roots",
        ),
    ],
)
def test_export_source_snapshot_files_mirrors_absolute_paths(
    tmp_path,
    snapshot_contents_by_absolute_path,
    expected_exported_files,
):
    """Lay every export out under the absolute path recorded for the file.

    Adding a file must not move the others, so the layout cannot depend on
    what the workload as a whole happens to reference.
    """
    workload_source_snapshot = make_workload_source_snapshot(
        tmp_path / "vector_copy" / "MI300X_A1",
        snapshot_contents_by_absolute_path,
    )
    export_directory = tmp_path / "per_kernel_pc_sampling"

    source_snapshot_analysis.export_source_snapshot_files(
        workload_source_snapshots=[workload_source_snapshot],
        export_directory=export_directory,
    )

    workload_export_directory = (
        export_directory / "vector_copy" / "MI300X_A1" / "source"
    )
    assert (
        common.read_binary_file_tree(workload_export_directory)
        == expected_exported_files
    )


def test_export_source_snapshot_files_copies_only_referenced_paths(tmp_path):
    """Leave a snapshot file no source row names out of the export."""
    workload_source_snapshot = make_workload_source_snapshot(
        tmp_path / "vector_copy" / "MI300X_A1",
        {
            "/home/u/app/kernel.cpp": b"kernel source\n",
            "/home/u/app/unreferenced.cpp": b"unreferenced source\n",
        },
        exported_absolute_paths=("/home/u/app/kernel.cpp",),
    )
    export_directory = tmp_path / "per_kernel_pc_sampling"

    source_snapshot_analysis.export_source_snapshot_files(
        workload_source_snapshots=[workload_source_snapshot],
        export_directory=export_directory,
    )

    workload_export_directory = (
        export_directory / "vector_copy" / "MI300X_A1" / "source"
    )
    assert common.read_binary_file_tree(workload_export_directory) == {
        Path("home/u/app/kernel.cpp"): b"kernel source\n"
    }


def test_export_source_snapshot_files_keeps_workloads_apart(tmp_path):
    """Two workloads holding the same absolute path each keep their own copy."""
    workload_source_snapshots = [
        make_workload_source_snapshot(
            tmp_path / workload_name / "MI300X_A1",
            {"/home/u/app/kernel.cpp": contents},
        )
        for workload_name, contents in (
            ("first_workload", b"first source\n"),
            ("second_workload", b"second source\n"),
        )
    ]
    export_directory = tmp_path / "per_kernel_pc_sampling"

    source_snapshot_analysis.export_source_snapshot_files(
        workload_source_snapshots=workload_source_snapshots,
        export_directory=export_directory,
    )

    assert common.read_binary_file_tree(export_directory) == {
        Path(
            "first_workload/MI300X_A1/source/home/u/app/kernel.cpp"
        ): b"first source\n",
        Path(
            "second_workload/MI300X_A1/source/home/u/app/kernel.cpp"
        ): b"second source\n",
    }


def test_export_source_snapshot_files_is_quiet_for_workload_without_sources(
    tmp_path,
    monkeypatch,
):
    """A workload with no source rows exports nothing and warns about nothing.

    Every workload profiled without PC sampling lands here, so it is the
    ordinary case rather than one worth reporting.
    """
    warning_messages = []
    monkeypatch.setattr(
        source_snapshot_analysis,
        "console_warning",
        warning_messages.append,
        raising=False,
    )
    workload_source_snapshot = make_workload_source_snapshot(
        tmp_path / "vector_copy" / "MI300X_A1",
        {},
    )
    export_directory = tmp_path / "per_kernel_pc_sampling"

    source_snapshot_analysis.export_source_snapshot_files(
        workload_source_snapshots=[workload_source_snapshot],
        export_directory=export_directory,
    )

    assert warning_messages == []
    assert not export_directory.exists()


def test_export_source_snapshot_files_leaves_exports_writable(tmp_path):
    """A read-only snapshot file must not export a read-only copy.

    Preserving the mode would make a rerun into the same folder fail for a
    user who cannot overwrite their own export.
    """
    workload_path = tmp_path / "vector_copy" / "MI300X_A1"
    workload_source_snapshot = make_workload_source_snapshot(
        workload_path,
        {"/home/u/app/kernel.cpp": b"kernel source\n"},
    )
    snapshot_path = resolve_snapshot_path(workload_path, "/home/u/app/kernel.cpp")
    snapshot_path.chmod(0o444)
    export_directory = tmp_path / "per_kernel_pc_sampling"

    source_snapshot_analysis.export_source_snapshot_files(
        workload_source_snapshots=[workload_source_snapshot],
        export_directory=export_directory,
    )

    exported_path = resolve_export_path(
        export_directory / "vector_copy" / "MI300X_A1" / "source",
        "/home/u/app/kernel.cpp",
    )
    assert exported_path.read_bytes() == b"kernel source\n"
    assert os.access(exported_path, os.W_OK)
