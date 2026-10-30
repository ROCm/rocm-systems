# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

import hashlib
import json
from pathlib import Path

import pytest

from pc_sampling import source_snapshot_analysis
from pc_sampling.source_snapshot_analysis import (
    load_source_path_map,
    parse_source_frames,
    read_source_file_digest_and_lines,
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


@pytest.mark.parametrize(
    ("map_contents", "warning_fragment"),
    [
        pytest.param("{not json", "Expecting property name", id="malformed_json"),
        pytest.param("null", "root is not an object", id="null_root"),
        pytest.param("[]", "root is not an object", id="list_root"),
        pytest.param("1", "root is not an object", id="scalar_root"),
        pytest.param(
            json.dumps({"source_paths": None}),
            "'source_paths' is not an object",
            id="null_source_paths",
        ),
        pytest.param(
            json.dumps({"source_paths": []}),
            "'source_paths' is not an object",
            id="empty_list_source_paths",
        ),
        pytest.param(
            json.dumps({"source_paths": ["x"]}),
            "'source_paths' is not an object",
            id="list_source_paths",
        ),
        pytest.param(
            json.dumps({"source_paths": "oops"}),
            "'source_paths' is not an object",
            id="scalar_source_paths",
        ),
        pytest.param(
            json.dumps({"source_paths": {"/app/kernel.cpp": 1}}),
            "entry for '/app/kernel.cpp' must map strings to strings",
            id="non_string_value",
        ),
    ],
)
def test_load_source_path_map_rejects_invalid_map(
    tmp_path,
    monkeypatch,
    map_contents,
    warning_fragment,
):
    """An invalid map warns and leaves its paths unresolved."""
    map_path = tmp_path / "src" / "303_source_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text(map_contents, encoding="utf-8")

    warnings = []
    monkeypatch.setattr(source_snapshot_analysis, "console_warning", warnings.append)

    assert load_source_path_map(tmp_path) == {}
    assert len(warnings) == 1
    assert str(map_path) in warnings[0]
    assert warning_fragment in warnings[0]


def test_load_source_path_map_allows_missing_source_paths(tmp_path, monkeypatch):
    """A map with no source-path member contributes no paths or warnings."""
    map_path = tmp_path / "src" / "303_source_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("{}", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(source_snapshot_analysis, "console_warning", warnings.append)

    assert load_source_path_map(tmp_path) == {}
    assert warnings == []


def test_load_source_path_map_with_unreadable_file(tmp_path, monkeypatch):
    """An unreadable map warns and leaves its paths unresolved."""
    map_path = tmp_path / "src" / "303_source_map.json"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("{}", encoding="utf-8")

    def raise_permission_error(_path, *_args, **_kwargs):
        raise PermissionError("permission denied")

    warnings = []
    monkeypatch.setattr(Path, "open", raise_permission_error)
    monkeypatch.setattr(source_snapshot_analysis, "console_warning", warnings.append)

    assert load_source_path_map(tmp_path) == {}
    assert len(warnings) == 1
    assert str(map_path) in warnings[0]
    assert "permission denied" in warnings[0]


def test_load_source_path_map_skips_invalid_and_keeps_valid(tmp_path, monkeypatch):
    """One invalid process map must not discard another process's valid paths."""
    invalid_map_path = tmp_path / "src" / "101_source_map.json"
    invalid_map_path.parent.mkdir(parents=True)
    invalid_map_path.write_text('{"source_paths": null}', encoding="utf-8")
    write_source_path_map(
        tmp_path,
        202,
        {"/app/bin/../kernel.cpp": "/app/kernel.cpp"},
    )

    warnings = []
    monkeypatch.setattr(source_snapshot_analysis, "console_warning", warnings.append)

    assert load_source_path_map(tmp_path) == {
        "/app/bin/../kernel.cpp": "/app/kernel.cpp"
    }
    assert len(warnings) == 1
    assert str(invalid_map_path) in warnings[0]


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
