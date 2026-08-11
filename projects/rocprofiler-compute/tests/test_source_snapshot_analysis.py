# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling.source_snapshot_analysis."""

import hashlib

import pytest

from pc_sampling.source_snapshot_analysis import (
    parse_source_frames,
    read_source_file_digest_and_lines,
    resolve_snapshot_path,
)


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
