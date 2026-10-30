# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/output_location.py."""

from datetime import datetime
from pathlib import Path

import pytest

from rocprof_compute_analyze.output_location import (
    InvalidOutputNameError,
    OutputFolderExistsError,
    resolve_run_folder,
    validate_output_name,
)

NOW = datetime(2026, 10, 30, 14, 5, 9)
TIMESTAMP_NAME = "rocprof_compute_analyze_20261030_140509"


def test_default_name_is_timestamped(tmp_path) -> None:
    """Without a name the run folder is named after the local time."""
    assert resolve_run_folder(tmp_path, None, NOW) == tmp_path / TIMESTAMP_NAME


def test_default_name_appends_first_free_suffix(tmp_path) -> None:
    """A taken timestamp name gets _1, then _2, until the name is free."""
    (tmp_path / TIMESTAMP_NAME).mkdir()
    assert resolve_run_folder(tmp_path, None, NOW) == tmp_path / f"{TIMESTAMP_NAME}_1"

    (tmp_path / f"{TIMESTAMP_NAME}_1").mkdir()
    assert resolve_run_folder(tmp_path, None, NOW) == tmp_path / f"{TIMESTAMP_NAME}_2"


def test_default_name_treats_a_file_as_taken(tmp_path) -> None:
    """A file at the timestamp name blocks it just as a folder does."""
    (tmp_path / TIMESTAMP_NAME).write_text("", encoding="utf-8")
    assert resolve_run_folder(tmp_path, None, NOW) == tmp_path / f"{TIMESTAMP_NAME}_1"


def test_missing_directory_is_resolved_but_not_created(tmp_path) -> None:
    """A missing parent is allowed, and resolving it creates nothing."""
    directory = tmp_path / "missing" / "parents"

    assert resolve_run_folder(directory, None, NOW) == directory / TIMESTAMP_NAME
    assert resolve_run_folder(directory, "X", NOW) == directory / "X"
    assert not (tmp_path / "missing").exists()


def test_directory_is_taken_literally(tmp_path) -> None:
    """A ~ in the directory is not expanded."""
    directory = tmp_path / "~"
    assert resolve_run_folder(directory, "X", NOW) == directory / "X"


def test_named_folder_has_no_timestamp_or_suffix(tmp_path) -> None:
    """A free name is used as given."""
    assert resolve_run_folder(tmp_path, "X", NOW) == tmp_path / "X"


def test_existing_named_folder_errors(tmp_path) -> None:
    """A user-chosen name is never reused."""
    (tmp_path / "X").mkdir()

    with pytest.raises(OutputFolderExistsError) as error:
        resolve_run_folder(tmp_path, "X", NOW)

    assert error.value.path == tmp_path / "X"
    assert "already exists" in str(error.value)


@pytest.mark.parametrize("name", ["", "a b", "a/b", "..", "x.db", "ü", "x\n"])
def test_invalid_name_errors(tmp_path, name) -> None:
    """Names outside [A-Za-z0-9_-] are rejected before any path is checked."""
    with pytest.raises(InvalidOutputNameError) as error:
        resolve_run_folder(tmp_path, name, NOW)

    assert error.value.name == name
    assert "only alphanumeric characters" in str(error.value)


@pytest.mark.parametrize("name", ["a-B_9", "-", "_"])
def test_valid_name_passes(name) -> None:
    """Letters, digits, underscores and hyphens are accepted."""
    validate_output_name(name)


def test_resolved_path_keeps_relative_directory() -> None:
    """A relative directory stays relative; the caller decides the base."""
    assert resolve_run_folder(Path("out"), "X", NOW) == Path("out") / "X"
