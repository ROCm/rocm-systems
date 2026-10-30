# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/output_location.py."""

from datetime import datetime
from pathlib import Path

import pytest

from rocprof_compute_analyze.output_location import (
    kernel_folder_name,
    plan_run_layout,
    resolve_run_folder,
    validate_output_name,
    workload_folder_name,
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


def test_existing_named_folder_appends_timestamp(tmp_path) -> None:
    """An occupied explicit name gets a timestamp suffix."""
    (tmp_path / "X").mkdir()

    assert resolve_run_folder(tmp_path, "X", NOW) == tmp_path / f"X_{NOW:%Y%m%d_%H%M%S}"


def test_existing_named_timestamp_appends_first_free_suffix(tmp_path) -> None:
    """An occupied timestamped name gets _1, then _2, until it is free."""
    timestamp_name = f"X_{NOW:%Y%m%d_%H%M%S}"
    (tmp_path / "X").mkdir()
    (tmp_path / timestamp_name).mkdir()
    assert resolve_run_folder(tmp_path, "X", NOW) == tmp_path / f"{timestamp_name}_1"

    (tmp_path / f"{timestamp_name}_1").mkdir()
    assert resolve_run_folder(tmp_path, "X", NOW) == tmp_path / f"{timestamp_name}_2"


def test_existing_named_file_appends_timestamp(tmp_path) -> None:
    """A file at the explicit name is treated as occupied."""
    (tmp_path / "X").write_text("", encoding="utf-8")

    assert resolve_run_folder(tmp_path, "X", NOW) == tmp_path / f"X_{NOW:%Y%m%d_%H%M%S}"


@pytest.mark.parametrize("name", ["", "a b", "a/b", "..", "x.db", "ü", "x\n"])
def test_invalid_name_errors(tmp_path, name) -> None:
    """Names outside [A-Za-z0-9_-] are rejected before any path is checked."""
    with pytest.raises(ValueError) as error:
        resolve_run_folder(tmp_path, name, NOW)

    assert "only alphanumeric characters" in str(error.value)


@pytest.mark.parametrize("name", ["a-B_9", "-", "_"])
def test_valid_name_passes(name) -> None:
    """Letters, digits, underscores and hyphens are accepted."""
    validate_output_name(name)


def test_resolved_path_keeps_relative_directory() -> None:
    """A relative directory stays relative; the caller decides the base."""
    assert resolve_run_folder(Path("out"), "X", NOW) == Path("out") / "X"


@pytest.mark.parametrize(
    ("position", "workload_count", "expected"),
    [(1, 1, "workload-1"), (2, 3, "workload-2"), (2, 12, "workload-02")],
    ids=["single", "one_digit", "two_digits"],
)
def test_workload_folder_name_pads_to_the_last_position(
    position, workload_count, expected
) -> None:
    """The 1-based -p position is zero-padded to the width of the last one."""
    assert workload_folder_name(position, workload_count) == expected


@pytest.mark.parametrize(
    ("position", "kernel_count", "expected"),
    [
        (0, 1, "kernel-0"),
        (3, 10, "kernel-3"),
        (3, 11, "kernel-03"),
        (42, 1200, "kernel-0042"),
    ],
    ids=["single", "last_rank_is_9", "last_rank_is_10", "four_digits"],
)
def test_kernel_folder_name_pads_to_the_last_rank(
    position, kernel_count, expected
) -> None:
    """The zero-based rank is zero-padded to the width of the last rank."""
    assert kernel_folder_name(position, kernel_count) == expected


def test_kernel_folder_names_sort_in_rank_order() -> None:
    """A sorted listing of the kernel folders follows the top-stats order."""
    folder_names = [kernel_folder_name(rank, 12) for rank in range(12)]
    assert sorted(folder_names) == folder_names


def test_plan_run_layout_numbers_workloads_in_path_order() -> None:
    """Each -p path gets the folder of its 1-based position, in the run folder."""
    layout = plan_run_layout(Path("run"), ["/data/b", "/data/a"])

    assert layout.run_folder == Path("run")
    assert layout.workload_folders == {
        "/data/b": Path("run") / "workload-1",
        "/data/a": Path("run") / "workload-2",
    }
