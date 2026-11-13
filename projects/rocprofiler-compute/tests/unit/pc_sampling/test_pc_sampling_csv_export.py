# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling_csv_export.py."""

import csv
from pathlib import Path

import pytest

from pc_sampling.pc_sampling_csv_export import (
    MAX_SHORT_NAME_LENGTH,
    _sanitize_short_name,
    build_header,
    resolve_isa_relative_path,
    write_grouped_csv_files,
)


def resolve_export_path_under(directory, file_key):
    """Name one file after its key, beneath a directory."""
    return directory.joinpath(*file_key).with_suffix(".csv")


def read_exported_rows(export_path):
    """Return one exported file as a list of string rows."""
    with export_path.open(newline="", encoding="utf-8") as export_file:
        return list(csv.reader(export_file))


def test_build_header_places_stall_reasons_between_leading_and_trailing_columns():
    """Each stall reason becomes a prefixed column of its own."""
    header = build_header(["Count"], ["NONE", "WAITCNT"], ["Source"])

    assert header == ["Count", "Stall NONE", "Stall WAITCNT", "Source"]


def test_build_header_carries_no_stall_columns_without_reasons():
    """A host_trap workload records no stall reason, so it gets no stall column."""
    assert build_header(["Count"], [], ["Source"]) == ["Count", "Source"]


def test_resolve_isa_relative_path_names_kernel_code_object_and_process():
    """An ISA file is filed by short name and uuid, then by object and process."""
    assert resolve_isa_relative_path(7, "vecCopy_2", 5, 42) == Path(
        "vecCopy_2_uuid_7/isa_code_object_id_5_pid_42.csv"
    )


def test_resolve_isa_relative_path_falls_back_when_no_short_name_was_captured():
    """A kernel without a short name keeps the folder name earlier releases used."""
    assert resolve_isa_relative_path(7, None, 5, 42).parent.name == "kernel_uuid_7"


def test_resolve_isa_relative_path_keeps_a_shared_short_name_unique():
    """Two overloads truncate to one short name, so the uuid separates them."""
    relative_paths = {
        resolve_isa_relative_path(kernel_uuid, "gemm", 5, 42) for kernel_uuid in (7, 8)
    }

    assert len(relative_paths) == 2


@pytest.mark.parametrize(
    ("short_name", "expected"),
    [
        ("vecCopy_2", "vecCopy_2"),
        ("__amd_rocclr_streamOpsDecrement.kd", "amd_rocclr_streamOpsDecrement"),
        ("operator/", "operator"),
        ("cub::DeviceReduceKernel", "cub_DeviceReduceKernel"),
        ("my kernel", "my_kernel"),
        ("$_0::__cxx11", "0_cxx11"),
        ("a" * 100, "a" * MAX_SHORT_NAME_LENGTH),
        (None, None),
    ],
    ids=[
        "plain_identifier",
        "kernel_descriptor_suffix",
        "operator_slash",
        "namespace_qualifier",
        "whitespace",
        "lambda_spelling",
        "over_length",
        "missing",
    ],
)
def test_sanitize_short_name(short_name, expected):
    """A short name reduces to a path component without losing its identity."""
    assert _sanitize_short_name(short_name) == expected


def test_write_grouped_csv_files_writes_one_file_per_file_key(tmp_path):
    """Each run of rows that share a file key is written to its own file."""
    rows = [("a", 1), ("a", 2), ("b", 3)]

    file_count = write_grouped_csv_files(
        iter(rows),
        1,
        ["Value"],
        lambda file_key: resolve_export_path_under(tmp_path, file_key),
        lambda file_key, file_rows: file_rows,
    )

    assert file_count == 2
    assert read_exported_rows(tmp_path / "a.csv") == [["Value"], ["1"], ["2"]]
    assert read_exported_rows(tmp_path / "b.csv") == [["Value"], ["3"]]


def test_write_grouped_csv_files_writes_nothing_without_rows(tmp_path):
    """No rows leave no folder behind."""
    file_count = write_grouped_csv_files(
        iter([]),
        1,
        ["Value"],
        lambda file_key: resolve_export_path_under(tmp_path / "out", file_key),
        lambda file_key, file_rows: file_rows,
    )

    assert file_count == 0
    assert not (tmp_path / "out").exists()
