# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for tests/common.py helpers."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from tests.common import (
    COUNTER_RESULT_COLUMNS,
    check_counter_results,
    check_sysinfo,
    read_counter_results,
    write_result_csv,
)


@pytest.fixture
def counter_frame():
    """One valid long-form dispatch with every field consumed by analysis."""
    row = dict.fromkeys(COUNTER_RESULT_COLUMNS, 0)
    row.update(
        Kernel_Name="vecCopy",
        Counter_Name="SQ_WAVES",
        Counter_Value=4,
        Start_Timestamp=10,
        End_Timestamp=20,
    )
    return pd.DataFrame([row])


@pytest.fixture
def sysinfo_frame():
    return pd.DataFrame([
        dict(
            gpu_model="MI350",
            gpu_arch="gfx950",
            cu_per_gpu=256,
            se_per_gpu=32,
            simd_per_cu=4,
        )
    ])


def test_check_resource_allocation_no_ctest(monkeypatch):
    """
    Test check_resource_allocation when CTEST_RESOURCE_GROUP_COUNT is not set.
    Should return without setting HIP_VISIBLE_DEVICES.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for modifying environment
    """
    monkeypatch.delenv("CTEST_RESOURCE_GROUP_COUNT", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    from tests.common import check_resource_allocation

    result = check_resource_allocation()

    assert result is None
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_check_resource_allocation_with_gpu_resource(monkeypatch):
    """
    Test check_resource_allocation when CTEST resource allocation is
    enabled with GPU resource. Should extract GPU ID and set HIP_VISIBLE_DEVICES.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for modifying environment
    """
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_COUNT", "1")
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_0_GPUS", "id:2,slots:1")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    from tests.common import check_resource_allocation

    result = check_resource_allocation()

    assert result is None
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"


def test_check_resource_allocation_no_gpu_resource(monkeypatch):
    """
    Test check_resource_allocation when CTEST is enabled but no GPU
    resource is specified.Should return without setting HIP_VISIBLE_DEVICES.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for modifying environment
    """
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_COUNT", "1")
    monkeypatch.delenv("CTEST_RESOURCE_GROUP_0_GPUS", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    from tests.common import check_resource_allocation

    result = check_resource_allocation()

    assert result is None
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_check_resource_allocation_malformed_resource(monkeypatch):
    """
    Test check_resource_allocation with malformed CTEST_RESOURCE_GROUP_0_GPUS format.
    Should handle gracefully without crashing.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for modifying environment
    """
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_COUNT", "1")
    monkeypatch.setenv("CTEST_RESOURCE_GROUP_0_GPUS", "malformed_resource_string")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

    from tests.common import check_resource_allocation

    try:
        result = check_resource_allocation()
        assert result is None
    except (ValueError, IndexError):
        pass


# =============================================================================
# FILE PATTERN MATCHING TESTS
# =============================================================================


def test_check_file_pattern_match_found():
    """
    Test check_file_pattern when the pattern is found in the file.
    Should return True.
    """
    from tests.common import check_file_pattern

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("This is a test file\nwith multiple lines\nand some pattern text\n")
        temp_file_path = f.name

    try:
        result = check_file_pattern("pattern", temp_file_path)
        assert result is True

        result = check_file_pattern(r"test.*file", temp_file_path)
        assert result is True

    finally:
        os.unlink(temp_file_path)


def test_check_file_pattern_file_not_found():
    """
    Test check_file_pattern when the file doesn't exist.
    Should raise FileNotFoundError.
    """
    from tests.common import check_file_pattern

    with pytest.raises(FileNotFoundError):
        check_file_pattern("pattern", "/nonexistent/file/path.txt")


@pytest.mark.parametrize("required", [(), ("SQ_WAVES",)])
def test_check_counter_results_valid(counter_frame, required):
    assert (
        check_counter_results(counter_frame, required_counters=required)
        is counter_frame
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("Counter_Value", "invalid"),
        ("Counter_Value", float("inf")),
        ("Counter_Value", float("nan")),
        ("Start_Timestamp", float("inf")),
        ("End_Timestamp", float("nan")),
        ("Start_Timestamp", 20),
        ("Start_Timestamp", 21),
        ("Kernel_Name", " "),
        ("Kernel_Name", None),
        ("Counter_Name", ""),
    ],
)
def test_check_counter_results_malformed(counter_frame, column, value):
    counter_frame[column] = value
    with pytest.raises(AssertionError):
        check_counter_results(counter_frame)


@pytest.mark.parametrize("column", COUNTER_RESULT_COLUMNS)
def test_check_counter_results_missing_column(counter_frame, column):
    with pytest.raises(AssertionError, match="Missing counter columns"):
        check_counter_results(counter_frame.drop(columns=column))


def test_check_counter_results_empty(counter_frame):
    with pytest.raises(AssertionError, match="empty"):
        check_counter_results(counter_frame.iloc[:0])


def test_check_counter_results_missing_required(counter_frame):
    with pytest.raises(AssertionError, match="Missing required counters"):
        check_counter_results(counter_frame, required_counters=("absent",))


def test_read_counter_results_sorted(tmp_path, counter_frame):
    for index in (2, 0, 1):
        frame = counter_frame.assign(Counter_Value=index)
        write_result_csv(tmp_path, frame.to_csv(index=False), pass_index=index)
    assert read_counter_results(tmp_path).Counter_Value.tolist() == [0, 1, 2]


def test_read_counter_results_missing(tmp_path):
    with pytest.raises(AssertionError, match="No counter result files"):
        read_counter_results(tmp_path)


@pytest.mark.parametrize("rows", [1, 2])
def test_check_sysinfo_valid(tmp_path, sysinfo_frame, rows):
    path = tmp_path / "sysinfo.csv"
    pd.concat([sysinfo_frame] * rows).to_csv(path, index=False)
    assert len(check_sysinfo(path)) == rows


@pytest.mark.parametrize(
    "column,value",
    [
        ("gpu_model", " "),
        ("gpu_model", None),
        ("gpu_arch", "unknown"),
        ("cu_per_gpu", 0),
        ("se_per_gpu", -1),
        ("simd_per_cu", "invalid"),
        ("cu_per_gpu", float("inf")),
        ("se_per_gpu", float("nan")),
    ],
)
def test_check_sysinfo_malformed_every_row(tmp_path, sysinfo_frame, column, value):
    frame = pd.concat([sysinfo_frame, sysinfo_frame], ignore_index=True)
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    path = tmp_path / "sysinfo.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_sysinfo(path)


@pytest.mark.parametrize("invalid", ["empty", "missing-column"])
def test_check_sysinfo_incomplete(tmp_path, sysinfo_frame, invalid):
    frame = (
        sysinfo_frame.iloc[:0]
        if invalid == "empty"
        else sysinfo_frame.drop(columns="gpu_model")
    )
    path = tmp_path / "sysinfo.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_sysinfo(path)


def test_profile_helpers_committed_mi350():
    workload = Path(__file__).parents[1] / "workloads/vcopy/MI350"
    results = check_counter_results(
        read_counter_results(workload), required_counters=("SQ_WAVES",)
    )
    assert (results.loc[results.Counter_Name == "SQ_WAVES", "Counter_Value"] > 0).all()
    assert set(check_sysinfo(workload / "sysinfo.csv").gpu_arch) == {"gfx950"}
