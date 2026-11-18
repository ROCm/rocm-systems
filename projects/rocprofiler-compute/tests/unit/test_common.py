# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for tests/common.py helpers."""

import os
import tempfile

import pytest


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
