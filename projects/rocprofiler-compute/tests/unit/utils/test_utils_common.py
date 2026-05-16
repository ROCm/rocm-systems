# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import io
import logging
import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import utils.utils_common as utils_common

class MockArgs:
    def __init__(self, **kwargs):
        # Set kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


logging.trace = lambda *args, **kwargs: None

# =============================================================================
# VERSION UTILITIES TESTS
# =============================================================================


def test_get_version_finds_version_in_home(tmp_path, monkeypatch):
    """Test that get_version correctly reads version and SHA from a VERSION file in the
    given directory.

    Args:
        tmp_path (Path): Temporary path provided by pytest for test isolation.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to modify or simulate behavior
            of modules/functions.

    Returns:
        None: Asserts correctness of version, SHA, and mode returned by get_version.
    """
    version_content = "1.2.3"
    version_file = tmp_path / "VERSION"
    version_file.write_text(version_content)
    monkeypatch.setattr(
        utils_common, "capture_subprocess_output", lambda *a, **k: (True, "abc123")
    )
    monkeypatch.setattr(
        utils_common,
        "console_error",
        lambda *a, **k: pytest.fail("console_error should not be called"),
    )
    result = utils_common.get_version(tmp_path)
    assert result["version"] == version_content
    assert result["sha"] == "abc123"
    assert result["mode"] == "dev"


def test_get_version_finds_version_in_parent(tmp_path, monkeypatch):
    """
    Test that get_version finds VERSION file in a parent directory when not present
    in the given directory.

    Args:
        tmp_path (Path): Temporary path provided by pytest for test isolation.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to modify or simulate behavior
            of modules/functions.

    Returns:
        None: Asserts correctness of version, SHA, and mode returned by get_version.
    """
    parent = tmp_path / "parent"
    parent.mkdir()
    version_content = "2.0.0"
    version_file = parent / "VERSION"
    version_file.write_text(version_content)
    monkeypatch.setattr(
        utils_common, "capture_subprocess_output", lambda *a, **k: (True, "def456")
    )
    monkeypatch.setattr(
        utils_common,
        "console_error",
        lambda *a, **k: pytest.fail("console_error should not be called"),
    )
    child = parent / "child"
    child.mkdir()
    result = utils_common.get_version(child)
    assert result["version"] == version_content
    assert result["sha"] == "def456"
    assert result["mode"] == "dev"


def test_get_version_console_error_when_no_version(monkeypatch):
    """
    Test that get_version calls console_error when no VERSION file is found in any
    directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture to modify or simulate
        behavior of modules/functions.

    Returns:
        None: Asserts that console_error is called with the expected message and
        raises RuntimeError.
    """
    fake_path = Path("/nonexistent/path")
    monkeypatch.setattr(builtins, "open", mock.Mock(side_effect=FileNotFoundError))
    called = {}

    def fake_console_error(msg, *args, **kwargs):
        called["msg"] = msg
        raise RuntimeError("console_error called")

    monkeypatch.setattr(utils_common, "console_error", fake_console_error)
    monkeypatch.setattr(
        utils_common, "capture_subprocess_output", lambda *a, **k: (False, "")
    )
    with pytest.raises(RuntimeError, match="console_error called"):
        utils_common.get_version(fake_path)
    assert "Cannot find VERSION file" in called["msg"]


def test_get_version_git_success(tmp_path, monkeypatch):
    """
    Test get_version returns correct version info when git command succeeds.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts version, sha, and mode are correct.
    """
    version_content = "1.0.0"
    version_file = tmp_path / "VERSION"
    version_file.write_text(version_content)
    monkeypatch.setattr(
        "utils.utils_common.capture_subprocess_output", lambda *a, **k: (True, "abc123")
    )
    monkeypatch.setattr(
        "utils.logger.console_error",
        lambda *a, **k: pytest.fail("console_error should not be called"),
    )
    result = utils_common.get_version(tmp_path)
    assert result["version"] == version_content
    assert result["sha"] == "abc123"
    assert result["mode"] == "dev"


def test_get_version_git_fails_sha_file(tmp_path, monkeypatch):
    """
    Test get_version returns correct version info when git fails but VERSION.sha exists.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts version, sha, and mode are correct.
    """
    version_content = "2.0.0"
    sha_content = "def456"
    version_file = tmp_path / "VERSION"
    sha_file = tmp_path / "VERSION.sha"
    version_file.write_text(version_content)
    sha_file.write_text(sha_content)

    def fail_git(*a, **k):
        return (False, "git error")

    monkeypatch.setattr("utils.utils_common.capture_subprocess_output", fail_git)
    monkeypatch.setattr(
        "utils.logger.console_error",
        lambda *a, **k: pytest.fail("console_error should not be called"),
    )
    result = utils_common.get_version(tmp_path)
    assert result["version"] == version_content
    assert result["sha"] == sha_content
    assert result["mode"] == "release"


def test_get_version_git_and_sha_fail(tmp_path, monkeypatch):
    """
    Test get_version returns unknown sha and mode when both git and VERSION.sha fail.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts version is correct, sha and mode are 'unknown'.
    """
    version_content = "3.0.0"
    version_file = tmp_path / "VERSION"
    version_file.write_text(version_content)

    def fail_git(*a, **k):
        return (False, "git error")

    monkeypatch.setattr("utils.utils_common.capture_subprocess_output", fail_git)
    monkeypatch.setattr(
        "utils.logger.console_error",
        lambda *a, **k: pytest.fail("console_error should not be called"),
    )

    result = utils_common.get_version(tmp_path)
    assert result["version"] == version_content
    assert result["sha"] == "unknown"
    assert result["mode"] == "unknown"

# =============================================================================
# ROCPROF DETECTION TESTS
# =============================================================================


def test_detect_rocprof_env_rocprof_not_found(monkeypatch):
    """
    Test detect_rocprof when ROCPROF is set to 'rocprof' but the binary cannot be
    found. Should revert to default 'rocprof' and call console_warning, then fail
    with console_error.
    """

    class DummyArgs:
        rocprofiler_sdk_tool_path = "/fake/path"

    # Set ROCPROF to 'rocprof'
    monkeypatch.setenv("ROCPROF", "rocprofv3")
    # shutil.which returns None for 'rocprof'
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    # Track calls to console_warning and console_error
    warnings = []
    errors = []
    monkeypatch.setattr(
        "utils.utils_common.console_warning", lambda msg, *a, **k: warnings.append(msg)
    )

    def fake_console_error(msg, *a, **k):
        errors.append(msg)
        raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_common.console_error", fake_console_error)

    with pytest.raises(RuntimeError, match="console_error called"):
        utils_common.detect_rocprof(DummyArgs())
    assert any(
        "Please verify installation or set ROCPROF environment variable" in e
        for e in errors
    )


def test_detect_rocprof_env_rocprof_found(monkeypatch):
    """
    Test detect_rocprof when ROCPROF is set to 'rocprof' and the binary is found.
    Should resolve the path and return 'rocprof'.
    """

    class DummyArgs:
        rocprofiler_sdk_tool_path = "/fake/path"

    monkeypatch.setenv("ROCPROF", "rocprof")
    # shutil.which returns a fake path for 'rocprof'
    monkeypatch.setattr(
        "shutil.which", lambda cmd: "/usr/bin/rocprof" if cmd == "rocprof" else None
    )
    # Path.resolve returns the same path for simplicity
    monkeypatch.setattr("pathlib.Path.resolve", lambda self: self)
    # Track debug logs
    logs = []
    monkeypatch.setattr(
        "utils.utils_common.console_debug", lambda msg, *a, **k: logs.append(str(msg))
    )

    result = utils_common.detect_rocprof(DummyArgs())
    assert result == "rocprof"
    assert any(
        "ROC Profiler: /usr/bin/rocprof" in log_entry
        or "rocprof_cmd is rocprof" in log_entry
        for log_entry in logs
    )


def test_detect_rocprof_env_not_set(monkeypatch):
    """
    Test detect_rocprof when ROCPROF is not set in the environment.
    Should default to 'rocprofv3' and resolve its path.
    """

    class DummyArgs:
        rocprofiler_sdk_tool_path = "/fake/path"

    monkeypatch.delenv("ROCPROF", raising=False)
    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    logs = []
    monkeypatch.setattr(
        "utils.utils_common.console_debug", lambda msg, *a, **k: logs.append(str(msg))
    )

    result = utils_common.detect_rocprof(DummyArgs())
    assert result == "rocprofiler-sdk"
    assert any(
        "rocprofiler_sdk_path is /fake/path" in log_entry
        or "rocprof_cmd is rocprofiler-sdk" in log_entry
        for log_entry in logs
    )


def test_detect_rocprof_sdk(monkeypatch):
    """
    Test detect_rocprof when ROCPROF is set
    to 'rocprofiler-sdk' and the library path exists.
    Should return 'rocprofiler-sdk'.
    """

    class DummyArgs:
        rocprofiler_sdk_tool_path = "/some/sdk/path"

    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    logs = []
    monkeypatch.setattr(
        "utils.utils_common.console_debug", lambda msg, *a, **k: logs.append(str(msg))
    )

    result = utils_common.detect_rocprof(DummyArgs())
    assert result == "rocprofiler-sdk"
    assert any("rocprof_cmd is rocprofiler-sdk" in log_entry for log_entry in logs)

# =============================================================================
# Tests for convert_metric_id_to_panel_info function
# ============================================================================


def test_convert_metric_id_to_panel_info_zero_values():
    """Test convert_metric_id_to_panel_info with zero values in different positions.

    Args:
        None
    Returns:
        None: Asserts that zero values are handled correctly in metric IDs.
    """
    assert utils_common.convert_metric_id_to_panel_info("0") == ("0000", None, None)
    assert utils_common.convert_metric_id_to_panel_info("0.0") == ("0000", 0, None)
    assert utils_common.convert_metric_id_to_panel_info("5.0") == ("0500", 500, None)
    assert utils_common.convert_metric_id_to_panel_info("0.5") == ("0000", 5, None)


def test_convert_metric_id_to_panel_info_leading_zeros():
    """Test convert_metric_id_to_panel_info with leading zeros in metric IDs.

    Args:
        None
    Returns:
        None: Asserts that leading zeros are handled correctly.
    """
    assert utils_common.convert_metric_id_to_panel_info("04") == ("0400", None, None)
    assert utils_common.convert_metric_id_to_panel_info("4.02") == ("0400", 402, None)
    assert utils_common.convert_metric_id_to_panel_info("01.05") == ("0100", 105, None)


def test_convert_metric_id_to_panel_info_invalid_empty_string():
    """Test convert_metric_id_to_panel_info with empty string raises exception.

    Args:
        None
    Returns:
        None: Asserts that empty string raises ValueError.
    """
    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("")


def test_convert_metric_id_to_panel_info_invalid_too_many_parts():
    """Test convert_metric_id_to_panel_info with more than two parts raises exception.

    Args:
        None
    Returns:
        None: Asserts that metric IDs with more than two parts raise Exception.
    """
    with pytest.raises(Exception, match="Invalid metric id"):
        utils_common.convert_metric_id_to_panel_info("4.02.1.5")

    with pytest.raises(Exception, match="Invalid metric id"):
        utils_common.convert_metric_id_to_panel_info("1.2.3.4")

    with pytest.raises(Exception, match="Invalid metric id"):
        utils_common.convert_metric_id_to_panel_info("4.02.1.5")


def test_convert_metric_id_to_panel_info_invalid_non_numeric():
    """Test convert_metric_id_to_panel_info with non-numeric values raises exception.

    Args:
        None
    Returns:
        None: Asserts that non-numeric metric IDs raise ValueError.
    """
    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("abc")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("4.abc")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("abc.02")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("4.02abc")


def test_convert_metric_id_to_panel_info_three_floating_point():
    """Test convert_metric_id_to_panel_info with floating
    point numbers in unexpected format.

    Args:
        None
    Returns:
        None: Asserts behavior with floating point representations.
    """
    assert utils_common.convert_metric_id_to_panel_info("4.0.2") == ("0400", 400, 2)
    assert utils_common.convert_metric_id_to_panel_info("4.2.0") == ("0400", 402, 0)
    assert utils_common.convert_metric_id_to_panel_info("4.0.3") == ("0400", 400, 3)


def test_convert_metric_id_to_panel_info_edge_case_whitespace():
    """Test convert_metric_id_to_panel_info with whitespace in metric IDs.

    Args:
        None
    Returns:
        None: Asserts that whitespace is handled (int() strips whitespace).
    """
    assert utils_common.convert_metric_id_to_panel_info(" 4") == ("0400", None, None)
    assert utils_common.convert_metric_id_to_panel_info("4 ") == ("0400", None, None)
    assert utils_common.convert_metric_id_to_panel_info("4 . 02") == ("0400", 402, None)


def test_convert_metric_id_to_panel_info_edge_case_dot_only():
    """Test convert_metric_id_to_panel_info with only dot character raises exception.

    Args:
        None
    Returns:
        None: Asserts that metric ID with only dot raises Exception.
    """
    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("..")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info(".")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info("4.")

    with pytest.raises(ValueError):
        utils_common.convert_metric_id_to_panel_info(".02")

# =============================================================================
# --- New test functions for add_counter_extra_config_input_yaml ---
# =============================================================================


def test_add_counter_invalid_architectures_type():
    """
    Test that add_counter_extra_config_input_yaml raises TypeError
    if 'architectures' is not a list.
    """
    data = {}
    with pytest.raises(TypeError, match="'architectures' must be a list, got str"):
        utils_common.add_counter_extra_config_input_yaml(
            data=data,
            counter_name="test_counter",
            description="A test counter",
            expression="expr1",
            architectures="not_a_list",  # Invalid type
            properties=["prop1"],
        )
    with pytest.raises(TypeError, match="'architectures' must be a list, got int"):
        utils_common.add_counter_extra_config_input_yaml(
            data=data,
            counter_name="test_counter_2",
            description="A test counter 2",
            expression="expr2",
            architectures=123,  # Invalid type
            properties=["prop1"],
        )


def test_add_counter_invalid_properties_type():
    """
    Test that add_counter_extra_config_input_yaml raises TypeError
    if 'properties' is not a list (and not None).
    """
    data = {}
    with pytest.raises(TypeError, match="'properties' must be a list, got str"):
        utils_common.add_counter_extra_config_input_yaml(
            data=data,
            counter_name="test_counter",
            description="A test counter",
            expression="expr1",
            architectures=["arch1"],
            properties="not_a_list",  # Invalid type
        )
    with pytest.raises(TypeError, match="'properties' must be a list, got dict"):
        utils_common.add_counter_extra_config_input_yaml(
            data=data,
            counter_name="test_counter_2",
            description="A test counter 2",
            expression="expr2",
            architectures=["arch1"],
            properties={"key": "value"},  # Invalid type
        )


def test_add_counter_overwrite_existing():
    """
    Test that add_counter_extra_config_input_yaml overwrites an existing counter
    with the same name.
    """
    data = {}
    counter_name = "MY_COUNTER"
    initial_description = "Initial version"
    initial_expression = "initial_expr"
    initial_architectures = ["gfx900"]
    initial_properties = ["P_INIT"]

    # Add the counter for the first time
    data = utils_common.add_counter_extra_config_input_yaml(
        data=data,
        counter_name=counter_name,
        description=initial_description,
        expression=initial_expression,
        architectures=initial_architectures,
        properties=initial_properties,
    )

    assert len(data["rocprofiler-sdk"]["counters"]) == 1
    assert data["rocprofiler-sdk"]["counters"][0]["name"] == counter_name
    assert data["rocprofiler-sdk"]["counters"][0]["description"] == initial_description
    assert (
        data["rocprofiler-sdk"]["counters"][0]["definitions"][0]["expression"]
        == initial_expression
    )

    updated_description = "Updated version"  # noqa
    updated_expression = "updated_expr"  # noqa
    updated_architectures = ["gfx908"]  # noqa
    updated_properties = ["P_UPDATED", "P_NEW"]  # noqa

# =============================================================================
# additional test detect_rocprof console error
# =============================================================================


@mock.patch.dict(os.environ, {"ROCPROF": "rocprofiler-sdk"}, clear=True)
@mock.patch("utils.utils_common.console_error")
@mock.patch("utils.utils_common.Path")
def test_detect_rocprof_calls_console_error_if_sdk_path_invalid(
    mock_path_constructor, mock_console_error_func
):
    """
    Tests that detect_rocprof calls console_error when ROCPROF is 'rocprofiler-sdk'
    and the rocprofiler_sdk_tool_path does not exist.
    Focuses on the console_error call.
    """
    mock_path_instance = mock.Mock()
    mock_path_instance.exists.return_value = False
    mock_path_constructor.return_value = mock_path_instance

    fake_library_path = "/some/invalid/path/to/librocprofiler_sdk.so"
    args = MockArgs(rocprofiler_sdk_tool_path=fake_library_path)

    with mock.patch("utils.utils_common.console_debug") as mock_console_debug:  # noqa
        utils_common.detect_rocprof(args)

    expected_error_message = (
        "Could not find rocprofiler-sdk tool at " + fake_library_path
    )
    mock_console_error_func.assert_called_once_with(expected_error_message)

    mock_path_constructor.assert_called_once_with(fake_library_path)
    mock_path_instance.exists.assert_called_once()

def test_set_parser():
    from utils.utils_common import parse_sets_yaml

    result = parse_sets_yaml("gfx90a")

    assert "compute_thruput_util" in result
    assert result["compute_thruput_util"]["title"] == "Compute Throughput Utilization"


@pytest.mark.sci_notion
def test_scientific_notation_trigger_below_lower_bound():
    value = 0.0001
    result = utils_common.format_scientific_notation_if_needed(value)
    assert pytest.approx(float(result.strip()), rel=1e-9) == value


@pytest.mark.sci_notion
def test_scientific_notation_trigger_at_lower_bound():
    value = 0.01
    result = utils_common.format_scientific_notation_if_needed(value)
    assert pytest.approx(float(result.strip()), rel=1e-9) == value


@pytest.mark.sci_notion
def test_scientific_notation_trigger_just_below_upper_bound():
    value = 999999
    result = utils_common.format_scientific_notation_if_needed(value, precision=6)
    assert pytest.approx(float(result.strip()), rel=1e-6) == value


@pytest.mark.sci_notion
def test_scientific_notation_trigger_zero():
    value = 0
    result = utils_common.format_scientific_notation_if_needed(value)
    assert float(result.strip()) == value  # Exact match for zero


@pytest.mark.sci_notion
def test_scientific_notation_trigger_slightly_below_lower_bound():
    value = 0.009
    result = utils_common.format_scientific_notation_if_needed(value)
    assert pytest.approx(float(result.strip()), rel=1e-9) == value


@pytest.mark.sci_notion
def test_scientific_notation_trigger_well_below_lower_bound():
    value = 1e-5
    result = utils_common.format_scientific_notation_if_needed(value)
    assert pytest.approx(float(result.strip()), rel=1e-9) == value


@pytest.mark.sci_notion
def test_scientific_notation_trigger_well_above_upper_bound():
    value = 1e10
    result = utils_common.format_scientific_notation_if_needed(value)
    assert pytest.approx(float(result.strip()), rel=1e-9) == value


@pytest.mark.sci_notion
def test_alignment_and_width():
    value = 1e10
    result = utils_common.format_scientific_notation_if_needed(
        value,
        align=">",
        width_align=12,
        precision=2,
        fmt_type_align="f",
        max_length=8,
    )
    assert pytest.approx(float(result.strip()), rel=1e-9) == value
# =============================================================================
# validate_roofline_csv TESTS
# =============================================================================


def test_validate_roofline_csv_valid():
    """
    Test validate_roofline_csv returns True for a valid roofline.csv file.
    Creates a temporary directory with a properly formatted CSV.
    """
    from utils.utils_common import validate_roofline_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "roofline.csv"
        csv_path.write_text(
            "device,HBMBw,L2Bw,L1Bw,FP32Flops,FP64Flops\n"
            "0,1000.0,2000.0,3000.0,4000.0,5000.0\n"
        )

        is_valid, error_msg = validate_roofline_csv(tmpdir)

        assert is_valid is True
        assert error_msg == ""


def test_validate_roofline_csv_invalid_inconsistent_columns():
    """
    Test validate_roofline_csv returns False for a CSV with inconsistent row lengths.
    This simulates corrupted or incomplete benchmark data.
    """
    from utils.utils_common import validate_roofline_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "roofline.csv"
        csv_path.write_text(
            "device,HBMBw,L2Bw,L1Bw,FP32Flops,FP64Flops\n0,1000.0,2000.0,3000.0\n"
        )

        is_valid, error_msg = validate_roofline_csv(tmpdir)

        assert is_valid is False
        assert "Inconsistent row length" in error_msg
        assert "row 2" in error_msg

# =============================================================================
# Test rocm library resolver
# =============================================================================


@pytest.mark.misc
def test_version_to_numeric():
    """Test version_to_numeric helper function."""
    from utils.utils_common import version_to_numeric

    # Test normalized to max_len=3
    max_len = 3

    # Single component versions
    assert version_to_numeric([2], max_len) == 2_000_000  # 2 * 1000^2
    assert version_to_numeric([10], max_len) == 10_000_000  # 10 * 1000^2
    assert version_to_numeric([15], max_len) == 15_000_000  # 15 * 1000^2

    # Multi-component versions
    assert version_to_numeric([1, 2, 3], max_len) == 1_002_003  # 1*1000^2 + 2*1000 + 3
    assert version_to_numeric([2, 5, 3], max_len) == 2_005_003  # 2*1000^2 + 5*1000 + 3
    assert version_to_numeric([1, 2], max_len) == 1_002_000  # 1*1000^2 + 2*1000

    # Version comparisons - higher version numbers should produce higher values
    assert version_to_numeric([10], max_len) > version_to_numeric([2], max_len)
    assert version_to_numeric([10], max_len) > version_to_numeric([1, 2, 3], max_len)
    assert version_to_numeric([2], max_len) > version_to_numeric([1, 2, 3], max_len)
    assert version_to_numeric([2, 5, 3], max_len) > version_to_numeric([2], max_len)
    assert version_to_numeric([1, 2, 3], max_len) > version_to_numeric([1, 2], max_len)

    # Edge case: version components support 0-999
    assert version_to_numeric([999, 999, 999], max_len) == 999_999_999


@pytest.mark.misc
def test_resolve_rocm_library_path(tmp_path):
    """Test resolve_rocm_library_path with various scenarios."""
    from utils.utils_common import resolve_rocm_library_path

    # Test case 1: Empty path returns as-is
    assert resolve_rocm_library_path("") == ""
    assert resolve_rocm_library_path(None) is None

    # Test case 2: Exact path exists (unversioned)
    unversioned = tmp_path / "libtest.so"
    unversioned.touch()
    assert resolve_rocm_library_path(str(unversioned)) == str(unversioned)

    # Test case 3: Exact path exists (already versioned)
    versioned = tmp_path / "libfoo.so.1"
    versioned.touch()
    assert resolve_rocm_library_path(str(versioned)) == str(versioned)

    # Test case 4: Unversioned doesn't exist, fallback to versioned variant
    nonexistent = tmp_path / "libbar.so"
    versioned_bar = tmp_path / "libbar.so.1"
    versioned_bar.touch()
    assert resolve_rocm_library_path(str(nonexistent)) == str(versioned_bar)

    # Test case 5: Multiple versioned files, pick highest version deterministically
    multi_base = tmp_path / "libmulti.so"
    v1 = tmp_path / "libmulti.so.1"
    v123 = tmp_path / "libmulti.so.1.2.3"
    v12 = tmp_path / "libmulti.so.1.2"
    v2 = tmp_path / "libmulti.so.2"
    v1.touch()
    v123.touch()
    v12.touch()
    v2.touch()
    # Should pick .so.2 (highest major version)
    assert resolve_rocm_library_path(str(multi_base)) == str(v2)

    # Test case 6: Filters out non-numeric suffixes (e.g., .so.debug)
    filter_base = tmp_path / "libfilter.so"
    numeric_version = tmp_path / "libfilter.so.1"
    debug_file = tmp_path / "libfilter.so.debug"
    numeric_version.touch()
    debug_file.touch()
    # Should pick .so.1, not .so.debug
    assert resolve_rocm_library_path(str(filter_base)) == str(numeric_version)

    # Test case 7: Version comparison edge cases
    # 10.0 should beat 2.5.3 (not string comparison)
    version_base = tmp_path / "libversion.so"
    v10 = tmp_path / "libversion.so.10"
    v253 = tmp_path / "libversion.so.2.5.3"
    v10.touch()
    v253.touch()
    # Should pick .so.10 (10 > 2 in first position)
    assert resolve_rocm_library_path(str(version_base)) == str(v10)

    # Test case 8: No match at all, returns original path
    missing = tmp_path / "libmissing.so"
    assert resolve_rocm_library_path(str(missing)) == str(missing)

# =============================================================================
# BUILD METRIC LIST TESTS
# =============================================================================


class TestBuildMetricList:
    """Tests for build_metric_list and _metric_has_valid_expr."""

    # Maps YAML metric expression keys to their SUPPORTED_FIELD display names.
    _EXPR_KEY_TO_HEADER_DISPLAY = {
        "value": "Value",
        "avg": "Avg",
        "min": "Min",
        "max": "Max",
        "expr": "Expression",
        "median": "Median",
        "count": "Count",
    }

    @classmethod
    def setup_class(cls):
        from utils.utils_common import build_metric_list

        cls.build_metric_list = staticmethod(build_metric_list)

    def _build_test_panel_configs_for_single_metric(
        self, metric_name: str, expression_values: dict
    ):
        """
        Build panel_configs containing a single metric for testing.
        """
        from collections import OrderedDict

        header = {"metric": "Metric"}
        for key in expression_values:
            if key in self._EXPR_KEY_TO_HEADER_DISPLAY:
                header[key] = self._EXPR_KEY_TO_HEADER_DISPLAY[key]

        table = {
            "id": 201,
            "title": "Test Table",
            "header": header,
            "metric": {metric_name: expression_values},
        }
        if "expr" in expression_values:
            table["cli_style"] = "simple_box"

        panel_configs = OrderedDict()
        panel_configs[200] = {
            "id": 200,
            "title": "Test Panel",
            "data source": [{"metric_table": table}],
        }

        return panel_configs

    @staticmethod
    def _extract_leaf_metric_entries(metric_list):
        """Return only leaf metric entries whose ID has format 'panel.table.index'."""
        return {k: v for k, v in metric_list.items() if k.count(".") == 2}

    def test_given_metric_with_valid_value__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Valid Metric A", {"value": "AVG(COUNTER_A)"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Valid Metric A" in leaf_entries.values()

    def test_given_metric_with_python_none__it_doesnt_present_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Unsupported Metric B", {"value": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Unsupported Metric B" not in leaf_entries.values()

    def test_given_metric_with_string_none__it_doesnt_present_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Unsupported Metric C", {"value": "None"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Unsupported Metric C" not in leaf_entries.values()

    def test_given_expr_metric__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Expr Metric", {"expr": "(100 * COUNTER_B / COUNTER_C)"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Expr Metric" in leaf_entries.values()

    def test_given_metric_with_partial_avg_min_max__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Partial Metric", {"avg": "AVG(COUNTER_E)", "min": None, "max": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Partial Metric" in leaf_entries.values()

    def test_given_metric_with_all_none_avg_min_max__it_doesnt_present_in_metric_list(
        self,
    ):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "All None Metric", {"avg": None, "min": None, "max": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "All None Metric" not in leaf_entries.values()

# =============================================================================
# format_table_ascii TESTS
# =============================================================================


def test_format_table_ascii_basic():
    """Test format_table_ascii produces correct ASCII table output."""
    from utils.utils_common import format_table_ascii

    data = [
        {"Spec": "GPU Model", "Value": "MI300X", "Description": "The GPU model name."},
        {"Spec": "Max SCLK", "Value": "2100", "Description": "Maximum clock speed."},
    ]
    columns = ["Spec", "Value", "Description"]

    result = format_table_ascii(data, columns)

    # Check table structure
    assert "+-------+" in result  # Has separators
    assert "| index |" in result  # Has index column header
    assert "| Spec" in result  # Has Spec column
    assert "| GPU Model" in result  # Has data
    assert "| MI300X" in result  # Has value
    assert "| 2100" in result  # Has second row value


def test_format_table_ascii_text_wrapping():
    """Test that long Description text is wrapped at 40 characters."""
    from utils.utils_common import format_table_ascii

    long_desc = (
        "This is a very long description that should be wrapped "
        "across multiple lines in the table output."
    )
    data = [{"Spec": "Test", "Value": "123", "Description": long_desc}]
    columns = ["Spec", "Value", "Description"]

    result = format_table_ascii(data, columns)
    lines = result.split("\n")

    # Find lines containing description content (not separator lines)
    desc_lines = [
        ln for ln in lines if "|" in ln and "Description" not in ln and "---" not in ln
    ]
    # Should have multiple lines for the wrapped description
    assert len(desc_lines) > 1, "Long description should wrap to multiple lines"
# =============================================================================
# TESTS FOR reconfigure_stdio_utf8 FUNCTION
# =============================================================================


def test_reconfigure_stdio_utf8_calls_reconfigure_on_both_streams():
    """Both sys.stdout and sys.stderr should be reconfigured to utf-8/replace."""
    fake_stdout = mock.MagicMock()
    fake_stderr = mock.MagicMock()
    with mock.patch("utils.utils_common.sys") as fake_sys:
        fake_sys.stdout = fake_stdout
        fake_sys.stderr = fake_stderr
        utils_common.reconfigure_stdio_utf8()
    fake_stdout.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")
    fake_stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


def test_reconfigure_stdio_utf8_swallows_attribute_error():
    """Streams without a reconfigure attribute (captured / wrapped) are skipped."""
    fake_stdout = mock.MagicMock(spec=[])  # no reconfigure attribute
    fake_stderr = mock.MagicMock()
    with mock.patch("utils.utils_common.sys") as fake_sys:
        fake_sys.stdout = fake_stdout
        fake_sys.stderr = fake_stderr
        utils_common.reconfigure_stdio_utf8()  # must not raise
    fake_stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


def test_reconfigure_stdio_utf8_swallows_unsupported_operation():
    """io.UnsupportedOperation from a captured stream must be swallowed."""
    fake_stdout = mock.MagicMock()
    fake_stdout.reconfigure.side_effect = io.UnsupportedOperation("not seekable")
    fake_stderr = mock.MagicMock()
    with mock.patch("utils.utils_common.sys") as fake_sys:
        fake_sys.stdout = fake_stdout
        fake_sys.stderr = fake_stderr
        utils_common.reconfigure_stdio_utf8()  # must not raise
    fake_stderr.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")


def test_reconfigure_stdio_utf8_end_to_end_makes_non_ascii_print_safe():
    """After reconfigure, encoding to bytes via the wrapper must not raise."""
    raw = io.BytesIO()
    wrapper = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    with mock.patch("utils.utils_common.sys") as fake_sys:
        fake_sys.stdout = wrapper
        fake_sys.stderr = wrapper
        utils_common.reconfigure_stdio_utf8()
    wrapper.write("│ box │\n")  # would raise UnicodeEncodeError under ascii/strict
    wrapper.flush()
    assert raw.getvalue() == "│ box │\n".encode("utf-8")
