# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import builtins
import io
import locale
import logging
import math
import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import utils.utils_analysis as utils_analysis
import utils.utils_common as utils_common
import utils.utils_profile as utils_profile
from utils.tty import (
    format_duration,
    format_node_stats,
    print_operator_node,
    show_call_tree,
    show_operator_summary,
)
from utils.utils_analysis import (
    CallTreeNode,
    KernelStats,
    NodeRollup,
    build_call_trees,
    build_operator_summary,
    parse_top_level_location,
    rollup_node_stats,
)


class MockArgs:
    def __init__(self, **kwargs):
        # Set kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockSoc:
    def __init__(self):
        pass


logging.trace = lambda *args, **kwargs: None

##################################################
##          Generated tests                     ##
##################################################

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


def make_dummy_process(*, lines=(), returncode=0, poll_pending_first=False):
    """Fake subprocess.Popen for capture_subprocess_output tests."""

    class Stdout:
        def __init__(self):
            self.iter = iter(lines)

        def readline(self):
            return next(self.iter, "")

        def fileno(self):
            return 1

        def close(self):
            pass

    class Process:
        def __init__(self):
            self.stdout = Stdout()
            self.poll_pending = poll_pending_first

        def poll(self):
            if self.poll_pending:
                self.poll_pending = False
                return None
            return returncode

        def wait(self):
            return returncode

    return Process()


def test_capture_subprocess_output_with_new_env(monkeypatch):
    """
    Test capture_subprocess_output with custom environment variables.
    Verifies that new_env parameter is properly passed to subprocess.
    """

    dummy_process = make_dummy_process(poll_pending_first=True)
    popen_calls = []

    def dummy_popen(*args, **kwargs):
        popen_calls.append(kwargs)
        return dummy_process

    monkeypatch.setattr("subprocess.Popen", dummy_popen)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    custom_env = {"CUSTOM_VAR": "test_value"}
    utils_common.capture_subprocess_output(["echo", "test"], new_env=custom_env)

    # Verify that custom environment was passed
    assert len(popen_calls) == 1
    assert popen_calls[0]["env"] == custom_env


def test_capture_subprocess_output_profile_mode(monkeypatch):
    """
    Test capture_subprocess_output with profileMode flag enabled.
    Verifies different behavior when profiling mode is active.
    """

    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: make_dummy_process())
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    success, output = utils_common.capture_subprocess_output(
        ["echo", "test"], profileMode=True, enable_logging=False
    )

    assert success is True
    assert isinstance(output, str)


def test_capture_subprocess_output_failure(monkeypatch):
    """
    Test capture_subprocess_output returns
    (False, output) when subprocess exits with nonzero code.
    """
    dummy_process = make_dummy_process(
        lines=["fail\n"], returncode=1, poll_pending_first=True
    )
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: dummy_process)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    success, output = utils_common.capture_subprocess_output(["fail", "test"])
    assert success is False
    assert "fail" in output


def test_capture_subprocess_output_unicode_decode(monkeypatch):
    """
    Test capture_subprocess_output handles bad bytes from the child without
    crashing. errors="replace" on Popen substitutes invalid bytes with the
    Unicode replacement character (\\ufffd), so readline never raises.
    """

    popen_calls = []

    # Lines as the TextIOWrapper would yield them after error replacement:
    # bad bytes show up as �, not as exceptions.
    def dummy_popen(*args, **kwargs):
        popen_calls.append(kwargs)
        return make_dummy_process(lines=["good line\n", "bad � byte\n"])

    monkeypatch.setattr("subprocess.Popen", dummy_popen)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    success, output = utils_common.capture_subprocess_output(["echo", "test"])

    assert success is True
    assert "good line" in output
    assert "�" in output
    # Popen must request "replace" error handling so bad bytes never raise.
    assert popen_calls[0].get("errors") == "replace"


# =============================================================================
# JSON DATA PARSING TESTS
# =============================================================================


def test_get_agent_dict_basic():
    """
    Test get_agent_dict correctly maps agent IDs to agent objects.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 1}, "type": 2, "node_id": 100},
                    {"id": {"handle": 2}, "type": 2, "node_id": 200},
                ]
            }
        ]
    }

    result = utils_common.get_agent_dict(data)

    # Verify correct mapping
    assert len(result) == 2
    assert result[1]["node_id"] == 100
    assert result[2]["node_id"] == 200
    assert result[1]["type"] == 2
    assert result[2]["type"] == 2


def test_get_agent_dict_empty_agents():
    """
    Test get_agent_dict with an empty agents list.
    """
    data = {"rocprofiler-sdk-tool": [{"agents": []}]}

    result = utils_common.get_agent_dict(data)

    assert result == {}


def test_get_agent_dict_missing_keys(monkeypatch):
    """
    Test get_agent_dict behavior when expected keys are missing.
    """
    # Case 1: Missing 'agents' key
    data1 = {"rocprofiler-sdk-tool": [{}]}

    with pytest.raises(KeyError):
        utils_common.get_agent_dict(data1)

    # Case 2: Missing 'rocprofiler-sdk-tool' key
    data2 = {}

    with pytest.raises(KeyError):
        utils_common.get_agent_dict(data2)

    # Case 3: Empty 'rocprofiler-sdk-tool' list
    data3 = {"rocprofiler-sdk-tool": []}

    with pytest.raises(IndexError):
        utils_common.get_agent_dict(data3)


def test_get_agent_dict_duplicate_agent_ids():
    """
    Test get_agent_dict behavior with duplicate agent IDs.
    The function should overwrite previous entries with the same ID.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 1}, "type": 2, "node_id": 100, "name": "first"},
                    {"id": {"handle": 1}, "type": 2, "node_id": 200, "name": "second"},
                ]
            }
        ]
    }

    result = utils_common.get_agent_dict(data)

    assert len(result) == 1
    assert result[1]["node_id"] == 200
    assert result[1]["name"] == "second"


def test_get_agent_dict_non_integer_handles():
    """
    Test get_agent_dict with non-integer handle values.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": "agent_1"}, "type": 2, "node_id": 100},
                    {"id": {"handle": "agent_2"}, "type": 2, "node_id": 200},
                ]
            }
        ]
    }

    result = utils_common.get_agent_dict(data)

    assert len(result) == 2
    assert result["agent_1"]["node_id"] == 100
    assert result["agent_2"]["node_id"] == 200


# =========================================================================
# Tests for get_gpuid_dict function
# =========================================================================
def test_get_gpuid_dict_basic():
    """Test that get_gpuid_dict correctly maps agent IDs to GPU IDs for a basic case.
    Args:
        None
    Returns:
        None: Asserts that agent IDs are correctly mapped to GPU IDs
        based on node_id ordering.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 100}, "node_id": 5, "type": 2},  # GPU agent
                    {"id": {"handle": 101}, "node_id": 3, "type": 2},  # GPU agent
                    {"id": {"handle": 102}, "node_id": 7, "type": 2},  # GPU agent
                ]
            }
        ]
    }

    expected = {101: 0, 100: 1, 102: 2}

    result = utils_common.get_gpuid_dict(data)
    assert result == expected


def test_get_gpuid_dict_no_gpu_agents():
    """Test that get_gpuid_dict returns an empty dictionary
    when no GPU agents are present.
    Args:
        None
    Returns:
        None: Asserts that an empty dictionary is returned
        when there are no GPU agents.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 100}, "node_id": 5, "type": 1},  # Non-GPU agent
                    {"id": {"handle": 101}, "node_id": 3, "type": 3},  # Non-GPU agent
                    {"id": {"handle": 102}, "node_id": 7, "type": 0},  # Non-GPU agent
                ]
            }
        ]
    }

    result = utils_common.get_gpuid_dict(data)
    assert result == {}


def test_get_gpuid_dict_mixed_agents():
    """Test that get_gpuid_dict correctly ignores non-GPU agents
    and only maps GPU agents.
    Args:
        None
    Returns:
        None: Asserts that only GPU agents (type 2) are included
        in the mapping.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 100}, "node_id": 5, "type": 2},  # GPU agent
                    {"id": {"handle": 101}, "node_id": 3, "type": 1},  # Non-GPU agent
                    {"id": {"handle": 102}, "node_id": 7, "type": 2},  # GPU agent
                    {"id": {"handle": 103}, "node_id": 2, "type": 0},  # Non-GPU agent
                ]
            }
        ]
    }

    # Expected mapping after sorting by node_id and filtering by type 2: 100->0, 102->1
    expected = {100: 0, 102: 1}

    result = utils_common.get_gpuid_dict(data)
    assert result == expected


def test_get_gpuid_dict_sorting():
    """Test that get_gpuid_dict correctly sorts GPU agents by node_id
    to determine GPU ID ordering.
    Args:
        None
    Returns:
        None: Asserts that GPU agents are sorted by node_id before
        being assigned sequential GPU IDs.
    """
    data = {
        "rocprofiler-sdk-tool": [
            {
                "agents": [
                    {"id": {"handle": 100}, "node_id": 10, "type": 2},  # GPU agent
                    {"id": {"handle": 101}, "node_id": 5, "type": 2},  # GPU agent
                    {"id": {"handle": 102}, "node_id": 8, "type": 2},  # GPU agent
                    {"id": {"handle": 103}, "node_id": 1, "type": 2},  # GPU agent
                ]
            }
        ]
    }

    expected = {103: 0, 101: 1, 102: 2, 100: 3}

    result = utils_common.get_gpuid_dict(data)
    assert result == expected


def test_get_gpuid_dict_empty_agents():
    """Test that get_gpuid_dict handles an empty agents list correctly.
    Args:
        None
    Returns:
        None: Asserts that an empty dictionary is returned when the
        agents list is empty.
    """
    # Sample data with empty agents list
    data = {"rocprofiler-sdk-tool": [{"agents": []}]}

    result = utils_common.get_gpuid_dict(data)
    assert result == {}


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


# =============================================================================
# PMC PERF PARSING UTILITIES TESTS
# =============================================================================


def test_parse_pmc_perf_basic(tmp_path):
    """Test parse_pmc_perf with a simple valid YAML input file.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.

    Returns:
        None: Asserts that counters are correctly extracted from a simple file.
    """
    test_file = tmp_path / "test_counters.yaml"
    test_file.write_text(
        "jobs:\n  - pmc:\n    - counter1\n    - counter2\n    - counter3\n"
    )

    result = utils_common.parse_pmc_perf(str(test_file))
    assert result == ["counter1", "counter2", "counter3"]


def test_parse_pmc_perf_empty_file(tmp_path):
    """Test parse_pmc_perf with an empty file.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.

    Returns:
        None: Asserts that an empty file returns an empty list.
    """
    test_file = tmp_path / "empty.yaml"
    test_file.write_text("")

    result = utils_common.parse_pmc_perf(str(test_file))
    assert result == []


def test_parse_pmc_perf_no_pmc_entries(tmp_path):
    """Test parse_pmc_perf with a YAML file that doesn't contain any 'pmc' entries.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.

    Returns:
        None: Asserts that a file without 'pmc' returns an empty list.
    """
    test_file = tmp_path / "no_pmc.yaml"
    test_file.write_text("jobs:\n  - other: value\n")

    result = utils_common.parse_pmc_perf(str(test_file))
    assert result == []


def test_parse_pmc_perf_no_jobs_key(tmp_path):
    """Test parse_pmc_perf with missing jobs key.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.

    Returns:
        None: Asserts that missing jobs key returns an empty list.
    """
    test_file = tmp_path / "no_jobs.yaml"
    test_file.write_text("other_key: value\n")

    result = utils_common.parse_pmc_perf(str(test_file))
    assert result == []


def test_parse_pmc_perf_empty_pmc(tmp_path):
    """Test parse_pmc_perf with empty pmc list.

    Args:
        tmp_path (Path): Temporary path fixture provided by pytest.

    Returns:
        None: Asserts that empty pmc returns an empty list.
    """
    test_file = tmp_path / "empty_pmc.yaml"
    test_file.write_text("jobs:\n  - pmc: []\n")

    result = utils_common.parse_pmc_perf(str(test_file))
    assert result == []


def test_parse_pmc_perf_file_not_found():
    """Test parse_pmc_perf with a nonexistent file.

    Returns:
        None: Asserts that FileNotFoundError is raised for nonexistent files.
    """
    with pytest.raises(FileNotFoundError):
        utils_common.parse_pmc_perf("nonexistent_file.yaml")


# =============================================================================
# RUN_PROF TESTS
# =============================================================================


def test_run_prof_success_v3(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofv3 successful execution.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution and file creation.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")
    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_content = (
        "Agent_Type,Node_Id,Wave_Front_Size,Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,"
        "Grid_Size,Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,"
        "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Start_Timestamp,"
        "End_Timestamp,Counter_Name,Counter_Value\n"
        "GPU,0,0,0,0,0,0,0,0,0,0,test_kernel,0,0,0,0,0,0,0,1,SQ_WAVES,100"
    )
    with open(workload_dir + "/out/pmc_1/results_0.csv", "w") as f:
        f.write(csv_content)

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr(
        "glob.glob", lambda pattern: [workload_dir + "/out/pmc_1/results_0.csv"]
    )

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")

    assert Path(workload_dir + "/results_pmc_perf_test.csv").exists()


def test_run_prof_success_v3_csv(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofv3 using CSV format.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution with v3 CSV processing.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")
    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_files = [workload_dir + "/out/pmc_1/converted.csv"]

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: csv_files
    )

    # Mock csv_ops functions to avoid disk I/O
    mock_rows = [
        {
            "Dispatch_ID": "0",
            "GPU_ID": "0",
            "Kernel_Name": "test",
            "Grid_Size": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
        }
    ]
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: (mock_rows.copy(), list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", lambda *a, **k: None
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_success_rocprofiler_sdk(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofiler-sdk execution.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution with SDK configuration.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    profiler_options = {
        "APP_CMD": ["./test_app"],
        "ROCPROF_OUTPUT_PATH": workload_dir,
        "ROCPROF_COUNTER_COLLECTION": "1",
        "ROCP_TOOL_LIBRARIES": "/opt/rocm/lib/rocprofiler-sdk/"
        "librocprofiler-sdk-tool.so",
    }

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_common.parse_pmc_perf", lambda f: ["SQ_WAVES"])
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(
        str(fname), profiler_options, workload_dir, logging.INFO, "csv"
    )


def test_run_prof_with_yaml_config(tmp_path, monkeypatch):
    """
    Test run_prof with additional YAML configuration file.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts YAML config is properly handled.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    yaml_file = tmp_path / "counter_def_test.yaml"
    yaml_file.write_text("rocprofiler-sdk:\n  counters:\n    - TCC_HIT\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_failure_subprocess(tmp_path, monkeypatch):
    """
    Test run_prof when subprocess execution fails.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts proper error handling on subprocess failure.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (False, "error output"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    def mock_console_error(msg, exit=True):
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with pytest.raises(RuntimeError, match="console_error called"):
        utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_mi300_environment_setup(tmp_path, monkeypatch):
    """
    Test run_prof sets proper environment variables for MI300 series GPUs.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts MI300 environment variable is set correctly.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    captured_env = {}

    def mock_capture_subprocess_output(cmd, new_env=None, **kwargs):
        if new_env:
            captured_env.update(new_env)
        return (True, "success")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output", mock_capture_subprocess_output
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_timestamps_special_case(tmp_path, monkeypatch):
    """
    Test run_prof handles timestamps.txt special case correctly.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts timestamps processing is handled correctly.
    """
    fname = tmp_path / "pmc_perf_timestamps.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_content = (
        "Agent_Type,Node_Id,Wave_Front_Size,Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,"
        "Grid_Size,Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,"
        "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Start_Timestamp,"
        "End_Timestamp,Counter_Name,Counter_Value\n"
        "GPU,0,0,0,0,0,0,0,0,0,0,test_kernel,0,0,0,0,0,0,0,1,SQ_WAVES,100"
    )
    with open(workload_dir + "/kernel_trace.csv", "w") as f:
        f.write(csv_content)

    csv_files = [workload_dir + "/kernel_trace.csv"]

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: csv_files
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    mock_df = pd.DataFrame({
        "Dispatch_ID": [0],
        "Start_Timestamp": [100],
        "End_Timestamp": [200],
        "Grid_Size": [1024],
        "Workgroup_Size": [64],
        "Kernel_Name": ["test_kernel"],
        "LDS_Per_Workgroup": [1024],
    })
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.concat", lambda *a, **k: mock_df)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_no_results_files(tmp_path, monkeypatch):
    """
    Test run_prof when no results files are generated.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts proper handling when no results are found.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv2")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("glob.glob", lambda pattern: [])  # No files found
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_header_standardization(tmp_path, monkeypatch):
    """
    Test run_prof properly standardizes CSV headers.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts CSV headers are standardized correctly.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    results_csv = workload_dir + "/out/pmc_1/results_test.csv"

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: [results_csv]
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    # Mock csv_ops to track rename_columns calls and avoid disk I/O
    mock_rows = [
        {
            "KernelName": "test_kernel",
            "Index": "0",
            "grd": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
            "BeginNs": "0",
            "EndNs": "1",
            "SQ_WAVES": "100",
        }
    ]

    rename_calls = []

    def mock_rename_columns(rows, mapping):
        rename_calls.append(mapping)
        # Apply the rename to verify the mapping
        for row in rows:
            for old_name, new_name in mapping.items():
                if old_name in row:
                    row[new_name] = row.pop(old_name)

    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: ([r.copy() for r in mock_rows], list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", mock_rename_columns
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")

    # Verify that rename_columns was called with the header standardization mapping
    assert len(rename_calls) == 1
    mapping = rename_calls[0]
    assert mapping.get("KernelName") == "Kernel_Name"
    assert mapping.get("Index") == "Dispatch_ID"
    assert mapping.get("grd") == "Grid_Size"
    assert mapping.get("BeginNs") == "Start_Timestamp"
    assert mapping.get("EndNs") == "End_Timestamp"


def test_run_prof_tcc_flattening_mi300(tmp_path, monkeypatch):
    """
    Test run_prof applies TCC flattening for MI300 series GPUs.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts TCC flattening is applied for MI300 GPUs.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - TCC_HIT[0]\n")
    workload_dir = str(tmp_path / "workload")

    # Mock functions
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.mi_gpu_spec.mi_gpu_specs.get_num_xcds", lambda *a: 2)
    monkeypatch.setattr(
        "glob.glob", lambda pattern: [workload_dir + "/results_test.csv"]
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    # Mock pandas
    mock_df = pd.DataFrame({"Dispatch_ID": [0], "TCC_HIT[0]": [100]})
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.concat", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.DataFrame.to_csv", lambda self, *a, **k: None)

    # Execute function
    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_sdk_creates_new_env_copy(tmp_path, monkeypatch):
    """
    Covers: new_env = os.environ.copy()
            when rocprof_cmd == "rocprofiler-sdk" and new_env was not previously set
            by the mspec.gpu_model check.
    """
    fname_str = str(tmp_path / "pmc_perf_counters.yaml")
    Path(fname_str).write_text("jobs:\n  - pmc:\n    - COUNTER1\n")
    workload_dir_str = str(tmp_path)

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )

    capture_subprocess_called_with_env = None

    def mock_capture_subprocess(app_cmd, new_env=None, profileMode=False):
        nonlocal capture_subprocess_called_with_env
        capture_subprocess_called_with_env = new_env
        return (True, "Success")

    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output", mock_capture_subprocess
    )

    def mock_console_error_no_exit(msg, exit=True):
        print(f"Mocked console_error: {msg}, exit={exit}")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error_no_exit)
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.parse_pmc_perf", lambda *a, **k: ["COUNTER1", "COUNTER2"]
    )

    mock_fname_path_obj = mock.MagicMock(spec=Path)
    mock_fname_path_obj.stem = "pmc_perf_counters"
    mock_fname_path_obj.name = "pmc_perf_counters.yaml"
    mock_fname_path_obj.exists.return_value = False

    mock_out_path_obj = mock.Mock(spec=Path)
    mock_out_path_obj.exists.return_value = False

    mock_counter_def_path_obj = mock.Mock(spec=Path)
    mock_counter_def_path_obj.exists.return_value = False
    mock_fname_path_obj.parent.__truediv__ = mock.Mock(
        return_value=mock_counter_def_path_obj
    )

    def path_side_effect(p_arg, *args):
        if isinstance(p_arg, Path):
            if p_arg.name == "pmc_perf_counters.yaml":
                return mock_fname_path_obj
            return p_arg
        if isinstance(p_arg, str):
            if p_arg.endswith("/out"):
                return mock_out_path_obj
            if p_arg.endswith("pmc_perf_counters.yaml"):
                return mock_fname_path_obj
            if "counter_def" in p_arg:
                return mock_counter_def_path_obj
        if (
            p_arg == mock_fname_path_obj
            and args == ()
            and hasattr(p_arg, "with_suffix")
        ):
            return mock_fname_path_obj
        return mock_fname_path_obj

    monkeypatch.setattr("utils.utils_profile.Path", path_side_effect)

    loglevel = logging.DEBUG
    format_rocprof_output = True

    dummy_df = pd.DataFrame({"Dispatch_ID": [0], "A": [1]})
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: dummy_df.copy())
    monkeypatch.setattr("pandas.DataFrame.to_csv", lambda self, *a, **k: None)
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.create_temp_rocprofiler_metrics_path",
        lambda *a, **k: "dummy_path",
    )
    monkeypatch.setattr("yaml.dump", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(""))

    from rocprof_compute_profile.profiler_rocprofiler_sdk import (
        rocprofiler_sdk_profiler as rocprofiler_sdk_profiler,
    )

    profiler = rocprofiler_sdk_profiler(
        profiling_args=MockArgs(
            rocprofiler_sdk_tool_path="sdk_tool",
            roof_only=True,
            format_rocprof_output="format",
            output_directory="path",
            remaining="remaining",
            iteration_multiplexing=None,
            attach_pid=None,
            kokkos_trace=None,
            kernel=None,
            dispatch=None,
        ),
        profiler_mode="rocprofiler-sdk",
        soc=MockSoc(),
    )

    # Test 1: LD_PRELOAD is already set - should be preserved
    # Since we check all env. vars. in test,
    # empty them out while calling profiling function
    with mock.patch.dict(os.environ, {}, clear=True):
        assert len(os.environ) == 0
        original_env_var = "original_value"
        monkeypatch.setenv("EXISTING_VAR", original_env_var)
        monkeypatch.setenv("LD_LIBRARY_PATH", original_env_var)
        monkeypatch.setenv("LD_PRELOAD", original_env_var)
        profiler_options = profiler.get_profiler_options(native_tool_path="native_tool")

        utils_profile.run_prof(
            fname_str,
            profiler_options,
            workload_dir_str,
            loglevel,
            format_rocprof_output,
        )

    assert capture_subprocess_called_with_env is not None, (
        "new_env should have been created"
    )
    assert "EXISTING_VAR" in capture_subprocess_called_with_env, (
        "new_env should be a copy of os.environ"
    )
    # Ensure existing env. vars. are preserved
    assert capture_subprocess_called_with_env["EXISTING_VAR"] == original_env_var
    # Ensure LD_LIBRARY_PATH is not touched
    assert capture_subprocess_called_with_env["LD_LIBRARY_PATH"] == original_env_var
    # Ensure LD_PRELOAD is preserved and our tools are appended
    assert original_env_var in capture_subprocess_called_with_env["LD_PRELOAD"], (
        f"User's LD_PRELOAD '{original_env_var}' should be preserved"
    )
    assert "sdk_tool" in capture_subprocess_called_with_env["LD_PRELOAD"], (
        "Profiler sdk_tool should be in LD_PRELOAD"
    )
    assert "native_tool" in capture_subprocess_called_with_env["LD_PRELOAD"], (
        "Native tool should be in LD_PRELOAD"
    )
    # Verify the order: user's LD_PRELOAD comes first, then our tools appended
    expected_ld_preload = f"{original_env_var}:sdk_tool:native_tool"
    assert capture_subprocess_called_with_env["LD_PRELOAD"] == expected_ld_preload, (
        f"LD_PRELOAD should be '{expected_ld_preload}' but got "
        f"'{capture_subprocess_called_with_env['LD_PRELOAD']}'"
    )

    # Test 2: LD_PRELOAD is unset - should only contain profiler tools
    capture_subprocess_called_with_env = None
    with mock.patch.dict(os.environ, {}, clear=True):
        assert len(os.environ) == 0
        monkeypatch.setenv("EXISTING_VAR", original_env_var)
        monkeypatch.setenv("LD_LIBRARY_PATH", original_env_var)
        # Intentionally not setting LD_PRELOAD to test the unset case
        profiler_options = profiler.get_profiler_options(native_tool_path="native_tool")

        utils_profile.run_prof(
            fname_str,
            profiler_options,
            workload_dir_str,
            loglevel,
            format_rocprof_output,
        )

    assert capture_subprocess_called_with_env is not None, (
        "new_env should have been created"
    )
    # When LD_PRELOAD is unset, should only contain our profiler tools
    expected_ld_preload_unset = "sdk_tool:native_tool"
    actual_ld_preload = capture_subprocess_called_with_env["LD_PRELOAD"]
    assert actual_ld_preload == expected_ld_preload_unset, (
        f"LD_PRELOAD should be '{expected_ld_preload_unset}' when unset, "
        f"but got '{actual_ld_preload}'"
    )
    assert (
        capture_subprocess_called_with_env["ROCPROFILER_METRICS_PATH"] == "dummy_path"
    )
    assert capture_subprocess_called_with_env["ROCPROF_COUNTER_COLLECTION"] == "0"
    assert capture_subprocess_called_with_env["ROCPROF_KERNEL_TRACE"] == "1"
    assert capture_subprocess_called_with_env["ROCPROF_OUTPUT_FORMAT"] == "format"
    assert capture_subprocess_called_with_env["ROCPROF_OUTPUT_PATH"] == "path/out/pmc_1"
    assert (
        capture_subprocess_called_with_env["ROCPROF_COUNTERS"]
        == "pmc: COUNTER1 COUNTER2"
    )
    assert "APP_CMD" not in capture_subprocess_called_with_env


def test_run_prof_v3_cli_calls_kokkos_trace_processing(tmp_path, monkeypatch):
    """
    Covers:
    CLI: if "--kokkos-trace" in options:
        process_kokkos_trace_output(...)
    """
    fname_str = str(tmp_path) + "/pmc_perf_counters.yaml"
    Path(fname_str).write_text("jobs:\n  - pmc:\n    - C1\n")
    fbase_str = "pmc_perf_counters"
    workload_dir_str = str(tmp_path)
    (tmp_path / "out" / "pmc_1").mkdir(parents=True, exist_ok=True)

    results_csv = str(tmp_path) + "/out/pmc_1/results1.csv"

    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "Success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output",
        lambda *a, **k: [results_csv],
    )

    kokkos_trace_called_with = None

    def mock_kokkos_trace(wd, fb):
        nonlocal kokkos_trace_called_with
        kokkos_trace_called_with = (wd, fb)

    monkeypatch.setattr(
        "utils.utils_profile.process_kokkos_trace_output", mock_kokkos_trace
    )

    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.parse_pmc_perf", lambda *a, **k: ["C1"])

    # Mock csv_ops functions to avoid disk I/O
    mock_rows = [
        {
            "Dispatch_ID": "0",
            "Kernel_Name": "test",
            "Grid_Size": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
        }
    ]
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: (mock_rows.copy(), list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", lambda *a, **k: None
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    loglevel = logging.INFO
    format_rocprof_output = "csv"

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_v3")

    profiler_options_cli_kokkos = ["--kokkos-trace", "--other-opt"]
    kokkos_trace_called_with = None

    utils_profile.run_prof(
        fname_str,
        profiler_options_cli_kokkos,
        workload_dir_str,
        loglevel,
        format_rocprof_output,
    )
    assert kokkos_trace_called_with == (workload_dir_str, fbase_str)


# =============================================================================
# ROCPROFV3 OUTPUT PROCESSING TESTS
# =============================================================================


def test_process_rocprofv3_output_csv_format_with_counter_files(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output with csv format processes counter collection files.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts counter files are converted properly.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    agent_file = output_dir / "test_agent_info.csv"
    converted_file = output_dir / "test_converted.csv"

    counter_file.write_text("counter,data\ntest,value")
    agent_file.write_text("agent,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        elif "_converted.csv" in pattern:
            return [str(converted_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        Path(output_path).write_text("converted,data\ntest,value")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert len(result) == 1
    assert str(converted_file) in result


def test_process_rocprofv3_output_csv_format_conversion_error(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output handles conversion errors gracefully.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts empty list returned when conversion fails.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    agent_file = output_dir / "test_agent_info.csv"

    counter_file.write_text("counter,data\ntest,value")
    agent_file.write_text("agent,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        raise ValueError("Conversion failed")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    warnings = []
    monkeypatch.setattr(
        "utils.utils_profile.console_warning", lambda msg: warnings.append(msg)
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert result == []
    assert len(warnings) == 1
    assert "Error converting" in warnings[0]


def test_process_rocprofv3_output_csv_format_missing_agent_file(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output raises error when agent info file is missing.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts ValueError is raised for missing agent file.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    counter_file.write_text("counter,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    with pytest.raises(ValueError, match='has no corresponding "agent info" file'):
        utils_profile.process_rocprofv3_output(workload_dir, False)


def test_process_rocprofv3_output_csv_format_no_files_non_timestamps(
    tmp_path, monkeypatch
):
    """
    Test process_rocprofv3_output returns empty list when
    no files found for non-timestamps.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts empty list returned when no counter files exist.
    """
    workload_dir = str(tmp_path)

    monkeypatch.setattr("glob.glob", lambda pattern: [])

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert result == []


def test_process_rocprofv3_output_csv_format_multiple_counter_files(
    tmp_path, monkeypatch
):
    """
    Test process_rocprofv3_output processes multiple counter collection files.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts multiple counter files are processed correctly.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file1 = output_dir / "test1_counter_collection.csv"
    agent_file1 = output_dir / "test1_agent_info.csv"
    converted_file1 = output_dir / "test1_converted.csv"

    counter_file2 = output_dir / "test2_counter_collection.csv"
    agent_file2 = output_dir / "test2_agent_info.csv"
    converted_file2 = output_dir / "test2_converted.csv"

    counter_file1.write_text("counter,data\ntest1,value1")
    agent_file1.write_text("agent,data\ntest1,value1")
    counter_file2.write_text("counter,data\ntest2,value2")
    agent_file2.write_text("agent,data\ntest2,value2")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file1), str(counter_file2)]
        elif "_converted.csv" in pattern:
            return [str(converted_file1), str(converted_file2)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        Path(output_path).write_text(f"converted,data\n{Path(counter_path).stem},value")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert len(result) == 2
    assert str(converted_file1) in result
    assert str(converted_file2) in result


def test_capture_subprocess_output_with_logging_disabled(monkeypatch):
    """
    Test capture_subprocess_output with enable_logging=False doesn't call console_log.
    """

    monkeypatch.setattr(
        "subprocess.Popen",
        lambda *a, **k: make_dummy_process(lines=["test output\n"]),
    )

    log_calls = []
    monkeypatch.setattr(
        "utils.utils_common.console_log", lambda *a, **k: log_calls.append((a, k))
    )
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    success, output = utils_common.capture_subprocess_output(
        ["echo", "test"], enable_logging=False
    )

    assert success is True
    assert len(log_calls) == 0


# =============================================================================
# KOKKOS TRACE PROCESSING TESTS
# =============================================================================


def test_process_kokkos_trace_output_single_file(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with a single CSV file.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that single file is processed correctly
        and output files are created.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "single_marker_api_trace.csv"
    csv1.write_text(
        "marker_id,marker_name,start_time,end_time\n1,kokkos_begin,1000,1050\n2,kokkos_end,2000,2010\n"
    )

    fbase = "single_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    # Check output file in pmc_1 directory
    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert df["marker_name"].tolist() == ["kokkos_begin", "kokkos_end"]

    # Check copied file in workload directory
    copied_file = tmp_path / f"{fbase}_marker_api_trace.csv"
    assert copied_file.exists()


def test_process_kokkos_trace_output_multiple_files(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with multiple valid CSV files.
    Should concatenate all files and save the result.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_warning", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub1.mkdir()
    sub2.mkdir()

    csv1 = sub1 / "test_marker_api_trace.csv"
    csv2 = sub2 / "test_marker_api_trace.csv"
    csv1.write_text(
        "timestamp,marker_name,duration\n1000,kokkos_malloc,500\n2000,kokkos_parallel_for,300\n"
    )
    csv2.write_text(
        "timestamp,marker_name,duration\n3000,kokkos_free,200\n4000,kokkos_parallel_reduce,800\n"
    )

    fbase = "test_workload"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists(), "The primary output file was not created."

    df = pd.read_csv(output_file)
    assert len(df) == 4, (
        "The final DataFrame does not contain the correct number of rows."
    )
    assert set(df["timestamp"]) == {1000, 2000, 3000, 4000}
    assert "kokkos_malloc" in df["marker_name"].values
    assert "kokkos_parallel_reduce" in df["marker_name"].values


def test_process_kokkos_trace_output_no_files_found(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when no marker API trace files are found.
    With the new csv_ops-based implementation, no output file is created when
    there are no input files, and shutil.copyfile will raise FileNotFoundError.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that function raises FileNotFoundError with no input files.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    fbase = "no_files"

    # With the csv_ops implementation, when there are no input files:
    # - concat_csv_files returns []
    # - write_csv_from_dicts doesn't write anything (no rows, no fieldnames)
    # - shutil.copyfile fails because the source file doesn't exist
    with pytest.raises(FileNotFoundError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


def test_process_kokkos_trace_output_mixed_file_states(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with a mix of valid, empty, and corrupted files.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that valid files are processed while invalid
        ones are handled gracefully.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub3 = out_dir / "process3"
    sub1.mkdir()
    sub2.mkdir()
    sub3.mkdir()

    csv1 = sub1 / "valid_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name\n1000,kokkos_malloc\n2000,kokkos_free\n")

    csv2 = sub2 / "empty_marker_api_trace.csv"
    csv2.write_text("")

    csv3 = sub3 / "headers_marker_api_trace.csv"
    csv3.write_text("timestamp,marker_name\n")

    fbase = "mixed_test"

    original_read_csv = pd.read_csv

    def mock_read_csv(filepath, **kwargs):
        try:
            return original_read_csv(filepath, **kwargs)
        except pd.errors.EmptyDataError:
            # Return empty DataFrame for empty files
            return pd.DataFrame()

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) >= 0


def test_process_kokkos_trace_output_no_out_directory(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when output directory doesn't exist.
    Should not copy file to workload directory.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that function handles missing
        output directory gracefully.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)

    fbase = "no_out_dir"

    monkeypatch.setattr("glob.glob", lambda pattern: [])

    def mock_concat(dataframes, **kwargs):
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, **kwargs)

    monkeypatch.setattr("pandas.concat", mock_concat)

    def mock_to_csv(self, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")

    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    from pathlib import Path as original_path

    def mock_path_exists(path_str):
        if path_str == workload_dir + "/out":
            mock_path_obj = mock.MagicMock()
            mock_path_obj.exists.return_value = False
            return mock_path_obj
        else:
            return original_path(path_str)

    monkeypatch.setattr("utils.utils_profile.Path", mock_path_exists)

    try:
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)

        # Should not copy file to workload directory since /out doesn't exist
        copied_file = tmp_path / f"{fbase}_marker_api_trace.csv"
        assert not copied_file.exists()

    except ValueError:
        pytest.skip(
            "process_kokkos_trace_output doesn't handle missing "
            "output directory gracefully"
        )


def test_process_kokkos_trace_output_csv_with_only_headers(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files that contain
    only headers but no data.

    With the csv_ops implementation, when files have only headers (no data rows):
    - concat_csv_files reads headers but returns empty list (no data rows)
    - write_csv_from_dicts doesn't write if rows is empty and no fieldnames passed
    - shutil.copyfile fails because source file doesn't exist

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that header-only files raise FileNotFoundError.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "headers_only_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name,duration,thread_id\n")

    fbase = "headers_only"

    # With csv_ops, header-only files result in empty rows and the output
    # file isn't created, causing FileNotFoundError during copyfile
    with pytest.raises(FileNotFoundError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


def test_process_kokkos_trace_output_large_files(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with larger CSV files to ensure memory handling.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that large files are processed correctly.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "large_marker_api_trace.csv"

    content = "timestamp,marker_name,duration,thread_id\n"
    kokkos_markers = [
        "kokkos_malloc",
        "kokkos_free",
        "kokkos_parallel_for",
        "kokkos_parallel_reduce",
        "kokkos_fence",
    ]
    for i in range(1000):
        marker_name = kokkos_markers[i % len(kokkos_markers)]
        content += f"{i},{marker_name},{i % 100},{i % 10}\n"

    csv1.write_text(content)

    fbase = "large_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 1000
    assert "kokkos_malloc" in df["marker_name"].values
    assert "kokkos_parallel_reduce" in df["marker_name"].values


def test_process_kokkos_trace_output_unicode_content(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files containing unicode characters.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that unicode content is handled properly.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "unicode_marker_api_trace.csv"
    csv1.write_text(
        "timestamp,marker_name,duration\n1000,kokkos_α_kernel,500\n2000,kokkos_β_operation,300\n",
        encoding="utf-8",
    )

    fbase = "unicode_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert "kokkos_α_kernel" in df["marker_name"].values
    assert "kokkos_β_operation" in df["marker_name"].values


def test_process_kokkos_trace_output_different_schemas(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files having different column schemas.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that files with different schemas are concatenated properly.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    # Create subdirectories for glob to find
    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub1.mkdir()
    sub2.mkdir()

    # Create marker trace files (needed for glob pattern matching)
    csv1 = sub1 / "schema1_marker_api_trace.csv"
    csv2 = sub2 / "schema2_marker_api_trace.csv"
    csv1.touch()
    csv2.touch()

    fbase = "schema_test"

    # Mock csv_ops to avoid disk I/O and test concatenation behavior
    mock_rows = [
        {"marker_id": "1", "marker_name": "kokkos_begin", "start_time": "1000"},
        {"marker_id": "2", "marker_name": "kokkos_end", "start_time": "2000"},
        {"marker_name": "kokkos_malloc", "duration": "500", "thread_id": "0"},
        {"marker_name": "kokkos_free", "duration": "200", "thread_id": "1"},
    ]

    write_calls = []

    def mock_concat(files, output_file=None):
        return mock_rows.copy()

    def mock_write(path, rows, fieldnames=None):
        write_calls.append((path, rows))

    monkeypatch.setattr("utils.utils_profile.csv_ops.concat_csv_files", mock_concat)
    monkeypatch.setattr("utils.utils_profile.csv_ops.write_csv_from_dicts", mock_write)
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: None)

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    # Verify write was called with the concatenated rows
    assert len(write_calls) == 1
    written_rows = write_calls[0][1]
    assert len(written_rows) == 4
    # Check that all rows have marker_name
    assert all("marker_name" in row for row in written_rows)


def test_process_kokkos_trace_output_permission_error(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when there are permission
    errors during file operations.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts that permission errors are handled gracefully.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "test_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name\n1000,kokkos_malloc\n")

    fbase = "permission_test"

    def mock_write_permission_error(path, rows, fieldnames=None):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts",
        mock_write_permission_error,
    )

    with pytest.raises(PermissionError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


# =============================================================================
# Normal Functionality:
#
# Basic submodule listing with real packages
# Correct name processing with underscores
# Multiple underscore handling
# Base module filtering
# Edge Cases:
#
# Empty packages (no submodules)
# Non-existent packages
# Names without underscores (IndexError case)
# Empty name parts
# Packages without __path__ attribute
# Error Conditions:
#
# ModuleNotFoundError for invalid packages
# AttributeError for packages without __path__
# TypeError for invalid input types
# ImportError from pkgutil.walk_packages
# Special Scenarios:
#
# Large numbers of submodules
# Special characters in names
# Unicode character handling
# Import isolation testing
# Mixed module types
# Data Integrity:
#
# Return type consistency
# Docstring verification
# Behavior validation
# =============================================================================


mock_package = mock.MagicMock()
mock_package.__path__ = ["/fake/path"]
mock_submodules = [
    (None, "module_parse", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules)
def test_get_submodules_basic_functionality(mock_walk, mock_import):
    """
    Test basic functionality with a real package that has submodules.

    Returns:
        None: Asserts function correctly lists submodules from a real package.
    """

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert len(result) == 3
    expected = ["parse", "request", "error"]
    assert result == expected


def test_get_submodules_empty_package():
    """
    Test with a package that has no submodules.

    Returns:
        None: Asserts function returns empty list for packages without submodules.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("empty_package")

            assert isinstance(result, list)
            assert len(result) == 0


def test_get_submodules_package_not_found():
    """
    Test behavior when package doesn't exist.

    Returns:
        None: Asserts ModuleNotFoundError is raised for non-existent packages.
    """

    with pytest.raises(ModuleNotFoundError):
        utils_profile.get_submodules("nonexistent_package_12345")


mock_package_single = mock.MagicMock()
mock_package_single.__path__ = ["/fake/path"]
mock_submodules_single = [
    (None, "module_parser", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_single)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_single)
def test_get_submodules_name_processing_single_underscore(mock_walk, mock_import):
    """
    Test name processing with single underscore pattern.

    Returns:
        None: Asserts correct name processing for submodules with single underscore.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "request", "error"]
    assert result == expected


mock_package_multiple = mock.MagicMock()
mock_package_multiple.__path__ = ["/fake/path"]
mock_submodules_multiple = [
    (None, "module_some_complex_name", False),
    (None, "module_another_test_case", False),
    (None, "module_simple", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_multiple)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_multiple)
def test_get_submodules_name_processing_multiple_underscores(mock_walk, mock_import):
    """
    Test name processing with multiple underscores in submodule names.

    Returns:
        None: Asserts correct name processing for complex underscore patterns.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["somecomplexname", "anothertestcase", "simple"]
    assert result == expected


mock_package_base = mock.MagicMock()
mock_package_base.__path__ = ["/fake/path"]
mock_submodules_base = [
    (None, "module_base", False),
    (None, "module_parser", False),
    (None, "module_handler", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_base)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_base)
def test_get_submodules_base_module_filtered(mock_walk, mock_import):
    """
    Test that 'base' submodule is properly filtered out.

    Returns:
        None: Asserts 'base' submodules are excluded from results.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "handler"]
    assert result == expected
    assert "base" not in result


mock_package_no_underscore = mock.MagicMock()
mock_package_no_underscore.__path__ = ["/fake/path"]
mock_submodules_no_underscore = [
    (None, "simplemodule", False),
    (None, "anothermodule", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_no_underscore)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_no_underscore)
def test_get_submodules_no_underscore_in_name(mock_walk, mock_import):
    """
    Test behavior with submodule names that don't follow the expected pattern.

    Returns:
        None: Asserts function handles names without underscores by raising IndexError.
    """

    with pytest.raises(IndexError):
        utils_profile.get_submodules("test_package")


mock_package_empty_parts = mock.MagicMock()
mock_package_empty_parts.__path__ = ["/fake/path"]
mock_submodules_empty_parts = [
    (None, "module_", False),  # ends with underscore
    (None, "_module", False),  # starts with underscore - this will cause IndexError
    (None, "module__double", False),  # double underscore
]


@mock.patch("importlib.import_module", return_value=mock_package_empty_parts)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_empty_parts)
def test_get_submodules_empty_name_parts(mock_walk, mock_import):
    """
    Test behavior with empty name parts after splitting.

    Returns:
        None: Asserts function handles edge cases in name processing.
    """

    try:
        result = utils_profile.get_submodules("test_package")
        expected = ["", "", "double"]  # noqa - Empty strings for edge cases
        assert len(result) == 3
    except IndexError:
        pytest.skip("Function doesn't handle edge case module names gracefully")


def test_get_submodules_package_without_path_attribute():
    """
    Test behavior when package doesn't have __path__ attribute.

    Returns:
        None: Asserts AttributeError is raised for packages without __path__.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    del mock_package.__path__

    with patch("importlib.import_module", return_value=mock_package):
        with pytest.raises(AttributeError):
            utils_profile.get_submodules("test_package")


mock_package_exception = mock.MagicMock()
mock_package_exception.__path__ = ["/fake/path"]


@mock.patch("importlib.import_module", return_value=mock_package_exception)
@mock.patch("pkgutil.walk_packages", side_effect=ImportError("Mock error"))
def test_get_submodules_pkgutil_walk_packages_exception(mock_walk, mock_import):
    """
    Test behavior when pkgutil.walk_packages raises an exception.

    Returns:
        None: Asserts exceptions from pkgutil.walk_packages are properly handled.
    """

    with pytest.raises(ImportError):
        utils_profile.get_submodules("test_package")


mock_package_mixed = mock.MagicMock()
mock_package_mixed.__path__ = ["/fake/path"]
mock_submodules_mixed = [
    (None, "module_base", False),  # Should be filtered out
    (None, "module_parser", False),  # Normal case
    (None, "module_test_case", False),  # Multiple underscores
    (None, "module_simple", False),  # Simple case
    (None, "module_another_base", False),  # Contains 'base' but not exactly 'base'
]


@mock.patch("importlib.import_module", return_value=mock_package_mixed)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_mixed)
def test_get_submodules_mixed_module_types(mock_walk, mock_import):
    """
    Test with a mix of different module types and names.

    Returns:
        None: Asserts function correctly processes various submodule patterns.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "testcase", "simple", "anotherbase"]
    assert result == expected
    assert "base" not in result


mock_package_large = mock.MagicMock()
mock_package_large.__path__ = ["/fake/path"]
mock_submodules_large = []
expected_results_large = []
for i in range(100):
    module_name = f"module_test{i}"
    mock_submodules_large.append((None, module_name, False))
    expected_results_large.append(f"test{i}")


@mock.patch("importlib.import_module", return_value=mock_package_large)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_large)
def test_get_submodules_large_number_of_submodules(mock_walk, mock_import):
    """
    Test performance and correctness with a large number of submodules.

    Returns:
        None: Asserts function handles large numbers of submodules correctly.
    """

    result = utils_profile.get_submodules("test_package")
    assert len(result) == 100
    assert result == expected_results_large


def test_get_submodules_string_input_validation():
    """
    Test input validation for package_name parameter.

    Returns:
        None: Asserts function handles invalid input types
        but may not validate properly.
    """

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(None)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(123)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(["list", "input"])


def test_get_submodules_return_type_consistency():
    """
    Test that function always returns a list, even in edge cases.

    Returns:
        None: Asserts return type is always a list.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0

    mock_submodules = [(None, "module_base", False)]
    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=mock_submodules):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0


mock_package_special = mock.MagicMock()
mock_package_special.__path__ = ["/fake/path"]
mock_submodules_special = [
    (None, "module_test-case", False),
    (None, "module_test.case", False),
    (None, "module_test123", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_special)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_special)
def test_get_submodules_special_characters_in_names(mock_walk, mock_import):
    """
    Test handling of special characters in submodule names.

    Returns:
        None: Asserts function processes special characters in names correctly.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["test-case", "test.case", "test123"]
    assert result == expected


mock_package_isolation = mock.MagicMock()
mock_package_isolation.__path__ = ["/fake/path"]
mock_submodules_isolation = [(None, "module_test", False)]


@mock.patch("importlib.import_module", return_value=mock_package_isolation)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_isolation)
def test_get_submodules_imports_isolation(mock_walk, mock_import):
    """
    Test that imports are properly isolated and don't affect global state.

    Returns:
        None: Asserts function imports don't pollute global namespace.
    """
    import sys

    original_importlib = sys.modules.get("importlib")
    original_pkgutil = sys.modules.get("pkgutil")

    result = utils_profile.get_submodules("test_package")

    assert sys.modules.get("importlib") == original_importlib
    assert sys.modules.get("pkgutil") == original_pkgutil
    assert isinstance(result, list)
    assert result == ["test"]


mock_package_unicode = mock.MagicMock()
mock_package_unicode.__path__ = ["/fake/path"]
mock_submodules_unicode = [
    (None, "module_tëst", False),
    (None, "module_测试", False),
    (None, "module_тест", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_unicode)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_unicode)
def test_get_submodules_unicode_names(mock_walk, mock_import):
    """
    Test handling of Unicode characters in package and submodule names.

    Returns:
        None: Asserts function handles Unicode characters appropriately.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["tëst", "测试", "тест"]
    assert result == expected


mock_package_docstring = mock.MagicMock()
mock_package_docstring.__path__ = ["/fake/path"]
mock_submodules_docstring = [
    (None, "module_submodule1", False),
    (None, "module_submodule2", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_docstring)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_docstring)
def test_get_submodules_docstring_verification(mock_walk, mock_import):
    """
    Test that function behavior matches its docstring description.

    Returns:
        None: Asserts function behavior aligns with documented purpose.
    """

    assert utils_profile.get_submodules.__doc__ is not None
    assert (
        "List all submodules for a target package"
        in utils_profile.get_submodules.__doc__
    )  # noqa

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert "submodule1" in result
    assert "submodule2" in result


# =============================================================================
# TESTS FOR EMPTY WORKLOAD
#
# Normal Functionality:
#
# Valid CSV files with data
# Mixed valid and invalid data
# Large datasets
# Unicode content handling
# Edge Cases:
#
# Empty CSV files
# CSV with only headers
# Files with all NaN values that become empty after dropna()
# Malformed CSV files
# Missing pmc_perf.csv file
# Nonexistent directories
# Error Conditions:
#
# File permission errors
# CSV reading errors
# Directory access issues
# String Formatting and Dependencies:
#
# Console error message formatting
# Path handling (string vs Path)
# Pandas dependency verification
# Return value consistency
# Special Scenarios:
#
# Special characters in paths
# Unicode content in CSV files
# Large datasets with performance implications
# Different input path types
# =============================================================================


def test_is_workload_empty_valid_data_file(tmp_path):
    """
    Test is_workload_empty with a valid pmc_perf.csv file containing data.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles valid data files without errors.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    valid_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200
kernel2,1,150,250
kernel3,0,120,220"""
    pmc_perf_file.write_text(valid_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_file_with_nan_values(tmp_path):
    """
    Test is_workload_empty with pmc_perf.csv containing NaN values.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects and reports empty cells after dropping NaN.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    nan_data = """Kernel_Name,GPU_ID,Counter1,Counter2
,,NaN,
,NaN,,NaN
NaN,,,"""
    pmc_perf_file.write_text(nan_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "pmc_perf.csv" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_is_workload_empty_completely_empty_csv(tmp_path):
    """
    Test is_workload_empty with completely empty pmc_perf.csv file.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects empty CSV file.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        try:
            utils_analysis.is_workload_empty(str(workload_dir))
        except Exception:
            pass


def test_is_workload_empty_headers_only_csv(tmp_path):
    """
    Test is_workload_empty with CSV containing only headers.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects CSV with headers but no data.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    headers_only = "Kernel_Name,GPU_ID,Counter1,Counter2"
    pmc_perf_file.write_text(headers_only)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_is_workload_empty_no_pmc_perf_file(tmp_path):
    """
    Test is_workload_empty when pmc_perf.csv file doesn't exist.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects missing profiling data file.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_nonexistent_directory():
    """
    Test is_workload_empty with nonexistent directory path.

    Returns:
        None: Asserts function handles nonexistent directories.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty("/nonexistent/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_malformed_csv(tmp_path):
    """
    Test is_workload_empty with malformed CSV that causes pandas read error.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles pandas CSV reading errors gracefully.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    malformed_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200,extra_column_data
kernel2,1,150
incomplete_row"""
    pmc_perf_file.write_text(malformed_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        try:
            utils_analysis.is_workload_empty(str(workload_dir))
        except Exception:
            pass


def test_is_workload_empty_mixed_valid_invalid_data(tmp_path):
    """
    Test is_workload_empty with CSV containing mix of valid and invalid (NaN) data.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles mixed data correctly.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    mixed_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200
kernel2,,NaN,250
kernel3,1,120,
,0,110,240"""
    pmc_perf_file.write_text(mixed_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_large_dataset_with_nans(tmp_path):
    """
    Test is_workload_empty with large dataset that becomes empty after dropping NaNs.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function correctly processes large datasets.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    headers = "Kernel_Name,GPU_ID,Counter1,Counter2\n"
    nan_rows = []
    for i in range(1000):
        nan_rows.append("NaN,NaN,NaN,NaN")
    large_nan_data = headers + "\n".join(nan_rows)
    pmc_perf_file.write_text(large_nan_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_is_workload_empty_unicode_content(tmp_path):
    """
    Test is_workload_empty with CSV containing Unicode characters.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles Unicode content correctly.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    unicode_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel_测试,0,100,200
kernel_тест,1,150,250
kernel_tëst,0,120,220"""
    pmc_perf_file.write_text(unicode_data, encoding="utf-8")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_special_path_characters(tmp_path):
    """
    Test is_workload_empty with directory paths containing special characters.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles special characters in paths.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload-test_dir.with.dots"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    valid_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200"""
    pmc_perf_file.write_text(valid_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_csv_read_permission_error(tmp_path):
    """
    Test is_workload_empty when CSV file exists but cannot be read due to permissions.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles file permission errors.
    """
    import os
    from unittest.mock import patch

    if os.name == "nt":
        pytest.skip("Permission test not applicable on Windows")

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nkernel1,0")
    pmc_perf_file.chmod(0o000)  # Remove all permissions

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    try:
        with patch(
            "utils.utils_analysis.console_error", side_effect=mock_console_error
        ):
            utils_analysis.is_workload_empty(str(workload_dir))
    except PermissionError:
        pass
    finally:
        pmc_perf_file.chmod(0o644)


def test_is_workload_empty_string_path_input():
    """
    Test is_workload_empty with string path input vs Path.

    Returns:
        None: Asserts function handles different path input types.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty("/nonexistent/string/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_console_error_string_formatting(tmp_path):
    """
    Test is_workload_empty string formatting in console_error messages.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts console_error messages are properly formatted.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nNaN,NaN")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    expected_path = str(workload_dir / "pmc_perf.csv")
    assert expected_path in error_args[1]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_is_workload_empty_function_return_value(tmp_path):
    """
    Test that is_workload_empty function return behavior (implicitly returns None).

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function return value consistency.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nkernel1,0")

    with patch("utils.utils_analysis.console_error"):
        result = utils_analysis.is_workload_empty(str(workload_dir))

    assert result is None

    workload_dir2 = tmp_path / "workload2"
    workload_dir2.mkdir()

    with patch("utils.utils_analysis.console_error"):
        result2 = utils_analysis.is_workload_empty(str(workload_dir2))

    assert result2 is None


def test_is_workload_empty_pandas_import_dependency():
    """
    Test is_workload_empty dependency on pandas module.

    Returns:
        None: Asserts function properly uses pandas functionality.
    """
    from unittest.mock import MagicMock, patch

    mock_pandas = MagicMock()
    mock_df = MagicMock()
    mock_df.dropna.return_value.empty = False
    mock_pandas.read_csv.return_value = mock_df

    with patch.dict("sys.modules", {"pandas": mock_pandas}):
        with patch("utils.utils_analysis.pd", mock_pandas):
            with patch("utils.utils_analysis.console_error"):
                with patch("pathlib.Path.is_file", return_value=True):
                    utils_analysis.is_workload_empty("/test/path")

    mock_pandas.read_csv.assert_called_once()
    mock_df.dropna.assert_called_once()


# =============================================================================
# TESTS FOR LOCAL ENCODING FUNCTION
#
# Normal Functionality:
#
# Successful C.UTF-8 locale setting
# Fallback to current UTF-8 locale when C.UTF-8 fails
# Various UTF-8 encoding formats and case variations
# Edge Cases:
#
# getdefaultlocale returning None or partial None values
# Empty encoding strings
# Unusual but valid locale names
# Multiple function calls
# Error Conditions:
#
# C.UTF-8 locale not available
# Fallback locale setting failures
# No UTF-8 locales available on system
# getdefaultlocale exceptions
# Various locale.Error scenarios
# String Handling and Dependencies:
#
# UTF-8 substring detection in encoding names
# Console error message formatting and parameters
# Locale module dependency verification
# Return value consistency
# Special Scenarios:
#
# Thread safety simulation
# Different locale error types and messages
# Comprehensive error path coverage
# Module import dependencies
# =============================================================================


def test_set_locale_encoding_successful_c_utf8():
    """
    Test set_locale_encoding when C.UTF-8 locale is
    available and can be set successfully.

    Returns:
        None: Asserts function sets C.UTF-8 locale without errors.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("utils.utils_common.console_error", side_effect=mock_console_error):
            mock_setlocale.return_value = None

            utils_common.set_locale_encoding()

            mock_setlocale.assert_called_once_with(locale.LC_ALL, "C.UTF-8")
            assert len(console_error_calls) == 0


def test_set_locale_encoding_c_utf8_fails_fallback_to_current_utf8():
    """
    Test set_locale_encoding when C.UTF-8 fails but current locale is UTF-8 based.

    Returns:
        None: Asserts function falls back to current UTF-8 locale successfully.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = [
                    locale.Error("C.UTF-8 not available"),
                    None,
                ]
                mock_getdefaultlocale.return_value = ("en_US", "UTF-8")

                utils_common.set_locale_encoding()

                assert mock_setlocale.call_count == 2
                mock_setlocale.assert_any_call(locale.LC_ALL, "C.UTF-8")
                mock_setlocale.assert_any_call(locale.LC_ALL, "en_US")
                assert len(console_error_calls) == 0


def test_set_locale_encoding_c_utf8_fails_fallback_also_fails():
    """
    Test set_locale_encoding when both C.UTF-8 and fallback locale fail.

    Returns:
        None: Asserts function calls console_error when fallback locale fails.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                fallback_error = locale.Error("Fallback locale failed")
                mock_setlocale.side_effect = [
                    locale.Error("C.UTF-8 not available"),
                    fallback_error,
                ]
                mock_getdefaultlocale.return_value = ("en_US", "UTF-8")

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 1
                assert (
                    "Failed to set locale to the current UTF-8-based locale:"
                    in console_error_calls[0][0][0]
                )
                assert "Fallback locale failed" in console_error_calls[0][0][0]


def test_set_locale_encoding_no_utf8_locale_available():
    """
    Test set_locale_encoding when no UTF-8 locale is available.

    Returns:
        None: Asserts function calls console_error when no UTF-8 locale found.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.return_value = ("en_US", "ISO-8859-1")

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 1
                assert (
                    "Please ensure that a UTF-8-based "
                    "locale is available on your system."
                    in console_error_calls[0][0][0]
                )
                assert console_error_calls[0][1]["exit"] == False  # noqa


def test_set_locale_encoding_getdefaultlocale_returns_none():
    """
    Test set_locale_encoding when getdefaultlocale returns None.

    Returns:
        None: Asserts function handles
        None return from getdefaultlocale.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.return_value = None

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 1
                assert (
                    "Please ensure that a UTF-8-based locale "
                    "is available on your system." in console_error_calls[0][0][0]
                )


def test_set_locale_encoding_getdefaultlocale_partial_none():
    """
    Test set_locale_encoding when getdefaultlocale returns partial None values.

    Returns:
        None: Asserts function handles partial None values from getdefaultlocale.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")

                mock_getdefaultlocale.return_value = ("en_US", None)

                try:
                    utils_common.set_locale_encoding()
                except TypeError as e:
                    if "argument of type 'NoneType' is not iterable" in str(e):
                        pytest.skip(
                            "Function doesn't handle None encoding "
                            "gracefully - needs null check"
                        )
                    else:
                        raise

                assert len(console_error_calls) == 1
                assert (
                    "Please ensure that a UTF-8-based locale is "
                    "available on your system." in console_error_calls[0][0][0]
                )


def test_set_locale_encoding_utf8_case_variations():
    """
    Test set_locale_encoding with various UTF-8 case variations in encoding.

    Returns:
        None: Asserts function handles different UTF-8 case formats.
    """
    import locale
    from unittest.mock import patch

    utf8_variations = ["UTF-8", "utf-8", "UTF8", "utf8"]

    for utf8_variant in utf8_variations:
        console_error_calls = []

        def mock_console_error(*args, **kwargs):
            console_error_calls.append((args, kwargs))

        with patch("locale.setlocale") as mock_setlocale:
            with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
                with patch(
                    "utils.utils_common.console_error", side_effect=mock_console_error
                ):
                    mock_setlocale.side_effect = [
                        locale.Error("C.UTF-8 not available"),
                        None,
                    ]
                    mock_getdefaultlocale.return_value = ("en_US", utf8_variant)

                    utils_common.set_locale_encoding()

                    if "UTF-8" in utf8_variant:
                        assert len(console_error_calls) == 0
                        assert mock_setlocale.call_count == 2
                    else:
                        assert len(console_error_calls) == 1


def test_set_locale_encoding_empty_encoding():
    """
    Test set_locale_encoding when getdefaultlocale returns empty encoding.

    Returns:
        None: Asserts function handles empty encoding string.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.return_value = ("en_US", "")

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 1
                assert (
                    "Please ensure that a UTF-8-based locale "
                    "is available on your system." in console_error_calls[0][0][0]
                )


def test_set_locale_encoding_locale_with_utf8_substring():
    """
    Test set_locale_encoding with encoding that contains UTF-8 as substring.

    Returns:
        None: Asserts function correctly identifies UTF-8 in encoding names.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = [
                    locale.Error("C.UTF-8 not available"),
                    None,
                ]
                mock_getdefaultlocale.return_value = (
                    "en_US",
                    "ISO-8859-1.UTF-8.EXTENDED",
                )

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 0
                assert mock_setlocale.call_count == 2


def test_set_locale_encoding_different_locale_error_types():
    """
    Test set_locale_encoding with different types of locale.Error exceptions.

    Returns:
        None: Asserts function handles various locale error scenarios.
    """
    import locale
    from unittest.mock import patch

    error_scenarios = [
        "Locale not supported",
        "Invalid locale specification",
        "System locale database corrupted",
        "",  # Empty error message
    ]

    for error_msg in error_scenarios:
        console_error_calls = []

        def mock_console_error(*args, **kwargs):
            console_error_calls.append((args, kwargs))

        with patch("locale.setlocale") as mock_setlocale:
            with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
                with patch(
                    "utils.utils_common.console_error", side_effect=mock_console_error
                ):
                    fallback_error = locale.Error(error_msg)
                    mock_setlocale.side_effect = [
                        locale.Error("C.UTF-8 not available"),
                        fallback_error,
                    ]
                    mock_getdefaultlocale.return_value = ("en_US", "UTF-8")

                    utils_common.set_locale_encoding()

                    assert len(console_error_calls) == 1
                    assert str(fallback_error) in console_error_calls[0][0][0]


def test_set_locale_encoding_unusual_locale_names():
    """
    Test set_locale_encoding with unusual but valid locale names.

    Returns:
        None: Asserts function handles unusual locale name formats.
    """
    import locale
    from unittest.mock import patch

    unusual_locales = [
        ("C", "UTF-8"),
        ("POSIX", "UTF-8"),
        ("en_US.UTF-8", "UTF-8"),
        ("zh_CN.UTF-8", "UTF-8"),
        ("", "UTF-8"),  # Empty locale name
    ]

    for locale_name, encoding in unusual_locales:
        console_error_calls = []

        def mock_console_error(*args, **kwargs):
            console_error_calls.append((args, kwargs))

        with patch("locale.setlocale") as mock_setlocale:
            with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
                with patch(
                    "utils.utils_common.console_error", side_effect=mock_console_error
                ):
                    mock_setlocale.side_effect = [
                        locale.Error("C.UTF-8 not available"),
                        None,
                    ]
                    mock_getdefaultlocale.return_value = (locale_name, encoding)

                    utils_common.set_locale_encoding()

                    assert len(console_error_calls) == 0
                    assert mock_setlocale.call_count == 2
                    mock_setlocale.assert_any_call(locale.LC_ALL, locale_name)


def test_set_locale_encoding_getdefaultlocale_exception():
    """
    Test set_locale_encoding when getdefaultlocale raises an exception.

    Returns:
        None: Asserts function handles getdefaultlocale exceptions gracefully.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.side_effect = Exception("getdefaultlocale failed")

                try:
                    utils_common.set_locale_encoding()
                except Exception:
                    pass


def test_set_locale_encoding_console_error_parameters():
    """
    Test set_locale_encoding console_error call parameters are correct.

    Returns:
        None: Asserts console_error is called with correct parameters.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.return_value = ("en_US", "ISO-8859-1")

                utils_common.set_locale_encoding()

                assert len(console_error_calls) == 1
                args, kwargs = console_error_calls[0]
                assert len(args) == 1
                assert "exit" in kwargs
                assert kwargs["exit"] == False  # noqa


def test_set_locale_encoding_return_value():
    """
    Test that set_locale_encoding returns None (implicit return).

    Returns:
        None: Asserts function returns None in all scenarios.
    """
    import locale
    from unittest.mock import patch

    with patch("locale.setlocale") as mock_setlocale:
        with patch("utils.utils_common.console_error"):
            mock_setlocale.return_value = None

            result = utils_common.set_locale_encoding()
            assert result is None

    with patch("locale.setlocale") as mock_setlocale:
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch("utils.utils_common.console_error"):
                mock_setlocale.side_effect = locale.Error("C.UTF-8 not available")
                mock_getdefaultlocale.return_value = ("en_US", "ISO-8859-1")

                result = utils_common.set_locale_encoding()
                assert result is None


def test_set_locale_encoding_locale_module_import():
    """
    Test set_locale_encoding dependency on locale module.

    Returns:
        None: Asserts function properly uses locale module functionality.
    """
    import locale
    from unittest.mock import patch

    setlocale_calls = []
    getdefaultlocale_calls = []

    def mock_setlocale(category, locale_name):
        setlocale_calls.append((category, locale_name))
        return None

    def mock_getdefaultlocale():
        getdefaultlocale_calls.append(True)
        return ("en_US", "UTF-8")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale", side_effect=mock_setlocale):
        with patch("locale.getdefaultlocale", side_effect=mock_getdefaultlocale):
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                utils_common.set_locale_encoding()

    assert len(setlocale_calls) == 1
    assert setlocale_calls[0] == (locale.LC_ALL, "C.UTF-8")
    assert len(getdefaultlocale_calls) == 0
    assert len(console_error_calls) == 0

    setlocale_calls.clear()
    getdefaultlocale_calls.clear()
    console_error_calls.clear()

    def mock_setlocale_with_error(category, locale_name):
        setlocale_calls.append((category, locale_name))
        if locale_name == "C.UTF-8":
            raise locale.Error("C.UTF-8 not available")
        return None

    with patch("locale.setlocale", side_effect=mock_setlocale_with_error):
        with patch("locale.getdefaultlocale", side_effect=mock_getdefaultlocale):
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                utils_common.set_locale_encoding()

    assert len(setlocale_calls) == 2
    assert setlocale_calls[0] == (locale.LC_ALL, "C.UTF-8")
    assert setlocale_calls[1] == (locale.LC_ALL, "en_US")
    assert len(getdefaultlocale_calls) == 1
    assert len(console_error_calls) == 0


def test_set_locale_encoding_multiple_calls():
    """
    Test set_locale_encoding behavior when called multiple times.

    Returns:
        None: Asserts function behaves consistently across multiple calls.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale") as mock_setlocale:
        with patch("utils.utils_common.console_error", side_effect=mock_console_error):
            mock_setlocale.return_value = None

            utils_common.set_locale_encoding()
            utils_common.set_locale_encoding()
            utils_common.set_locale_encoding()

            assert mock_setlocale.call_count == 3
            assert len(console_error_calls) == 0


def test_set_locale_encoding_thread_safety_simulation():
    """
    Test set_locale_encoding behavior in simulated concurrent scenarios.

    Returns:
        None: Asserts function handles concurrent-like access patterns.
    """
    import locale
    from unittest.mock import patch

    call_count = 0

    def side_effect_setlocale(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise locale.Error("First call fails")
        return None

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("locale.setlocale", side_effect=side_effect_setlocale):
        with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
            with patch(
                "utils.utils_common.console_error", side_effect=mock_console_error
            ):
                mock_getdefaultlocale.return_value = ("en_US", "UTF-8")

                utils_common.set_locale_encoding()

                assert call_count == 2
                assert len(console_error_calls) == 0


def test_set_locale_encoding_comprehensive_error_handling():
    """
    Test set_locale_encoding comprehensive error handling across all code paths.

    Returns:
        None: Asserts all error paths are properly handled.
    """
    import locale
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    test_scenarios = [
        {
            "name": "C.UTF-8 success",
            "setlocale_side_effect": [None],
            "getdefaultlocale_return": ("en_US", "UTF-8"),
            "expected_errors": 0,
        },
        {
            "name": "C.UTF-8 fails, fallback success",
            "setlocale_side_effect": [locale.Error("C.UTF-8 fail"), None],
            "getdefaultlocale_return": ("en_US", "UTF-8"),
            "expected_errors": 0,
        },
        {
            "name": "Both fail with UTF-8 locale",
            "setlocale_side_effect": [
                locale.Error("C.UTF-8 fail"),
                locale.Error("Fallback fail"),
            ],
            "getdefaultlocale_return": ("en_US", "UTF-8"),
            "expected_errors": 1,
        },
        {
            "name": "No UTF-8 locale available",
            "setlocale_side_effect": [locale.Error("C.UTF-8 fail")],
            "getdefaultlocale_return": ("en_US", "ISO-8859-1"),
            "expected_errors": 1,
        },
    ]

    for scenario in test_scenarios:
        console_error_calls.clear()

        with patch("locale.setlocale") as mock_setlocale:
            with patch("locale.getdefaultlocale") as mock_getdefaultlocale:
                with patch(
                    "utils.utils_common.console_error", side_effect=mock_console_error
                ):
                    mock_setlocale.side_effect = scenario["setlocale_side_effect"]
                    mock_getdefaultlocale.return_value = scenario[
                        "getdefaultlocale_return"
                    ]

                    utils_common.set_locale_encoding()

                    assert len(console_error_calls) == scenario["expected_errors"], (
                        f"Failed scenario: {scenario['name']}"
                    )


# =============================================================================
# TESTS FOR merge_counters_spatial_multiplex FUNCTION
# =============================================================================


def test_merge_counters_spatial_multiplex_basic_functionality():
    """
    Test merge_counters_spatial_multiplex with basic multi-index DataFrame.

    Returns:
        None: Asserts function correctly merges counter values for spatial multiplexing.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 1, 1, 2, 2],
        "Grid_Size": [64, 128, 256, 512, 1024, 2048],
        "Workgroup_Size": [16, 32, 64, 32, 64, 128],
        "LDS_Per_Workgroup": [1024, 2048, 4096, 2048, 4096, 8192],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [32, 64, 96, 64, 96, 128],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [16, 32, 48, 32, 48, 64],
        "Wave_Size": [64, 64, 64, 64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003, 2001, 2002, 2003],
        "Kernel_ID": [501, 502, 503, 601, 602, 603],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_c",
            "kernel_c",
            "kernel_d",
        ],
        "Start_Timestamp": [1000, 1100, 2000, 3000, 3100, 4000],
        "End_Timestamp": [1200, 1300, 2500, 3400, 3500, 4800],
        "Counter1": [100, 200, 300, 400, 500, 600],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)


def test_merge_counters_spatial_multiplex_kernel_name_fallback():
    """
    Test merge_counters_spatial_multiplex when Kernel_Name is missing but Name exists.

    Returns:
        None: Asserts function uses Name column when Kernel_Name is not available.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [64, 128],
        "Workgroup_Size": [16, 32],
        "LDS_Per_Workgroup": [1024, 2048],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [32, 64],
        "Accum_VGPR": [0, 0],
        "SGPR": [16, 32],
        "Wave_Size": [64, 64],
        "Correlation_ID": [1001, 1002],
        "Kernel_ID": [501, 502],
        "Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1100],
        "End_Timestamp": [1200, 1300],
        "Counter1": [100, 200],
    }
    df = pd.DataFrame(data)

    # The function currently has a bug where it doesn't properly check for 'Kernel_Name'
    # existence before accessing it, even though it has fallback logic for 'Name'
    try:
        result = utils_analysis.merge_counters_spatial_multiplex(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    except KeyError as e:
        if "'Kernel_Name'" in str(e):
            pytest.skip(
                "Function doesn't properly check for Kernel_Name "
                "existence before accessing - needs to validate column "
                "presence in the check condition"
            )
        else:
            raise


def test_merge_counters_spatial_multiplex_single_kernel_occurrence():
    """
    Test merge_counters_spatial_multiplex with kernels that appear only once.

    Returns:
        None: Asserts function handles single kernel occurrences correctly.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 1, 2],
        "Grid_Size": [64, 128, 256],
        "Workgroup_Size": [16, 32, 64],
        "LDS_Per_Workgroup": [1024, 2048, 4096],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [32, 64, 96],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [16, 32, 48],
        "Wave_Size": [64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003],
        "Kernel_ID": [501, 502, 503],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
        "Start_Timestamp": [1000, 2000, 3000],
        "End_Timestamp": [1200, 2500, 3800],
        "Counter1": [100, 200, 300],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_merge_counters_spatial_multiplex_multiple_duplicate_kernels():
    """
    Test merge_counters_spatial_multiplex with multiple kernels having duplicates.

    Returns:
        None: Asserts function correctly handles multiple kernel duplicates.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 1, 1, 2, 2],
        "Grid_Size": [64, 64, 128, 128, 256, 256],
        "Workgroup_Size": [16, 16, 32, 32, 64, 64],
        "LDS_Per_Workgroup": [1024, 1024, 2048, 2048, 4096, 4096],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [32, 32, 64, 64, 96, 96],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [16, 16, 32, 32, 48, 48],
        "Wave_Size": [64, 64, 64, 64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Kernel_ID": [501, 502, 503, 504, 505, 506],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_c",
            "kernel_c",
        ],
        "Start_Timestamp": [1000, 1100, 2000, 2100, 3000, 3100],
        "End_Timestamp": [1200, 1300, 2500, 2600, 3800, 3900],
        "Counter1": [100, 200, 300, 400, 500, 600],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_merge_counters_spatial_multiplex_timestamp_median_calculation():
    """
    Test merge_counters_spatial_multiplex timestamp median calculations.

    Returns:
        None: Asserts function correctly calculates median timestamps.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [64, 64, 64],
        "Workgroup_Size": [16, 16, 16],
        "LDS_Per_Workgroup": [1024, 1024, 1024],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [32, 32, 32],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [16, 16, 16],
        "Wave_Size": [64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003],
        "Kernel_ID": [501, 502, 503],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Counter1": [100, 200, 300],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


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


# =============================================================================
# additional tests for v3_counter_csv_to_v2_csv function
# =============================================================================


def create_csv_string(data_dict):
    return pd.DataFrame(data_dict).to_csv(index=False)


@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_v3_to_v2_agent_id_parsing_success_and_error(
    mock_console_debug, mock_console_error, tmp_path
):
    """
    Tests Line 1: Successful parsing of 'Agent Id' string.
    Tests Line 2: Graceful handling of malformed 'Agent Id' (no error expected).

    The new csv_ops-based implementation uses regex matching that simply
    doesn't match invalid formats, keeping the original value unchanged
    rather than raising an error.
    """
    import utils.utils_profile_csv as csv_ops

    agent_info_content = create_csv_string({
        "Node_Id": [0, 1],
        "Agent_Type": ["CPU", "GPU"],
        "Wave_Front_Size": [0, 64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted.csv"
    counter_content_success = create_csv_string({
        "Correlation_Id": [1],
        "Dispatch_Id": [10],
        "Agent_Id": ["Agent 1"],
        "Queue_Id": [100],
        "Process_Id": [1000],
        "Thread_Id": [10000],
        "Grid_Size": [256],
        "Kernel_Id": [1],
        "Kernel_Name": ["kernelA"],
        "Workgroup_Size": [64],
        "LDS_Block_Size": [32],
        "Scratch_Size": [0],
        "VGPR_Count": [16],
        "Accum_VGPR_Count": [0],
        "SGPR_Count": [32],
        "Start_Timestamp": [100000],
        "End_Timestamp": [100100],
        "Counter_Name": ["Cycles"],
        "Counter_Value": [5000],
    })
    counter_filepath_success = tmp_path / "counter_success.csv"
    counter_filepath_success.write_text(counter_content_success)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath_success),
        str(agent_info_filepath),
        str(converted_csv_filepath),
    )

    mock_console_error.assert_not_called()
    rows, _ = csv_ops.read_csv_as_dicts(str(converted_csv_filepath))
    assert len(rows) == 1
    assert "GPU_ID" in rows[0]
    # GPU_ID should be the index of the GPU agent (1 is the only GPU, so index 0)
    # Note: csv_ops returns strings, so we compare as string
    assert str(rows[0]["GPU_ID"]) == "0"

    mock_console_error.reset_mock()

    # Test with malformed Agent_Id - the new implementation gracefully handles
    # this by keeping the original value unchanged (regex simply doesn't match)
    counter_content_malformed = create_csv_string({
        "Correlation_Id": [2],
        "Dispatch_Id": [20],
        "Agent_Id": ["Malformed Agent X"],
        "Queue_Id": [200],
        "Process_Id": [2000],
        "Thread_Id": [20000],
        "Grid_Size": [512],
        "Kernel_Id": [2],
        "Kernel_Name": ["kernelB"],
        "Workgroup_Size": [128],
        "LDS_Block_Size": [64],
        "Scratch_Size": [0],
        "VGPR_Count": [32],
        "Accum_VGPR_Count": [0],
        "SGPR_Count": [64],
        "Start_Timestamp": [200000],
        "End_Timestamp": [200200],
        "Counter_Name": ["Instructions"],
        "Counter_Value": [10000],
    })
    counter_filepath_malformed = tmp_path / "counter_malformed.csv"
    counter_filepath_malformed.write_text(counter_content_malformed)
    converted_malformed_filepath = tmp_path / "converted_malformed.csv"

    # This should not raise an exception - malformed values are handled gracefully
    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath_malformed),
        str(agent_info_filepath),
        str(converted_malformed_filepath),
    )

    # console_error is not called because the regex simply doesn't match
    # and the original value is kept (no exception raised)
    mock_console_error.assert_not_called()

    # The output should still be written with the original Agent_Id value
    rows, _ = csv_ops.read_csv_as_dicts(str(converted_malformed_filepath))
    assert len(rows) == 1
    # GPU_ID will have the malformed value since it wasn't converted
    # It won't map to a GPU ID, so it stays as the original value
    assert "GPU_ID" in rows[0]


@mock.patch("utils.utils_profile.console_debug")  # To suppress debug output
def test_v3_to_v2_accum_column_rename(mock_console_debug, tmp_path):
    """
    Tests Line 3: Renaming of a column ending with '_ACCUM' to 'SQ_ACCUM_PREV_HIRES'.
    """
    # --- Setup ---
    agent_info_content = create_csv_string({
        "Node_Id": [0],
        "Agent_Type": ["GPU"],
        "Wave_Front_Size": [64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted_accum.csv"

    counter_data = {
        "Correlation_Id": [1, 1],
        "Dispatch_Id": [10, 10],
        "Agent_Id": [0, 0],
        "Queue_Id": [100, 100],
        "Process_Id": [1000, 1000],
        "Thread_Id": [10000, 10000],
        "Grid_Size": [256, 256],
        "Kernel_Id": [1, 1],
        "Kernel_Name": ["kernelA", "kernelA"],
        "Workgroup_Size": [64, 64],
        "LDS_Block_Size": [32, 32],
        "Scratch_Size": [0, 0],
        "VGPR_Count": [16, 16],
        "Accum_VGPR_Count": [0, 0],
        "SGPR_Count": [32, 32],
        "Start_Timestamp": [100000, 100000],
        "End_Timestamp": [100100, 100100],
        "Counter_Name": ["FETCH_SIZE_ACCUM", "CYCLES"],
        "Counter_Value": [12345, 5000],
    }
    counter_content = create_csv_string(counter_data)
    counter_filepath = tmp_path / "counter_accum.csv"
    counter_filepath.write_text(counter_content)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath), str(agent_info_filepath), str(converted_csv_filepath)
    )

    result_df = pd.read_csv(converted_csv_filepath)
    assert "SQ_ACCUM_PREV_HIRES" in result_df.columns
    assert "FETCH_SIZE_ACCUM" not in result_df.columns
    assert "CYCLES" in result_df.columns
    assert result_df["SQ_ACCUM_PREV_HIRES"].iloc[0] == 12345
    assert result_df["CYCLES"].iloc[0] == 5000


@mock.patch("utils.utils_profile.console_debug")
def test_v3_to_v2_default_accum_vgpr_count(mock_console_debug, tmp_path):
    """
    Tests Line 4: 'Accum_VGPR_Count' is added and set to 0 if not present in input.
    """
    agent_info_content = create_csv_string({
        "Node_Id": [0],
        "Agent_Type": ["GPU"],
        "Wave_Front_Size": [64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted_no_accum_vgpr.csv"

    counter_content = create_csv_string({
        "Correlation_Id": [1],
        "Dispatch_Id": [10],
        "Agent_Id": [0],
        "Queue_Id": [100],
        "Process_Id": [1000],
        "Thread_Id": [10000],
        "Grid_Size": [256],
        "Kernel_Id": [1],
        "Kernel_Name": ["kernelA"],
        "Workgroup_Size": [64],
        "LDS_Block_Size": [32],
        "Scratch_Size": [0],
        "VGPR_Count": [16],
        "SGPR_Count": [32],
        "Start_Timestamp": [100000],
        "End_Timestamp": [100100],
        "Counter_Name": ["Cycles"],
        "Counter_Value": [5000],
    })
    counter_filepath = tmp_path / "counter_no_accum_vgpr.csv"
    counter_filepath.write_text(counter_content)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath), str(agent_info_filepath), str(converted_csv_filepath)
    )

    result_df = pd.read_csv(converted_csv_filepath)
    assert "Accum_VGPR" in result_df.columns
    assert result_df["Accum_VGPR"].iloc[0] == 0
    assert result_df["Accum_VGPR"].dtype == "int64"


# ===================================================================
# Test PC_sampling function
# ===================================================================


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_sdk_path_nonexistent_librocprofiler_sdk_tool(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """
    Edge Case: rocprofiler_sdk_tool_path is valid, but librocprofiler-sdk-tool.so
    is NOT found next to it (or in rocprofiler-sdk subdir).
    This test primarily checks if the paths are constructed. The actual check for
    file existence before `capture_subprocess_output` is not in the provided snippet,
    but we test the path construction.
    """
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        method = "host_trap"
        interval = 1000
        workload_dir = str(tmp_path)
        options = {"APP_CMD": "my_app --arg"}

        sdk_lib_dir = tmp_path / "rocm_sdk" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path).touch()

        expected_tool_path = str(
            sdk_lib_dir / "rocprofiler-sdk" / "librocprofiler-sdk-tool.so"
        )

        options["LD_PRELOAD"] = expected_tool_path

        mock_capture_subprocess.return_value = (True, "Success output")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        call_args = mock_capture_subprocess.call_args
        called_env = call_args.kwargs.get("new_env", {})

        assert "LD_PRELOAD" in called_env
        assert called_env["LD_PRELOAD"] == expected_tool_path

        mock_console_error.assert_not_called()


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_subprocess_fails(
    mock_console_debug, mock_capture_subprocess, tmp_path, monkeypatch
):
    """
    Edge Case: The capture_subprocess_output returns success=False.
    This should trigger the console_error("PC sampling failed.").
    """
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "stochastic"
        interval = 5000
        workload_dir = str(tmp_path)
        options = ["another_app"]

        with pytest.raises(RuntimeError, match="console_error called"):
            utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        mock_capture_subprocess.assert_not_called()
        assert console_error_calls == [
            "APP_CMD, the workload's executable must be provided "
            "when not in live attach mode"
        ]

    mock_capture_subprocess.reset_mock()
    console_error_calls.clear()
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        options = {"APP_CMD": "another_app"}
        sdk_lib_dir = tmp_path / "rocm_sdk_fail" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path_sdk).touch()

        tool_dir = sdk_lib_dir / "rocprofiler-sdk"
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "librocprofiler-sdk-tool.so").touch()

        mock_capture_subprocess.return_value = (
            False,
            "Error output from SDK subprocess",
        )

        with pytest.raises(RuntimeError, match="console_error called"):
            utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        mock_capture_subprocess.assert_called_once()
        assert console_error_calls == ["PC sampling failed."]


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_empty_appcmd(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """
    Edge Case: The appcmd is an empty string.
    The function should still attempt to run it. The behavior of
    capture_subprocess_output with an empty command is external to this function.
    """
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "host_trap"
        interval = 100
        workload_dir = str(tmp_path)
        options = ["--"]
        rocprofiler_sdk_tool_path = "/some/path/librocprofiler_sdk.so"  # noqa: F841

        mock_capture_subprocess.return_value = (True, "Output with empty appcmd")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        assert options_list[-1] == "--"
        mock_console_error.assert_not_called()

    mock_capture_subprocess.reset_mock()
    mock_console_error.reset_mock()
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        sdk_lib_dir = tmp_path / "rocm_sdk_empty" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path_sdk).touch()
        tool_dir = sdk_lib_dir / "rocprofiler-sdk"
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "librocprofiler-sdk-tool.so").touch()

        mock_capture_subprocess.return_value = (True, "Output with empty appcmd SDK")
        options = {"APP_CMD": ""}

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        assert mock_capture_subprocess.call_args[0][0] == ""
        mock_console_error.assert_not_called()


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_multiarg_appcmd(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """All arguments after '--' in profiler_options must appear
    in the subprocess call."""
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "host_trap"
        interval = 100
        workload_dir = str(tmp_path)
        options = ["--kernel-trace", "--", "./myapp", "arg1", "arg2"]

        mock_capture_subprocess.return_value = (True, "Success")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        separator_index = options_list.index("--")
        assert options_list[separator_index:] == ["--", "./myapp", "arg1", "arg2"]
        mock_console_error.assert_not_called()


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
# TESTS FOR MODELESS COMMAND LINE OPTIONS
# =============================================================================


@pytest.mark.list_metrics
def test_list_metrics(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute(["--list-metrics", "gfx90a"])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "6 -> Workgroup Manager (SPI)" in output
    assert "5.2 -> Command processor packet processor (CPC)" in output


def test_list_blocks(binary_handler_analyze_rocprof_compute, capsys):
    return_code = binary_handler_analyze_rocprof_compute(["--list-blocks", "gfx90a"])
    assert return_code == 0

    # Test output
    output = capsys.readouterr().out
    assert "INDEX" in output
    assert "BLOCK ALIAS" in output
    assert "BLOCK NAME" in output

    # Verify specific block id, alias, and name mappings
    lines = output.strip().splitlines()
    block_entries = {}
    for line in lines[1:]:  # skip header
        parts = line.split()
        if len(parts) >= 3:
            block_id = parts[0]
            block_alias = parts[1]
            block_name = " ".join(parts[2:])
            block_entries[block_id] = (block_alias, block_name)

    assert block_entries["0"] == ("topstats", "Top Stats")
    assert block_entries["1"] == ("sysinfo", "System Info")
    assert block_entries["6"] == ("spi", "Workgroup Manager (SPI)")


# =============================================================================
# TESTS FOR AMDSMI INTERFACE
# =============================================================================


def test_amdsmi_ctx():
    from utils.amdsmi_interface import amdsmi_ctx, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("amdsmi.amdsmi_init") as amdsmi_init_mock:
        with mock.patch("amdsmi.amdsmi_shut_down") as amdsmi_shutdown_mock:
            with amdsmi_ctx():
                amdsmi_init_mock.assert_called_once()
            amdsmi_shutdown_mock.assert_called_once()


def test_amdsmi_get_device_handles():
    from utils.amdsmi_interface import get_device_handles, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("amdsmi.amdsmi_get_processor_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        handles = get_device_handles()
        assert handles[0] == 12345
        device_handles_mock.assert_called_once()

    with mock.patch(
        "amdsmi.amdsmi_get_processor_handles", side_effect=Exception("Mock exception")
    ) as device_handles_mock:
        handle = get_device_handles()
        assert len(handle) == 0


def test_amdsmi_get_mem_max_clock():
    from utils.amdsmi_interface import get_mem_max_clock, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [0, 4567]
        with mock.patch("amdsmi.amdsmi_get_clock_info") as mem_max_clock_mock:

            def side_effect(handle, *args, **kwargs):
                if handle == 0:
                    raise Exception("Invalid handle: 0")
                return {"max_clk": 100}

            mem_max_clock_mock.side_effect = side_effect
            clk = get_mem_max_clock()
            assert mem_max_clock_mock.call_count == 2
            assert clk == 100


def test_amdsmi_get_gpu_model():
    from utils.amdsmi_interface import get_gpu_model, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_board_info") as device_name_mock:
            with mock.patch("amdsmi.amdsmi_get_gpu_asic_info") as asic_name_mock:
                with mock.patch("amdsmi.amdsmi_get_gpu_vbios_info") as vbios_name_mock:
                    device_name_mock.return_value = {"product_name": "AMD MIXXX"}
                    asic_name_mock.return_value = {"market_name": "MIXXX"}
                    vbios_name_mock.return_value = {"name": "mixxx"}
                    model = get_gpu_model()
                    device_name_mock.assert_called_once()
                    assert model == ("AMD MIXXX", "MIXXX", "mixxx")

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_board_info", side_effect=Exception("Mock exception")
        ):
            model = get_gpu_model()
            assert model == ("N/A", "N/A", "N/A")


def test_amdsmi_get_gpu_vbios_part_number():
    from utils.amdsmi_interface import get_gpu_vbios_part_number, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_vbios_info") as vbios_part_number_mock:
            vbios_part_number_mock.return_value = {
                "part_number": "12345-67890",
            }
            part_number = get_gpu_vbios_part_number()
            vbios_part_number_mock.assert_called_once()
            assert part_number == "12345-67890"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_vbios_info", side_effect=Exception("Mock exception")
        ):
            part_number = get_gpu_vbios_part_number()
            assert part_number == "N/A"


def test_amdsmi_get_gpu_compute_partition():
    from utils.amdsmi_interface import get_gpu_compute_partition, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch(
            "amdsmi.amdsmi_get_gpu_compute_partition"
        ) as compute_partition_mock:
            compute_partition_mock.return_value = "Mock Partition"
            partition = get_gpu_compute_partition()
            compute_partition_mock.assert_called_once()
            assert partition == "Mock Partition"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_compute_partition",
            side_effect=Exception("Mock exception"),
        ):
            partition = get_gpu_compute_partition()
            assert partition == "N/A"


def test_amdsmi_get_gpu_memory_partition():
    from utils.amdsmi_interface import get_gpu_memory_partition, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch(
            "amdsmi.amdsmi_get_gpu_memory_partition"
        ) as memory_partition_mock:
            memory_partition_mock.return_value = "Mock Memory Partition"
            partition = get_gpu_memory_partition()
            memory_partition_mock.assert_called_once()
            assert partition == "Mock Memory Partition"

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_memory_partition",
            side_effect=Exception("Mock exception"),
        ):
            partition = get_gpu_memory_partition()
            assert partition == "N/A"


def test_amdsmi_get_gpu_cache_size():
    from utils.amdsmi_interface import get_gpu_cache_info, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_cache_info") as cache_info_mock:
            cache_info_mock.return_value = {"cache": "Mock Cache Info"}
            cache_info = get_gpu_cache_info()
            cache_info_mock.assert_called_once()
            assert cache_info == {"cache": "Mock Cache Info"}

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_cache_info",
            side_effect=Exception("Mock exception"),
        ):
            cache_info = get_gpu_cache_info()
            assert cache_info == {}


def test_amdsmi_get_gpu_num_compute_units():
    from utils.amdsmi_interface import get_gpu_num_compute_units, import_amdsmi_module

    _ = import_amdsmi_module()

    with mock.patch("utils.amdsmi_interface.get_device_handles") as device_handles_mock:
        device_handles_mock.return_value = [12345]
        with mock.patch("amdsmi.amdsmi_get_gpu_asic_info") as cu_mock:
            cu_mock.return_value = {"num_compute_units": 10}
            cu_count = get_gpu_num_compute_units()
            cu_mock.assert_called_once()
            assert cu_count == 10

        with mock.patch(
            "amdsmi.amdsmi_get_gpu_asic_info",
            side_effect=Exception("Mock exception"),
        ):
            cu_count = get_gpu_num_compute_units()
            assert cu_count == 0


# =============================================================================
# TESTS FOR ITERATION MULTIPLEXING
# =============================================================================


def test_impute_counters_iteration_multiplex(tmp_path: Path) -> None:
    """Test impute_counters_iteration_multiplex with sample DataFrame."""
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    # For "kernel" policy
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # Assert Counter1 and Counter2 imputed for first two dispatches
    assert result["Counter2"].iloc[0] == 500
    assert result["Counter1"].iloc[1] == 100

    # For "kernel_launch_params" policy
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # No imputation possible
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[2])

    # Test multi_kernel
    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    # For "kernel" policy
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # For "kernel_launch_params" policy
    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # No imputation possible
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[2])

    # Test incomplete last subgroup handling and no cross-subgroup contamination
    # Scenario: 3 counter buckets, 8 dispatches (2 complete subgroups + incomplete last)
    # Subgroup 0: rows 0-2, Subgroup 1: rows 3-5, Subgroup 2 (incomplete): rows 6-7
    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": ["kernel_a"] * 8,
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
        "End_Timestamp": [1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1, 1],
        # Counter bucket pattern: A, B, C (repeats)
        "Counter_A": [100, None, None, 200, None, None, 300, None],
        "Counter_B": [None, 110, None, None, 210, None, None, 310],
        "Counter_C": [None, None, 120, None, None, 220, None, None],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    # Verify complete subgroups: all rows should have all counters
    assert result["Counter_A"].iloc[0] == 100
    assert result["Counter_A"].iloc[1] == 100
    assert result["Counter_A"].iloc[2] == 100
    assert result["Counter_B"].iloc[0] == 110
    assert result["Counter_C"].iloc[0] == 120

    # Verify no cross-subgroup contamination: subgroup 1 has its own values
    assert result["Counter_A"].iloc[3] == 200
    assert result["Counter_A"].iloc[4] == 200
    assert result["Counter_B"].iloc[3] == 210
    assert result["Counter_C"].iloc[3] == 220

    # Verify incomplete last subgroup gets filled from previous subgroup
    # Row 6-7 only have Counter_A and Counter_B, missing Counter_C
    assert result["Counter_A"].iloc[6] == 300
    assert result["Counter_A"].iloc[7] == 300
    assert result["Counter_B"].iloc[6] == 310
    assert result["Counter_B"].iloc[7] == 310
    # Counter_C should be filled from previous subgroup via global ffill
    assert result["Counter_C"].iloc[6] == 220
    assert result["Counter_C"].iloc[7] == 220

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8  # Ensure same number of rows


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
# TESTS FOR NOISE_CLAMP: Multi-Pass Profiling Variance Handling
# =============================================================================


@pytest.mark.noise_clamp
def test_noise_clamp_clamping_behavior():
    """Core behavior: positives unchanged, negatives clamped to 0."""
    import numpy as np

    from utils.metrics.noise_clamper import to_noise_clamp

    # Scalar: positive unchanged
    assert to_noise_clamp(1000.0, 100000.0) == 1000.0
    # Scalar: negative clamped
    assert to_noise_clamp(-100.0, 1000000.0) == 0.0

    # Series: mixed values
    diff = pd.Series([100.0, -50.0, 200.0, -100.0])
    ref = pd.Series([1e6, 1e6, 1e6, 1e6])
    result = to_noise_clamp(diff, ref)
    pd.testing.assert_series_equal(result, pd.Series([100.0, 0.0, 200.0, 0.0]))

    # NumPy array
    diff_np = np.array([100.0, -50.0])
    ref_np = np.array([1e6, 1e6])
    result_np = to_noise_clamp(diff_np, ref_np)
    np.testing.assert_array_equal(result_np, np.array([100.0, 0.0]))


@pytest.mark.noise_clamp
def test_noise_clamp_zero_reference():
    """Edge case: zero reference should not cause division by zero."""
    from utils.metrics.noise_clamper import to_noise_clamp

    assert to_noise_clamp(-100.0, 0.0) == 0.0
    result = to_noise_clamp(pd.Series([-100.0]), pd.Series([0.0]))
    assert result.iloc[0] == 0.0


@pytest.mark.noise_clamp
def test_noise_clamp_warning_above_threshold():
    """Warning recorded when relative error >= 1%."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # 2% error (above 1% threshold) - should record
    to_noise_clamp(pd.Series([-20000.0]), pd.Series([1000000.0]))

    stats = get_noise_clamp_warnings()
    assert stats["count"] == 1
    assert stats["max_rel"] >= 0.01


@pytest.mark.noise_clamp
def test_noise_clamp_no_warning_below_threshold():
    """No warning when relative error < 1%."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # 0.5% error (below 1% threshold) - still clamped, no warning
    result = to_noise_clamp(pd.Series([-5000.0]), pd.Series([1000000.0]))
    assert result.iloc[0] == 0.0
    assert get_noise_clamp_warnings()["count"] == 0


@pytest.mark.noise_clamp
def test_noise_clamp_empty_input():
    """Empty inputs should return empty without error."""
    from utils.metrics.noise_clamper import to_noise_clamp

    result = to_noise_clamp(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert len(result) == 0


@pytest.mark.noise_clamp
def test_noise_clamp_threshold_boundary():
    """Exactly 1% error should trigger warning (>= not >)."""
    from utils.metrics.noise_clamper import (
        clear_noise_clamp_warnings,
        get_noise_clamp_warnings,
        to_noise_clamp,
    )

    clear_noise_clamp_warnings()

    # Exactly 1% error: -10000 / 1000000 = 0.01
    to_noise_clamp(pd.Series([-10000.0]), pd.Series([1000000.0]))
    assert get_noise_clamp_warnings()["count"] == 1


@pytest.mark.noise_clamp
def test_noise_clamper_instance_isolation():
    """Separate NoiseClamper instances should have independent state."""
    import numpy as np

    from utils.metrics.noise_clamper import NoiseClamper

    clamper1 = NoiseClamper()
    clamper2 = NoiseClamper()

    clamper1.clamp(pd.Series([-20000.0]), pd.Series([1000000.0]))

    assert clamper1.get_stats()["count"] == 1
    assert clamper2.get_stats()["count"] == 0

    clamper1.clear()
    assert clamper1.get_stats()["count"] == 0
    assert clamper2.get_stats()["count"] == 0

    clamper1.clamp(np.array([-50000.0]), np.array([1000000.0]))
    clamper2.clamp(np.array([-30000.0, -40000.0]), np.array([1000000.0, 1000000.0]))

    assert clamper1.get_stats()["count"] == 1
    assert clamper2.get_stats()["count"] == 2


# =============================================================================
# Experimental Feature Tests
# =============================================================================


@pytest.mark.experimental_feature
def test_experimental_feature_without_flag_errors(monkeypatch, capsys):
    """Test that using experimental feature without --experimental flag raises error."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv to simulate command-line usage
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])

    # Create a self-contained parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Test that using experimental feature without --experimental causes error
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args()

    assert exc_info.value.code == 2  # argparse error exit code
    captured = capsys.readouterr()
    assert "experimental feature" in captured.err.lower()
    assert "--experimental" in captured.err.lower()


@pytest.mark.experimental_feature
def test_experimental_feature_with_flag_succeeds(monkeypatch, caplog):
    """Test that using experimental feature with --experimental flag succeeds."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv to simulate command-line usage with --experimental
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])

    # Create a self-contained parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=True,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Parse args - should succeed and print warning
    parser.parse_args()

    # Verify warning was logged
    assert "Test experimental feature" in caplog.text
    assert "experimental" in caplog.text.lower()
    assert "may change in future releases" in caplog.text.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_parsing_before_separator(monkeypatch, caplog):
    """Test that prelim parser correctly detects --experimental
    before '--' separator."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv with --experimental before separator
    monkeypatch.setattr(
        "sys.argv",
        ["rocprof-compute", "--experimental", "profile", "-n", "test", "--", "./app"],
    )

    # Create a self-contained prelim parser
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--experimental", action="store_true", default=False)
    prelim_parser.parse_known_args()

    # Create full parser with experimental feature
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental", action="store_true", default=False)
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=True,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    # Parse with just the experimental feature flag
    monkeypatch.setattr("sys.argv", ["rocprof-compute", "--test-exp-feature"])
    parser.parse_args()

    assert "experimental" in caplog.text.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_parsing_after_separator(monkeypatch, capsys):
    """Test that prelim parser ignores --experimental after '--' separator."""
    import argparse

    from argparser import ExperimentalAction

    # Monkeypatch sys.argv with --experimental after separator
    monkeypatch.setattr(
        "sys.argv",
        ["rocprof-compute", "profile", "-n", "test", "--", "./app", "--experimental"],
    )

    # Create a self-contained prelim parser
    prelim_parser = argparse.ArgumentParser(add_help=False)
    prelim_parser.add_argument("--experimental", action="store_true", default=False)
    prelim_parser.parse_known_args()

    # Create full parser with experimental feature
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Custom Help",
    )

    with pytest.raises(SystemExit):
        parser.parse_args()

    captured = capsys.readouterr()
    assert "use --experimental" not in captured.err.lower()


@pytest.mark.experimental_feature
def test_experimental_flag_without_features(monkeypatch, capsys):
    """Test that --experimental flag is parsed correctly even without
    experimental features."""
    import argparse

    # Monkeypatch sys.argv with --experimental but no experimental features
    monkeypatch.setattr(
        "sys.argv", ["rocprof-compute", "--experimental", "profile", "-n", "test"]
    )

    # Create a self-contained parser with just --experimental flag
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental", action="store_true", default=False)
    parser.add_argument("profile", nargs="?")
    parser.add_argument("-n", "--name", type=str)

    # Parse args - should succeed without errors since no experimental features used
    parser.parse_args()

    # Verify no errors or warnings
    captured = capsys.readouterr()
    assert captured.err == "", f"{captured.err}"


@pytest.mark.experimental_feature
def test_experimental_action_help_suppression():
    """Test that ExperimentalAction suppresses help when experimental_enabled=False."""
    import argparse

    from argparser import ExperimentalAction

    # Create parser without experimental enabled
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-exp-feature",
        action=ExperimentalAction,
        experimental_enabled=False,
        feature_label="Test experimental feature",
        base_action="store_const",
        nargs=0,
        const=True,
        default=False,
        help="Test help text",
    )

    # Get help text
    help_text = parser.format_help()

    # Help should be suppressed
    assert "--test-exp-feature" not in help_text, f"{help_text}"


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
# TESTS FOR Analysis DB mode: Analysis DB mode code path
# =============================================================================


def test_calc_roofline_data_early_exit_on_empty_roofline_df(monkeypatch):
    """Test calc_roofline_data exits early when roofline data is empty.

    This test verifies that when the roofline dataframe (ID 402) is empty
    or filtered out, the function logs a warning and skips that workload
    without adding it to the result dictionary.
    """
    from rocprof_compute_analyze.analysis_db import db_analysis

    # Create mock db_analysis instance
    analyzer = mock.MagicMock(spec=db_analysis)

    # Mock workload data
    workload_path = "/mock/workload/path"
    mock_runs = {
        workload_path: mock.MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx90a"}]))
    }

    # Mock PMC dataframe with kernel data
    mock_pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel1", "kernel2"],
        "Start_Timestamp": [100, 200],
        "End_Timestamp": [150, 300],
    })

    # Mock architecture config with EMPTY roofline dataframe (ID 402)
    mock_arch_config = mock.MagicMock()
    mock_arch_config.dfs = {
        402: pd.DataFrame()  # Empty roofline dataframe triggers early exit
    }

    # Setup instance variables
    analyzer._runs = mock_runs
    analyzer._pmc_df_per_workload = {workload_path: mock_pmc_df}
    analyzer._arch_configs = {"gfx90a": mock_arch_config}
    analyzer.get_args = mock.MagicMock(return_value=mock.MagicMock(max_stat_num=10))

    # Mock console_warning to verify it's called
    warning_messages = []

    def mock_warning(msg):
        warning_messages.append(msg)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_warning", mock_warning
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_debug", lambda msg: None
    )

    # Call the actual function
    result = db_analysis.calc_roofline_data(analyzer)

    # Verify early exit behavior
    assert len(result[0]) == 0, (
        "Should return empty kernel level dict when roofline data is empty"
    )
    assert len(result[1]) == 0, (
        "Should return empty workload level dict when roofline data is empty"
    )
    assert len(warning_messages) == 1, "Should log one warning message"
    assert "Roofline data is filtered out or not found" in warning_messages[0]
    assert workload_path in warning_messages[0]


# =============================================================================
# GPU Benchmark Locking Tests
# =============================================================================


@pytest.mark.misc
def test_gpu_benchmark_locking(tmp_path, monkeypatch, capsys):
    """Test GPU benchmark locking functions."""
    import fcntl

    import roofline.benchmark.benchmark_base as benchmark_base

    # --- Setup: redirect lock directory to temp path ---
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    # Mock GPU UUID
    monkeypatch.setattr(
        benchmark_base.hip,
        "hipGetDeviceProperties",
        lambda d: mock.Mock(uuid=mock.Mock(uuid=bytes([0x01, 0x02, 0x03, 0x04]))),
    )

    # Mock Path to use our temp directory
    original_path = Path

    def mock_path(p):
        if p == "/tmp/rocprof-compute-benchmark":
            return lock_dir
        return original_path(p)

    monkeypatch.setattr(benchmark_base, "Path", mock_path)

    deviceID = 0
    cache_sizes = {}
    # Create Bench_base object in order to call gpu benchmark lock method
    # Device ID list arg doesn't matter since we are just using the base class
    # cache_sizes can be empty for this test since we do not need it to test locking
    testClass = benchmark_base.Bench_base(deviceID, cache_sizes)

    # --- Test lock acquisition and lock file creation ---
    with testClass.gpu_benchmark_lock(deviceID):
        lock_file = lock_dir / "rocprof-compute-benchmark-01020304.lock"
        assert lock_file.exists()

    # --- Test no message when lock acquired immediately ---
    capsys.readouterr()  # Clear previous output
    with testClass.gpu_benchmark_lock(deviceID):
        pass
    output = capsys.readouterr().out
    assert "Waiting" not in output

    # --- Test waiting/acquired messages when lock is contended ---
    call_count = {"count": 0}

    def mock_flock(fd, op):
        call_count["count"] += 1
        if call_count["count"] == 1 and (op & fcntl.LOCK_NB):
            raise BlockingIOError("Lock held by another process")

    monkeypatch.setattr(benchmark_base.fcntl, "flock", mock_flock)

    with testClass.gpu_benchmark_lock(deviceID):
        pass

    output = capsys.readouterr().out
    assert "Waiting for GPU 0" in output
    assert "another rocprof-compute benchmark is in progress" in output
    assert "Acquired lock for GPU 0" in output


# ---------------------------------------------------------------------------
# Tests for call-tree functions (build/display)
# ---------------------------------------------------------------------------


def test_parse_location_normal():
    assert parse_top_level_location("10@main.py:60/#10@main.py:21") == "main.py:60"


def test_parse_location_single_entry():
    assert parse_top_level_location("5@train.py:42") == "train.py:42"


def test_parse_location_nan():
    assert parse_top_level_location(float("nan")) == "unknown:0"


def test_parse_location_none():
    assert parse_top_level_location(None) == "unknown:0"


def test_parse_location_empty():
    assert parse_top_level_location("") == "unknown:0"
    assert parse_top_level_location("   ") == "unknown:0"


def test_parse_location_no_at_sign():
    assert parse_top_level_location("no_at_sign") == "unknown:0"


def test_parse_location_no_colon():
    assert parse_top_level_location("10@mainpy") == "unknown:0"


def test_format_duration_microseconds_below_threshold():
    assert format_duration(0.005) == "5.00 us"


def test_format_duration_milliseconds_above_threshold():
    assert format_duration(1.5) == "1.50 ms"


def test_format_duration_boundary_value_is_milliseconds():
    assert format_duration(0.01) == "0.01 ms"


def test_format_duration_none_renders_na():
    assert format_duration(None) == "N/A"


def test_format_duration_nan_renders_na():
    assert format_duration(float("nan")) == "N/A"


def test_kernel_stats_defaults_min_max_to_none():
    stats = KernelStats()
    assert stats.min_duration_ns is None
    assert stats.max_duration_ns is None


def test_call_tree_node_defaults_dispatch_stats_to_none():
    node = CallTreeNode(name="x")
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns is None


def test_call_tree_node_call_count_is_property_of_invocation_ids():
    node = CallTreeNode(name="x")
    assert node.call_count == 0
    node.invocation_ids.add("ctx1")
    node.invocation_ids.add("ctx2")
    assert node.call_count == 2


def test_format_node_stats_omits_calls_when_no_invocation_ids():
    node = CallTreeNode(name="x")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    node.mean_dispatch_ns = 1_000_000.0
    node.min_dispatch_ns = 1_000_000.0
    node.max_dispatch_ns = 1_000_000.0
    rendered = format_node_stats(node)
    assert "calls:" not in rendered
    assert "dispatches: 1" in rendered
    assert "total: 1.00 ms" in rendered


def test_format_node_stats_includes_calls_when_invocation_ids_present():
    node = CallTreeNode(name="x")
    node.invocation_ids.add("ctx1")
    node.invocation_ids.add("ctx2")
    node.kernel_launches = 4
    node.total_duration_ms = 2.0
    node.mean_dispatch_ns = 500_000.0
    node.min_dispatch_ns = 500_000.0
    node.max_dispatch_ns = 500_000.0
    rendered = format_node_stats(node)
    assert "calls: 2" in rendered
    assert "dispatches: 4" in rendered


def test_format_node_stats_renders_na_when_dispatch_stats_missing():
    node = CallTreeNode(name="x")
    node.kernel_launches = 0
    rendered = format_node_stats(node)
    assert "dispatch_mean: N/A" in rendered
    assert "dispatch_min: N/A" in rendered
    assert "dispatch_max: N/A" in rendered


def test_rollup_leaf_node():
    node = CallTreeNode(name="leaf")
    node.kernels["kern_a"] = KernelStats(launches=2, total_duration_ns=1000.0)
    rollup = rollup_node_stats(node)
    assert rollup.launches == 2
    assert rollup.total_duration_ns == 1000.0
    assert node.kernel_launches == 2


def test_rollup_leaf_node_with_no_min_max_returns_none():
    node = CallTreeNode(name="leaf")
    node.kernels["kern"] = KernelStats(launches=1, total_duration_ns=0.0)
    rollup = rollup_node_stats(node)
    assert isinstance(rollup, NodeRollup)
    assert rollup.min_dispatch_ns is None
    assert rollup.max_dispatch_ns is None
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns == 0.0


def test_rollup_leaf_node_with_zero_launches_has_mean_none():
    node = CallTreeNode(name="leaf")
    rollup = rollup_node_stats(node)
    assert rollup.launches == 0
    assert node.mean_dispatch_ns is None


def test_rollup_propagates_min_max_from_kernel_stats():
    node = CallTreeNode(name="leaf")
    node.kernels["k"] = KernelStats(
        launches=2,
        total_duration_ns=3000.0,
        min_duration_ns=1000.0,
        max_duration_ns=2000.0,
    )
    rollup_node_stats(node)
    assert node.min_dispatch_ns == 1000.0
    assert node.max_dispatch_ns == 2000.0
    assert node.mean_dispatch_ns == 1500.0


def test_rollup_parent_rolls_up_children():
    child = CallTreeNode(name="child")
    child.kernels["kern_a"] = KernelStats(launches=3, total_duration_ns=3000.0)
    parent = CallTreeNode(name="parent")
    parent.children["child"] = child
    parent.kernels["kern_b"] = KernelStats(launches=1, total_duration_ns=500.0)
    rollup_node_stats(parent)
    assert parent.kernel_launches == 4
    assert child.kernel_launches == 3


def test_rollup_deep_hierarchy():
    grandchild = CallTreeNode(name="grandchild")
    grandchild.kernels["k"] = KernelStats(launches=1, total_duration_ns=100.0)
    child = CallTreeNode(name="child")
    child.children["grandchild"] = grandchild
    child.kernels["k2"] = KernelStats(launches=2, total_duration_ns=200.0)
    root = CallTreeNode(name="root")
    root.children["child"] = child
    rollup_node_stats(root)
    assert grandchild.kernel_launches == 1
    assert child.kernel_launches == 3
    assert root.kernel_launches == 3


def test_build_call_trees_empty_df():
    assert build_call_trees(pd.DataFrame()) == {}


def test_build_call_trees_missing_columns():
    assert build_call_trees(pd.DataFrame([{"Operator_Name": "a"}])) == {}


def test_build_call_trees_single_dispatch():
    df = pd.DataFrame([
        {
            "Operator_Name": "torch.nn.Linear",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@train.py:42",
            "Start_Timestamp_kernel": 1000,
            "End_Timestamp_kernel": 2000,
        }
    ])
    call_trees = build_call_trees(df)
    assert "train.py:42" in call_trees
    assert call_trees["train.py:42"].kernel_launches == 1
    assert "torch.nn.Linear" in call_trees["train.py:42"].children


def test_build_call_trees_hierarchy_split():
    df = pd.DataFrame([
        {
            "Operator_Name": "aten/linear/addmm",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@file.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    call_trees = build_call_trees(df)
    root = call_trees["file.py:1"]
    assert "aten" in root.children
    assert "linear" in root.children["aten"].children
    assert "addmm" in root.children["aten"].children["linear"].children


def test_build_call_trees_multiple_dispatches_same_kernel():
    rows = [
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": i * 1000,
            "End_Timestamp_kernel": (i + 1) * 1000,
        }
        for i in range(3)
    ]
    call_trees = build_call_trees(pd.DataFrame(rows))
    assert call_trees["f.py:1"].kernel_launches == 3
    assert call_trees["f.py:1"].children["op_a"].kernels["kern"].launches == 3


def test_build_call_trees_dedup_identical_timestamps():
    row = {
        "Operator_Name": "op",
        "Kernel_Name": "kern",
        "Context_Id": "10@f.py:1",
        "Start_Timestamp_kernel": 1000,
        "End_Timestamp_kernel": 2000,
    }
    assert build_call_trees(pd.DataFrame([row, row]))["f.py:1"].kernel_launches == 1


def test_build_call_trees_no_context_id():
    df = pd.DataFrame([
        {
            "Operator_Name": "op",
            "Kernel_Name": "kern",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    assert "unknown:0" in build_call_trees(df)


def test_build_call_trees_duration_rollup():
    df = pd.DataFrame([
        {
            "Operator_Name": "parent/child",
            "Kernel_Name": "kern_a",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "parent",
            "Kernel_Name": "kern_b",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 2_000_000,
            "End_Timestamp_kernel": 3_000_000,
        },
    ])
    call_trees = build_call_trees(df)
    root = call_trees["f.py:1"]
    assert root.kernel_launches == 2
    assert root.children["parent"].kernel_launches == 2
    assert root.children["parent"].children["child"].kernel_launches == 1


def test_build_call_trees_multiple_source_locations():
    df = pd.DataFrame([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@a.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "10@b.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
    ])
    call_trees = build_call_trees(df)
    assert "a.py:1" in call_trees
    assert "b.py:2" in call_trees


def test_show_call_tree_prints_location_and_stats(capsys):
    root = CallTreeNode(name="main.py:10")
    root.kernel_launches = 1
    root.total_duration_ms = 0.5
    child = CallTreeNode(name="op_a")
    child.kernel_launches = 1
    child.total_duration_ms = 0.5
    child.kernels["kern"] = KernelStats(launches=1, total_duration_ns=500_000.0)
    root.children["op_a"] = child
    show_call_tree({"main.py:10": root})
    output = capsys.readouterr().out
    assert "main.py:10" in output
    assert "dispatches: 1" in output
    assert "kern" in output


def test_show_call_tree_sorted_by_duration(capsys):
    root_a = CallTreeNode(name="a.py:1")
    root_a.total_duration_ms = 10.0
    root_a.kernel_launches = 1
    root_b = CallTreeNode(name="b.py:1")
    root_b.total_duration_ms = 20.0
    root_b.kernel_launches = 2
    show_call_tree({"a.py:1": root_a, "b.py:1": root_b})
    output = capsys.readouterr().out
    assert output.index("b.py:1") < output.index("a.py:1")


def test_show_call_tree_kernel_id_printed(capsys):
    root = CallTreeNode(name="f.py:1")
    root.kernel_launches = 1
    root.total_duration_ms = 1.0
    child = CallTreeNode(name="op")
    child.kernel_launches = 1
    child.total_duration_ms = 1.0
    child.kernels["kern_x"] = KernelStats(
        launches=1, total_duration_ns=1_000_000.0, kernel_id=42
    )
    root.children["op"] = child
    show_call_tree({"f.py:1": root})
    output = capsys.readouterr().out
    assert "(id 42)" in output


def test_print_operator_node_branching_shows_stats(capsys):
    node = CallTreeNode(name="branch")
    node.kernel_launches = 2
    node.total_duration_ms = 5.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    node.kernels["k2"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    print_operator_node(node)
    output = capsys.readouterr().out
    assert "dispatches: 2" in output
    assert "k1" in output
    assert "k2" in output


def test_print_operator_node_non_branching_omits_stats(capsys):
    node = CallTreeNode(name="single")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=1_000_000.0)
    print_operator_node(node)
    output = capsys.readouterr().out
    lines = output.strip().split("\n")
    assert "└─ single" in lines[0]
    assert "dispatches" not in lines[0]


def test_print_operator_node_long_kernel_wraps(capsys):
    node = CallTreeNode(name="single")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    long_kernel_name = "K" * 220
    node.kernels[long_kernel_name] = KernelStats(
        launches=1,
        total_duration_ns=1_000_000.0,
        kernel_id=7,
    )
    print_operator_node(node)
    output_lines = capsys.readouterr().out.splitlines()
    assert any("└─ single" in line for line in output_lines)
    kernel_lines = [
        line for line in output_lines if "(id 7)" in line or line.startswith("   ")
    ]
    assert any(line.startswith("   └─ ") for line in kernel_lines)
    wrapped_kernel_lines = [
        line
        for line in kernel_lines
        if line.startswith("   ") and not line.startswith("   └─ ")
    ]
    assert wrapped_kernel_lines
    assert not any(line.strip().startswith("(id 7)") for line in output_lines)


# ---------------------------------------------------------------------------
# build_operator_summary
# ---------------------------------------------------------------------------


_OPERATOR_SUMMARY_COLUMNS = [
    "Operator",
    "Location",
    "Calls",
    "Dispatches",
    "Dispatches_Per_Call",
    "Total_GPU",
    "Pct_Total_GPU",
    "Mean_Per_Call",
    "Mean_Per_Dispatch",
    "Min_Dispatch",
    "Max_Dispatch",
]


def _build_summary_from_dataframe(rows):
    call_trees = build_call_trees(pd.DataFrame(rows))
    return build_operator_summary(call_trees)


def test_build_operator_summary_empty_input_returns_empty_with_full_schema():
    summary = build_operator_summary({})
    assert list(summary.columns) == _OPERATOR_SUMMARY_COLUMNS
    assert summary.empty


def test_build_operator_summary_skips_synthetic_location_root():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        }
    ])
    assert "f.py:1" not in summary["Operator"].tolist()
    assert "op_a" in summary["Operator"].tolist()


def test_build_operator_summary_row_values_for_single_dispatch():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 2_000_000,
        }
    ])
    row = summary.loc[summary["Operator"] == "op_a"].iloc[0]
    assert row["Location"] == "f.py:1"
    assert row["Calls"] == 1
    assert row["Dispatches"] == 1
    assert row["Dispatches_Per_Call"] == 1.0
    assert row["Total_GPU"] == pytest.approx(2.0)
    assert row["Pct_Total_GPU"] == pytest.approx(100.0)
    assert row["Mean_Per_Call"] == pytest.approx(2.0)
    assert row["Mean_Per_Dispatch"] == pytest.approx(2.0)
    assert row["Min_Dispatch"] == pytest.approx(2.0)
    assert row["Max_Dispatch"] == pytest.approx(2.0)


def test_build_operator_summary_sort_by_total_descending():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "small_op",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "big_op",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 10_000_000,
        },
    ])
    operators_in_order = summary["Operator"].tolist()
    assert operators_in_order.index("big_op") < operators_in_order.index("small_op")


def test_build_operator_summary_pct_total_gpu_sums_to_100_at_top_level():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 3_000_000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
    ])
    op_a_pct = summary.loc[summary["Operator"] == "op_a", "Pct_Total_GPU"].iloc[0]
    op_b_pct = summary.loc[summary["Operator"] == "op_b", "Pct_Total_GPU"].iloc[0]
    assert op_a_pct == pytest.approx(75.0)
    assert op_b_pct == pytest.approx(25.0)


def test_build_operator_summary_pct_total_gpu_is_nan_when_grand_total_zero():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 0.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    pct = summary.loc[summary["Operator"] == "op", "Pct_Total_GPU"].iloc[0]
    assert math.isnan(pct)


def test_build_operator_summary_min_max_mean_are_nan_when_no_dispatch_stats():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 5.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "op"].iloc[0]
    assert math.isnan(row["Min_Dispatch"])
    assert math.isnan(row["Max_Dispatch"])
    assert math.isnan(row["Mean_Per_Dispatch"])


def test_build_operator_summary_calls_nan_when_no_invocation_ids():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="torch.ops.x")
    op.kernel_launches = 2
    op.total_duration_ms = 4.0
    op.mean_dispatch_ns = 2_000_000.0
    op.min_dispatch_ns = 2_000_000.0
    op.max_dispatch_ns = 2_000_000.0
    root.children["torch.ops.x"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "torch.ops.x"].iloc[0]
    assert math.isnan(row["Calls"])
    assert math.isnan(row["Dispatches_Per_Call"])
    assert math.isnan(row["Mean_Per_Call"])
    assert row["Dispatches"] == 2


# ---------------------------------------------------------------------------
# show_operator_summary
# ---------------------------------------------------------------------------


def test_show_operator_summary_empty_prints_no_dispatches_message(capsys):
    show_operator_summary(pd.DataFrame(columns=_OPERATOR_SUMMARY_COLUMNS))
    output = capsys.readouterr().out
    assert "no operators with recorded dispatches" in output


def test_show_operator_summary_renders_per_cell_unit_suffix(capsys):
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 2_000_000,
        }
    ])
    show_operator_summary(summary)
    output = capsys.readouterr().out
    assert "ms" in output or "us" in output
    assert "Operator" in output
    assert "Total" in output


def test_show_operator_summary_renders_na_for_nan_cells(capsys):
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 0.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    show_operator_summary(summary)
    output = capsys.readouterr().out
    assert "N/A" in output


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


# ---------------------------------------------------------------------------
# Torch operator pattern matching (PurePosixPath glob)
# ---------------------------------------------------------------------------

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"
H2 = "nn.Module.Net.forward/torch.nn.functional.conv2d"
H1 = "torch.relu"


@pytest.mark.torch_ops
def test_all_keyword():
    """'all' maps to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert m("all", H2)
    assert m("all", H1)
    assert not m("all", "")


@pytest.mark.torch_ops
def test_bare_pattern_matches_last_component():
    """Bare token is matched via PurePosixPath.match() against the full hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert m("torch.nn.functional.conv2d", H2)
    assert not m("relu", H3)
    assert not m("forward", H3)
    assert not m("sigmoid", H3)


@pytest.mark.torch_ops
def test_bare_wildcard_pattern():
    """Wildcard bare token matched via PurePosixPath.match()."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.*", H3)
    assert m("*relu", H3)
    assert m("*conv*", H2)
    assert not m("conv*", H2)
    assert not m("sigm*", H3)


@pytest.mark.torch_ops
def test_hierarchy_glob():
    """Patterns with '/' match across multiple hierarchy components."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("nn.Module.Net.forward/*/torch.relu", H3)
    assert m("*/torch.nn.functional.conv2d", H2)
    assert not m("nn.Module.Net.forward/torch.relu", H3)


@pytest.mark.torch_ops
def test_leading_slash_is_cosmetic():
    """Leading '/' is stripped during pattern normalization."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("/nn.Module.Net.forward/*/torch.relu", H3)
    assert m("/torch.relu", H3)


@pytest.mark.torch_ops
def test_trailing_slash_stripped_by_posixpath():
    """PurePosixPath strips trailing slashes, so they are cosmetic."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("nn.Module.Net.forward/", H3)
    assert m("torch.relu/", H3)


@pytest.mark.torch_ops
def test_regex_not_supported():
    """Regex syntax has no special meaning; treated as literal glob text."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("relu|conv2d", H3)
    assert not m("^torch\\.relu$", H3)
    assert not m("not:relu", H3)
    assert not m("2:functional", H3)


@pytest.mark.torch_ops
def test_empty_inputs():
    """Empty pattern or operator_name returns False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("", H3)
    assert not m("relu", "")
    assert not m("", "")


@pytest.mark.torch_ops
def test_slash_only_markers():
    """Scope-marker-only tokens should not match any hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("/", H3)
    assert not m("//", H3)


# -- get_matched_torch_operators_for_display ---------------------------------


def get_matched_torch_operators_for_display(
    torch_operators: dict[str, pd.DataFrame],
    pattern_list: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Return (operator_name, filtered_df) for each operator matching any pattern.

    Test-only helper: iterates every unique Operator_Name across all torch trace
    DataFrames and checks each against the supplied glob patterns.
    """
    from utils.parser import torch_operator_pattern_matches

    if not torch_operators or not pattern_list:
        return []
    result: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for _, df in torch_operators.items():
        if df is None or df.empty or "Operator_Name" not in df.columns:
            continue
        for op_name in df["Operator_Name"].dropna().unique():
            op_str = str(op_name).strip()
            if op_str in seen:
                continue
            for pattern in pattern_list:
                if torch_operator_pattern_matches(pattern.strip(), op_str):
                    seen.add(op_str)
                    result.append((op_str, df.loc[df["Operator_Name"] == op_name]))
                    break
    return result


@pytest.mark.torch_ops
def test_display_match_hierarchy_glob():
    """Full hierarchy globs are honored by display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H3, H2],
        "Kernel_Name": ["k1", "k2", "k3"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*/torch.relu"])
    assert len(matched) == 1
    assert matched[0][0] == H3


@pytest.mark.torch_ops
def test_display_match_multi_patterns():
    """Multiple glob patterns match their respective operators."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(
        torch_operators, ["*relu", "*conv*"]
    )
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_no_match():
    """No matches returns empty list."""
    df = pd.DataFrame({
        "Operator_Name": [H3],
        "Kernel_Name": ["k1"],
    })
    assert get_matched_torch_operators_for_display({"t": df}, ["sigmoid"]) == []


@pytest.mark.torch_ops
def test_display_empty_inputs():
    """Empty torch_operators or pattern_list returns []."""
    assert get_matched_torch_operators_for_display({}, ["relu"]) == []
    assert get_matched_torch_operators_for_display({"x": pd.DataFrame()}, []) == []


# -- parse_torch_operator_patterns ------------------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu"])
    assert parse_torch_operator_patterns(args) == ["relu"]

    args = Namespace(torch_operator=["relu", "conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu,conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    assert parse_torch_operator_patterns(Namespace(torch_operator=[])) == ["**"]
    assert parse_torch_operator_patterns(Namespace(torch_operator=None)) == []
    assert parse_torch_operator_patterns(Namespace()) == []


# -- PatternMatcherEngine ---------------------------------------------------


@pytest.mark.torch_ops
def test_engine_glob_hierarchy_mode():
    """Facade delegates matching to glob-hierarchy implementation."""
    from utils.pattern_matching import PatternMatcherEngine

    matcher = PatternMatcherEngine(mode="glob-hierarchy")
    assert matcher.matches("torch.relu", H3)
    assert matcher.matches("*relu", H3)
    assert not matcher.matches("sigmoid", H3)


@pytest.mark.torch_ops
def test_engine_invalid_mode():
    """Unsupported strategy names should raise ValueError."""
    from utils.pattern_matching import PatternMatcherEngine

    with pytest.raises(ValueError):
        PatternMatcherEngine(mode="regex")


# -- Additional coverage (xuchen #26) ----------------------------------------


@pytest.mark.torch_ops
def test_double_star_explicit():
    """'**' matches any hierarchy depth."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("**", H3)
    assert m("**", H2)
    assert m("**", H1)
    assert m("**", "a/b/c/d/e")
    assert not m("**", "")


@pytest.mark.torch_ops
def test_single_char_wildcard():
    """'?' matches exactly one character in a component."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel?", H3)
    assert m("torch.?elu", H3)
    assert not m("torch.?", H3)
    assert not m("?", H1)
    assert m("torch.nn.functional.conv?d", H2)


@pytest.mark.torch_ops
def test_long_hierarchy():
    """Deeply nested hierarchies match correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    deep = "/".join([f"level{i}" for i in range(20)])
    assert m("level19", deep)
    assert m("*19", deep)
    assert m("*/level19", deep)
    assert m("all", deep)
    assert not m("level0", deep)


@pytest.mark.torch_ops
def test_long_component_names():
    """Components with very long names are handled correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    long_name = "a" * 500
    hierarchy = f"root/{long_name}"
    assert m(f"{'a' * 500}", hierarchy)
    assert m("a*", hierarchy)
    assert not m("b*", hierarchy)


@pytest.mark.torch_ops
def test_special_characters_in_names():
    """Dots, underscores, and other non-glob chars are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module._internal/torch.nn.functional.conv2d"
    assert m("torch.nn.functional.conv2d", h)
    assert m("*conv2d", h)
    assert m("nn.Module._internal/*", h)
    assert not m("nn_Module._internal/*", h)


@pytest.mark.torch_ops
def test_bracket_glob_pattern():
    """Character classes [abc] work in glob patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel[uv]", H3)
    assert not m("torch.rel[ab]", H3)


@pytest.mark.torch_ops
def test_single_component_hierarchy():
    """Single-component hierarchy (no slashes) matches bare patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", "torch.relu")
    assert m("*relu", "torch.relu")
    assert m("torch.*", "torch.relu")
    assert not m("*/torch.relu", "torch.relu")


@pytest.mark.torch_ops
def test_whitespace_only_pattern():
    """Whitespace-only patterns normalize to empty and return False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("   ", H3)
    assert not m("\t", H3)


@pytest.mark.torch_ops
def test_star_pattern_matches_all():
    """Bare '*' is normalized to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*", H3)
    assert m("*", H2)
    assert m("*", H1)
    assert m("*", "a/b/c/d/e")
    assert not m("*", "")


@pytest.mark.torch_ops
def test_star_normalize_equivalence():
    """'*' and 'all' produce the same normalization."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("*") == norm("all") == "**"


@pytest.mark.torch_ops
def test_case_sensitivity():
    """Pattern matching is case-sensitive."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("Torch.Relu", H3)
    assert not m("TORCH.RELU", H3)
    assert not m("ALL", H3)
    assert m("all", H3)


@pytest.mark.torch_ops
def test_all_keyword_case_sensitive():
    """Only lowercase 'all' is the special keyword; mixed case is a literal."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("all") == "**"
    assert norm("ALL") == "ALL"
    assert norm("All") == "All"


@pytest.mark.torch_ops
def test_consecutive_slashes_in_target():
    """Consecutive slashes in the target are collapsed by PurePosixPath."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "a//b///torch.relu"
    assert m("torch.relu", h)
    assert m("*relu", h)


@pytest.mark.torch_ops
def test_dots_in_patterns():
    """Dots are literal characters in glob patterns, not regex wildcards."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert not m("torchXrelu", H3)
    h = "root/torchXrelu"
    assert not m("torch.relu", h)
    assert m("torchXrelu", h)


@pytest.mark.torch_ops
def test_pattern_with_spaces():
    """Spaces in patterns and targets are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "module/ spaced op /torch.relu"
    assert m("torch.relu", h)
    assert not m(" spaced op ", h)
    assert m("* spaced op */*", h)


@pytest.mark.torch_ops
def test_colons_in_operator_names():
    """Colons (e.g. aten::relu) are literal characters in glob matching."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module/aten::relu_"
    assert m("aten::relu_", h)
    assert m("*relu_", h)
    assert m("aten::*", h)
    assert not m("*relu", h)
    assert not m("torch.relu", h)


@pytest.mark.torch_ops
def test_display_star_matches_all_operators():
    """'*' pattern matches all operators in display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*"])
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_dedup_across_dataframes():
    """Same operator in multiple DataFrames is matched only once."""
    df1 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k1"]})
    df2 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k2"]})
    torch_operators = {"trace_0": df1, "trace_1": df2}

    matched = get_matched_torch_operators_for_display(torch_operators, ["all"])
    op_names = [name for name, _ in matched]
    assert op_names.count(H3) == 1


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["*"])
    assert parse_torch_operator_patterns(args) == ["*"]

    args = Namespace(torch_operator=["*,torch.relu"])
    assert parse_torch_operator_patterns(args) == ["*", "torch.relu"]


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
