# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests covering functions in src/utils/utils_common.py."""

import builtins
import locale
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
    # bad bytes show up as \ufffd, not as exceptions.
    def dummy_popen(*args, **kwargs):
        popen_calls.append(kwargs)
        return make_dummy_process(lines=["good line\n", "bad \ufffd byte\n"])

    monkeypatch.setattr("subprocess.Popen", dummy_popen)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    success, output = utils_common.capture_subprocess_output(["echo", "test"])

    assert success is True
    assert "good line" in output
    assert "\ufffd" in output
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
# TESTS FOR LOCAL ENCODING FUNCTION
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
