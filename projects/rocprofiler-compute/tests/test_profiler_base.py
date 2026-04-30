# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse

import pytest

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.utils_exceptions import (
    ExecutableNotFoundError,
    NoScriptInCommandError,
    PythonScriptNotFoundError,
)


def _make_sanitize_args(remaining, torch_trace=False):
    """Build a minimal argparse.Namespace for sanitize() unit tests."""
    return argparse.Namespace(
        filter_blocks=[],
        set_selected=None,
        roof_only=False,
        path="/tmp/test_workload",
        no_native_tool=False,
        iteration_multiplexing=None,
        attach_pid=None,
        remaining=["--"] + remaining,
        torch_trace=torch_trace,
    )


def _setup_test_files(tmp_path, remaining, setup):
    """Create temporary files and substitute placeholders in remaining."""
    if setup == "script":
        script = tmp_path / "good_script.py"
        script.write_text("print('ok')\n")
        return [s.replace("{script}", str(script)) for s in remaining]
    elif setup == "exec_script":
        script = tmp_path / "main.py"
        script.write_text("print('ok')\n")
        script.chmod(0o755)
        return [s.replace("{exec_script}", str(script)) for s in remaining]
    elif setup == "binary":
        binary = tmp_path / "my_binary"
        binary.write_text("#!/bin/sh\necho hello\n")
        binary.chmod(0o755)
        return [s.replace("{binary}", str(binary)) for s in remaining]
    return remaining


# ---------------------------------------------------------------------------
# sanitize() with --torch-trace
# ---------------------------------------------------------------------------
@pytest.mark.torch_trace
@pytest.mark.parametrize(
    "remaining, expected_exception, setup",
    [
        # Should raise
        pytest.param(
            ["python3", "nonexistent_script_abc.py"],
            PythonScriptNotFoundError,
            None,
            id="missing_script",
        ),
        pytest.param(
            ["python3"],
            NoScriptInCommandError,
            None,
            id="bare_interpreter",
        ),
        pytest.param(
            ["python3", "-u", "-v"],
            NoScriptInCommandError,
            None,
            id="flags_only",
        ),
        pytest.param(
            ["python3", "-u", "nonexistent_script_abc.py"],
            PythonScriptNotFoundError,
            None,
            id="missing_script_after_flags",
        ),
        pytest.param(
            ["nonexistentpython3", "script.py"],
            ExecutableNotFoundError,
            None,
            id="nonexistent_executable",
        ),
        pytest.param(
            ["./no_such_binary"],
            ExecutableNotFoundError,
            None,
            id="nonexistent_binary",
        ),
        # Should not raise
        pytest.param(
            ["python3", "-c", "print(1)"],
            None,
            None,
            id="dash_c",
        ),
        pytest.param(
            ["python3", "-m", "json.tool", "--help"],
            None,
            None,
            id="dash_m",
        ),
        pytest.param(
            ["python3", "-u", "{script}"],
            None,
            "script",
            id="script_after_single_flag",
        ),
        pytest.param(
            ["python3", "-W", "ignore", "-u", "{script}"],
            None,
            "script",
            id="script_after_multi_flags",
        ),
        pytest.param(
            ["{exec_script}"],
            None,
            "exec_script",
            id="direct_py_script",
        ),
        pytest.param(
            ["{binary}"],
            None,
            "binary",
            id="non_python_binary",
        ),
    ],
)
def test_sanitize_torch_trace(tmp_path, remaining, expected_exception, setup):
    """Unit test: sanitize() behavior with --torch-trace enabled."""
    remaining = _setup_test_files(tmp_path, remaining, setup)
    args = _make_sanitize_args(remaining, torch_trace=True)
    profiler = RocProfCompute_Base(args, profiler_mode="rocprofiler-sdk", soc=None)
    if expected_exception:
        with pytest.raises(expected_exception):
            profiler.sanitize()
    else:
        profiler.sanitize()


# ---------------------------------------------------------------------------
# sanitize() without --torch-trace
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "remaining, expected_exception, setup",
    [
        # Should raise
        pytest.param(
            ["python3"],
            NoScriptInCommandError,
            None,
            id="bare_interpreter",
        ),
        pytest.param(
            ["python3", "-u", "-v"],
            NoScriptInCommandError,
            None,
            id="flags_only",
        ),
        pytest.param(
            ["nonexistentpython3", "script.py"],
            ExecutableNotFoundError,
            None,
            id="nonexistent_executable",
        ),
        pytest.param(
            ["./no_such_binary"],
            ExecutableNotFoundError,
            None,
            id="nonexistent_binary",
        ),
        # Should not raise
        pytest.param(
            ["python3", "-c", "print(1)"],
            None,
            None,
            id="dash_c",
        ),
        pytest.param(
            ["python3", "-m", "json.tool", "--help"],
            None,
            None,
            id="dash_m",
        ),
        pytest.param(
            ["python3", "nonexistent_script_abc.py"],
            None,
            None,
            id="missing_script",
        ),
        pytest.param(
            ["python3", "-u", "nonexistent_script_abc.py"],
            None,
            None,
            id="missing_script_after_flags",
        ),
        pytest.param(
            ["python3", "-u", "{script}"],
            None,
            "script",
            id="script_after_single_flag",
        ),
        pytest.param(
            ["python3", "-W", "ignore", "-u", "{script}"],
            None,
            "script",
            id="script_after_multi_flags",
        ),
    ],
)
def test_sanitize_no_torch_trace(tmp_path, remaining, expected_exception, setup):
    """Unit test: sanitize() behavior without --torch-trace."""
    remaining = _setup_test_files(tmp_path, remaining, setup)
    args = _make_sanitize_args(remaining, torch_trace=False)
    profiler = RocProfCompute_Base(args, profiler_mode="rocprofiler-sdk", soc=None)
    if expected_exception:
        with pytest.raises(expected_exception):
            profiler.sanitize()
    else:
        profiler.sanitize()
