# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from unittest.mock import patch

import common
import pytest

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_profile.profiler_rocprofiler_sdk import rocprofiler_sdk_profiler
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
        output_directory="/tmp/test_workload",
        no_native_tool=False,
        iteration_multiplexing=None,
        attach_pid=None,
        attach_duration_msec=None,
        spatial_multiplexing=None,
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


# ---------------------------------------------------------------------------
# get_profiler_options(): live-attach library resolution with fallback
# ---------------------------------------------------------------------------
def test_attach_library_resolution_with_fallback():
    """Unit test: attach branch picks new lib first, falls back to old, errors if
    neither exists. resolve_rocm_library_path is mocked so the actual library
    locations are controlled by the test, independent of the configured tool path."""
    output_dir = Path(common.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        remaining="-- /bin/true",
        rocprofiler_sdk_tool_path="/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        format_rocprof_output="csv",
        output_directory=str(output_dir),
        iteration_multiplexing=None,
        attach_pid=12345,
        attach_duration_msec=None,
        kokkos_trace=False,
        kernel=None,
        dispatch=None,
        torch_trace=False,
    )
    profiler = rocprofiler_sdk_profiler(args, profiler_mode="rocprofiler-sdk", soc=None)
    resolve_target = (
        "rocprof_compute_profile.profiler_rocprofiler_sdk.resolve_rocm_library_path"
    )
    new_lib = output_dir / "librocprofiler-sdk-rocattach.so"
    old_lib = output_dir / "librocprofv3-attach.so"

    # Case 1: new library present -> selected, fallback lookup never happens.
    new_lib.write_text("")
    with patch(
        resolve_target, side_effect=[str(new_lib), str(old_lib)]
    ) as mock_resolve:
        options = profiler.get_profiler_options()
    assert options["ROCPROF_ATTACH_LIBRARY"] == str(new_lib)
    assert mock_resolve.call_count == 1

    # Case 2: only old library present -> falls back to it.
    new_lib.unlink()
    old_lib.write_text("")
    with patch(
        resolve_target, side_effect=[str(new_lib), str(old_lib)]
    ) as mock_resolve:
        options = profiler.get_profiler_options()
    assert options["ROCPROF_ATTACH_LIBRARY"] == str(old_lib)
    assert mock_resolve.call_count == 2

    # Case 3: neither library present -> console_error exits the process.
    old_lib.unlink()
    with patch(resolve_target, side_effect=[str(new_lib), str(old_lib)]):
        with pytest.raises(SystemExit):
            profiler.get_profiler_options()


# ---------------------------------------------------------------------------
# get_profiler_options(): ROCPROF_SHELL_TARGET injection
# ---------------------------------------------------------------------------
def _make_sdk_profiler_options_args(
    tmp_path: Path, remaining: str
) -> argparse.Namespace:
    """Build a minimal Namespace for get_profiler_options() unit tests."""
    return argparse.Namespace(
        remaining=remaining,
        rocprofiler_sdk_tool_path="/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so",
        format_rocprof_output="csv",
        output_directory=str(tmp_path),
        iteration_multiplexing=None,
        attach_pid=None,
        attach_duration_msec=None,
        kokkos_trace=False,
        kernel=None,
        dispatch=None,
        torch_trace=False,
    )


@pytest.mark.parametrize(
    "remaining",
    [
        pytest.param("bash script.sh", id="bash_with_script"),
        pytest.param("./run.sh", id="relative_shell_script"),
        pytest.param("/usr/bin/zsh -c true", id="absolute_zsh_dash_c"),
    ],
)
def test_shell_target_set_with_native_tool(tmp_path, remaining, mock_etc_shells):
    """ROCPROF_SHELL_TARGET=1 when native tool is used and command is shell-like."""
    args = _make_sdk_profiler_options_args(tmp_path, remaining)
    profiler = rocprofiler_sdk_profiler(args, profiler_mode="rocprofiler-sdk", soc=None)
    options = profiler.get_profiler_options(native_tool_path="/path/to/native_tool.so")
    assert options.get("ROCPROF_SHELL_TARGET") == "1"


@pytest.mark.parametrize(
    "remaining",
    [
        pytest.param("/bin/true", id="non_shell_binary"),
        pytest.param("python3 app.py", id="python_script"),
        pytest.param("./vcopy -n 1024", id="custom_binary"),
    ],
)
def test_shell_target_absent_with_native_tool_non_shell(
    tmp_path, remaining, mock_etc_shells
):
    """ROCPROF_SHELL_TARGET is omitted when the workload is not a shell target."""
    args = _make_sdk_profiler_options_args(tmp_path, remaining)
    profiler = rocprofiler_sdk_profiler(args, profiler_mode="rocprofiler-sdk", soc=None)
    options = profiler.get_profiler_options(native_tool_path="/path/to/native_tool.so")
    assert "ROCPROF_SHELL_TARGET" not in options


def test_shell_target_absent_without_native_tool(tmp_path, mock_etc_shells):
    """Even with a shell-like command, no flag is set without a native tool."""
    args = _make_sdk_profiler_options_args(tmp_path, "bash script.sh")
    profiler = rocprofiler_sdk_profiler(args, profiler_mode="rocprofiler-sdk", soc=None)
    options = profiler.get_profiler_options()
    assert "ROCPROF_SHELL_TARGET" not in options
