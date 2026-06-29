# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from unittest.mock import Mock

import pytest
from common import patch_console

from pc_sampling.pc_sampling_profile import PCSamplingProfile

MODULE = "pc_sampling.pc_sampling_profile"


class MockArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_pc_sampling_profile(method, interval, workload_dir, profiler):
    """Build a PCSamplingProfile with a minimal args namespace for launch tests."""
    return PCSamplingProfile(
        args=MockArgs(
            pc_sampling_method=method,
            pc_sampling_interval=interval,
            filter_blocks=["21"],
        ),
        profiler=profiler,
        workload_dir=workload_dir,
    )


def test_pc_sampling_profile_sdk_forwards_env_and_ld_preload(tmp_path, monkeypatch):
    """sdk non-live-attach launch forwards LD_PRELOAD plus the PC sampling
    env vars (method/interval and the host_trap->time unit mapping) into the
    subprocess env on success."""
    method = "host_trap"
    interval = 1000
    workload_dir = str(tmp_path)
    options = {"APP_CMD": "my_app --arg"}

    expected_tool_path = str(
        tmp_path / "rocm_sdk" / "lib" / "rocprofiler-sdk" / "librocprofiler-sdk-tool.so"
    )
    options["LD_PRELOAD"] = expected_tool_path

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    assert mock_capture.called
    called_env = mock_capture.call_args.kwargs.get("new_env", {})

    assert called_env["LD_PRELOAD"] == expected_tool_path
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "time"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)
    assert called_env["ROCPROF_OUTPUT_PATH"] == workload_dir
    assert called_env["ROCPROF_OUTPUT_FILE_NAME"] == "ps_file"

    mock_error.assert_not_called()


def test_pc_sampling_profile_env_log_excludes_user_env(tmp_path, monkeypatch):
    """The sdk launch must log only profiler-added env vars, never the user's
    full environment, to avoid leaking secrets into shared workload logs."""
    monkeypatch.setenv("LEAK_CANARY_TOKEN", "SHOULD_NOT_APPEAR")
    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    _mocks = patch_console(monkeypatch, MODULE, "debug", "error")
    mock_debug, mock_error = _mocks["debug"], _mocks["error"]

    profiler = _make_pc_sampling_profile(
        "host_trap", 1000, str(tmp_path), "rocprofiler-sdk"
    )
    profiler._launch({"APP_CMD": "my_app"})

    logs = [str(call.args[0]) for call in mock_debug.call_args_list]
    env_log_lines = [m for m in logs if "env vars" in m]
    assert env_log_lines
    assert any("ROCPROF_PC_SAMPLING_METHOD" in m for m in env_log_lines)
    assert not any("SHOULD_NOT_APPEAR" in m for m in logs)

    mock_error.assert_not_called()


def test_pc_sampling_profile_sdk_stochastic_unit_is_cycles(tmp_path, monkeypatch):
    """A non-host_trap (stochastic) method maps to the ``cycles`` sampling unit
    in the forwarded sdk env."""
    method = "stochastic"
    interval = 5000
    workload_dir = str(tmp_path)
    options = {"APP_CMD": "my_app"}

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    called_env = mock_capture.call_args.kwargs.get("new_env", {})
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "cycles"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)

    mock_error.assert_not_called()


def test_pc_sampling_profile_subprocess_fails(tmp_path, monkeypatch):
    """
    Edge Case: The capture_subprocess_output returns success=False.
    This should trigger the console_error("PC sampling failed.").
    """
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=mock_console_error)

    method = "stochastic"
    interval = 5000
    workload_dir = str(tmp_path)
    options = ["another_app"]

    profiler = _make_pc_sampling_profile(method, interval, workload_dir, "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(options)

    mock_capture.assert_not_called()
    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]

    mock_capture.reset_mock()
    console_error_calls.clear()
    options = {"APP_CMD": "another_app"}
    sdk_lib_dir = tmp_path / "rocm_sdk_fail" / "lib"
    sdk_lib_dir.mkdir(parents=True, exist_ok=True)
    rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
    Path(rocprofiler_sdk_tool_path_sdk).touch()

    tool_dir = sdk_lib_dir / "rocprofiler-sdk"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "librocprofiler-sdk-tool.so").touch()

    mock_capture.return_value = (False, "Error output from SDK subprocess")

    profiler = _make_pc_sampling_profile(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(options)

    mock_capture.assert_called_once()
    assert console_error_calls == ["PC sampling failed."]


def test_pc_sampling_profile_empty_appcmd(tmp_path, monkeypatch):
    """
    Edge Case: The appcmd is an empty string.
    The function should still attempt to run it. The behavior of
    capture_subprocess_output with an empty command is external to this function.
    """
    method = "host_trap"
    interval = 100
    workload_dir = str(tmp_path)
    options = ["--"]

    mock_capture = Mock(return_value=(True, "Output with empty appcmd"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(method, interval, workload_dir, "rocprofv3")
    profiler._launch(options)

    assert mock_capture.called
    options_list = mock_capture.call_args[0][0]
    assert options_list[-1] == "--"
    mock_error.assert_not_called()

    mock_capture.reset_mock()
    mock_error.reset_mock()
    sdk_lib_dir = tmp_path / "rocm_sdk_empty" / "lib"
    sdk_lib_dir.mkdir(parents=True, exist_ok=True)
    rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
    Path(rocprofiler_sdk_tool_path_sdk).touch()
    tool_dir = sdk_lib_dir / "rocprofiler-sdk"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "librocprofiler-sdk-tool.so").touch()

    mock_capture.return_value = (True, "Output with empty appcmd SDK")
    options = {"APP_CMD": ""}

    profiler = _make_pc_sampling_profile(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    assert mock_capture.called
    assert mock_capture.call_args[0][0] == ""
    mock_error.assert_not_called()


def test_pc_sampling_profile_multiarg_appcmd(tmp_path, monkeypatch):
    """All arguments after '--' in profiler_options must appear
    in the subprocess call."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_tool")
    method = "host_trap"
    interval = 100
    workload_dir = str(tmp_path)
    options = ["--kernel-trace", "--", "./myapp", "arg1", "arg2"]

    mock_capture = Mock(return_value=(True, "Success"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(method, interval, workload_dir, "rocprofv3")
    profiler._launch(options)

    assert mock_capture.called
    options_list = mock_capture.call_args[0][0]
    assert options_list[0] == "rocprof_cli_tool"
    separator_index = options_list.index("--")
    assert options_list[separator_index:] == ["--", "./myapp", "arg1", "arg2"]
    mock_error.assert_not_called()


def test_pc_sampling_profile_is_requested():
    workload_dir = "/tmp"
    for blocks in (["21"], ["pc_sampling"], ["2", "21"]):
        profiler = PCSamplingProfile(
            args=MockArgs(filter_blocks=blocks),
            profiler="rocprofv3",
            workload_dir=workload_dir,
        )
        assert profiler.is_requested() is True

    profiler = PCSamplingProfile(
        args=MockArgs(filter_blocks=["2"]),
        profiler="rocprofv3",
        workload_dir=workload_dir,
    )
    assert profiler.is_requested() is False


def test_pc_sampling_profile_cleanup_stale_output_removes_dir(tmp_path, monkeypatch):
    """Exclusive sdk run with a dict ROCPROF_OUTPUT_PATH removes the stale dir."""
    patch_console(monkeypatch, MODULE, "debug")
    stale = tmp_path / "out" / "pmc_1"
    stale.mkdir(parents=True, exist_ok=True)
    options = {"ROCPROF_OUTPUT_PATH": str(stale)}

    profiler = PCSamplingProfile(
        args=MockArgs(filter_blocks=["21"]),
        profiler="rocprofiler-sdk",
        workload_dir=str(tmp_path),
    )
    profiler._cleanup_stale_output(options)

    assert not stale.exists()


def test_pc_sampling_profile_cleanup_stale_output_noop_cases(tmp_path, monkeypatch):
    """Cleanup is a no-op outside the exclusive-sdk-dict-with-key case."""
    patch_console(monkeypatch, MODULE, "debug")
    sdk_args = MockArgs(filter_blocks=["21"])

    # Non-sdk profiler: no removal even when exclusive.
    stale = tmp_path / "non_sdk"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfile(
        args=sdk_args, profiler="rocprofv3", workload_dir=str(tmp_path)
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # Non-exclusive: no removal.
    stale = tmp_path / "mixed"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfile(
        args=MockArgs(filter_blocks=["2", "21"]),
        profiler="rocprofiler-sdk",
        workload_dir=str(tmp_path),
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # List options (v3): no removal.
    stale = tmp_path / "list_opts"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfile(
        args=sdk_args, profiler="rocprofiler-sdk", workload_dir=str(tmp_path)
    )._cleanup_stale_output(["--kernel-trace"])
    assert stale.exists()

    # Missing key: no error, no removal.
    PCSamplingProfile(
        args=sdk_args, profiler="rocprofiler-sdk", workload_dir=str(tmp_path)
    )._cleanup_stale_output({"APP_CMD": "app"})


def test_pc_sampling_profile_v3_live_attach(tmp_path, monkeypatch):
    """v3 live-attach appends --pid and --attach-duration-msec, no APP_CMD '--'."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_tool")
    options = ["--pid", "1234", "--attach-duration-msec", "500"]

    mock_capture = Mock(return_value=(True, "Success"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile("host_trap", 100, str(tmp_path), "rocprofv3")
    profiler._launch(options)

    assert mock_capture.called
    options_list = mock_capture.call_args[0][0]
    pid_idx = options_list.index("--pid")
    assert options_list[pid_idx + 1] == "1234"
    dur_idx = options_list.index("--attach-duration-msec")
    assert options_list[dur_idx + 1] == "500"
    assert "--" not in options_list
    mock_error.assert_not_called()


def test_pc_sampling_profile_v3_live_attach_missing_pid_value(tmp_path, monkeypatch):
    """v3 live-attach with --pid but no trailing value triggers console_error."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=mock_console_error)

    profiler = _make_pc_sampling_profile("host_trap", 100, str(tmp_path), "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(["--pid"])

    assert console_error_calls == [
        "--pid or --attach-duration-msec option not found in "
        "profiler arguments for live attach mode"
    ]
    mock_capture.assert_not_called()


def test_pc_sampling_profile_sdk_live_attach(tmp_path, monkeypatch):
    """sdk live-attach calls perform_attach_detach and returns before launching."""
    options = {
        "ROCPROF_ATTACH_PID": "1234",
        "ROCPROF_ATTACH_LIBRARY": "lib.so",
    }

    mock_capture = Mock()
    mock_attach = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.perform_attach_detach", mock_attach)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )
    profiler._launch(options)

    mock_attach.assert_called_once()
    mock_capture.assert_not_called()
    mock_error.assert_not_called()


def test_pc_sampling_profile_sdk_missing_app_cmd(tmp_path, monkeypatch):
    """sdk non-live-attach without APP_CMD errors before launching."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=mock_console_error)

    profiler = _make_pc_sampling_profile(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch({"LD_PRELOAD": "x"})

    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]
    mock_capture.assert_not_called()


def test_pc_sampling_profile_v3_missing_separator(tmp_path, monkeypatch):
    """v3 non-live-attach without a '--' separator errors before launching."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=mock_console_error)

    profiler = _make_pc_sampling_profile("host_trap", 100, str(tmp_path), "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(["--something"])

    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]
    mock_capture.assert_not_called()


def test_pc_sampling_profile_run_cleanup_before_launch(tmp_path, monkeypatch):
    """run() removes stale output before reaching the subprocess launch seam, and
    emits the run header and a timing debug."""
    stale = tmp_path / "out" / "pmc_1"
    stale.mkdir(parents=True, exist_ok=True)
    options = {"ROCPROF_OUTPUT_PATH": str(stale), "APP_CMD": "my_app"}

    profiler = _make_pc_sampling_profile(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )

    stale_existed_at_launch = []

    def record(*args, **kwargs):
        # Cleanup must already have run by the time we launch the subprocess.
        stale_existed_at_launch.append(stale.exists())
        return (True, "")

    mock_capture = Mock(side_effect=record)
    mock_log = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.console_log", mock_log)
    _mocks = patch_console(monkeypatch, MODULE, "debug", "error")
    mock_debug, mock_error = _mocks["debug"], _mocks["error"]

    profiler.run(options, prior_run_count=0)

    assert stale_existed_at_launch == [False]
    assert not stale.exists()
    mock_error.assert_not_called()

    mock_log.assert_any_call("[Run 1/1][PC sampling profile run]")
    assert any(
        call.args and call.args[0] == "profiling" for call in mock_debug.call_args_list
    )
