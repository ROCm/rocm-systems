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


def _make_pc_sampling_profile(
    workload_dir,
    *,
    method="host_trap",
    interval=1000,
    profiler="rocprofiler-sdk",
    native_tool_path=None,
    filter_blocks=("21",),
):
    """Build a PCSamplingProfile; defaults target the sdk launch/cleanup case."""
    return PCSamplingProfile(
        args=MockArgs(
            pc_sampling_method=method,
            pc_sampling_interval=interval,
            filter_blocks=list(filter_blocks),
        ),
        profiler=profiler,
        workload_dir=workload_dir,
        native_tool_path=native_tool_path,
    )


def _record_and_raise_console_error():
    """Return a console_error stub that records messages and raises on exit."""
    calls = []

    def stub(msg, exit=True):
        calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    return stub, calls


def test_native_backend_dispatch_sets_env_and_ld_preload(tmp_path, monkeypatch):
    """Native backend: .so on LD_PRELOAD, PC sampling env set, no sdk output vars."""
    method = "host_trap"
    interval = 1000
    workload_dir = str(tmp_path)
    native_tool_path = "/n/native.so"
    options = {"APP_CMD": "my_app --arg"}

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        workload_dir,
        method=method,
        interval=interval,
        native_tool_path=native_tool_path,
    )
    profiler._launch(options)

    assert mock_capture.called
    called_env = mock_capture.call_args.kwargs.get("new_env", {})

    assert native_tool_path in called_env["LD_PRELOAD"].split(":")
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "time"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)
    assert called_env["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] == "1"
    assert called_env["ROCPROF_COUNTER_COLLECTION"] == "0"
    assert "ROCPROF_OUTPUT_FILE_NAME" not in called_env
    assert "ROCPROF_OUTPUT_FORMAT" not in called_env

    mock_error.assert_not_called()


def test_native_backend_preserves_existing_ld_preload(tmp_path, monkeypatch):
    """The native .so is appended after the user's own LD_PRELOAD."""
    native_tool_path = "/n/native.so"
    options = {"APP_CMD": "my_app"}
    monkeypatch.setenv("LD_PRELOAD", "/user/existing.so")

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path), native_tool_path=native_tool_path
    )
    profiler._launch(options)

    called_env = mock_capture.call_args.kwargs.get("new_env", {})
    assert called_env["LD_PRELOAD"] == "/user/existing.so:/n/native.so"
    mock_error.assert_not_called()


@pytest.mark.parametrize("native_tool_path", [None, "/n/native.so"])
def test_stochastic_method_maps_to_cycles_unit(tmp_path, monkeypatch, native_tool_path):
    """A stochastic method maps to the cycles unit on both backends."""
    method = "stochastic"
    interval = 5000
    options = {"APP_CMD": "my_app"}

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path),
        method=method,
        interval=interval,
        native_tool_path=native_tool_path,
    )
    profiler._launch(options)

    called_env = mock_capture.call_args.kwargs.get("new_env", {})
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "cycles"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)
    mock_error.assert_not_called()


@pytest.mark.parametrize(
    "native_tool_path", [None, "/n/native.so"], ids=["sdk", "native"]
)
def test_subprocess_failure_errors(tmp_path, monkeypatch, native_tool_path):
    """A failed subprocess reports the standard PC sampling failure on both backends."""
    mock_capture = Mock(return_value=(False, "Error output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path), native_tool_path=native_tool_path
    )
    profiler._launch({"APP_CMD": "my_app"})

    assert mock_error.called
    assert "PC sampling failed." in mock_error.call_args.args[0]


@pytest.mark.parametrize(
    "native_tool_path", [None, "/n/native.so"], ids=["sdk", "native"]
)
def test_env_log_excludes_user_env(tmp_path, monkeypatch, native_tool_path):
    """The debug env log records the PC sampling vars but never the user's env."""
    monkeypatch.setenv("LEAK_CANARY_TOKEN", "SHOULD_NOT_APPEAR")
    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    _mocks = patch_console(monkeypatch, MODULE, "debug", "error")
    mock_debug, mock_error = _mocks["debug"], _mocks["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path), native_tool_path=native_tool_path
    )
    profiler._launch({"APP_CMD": "my_app"})

    logs = [str(call.args[0]) for call in mock_debug.call_args_list]
    env_log_lines = [m for m in logs if "env vars" in m]
    assert env_log_lines
    assert any("ROCPROF_PC_SAMPLING_METHOD" in m for m in env_log_lines)
    assert not any("SHOULD_NOT_APPEAR" in m for m in logs)
    mock_error.assert_not_called()


@pytest.mark.parametrize("native", [True, False])
def test_live_attach_performs_attach_detach(tmp_path, monkeypatch, native):
    """Live-attach calls perform_attach_detach instead of launching, both backends."""
    options = {
        "ROCPROF_ATTACH_PID": "1234",
        "ROCPROF_ATTACH_LIBRARY": "lib.so",
    }

    mock_capture = Mock()
    mock_attach = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.perform_attach_detach", mock_attach)
    if native:
        monkeypatch.setattr(f"{MODULE}.is_live_attach", Mock(return_value=True))
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path),
        interval=100,
        native_tool_path="/n/native.so" if native else None,
    )
    profiler._launch(options)

    mock_attach.assert_called_once()
    mock_capture.assert_not_called()
    mock_error.assert_not_called()


def test_no_native_tool_path_uses_sdk_standard(tmp_path, monkeypatch):
    """native_tool_path=None uses the sdk standard backend (no native .so forced on)."""
    native_tool_path = "/n/native.so"
    options = {"APP_CMD": "my_app"}

    mock_capture = Mock(return_value=(True, "Success output"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(str(tmp_path), native_tool_path=None)
    profiler._launch(options)

    called_env = mock_capture.call_args.kwargs.get("new_env", {})
    assert called_env["ROCPROF_OUTPUT_FILE_NAME"] == "ps_file"
    assert native_tool_path not in called_env.get("LD_PRELOAD", "").split(":")
    mock_error.assert_not_called()


@pytest.mark.parametrize(
    "filter_blocks, native_tool_path, expect_removed",
    [
        (["21"], "/n/native.so", True),
        (["2", "21"], "/n/native.so", True),
        (["2", "21"], None, False),
    ],
)
def test_cleanup_native_artifacts(
    tmp_path, monkeypatch, filter_blocks, native_tool_path, expect_removed
):
    """Native artifacts are purged iff native_tool_path is set."""
    patch_console(monkeypatch, MODULE, "debug")

    results = tmp_path / "1234_ps_file_results.json"
    results.touch()
    code_obj_info = tmp_path / "1234_code_obj_info.json"
    code_obj_info.touch()
    sources_dir = tmp_path / "code_obj_sources"
    sources_dir.mkdir()
    (sources_dir / "src.txt").touch()

    profiler = _make_pc_sampling_profile(
        str(tmp_path),
        native_tool_path=native_tool_path,
        filter_blocks=filter_blocks,
    )
    profiler._cleanup_stale_output({"APP_CMD": "my_app"})

    assert results.exists() is not expect_removed
    assert code_obj_info.exists() is not expect_removed
    assert sources_dir.exists() is not expect_removed


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

    profiler = _make_pc_sampling_profile(workload_dir, method=method, interval=interval)
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

    profiler = _make_pc_sampling_profile(
        workload_dir, method=method, interval=interval, profiler="rocprofv3"
    )
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

    profiler = _make_pc_sampling_profile(workload_dir, method=method, interval=interval)
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

    profiler = _make_pc_sampling_profile(
        workload_dir, method=method, interval=interval, profiler="rocprofv3"
    )
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
        profiler = _make_pc_sampling_profile(
            workload_dir, profiler="rocprofv3", filter_blocks=blocks
        )
        assert profiler.is_requested() is True

    profiler = _make_pc_sampling_profile(
        workload_dir, profiler="rocprofv3", filter_blocks=["2"]
    )
    assert profiler.is_requested() is False


def test_pc_sampling_profile_cleanup_stale_output_removes_dir(tmp_path, monkeypatch):
    """Exclusive sdk run with a dict ROCPROF_OUTPUT_PATH removes the stale dir."""
    patch_console(monkeypatch, MODULE, "debug")
    stale = tmp_path / "out" / "pmc_1"
    stale.mkdir(parents=True, exist_ok=True)
    options = {"ROCPROF_OUTPUT_PATH": str(stale)}

    profiler = _make_pc_sampling_profile(str(tmp_path))
    profiler._cleanup_stale_output(options)

    assert not stale.exists()


def test_pc_sampling_profile_cleanup_stale_output_noop_cases(tmp_path, monkeypatch):
    """Cleanup is a no-op outside the exclusive-sdk-dict-with-key case."""
    patch_console(monkeypatch, MODULE, "debug")

    # Non-sdk profiler: no removal even when exclusive.
    stale = tmp_path / "non_sdk"
    stale.mkdir(parents=True, exist_ok=True)
    _make_pc_sampling_profile(
        str(tmp_path), profiler="rocprofv3"
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # Non-exclusive: no removal.
    stale = tmp_path / "mixed"
    stale.mkdir(parents=True, exist_ok=True)
    _make_pc_sampling_profile(
        str(tmp_path), filter_blocks=["2", "21"]
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # List options (v3): no removal.
    stale = tmp_path / "list_opts"
    stale.mkdir(parents=True, exist_ok=True)
    _make_pc_sampling_profile(str(tmp_path))._cleanup_stale_output(["--kernel-trace"])
    assert stale.exists()

    # Missing key: no error, no removal.
    _make_pc_sampling_profile(str(tmp_path))._cleanup_stale_output({"APP_CMD": "app"})


def test_pc_sampling_profile_v3_live_attach(tmp_path, monkeypatch):
    """v3 live-attach appends attach args and no APP_CMD '--'."""
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_tool")
    options = ["--pid", "1234", "--attach-duration-msec", "500"]

    mock_capture = Mock(return_value=(True, "Success"))
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path), interval=100, profiler="rocprofv3"
    )
    profiler._launch(options)

    assert mock_capture.called
    options_list = mock_capture.call_args[0][0]
    assert "--attach-sync-output" in options_list
    pid_idx = options_list.index("--pid")
    assert options_list[pid_idx + 1] == "1234"
    dur_idx = options_list.index("--attach-duration-msec")
    assert options_list[dur_idx + 1] == "500"
    assert "--" not in options_list
    mock_error.assert_not_called()


def test_pc_sampling_profile_v3_live_attach_missing_pid_value(tmp_path, monkeypatch):
    """v3 live-attach with --pid but no trailing value triggers console_error."""
    stub, console_error_calls = _record_and_raise_console_error()

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=stub)

    profiler = _make_pc_sampling_profile(
        str(tmp_path), interval=100, profiler="rocprofv3"
    )
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
        "ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC": "1",
    }

    mock_capture = Mock()
    mock_attach = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    monkeypatch.setattr(f"{MODULE}.perform_attach_detach", mock_attach)
    mock_error = patch_console(monkeypatch, MODULE, "debug", "error")["error"]

    profiler = _make_pc_sampling_profile(
        str(tmp_path), method="host_trap", interval=100, profiler="rocprofiler-sdk"
    )
    profiler._launch(options)

    mock_attach.assert_called_once()
    new_env, attach_options = mock_attach.call_args.args
    assert new_env["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    assert attach_options["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    mock_capture.assert_not_called()
    mock_error.assert_not_called()


def test_pc_sampling_profile_sdk_missing_app_cmd(tmp_path, monkeypatch):
    """sdk non-live-attach without APP_CMD errors before launching."""
    stub, console_error_calls = _record_and_raise_console_error()

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=stub)

    profiler = _make_pc_sampling_profile(str(tmp_path), interval=100)
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch({"LD_PRELOAD": "x"})

    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]
    mock_capture.assert_not_called()


def test_pc_sampling_profile_v3_missing_separator(tmp_path, monkeypatch):
    """v3 non-live-attach without a '--' separator errors before launching."""
    stub, console_error_calls = _record_and_raise_console_error()

    mock_capture = Mock()
    monkeypatch.setattr(f"{MODULE}.capture_subprocess_output", mock_capture)
    patch_console(monkeypatch, MODULE, "debug", "error", error=stub)

    profiler = _make_pc_sampling_profile(
        str(tmp_path), interval=100, profiler="rocprofv3"
    )
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

    profiler = _make_pc_sampling_profile(str(tmp_path), interval=100)

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
