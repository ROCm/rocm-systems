# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import common
import pytest

from rocprof_compute_profile.pc_sampling_profiler import PCSamplingProfiler
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_profile.profiler_rocprofiler_sdk import rocprofiler_sdk_profiler

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1


class MockArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _require_pc_sampling_gpu(monkeypatch, is_stochastic=False):
    """Skip when no supported GPU is present, apply SoC-specific PC sampling
    skips, then select the rocprofiler-sdk profiler for the run.

    Used only by the GPU integration tests below; the mock-based unit tests run
    everywhere and must not depend on hardware or the ROCPROF environment. The
    ROCPROF override is scoped to the calling test via monkeypatch so it does
    not leak into unrelated tests.
    """
    _, soc = common.gpu_soc()
    if not soc:
        pytest.skip("GPU not supported")
    common.skip_unsupported_pc_sampling_soc(is_stochastic=is_stochastic)
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")


PC_SAMPLING_HOST_TRAP_FILES = sorted([
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_host_trap.csv",
    "ps_file_results.json",
    "sysinfo.csv",
])

PC_SAMPLING_STOCHASTIC_FILES = sorted([
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_stochastic.csv",
    "ps_file_results.json",
    "sysinfo.csv",
])


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return "Given PC sampling configuration is not supported" in output


def _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir):
    if is_pc_sampling_not_supported(f"{stdout}\n{stderr}"):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    _require_pc_sampling_gpu(monkeypatch)

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    _require_pc_sampling_gpu(monkeypatch, is_stochastic=True)

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
        "--pc-sampling-interval",
        "1048576",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_STOCHASTIC_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    _require_pc_sampling_gpu(monkeypatch)

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" not in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_warning_pc_sampling_with_counters(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a multi-rank warning is printed when running with --block 21
    and another block (PC sampling with counters mode requires multiple passes)
    with multi-rank.
    """
    _require_pc_sampling_gpu(monkeypatch)

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_profile_then_analyze(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    End-to-end: profile with PC sampling (host_trap), then
    run analysis on the profiling output.
    """
    _require_pc_sampling_gpu(monkeypatch)

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    workload_path = Path(workload_dir)

    kernel_top_csv = workload_path / "pmc_kernel_top.csv"
    assert kernel_top_csv.exists()
    kernel_top_header = kernel_top_csv.read_text().splitlines()[0]
    assert "Kernel_Name" in kernel_top_header
    assert "Count" in kernel_top_header
    assert "Percent" in kernel_top_header

    dispatch_info_csv = workload_path / "pmc_dispatch_info.csv"
    assert dispatch_info_csv.exists()
    dispatch_info_header = dispatch_info_csv.read_text().splitlines()[0]
    assert "Dispatch_ID" in dispatch_info_header
    assert "Kernel_Name" in dispatch_info_header
    assert "GPU_ID" in dispatch_info_header

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out
    assert "21. PC Sampling" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_with_sol_block(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that PC sampling works with --block 21 and --block 2
    (PC sampling with counter collection)
    """
    _require_pc_sampling_gpu(monkeypatch)

    options = [
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    assert common.check_file_pattern("- '21'", f"{workload_dir}/profiling_config.yaml")
    assert common.check_file_pattern("- '2'", f"{workload_dir}/profiling_config.yaml")

    common.clean_output_dir(config["cleanup"], workload_dir)


# ===================================================================
# Test PCSamplingProfiler class (mock-based unit tests, no GPU required)
# ===================================================================


def _make_pc_sampling_profiler(method, interval, workload_dir, profiler):
    """Build a PCSamplingProfiler with a minimal args namespace for launch tests."""
    return PCSamplingProfiler(
        args=MockArgs(
            pc_sampling_method=method,
            pc_sampling_interval=interval,
            filter_blocks=["21"],
        ),
        profiler=profiler,
        workload_dir=workload_dir,
    )


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_sdk_forwards_env_and_ld_preload(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
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

    mock_capture_subprocess.return_value = (True, "Success output")

    profiler = _make_pc_sampling_profiler(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    assert mock_capture_subprocess.called
    call_args = mock_capture_subprocess.call_args
    called_env = call_args.kwargs.get("new_env", {})

    assert called_env["LD_PRELOAD"] == expected_tool_path
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "time"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)
    assert called_env["ROCPROF_OUTPUT_PATH"] == workload_dir
    assert called_env["ROCPROF_OUTPUT_FILE_NAME"] == "ps_file"

    mock_console_error.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_sdk_stochastic_unit_is_cycles(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """A non-host_trap (stochastic) method maps to the ``cycles`` sampling unit
    in the forwarded sdk env."""
    method = "stochastic"
    interval = 5000
    workload_dir = str(tmp_path)
    options = {"APP_CMD": "my_app"}

    mock_capture_subprocess.return_value = (True, "Success output")

    profiler = _make_pc_sampling_profiler(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    called_env = mock_capture_subprocess.call_args.kwargs.get("new_env", {})
    assert called_env["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert called_env["ROCPROF_PC_SAMPLING_UNIT"] == "cycles"
    assert called_env["ROCPROF_PC_SAMPLING_INTERVAL"] == str(interval)

    mock_console_error.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_subprocess_fails(
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

    monkeypatch.setattr(
        "rocprof_compute_profile.pc_sampling_profiler.console_error",
        mock_console_error,
    )

    method = "stochastic"
    interval = 5000
    workload_dir = str(tmp_path)
    options = ["another_app"]

    profiler = _make_pc_sampling_profiler(method, interval, workload_dir, "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(options)

    mock_capture_subprocess.assert_not_called()
    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]

    mock_capture_subprocess.reset_mock()
    console_error_calls.clear()
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

    profiler = _make_pc_sampling_profiler(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(options)

    mock_capture_subprocess.assert_called_once()
    assert console_error_calls == ["PC sampling failed."]


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_empty_appcmd(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """
    Edge Case: The appcmd is an empty string.
    The function should still attempt to run it. The behavior of
    capture_subprocess_output with an empty command is external to this function.
    """
    method = "host_trap"
    interval = 100
    workload_dir = str(tmp_path)
    options = ["--"]

    mock_capture_subprocess.return_value = (True, "Output with empty appcmd")

    profiler = _make_pc_sampling_profiler(method, interval, workload_dir, "rocprofv3")
    profiler._launch(options)

    assert mock_capture_subprocess.called
    options_list = mock_capture_subprocess.call_args[0][0]
    assert options_list[-1] == "--"
    mock_console_error.assert_not_called()

    mock_capture_subprocess.reset_mock()
    mock_console_error.reset_mock()
    sdk_lib_dir = tmp_path / "rocm_sdk_empty" / "lib"
    sdk_lib_dir.mkdir(parents=True, exist_ok=True)
    rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
    Path(rocprofiler_sdk_tool_path_sdk).touch()
    tool_dir = sdk_lib_dir / "rocprofiler-sdk"
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "librocprofiler-sdk-tool.so").touch()

    mock_capture_subprocess.return_value = (True, "Output with empty appcmd SDK")
    options = {"APP_CMD": ""}

    profiler = _make_pc_sampling_profiler(
        method, interval, workload_dir, "rocprofiler-sdk"
    )
    profiler._launch(options)

    assert mock_capture_subprocess.called
    assert mock_capture_subprocess.call_args[0][0] == ""
    mock_console_error.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_multiarg_appcmd(
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

        profiler = _make_pc_sampling_profiler(
            method, interval, workload_dir, "rocprofv3"
        )
        profiler._launch(options)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        assert options_list[0] == "rocprof_cli_tool"
        separator_index = options_list.index("--")
        assert options_list[separator_index:] == ["--", "./myapp", "arg1", "arg2"]
        mock_console_error.assert_not_called()


def test_pc_sampling_profiler_is_requested():
    workload_dir = "/tmp"
    for blocks in (["21"], ["pc_sampling"], ["2", "21"]):
        profiler = PCSamplingProfiler(
            args=MockArgs(filter_blocks=blocks),
            profiler="rocprofv3",
            workload_dir=workload_dir,
        )
        assert profiler.is_requested() is True

    profiler = PCSamplingProfiler(
        args=MockArgs(filter_blocks=["2"]),
        profiler="rocprofv3",
        workload_dir=workload_dir,
    )
    assert profiler.is_requested() is False


def test_pc_sampling_profiler_is_exclusive():
    workload_dir = "/tmp"
    exclusive = PCSamplingProfiler(
        args=MockArgs(filter_blocks=["21"]),
        profiler="rocprofv3",
        workload_dir=workload_dir,
    )
    assert exclusive.is_exclusive() is True

    mixed = PCSamplingProfiler(
        args=MockArgs(filter_blocks=["2", "21"]),
        profiler="rocprofv3",
        workload_dir=workload_dir,
    )
    assert mixed.is_exclusive() is False


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_cleanup_stale_output_removes_dir(
    mock_console_debug, tmp_path
):
    """Exclusive sdk run with a dict ROCPROF_OUTPUT_PATH removes the stale dir."""
    stale = tmp_path / "out" / "pmc_1"
    stale.mkdir(parents=True, exist_ok=True)
    options = {"ROCPROF_OUTPUT_PATH": str(stale)}

    profiler = PCSamplingProfiler(
        args=MockArgs(filter_blocks=["21"]),
        profiler="rocprofiler-sdk",
        workload_dir=str(tmp_path),
    )
    profiler._cleanup_stale_output(options)

    assert not stale.exists()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_cleanup_stale_output_noop_cases(
    mock_console_debug, tmp_path
):
    """Cleanup is a no-op outside the exclusive-sdk-dict-with-key case."""
    sdk_args = MockArgs(filter_blocks=["21"])

    # Non-sdk profiler: no removal even when exclusive.
    stale = tmp_path / "non_sdk"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfiler(
        args=sdk_args, profiler="rocprofv3", workload_dir=str(tmp_path)
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # Non-exclusive: no removal.
    stale = tmp_path / "mixed"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfiler(
        args=MockArgs(filter_blocks=["2", "21"]),
        profiler="rocprofiler-sdk",
        workload_dir=str(tmp_path),
    )._cleanup_stale_output({"ROCPROF_OUTPUT_PATH": str(stale)})
    assert stale.exists()

    # List options (v3): no removal.
    stale = tmp_path / "list_opts"
    stale.mkdir(parents=True, exist_ok=True)
    PCSamplingProfiler(
        args=sdk_args, profiler="rocprofiler-sdk", workload_dir=str(tmp_path)
    )._cleanup_stale_output(["--kernel-trace"])
    assert stale.exists()

    # Missing key: no error, no removal.
    PCSamplingProfiler(
        args=sdk_args, profiler="rocprofiler-sdk", workload_dir=str(tmp_path)
    )._cleanup_stale_output({"APP_CMD": "app"})


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_v3_live_attach(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """v3 live-attach appends --pid and --attach-duration-msec, no APP_CMD '--'."""
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        options = ["--pid", "1234", "--attach-duration-msec", "500"]
        mock_capture_subprocess.return_value = (True, "Success")

        profiler = _make_pc_sampling_profiler(
            "host_trap", 100, str(tmp_path), "rocprofv3"
        )
        profiler._launch(options)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        pid_idx = options_list.index("--pid")
        assert options_list[pid_idx + 1] == "1234"
        dur_idx = options_list.index("--attach-duration-msec")
        assert options_list[dur_idx + 1] == "500"
        assert "--" not in options_list
        mock_console_error.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_v3_live_attach_missing_pid_value(
    mock_console_debug, mock_capture_subprocess, tmp_path, monkeypatch
):
    """v3 live-attach with --pid but no trailing value triggers console_error."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr(
        "rocprof_compute_profile.pc_sampling_profiler.console_error",
        mock_console_error,
    )

    profiler = _make_pc_sampling_profiler("host_trap", 100, str(tmp_path), "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(["--pid"])

    assert console_error_calls == [
        "--pid or --attach-duration-msec option not found in "
        "profiler arguments for live attach mode"
    ]
    mock_capture_subprocess.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.perform_attach_detach")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_sdk_live_attach(
    mock_console_debug,
    mock_console_error,
    mock_perform_attach_detach,
    mock_capture_subprocess,
    tmp_path,
):
    """sdk live-attach calls perform_attach_detach and returns before launching."""
    options = {
        "ROCPROF_ATTACH_PID": "1234",
        "ROCPROF_ATTACH_LIBRARY": "lib.so",
    }

    profiler = _make_pc_sampling_profiler(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )
    profiler._launch(options)

    mock_perform_attach_detach.assert_called_once()
    mock_capture_subprocess.assert_not_called()
    mock_console_error.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_sdk_missing_app_cmd(
    mock_console_debug, mock_capture_subprocess, tmp_path, monkeypatch
):
    """sdk non-live-attach without APP_CMD errors before launching."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr(
        "rocprof_compute_profile.pc_sampling_profiler.console_error",
        mock_console_error,
    )

    profiler = _make_pc_sampling_profiler(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch({"LD_PRELOAD": "x"})

    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]
    mock_capture_subprocess.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_v3_missing_separator(
    mock_console_debug, mock_capture_subprocess, tmp_path, monkeypatch
):
    """v3 non-live-attach without a '--' separator errors before launching."""
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr(
        "rocprof_compute_profile.pc_sampling_profiler.console_error",
        mock_console_error,
    )

    profiler = _make_pc_sampling_profiler("host_trap", 100, str(tmp_path), "rocprofv3")
    with pytest.raises(RuntimeError, match="console_error called"):
        profiler._launch(["--something"])

    assert console_error_calls == [
        "APP_CMD, the workload's executable must be provided "
        "when not in live attach mode"
    ]
    mock_capture_subprocess.assert_not_called()


@mock.patch("rocprof_compute_profile.pc_sampling_profiler.capture_subprocess_output")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_error")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_log")
@mock.patch("rocprof_compute_profile.pc_sampling_profiler.console_debug")
def test_pc_sampling_profiler_run_cleanup_before_launch(
    mock_console_debug,
    mock_console_log,
    mock_console_error,
    mock_capture_subprocess,
    tmp_path,
):
    """run() removes stale output before reaching the subprocess launch seam, and
    emits the run header and a timing debug."""
    stale = tmp_path / "out" / "pmc_1"
    stale.mkdir(parents=True, exist_ok=True)
    options = {"ROCPROF_OUTPUT_PATH": str(stale), "APP_CMD": "my_app"}

    profiler = _make_pc_sampling_profiler(
        "host_trap", 100, str(tmp_path), "rocprofiler-sdk"
    )

    stale_existed_at_launch = []

    def record(*args, **kwargs):
        # Cleanup must already have run by the time we launch the subprocess.
        stale_existed_at_launch.append(stale.exists())
        return (True, "")

    mock_capture_subprocess.side_effect = record
    profiler.run(options, prior_run_count=0)

    assert stale_existed_at_launch == [False]
    assert not stale.exists()
    mock_console_error.assert_not_called()

    mock_console_log.assert_any_call("[Run 1/1][PC sampling profile run]")
    assert any(
        call.args and call.args[0] == "profiling"
        for call in mock_console_debug.call_args_list
    )


def _make_soc(rocm_version: str) -> SimpleNamespace:
    """Minimal soc stub exposing the rocm_version path read by
    __is_native_tool_supported."""
    return SimpleNamespace(_mspec=SimpleNamespace(rocm_version=rocm_version))


def _make_supported_args(filter_blocks: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        filter_blocks=filter_blocks,
        attach_pid=None,
        no_native_tool=False,
    )


# ---------------------------------------------------------------------------
# __is_native_tool_supported no longer excludes PC-sampling-only runs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filter_blocks",
    [["21"], ["pc_sampling"], ["21", "pc_sampling"]],
    ids=["block_21", "block_pc_sampling", "both_aliases"],
)
def test_native_tool_supported_for_pc_sampling_only(filter_blocks):
    """PC-sampling-only filter_blocks must NOT disqualify the native tool path.

    Regression guard: the old `is_only_pc_sampling(...)` exclusion in
    `__is_native_tool_supported` is gone."""
    args = _make_supported_args(filter_blocks)
    profiler = RocProfCompute_Base(
        args, profiler_mode="rocprofiler-sdk", soc=_make_soc("7.0.0")
    )
    # Access the name-mangled private method.
    assert profiler._RocProfCompute_Base__is_native_tool_supported(args) is True


@pytest.mark.parametrize(
    "rocm_version, attach_pid, expected",
    [
        ("6.4.0", None, False),  # ROCm < 7
        ("7.0.0", 12345, False),  # attach mode
        ("7.0.0", None, True),  # supported
    ],
    ids=["rocm_lt_7", "attach_pid_set", "supported"],
)
def test_native_tool_supported_preconditions(rocm_version, attach_pid, expected):
    """Other preconditions (ROCm >= 7, no --attach-pid) still gate the path."""
    args = _make_supported_args(["21"])
    args.attach_pid = attach_pid
    profiler = RocProfCompute_Base(
        args, profiler_mode="rocprofiler-sdk", soc=_make_soc(rocm_version)
    )
    assert profiler._RocProfCompute_Base__is_native_tool_supported(args) is expected


# ---------------------------------------------------------------------------
# get_profiler_options(pc_sampling=True) populates the PC sampling env vars
# ---------------------------------------------------------------------------
def _make_sdk_args(
    output_dir: Path, pc_sampling_method: str, pc_sampling_interval: int = 1048576
) -> argparse.Namespace:
    return argparse.Namespace(
        remaining="-- /bin/true",
        rocprofiler_sdk_tool_path=(
            "/opt/rocm/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so"
        ),
        format_rocprof_output="csv",
        output_directory=str(output_dir),
        iteration_multiplexing=None,
        attach_pid=None,
        attach_duration_msec=None,
        kokkos_trace=False,
        kernel=None,
        dispatch=None,
        torch_trace=False,
        pc_sampling_method=pc_sampling_method,
        pc_sampling_interval=pc_sampling_interval,
    )


@pytest.mark.parametrize(
    "method, expected_unit",
    [("host_trap", "time"), ("stochastic", "cycles")],
)
def test_get_profiler_options_pc_sampling_true_adds_env_vars(method, expected_unit):
    """When pc_sampling=True, the returned dict carries the PC sampling env vars
    the native tool (and the rocprofiler-sdk PC sampling service) read."""
    output_dir = Path(common.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        args = _make_sdk_args(output_dir, method, pc_sampling_interval=4096)
        profiler = rocprofiler_sdk_profiler(
            args, profiler_mode="rocprofiler-sdk", soc=None
        )
        options = profiler.get_profiler_options(
            native_tool_path="/tmp/fake_native_tool.so", pc_sampling=True
        )
        assert options["ROCPROF_PC_SAMPLING_METHOD"] == method
        assert options["ROCPROF_PC_SAMPLING_INTERVAL"] == "4096"
        assert options["ROCPROF_PC_SAMPLING_UNIT"] == expected_unit
        assert options["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] == "1"
        # No PMC collection on the PC sampling run.
        assert options["ROCPROF_COUNTER_COLLECTION"] == "0"
        # Native tool still gets onto LD_PRELOAD alongside the SDK tool.
        assert "/tmp/fake_native_tool.so" in options["LD_PRELOAD"]
    finally:
        common.clean_output_dir(True, str(output_dir))


def test_get_profiler_options_pc_sampling_false_preserves_pmc_behavior():
    """Default pc_sampling=False must not emit any PC-sampling env vars."""
    output_dir = Path(common.get_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        args = _make_sdk_args(output_dir, pc_sampling_method="host_trap")
        profiler = rocprofiler_sdk_profiler(
            args, profiler_mode="rocprofiler-sdk", soc=None
        )
        options = profiler.get_profiler_options(native_tool_path=None)
        for key in (
            "ROCPROF_PC_SAMPLING_METHOD",
            "ROCPROF_PC_SAMPLING_INTERVAL",
            "ROCPROF_PC_SAMPLING_UNIT",
            "ROCPROFILER_PC_SAMPLING_BETA_ENABLED",
        ):
            assert key not in options
        # PMC path: counter collection stays on (SDK collects).
        assert options["ROCPROF_COUNTER_COLLECTION"] == "1"
    finally:
        common.clean_output_dir(True, str(output_dir))
