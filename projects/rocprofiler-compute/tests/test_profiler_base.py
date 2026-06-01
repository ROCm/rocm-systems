# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
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
        dispatch=None,
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

    common.clean_output_dir(True, str(output_dir))


# ---------------------------------------------------------------------------
# run_profiling(): PC sampling gating / increment / delegation
# ---------------------------------------------------------------------------
def _make_run_profiling_profiler(tmp_path, filter_blocks, perfmon_files=0):
    """Build a RocProfCompute_Base ready to drive run_profiling() in isolation.

    Uses the rocprofv3 profiler mode so the native-tool path resolves to None
    without touching the filesystem, and seeds `perfmon_files` empty pmc_perf
    yaml files so total_runs reflects the counter-collection pass count.
    """
    if perfmon_files:
        perfmon_dir = Path(tmp_path) / "perfmon"
        perfmon_dir.mkdir(parents=True, exist_ok=True)
        for i in range(perfmon_files):
            (perfmon_dir / f"pmc_perf_{i}.yaml").write_text("")

    args = argparse.Namespace(
        filter_blocks=filter_blocks,
        output_directory=str(tmp_path),
        roof_only=False,
        no_native_tool=True,
        iteration_multiplexing=None,
        kernel=None,
        dispatch=None,
        remaining="-- ./app",
    )
    soc = SimpleNamespace(_mspec=SimpleNamespace(gpu_model="MI300"))
    profiler = RocProfCompute_Base(args, profiler_mode="rocprofv3", soc=soc)
    profiler._filter_blocks = filter_blocks
    return profiler


def test_run_profiling_delegates_to_pc_sampling_when_requested(tmp_path):
    """When a PC sampling block is requested, run_profiling delegates to
    PCSamplingProfiler.run() and does not emit the skip warning."""
    profiler = _make_run_profiling_profiler(tmp_path, ["pc_sampling"])
    base = "rocprof_compute_profile.profiler_base"
    with (
        mock.patch(f"{base}.PCSamplingProfiler") as mock_pc_cls,
        mock.patch(f"{base}.print_status"),
        mock.patch(f"{base}.console_log"),
        mock.patch(f"{base}.console_debug"),
        mock.patch(f"{base}.console_warning") as mock_warning,
        mock.patch(f"{base}.get_job_rank_and_size", return_value=(None, None)),
    ):
        instance = mock_pc_cls.return_value
        instance.is_exclusive.return_value = True
        instance.is_requested.return_value = True

        profiler.run_profiling(version="1.0.0", prog="rocprof-compute")

        instance.run.assert_called_once()
        for call in mock_warning.call_args_list:
            assert "PC sampling data collection skipped" not in str(call)


def test_run_profiling_skips_pc_sampling_when_not_requested(tmp_path):
    """When no PC sampling block is requested, run_profiling emits the skip
    warning and never calls PCSamplingProfiler.run()."""
    profiler = _make_run_profiling_profiler(tmp_path, ["2"], perfmon_files=1)
    base = "rocprof_compute_profile.profiler_base"
    with (
        mock.patch(f"{base}.PCSamplingProfiler") as mock_pc_cls,
        mock.patch(f"{base}.print_status"),
        mock.patch(f"{base}.console_log"),
        mock.patch(f"{base}.console_debug"),
        mock.patch(f"{base}.console_warning") as mock_warning,
        mock.patch(f"{base}.get_job_rank_and_size", return_value=(None, None)),
        mock.patch.object(RocProfCompute_Base, "profile", return_value=0.0),
    ):
        instance = mock_pc_cls.return_value
        instance.is_exclusive.return_value = False
        instance.is_requested.return_value = False

        profiler.run_profiling(version="1.0.0", prog="rocprof-compute")

        instance.run.assert_not_called()
        skip_warned = any(
            "PC sampling data collection skipped" in str(call)
            for call in mock_warning.call_args_list
        )
        assert skip_warned


def test_run_profiling_pc_sampling_increments_workload_runs_for_multirank(tmp_path):
    """A requested PC sampling pass counts as an extra workload run, so with a
    single counter-collection pass and >=2 ranks the multi-rank warning fires;
    without PC sampling that same single pass stays at 1 run and is silent."""
    base = "rocprof_compute_profile.profiler_base"

    def run_with(filter_blocks, is_requested):
        profiler = _make_run_profiling_profiler(
            tmp_path, filter_blocks, perfmon_files=1
        )
        with (
            mock.patch(f"{base}.PCSamplingProfiler") as mock_pc_cls,
            mock.patch(f"{base}.print_status"),
            mock.patch(f"{base}.console_log"),
            mock.patch(f"{base}.console_debug"),
            mock.patch(f"{base}.console_warning") as mock_warning,
            mock.patch(f"{base}.get_job_rank_and_size", return_value=("0", 2)),
            mock.patch.object(RocProfCompute_Base, "profile", return_value=0.0),
        ):
            instance = mock_pc_cls.return_value
            instance.is_exclusive.return_value = False
            instance.is_requested.return_value = is_requested
            profiler.run_profiling(version="1.0.0", prog="rocprof-compute")
            return any(
                "Multi-rank application detected" in str(call)
                for call in mock_warning.call_args_list
            )

    # PC sampling requested -> 1 counter pass + 1 PC sampling pass = 2 runs -> warn.
    assert run_with(["21", "2"], is_requested=True) is True
    # Not requested -> only the single counter pass -> no multi-rank warning.
    assert run_with(["2"], is_requested=False) is False


def test_run_profiling_sdk_routes_native_tool_into_pc_sampling(tmp_path):
    """In rocprofiler-sdk mode, run_profiling resolves the native tool path and
    threads it (with pc_sampling=True) into the options handed to
    PCSamplingProfiler.run() — the native-tool PC sampling routing."""
    args = argparse.Namespace(
        filter_blocks=["21"],
        output_directory=str(tmp_path),
        roof_only=False,
        no_native_tool=False,
        iteration_multiplexing=None,
        kernel=None,
        dispatch=None,
        remaining="-- ./app",
    )
    soc = SimpleNamespace(_mspec=SimpleNamespace(gpu_model="MI300"))
    profiler = RocProfCompute_Base(args, profiler_mode="rocprofiler-sdk", soc=soc)
    profiler._filter_blocks = ["21"]

    fake_native = "/opt/rocm/lib/rocprofiler-compute/librocprofiler-compute-tool.so"
    pc_sampling_opts = {"LD_PRELOAD": f"sdk_tool:{fake_native}", "tag": "pc"}
    base = "rocprof_compute_profile.profiler_base"

    def fake_get_options(native_tool_path=None, pc_sampling=False):
        assert native_tool_path == fake_native
        return pc_sampling_opts if pc_sampling else {"tag": "pmc"}

    with (
        mock.patch(f"{base}.PCSamplingProfiler") as mock_pc_cls,
        mock.patch(f"{base}.print_status"),
        mock.patch(f"{base}.console_log"),
        mock.patch(f"{base}.console_debug"),
        mock.patch(f"{base}.console_warning"),
        mock.patch(f"{base}.get_job_rank_and_size", return_value=(None, None)),
        mock.patch.object(
            RocProfCompute_Base,
            "_RocProfCompute_Base__get_native_tool_path",
            return_value=fake_native,
        ),
        mock.patch.object(
            RocProfCompute_Base, "get_profiler_options", side_effect=fake_get_options
        ),
    ):
        instance = mock_pc_cls.return_value
        instance.is_exclusive.return_value = True
        instance.is_requested.return_value = True

        profiler.run_profiling(version="1.0.0", prog="rocprof-compute")

        instance.run.assert_called_once()
        passed_opts = instance.run.call_args[0][0]
        assert passed_opts is pc_sampling_opts
        assert fake_native in passed_opts["LD_PRELOAD"]
