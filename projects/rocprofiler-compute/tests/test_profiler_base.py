# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import pytest

from rocprof_compute_base import RocProfCompute
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_profile.profiler_rocprof_v3 import rocprof_v3_profiler
from rocprof_compute_profile.profiler_rocprofiler_sdk import rocprofiler_sdk_profiler
from utils.utils_exceptions import (
    ExecutableNotFoundError,
    NoScriptInCommandError,
    PythonScriptNotFoundError,
)


def _make_sanitize_args(remaining, torch_trace=False, **overrides):
    """Build a minimal argparse.Namespace for profiler_base unit tests.

    Defaults satisfy sanitize(); pass **overrides to set or replace fields the
    run_profiling() tests need (e.g. output_directory, no_native_tool, kernel,
    or a pre-joined string ``remaining``).
    """
    defaults = dict(
        filter_blocks=[],
        set_selected=None,
        roof_only=False,
        output_directory="/tmp/test_workload",
        no_native_tool=False,
        iteration_multiplexing=None,
        attach_pid=None,
        attach_duration_msec=None,
        remaining=["--"] + remaining,
        torch_trace=torch_trace,
        dispatch=None,
        kernel=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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
    assert options["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    assert mock_resolve.call_count == 1

    # Case 2: only old library present -> falls back to it.
    new_lib.unlink()
    old_lib.write_text("")
    with patch(
        resolve_target, side_effect=[str(new_lib), str(old_lib)]
    ) as mock_resolve:
        options = profiler.get_profiler_options()
    assert options["ROCPROF_ATTACH_LIBRARY"] == str(old_lib)
    assert options["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] == "1"
    assert mock_resolve.call_count == 2

    # Case 3: neither library present -> console_error exits the process.
    old_lib.unlink()
    with patch(resolve_target, side_effect=[str(new_lib), str(old_lib)]):
        with pytest.raises(SystemExit):
            profiler.get_profiler_options()

    common.clean_output_dir(True, str(output_dir))


def test_rocprofv3_live_attach_uses_sync_output():
    """Unit test: rocprofv3 live attach requests synchronous output generation."""
    args = _make_sanitize_args(
        ["/bin/true"],
        attach_pid="12345",
        attach_duration_msec="500",
        format_rocprof_output="csv",
        kokkos_trace=False,
    )
    args.remaining = "-- /bin/true"
    profiler = rocprof_v3_profiler(args, profiler_mode="rocprofv3", soc=None)

    options = profiler.get_profiler_options()

    assert "--attach-sync-output" in options
    pid_idx = options.index("--pid")
    assert options[pid_idx + 1] == "12345"
    duration_idx = options.index("--attach-duration-msec")
    assert options[duration_idx + 1] == "500"
    assert "--" not in options


# ---------------------------------------------------------------------------
# get_pc_sampling_profiler_options(): sdk + v3 backends
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "native_tool_path, method, expected_unit, expected_ld_preload",
    [
        pytest.param(None, "host_trap", "time", ["/opt/sdk/tool.so"], id="sdk_only"),
        pytest.param(
            "/n/native.so",
            "stochastic",
            "cycles",
            ["/opt/sdk/tool.so", "/n/native.so"],
            id="native_appended",
        ),
    ],
)
def test_sdk_pc_sampling_options(
    tmp_path, native_tool_path, method, expected_unit, expected_ld_preload
):
    """sdk PC sampling options set the PC sampling env, json/ps_file output, and
    append the native tool (when given) to the upstream LD_PRELOAD."""
    args = _make_sanitize_args(
        ["/bin/true"],
        rocprofiler_sdk_tool_path="/opt/sdk/tool.so",
        format_rocprof_output="csv",
        output_directory=str(tmp_path),
        pc_sampling_method=method,
        pc_sampling_interval=1000,
        kokkos_trace=False,
    )
    args.remaining = "/bin/true"
    profiler = rocprofiler_sdk_profiler(args, profiler_mode="rocprofiler-sdk", soc=None)

    options = profiler.get_pc_sampling_profiler_options(
        native_tool_path=native_tool_path
    )

    assert options["ROCPROF_COUNTER_COLLECTION"] == "0"
    assert options["ROCPROF_KERNEL_TRACE"] == "1"
    assert options["ROCPROF_OUTPUT_FORMAT"] == "json"
    assert options["ROCPROF_OUTPUT_PATH"] == str(tmp_path)
    assert options["ROCPROF_OUTPUT_FILE_NAME"] == "ps_file"
    assert options["ROCPROFILER_PC_SAMPLING_BETA_ENABLED"] == "1"
    assert options["ROCPROF_PC_SAMPLING_METHOD"] == method
    assert options["ROCPROF_PC_SAMPLING_UNIT"] == expected_unit
    assert options["ROCPROF_PC_SAMPLING_INTERVAL"] == "1000"
    ld_preload = options["LD_PRELOAD"].split(":")
    assert all(part in ld_preload for part in expected_ld_preload)


@pytest.mark.parametrize(
    "method, expected_unit, attach_pid, attach_duration_msec",
    [
        pytest.param("host_trap", "time", None, None, id="app_cmd"),
        pytest.param("stochastic", "cycles", "1234", "500", id="live_attach"),
    ],
)
def test_v3_pc_sampling_options(
    tmp_path, method, expected_unit, attach_pid, attach_duration_msec
):
    """v3 PC sampling options build the CLI flags and append either the -- app
    cmd or the live-attach flags."""
    args = _make_sanitize_args(
        ["./myapp", "arg1"],
        output_directory=str(tmp_path),
        pc_sampling_method=method,
        pc_sampling_interval=1000,
        attach_pid=attach_pid,
        attach_duration_msec=attach_duration_msec,
    )
    args.remaining = "./myapp arg1"
    profiler = rocprof_v3_profiler(args, profiler_mode="rocprofv3", soc=None)

    options = profiler.get_pc_sampling_profiler_options()

    assert "--kernel-trace" in options
    assert "--pc-sampling-beta-enabled" in options
    assert options[options.index("--pc-sampling-method") + 1] == method
    assert options[options.index("--pc-sampling-unit") + 1] == expected_unit
    assert options[options.index("--pc-sampling-interval") + 1] == "1000"
    assert options[options.index("-d") + 1] == str(tmp_path)
    assert options[options.index("-o") + 1] == "ps_file"
    if attach_pid:
        assert "--attach-sync-output" in options
        assert options[options.index("--pid") + 1] == attach_pid
        assert options[options.index("--attach-duration-msec") + 1] == (
            attach_duration_msec
        )
        assert "--" not in options
    else:
        sep = options.index("--")
        assert options[sep:] == ["--", "./myapp", "arg1"]


# ---------------------------------------------------------------------------
# RocProfCompute.sanitize(): block 21 / block 30 experimental-gating
# ---------------------------------------------------------------------------
def _make_rpc_args(
    *,
    filter_blocks=None,
    filter_metrics=None,
    pc_sampling=False,
    pc_sampling_method="stochastic",
    pc_sampling_interval=None,
    membw_analysis=False,
    experimental=False,
    mode="profile",
) -> argparse.Namespace:
    """Build a minimal Namespace for RocProfCompute.sanitize() unit tests."""
    return argparse.Namespace(
        mode=mode,
        list_metrics=None,
        list_blocks=None,
        list_available_metrics=False,
        list_sets=False,
        specs=False,
        filter_blocks=filter_blocks,
        filter_metrics=filter_metrics,
        pc_sampling=pc_sampling,
        pc_sampling_method=pc_sampling_method,
        pc_sampling_interval=pc_sampling_interval,
        membw_analysis=membw_analysis,
        experimental=experimental,
        set_selected=None,
        roof_only=False,
        bench_only=False,
        no_roof=False,
        format_rocprof_output="csv",
        name="unit-test",
        output_directory="/tmp/unit-test",
    )


def _make_rpc_with_args(args: argparse.Namespace) -> RocProfCompute:
    """Construct a RocProfCompute without invoking __init__."""
    instance = RocProfCompute.__new__(RocProfCompute)
    # Name-mangled private attributes consumed by sanitize().
    instance._RocProfCompute__args = args
    instance._RocProfCompute__mode = args.mode
    return instance


@pytest.mark.parametrize(
    "args, expect_error, expected_filter_blocks",
    [
        pytest.param(
            _make_rpc_args(filter_blocks=["21"]),
            True,
            None,
            id="block21_without_pc_sampling_errors",
        ),
        pytest.param(
            _make_rpc_args(filter_blocks=["21"], pc_sampling=True),
            True,
            None,
            id="pc_sampling_without_experimental_errors",
        ),
        pytest.param(
            _make_rpc_args(filter_blocks=["pc_sampling"]),
            True,
            None,
            id="block_alias_without_pc_sampling_errors",
        ),
        pytest.param(
            _make_rpc_args(pc_sampling=True, experimental=True, filter_blocks=[]),
            False,
            ["21"],
            id="pc_sampling_with_experimental_injects_21",
        ),
        pytest.param(
            _make_rpc_args(filter_blocks=["21"], pc_sampling=True, experimental=True),
            False,
            ["21"],
            id="block21_with_pc_sampling_experimental_passes",
        ),
        pytest.param(
            _make_rpc_args(filter_blocks=["30"]),
            True,
            None,
            id="block30_without_membw_analysis_errors",
        ),
        pytest.param(
            _make_rpc_args(filter_blocks=["30"], membw_analysis=True),
            True,
            None,
            id="membw_analysis_without_experimental_errors",
        ),
        pytest.param(
            _make_rpc_args(
                filter_blocks=["30"], membw_analysis=True, experimental=True
            ),
            False,
            ["30"],
            id="block30_with_membw_analysis_experimental_passes",
        ),
    ],
)
def test_sanitize_block_experimental_gating(args, expect_error, expected_filter_blocks):
    """Unit test: block 21 and block 30 require their experimental flags."""
    instance = _make_rpc_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.sanitize()
    else:
        instance.sanitize()
        assert args.filter_blocks == expected_filter_blocks


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

    args = _make_sanitize_args(
        ["./app"],
        filter_blocks=filter_blocks,
        output_directory=str(tmp_path),
        no_native_tool=True,
    )
    # run_profiling() consumes remaining as the already-joined command string.
    args.remaining = "-- ./app"
    soc = SimpleNamespace(_mspec=SimpleNamespace(gpu_model="MI300"))
    profiler = RocProfCompute_Base(args, profiler_mode="rocprofv3", soc=soc)
    profiler._filter_blocks = filter_blocks
    return profiler


@pytest.mark.parametrize(
    "filter_blocks, perfmon_files, is_requested, ranks, "
    "expect_run, expect_skip_warning, expect_multirank_warning",
    [
        # A requested PC sampling block delegates to PCSamplingProfile.run()
        # and suppresses the skip warning.
        pytest.param(
            ["pc_sampling"],
            0,
            True,
            (None, None),
            True,
            False,
            False,
            id="delegates_when_requested",
        ),
        # No PC sampling block requested -> skip warning, run() never called.
        pytest.param(
            ["2"],
            1,
            False,
            (None, None),
            False,
            True,
            False,
            id="skips_when_not_requested",
        ),
        # Requested PC sampling adds a workload run, so 1 counter pass + 1 PC
        # sampling pass = 2 runs with >=2 ranks -> multi-rank warning.
        pytest.param(
            ["21", "2"],
            1,
            True,
            ("0", 2),
            True,
            False,
            True,
            id="multirank_warns_with_pc_sampling",
        ),
        # Not requested -> only the single counter pass -> no multi-rank warning.
        pytest.param(
            ["2"],
            1,
            False,
            ("0", 2),
            False,
            True,
            False,
            id="multirank_silent_without_pc_sampling",
        ),
    ],
)
def test_run_profiling_pc_sampling_gating(
    tmp_path,
    monkeypatch,
    filter_blocks,
    perfmon_files,
    is_requested,
    ranks,
    expect_run,
    expect_skip_warning,
    expect_multirank_warning,
):
    """run_profiling() delegates to PCSamplingProfile.run() only when a PC
    sampling block is requested, emits the skip warning otherwise, and counts a
    requested PC sampling pass as an extra workload run for the multi-rank
    warning."""
    profiler = _make_run_profiling_profiler(
        tmp_path, filter_blocks, perfmon_files=perfmon_files
    )
    base = "rocprof_compute_profile.profiler_base"

    mock_pc_cls = Mock()
    instance = mock_pc_cls.return_value
    instance.is_requested.return_value = is_requested
    mock_warning = Mock()

    monkeypatch.setattr(f"{base}.PCSamplingProfile", mock_pc_cls)
    monkeypatch.setattr(f"{base}.print_status", Mock())
    common.patch_console(
        monkeypatch, base, "log", "debug", "warning", warning=mock_warning
    )
    monkeypatch.setattr(f"{base}.get_job_rank_and_size", Mock(return_value=ranks))
    monkeypatch.setattr(RocProfCompute_Base, "profile", Mock(return_value=0.0))
    monkeypatch.setattr(
        RocProfCompute_Base,
        "get_pc_sampling_profiler_options",
        Mock(return_value=[]),
        raising=False,
    )

    profiler.run_profiling(version="1.0.0", prog="rocprof-compute")

    assert instance.run.called is expect_run

    skip_warned = any(
        "PC sampling data collection skipped" in str(call)
        for call in mock_warning.call_args_list
    )
    assert skip_warned is expect_skip_warning

    multirank_warned = any(
        "Multi-rank application detected" in str(call)
        for call in mock_warning.call_args_list
    )
    assert multirank_warned is expect_multirank_warning


@pytest.mark.parametrize(
    "method, interval, expect_error, expected_interval",
    [
        pytest.param("host_trap", None, False, 512, id="host_trap_unset_default"),
        pytest.param("stochastic", None, False, 1048576, id="stochastic_unset_default"),
        pytest.param("stochastic", 65536, False, 65536, id="stochastic_min_accepted"),
        pytest.param("stochastic", 12345, True, None, id="stochastic_not_pow2"),
        pytest.param("stochastic", 32768, True, None, id="stochastic_below_min"),
        pytest.param("host_trap", 100, False, 100, id="host_trap_positive_accepted"),
        pytest.param("host_trap", 0, True, None, id="host_trap_zero_rejected"),
        pytest.param("host_trap", -1, True, None, id="host_trap_negative_rejected"),
    ],
)
def test_sanitize_pc_sampling_interval(
    method, interval, expect_error, expected_interval
):
    """Unit test: --pc-sampling-interval default and stochastic validation."""
    args = _make_rpc_args(
        pc_sampling=True,
        experimental=True,
        filter_blocks=[],
        pc_sampling_method=method,
        pc_sampling_interval=interval,
    )
    instance = _make_rpc_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.sanitize()
    else:
        instance.sanitize()
        assert args.pc_sampling_interval == expected_interval


# ---------------------------------------------------------------------------
# run_profiling(): native_tool_path reaches get_pc_sampling_profiler_options
# ---------------------------------------------------------------------------
def _make_sdk_run_profiling_profiler(
    tmp_path,
    monkeypatch,
    *,
    filter_blocks,
    perfmon_files=0,
    rocm_version="7.0.0",
    profiler_mode="rocprofiler-sdk",
    no_native_tool=False,
    attach_pid=None,
    native_finder_raises=False,
):
    """Drive run_profiling() on the sdk native-tool path; returns (profiler, mocks)."""
    if perfmon_files:
        perfmon_dir = Path(tmp_path) / "perfmon"
        perfmon_dir.mkdir(parents=True, exist_ok=True)
        for i in range(perfmon_files):
            (perfmon_dir / f"pmc_perf_{i}.yaml").write_text("")

    args = _make_sanitize_args(
        ["./app"],
        filter_blocks=filter_blocks,
        output_directory=str(tmp_path),
        no_native_tool=no_native_tool,
        attach_pid=attach_pid,
        pc_sampling_method="host_trap",
        pc_sampling_interval=1000,
    )
    args.remaining = "-- ./app"
    soc = SimpleNamespace(
        _mspec=SimpleNamespace(gpu_model="MI300", rocm_version=rocm_version)
    )
    profiler = RocProfCompute_Base(args, profiler_mode=profiler_mode, soc=soc)
    profiler._filter_blocks = filter_blocks

    base = "rocprof_compute_profile.profiler_base"

    mock_pc_cls = Mock()
    mock_pc_cls.return_value.is_requested.return_value = any(
        block in ("21", "pc_sampling") for block in filter_blocks
    )

    mock_finder_cls = Mock()
    finder_instance = mock_finder_cls.return_value
    if native_finder_raises:
        finder_instance.get_collector_library_path.side_effect = RuntimeError("boom")
    else:
        finder_instance.get_collector_library_path.return_value = "/n/native.so"

    mock_profile = Mock(return_value=0.0)

    monkeypatch.setattr(f"{base}.PCSamplingProfile", mock_pc_cls)
    monkeypatch.setattr(f"{base}.NativeToolFinder", mock_finder_cls)
    monkeypatch.setattr(f"{base}.print_status", Mock())
    monkeypatch.setattr(
        f"{base}.get_job_rank_and_size", Mock(return_value=(None, None))
    )
    consoles = common.patch_console(
        monkeypatch, base, "log", "debug", "warning", "error"
    )
    monkeypatch.setattr(RocProfCompute_Base, "profile", mock_profile)
    # Base get_profiler_options ignores native_tool_path kwarg passed in sdk mode.
    monkeypatch.setattr(
        RocProfCompute_Base,
        "get_profiler_options",
        lambda self, *a, **k: {},
    )
    mock_pc_options = Mock(return_value={})
    monkeypatch.setattr(
        RocProfCompute_Base,
        "get_pc_sampling_profiler_options",
        mock_pc_options,
        raising=False,
    )

    mocks = SimpleNamespace(
        pc_cls=mock_pc_cls,
        pc_options=mock_pc_options,
        finder_cls=mock_finder_cls,
        profile=mock_profile,
        console_error=consoles["error"],
    )
    return profiler, mocks


def _pc_native_tool_path(mock_pc_options):
    """native_tool_path passed to get_pc_sampling_profiler_options (or None)."""
    assert mock_pc_options.called, "get_pc_sampling_profiler_options was never called"
    return mock_pc_options.call_args.kwargs.get("native_tool_path")


@pytest.mark.parametrize(
    "kwargs, expect_native_path, expect_profile_called, expect_error",
    [
        # PC-sampling-only: native tool threaded, no counter pass.
        pytest.param(
            dict(filter_blocks=["21"], perfmon_files=0),
            True,
            False,
            None,
            id="pc_only_threads_path_no_counter_pass",
        ),
        # Mixed blocks: native tool threaded and the counter pass runs.
        pytest.param(
            dict(filter_blocks=["2", "21"], perfmon_files=1),
            True,
            True,
            None,
            id="mixed_threads_and_runs_counter_pass",
        ),
        # --no-native-tool: native tool not threaded.
        pytest.param(
            dict(filter_blocks=["21"], perfmon_files=0, no_native_tool=True),
            False,
            None,
            None,
            id="no_native_tool_threads_none",
        ),
        # Live attach: native tool not threaded.
        pytest.param(
            dict(filter_blocks=["2", "21"], perfmon_files=1, attach_pid="123"),
            False,
            None,
            None,
            id="attach_pid_threads_none",
        ),
        # ROCm major < 7: native tool not threaded.
        pytest.param(
            dict(filter_blocks=["2", "21"], perfmon_files=1, rocm_version="6.4.0"),
            False,
            None,
            None,
            id="rocm_major_lt_7_threads_none",
        ),
        # rocprofv3 backend: native tool not threaded.
        pytest.param(
            dict(filter_blocks=["2", "21"], perfmon_files=1, profiler_mode="rocprofv3"),
            False,
            None,
            None,
            id="rocprofv3_threads_none",
        ),
        # Native finder failure: console_error fires.
        pytest.param(
            dict(filter_blocks=["2", "21"], perfmon_files=1, native_finder_raises=True),
            None,
            None,
            True,
            id="native_resolution_failure_errors",
        ),
    ],
)
def test_run_profiling_native_tool_path(
    tmp_path,
    monkeypatch,
    kwargs,
    expect_native_path,
    expect_profile_called,
    expect_error,
):
    """run_profiling() threads native_tool_path only on the supported sdk path,
    runs the counter pass for non-PC blocks, and errors on finder failure."""
    profiler, mocks = _make_sdk_run_profiling_profiler(tmp_path, monkeypatch, **kwargs)
    profiler.run_profiling(version="1.0.0", prog="rocprof-compute")

    if expect_native_path is not None:
        path = _pc_native_tool_path(mocks.pc_options)
        assert (path is not None) is expect_native_path
    if expect_profile_called is not None:
        assert mocks.profile.called is expect_profile_called
    if expect_error is not None:
        assert mocks.console_error.called is expect_error
