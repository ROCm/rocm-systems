# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path
from types import SimpleNamespace

import common
import pytest

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_profile.profiler_rocprofiler_sdk import rocprofiler_sdk_profiler


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
