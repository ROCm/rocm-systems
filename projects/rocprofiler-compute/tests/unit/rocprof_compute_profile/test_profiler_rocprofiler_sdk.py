# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from pathlib import Path

import pytest

from rocprof_compute_profile import profiler_rocprofiler_sdk as sdk
from utils.utils_common import PROFILE_OUTPUT_FORMAT


@pytest.mark.parametrize(
    "enabled",
    [
        pytest.param(None, id="direct_construction"),
        pytest.param(False, id="disabled"),
        pytest.param(True, id="enabled"),
    ],
)
@pytest.mark.parametrize("user_preload", [None, "user_a.so:user_b.so"])
@pytest.mark.parametrize("native_tool", [None, "native.so"])
def test_marker_trace_preserves_preload_order_and_direct_construction(
    tmp_path, monkeypatch, enabled, user_preload, native_tool
):
    args = argparse.Namespace(
        remaining="application --name 'two words'",
        rocprofiler_sdk_tool_path=str(tmp_path / "sdk" / "tool.so"),
        output_directory=str(tmp_path),
        iteration_multiplexing=None,
        attach_pid=None,
        attach_duration_msec=None,
        kernel=None,
        kernel_iteration_range=None,
        trace_ml_api=[] if enabled else None,
    )
    profiler = sdk.rocprofiler_sdk_profiler(args, "rocprofiler-sdk", None)
    if user_preload is None:
        monkeypatch.delenv("LD_PRELOAD", raising=False)
    else:
        monkeypatch.setenv("LD_PRELOAD", user_preload)
    monkeypatch.setattr(sdk, "resolve_rocm_library_path", str)

    options = profiler.get_profiler_options(native_tool_path=native_tool)

    parts = [user_preload, args.rocprofiler_sdk_tool_path, native_tool]
    if enabled:
        parts.append(
            str(
                Path(args.rocprofiler_sdk_tool_path).parent.parent
                / "librocprofiler-sdk-roctx.so"
            )
        )
    expected = {
        "LD_PRELOAD": ":".join(part for part in parts if part),
        "ROCPROF_KERNEL_TRACE": "1",
        "ROCPROF_COUNTER_COLLECTION": "0" if native_tool else "1",
        "ROCPROF_OUTPUT_FORMAT": PROFILE_OUTPUT_FORMAT,
        "ROCPROF_OUTPUT_PATH": f"{tmp_path}/out/pmc_1",
        "APP_CMD": ["application", "--name", "two words"],
    }
    if enabled:
        expected["ROCPROF_MARKER_API_TRACE"] = "1"
    assert options == expected
