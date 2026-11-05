# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse

import pytest

from rocprof_compute_profile.profiler_rocprof_v3 import rocprof_v3_profiler
from utils.utils_common import PROFILE_OUTPUT_FORMAT


@pytest.mark.parametrize("selected", [None, set(), {"torch", "triton"}])
def test_marker_choice_preserves_complete_options_and_pc_sampling_path(
    tmp_path, selected
):
    args = argparse.Namespace(
        remaining="application --name 'two words'",
        output_directory=str(tmp_path),
        attach_pid=None,
        attach_duration_msec=None,
        kernel=["first", "second"],
        kernel_iteration_range=["1", "3:5"],
        pc_sampling_method="stochastic",
        pc_sampling_interval=1024,
        ml_api_trace=True,
    )
    profiler = rocprof_v3_profiler(args, "rocprofv3", None)
    if selected is not None:
        profiler._selected_frameworks = selected

    assert profiler.get_profiler_options() == [
        "-d",
        f"{tmp_path}/out",
        "--marker-trace" if selected else "--kernel-trace",
        "--output-format",
        PROFILE_OUTPUT_FORMAT,
        "--kernel-include-regex",
        "first|second",
        "--kernel-iteration-range",
        "[1,3-5]",
        "--",
        "application",
        "--name",
        "two words",
    ]
    assert profiler.get_pc_sampling_profiler_options() == [
        "--kernel-trace",
        "--pc-sampling-beta-enabled",
        "--pc-sampling-method",
        "stochastic",
        "--pc-sampling-unit",
        "cycles",
        "--output-format",
        "json",
        "--pc-sampling-interval",
        "1024",
        "-d",
        str(tmp_path),
        "-o",
        "%pid%_ps_file",
        "--",
        "application",
        "--name",
        "two words",
    ]
