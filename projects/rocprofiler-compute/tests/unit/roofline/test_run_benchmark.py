# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for roofline/run_benchmark.py. No GPU is required."""

import pytest

from roofline import run_benchmark
from roofline.benchmark import ceilings


def test_aliases_load_the_gfx942_class():
    try:
        gfx942_class = run_benchmark.bench_class("gfx942")
    except OSError as hip_load_error:
        # hip_interface and hiprtc_interface dlopen the ROCm libraries at import.
        pytest.skip(f"ROCm runtime libraries unavailable: {hip_load_error}")
    for arch in ceilings.BENCH_DEVICE_ALIASES:
        assert run_benchmark.bench_class(arch) is gfx942_class
