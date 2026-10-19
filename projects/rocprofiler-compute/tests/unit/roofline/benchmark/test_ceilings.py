# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for roofline/benchmark/ceilings.py. No GPU or ROCm library is required."""

import subprocess
import sys

from common import SRC

from roofline.benchmark import ceilings


def test_unsupported_data_types_are_not_measured():
    assert "MFMAF8Flops" not in ceilings.measured_ceilings("gfx90a")
    assert "MFMAF8Flops" in ceilings.measured_ceilings("gfx942")


def test_aliases_measure_what_gfx942_measures():
    for arch in ceilings.BENCH_DEVICE_ALIASES:
        assert ceilings.measured_ceilings(arch) == ceilings.measured_ceilings("gfx942")


def test_arch_without_a_benchmark_measures_nothing():
    assert ceilings.measured_ceilings("gfx908") == set()


def test_unsupported_data_types_are_family_data_types():
    for device, data_types in ceilings.UNSUPPORTED_DATA_TYPES.items():
        assert set(data_types) <= set(ceilings.FAMILY_CEILINGS[device[:-2]]), device


def test_imports_no_rocm_library():
    """tools/validate_metric_library.py reads ceilings on hosts without ROCm."""
    code = (
        "import sys; import roofline.run_benchmark, roofline.benchmark.ceilings; "
        "sys.exit(any(name.startswith('utils.hip') for name in sys.modules))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=SRC, check=True)
