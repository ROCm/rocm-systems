# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for gfx1201 (RDNA 4 / Navi48) SoC registration."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rocprof_compute_soc.soc_gfx1201 import gfx1201_soc
from roofline.run_benchmark import BENCHMARKING_SUPPORTED
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import SUPPORTED_DATATYPES, OpsSupport
from utils.specs import MachineSpecsRDNA35, spec_family_for_arch


def make_soc() -> gfx1201_soc:
    args = SimpleNamespace()
    mspec = SimpleNamespace(rocminfo_lines=None, l2_banks=0)
    with patch("rocprof_compute_soc.soc_base.console_debug"):
        return gfx1201_soc(args, mspec)


@pytest.mark.misc
def test_gfx1201_soc_uses_navi48_limits():
    soc = make_soc()
    assert soc.get_arch() == "gfx1201"
    assert soc._mspec.l2_banks == 16
    assert soc._mspec.lds_banks_per_cu == 32
    assert soc._mspec.pipes_per_gpu == 2
    assert soc.get_compatible_profilers() == ["rocprofv3", "rocprofiler-sdk"]

    perfmon = mi_gpu_specs.get_perfmon_config("gfx1201")
    assert perfmon["SQ"] == 8
    assert perfmon["TCP"] == 4
    assert perfmon["GL2C"] == 4
    assert "SQG" not in perfmon


@pytest.mark.misc
def test_gfx1201_spec_registry_and_roofline_lists():
    assert mi_gpu_specs.get_gpu_series("gfx1201") == "RDNA4"
    assert mi_gpu_specs.get_gpu_model("gfx1201", "30032") == "NAVI48"
    assert spec_family_for_arch("gfx1201") is MachineSpecsRDNA35
    assert spec_family_for_arch("gfx1250") is not MachineSpecsRDNA35

    assert "gfx1201" in BENCHMARKING_SUPPORTED
    gfx1201_types = SUPPORTED_DATATYPES["gfx1201"]
    assert gfx1201_types["FP32"] == OpsSupport.VALU
    assert OpsSupport.MATRIX not in gfx1201_types.values()
    assert "FP8" not in gfx1201_types
