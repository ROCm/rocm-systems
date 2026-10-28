# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse

from rocprof_compute_soc.soc_gfx1201 import gfx1201_soc
from utils.specs import MachineSpecs


def test_gfx1201_soc_initializes_with_current_base_api() -> None:
    specs = MachineSpecs()

    soc = gfx1201_soc(argparse.Namespace(), specs)

    assert soc.get_arch() == "gfx1201"
    assert specs.l2_banks == 32
    assert specs.lds_banks_per_cu == 32
