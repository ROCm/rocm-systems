# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse

from rocprof_compute_soc.soc_gfx1152 import gfx1152_soc
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs


class gfx1153_soc(gfx1152_soc):
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx1153")
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx1153"))
