# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
from typing import Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_error, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs


class gfx908_soc(OmniSoC_Base):
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx908")
        self.set_compatible_profilers(["rocprofv3", "rocprofiler-sdk"])
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx908"))

        # Set arch specific specs
        self._mspec.l2_banks = 32
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 4

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        if self.get_args().roof_only:
            console_error(f"{self.get_arch()} does not support roofline analysis")

        # Perfmon filtering
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()
