##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import argparse
from typing import Any, Optional

from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs


class gfx942_soc(OmniSoC_Base):
    def __init__(self, args: argparse.Namespace, mspec: MachineSpecs) -> None:
        super().__init__(args, mspec)
        self.set_arch("gfx942")
        self.set_compatible_profilers([
            "rocprofv3",
            "rocprofiler-sdk",
        ])
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx942"))

        # Set arch specific specs
        self._mspec.l2_banks = 16
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 4

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self) -> Optional[list[str]]:
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        # Performance counter filtering
        filter_blocks = self.perfmon_filter()
        return filter_blocks

    @demarcate
    def post_profiling(self) -> None:
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()

    @demarcate
    def analysis_setup(
        self, roofline_parameters: Optional[dict[str, Any]] = None
    ) -> None:
        """Perform any SoC-specific setup prior to analysis."""
        super().analysis_setup(roofline_parameters=roofline_parameters)
