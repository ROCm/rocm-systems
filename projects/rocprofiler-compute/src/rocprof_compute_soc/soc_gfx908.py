##############################################################################bl
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

from pathlib import Path

import config
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_error, demarcate
from utils.mi_gpu_spec import mi_gpu_specs


class gfx908_soc(OmniSoC_Base):
    def __init__(self, args, mspec):
        super().__init__(args, mspec)
        self.set_arch("gfx908")
        self.set_compatible_profilers(
            ["rocprofv1", "rocscope", "rocprofv3", "rocprofiler-sdk"]
        )
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(mi_gpu_specs.get_perfmon_config("gfx908"))

        # Set arch specific specs
        self._mspec._l2_banks = 32
        self._mspec.lds_banks_per_cu = 32
        self._mspec.pipes_per_gpu = 4

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def profiling_setup(self):
        """Perform any SoC-specific setup prior to profiling."""
        super().profiling_setup()
        if self.get_args().roof_only:
            console_error("%s does not support roofline analysis" % self.get_arch())
        # Perfmon filtering
        self.perfmon_filter(self.get_args().roof_only)

    @demarcate
    def post_profiling(self):
        """Perform any SoC-specific post profiling activities."""
        super().post_profiling()

    @demarcate
    def analysis_setup(self):
        """Perform any SoC-specific setup prior to analysis."""
        super().analysis_setup()
