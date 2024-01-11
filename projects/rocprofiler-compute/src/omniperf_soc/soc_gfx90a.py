##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

import os
import config
from omniperf_soc.soc_base import OmniSoC_Base
from utils.utils import demarcate, mibench
from roofline import Roofline
import logging

class gfx90a_soc (OmniSoC_Base):
    def __init__(self,args):
        super().__init__(args)
        soc = "gfx90a"
        self.set_soc(soc)
        if hasattr(self.get_args(), 'roof_only') and self.get_args().roof_only:
            self.set_perfmon_dir(os.path.join(str(config.omniperf_home), "omniperf_soc", "profile_configs", "roofline"))
        else:
            self.set_perfmon_dir(os.path.join(str(config.omniperf_home), "omniperf_soc", "profile_configs", soc)) 
        # Per IP block max number of simultaneous counters. GFX IP Blocks
        self.set_perfmon_config(
            {
                "SQ": 8,
                "TA": 2,
                "TD": 2,
                "TCP": 4,
                "TCC": 4,
                "CPC": 2,
                "CPF": 2,
                "SPI": 2,
                "GRBM": 2,
                "GDS": 4,
                "TCC_channels": 32
            }
        )
        self.set_soc_param(
            {
                "numSE": 8,
                "numCU": 110,
                "numSIMD": 440,
                "numWavesPerCU": 32,
                "numSQC": 56,
                "L2Banks": 32,
                "LDSBanks": 32,
                "Freq": 1700,
                "mclk": 1600
            }
        )
        self.roofline_obj = Roofline(args)

    #-----------------------
    # Required child methods
    #-----------------------
    @demarcate
    def profiling_setup(self):
        """Perform any SoC-specific setup prior to profiling.
        """
        super().profiling_setup()
        # Performance counter filtering
        self.perfmon_filter(self.get_args().roof_only)
        

    @demarcate
    def post_profiling(self):
        """Perform any SoC-specific post profiling activities.
        """
        super().post_profiling()
        if not self.get_args().no_roof:
            logging.info("[roofline] Checking for roofline.csv in " + str(self.get_args().path))
            if not os.path.isfile(os.path.join(self.get_args().path, "roofline.csv")):
                mibench(self.get_args())
            self.roofline_obj.post_processing()
        else:
            logging.info("[roofline] Skipping roofline")


    @demarcate
    def analysis_setup(self, roofline_parameters=None):
        """Perform any SoC-specific setup prior to analysis.
        """
        super().analysis_setup()
        # configure roofline for analysis
        if roofline_parameters:
            self.roofline_obj = Roofline(self.get_args(), roofline_parameters)



