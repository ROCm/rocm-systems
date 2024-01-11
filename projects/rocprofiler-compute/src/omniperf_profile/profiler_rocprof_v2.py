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
import logging
from omniperf_profile.profiler_base import OmniProfiler_Base
from utils.utils import demarcate
from utils.csv_processor import kernel_name_shortener

class rocprof_v2_profiler(OmniProfiler_Base):
    def __init__(self,profiling_args,profiler_mode,soc):
        super().__init__(profiling_args,profiler_mode,soc)
        self.ready_to_profile = (self.get_args().roof_only and not os.path.isfile(os.path.join(self.get_args().path, "pmc_perf.csv"))
                            or not self.get_args().roof_only)

    def get_profiler_options(self, fname):
        fbase = os.path.splitext(os.path.basename(fname))[0]
        app_cmd = self.get_args().remaining
        args = [
            # v2 requires output directory argument
            "-d", self.get_args().path + "/" + "out",
            # v2 does not require csv extension
            "-o", fbase,
            # v2 doen not require quotes on cmd
            app_cmd
        ]
        return args
    #-----------------------
    # Required child methods
    #-----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling.
        """
        super().pre_processing()
        if self.ready_to_profile:
            self.pmc_perf_split(self.get_args().path)

    @demarcate
    def run_profiling(self, version, prog):
        """Run profiling.
        """
        if self.ready_to_profile:
            if self.get_args().roof_only:
                logging.info("[roofline] Generating pmc_perf.csv")
            super().run_profiling(version, prog)
        else:
            logging.info("[roofline] Detected existing pmc_perf.csv")

        # [Run] Get any SoC specific rocprof options
            # Pass profiler name and throw error if not supported
        soc_options = self._soc.get_rocprof_options(rocprof_version)
            
        # [Run] Load any rocprof version rocprof options
            # -i
            # -d
            # -o
        profiler_options = [
            "-i", fname, 
            "-d", workload_dir,
            "-o", fbase,
            cmd
        ]
            
        # [Run] Call run_prof() util

        


    @demarcate
    def post_processing(self):
        """Perform any post-processing steps prior to profiling.
        """
        super().post_processing()

        # Different rocprof versions have different headers. Set mapping for profiler output 
        output_headers = {
            "Kernel_Name": "Kernel_Name",
            "Grid_Size": "Grid_Size",
            "GPU_ID": "GPU_ID",
            "Workgroup_Size": "Workgroup_Size",
            "LDS_Per_Workgroup": "LDS_Per_Workgroup",
            "Scratch_Per_Workitem": "Scratch_Per_Workitem",
            "SGPR": "SGPR",
            "Arch_VGPR": "Arch_VGPR",
            "Accum_VGPR": "Accum_VGPR",
            "Start_Timestamp": "Start_Timestamp",
            "End_Timestamp": "End_Timestamp",
        }
        
        if self.ready_to_profile:
            # Pass headers to join on 
            self.join_prof(output_headers)
            # Demangle and overwrite original KernelNames
            kernel_name_shortener(self.get_args().path, self.get_args().kernel_verbose)

