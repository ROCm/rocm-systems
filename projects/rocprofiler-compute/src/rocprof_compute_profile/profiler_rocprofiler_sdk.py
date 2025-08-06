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


import os
import shlex
from pathlib import Path

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.logger import console_error, console_log, demarcate


class rocprofiler_sdk_profiler(RocProfCompute_Base):
    def __init__(self, profiling_args, profiler_mode, soc, supported_archs):
        super().__init__(profiling_args, profiler_mode, soc, supported_archs)
        self.ready_to_profile = (
            self.get_args().roof_only
            and not Path(self.get_args().path).joinpath("pmc_perf.csv").is_file()
            or not self.get_args().roof_only
        )

    def get_profiler_options(self, fname, soc):
        app_cmd = shlex.split(self.get_args().remaining)
        rocm_libdir = str(Path(self.get_args().rocprofiler_sdk_library_path).parent)
        rocprofiler_sdk_tool_path = str(
            Path(rocm_libdir).joinpath("rocprofiler-sdk/librocprofiler-sdk-tool.so")
        )
        ld_preload = [
            rocprofiler_sdk_tool_path,
            self.get_args().rocprofiler_sdk_library_path,
        ]
        options = {
            "ROCPROFILER_LIBRARY_CTOR": "1",
            "LD_PRELOAD": ":".join(ld_preload),
            "ROCP_TOOL_LIBRARIES": rocprofiler_sdk_tool_path,
            "LD_LIBRARY_PATH": rocm_libdir,
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_OUTPUT_FORMAT": self.get_args().format_rocprof_output,
            "ROCPROF_OUTPUT_PATH": self.get_args().path + "/out/pmc_1",
        }

        if self.get_args().kokkos_trace:
            # NOTE: --kokkos-trace feature is incomplete and is disabled for now.
            console_error(
                "The option '--kokkos-trace' is not supported in the current version of rocprof-compute. This functionality is planned for a future release. Please adjust your profiling options accordingly."
            )
        if self.get_args().hip_trace:
            options["ROCPROF_HIP_COMPILER_API_TRACE"] = "1"
            options["ROCPROF_HIP_RUNTIME_API_TRACE"] = "1"

        # Kernel filtering
        if self.get_args().kernel:
            options["ROCPROF_KERNEL_FILTER_INCLUDE_REGEX"] = "|".join(
                self.get_args().kernel
            )
        # Dispatch filtering
        dispatch = []
        # rocprof sdk dispatch indexing is inclusive and starts from 1
        if self.get_args().dispatch:
            for dispatch_id in self.get_args().dispatch:
                if ":" in dispatch_id:
                    tokens = dispatch_id.split(":")
                    # 4:7 -> 5-7
                    dispatch.append(f"{int(tokens[0]) + 1}-{tokens[1]}")
                else:
                    # 4 -> 5
                    dispatch.append(f"{int(dispatch_id) + 1}")
        if dispatch:
            options["ROCPROF_KERNEL_FILTER_RANGE"] = f"[{','.join(dispatch)}]"
        options["APP_CMD"] = app_cmd
        return options

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to profiling."""
        super().pre_processing()

    @demarcate
    def run_profiling(self, version, prog):
        """Run profiling."""
        if self.ready_to_profile:
            if self.get_args().roof_only:
                console_log(
                    "roofline", "Generating pmc_perf.csv (roofline counters only)."
                )
            # Log profiling options and setup filtering
            super().run_profiling(version, prog)
        else:
            console_log("roofline", "Detected existing pmc_perf.csv")

    @demarcate
    def post_processing(self):
        """Perform any post-processing steps prior to profiling."""
        super().post_processing()

        if self.ready_to_profile:
            # Manually join each pmc_perf*.csv output
            self.join_prof()
            # Replace timestamp data to solve a known rocprof bug
            # replace_timestamps(self.get_args().path)
