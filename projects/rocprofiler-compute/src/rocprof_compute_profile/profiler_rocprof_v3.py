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

import config
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.logger import console_error, console_log, demarcate


class rocprof_v3_profiler(RocProfCompute_Base):
    def __init__(self, profiling_args, profiler_mode, soc, supported_archs):
        super().__init__(profiling_args, profiler_mode, soc, supported_archs)
        self.ready_to_profile = (
            self.get_args().roof_only
            and not Path(self.get_args().path).joinpath("pmc_perf.csv").is_file()
            or not self.get_args().roof_only
        )

    def get_profiler_options(self, fname, soc):
        app_cmd = shlex.split(self.get_args().remaining)
        trace_option = "--kernel-trace"
        if self.get_args().kokkos_trace:
            trace_option = "--kokkos-trace"
            # NOTE: --kokkos-trace feature is incomplete and is disabled for now.
            console_error(
                "The option '--kokkos-trace' is not supported in the current version of rocprof-compute. This functionality is planned for a future release. Please adjust your profiling options accordingly."
            )
        if self.get_args().hip_trace:
            trace_option = "--hip-trace"

        args = [
            # v3 requires output directory argument
            "-d",
            self.get_args().path + "/" + "out",
            trace_option,
            "--output-format",
            self.get_args().format_rocprof_output,
        ]
        # Kernel filtering
        if self.get_args().kernel:
            args.extend(["--kernel-include-regex", "|".join(self.get_args().kernel)])
        # Dispatch filtering
        dispatch = []
        # rocprofv3 dispatch indexing is inclusive and starts from 1
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
            args.extend(["--kernel-iteration-range", f"[{','.join(dispatch)}]"])
        args.append("--")
        args.extend(app_cmd)
        return args

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
