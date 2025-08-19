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

import shlex
from pathlib import Path

import config
from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from utils.logger import console_log, demarcate
from utils.utils import replace_timestamps, store_app_cmd


class rocprof_v2_profiler(RocProfCompute_Base):
    def __init__(self, profiling_args, profiler_mode, soc, supported_archs):
        super().__init__(profiling_args, profiler_mode, soc, supported_archs)
        self.ready_to_profile = (
            self.get_args().roof_only
            and not Path(self.get_args().path).joinpath("pmc_perf.csv").is_file()
            or not self.get_args().roof_only
        )

    def get_profiler_options(self, fname, soc):
        app_cmd = shlex.split(self.get_args().remaining)

        args = []
        # rocprof v2 does not support some counters on gfx 908 architecture
        if soc.get_arch() == "gfx908":
            metrics_path = str(
                Path(str(config.rocprof_compute_home)).joinpath(
                    "rocprof_compute_soc", "profile_configs", "metrics.xml"
                )
            )
            args += ["-m", metrics_path]

        args += [
            # v2 requires output directory argument
            "-d",
            self.get_args().path + "/" + "out",
        ]
        args.extend(app_cmd)
        # store args for debug message
        store_app_cmd(args)
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
            replace_timestamps(self.get_args().path)
