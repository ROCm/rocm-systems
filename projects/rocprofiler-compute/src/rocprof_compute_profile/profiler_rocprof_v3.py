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
import shlex
from pathlib import Path

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_error, console_log, demarcate


class rocprof_v3_profiler(RocProfCompute_Base):
    def __init__(
        self,
        profiling_args: argparse.Namespace,
        profiler_mode: str,
        soc: OmniSoC_Base,
    ) -> None:
        super().__init__(profiling_args, profiler_mode, soc)
        self.ready_to_profile = (
            self.get_args().roof_only
            and not (Path(self.get_args().path) / "pmc_perf.csv").is_file()
            or not self.get_args().roof_only
        )

    def get_profiler_options(self, fname: str, soc: OmniSoC_Base) -> list[str]:
        args = self.get_args()
        app_cmd = shlex.split(args.remaining)

        if args.kokkos_trace:
            trace_option = "--kokkos-trace"
            # NOTE: --kokkos-trace feature is incomplete and is disabled for now.
            console_error(
                "The option '--kokkos-trace' is not supported in the current "
                "version of rocprof-compute. This functionality is planned for a "
                "future release. Please adjust your profiling options accordingly."
            )
        elif args.hip_trace:
            trace_option = "--hip-trace"
        else:
            trace_option = "--kernel-trace"

        profiling_options = [
            # v3 requires output directory argument
            "-d",
            f"{self.get_args().path}/out",
            trace_option,
            "--output-format",
            args.format_rocprof_output,
        ]

        if args.attach_pid:
            profiling_options.append("--pid")
            profiling_options.append(args.attach_pid)

            if args.attach_duration_msec:
                profiling_options.append("--attach-duration-msec")
                profiling_options.append(args.attach_duration_msec)

        # Kernel filtering
        if args.kernel:
            profiling_options.extend(["--kernel-include-regex", "|".join(args.kernel)])

        # Dispatch filtering
        dispatch = []
        # rocprofv3 dispatch indexing is inclusive and starts from 1
        if args.dispatch:
            for dispatch_id in args.dispatch:
                if ":" in dispatch_id:
                    # 4:7 -> 5-7
                    start, end = dispatch_id.split(":")
                    dispatch.append(f"{int(start) + 1}-{end}")
                else:
                    # 4 -> 5
                    dispatch.append(f"{int(dispatch_id) + 1}")
        if dispatch:
            profiling_options.extend([
                "--kernel-iteration-range",
                f"[{','.join(dispatch)}]",
            ])

        if not args.attach_pid:
            profiling_options.append("--")
            profiling_options.extend(app_cmd)
        return profiling_options

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to profiling."""
        super().pre_processing()

    @demarcate
    def run_profiling(self, version: str, prog: str) -> None:
        """Run profiling."""
        if not self.ready_to_profile:
            console_log("roofline", "Detected existing pmc_perf.csv")
            return

        if self.get_args().roof_only:
            console_log("roofline", "Generating pmc_perf.csv (roofline counters only).")

        # Log profiling options and setup filtering
        super().run_profiling(version, prog)

    @demarcate
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        if self.ready_to_profile:
            # Manually join each pmc_perf*.csv output
            self.join_prof()
            # Run roofline microbenchmark
            super().post_processing()
        else:
            console_log("roofline", "Detected existing pmc_perf.csv")
