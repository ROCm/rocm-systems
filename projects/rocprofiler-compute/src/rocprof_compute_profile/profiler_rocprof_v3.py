# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import shlex

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

    def get_profiler_options(self) -> list[str]:
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
        elif getattr(args, "torch_trace", False):
            trace_option = "--marker-trace"
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
                    # 4:7 -> 4-7
                    start, end = dispatch_id.split(":")
                    dispatch.append(f"{start}-{end}")
                else:
                    # 4 -> 4
                    dispatch.append(f"{dispatch_id}")
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
        if self.get_args().roof_only:
            console_log("roofline", "Profiling roofline counters only.")

        # Log profiling options and setup filtering
        super().run_profiling(version, prog)

    @demarcate
    def post_processing(self) -> None:
        """Perform any post-processing steps prior to profiling."""
        super().post_processing()
