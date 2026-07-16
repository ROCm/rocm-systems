# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import shlex

from rocprof_compute_profile.profiler_base import RocProfCompute_Base
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils.logger import console_error, console_log, demarcate
from utils.utils_profile import pc_sampling_unit


class rocprof_v3_profiler(RocProfCompute_Base):
    def __init__(
        self,
        profiling_args: argparse.Namespace,
        profiler_mode: str,
        soc: OmniSoC_Base,
    ) -> None:
        super().__init__(profiling_args, profiler_mode, soc)

    def _live_attach_options(self) -> list[str]:
        args = self.get_args()
        options = ["--pid", args.attach_pid, "--attach-sync-output"]
        if args.attach_duration_msec:
            options += ["--attach-duration-msec", args.attach_duration_msec]
        return options

    def _app_cmd_options(self) -> list[str]:
        return ["--", *shlex.split(self.get_args().remaining)]

    def get_profiler_options(self) -> list[str]:
        args = self.get_args()
        if args.kokkos_trace:
            trace_option = "--kokkos-trace"
            # NOTE: --kokkos-trace feature is incomplete and is disabled for now.
            console_error(
                "The option '--kokkos-trace' is not supported in the current "
                "version of rocprof-compute. This functionality is planned for a "
                "future release. Please adjust your profiling options accordingly."
            )
        elif getattr(self, "_selected_frameworks", set()):
            trace_option = "--marker-trace"
        else:
            trace_option = "--kernel-trace"
        profiling_options = [
            # v3 requires output directory argument
            "-d",
            f"{self.get_args().output_directory}/out",
            trace_option,
            "--output-format",
            args.format_rocprof_output,
        ]

        if args.attach_pid:
            profiling_options += self._live_attach_options()

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
            profiling_options += self._app_cmd_options()
        return profiling_options

    def get_pc_sampling_profiler_options(self) -> list[str]:
        args = self.get_args()
        method = args.pc_sampling_method

        profiling_options = [
            "--kernel-trace",
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            method,
            "--pc-sampling-unit",
            pc_sampling_unit(method),
            "--output-format",
            "json",
            "--pc-sampling-interval",
            str(args.pc_sampling_interval),
            "-d",
            args.output_directory,
            "-o",
            "ps_file",  # TODO: sync up with the name from source in 2100_.yaml
        ]

        if args.attach_pid:
            profiling_options += self._live_attach_options()
        else:
            profiling_options += self._app_cmd_options()
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
