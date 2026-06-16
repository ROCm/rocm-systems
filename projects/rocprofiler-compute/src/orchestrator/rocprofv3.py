# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""rocprofv3 profile and analysis orchestration."""

from __future__ import annotations

import argparse
import shlex
import shutil
import time
from pathlib import Path
from typing import cast

from interface.factory import create_profile_data_writer
from interface.profile_data import ProfileDataReaderOptions
from orchestrator.common import (
    ProfileDataAnalysisOrchestrator,
    ProfilerOptions,
    build_counter_collection_environment,
    build_profile_pass_context,
    classify_failed_profile_output,
    cleanup_counter_collection_environment,
    get_profile_input_info,
    is_live_attach,
)
from utils.logger import console_debug, console_error
from utils.utils_common import capture_subprocess_output


class Rocprofv3ProfileOrchestrator:
    """Coordinate rocprofv3 execution and profile data finalization."""

    profiler_command = "rocprofv3"

    def build_profiler_options(self, args: argparse.Namespace) -> list[str]:
        """Build rocprofv3 command-line options for a profiling run."""
        app_cmd = shlex.split(args.remaining)
        trace_option = self._trace_option(args)
        profiling_options = [
            "-d",
            f"{args.output_directory}/out",
            trace_option,
            "--output-format",
            args.format_rocprof_output,
        ]

        if args.attach_pid:
            profiling_options.extend(["--pid", args.attach_pid])
            if args.attach_duration_msec:
                profiling_options.extend([
                    "--attach-duration-msec",
                    args.attach_duration_msec,
                ])

        if args.kernel:
            profiling_options.extend(["--kernel-include-regex", "|".join(args.kernel)])

        dispatch = self._dispatch_ranges(args.dispatch)
        if dispatch:
            profiling_options.extend([
                "--kernel-iteration-range",
                f"[{','.join(dispatch)}]",
            ])

        if not args.attach_pid:
            profiling_options.append("--")
            profiling_options.extend(app_cmd)
        return profiling_options

    def run_pass(
        self,
        fnames: list[str] | str,
        profiler_options: ProfilerOptions,
        workload_dir: str,
        format_rocprof_output: str,
        torch_trace_enabled: bool = False,
        retain_rocpd_output: bool = False,
    ) -> None:
        """Run one rocprofv3 profile pass and finalize its profile data."""
        multiple_files, fbase = get_profile_input_info(fnames)
        if multiple_files:
            console_error(
                "Multiple pmc files detected but rocprofv3 does not "
                "support multiple input files."
            )
            return

        options = ["-A", "absolute", "-i", cast(str, fnames)]
        options.extend(cast(list[str], profiler_options))
        new_env = build_counter_collection_environment(fnames)
        output_path = Path(workload_dir) / "out" / "pmc_1"
        output_path.mkdir(parents=True, exist_ok=True)

        time_1 = time.time()
        console_debug(
            f"rocprof command: {shlex.join([self.profiler_command] + options)}"
        )
        success, output = capture_subprocess_output(
            [self.profiler_command] + options,
            new_env=new_env,
            profileMode=True,
        )
        self._log_profile_duration(time_1)
        self._normalize_pass_directory(workload_dir)
        cleanup_counter_collection_environment(new_env)

        if (not is_live_attach(profiler_options)) and (not success):
            classify_failed_profile_output(output)
            console_error("Profiling execution failed.")

        pass_context = build_profile_pass_context(
            workload_dir=workload_dir,
            fbase=fbase,
            profiler_command=self.profiler_command,
            options=options,
            torch_trace_enabled=torch_trace_enabled,
            retain_rocpd_output=retain_rocpd_output,
        )
        create_profile_data_writer(format_rocprof_output).finalize_pass(pass_context)

    def _trace_option(self, args: argparse.Namespace) -> str:
        if args.kokkos_trace:
            console_error(
                "The option '--kokkos-trace' is not supported in the current "
                "version of rocprof-compute. This functionality is planned for a "
                "future release. Please adjust your profiling options accordingly."
            )
            return "--kokkos-trace"
        if getattr(args, "torch_trace", False):
            return "--marker-trace"
        return "--kernel-trace"

    def _dispatch_ranges(self, dispatch_args: list[str]) -> list[str]:
        dispatch = []
        if dispatch_args:
            for dispatch_id in dispatch_args:
                if ":" in dispatch_id:
                    start, end = dispatch_id.split(":")
                    dispatch.append(f"{start}-{end}")
                else:
                    dispatch.append(f"{dispatch_id}")
        return dispatch

    def _log_profile_duration(self, time_1: float) -> None:
        time_2 = time.time()
        console_debug(
            f"Finishing subprocess of pmc file(s), the time taken is "
            f"{int((time_2 - time_1) / 60)} m {str((time_2 - time_1) % 60)} sec "
        )

    def _normalize_pass_directory(self, workload_dir: str) -> None:
        pass_1 = Path(workload_dir) / "out" / "pass_1"
        if pass_1.exists():
            shutil.copytree(
                pass_1,
                Path(workload_dir) / "out" / "pmc_1",
                dirs_exist_ok=True,
            )


class Rocprofv3AnalysisOrchestrator(ProfileDataAnalysisOrchestrator):
    """Coordinate rocprofv3 analysis profile data access."""

    def __init__(self, options: ProfileDataReaderOptions | None = None) -> None:
        self.options = options or ProfileDataReaderOptions()
