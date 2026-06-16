# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""rocprofiler-sdk profile and analysis orchestration."""

from __future__ import annotations

import argparse
import os
import shlex
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
    build_rocpd_counter_options,
    classify_failed_profile_output,
    cleanup_counter_collection_environment,
    get_profile_input_info,
    is_live_attach,
)
from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    capture_subprocess_output,
    perform_attach_detach,
    resolve_rocm_library_path,
)


class RocprofilerSdkProfileOrchestrator:
    """Coordinate rocprofiler-sdk execution and profile data finalization."""

    profiler_command = "rocprofiler-sdk"

    def build_profiler_options(
        self,
        args: argparse.Namespace,
        native_tool_path: str | None = None,
    ) -> dict[str, str | list[str]]:
        """Build rocprofiler-sdk environment options for a profiling run."""
        app_cmd = shlex.split(args.remaining)
        ld_preload_value = self._ld_preload_value(args, native_tool_path)
        if native_tool_path:
            options = {"ROCPROF_COUNTER_COLLECTION": "0"}
            console_log(
                f"Using native counter collection tool: {str(native_tool_path)}"
            )
        else:
            options = {"ROCPROF_COUNTER_COLLECTION": "1"}

        options.update({
            "LD_PRELOAD": ld_preload_value,
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_OUTPUT_FORMAT": args.format_rocprof_output,
            "ROCPROF_OUTPUT_PATH": f"{args.output_directory}/out/pmc_1",
        })

        if getattr(args, "torch_trace", False):
            options["ROCPROF_MARKER_API_TRACE"] = "1"
        Path(options["ROCPROF_OUTPUT_PATH"]).mkdir(parents=True, exist_ok=True)

        if args.iteration_multiplexing:
            options["ROCPROF_ITERATION_MULTIPLEXING"] = args.iteration_multiplexing

        if args.attach_pid:
            self._add_attach_options(options, args, native_tool_path)

        if args.kokkos_trace:
            console_error(
                "The option '--kokkos-trace' is not supported in the current "
                "version of rocprof-compute. This functionality is planned for a "
                "future release. Please adjust your profiling options accordingly."
            )

        if args.kernel:
            options["ROCPROF_KERNEL_FILTER_INCLUDE_REGEX"] = "|".join(args.kernel)

        dispatch = self._dispatch_ranges(args.dispatch)
        if dispatch:
            options["ROCPROF_KERNEL_FILTER_RANGE"] = f"[{','.join(dispatch)}]"
        if not args.attach_pid:
            options["APP_CMD"] = app_cmd
        return options

    def run_pass(
        self,
        fnames: list[str] | str,
        profiler_options: ProfilerOptions,
        workload_dir: str,
        format_rocprof_output: str,
        torch_trace_enabled: bool = False,
        retain_rocpd_output: bool = False,
    ) -> None:
        """Run one rocprofiler-sdk profile pass and finalize its profile data."""
        _, fbase = get_profile_input_info(fnames)
        options = build_rocpd_counter_options(
            fnames,
            cast(dict[str, str | list[str]], profiler_options),
        )
        new_env = build_counter_collection_environment(fnames)
        output_path = Path(workload_dir) / "out" / "pmc_1"
        output_path.mkdir(parents=True, exist_ok=True)

        time_1 = time.time()
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        for key, value in options.items():
            new_env[key] = value
        env_delta = {k: v for k, v in new_env.items() if os.environ.get(k) != v}
        console_debug(f"rocprof sdk env vars: {env_delta}")

        success = True
        output = ""
        if is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
        else:
            if app_cmd is None:
                console_error(
                    "APP_CMD, the workload's executable must be provided "
                    "when not in live attach mode"
                )
            console_debug(f"rocprof sdk user provided command: {app_cmd}")
            success, output = capture_subprocess_output(
                app_cmd,
                new_env=new_env,
                profileMode=True,
            )

        self._log_profile_duration(time_1)
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

    def _ld_preload_value(
        self,
        args: argparse.Namespace,
        native_tool_path: str | None,
    ) -> str:
        ld_preload_parts = [
            os.environ.get("LD_PRELOAD"),
            args.rocprofiler_sdk_tool_path,
            native_tool_path,
        ]
        return ":".join(part for part in ld_preload_parts if part)

    def _add_attach_options(
        self,
        options: dict[str, str | list[str]],
        args: argparse.Namespace,
        native_tool_path: str | None,
    ) -> None:
        attach_tools = [args.rocprofiler_sdk_tool_path]
        if native_tool_path:
            attach_tools.append(native_tool_path)
        options["ROCPROF_ATTACH_TOOL_LIBRARY"] = ":".join(attach_tools)
        options.pop("LD_PRELOAD", None)
        options["ROCPROF_ATTACH_LIBRARY"] = self._attach_library_path(args)
        options["ROCPROF_ATTACH_PID"] = args.attach_pid
        if args.attach_duration_msec:
            options["ROCPROF_ATTACH_DURATION"] = args.attach_duration_msec

    def _attach_library_path(self, args: argparse.Namespace) -> str:
        rocprofiler_attach_library_path = resolve_rocm_library_path(
            str(
                Path(args.rocprofiler_sdk_tool_path).parent.parent
                / "librocprofiler-sdk-rocattach.so"
            )
        )
        if not Path(rocprofiler_attach_library_path).exists():
            console_debug(
                f"Latest live attach library not found at "
                f"{rocprofiler_attach_library_path}, "
                "searching for legacy live attach library"
            )
            rocprofiler_attach_library_path = resolve_rocm_library_path(
                str(
                    Path(args.rocprofiler_sdk_tool_path).parent
                    / "librocprofv3-attach.so"
                )
            )
        if not Path(rocprofiler_attach_library_path).exists():
            console_error(
                f"No live attach library found at {rocprofiler_attach_library_path}."
            )
        return rocprofiler_attach_library_path

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


class RocprofilerSdkAnalysisOrchestrator(ProfileDataAnalysisOrchestrator):
    """Coordinate rocprofiler-sdk analysis profile data access."""

    def __init__(self, options: ProfileDataReaderOptions | None = None) -> None:
        self.options = options or ProfileDataReaderOptions()
