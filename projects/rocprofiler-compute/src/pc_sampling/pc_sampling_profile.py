# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import Union, cast

from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    capture_subprocess_output,
    get_rocprof_cmd,
    is_only_pc_sampling,
    perform_attach_detach,
)
from utils.utils_profile import ProfilerOptions, is_live_attach


class PCSamplingProfile:
    """Standalone PC sampling profile pass.

    Encapsulates the rocprof launch, env/option construction, output-path
    cleanup, and timing/logging for a single PC sampling collection.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        profiler: str,
        workload_dir: Union[str, Path],
    ) -> None:
        """Store the run config (args, profiler backend, output directory)."""
        self._args = args
        self._profiler = profiler
        self._workload_dir = str(workload_dir)

    def is_requested(self) -> bool:
        """Return True if a PC sampling block (21 / pc_sampling) was requested."""
        return any(block in PC_SAMPLING_BLOCK_IDS for block in self._args.filter_blocks)

    def run(
        self,
        profiler_options: ProfilerOptions,
        prior_run_count: int,
    ) -> None:
        """Execute the PC sampling pass and log timing."""
        console_log(
            f"[Run {prior_run_count + 1}/{prior_run_count + 1}]"
            "[PC sampling profile run]"
        )

        self._cleanup_stale_output(profiler_options)

        start_time = time.time()
        self._launch(profiler_options)
        duration = time.time() - start_time

        console_debug(
            "profiling",
            f"The time of pc sampling profiling is {int(duration / 60)} m "
            f"{duration % 60} sec",
        )

    def _cleanup_stale_output(
        self,
        profiler_options: ProfilerOptions,
    ) -> None:
        """Remove a leftover ROCPROF_OUTPUT_PATH in PC-sampling-only sdk runs."""
        if not (
            is_only_pc_sampling(self._args.filter_blocks)
            and self._profiler == "rocprofiler-sdk"
        ):
            return
        if not isinstance(profiler_options, dict):
            return
        rocprof_output_path = profiler_options.get("ROCPROF_OUTPUT_PATH")
        if rocprof_output_path is None:
            return
        rocprof_output_path = Path(rocprof_output_path)
        if rocprof_output_path.exists():
            shutil.rmtree(rocprof_output_path, ignore_errors=True)
            if rocprof_output_path.exists():
                console_debug(
                    "Failed to remove existing ROCProf output path: "
                    f"{rocprof_output_path}"
                )
            else:
                console_debug(
                    f"Removed existing ROCProf output path: {rocprof_output_path}"
                )

    def _launch(
        self,
        profiler_options: ProfilerOptions,
    ) -> None:
        """Run rocprof with pc sampling. Current support v3 only."""
        # Todo:
        #   - precheck with rocprofv3 --list-avail
        method = self._args.pc_sampling_method
        interval = self._args.pc_sampling_interval
        unit = "time" if method == "host_trap" else "cycles"

        if self._profiler == "rocprofiler-sdk":
            self._launch_sdk(
                cast(dict[str, Union[str, list[str]]], profiler_options),
                method,
                interval,
                unit,
            )
        else:
            self._launch_v3(cast(list[str], profiler_options), method, interval, unit)

    def _launch_sdk(
        self,
        profiler_options: dict[str, Union[str, list[str]]],
        method: str,
        interval: int,
        unit: str,
    ) -> None:
        """Launch the rocprofiler-sdk backend for PC sampling via env vars."""
        options = profiler_options.copy()
        options.update({
            # no counter collection for pc sampling
            "ROCPROF_COUNTER_COLLECTION": "0",
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_OUTPUT_FORMAT": "csv,json",
            "ROCPROF_OUTPUT_PATH": self._workload_dir,
            "ROCPROF_OUTPUT_FILE_NAME": "ps_file",
            "ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "1",
            "ROCPROF_PC_SAMPLING_UNIT": unit,
            "ROCPROF_PC_SAMPLING_INTERVAL": str(interval),
            "ROCPROF_PC_SAMPLING_METHOD": method,
        })
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        # Log only the os.environ delta to avoid leaking secrets in shared logs.
        env_delta = {k: v for k, v in new_env.items() if os.environ.get(k) != v}
        console_debug(f"pc sampling rocprof sdk env vars: {env_delta}")

        if is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
            return

        if app_cmd is None:
            console_error(
                "APP_CMD, the workload's executable must be provided "
                "when not in live attach mode"
            )
            return

        console_debug(f"pc sampling rocprof sdk user provided command: {app_cmd}")
        success, _ = capture_subprocess_output(
            app_cmd, new_env=new_env, profileMode=True
        )
        if not success:
            console_error("PC sampling failed.")

    def _launch_v3(
        self,
        profiler_options: list[str],
        method: str,
        interval: int,
        unit: str,
    ) -> None:
        """Launch the rocprofv3 CLI backend for PC sampling via flags."""
        options = [
            "--kernel-trace",
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            method,
            "--pc-sampling-unit",
            unit,
            "--output-format",
            "csv",
            "json",
            "--pc-sampling-interval",
            str(interval),
            "-d",
            self._workload_dir,
            "-o",
            "ps_file",  # TODO: sync up with the name from source in 2100_.yaml
        ]

        if is_live_attach(profiler_options):
            try:
                pid_idx = profiler_options.index("--pid")
                options += ["--pid", profiler_options[pid_idx + 1]]
                if "--attach-duration-msec" in profiler_options:
                    dur_idx = profiler_options.index("--attach-duration-msec")
                    options += [
                        "--attach-duration-msec",
                        profiler_options[dur_idx + 1],
                    ]
            except (ValueError, IndexError):
                console_error(
                    "--pid or --attach-duration-msec option not found in "
                    "profiler arguments for live attach mode"
                )
                return
        else:
            try:
                app_cmd_with_separator = profiler_options[
                    profiler_options.index("--") :
                ]
                options += app_cmd_with_separator
            except ValueError:
                console_error(
                    "APP_CMD, the workload's executable must be provided "
                    "when not in live attach mode"
                )
                return

        rocprof_cmd = get_rocprof_cmd()
        console_debug(f"rocprof command: {shlex.join([rocprof_cmd] + options)}")
        success, _ = capture_subprocess_output(
            [rocprof_cmd] + options,
            new_env=os.environ.copy(),
            profileMode=True,
        )
        if not success:
            console_error("PC sampling failed.")
