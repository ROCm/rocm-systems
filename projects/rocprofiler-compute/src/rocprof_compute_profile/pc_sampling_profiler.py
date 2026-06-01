# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
import shlex
import time
from pathlib import Path
from typing import Union, cast

from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    PC_SAMPLING_OUTPUT_FILE_NAME,
    capture_subprocess_output,
    get_rocprof_cmd,
    is_only_pc_sampling,
    pc_sampling_unit,
    perform_attach_detach,
)
from utils.utils_profile import ProfilerOptions, is_live_attach


class PCSamplingProfiler:
    """Standalone PC sampling profile pass.

    Encapsulates the rocprof launch, env/option construction, output-path
    cleanup, and timing/logging for a single PC sampling collection. The
    counter-collection profiler delegates to this class when block 21 is
    requested.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        profiler: str,
        workload_dir: Union[str, Path],
    ) -> None:
        self._args = args
        self._profiler = profiler
        self._workload_dir = str(workload_dir)

    def is_requested(self) -> bool:
        return any(block in PC_SAMPLING_BLOCK_IDS for block in self._args.filter_blocks)

    def is_exclusive(self) -> bool:
        return is_only_pc_sampling(self._args.filter_blocks)

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
        """Remove leftover ``ps_file_*`` outputs from a prior PC-sampling-only
        sdk run.

        PC sampling writes ``ps_file_*`` directly into the workload directory
        (see ``_launch_sdk``). When re-profiling into an existing directory we
        delete those stale files so the new run does not mix in old results.
        """
        if not (self.is_exclusive() and self._profiler == "rocprofiler-sdk"):
            return
        workload_dir = Path(self._workload_dir)
        if not workload_dir.is_dir():
            return
        for stale in workload_dir.glob(f"{PC_SAMPLING_OUTPUT_FILE_NAME}_*"):
            try:
                stale.unlink()
            except OSError:
                console_debug(f"Failed to remove stale PC sampling output: {stale}")
            else:
                console_debug(f"Removed stale PC sampling output: {stale}")

    def _launch(
        self,
        profiler_options: ProfilerOptions,
    ) -> None:
        """Run rocprof with pc sampling. Current support v3 only."""
        # Todo:
        #   - precheck with rocprofv3 --list-avail
        if self._profiler == "rocprofiler-sdk":
            self._launch_sdk(cast(dict[str, Union[str, list[str]]], profiler_options))
        else:
            self._launch_v3(cast(list[str], profiler_options))

    def _launch_sdk(
        self,
        profiler_options: dict[str, Union[str, list[str]]],
    ) -> None:
        """Launch the sdk PC sampling pass.

        ``profiler_options`` is the authoritative env dict built by
        ``rocprofiler_sdk_profiler.get_profiler_options(pc_sampling=True)``; this
        method merges it into the environment verbatim and never re-derives PC
        sampling keys.
        """
        options = profiler_options.copy()
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        console_debug(f"pc sampling rocprof sdk env vars: {new_env}")

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
    ) -> None:
        method = self._args.pc_sampling_method
        interval = self._args.pc_sampling_interval
        options = [
            "--kernel-trace",
            "--pc-sampling-beta-enabled",
            "--pc-sampling-method",
            method,
            "--pc-sampling-unit",
            pc_sampling_unit(method),
            "--output-format",
            "csv",
            "json",
            "--pc-sampling-interval",
            str(interval),
            "-d",
            self._workload_dir,
            "-o",
            PC_SAMPLING_OUTPUT_FILE_NAME,
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

        rocprof_cmd = get_rocprof_cmd()
        console_debug(f"rocprof command: {shlex.join([rocprof_cmd] + options)}")
        success, _ = capture_subprocess_output(
            [rocprof_cmd] + options,
            new_env=os.environ.copy(),
            profileMode=True,
        )
        if not success:
            console_error("PC sampling failed.")
