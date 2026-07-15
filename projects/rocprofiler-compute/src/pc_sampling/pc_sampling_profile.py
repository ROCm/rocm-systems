# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
import shlex
import time
from typing import Optional, Union, cast

from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    capture_subprocess_output,
    get_rocprof_cmd,
    perform_attach_detach,
)
from utils.utils_profile import ProfilerOptions, is_live_attach


class PCSamplingProfile:
    """Standalone PC sampling profile pass.

    Runs the rocprof launch and timing/logging for a single PC sampling
    collection. The backend builds the profiler options upstream.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        profiler: str,
    ) -> None:
        """Store the run config (args, profiler backend)."""
        self._args = args
        self._profiler = profiler

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

        start_time = time.time()
        self._launch(profiler_options)
        duration = time.time() - start_time

        console_debug(
            "profiling",
            f"The time of pc sampling profiling is {int(duration / 60)} m "
            f"{duration % 60} sec",
        )

    def _launch(
        self,
        profiler_options: ProfilerOptions,
    ) -> None:
        """Run rocprof with pc sampling."""
        if self._profiler == "rocprofiler-sdk":
            self._launch_sdk(cast(dict[str, Union[str, list[str]]], profiler_options))
        else:
            self._launch_v3(cast(list[str], profiler_options))

    def _build_env(
        self,
        options: dict[str, Union[str, list[str]]],
        log_label: str,
    ) -> tuple[Optional[Union[str, list[str]]], dict[str, str]]:
        """Pop APP_CMD, overlay options onto the environment, log the delta."""
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        # Log only the os.environ delta to avoid leaking secrets in shared logs.
        env_delta = {k: v for k, v in new_env.items() if os.environ.get(k) != v}
        console_debug(f"{log_label}: {env_delta}")
        return app_cmd, new_env

    def _run_app(
        self,
        app_cmd: Optional[Union[str, list[str]]],
        new_env: dict[str, str],
    ) -> None:
        """Run the workload under the prepared environment."""
        if app_cmd is None:
            console_error(
                "APP_CMD, the workload's executable must be provided "
                "when not in live attach mode"
            )
            return

        success, _ = capture_subprocess_output(
            app_cmd, new_env=new_env, profileMode=True
        )
        if not success:
            console_error("PC sampling failed.")

    def _launch_sdk(
        self,
        profiler_options: dict[str, Union[str, list[str]]],
    ) -> None:
        """Launch the rocprofiler-sdk backend for PC sampling via env vars."""
        options = profiler_options.copy()
        app_cmd, new_env = self._build_env(options, "pc sampling rocprof sdk env vars")

        if is_live_attach(profiler_options):
            perform_attach_detach(new_env, options)
            return

        if app_cmd is not None:
            console_debug(f"pc sampling rocprof sdk user provided command: {app_cmd}")
        self._run_app(app_cmd, new_env)

    def _launch_v3(
        self,
        profiler_options: list[str],
    ) -> None:
        """Launch the rocprofv3 CLI backend for PC sampling via flags."""
        rocprof_cmd = get_rocprof_cmd()
        console_debug(
            f"rocprof command: {shlex.join([rocprof_cmd] + profiler_options)}"
        )
        success, _ = capture_subprocess_output(
            [rocprof_cmd] + profiler_options,
            new_env=os.environ.copy(),
            profileMode=True,
        )
        if not success:
            console_error("PC sampling failed.")
