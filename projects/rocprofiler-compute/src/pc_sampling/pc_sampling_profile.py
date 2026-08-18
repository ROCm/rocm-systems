# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import os
import shlex
import time
from dataclasses import dataclass
from typing import Optional, Union, cast

from utils import rocprofv3_avail_interface
from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    capture_subprocess_output,
    get_rocprof_cmd,
    perform_attach_detach,
)
from utils.utils_profile import ProfilerOptions, is_live_attach

# Interval defaults: cycles for stochastic, microseconds for host_trap.
PC_SAMPLING_DEFAULT_INTERVALS = {"stochastic": 1048576, "host_trap": 512}


@dataclass
class PCSamplingLimits:
    """Interval bounds a sampling method accepts, and whether it needs pow2."""

    min_interval: int
    max_interval: int
    interval_pow2: bool = False


def pc_sampling_interval_limits(
    method: str,
    sdk_tool_path: Optional[str] = None,
) -> Optional[PCSamplingLimits]:
    """Return the interval limits the GPUs report for one sampling method.

    Mirrors `rocprofv3-avail info --pc-sampling`.

    None means the agents were queried and none of them offers a configuration
    for `method`. That is distinct from being unable to query at all, which
    yields the SDK fallback limits: rocprofiler-sdk rejects a configuration no
    agent accepts, so the caller has to stop rather than guess a range.
    """
    # Limits rocprofiler-sdk falls back to, see its
    # source/lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.cpp
    fallback = PCSamplingLimits(
        min_interval=1,
        max_interval=1048576,
        interval_pow2=method == "stochastic",
    )

    try:
        configs = rocprofv3_avail_interface.get_pc_sample_configs(sdk_tool_path)
    except (AttributeError, OSError, ValueError) as err:
        console_debug(f"PC sampling interval limit query failed: {err}")
        return fallback

    if configs is None:
        return fallback

    return _merge_interval_limits(configs).get(method)


def _merge_interval_limits(
    configs: list[tuple[int, ...]],
) -> dict[str, PCSamplingLimits]:
    """Merge every agent's configurations into per-method interval limits.

    The SDK configures PC sampling when any single agent supports the request,
    so the accepted range is the union across agents.
    """
    # Keyed on (method id, unit id) from rocprofiler-sdk/fwd.h, since the SDK
    # matches a requested configuration on both fields.
    supported = {(1, 2): "stochastic", (2, 3): "host_trap"}
    limits: dict[str, PCSamplingLimits] = {}
    for method_id, unit, minimum, maximum, flags in configs:
        method = supported.get((method_id, unit))
        if method is None:
            continue
        known = limits.setdefault(
            method, PCSamplingLimits(min_interval=minimum, max_interval=maximum)
        )
        known.min_interval = min(known.min_interval, minimum)
        known.max_interval = max(known.max_interval, maximum)
        # INTERVAL_POW2 is bit 0 of the configuration flags.
        known.interval_pow2 = known.interval_pow2 or bool(flags & 1)
    return limits


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
