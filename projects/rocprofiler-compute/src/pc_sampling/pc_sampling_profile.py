# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import ctypes
import os
import shlex
import time
from pathlib import Path
from typing import Optional, Union, cast

from utils.logger import console_debug, console_error, console_log
from utils.utils_common import (
    PC_SAMPLING_BLOCK_IDS,
    capture_subprocess_output,
    get_rocprof_cmd,
    perform_attach_detach,
    resolve_rocm_library_path,
)
from utils.utils_profile import ProfilerOptions, is_live_attach

# Interval defaults: cycles for stochastic, microseconds for host_trap.
PC_SAMPLING_DEFAULT_INTERVALS = {"stochastic": 1048576, "host_trap": 512}


def pc_sampling_interval_limits(
    method: str,
    sdk_tool_path: Optional[str] = None,
) -> dict[str, int]:
    """Return the interval limits the GPUs report for one sampling method.

    Mirrors `rocprofv3-avail info --pc-sampling`.
    """
    # Limits rocprofiler-sdk falls back to, see its
    # source/lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.cpp
    fallback = {
        "min_interval": 1,
        "max_interval": 1048576,
        "interval_pow2": method == "stochastic",
    }

    library = _load_avail_library(sdk_tool_path)
    if library is None:
        return fallback

    try:
        return _query_agent_interval_limits(library).get(method, fallback)
    except (AttributeError, OSError, ValueError) as err:
        console_debug(f"PC sampling interval limit query failed: {err}")
        return fallback


def _load_avail_library(sdk_tool_path: Optional[str]) -> Optional[ctypes.CDLL]:
    """Load the avail library sitting beside the rocprofiler-sdk tool.

    The library reports no PC sampling configuration unless the beta gate is
    set before it initializes, so it is warmed up while the gate is in place.
    """
    if not sdk_tool_path:
        return None

    library_path = resolve_rocm_library_path(
        str(Path(sdk_tool_path).parent / "librocprofv3-list-avail.so")
    )
    if not library_path or not Path(library_path).exists():
        console_debug("PC sampling: librocprofv3-list-avail.so not found")
        return None

    beta_gate = "ROCPROFILER_PC_SAMPLING_BETA_ENABLED"
    previous_gate = os.environ.get(beta_gate)
    os.environ[beta_gate] = "on"
    try:
        library = ctypes.CDLL(library_path)
        library.get_number_of_agents.restype = ctypes.c_ulong
        library.get_number_of_pc_sample_configs.restype = ctypes.c_ulong
        library.pc_sample_config.argtypes = [ctypes.c_ulong, ctypes.c_ulong] + [
            ctypes.POINTER(ctypes.c_ulong)
        ] * 5
        library.get_number_of_agents()
        return library
    except OSError as err:
        console_debug(f"PC sampling: unable to load {library_path}: {err}")
        return None
    finally:
        if previous_gate is None:
            os.environ.pop(beta_gate, None)
        else:
            os.environ[beta_gate] = previous_gate


def _query_agent_configs(
    library: ctypes.CDLL,
    agent_handle: int,
) -> list[tuple[int, ...]]:
    """Return (method, unit, min, max, flags) for each config of one agent."""
    configs = []
    for config_index in range(library.get_number_of_pc_sample_configs(agent_handle)):
        fields = [ctypes.c_ulong() for _ in range(5)]
        library.pc_sample_config(
            agent_handle, config_index, *(ctypes.byref(field) for field in fields)
        )
        configs.append(tuple(field.value for field in fields))
    return configs


def _query_agent_interval_limits(library: ctypes.CDLL) -> dict[str, dict[str, int]]:
    """Merge every agent's configurations into per-method interval limits.

    The SDK configures PC sampling when any single agent supports the request,
    so the accepted range is the union across agents.
    """
    agent_count = library.get_number_of_agents()
    library.agent_handles.argtypes = [ctypes.c_ulong * agent_count, ctypes.c_ulong]
    agent_handles = (ctypes.c_ulong * agent_count)()
    library.agent_handles(agent_handles, agent_count)

    # (method, unit) ids from rocprofiler-sdk/pc_sampling.h
    supported = {(1, 2): "stochastic", (2, 3): "host_trap"}
    limits: dict[str, dict[str, int]] = {}
    for agent_handle in agent_handles:
        for method_id, unit, minimum, maximum, flags in _query_agent_configs(
            library, agent_handle
        ):
            method = supported.get((method_id, unit))
            if method is None:
                continue
            known = limits.setdefault(
                method,
                {
                    "min_interval": minimum,
                    "max_interval": maximum,
                    "interval_pow2": False,
                },
            )
            known["min_interval"] = min(known["min_interval"], minimum)
            known["max_interval"] = max(known["max_interval"], maximum)
            # INTERVAL_POW2 is bit 0 of the configuration flags.
            known["interval_pow2"] = known["interval_pow2"] or bool(flags & 1)
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
