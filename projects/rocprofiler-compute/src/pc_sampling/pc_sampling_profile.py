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

AVAIL_LIBRARY_NAME = "librocprofv3-list-avail.so"

# The avail library reports no PC sampling configurations unless the beta gate
# is set before it initializes, the same way rocprofv3-avail sets it.
PC_SAMPLING_BETA_ENV = "ROCPROFILER_PC_SAMPLING_BETA_ENABLED"

# Enum values from rocprofiler_pc_sampling_method_t and
# rocprofiler_pc_sampling_unit_t in rocprofiler-sdk/pc_sampling.h.
PC_SAMPLING_METHOD_IDS = {1: "stochastic", 2: "host_trap"}
PC_SAMPLING_METHOD_UNIT_IDS = {"stochastic": 2, "host_trap": 3}
PC_SAMPLING_FLAG_INTERVAL_POW2 = 1

PC_SAMPLING_DEFAULT_INTERVALS = {"stochastic": 1048576, "host_trap": 512}

# Used when the GPU cannot be queried. The ceiling mirrors the 1 MB clamp the
# SDK applies to the driver-reported maximum.
PC_SAMPLING_STATIC_INTERVAL_LIMITS = {
    "stochastic": {
        "min_interval": 65536,
        "max_interval": 1048576,
        "interval_pow2": True,
    },
    "host_trap": {
        "min_interval": 1,
        "max_interval": 1048576,
        "interval_pow2": False,
    },
}


def query_pc_sampling_configs(
    sdk_tool_path: Optional[str] = None,
) -> dict[str, dict[str, int]]:
    """Return the PC sampling interval limits reported by the local GPUs,
    keyed by sampling method.

    Mirrors `rocprofv3-avail info --pc-sampling` by loading
    librocprofv3-list-avail.so and walking each agent's configurations.
    Returns an empty dict when no GPU agent can be queried.
    """
    previous_beta_gate = os.environ.get(PC_SAMPLING_BETA_ENV)
    os.environ[PC_SAMPLING_BETA_ENV] = "on"
    try:
        library = _load_avail_library(sdk_tool_path)
        if library is None:
            return {}
        return _merge_agent_interval_limits(library)
    except (AttributeError, OSError, ValueError) as err:
        console_debug(f"PC sampling configuration query failed: {err}")
        return {}
    finally:
        if previous_beta_gate is None:
            os.environ.pop(PC_SAMPLING_BETA_ENV, None)
        else:
            os.environ[PC_SAMPLING_BETA_ENV] = previous_beta_gate


def pc_sampling_interval_limits(
    method: str,
    sdk_tool_path: Optional[str] = None,
) -> dict[str, int]:
    """Return the interval limits for one sampling method, falling back to
    static limits when the GPU cannot be queried."""
    queried = query_pc_sampling_configs(sdk_tool_path).get(method)
    if queried is not None:
        return queried

    console_debug(f"PC sampling: using static interval limits for {method}")
    return PC_SAMPLING_STATIC_INTERVAL_LIMITS[method]


def _resolve_avail_library_path(sdk_tool_path: Optional[str]) -> Optional[str]:
    """Locate the avail library next to the SDK tool, or under ROCM_PATH."""
    if sdk_tool_path:
        candidate = Path(sdk_tool_path).parent / AVAIL_LIBRARY_NAME
    else:
        candidate = (
            Path(os.getenv("ROCM_PATH", "/opt/rocm"))
            / "lib/rocprofiler-sdk"
            / AVAIL_LIBRARY_NAME
        )
    return resolve_rocm_library_path(str(candidate))


def _load_avail_library(sdk_tool_path: Optional[str]) -> Optional[ctypes.CDLL]:
    """Load the avail library, returning None when it is unusable."""
    library_path = _resolve_avail_library_path(sdk_tool_path)
    if not library_path or not Path(library_path).exists():
        console_debug(f"PC sampling: {AVAIL_LIBRARY_NAME} not found")
        return None

    try:
        return ctypes.CDLL(library_path)
    except OSError as err:
        console_debug(f"PC sampling: unable to load {library_path}: {err}")
        return None


def _query_agent_handles(library: ctypes.CDLL) -> list[int]:
    """Return the handles of every agent the avail library reports."""
    library.get_number_of_agents.restype = ctypes.c_ulong
    agent_count = library.get_number_of_agents()
    if agent_count == 0:
        return []

    library.agent_handles.argtypes = [ctypes.c_ulong * agent_count, ctypes.c_ulong]
    agent_handles = (ctypes.c_ulong * agent_count)()
    library.agent_handles(agent_handles, agent_count)
    return list(agent_handles)


def _query_agent_pc_sampling_configs(
    library: ctypes.CDLL,
    agent_handle: int,
) -> list[dict[str, int]]:
    """Return every PC sampling configuration supported by one agent."""
    library.get_number_of_pc_sample_configs.argtypes = [ctypes.c_ulong]
    library.get_number_of_pc_sample_configs.restype = ctypes.c_ulong
    config_count = library.get_number_of_pc_sample_configs(agent_handle)
    if config_count == 0:
        return []

    library.pc_sample_config.argtypes = [ctypes.c_ulong, ctypes.c_ulong] + [
        ctypes.POINTER(ctypes.c_ulong)
    ] * 5

    configs = []
    for config_index in range(config_count):
        method = ctypes.c_ulong()
        unit = ctypes.c_ulong()
        min_interval = ctypes.c_ulong()
        max_interval = ctypes.c_ulong()
        flags = ctypes.c_ulong()
        library.pc_sample_config(
            agent_handle,
            config_index,
            ctypes.byref(method),
            ctypes.byref(unit),
            ctypes.byref(min_interval),
            ctypes.byref(max_interval),
            ctypes.byref(flags),
        )
        configs.append({
            "method": method.value,
            "unit": unit.value,
            "min_interval": min_interval.value,
            "max_interval": max_interval.value,
            "flags": flags.value,
        })
    return configs


def _merge_agent_interval_limits(library: ctypes.CDLL) -> dict[str, dict[str, int]]:
    """Widen the interval range across agents and keep the strictest flags.

    The SDK configures PC sampling when any single agent supports the
    requested configuration, so the accepted range is the union across agents.
    """
    limits: dict[str, dict[str, int]] = {}
    for agent_handle in _query_agent_handles(library):
        for config in _query_agent_pc_sampling_configs(library, agent_handle):
            method = PC_SAMPLING_METHOD_IDS.get(config["method"])
            if method is None:
                continue
            if config["unit"] != PC_SAMPLING_METHOD_UNIT_IDS[method]:
                continue

            requires_pow2 = bool(config["flags"] & PC_SAMPLING_FLAG_INTERVAL_POW2)
            known = limits.get(method)
            if known is None:
                limits[method] = {
                    "min_interval": config["min_interval"],
                    "max_interval": config["max_interval"],
                    "interval_pow2": requires_pow2,
                }
                continue

            known["min_interval"] = min(known["min_interval"], config["min_interval"])
            known["max_interval"] = max(known["max_interval"], config["max_interval"])
            known["interval_pow2"] = known["interval_pow2"] or requires_pow2
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
