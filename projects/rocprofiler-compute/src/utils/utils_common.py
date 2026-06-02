# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import csv
import ctypes
import errno
import io
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import config
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
)
from vendored import yaml

# Global constants
METRIC_ID_RE = re.compile(pattern=r"^\d{1,2}(?:\.\d{1,2}){0,2}$")


def canonical_config_arch(gpu_arch: Optional[str]) -> Optional[str]:
    """Map GPU architectures to the shared analysis-config directory name."""
    if gpu_arch is None:
        return None
    if gpu_arch.startswith("gfx115"):
        return "gfx115x"
    return gpu_arch


# Supported expression field names for metric tables
SUPPORTED_FIELD: list[str] = [
    "Value",
    "Minimum",
    "Maximum",
    "Average",
    "Median",
    "Min",
    "Max",
    "Avg",
    "Peak",
    "Peak (Empirical)",
    "Count",
    "Mean",
    "Percent",
    "Std Dev",
    "Q1",
    "Q3",
    "Expression",
    "Pct of Peak",
    # Special keywords for L2 channel
    "Channel",
    "L2 Cache Hit Rate",
    "Requests",
    "L2 Read",
    "L2 Write",
    "L2 Atomic",
    "L2-Fabric Requests",
    "L2-Fabric Read",
    "L2-Fabric Write and Atomic",
    "L2-Fabric Atomic",
    "L2 Read Req",
    "L2 Write Req",
    "L2 Atomic Req",
    "L2-Fabric Read Req",
    "L2-Fabric Write and Atomic Req",
    "L2-Fabric Atomic Req",
    "L2-Fabric Read Latency",
    "L2-Fabric Write Latency",
    "L2-Fabric Atomic Latency",
    "L2-Fabric Read Stall (PCIe)",
    "L2-Fabric Read Stall (Infinity Fabric™)",
    "L2-Fabric Read Stall (HBM)",
    "L2-Fabric Write Stall (PCIe)",
    "L2-Fabric Write Stall (Infinity Fabric™)",
    "L2-Fabric Write Stall (HBM)",
    "L2-Fabric Write Starve",
]

# Global state: rocprof being used by profile mode
_rocprof_cmd = ""


def get_rocprof_cmd() -> str:
    """Get the current rocprof command."""
    return _rocprof_cmd


def set_rocprof_cmd(cmd: str) -> None:
    """Set the rocprof command."""
    global _rocprof_cmd
    _rocprof_cmd = cmd


def version_to_numeric(version_parts: list[int], max_len: int) -> int:
    """Convert version tuple to numeric value using base-1000 positional system."""
    version_numeric = 0
    for i, part in enumerate(version_parts):
        version_numeric += part * (1000 ** (max_len - i - 1))
    return version_numeric


def resolve_rocm_library_path(library_path: Optional[str]) -> Optional[str]:
    """
    Resolve ROCm library path with automatic version fallback.
    Tries exact path first, then falls back to versioned variants
    (e.g., .so.1, .so.1.2.3).
    """
    if not library_path:
        return library_path

    path = Path(library_path)

    # Try exact path first (handles both unversioned and explicit versioned paths)
    if path.exists():
        console_debug(f"Resolved library (exact match): {path}")
        return str(path)

    # Use iterdir to avoid glob metacharacter issues in library_path.
    if not path.parent.is_dir():
        return None
    prefix = f"{path.name}."
    matches = [str(p) for p in path.parent.iterdir() if p.name.startswith(prefix)]

    # First pass: filter to numeric versions and collect version tuples
    version_tuples: list[tuple[list[int], str]] = []
    for candidate in matches:
        # Compute the suffix relative to the requested library path.
        if not candidate.startswith(library_path):
            continue
        suffix = candidate[len(library_path) :]
        # Expect a suffix like ".1" or ".1.2.3"
        if not suffix.startswith("."):
            continue
        parts = suffix.split(".")[1:]  # drop leading empty element
        if not parts:
            continue
        if not all(part.isdigit() for part in parts):
            continue
        version_tuples.append(([int(p) for p in parts], candidate))

    # Find max version length to normalize all versions
    if not version_tuples:
        console_debug(f"ROCm library .so file not found: {library_path}")
        return library_path

    # Second pass: convert to numeric values with normalized length
    max_version_len = max(len(vt[0]) for vt in version_tuples)
    versioned_candidates: list[tuple[int, str]] = []
    for version_parts, candidate in version_tuples:
        version_numeric = version_to_numeric(version_parts, max_version_len)
        versioned_candidates.append((version_numeric, candidate))

    # Select the candidate with the highest numeric version.
    versioned_candidates.sort(key=lambda item: item[0], reverse=True)
    resolved = versioned_candidates[0][1]
    console_debug(f"Resolved library (versioned): {library_path} -> {resolved}")
    return resolved


def is_tcc_channel_counter(counter: str) -> bool:
    return counter.startswith("TCC") and counter.endswith("]")


def add_counter_extra_config_input_yaml(
    data: dict[str, Any],
    counter_name: str,
    description: str,
    expression: str,
    architectures: list[str],
    properties: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Add a new counter to the rocprofiler-sdk dictionary.
    Initialize missing parts if data is empty or incomplete.
    Enforces that 'architectures' and 'properties' are lists
    for correct YAML list serialization.
    Overwrites the counter if it already exists.

    Args:
        data (dict): The loaded YAML dictionary (can be empty).
        counter_name (str): The name of the new counter.
        description (str): Description of the new counter.
        architectures (list): List of architectures for the definitions.
        expression (str): Expression string for the counter.
        properties (list, optional): Optional list of properties, default to empty list.

    Returns:
        dict: Updated YAML dictionary.
    """
    if properties is None:
        properties = []

    # Enforce type checks for YAML list serialization
    if not isinstance(architectures, list):
        raise TypeError(
            f"'architectures' must be a list, got {type(architectures).__name__}"
        )
    if not isinstance(properties, list):
        raise TypeError(f"'properties' must be a list, got {type(properties).__name__}")

    # Initialize the top-level 'rocprofiler-sdk' dict if missing
    if "rocprofiler-sdk" not in data or not isinstance(data["rocprofiler-sdk"], dict):
        data["rocprofiler-sdk"] = {}

    sdk = data["rocprofiler-sdk"]

    # Initialize schema version if missing
    if "counters-schema-version" not in sdk:
        sdk["counters-schema-version"] = 1

    # Initialize counters list if missing or not a list
    if "counters" not in sdk or not isinstance(sdk["counters"], list):
        sdk["counters"] = []

    # Build the new counter dictionary
    new_counter = {
        "name": counter_name,
        "description": description,
        "properties": properties,
        "definitions": [
            {
                "architectures": architectures,
                "expression": expression,
            }
        ],
    }

    # Check if the counter already exists and overwrite if found
    for idx, counter in enumerate(sdk["counters"]):
        if counter.get("name") == counter_name:
            sdk["counters"][idx] = new_counter
            break
    else:
        # Not found, append new counter
        sdk["counters"].append(new_counter)

    return data


def get_version(rocprof_compute_home: Path) -> dict[str, str]:
    """Return ROCm Compute Profiler versioning info"""

    # semantic version info - note that version file(s) can reside in
    # two locations depending on development vs formal install
    search_dirs = [rocprof_compute_home, rocprof_compute_home.parent]
    found = False
    version_dir: Optional[Path] = None
    VER = "unknown"
    SHA = "unknown"
    MODE = "unknown"

    for directory in search_dirs:
        version_file = directory / "VERSION"
        try:
            with open(version_file, encoding="utf-8") as file:
                VER = file.read().replace("\n", "")
                found = True
                version_dir = directory
                break
        except Exception:
            pass
    if not found:
        console_error(f"Cannot find VERSION file at {search_dirs}")

    # git version info
    if version_dir is not None:
        try:
            success, output = capture_subprocess_output(
                ["git", "-C", version_dir, "log", "--pretty=format:%h", "-n", "1"],
            )
            if success:
                SHA = output
                MODE = "dev"
            else:
                raise Exception(output)
        except Exception:
            try:
                sha_file = version_dir / "VERSION.sha"
                with open(sha_file, encoding="utf-8") as file:
                    SHA = file.read().replace("\n", "")
                    MODE = "release"
            except Exception:
                pass

    return {"version": VER, "sha": SHA, "mode": MODE}


def get_version_display(version: str, sha: str, mode: str) -> str:
    """Pretty print versioning info"""
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print(f"rocprofiler-compute version: {version} ({mode})", file=buf)
    print(f"Git revision:     {sha}", file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()


def detect_rocprof(args: argparse.Namespace) -> str:
    """Detect loaded rocprof version. Resolve path and set cmd globally."""
    # Default is rocprofiler-sdk
    if os.environ.get("ROCPROF", "rocprofiler-sdk") == "rocprofiler-sdk":
        if not Path(args.rocprofiler_sdk_tool_path).exists():
            console_error(
                "Could not find rocprofiler-sdk tool at "
                f"{args.rocprofiler_sdk_tool_path}"
            )
        set_rocprof_cmd("rocprofiler-sdk")
        console_debug(f"rocprof_cmd is {get_rocprof_cmd()}")
        console_debug(f"rocprofiler_sdk_tool_path is {args.rocprofiler_sdk_tool_path}")
    else:
        # If ROCPROF is not set to rocprofiler-sdk
        set_rocprof_cmd(os.environ["ROCPROF"])
        rocprof_path = shutil.which(get_rocprof_cmd())
        if not rocprof_path:
            console_error(
                f"Unable to resolve path to {get_rocprof_cmd()} binary. "
                "Please verify installation or set ROCPROF "
                "environment variable with full path."
            )
        rocprof_path = str(Path(rocprof_path.rstrip("\n")).resolve())
        console_debug(f"rocprof_cmd is {get_rocprof_cmd()}")
        console_debug(f"ROC Profiler: {rocprof_path}")
    return get_rocprof_cmd()


def perform_attach_detach(new_env: dict[str, str], options: dict[str, Any]) -> None:
    @contextmanager
    def temporary_env(env_vars: dict[str, str]) -> Generator[None, None, None]:
        """
        Temporarily change the environment variable of this application.
        """
        original_env = os.environ.copy()
        os.environ.update({k: str(v) for k, v in env_vars.items()})
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    with temporary_env(new_env):
        libname = options["ROCPROF_ATTACH_LIBRARY"]

        try:
            c_lib = ctypes.CDLL(libname)
            if c_lib is None:
                console_error(f"Error opening {libname}")
        except Exception as e:
            console_error(f"Error loading {libname}: {e}")

        # Set argument and return types for live attach functions
        try:
            # new live attach API
            c_lib.rocattach_attach.restype = ctypes.c_int
            c_lib.rocattach_attach.argtypes = [ctypes.c_int]
            c_lib.rocattach_detach.restype = ctypes.c_int
            c_lib.rocattach_detach.argtypes = [ctypes.c_int]
        except Exception as e:
            console_debug(
                "Error setting new live attach API argument "
                f"types: {e}, trying legacy live attach API"
            )
            try:
                # old live attach API
                c_lib.attach.restype = ctypes.c_int
                c_lib.attach.argtypes = [ctypes.c_uint]
                c_lib.detach.restype = ctypes.c_int
                c_lib.detach.argtypes = [ctypes.c_uint]
            except Exception as e:
                console_error(f"Error setting live attach function argument types: {e}")

        pid = options["ROCPROF_ATTACH_PID"]
        if pid is None:
            console_error("Live attach mode requires a process ID (ROCPROF_ATTACH_PID)")

        try:
            # new live attach API
            attach_status = c_lib.rocattach_attach(int(pid))
            if attach_status != 0:
                console_error(
                    f"Error attaching to process {pid}, "
                    f"rocattach_attach returned {attach_status}"
                )
        except Exception as e:
            console_debug(
                "Error attaching with latest live attach "
                f"API: {e}, trying legacy live attach API"
            )
            try:
                # old live attach API
                c_lib.attach(int(pid))
            except Exception as e:
                console_error(f"Error attaching to process {pid}: {e}")

        duration = os.environ.get("ROCPROF_ATTACH_DURATION", None)
        if duration is None:
            console_log(
                f"\033[93mAttach to process with ID {pid} is successful, "
                "Press Enter to detach...\033[0m"
            )
            input()
        else:
            console_log(
                f"\033[93mAttach to process with ID {pid} is successful, "
                f"detach will happen in {duration} milliseconds...\033[0m"
            )
            time.sleep(int(duration) / 1000)

        try:
            # new live attach API
            detach_status = c_lib.rocattach_detach(int(pid))
            if detach_status != 0:
                console_error(
                    f"Error detaching from process {pid}, "
                    f"rocattach_detach returned {detach_status}"
                )
        except Exception as e:
            console_debug(
                f"Error detaching with latest live attach API: {e}, "
                "trying detach with legacy live attach API"
            )
            try:
                # old live attach API
                c_lib.detach(int(pid))
            except Exception as e:
                console_error(f"Error detaching from process {pid}: {e}")


def capture_subprocess_output(
    subprocess_args: list[str],
    new_env: Optional[dict[str, str]] = None,
    profileMode: bool = False,
    enable_logging: bool = True,
) -> tuple[bool, str]:
    sanitized_env = (
        None
        if new_env is None
        else {
            k: ":".join(str(i) for i in v) if isinstance(v, list) else str(v)
            for k, v in new_env.items()
        }
    )

    # Use a PTY in profile mode to prevent instrumentation output from
    # being interleaved with workload output.
    if profileMode:
        pty_parent_fd, pty_child_fd = pty.openpty()
        stdout_arg = pty_child_fd
        stderr_arg = pty_child_fd
    else:
        stdout_arg = subprocess.PIPE
        stderr_arg = subprocess.STDOUT

    # env=None is Popen's default (inherit parent env).
    try:
        process = subprocess.Popen(
            subprocess_args,
            stdin=subprocess.PIPE,
            stdout=stdout_arg,
            stderr=stderr_arg,
            universal_newlines=True,
            errors="replace",
            env=sanitized_env,
        )
    except BaseException:
        # Close PTY fds if Popen fails.
        if profileMode:
            os.close(pty_parent_fd)
            os.close(pty_child_fd)
        raise

    if profileMode:
        # Close the child end; parent only reads. errors="replace" skips
        # bad bytes instead of crashing the read loop.
        os.close(pty_child_fd)
        process_stdout = os.fdopen(pty_parent_fd, "r", errors="replace")
    else:
        process_stdout = process.stdout

    # Create buffer for captured process output
    buf = io.StringIO()

    def forward_input() -> None:
        """
        Forward the keyboard input from the terminal to the inside subprocess
        """

        try:
            sys.stdin.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # Stdin can't be used in select; skip input forwarding
            return

        if sys.stdin.isatty():
            for line in sys.stdin:
                if process.poll() is not None:
                    break
                process.stdin.write(line)
                process.stdin.flush()
        else:
            while process.poll() is None:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                except (io.UnsupportedOperation, AttributeError):
                    break
                if rlist:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    process.stdin.write(line)
                    process.stdin.flush()
        try:
            process.stdin.close()
        except Exception:
            console_warning("forward_input: the stdin did not close properly!")

    input_thread = threading.Thread(target=forward_input, daemon=True)
    input_thread.start()

    # Read until the child closes its end. Pipes signal EOF with an empty
    # string; PTYs signal it with OSError(EIO). The two never overlap.
    while True:
        try:
            line = process_stdout.readline()
        except OSError as e:
            if e.errno == errno.EIO:
                break
            raise
        if not line:
            break
        buf.write(line)
        if not enable_logging:
            continue
        if profileMode:
            console_log(get_rocprof_cmd(), line.strip(), indent_level=1)
        else:
            console_log(line.strip())

    process_stdout.close()
    input_thread.join(timeout=1)

    # Get process return code
    return_code = process.wait()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return success, output


def get_agent_dict(data: dict[str, Any]) -> dict[Any, Any]:
    """Create a dictionary that maps agent ID to agent objects."""
    agents = data["rocprofiler-sdk-tool"][0]["agents"]
    agent_map: dict[Any, Any] = {}

    for agent in agents:
        agent_id = agent["id"]["handle"]
        agent_map[agent_id] = agent

    return agent_map


def get_gpuid_dict(data: dict[str, Any]) -> dict[Any, int]:
    """
    Returns a dictionary that maps agent ID to GPU ID starting at 0.
    """
    agents = data["rocprofiler-sdk-tool"][0]["agents"]
    agent_list: list[tuple[Any, int]] = []

    # Get agent ID and node_id for GPU agents only
    for agent in agents:
        if agent["type"] == 2:
            agent_id = agent["id"]["handle"]
            node_id = agent["node_id"]
            agent_list.append((agent_id, node_id))

    # Sort by node ID
    agent_list.sort(key=lambda x: x[1])

    # Map agent ID to node id
    gpu_map: dict[Any, int] = {}
    gpu_id = 0
    for agent_id, _ in agent_list:
        gpu_map[agent_id] = gpu_id
        gpu_id += 1

    return gpu_map


def parse_pmc_perf(pmc_perf_file: str) -> list[str]:
    """
    Parse the YAML file to get the pmc counters.
    Assumes only one job per file.
    """
    with open(pmc_perf_file, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    jobs = data.get("jobs", [])
    if not jobs:
        return []
    return jobs[0].get("pmc") or []


def is_only_pc_sampling(filter_blocks: list[str]) -> bool:
    """Return True if all requested blocks are PC sampling (block 21)."""
    return bool(filter_blocks) and all(
        block in ["21", "pc_sampling"] for block in filter_blocks
    )


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts: list[str] = []

    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    if len(parts) <= 1:
        return parts[0] if parts else "0 seconds"

    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def parse_sets_yaml(arch: str) -> dict[str, Any]:
    config_arch = canonical_config_arch(arch) or arch
    filename = (
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sets"
        / f"{config_arch}_sets.yaml"
    )
    with open(filename, encoding="utf-8") as file:
        content = file.read()
    data = yaml.safe_load(content)

    sets_data = data.get("sets", [])

    sets_info: dict[str, Any] = {}
    for set_item in sets_data:
        set_option = set_item.get("set_option", "")
        if set_option:
            sets_info[set_option] = set_item
    return sets_info


def load_panel_configs(
    dirs: list[str],
) -> OrderedDict[int, dict[str, Any]]:
    """
    Load all panel configs from yaml file.
    """
    configs: dict[int, dict[str, Any]] = {}
    for dir_path in dirs:
        for yaml_file in Path(dir_path).glob("*.yaml"):
            with open(yaml_file, encoding="utf-8") as file:
                config_yml = yaml.safe_load(file)
                # metric key can be None due to some metric-
                # tables not having any metrics
                # metric key should be empty dict instead of None
                panel_config = config_yml["Panel Config"]
                for data_source in panel_config["data source"]:
                    metric_table = data_source.get("metric_table")
                    if metric_table and metric_table["metric"] is None:
                        metric_table["metric"] = {}
                configs[panel_config["id"]] = panel_config

    # TODO: sort metrics as the header order in case they-
    # are not defined in the same order
    return OrderedDict(sorted(configs.items()))


def expand_placeholder_ranges(
    panel_configs: OrderedDict[int, dict[str, Any]],
    sys_info: Optional[dict[str, Any]],
) -> OrderedDict[int, dict[str, Any]]:
    """
    Expand placeholder_range entries in metric_table data configs in-place.

    Mutates panel_configs directly and returns the same object for convenience.

    Some metric tables define a range of metrics via a placeholder key that is
    expanded into individual entries at load time. sys_info is required to
    resolve built-in range variables; if it is None the table is cleared.
    """
    for _panel_id, panel in panel_configs.items():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if type_key != "metric_table":
                    continue
                if "metric" not in data_config:
                    continue
                if "placeholder_range" not in data_config["metric"]:
                    continue
                if sys_info is None:
                    data_config["metric"] = {}
                    continue

                # Resolved here (not in eval_metric) because the range may
                # itself be a built-in var. Single placeholder only.
                p_range = data_config["metric"].pop("placeholder_range")
                metric, metric_expr = data_config["metric"].popitem()
                new_metrics: dict[str, Any] = {}
                for p, r in p_range.items():
                    if isinstance(r, int):
                        var = r
                    elif isinstance(r, str) and r.startswith("$total_l2_chan"):
                        var = int(sys_info["total_l2_chan"])
                    else:
                        console_error(f'Built-in var "{r}" is not supported')
                    for i in range(var):
                        new_key = metric.replace(p, str(i))
                        new_val = {
                            k: v.replace(p, str(i)) for k, v in metric_expr.items()
                        }
                        new_metrics[new_key] = new_val
                data_config["metric"] = new_metrics

    return panel_configs


def _metric_has_valid_expr(entries: dict, data_config: dict) -> bool:
    """
    Return True if a metric entry has at least one evaluatable expression field
    that is not None and not the string "None".

    Expression fields are identified by matching the header display name against
    SUPPORTED_FIELD, excluding Peak-prefixed fields which are empirical values.
    """
    for header_key, header_display in data_config["header"].items():
        if header_display in SUPPORTED_FIELD and not header_display.startswith("Peak"):
            expr_value = entries.get(header_key)
            if expr_value is not None and expr_value != "None":
                return True
    return False


def build_metric_list(
    panel_configs: OrderedDict[int, dict[str, Any]],
    sys_info: Optional[dict[str, Any]],
) -> dict[str, str]:
    """
    Build metric_list from the panel configs.

    Returns a mapping of (panel/table/metric IDs -> display names)
    without constructing DataFrames or metric_counters. Use this directly when
    only the metric listing is needed (e.g. --list-metrics, --list-blocks).
    """
    metric_list: dict[str, str] = {}

    expanded_configs = expand_placeholder_ranges(panel_configs, sys_info)

    for panel_id, panel in expanded_configs.items():
        for data_source in panel["data source"]:
            for type_key, data_config in data_source.items():
                if type_key == "metric_table":
                    data_source_idx = str(data_config["id"] // 100)
                    if data_source_idx != "0":
                        metric_list[data_source_idx] = panel["title"]

                    table_idx = f"{data_config['id'] // 100}.{data_config['id'] % 100}"
                    metric_list[table_idx] = data_config["title"]

                    for i, (key, entries) in enumerate(data_config["metric"].items()):
                        metric_idx = f"{table_idx}.{i}"
                        if _metric_has_valid_expr(entries, data_config):
                            metric_list[metric_idx] = key

                elif type_key in ("raw_csv_table", "pc_sampling_table"):
                    data_source_idx = str(data_config["id"] // 100)
                    metric_list[data_source_idx] = panel["title"]

    return metric_list


def get_uuid(length: int = 8) -> str:
    return uuid.uuid4().hex[:length]


def format_scientific_notation_if_needed(
    value: Union[int, float],
    align: str = ">",
    width_align: int = 6,
    precision: int = 2,
    fmt_type_align: str = "f",
    max_length: int = 6,  # Deprecated: kept for backward compatibility
    sci_lower_bound: float = 1e-2,
    sci_upper_bound: float = 1e6,
) -> str:
    """
    Format a numeric value as normal or scientific notation string.

    Uses scientific notation only if it results in a shorter string than
    normal notation, or if the value falls outside the bounds:
    - abs(value) < sci_lower_bound (but not zero)
    - abs(value) >= sci_upper_bound

    Parameters:
    - value: numeric value to format
    - align: alignment character ('<', '>', '^', '=')
    - width_align: total width of formatted output
    - precision: number of digits after decimal point
    - fmt_type_align: format type, e.g., 'f', 'e', 'g'
    - max_length: deprecated, no longer used
    - sci_lower_bound: lower bound for scientific notation usage
    - sci_upper_bound: upper bound for scientific notation usage

    Returns:
    - formatted string according to the criteria, respecting alignment
    """
    del max_length  # Unused, kept for backward compatibility

    abs_val = abs(value)
    use_sci = False

    # Build format specifiers
    normal_format_spec = f"{align}{width_align}.{precision}{fmt_type_align}"
    sci_format_spec = f"{align}{width_align}.{precision}e"

    normal_str = None  # will hold formatted normal string (with padding)
    sci_str = None  # will hold formatted scientific string (with padding)

    if abs_val != 0:
        if abs_val < sci_lower_bound or abs_val >= sci_upper_bound:
            use_sci = True
        else:
            try:
                normal_str = format(value, normal_format_spec)
                normal_str_strip = normal_str.strip()

                sci_str = format(value, sci_format_spec)
                sci_str_strip = sci_str.strip()

                # Decide based on length of stripped strings (ignore padding)
                # Only use scientific notation if it's actually shorter
                if len(sci_str_strip) < len(normal_str_strip):
                    use_sci = True
            except Exception:
                # Fallback to scientific if formatting fails
                use_sci = True

    if use_sci:
        if sci_str is None:
            sci_str = format(value, sci_format_spec)
        formatted = sci_str
    else:
        if normal_str is None:
            normal_str = format(value, normal_format_spec)
        formatted = normal_str

    return formatted


def convert_filter_blocks_to_panel_ids(
    filter_blocks: list[str], arch: Optional[str] = None
) -> set[int]:
    """Inverse of convert_metric_id_to_panel_info: map metric ids like
    "2" or "11.1" to the set of file_id integers (e.g. {200, 1100}).
    Tokens that are not metric ids are looked up as panel aliases (e.g.
    "lds", "roofline") for the given arch.
    """
    alias_map: dict[str, str] = (
        get_arch_alias_to_panel_id(arch)
        if arch and any(not METRIC_ID_RE.match(str(bid)) for bid in filter_blocks)
        else {}
    )
    resolved: set[int] = set()
    for bid in filter_blocks:
        token = str(bid)
        if not METRIC_ID_RE.match(token):
            if token not in alias_map:
                console_error(
                    f"Invalid --block value {token}. "
                    "Run rocprof-compute --list-blocks to see valid values."
                )
            token = alias_map[token]
        resolved.add(int(convert_metric_id_to_panel_info(token)[0]))
    return resolved


def convert_metric_id_to_panel_info(
    metric_id: str,
) -> tuple[str, Optional[int], Optional[int]]:
    """
    Convert metric id into panel information.
    Output is a tuples of the form (file_id, panel_id, metric_id).

    For example:

    Input: "2"
    Output: ("0200", None, None)

    Input: "11"
    Output: ("1100", None, None)

    Input: "11.1"
    Output: ("1100", 1101, None)

    Input: "11.1.1"
    Output: ("1100", 1101, 1)

    Raises exception for invalid metric id.
    """
    tokens = metric_id.split(".")
    if not (0 < len(tokens) < 4):
        raise ValueError(f"Invalid metric id: {metric_id}")

    # File id
    file_id = str(int(tokens[0]))
    # 4 -> 04
    if len(file_id) == 1:
        file_id = f"0{file_id}"
    # Multiply integer by 100
    file_id = f"{file_id}00"

    # Panel id
    panel_id = None
    if len(tokens) > 1:
        panel_id = int(tokens[0]) * 100 + int(tokens[1])

    # Metric id
    metric_id_int = None
    if len(tokens) > 2:
        metric_id_int = int(tokens[2])

    return (file_id, panel_id, metric_id_int)


def load_yaml(filepath: Union[str, os.PathLike]) -> dict[str, Any]:
    """Load YAML file and return as dictionary."""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_arch_panel_id_to_alias(arch: str) -> dict[str, str]:
    """Return panel_id_str -> alias from the *_config_template.yaml whose
    filename prefix matches arch. Empty/None aliases stay as "".
    Returns {} when no template matches the arch."""
    analysis_dir = (
        Path(config.rocprof_compute_home) / "rocprof_compute_soc" / "analysis_configs"
    )
    for path in sorted(analysis_dir.glob("*_config_template.yaml")):
        m = re.match(r"(gfx\d+)_config_template\.yaml$", path.name)
        if m and arch.startswith(m.group(1)):
            panel_yaml = load_yaml(path) or {}
            panels = panel_yaml.get("panels") or []
            return {str(p["panel_id"]): (p.get("panel_alias") or "") for p in panels}
    return {}


def get_arch_alias_to_panel_id(arch: str) -> dict[str, str]:
    """Inverse of get_arch_panel_id_to_alias for resolving --block
    <alias> tokens: alias -> panel_id_str. Skips panels with empty or
    None aliases so unknown aliases naturally raise KeyError on lookup."""
    return {
        alias: pid for pid, alias in get_arch_panel_id_to_alias(arch).items() if alias
    }


def get_job_rank_and_size() -> tuple[Optional[str], Optional[int]]:
    """Detect job rank and total ranks from runtime environment variables.

    Returns a (rank, total_ranks) tuple, ensuring both values come from the same
    runtime.
    """
    # Note: PMIX_RANK is intentionally excluded. PMIx has no standard size env var,
    # and PMIx is never a standalone launcher — it always runs behind SLURM, OpenMPI,
    # PALS, etc., which set their own paired rank/size vars checked here. If only
    # PMIX_RANK is set, the launcher also sets generic MPI_RANK/MPI_SIZE caught below.
    rank_size_env_vars = [
        ("PBS_NODENUM", "PBS_O_TASKNUM"),  # PBS/Torque
        ("SLURM_PROCID", "SLURM_NTASKS"),  # SLURM
        ("FLUX_TASK_RANK", "FLUX_JOB_SIZE"),  # Flux
        ("PMI_RANK", "PMI_SIZE"),  # PMI
        ("PALS_RANKID", "PALS_WORLD_SIZE"),  # PALS (HPE Cray)
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),  # OpenMPI
        ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),  # MVAPICH2
        ("MPI_RANKID", "MPI_NRANKS"),  # Generic
        ("MPI_LOCALRANKID", "MPI_LOCALNRANKS"),  # Generic (local)
        ("MPI_RANK", "MPI_SIZE"),  # Generic
    ]
    matched_rank = None
    matched_rank_var = None
    matched_size = None
    matched_size_var = None
    for rank_var, size_var in rank_size_env_vars:
        rank_value = os.environ.get(rank_var)
        try:
            _ = int(rank_value)
        except (TypeError, ValueError):
            continue

        # Rank is valid; try to get a matching size for a complete pair
        size_value = os.environ.get(size_var)
        try:
            matched_size = int(size_value)
        except (TypeError, ValueError):
            # Size missing or invalid — remember rank as fallback but keep
            # searching for a runtime that provides both rank and size
            if matched_rank is None:
                matched_rank = rank_value
                matched_rank_var = rank_var
                matched_size_var = size_var
            continue

        # Complete pair found
        matched_rank = rank_value
        matched_rank_var = rank_var
        matched_size_var = size_var
        break

    console_debug(
        f"Parallel runtime detected: {matched_rank_var}='{matched_rank}',"
        f" {matched_size_var}={matched_size}"
    )

    return (matched_rank, matched_size)


def replace_rank(name: str) -> str:
    def rank(match: re.Match[str]) -> str:
        value, _ = get_job_rank_and_size()
        if value is not None:
            return value + match.group(1)  # preserve trailing slash
        else:
            return ""  # Ignore %rank% and trailing slash

    # Replace %rank% (and optional trailing slash) with MPI process rank
    pattern = re.compile(r"%rank%(/?)")

    return pattern.sub(rank, name)


def replace_env(name: str) -> str:
    def env(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")  # Default to empty string if not found

    # Replace %env{VAR}% with environment variable values
    pattern = re.compile(r"%env{([^}]+)}%")

    return pattern.sub(env, name)


def normalize_filter_to_str_list(value: Any) -> list[str]:  # noqa: ANN401
    """Normalize a filter value (scalar or list) to a list of strings."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def print_status(msg: str) -> None:
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def create_temp_rocprofiler_metrics_path(sdk_config: dict[str, Any]) -> str:
    """
    Create temporary directory with rocprofiler metrics config files.

    Writes two config files:
    - counter_defs.yaml: For backward compatibility (excludes firmware restrictions)
    - config.yaml: Current version with full config

    Args:
        sdk_config: The rocprofiler-sdk configuration dictionary.

    Returns:
        Path to the temporary directory (for ROCPROFILER_METRICS_PATH env var).
    """
    tmpfile_parent = Path(
        tempfile.mkdtemp(prefix="rocprof_compute_sdk_config_", dir="/tmp")
    )

    # counter_defs.yaml is for backward compatibility with previous versions of sdk
    tmpfile_path_old = tmpfile_parent / "counter_defs.yaml"
    # Current version of sdk uses config.yaml instead of counter_defs.yaml
    tmpfile_path_new = tmpfile_parent / "config.yaml"

    with open(tmpfile_path_old, "w", encoding="utf-8") as tmpfile:
        # Old sdk does not support firmware restrictions
        sdk_config_old = {
            **sdk_config,
            "rocprofiler-sdk": {
                k: v
                for k, v in sdk_config["rocprofiler-sdk"].items()
                if k not in {"fw-restriction-schema-version", "firmware_restrictions"}
            },
        }
        yaml.dump(sdk_config_old, tmpfile, default_flow_style=False, sort_keys=False)

    with open(tmpfile_path_new, "w", encoding="utf-8") as tmpfile:
        yaml.dump(sdk_config, tmpfile, default_flow_style=False, sort_keys=False)

    return str(tmpfile_parent)


def reconfigure_stdio_utf8() -> None:
    """Force sys.stdout and sys.stderr to UTF-8 so tabulated output never
    crashes with UnicodeEncodeError on systems started under a non-UTF-8
    locale.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass


def validate_roofline_csv(workload_dir: Union[str, Path, list]) -> tuple[bool, str]:
    """
    Validate roofline.csv exists and has consistent structure.

    Returns:
        tuple: (is_valid, error_message)
               is_valid=True if CSV is valid, False otherwise
               error_message contains description if invalid
    """
    if isinstance(workload_dir, list):
        base_dir = (
            workload_dir[0][0]
            if isinstance(workload_dir[0], (list, tuple))
            else workload_dir[0]
        )
    else:
        base_dir = workload_dir

    benchmark_results = Path(base_dir) / "roofline.csv"

    # Check if file exists
    if not benchmark_results.exists():
        return False, f"Benchmark results file not found: {benchmark_results}"

    # Validate CSV structure
    try:
        with open(benchmark_results, newline="", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=",")
            row_count = 0
            num_headers = 0

            for row in csv_reader:
                if row_count == 0:
                    num_headers = len(row) - 1
                    if num_headers <= 0:
                        return (
                            False,
                            "Empty or invalid header row in benchmark_results",
                        )
                else:
                    if len(row) - 1 != num_headers:
                        return (
                            False,
                            f"Inconsistent row length in benchmark_results at "
                            f"row {row_count + 1}. "
                            f"Expected {num_headers + 1} columns, "
                            f"found {len(row)}. "
                            "Roofline data appears corrupted or incomplete.",
                        )
                row_count += 1

            if row_count < 2:
                return (
                    False,
                    f"Insufficient data in benchmark_results. "
                    f"Found {row_count} rows (need at least 2)."
                    f" Roofline data appears corrupted or incomplete.",
                )
    except Exception as e:
        return False, f"Failed to read benchmark_results: {e}"

    return True, ""


def format_table_ascii(
    data: list[dict[str, Any]],
    columns: list[str],
    description_wrap_width: int = 40,
    decimal: int = 2,
) -> str:
    """
    Format a list of dicts as an ASCII table string using only stdlib.

    Args:
        data: List of dictionaries containing row data
        columns: List of column names to display (in order)
        description_wrap_width: Width to wrap Description column (default 40)
        decimal: Number of decimal places for floats (default 2)

    Returns:
        Formatted ASCII table string
    """
    if not data:
        return ""

    # Calculate column widths
    col_widths: dict[str, int] = {}
    wrapped_data: list[dict[str, list[str]]] = []

    for col in columns:
        col_widths[col] = len(col)

    # Process each row and wrap text as needed
    for row in data:
        wrapped_row: dict[str, list[str]] = {}
        for col in columns:
            value = row.get(col, "")
            if value is None:
                value = ""

            # Format floats with specified decimal places (matches tabulate floatfmt)
            if isinstance(value, float):
                value = f"{value:.{decimal}f}"
            elif isinstance(value, int):
                value = str(value)
            else:
                value = str(value)

            # Wrap Description column using textwrap.fill behavior
            # (collapses whitespace, wraps at word boundaries)
            if col == "Description":
                # Use textwrap.fill to match original behavior exactly
                wrapped_text = textwrap.fill(value, width=description_wrap_width)
                lines = wrapped_text.split("\n") if wrapped_text else [""]
            else:
                lines = [value]

            wrapped_row[col] = lines

            # Update column width based on longest line
            for line in lines:
                col_widths[col] = max(col_widths[col], len(line))

        wrapped_data.append(wrapped_row)

    # Build the table
    result: list[str] = []

    # Index column width: max of "index" (5) and the widest row number
    idx_width = max(5, len(str(len(data) - 1)))

    # Create separator line
    def make_separator() -> str:
        parts = ["+"]
        # Index column
        parts.append("-" * (idx_width + 2) + "+")
        for col in columns:
            parts.append("-" * (col_widths[col] + 2) + "+")
        return "".join(parts)

    # Create header
    def make_header() -> str:
        parts = ["|"]
        parts.append(f" {'index':^{idx_width}} |")
        for col in columns:
            parts.append(f" {col:<{col_widths[col]}} |")
        return "".join(parts)

    # Create data row (handles multi-line cells)
    def make_row(row_idx: int, wrapped_row: dict[str, list[str]]) -> list[str]:
        # Find max lines in this row
        max_lines = max(len(lines) for lines in wrapped_row.values())

        row_lines: list[str] = []
        for line_idx in range(max_lines):
            parts = ["|"]
            # Index column - only show on first line
            if line_idx == 0:
                parts.append(f" {row_idx:>{idx_width}} |")
            else:
                parts.append(" " * (idx_width + 1) + " |")

            for col in columns:
                lines = wrapped_row[col]
                if line_idx < len(lines):
                    cell_value = lines[line_idx]
                else:
                    cell_value = ""
                parts.append(f" {cell_value:<{col_widths[col]}} |")
            row_lines.append("".join(parts))
        return row_lines

    # Build table
    separator = make_separator()
    result.append(separator)
    result.append(make_header())
    result.append(separator)

    for idx, wrapped_row in enumerate(wrapped_data):
        row_lines = make_row(idx, wrapped_row)
        for line in row_lines:
            result.append(line)
        result.append(separator)

    return "\n".join(result)
