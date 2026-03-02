##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import argparse
import ctypes
import glob
import io
import json
import locale
import logging
import os
import re
import select
import selectors
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union, cast

import pandas as pd
import yaml

import config
from utils import rocpd_data
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)

METRIC_ID_RE = re.compile(pattern=r"^\d{1,2}(?:\.\d{1,2}){0,2}$")

rocprof_cmd = ""
rocprof_args = ""


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

    # Escape the input path so any glob metacharacters are treated literally.
    matches = glob.glob(f"{glob.escape(library_path)}.*")

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
            with open(version_file) as file:
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
                with open(sha_file) as file:
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
    global rocprof_cmd

    # Default is rocprofiler-sdk
    if os.environ.get("ROCPROF", "rocprofiler-sdk") == "rocprofiler-sdk":
        if not Path(args.rocprofiler_sdk_tool_path).exists():
            console_error(
                "Could not find rocprofiler-sdk tool at "
                f"{args.rocprofiler_sdk_tool_path}"
            )
        rocprof_cmd = "rocprofiler-sdk"
        console_debug(f"rocprof_cmd is {rocprof_cmd}")
        console_debug(f"rocprofiler_sdk_tool_path is {args.rocprofiler_sdk_tool_path}")
    else:
        # If ROCPROF is not set to rocprofiler-sdk
        rocprof_cmd = os.environ["ROCPROF"]
        rocprof_path = shutil.which(rocprof_cmd)
        if not rocprof_path:
            console_error(
                f"Unable to resolve path to {rocprof_cmd} binary. "
                "Please verify installation or set ROCPROF "
                "environment variable with full path."
            )
        rocprof_path = str(Path(rocprof_path.rstrip("\n")).resolve())
        console_debug(f"rocprof_cmd is {str(rocprof_cmd)}")
        console_debug(f"ROC Profiler: {rocprof_path}")
    return rocprof_cmd


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

        # Set argument and return types for attach/detach functions
        try:
            # old attach/detach API
            c_lib.attach.argtypes = [ctypes.c_uint]
        except Exception as e:
            console_debug(
                "Error setting old attach/detach API argument "
                f"types: {e}, trying new API"
            )
            try:
                # new attach/detach API
                c_lib.rocattach_attach.restype = ctypes.c_int
                c_lib.rocattach_attach.argtypes = [ctypes.c_int]
                c_lib.rocattach_detach.restype = ctypes.c_int
                c_lib.rocattach_detach.argtypes = [ctypes.c_int]
            except Exception as e:
                console_error(
                    f"Error setting attach/detach function argument types: {e}"
                )

        pid = options["ROCPROF_ATTACH_PID"]
        if pid is None:
            console_error("Mode of attach/detach must have setup for process ID")

        try:
            # old attach/detach API
            c_lib.attach(int(pid))
        except Exception as e:
            console_debug(f"Error attaching with old API: {e}, trying new API")
            try:
                # new attach/detach API
                attach_status = c_lib.rocattach_attach(int(pid))
                if attach_status != 0:
                    console_error(
                        f"Error attaching to process {pid}, "
                        f"rocattach_attach returned {attach_status}"
                    )
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
            # old attach/detach API
            c_lib.detach(int(pid))
        except Exception as e:
            console_debug(f"Error detaching with old API: {e}, trying new API")
            try:
                # new attach/detach API
                detach_status = c_lib.rocattach_detach(int(pid))
                if detach_status != 0:
                    console_error(
                        f"Error detaching from process {pid}, "
                        f"rocattach_detach returned {detach_status}"
                    )
            except Exception as e:
                console_error(f"Error detaching from process {pid}: {e}")


def capture_subprocess_output(
    subprocess_args: list[str],
    new_env: Optional[dict[str, str]] = None,
    profileMode: bool = False,
    enable_logging: bool = True,
) -> tuple[bool, str]:
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    sanitized_env = (
        None
        if new_env is None
        else {
            k: ":".join(str(i) for i in v) if isinstance(v, list) else str(v)
            for k, v in new_env.items()
        }
    )

    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if sanitized_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=sanitized_env,
        )
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream: io.TextIOWrapper, _mask) -> None:
        try:
            # Because the process' output is line buffered, there's only ever one
            # line to read when this function is called
            line = stream.readline()
            if not line:
                return
            buf.write(line)
            if enable_logging:
                if profileMode:
                    console_log(rocprof_cmd, line.strip(), indent_level=1)
                else:
                    console_log(line.strip())
        except UnicodeDecodeError:
            # Skip this line
            pass

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    if process.stdout is not None:
        selector.register(process.stdout, selectors.EVENT_READ, handle_output)

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

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    input_thread.join(timeout=1)

    # Get process return code
    return_code = process.wait()
    selector.close()

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


def v3_json_get_counters(data: dict[str, Any]) -> dict[tuple[Any, Any], Any]:
    """Create a dictionary that maps (agent_id, counter_id) to counter objects."""
    counters = data["rocprofiler-sdk-tool"][0]["counters"]
    counter_map: dict[tuple[Any, Any], Any] = {}

    for counter in counters:
        counter_id = counter["id"]["handle"]
        agent_id = counter["agent_id"]["handle"]
        counter_map[(agent_id, counter_id)] = counter

    return counter_map


def v3_json_get_dispatches(data: dict[str, Any]) -> dict[Any, Any]:
    """Create a dictionary that maps correlation_id to dispatch records."""
    records = data["rocprofiler-sdk-tool"][0]["buffer_records"]
    records_map: dict[Any, Any] = {}

    for rec in records["kernel_dispatch"]:
        id = rec["correlation_id"]["internal"]
        records_map[id] = rec

    return records_map


def v3_json_to_csv(json_file_path: str, csv_file_path: str) -> None:
    with open(json_file_path) as f:
        data = json.load(f)

    dispatch_records = v3_json_get_dispatches(data)
    dispatches = data["rocprofiler-sdk-tool"][0]["callback_records"][
        "counter_collection"
    ]
    kernel_symbols = data["rocprofiler-sdk-tool"][0]["kernel_symbols"]
    agents = get_agent_dict(data)
    pid = data["rocprofiler-sdk-tool"][0]["metadata"]["pid"]
    gpuid_map = get_gpuid_dict(data)
    counter_info = v3_json_get_counters(data)

    # CSV headers. If there are no dispatches we still end up with a valid CSV file.
    csv_data: dict[str, list[Any]] = {
        key: []
        for key in [
            "Dispatch_ID",
            "GPU_ID",
            "Queue_ID",
            "PID",
            "TID",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
            "Scratch_Per_Workitem",
            "Arch_VGPR",
            "Accum_VGPR",
            "SGPR",
            "Wave_Size",
            "Kernel_Name",
            "Start_Timestamp",
            "End_Timestamp",
            "Correlation_ID",
        ]
    }

    for d in dispatches:
        dispatch_info = d["dispatch_data"]["dispatch_info"]
        agent_id = dispatch_info["agent_id"]["handle"]
        kernel_id = dispatch_info["kernel_id"]

        row: dict[str, Any] = {}
        row["Dispatch_ID"] = dispatch_info["dispatch_id"]
        row["GPU_ID"] = gpuid_map[agent_id]
        row["Queue_ID"] = dispatch_info["queue_id"]["handle"]
        row["PID"] = pid
        row["TID"] = d["thread_id"]

        grid_size = dispatch_info["grid_size"]
        row["Grid_Size"] = grid_size["x"] * grid_size["y"] * grid_size["z"]

        wg = dispatch_info["workgroup_size"]
        row["Workgroup_Size"] = wg["x"] * wg["y"] * wg["z"]

        row["LDS_Per_Workgroup"] = d["lds_block_size_v"]
        row["Scratch_Per_Workitem"] = kernel_symbols[kernel_id]["private_segment_size"]
        row["Arch_VGPR"] = d["arch_vgpr_count"]
        row["Accum_VGPR"] = 0  # TODO: Accum VGPR is missing from rocprofv3 output.
        row["SGPR"] = d["sgpr_count"]
        row["Wave_Size"] = agents[agent_id]["wave_front_size"]
        row["Kernel_Name"] = kernel_symbols[kernel_id]["formatted_kernel_name"]

        id = d["dispatch_data"]["correlation_id"]["internal"]
        rec = dispatch_records[id]

        row["Start_Timestamp"] = rec["start_timestamp"]
        row["End_Timestamp"] = rec["end_timestamp"]
        row["Correlation_ID"] = d["dispatch_data"]["correlation_id"]["external"]

        # Get counters, summing repeated names.
        ctrs: dict[str, Any] = {}

        for r in d["records"]:
            ctr_id = r["counter_id"]["handle"]
            value = r["value"]
            name = counter_info[(agent_id, ctr_id)]["name"]
            if name.endswith("_ACCUM"):
                # Omniperf expects accumulated value in SQ_ACCUM_PREV_HIRES.
                name = "SQ_ACCUM_PREV_HIRES"
            ctrs[name] = ctrs.get(name, 0) + value

        # Append counter values
        for ctr, value in ctrs.items():
            row[ctr] = value

        # Add row to CSV data
        for col_name, value in row.items():
            if col_name not in csv_data:
                csv_data[col_name] = []
            csv_data[col_name].append(value)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file_path, index=False)


def v3_counter_csv_to_v2_csv(
    counter_file: str, agent_info_filepath: str, converted_csv_file: str
) -> None:
    """
    Convert the counter file of csv output for a certain csv from rocprofv3 format
    to rocprfv2 format.
    This function is not for use of other csv out file such as kernel trace file.
    """
    pd_counter_collections = pd.read_csv(counter_file)
    pd_agent_info = pd.read_csv(agent_info_filepath)

    # For backwards compatability. Older rocprof versions do not provide this.
    if not "Accum_VGPR_Count" in pd_counter_collections.columns:
        pd_counter_collections["Accum_VGPR_Count"] = 0

    result = pd_counter_collections.pivot_table(
        index=[
            "Correlation_Id",
            "Dispatch_Id",
            "Agent_Id",
            "Queue_Id",
            "Process_Id",
            "Thread_Id",
            "Grid_Size",
            "Kernel_Id",
            "Kernel_Name",
            "Workgroup_Size",
            "LDS_Block_Size",
            "Scratch_Size",
            "VGPR_Count",
            "Accum_VGPR_Count",
            "SGPR_Count",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        columns="Counter_Name",
        values="Counter_Value",
    ).reset_index()

    # NB: Agent_Id is int in older rocporfv3, now switched to string with prefix
    # "Agent ". We need to make sure handle both cases.
    console_debug(
        f"The type of Agent ID from counter csv file is {result['Agent_Id'].dtype}"
    )

    if result["Agent_Id"].dtype == "object":
        # Apply the function to the 'Agent_Id' column and store it as int64
        try:
            result["Agent_Id"] = (
                result["Agent_Id"]
                .apply(lambda x: int(re.search(r"Agent (\d+)", x).group(1)))
                .astype("int64")
            )
        except Exception as e:
            console_error(
                "v3_counter_csv_to_v2_csv",
                f'Error getting "Agent_Id": {e}',
            )

    # Grab the Wave_Front_Size column from agent info
    result = result.merge(
        pd_agent_info[["Node_Id", "Wave_Front_Size"]],
        left_on="Agent_Id",
        right_on="Node_Id",
        how="left",
    )

    # Create GPU ID mapping from agent info
    gpu_agents = pd_agent_info[pd_agent_info["Agent_Type"] == "GPU"].copy()
    gpu_agents = gpu_agents.reset_index(drop=True)
    gpu_id_map = dict(zip(gpu_agents["Node_Id"], gpu_agents.index))

    # Map Agent_Id to GPU_ID using vectorized operation
    result["Agent_Id"] = result["Agent_Id"].map(gpu_id_map)

    # Drop the temporary Node_Id column
    result = result.drop(columns="Node_Id")

    name_mapping = {
        "Dispatch_Id": "Dispatch_ID",
        "Agent_Id": "GPU_ID",
        "Queue_Id": "Queue_ID",
        "Process_Id": "PID",
        "Thread_Id": "TID",
        "Grid_Size": "Grid_Size",
        "Workgroup_Size": "Workgroup_Size",
        "LDS_Block_Size": "LDS_Per_Workgroup",
        "Scratch_Size": "Scratch_Per_Workitem",
        "VGPR_Count": "Arch_VGPR",
        "Accum_VGPR_Count": "Accum_VGPR",
        "SGPR_Count": "SGPR",
        "Wave_Front_Size": "Wave_Size",
        "Kernel_Name": "Kernel_Name",
        "Start_Timestamp": "Start_Timestamp",
        "End_Timestamp": "End_Timestamp",
        "Correlation_Id": "Correlation_ID",
        "Kernel_Id": "Kernel_ID",
    }
    result.rename(columns=name_mapping, inplace=True)

    index = [
        "Dispatch_ID",
        "GPU_ID",
        "Queue_ID",
        "PID",
        "TID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Wave_Size",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Correlation_ID",
        "Kernel_ID",
    ]

    remaining_column_names = [col for col in result.columns if col not in index]
    index = index + remaining_column_names
    result = result.reindex(columns=index)

    # Rename accumulate counters to standard format
    accum_columns = {
        col: "SQ_ACCUM_PREV_HIRES" for col in result.columns if col.endswith("_ACCUM")
    }
    if accum_columns:
        result = result.rename(columns=accum_columns)

    result.to_csv(converted_csv_file, index=False)


def parse_text(text_file: str) -> list[str]:
    """
    Parse the text file to get the pmc counters.
    """

    def process_line(line: str) -> list[str]:
        if "pmc:" not in line:
            return []
        line = line.strip()
        pos = line.find("#")
        if pos >= 0:
            line = line[0:pos]

        def _dedup(_line: str, _sep: list[str]) -> str:
            for itr in _sep:
                _line = " ".join(_line.split(itr))
            return _line.strip()

        # remove tabs and duplicate spaces
        return _dedup(line.replace("pmc:", ""), ["\n", "\t", " "]).split(" ")

    with open(text_file) as file:
        return [
            counter
            for litr in [process_line(itr) for itr in file.readlines()]
            for counter in litr
        ]


def run_prof(
    fnames: Union[list[str], str],
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
    workload_dir: str,
    mspec: Any,  # noqa: ANN401
    loglevel: int,
    format_rocprof_output: str,
    torch_trace_enabled: bool = False,
    retain_rocpd_output: bool = False,
) -> None:
    multiple_files = isinstance(fnames, list)
    if multiple_files and (
        (
            isinstance(profiler_options, dict)
            and profiler_options.get("ROCPROF_ITERATION_MULTIPLEXING") is None
        )
        or (
            isinstance(profiler_options, list)
            and "--iteration-multiplexing" not in profiler_options
        )
    ):
        console_error(
            "Multiple pmc files detected but ROCPROF_ITERATION_MULTIPLEXING is not set."
        )
        return

    fpath = Path(fnames[0]) if multiple_files else Path(fnames)
    fbase = fpath.stem
    if multiple_files:
        console_debug(f"pmc files: {', '.join([Path(fname).name for fname in fnames])}")
    else:
        console_debug(f"pmc file: {fpath.name}")

    is_mode_live_attach = (
        isinstance(profiler_options, list) and "--pid" in profiler_options
    ) or (
        isinstance(profiler_options, dict)
        and profiler_options.get("ROCPROF_ATTACH_PID") is not None
    )

    # standard rocprof options
    if rocprof_cmd == "rocprofiler-sdk":
        options = cast(dict[str, Union[str, list[str]]], profiler_options).copy()
        if multiple_files:
            options["ROCPROF_COUNTERS"] = ", ".join([
                f"pmc: {' '.join(parse_text(fname))}" for fname in fnames
            ])
        else:
            options["ROCPROF_COUNTERS"] = f"pmc: {' '.join(parse_text(fnames))}"
        options["ROCPROF_AGENT_INDEX"] = "absolute"
    else:
        if multiple_files:
            console_error(
                "Multiple pmc files detected but rocprofv3 does not "
                "support multiple input files."
            )
            return
        default_options = ["-i", fnames]
        options = default_options + cast(list[str], profiler_options)
        options = ["-A", "absolute"] + options

    new_env = os.environ.copy()

    # Counter definitions
    with open(
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "counter_defs.yaml",
    ) as file:
        counter_defs = yaml.safe_load(file)
    # Extra counter definitions
    for fname in fnames if multiple_files else [fnames]:
        if Path(fname).with_suffix(".yaml").exists():
            with open(Path(fname).with_suffix(".yaml")) as file:
                counter_defs["rocprofiler-sdk"]["counters"].extend(
                    yaml.safe_load(file)["rocprofiler-sdk"]["counters"]
                )
    # TODO: Write counter definitions to a user specified path
    # Write counter definitions to a temporary file
    tmpfile_path = (
        Path(tempfile.mkdtemp(prefix="rocprof_counter_defs_", dir="/tmp"))
        / "counter_defs.yaml"
    )
    with open(tmpfile_path, "w") as tmpfile:
        yaml.dump(counter_defs, tmpfile, default_flow_style=False, sort_keys=False)
    # Set counter definitions
    new_env["ROCPROFILER_METRICS_PATH"] = str(tmpfile_path.parent)
    console_debug(
        "Adding env var for counter definitions: "
        f"ROCPROFILER_METRICS_PATH={new_env['ROCPROFILER_METRICS_PATH']}"
    )

    time_1 = time.time()

    output_path = Path(workload_dir + "/out/pmc_1")
    output_path.mkdir(parents=True, exist_ok=True)

    if rocprof_cmd == "rocprofiler-sdk":
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        for key, value in options.items():
            new_env[key] = value
        console_debug(f"rocprof sdk env vars: {new_env}")

        if is_mode_live_attach:
            perform_attach_detach(new_env, options)
        else:
            if app_cmd is None:
                console_error(
                    "APP_CMD, the workload's execuatble must be provided "
                    "when not in live attach mode"
                )

            console_debug(f"rocprof sdk user provided command: {app_cmd}")
            success, output = capture_subprocess_output(
                app_cmd, new_env=new_env, profileMode=True
            )
    else:
        # print in readable format using shlex
        console_debug(f"rocprof command: {shlex.join([rocprof_cmd] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=new_env, profileMode=True
        )

    time_2 = time.time()
    console_debug(
        f"Finishing subprocess of fname {fname}, the time taken is "
        f"{int((time_2 - time_1) / 60)} m {str((time_2 - time_1) % 60)} sec "
    )

    # Delete counter definition temporary directory
    if new_env.get("ROCPROFILER_METRICS_PATH"):
        shutil.rmtree(new_env["ROCPROFILER_METRICS_PATH"], ignore_errors=True)

    if (not is_mode_live_attach) and (not success):
        if loglevel > logging.INFO:
            for line in output.splitlines():
                console_error(line, exit=False)
        console_error("Profiling execution failed.")

    results_files: list[str] = []

    if format_rocprof_output == "rocpd":
        # If using native tool for counter collection
        if (
            rocprof_cmd == "rocprofiler-sdk"
            and options["ROCPROF_COUNTER_COLLECTION"] == "0"
        ):
            for db_name in glob.glob(workload_dir + "/out/pmc_1/*/*.db"):
                pid = Path(db_name).stem.split("_")[0]
                rocpd_data.update_rocpd_pmc_events(
                    pd.read_csv(
                        f"{workload_dir}/out/pmc_1/{pid}_native_counter_collection.csv"
                    ),
                    db_name,
                )
                console_debug(f"Updated rocpd db {db_name} with native tool counters.")
        # Write results_fbase.csv
        rocpd_data.convert_dbs_to_csv(
            glob.glob(workload_dir + "/out/pmc_1/*/*.db"),
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv",
            workload_dir + f"/out/pmc_1/{fbase}_marker_api_trace.csv",
        )
        combined_df = pd.read_csv(
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv"
        )
        # Reset Dispatch_ID based on PID, Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup, Start_Timestamp, End_Timestamp
        combined_df["Dispatch_ID"] = combined_df.groupby(
            [
                "PID",
                "Kernel_Name",
                "Grid_Size",
                "Workgroup_Size",
                "LDS_Per_Workgroup",
                "Start_Timestamp",
                "End_Timestamp",
            ],
            sort=False,
        ).ngroup()
        # Reset Kernel_ID based on Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup
        combined_df["Kernel_ID"] = combined_df.groupby(
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            sort=False,
        ).ngroup()
        # Drop PID since its not required
        combined_df = combined_df.drop(columns=["PID"])
        combined_df.to_csv(
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv", index=False
        )
        combined_df.to_csv(workload_dir + f"/results_{fbase}.csv", index=False)
        if torch_trace_enabled:
            # move counter collection and marker trace to workload dir
            save_torch_trace_inputs(workload_dir, fbase, format_rocprof_output)
        if retain_rocpd_output:
            for db_path in glob.glob(workload_dir + "/out/pmc_1/*/*.db"):
                pid = Path(db_path).stem.split("_")[0]
                shutil.copyfile(
                    db_path,
                    workload_dir + f"/{fbase}_{pid}.db",
                )
                console_warning(
                    f"Retaining large raw rocpd database: "
                    f"{workload_dir}/{fbase}_{pid}.db"
                )
        # Remove temp directory
        shutil.rmtree(workload_dir + "/" + "out")
        return
    elif format_rocprof_output == "csv":
        if rocprof_cmd == "rocprofiler-sdk":
            # rocprofv3 requires additional processing for each process
            results_files = process_rocprofv3_output(
                workload_dir,
                # counter data collected using native tool
                using_native_tool=options["ROCPROF_COUNTER_COLLECTION"] == "0",
            )
            # TODO: as rocprofv3 --kokkos-trace feature improves,
            # rocprof-compute should make updates accordingly
            if "ROCPROF_HIP_RUNTIME_API_TRACE" in options:
                process_hip_trace_output(workload_dir, fbase)
        else:
            # rocprofv3 requires additional processing for each process
            # rocprofv3 cannot use native tool
            results_files = process_rocprofv3_output(
                workload_dir, using_native_tool=False
            )
            if "--kokkos-trace" in options:
                # TODO: as rocprofv3 --kokkos-trace feature improves,
                # rocprof-compute should make updates accordingly
                process_kokkos_trace_output(workload_dir, fbase)
            elif "--hip-trace" in options:
                process_hip_trace_output(workload_dir, fbase)
        # Add torch operator trace processing
        if torch_trace_enabled:
            # move counter collection and marker trace to workload dir
            save_torch_trace_inputs(workload_dir, fbase, format_rocprof_output)
        # Combine results into single CSV file
        if results_files:
            combined_results = pd.concat(
                [pd.read_csv(f) for f in results_files], ignore_index=True
            )
        else:
            console_warning(
                f"Cannot write results for {fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        # Reset Kernel_ID based on Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup
        combined_results["Kernel_ID"] = combined_results.groupby(
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            sort=False,
        ).ngroup()

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )

        if Path(f"{workload_dir}/out").exists():
            # copy and remove out directory if needed
            shutil.copyfile(
                f"{workload_dir}/out/pmc_1/results_{fbase}.csv",
                f"{workload_dir}/{fbase}.csv",
            )
            # Remove temp directory
            shutil.rmtree(f"{workload_dir}/out")

        # Standardize rocprof headers via overwrite
        # {<key to remove>: <key to replace>}
        output_headers = {
            # ROCm-6.1.0 specific csv headers
            "KernelName": "Kernel_Name",
            "Index": "Dispatch_ID",
            "grd": "Grid_Size",
            "gpu-id": "GPU_ID",
            "wgr": "Workgroup_Size",
            "lds": "LDS_Per_Workgroup",
            "scr": "Scratch_Per_Workitem",
            "sgpr": "SGPR",
            "arch_vgpr": "Arch_VGPR",
            "accum_vgpr": "Accum_VGPR",
            "BeginNs": "Start_Timestamp",
            "EndNs": "End_Timestamp",
            # ROCm-6.0.0 specific csv headers
            "GRD": "Grid_Size",
            "WGR": "Workgroup_Size",
            "LDS": "LDS_Per_Workgroup",
            "SCR": "Scratch_Per_Workitem",
            "ACCUM_VGPR": "Accum_VGPR",
        }
        csv_path = Path(workload_dir) / f"{fbase}.csv"
        df = pd.read_csv(csv_path)
        df.rename(columns=output_headers, inplace=True)
        df.to_csv(csv_path, index=False)
    else:
        console_error(f"Unknown format_rocprof_output: {format_rocprof_output}")


def pc_sampling_prof(
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
    method: str,
    interval: int,
    workload_dir: str,
) -> None:
    """
    Run rocprof with pc sampling. Current support v3 only.
    """
    # Todo:
    #   - precheck with rocprofv3 –-list-avail

    unit = "time" if method == "host_trap" else "cycles"

    if rocprof_cmd == "rocprofiler-sdk":
        options = cast(dict[str, Union[str, list[str]]], profiler_options).copy()
        options.update({
            # no counter collection for pc sampling
            "ROCPROF_COUNTER_COLLECTION": "0",
            "ROCPROF_KERNEL_TRACE": "1",
            "ROCPROF_OUTPUT_FORMAT": "csv,json",
            "ROCPROF_OUTPUT_PATH": workload_dir,
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
        console_debug(f"pc sampling rocprof sdk env vars: {new_env}")
        console_debug(f"pc sampling rocprof sdk user provided command: {app_cmd}")
        success, output = capture_subprocess_output(
            app_cmd, new_env=new_env, profileMode=True
        )
    else:
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
            workload_dir,
            "-o",
            "ps_file",  # TODO: sync up with the name from source in 2100_.yaml
            "--",
            cast(str, profiler_options[-1]),  # app command
        ]

        console_debug(f"rocprof command: {shlex.join([rocprof_cmd] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=os.environ.copy(), profileMode=True
        )

    if not success:
        console_error("PC sampling failed.")


def convert_native_counter_collection_csv(workload_dir: str) -> None:
    """
    Use native counter collection csv and rocprofiler-sdk kernel
    trace to write counter collection csv in rocprofiler-sdk format
    for further processing to pmc_perf.csv file
    """
    for native_filename in glob.glob(
        f"{workload_dir}/out/pmc_1/*_native_counter_collection.csv"
    ):
        counter_data = pd.read_csv(native_filename, index_col=False)
        # Group by on dispatch_id and counter_id and sum the counter_value,
        # Other rows in group have the same value, so take the first one
        groupby_cols = ["dispatch_id", "counter_name"]
        agg_dict = {
            col: "first" for col in counter_data.columns if col not in groupby_cols
        }
        # Overwrite counter_value aggregation to sum
        agg_dict["counter_value"] = "sum"
        counter_data = counter_data.groupby(groupby_cols, as_index=False).agg(agg_dict)

        pid = Path(native_filename).stem.split("_")[0]
        kernel_data_filename = glob.glob(
            f"{workload_dir}/out/pmc_1/*/{pid}_kernel_trace.csv"
        )[0]
        kernel_data = pd.read_csv(kernel_data_filename)

        # Merge counter_data with kernel_data on dispatch_id
        merged_data = pd.merge(
            counter_data,
            kernel_data,
            left_on="dispatch_id",
            right_on="Dispatch_Id",
            how="inner",
        )

        rocprofv3_counter_data = pd.DataFrame({
            "Correlation_Id": merged_data["Correlation_Id"],
            "Dispatch_Id": merged_data["dispatch_id"],
            "Agent_Id": merged_data["Agent_Id"],
            "Queue_Id": merged_data["Queue_Id"],
            "Process_Id": merged_data["Thread_Id"],
            "Thread_Id": merged_data["Thread_Id"],
            "Grid_Size": (
                merged_data[["Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"]].prod(axis=1)
            ),
            "Kernel_Id": merged_data["Kernel_Id"],
            "Kernel_Name": merged_data["Kernel_Name"],
            "Workgroup_Size": (
                merged_data[
                    ["Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z"]
                ].prod(axis=1)
            ),
            "LDS_Block_Size": merged_data["LDS_Block_Size"],
            "Scratch_Size": merged_data["Scratch_Size"],
            "VGPR_Count": merged_data["VGPR_Count"],
            "Accum_VGPR_Count": merged_data["Accum_VGPR_Count"],
            "SGPR_Count": merged_data["SGPR_Count"],
            "Counter_Name": merged_data["counter_name"],
            "Counter_Value": merged_data["counter_value"],
            "Start_Timestamp": merged_data["Start_Timestamp"],
            "End_Timestamp": merged_data["End_Timestamp"],
        })
        rocprofv3_counter_data.to_csv(
            kernel_data_filename.replace("kernel_trace", "counter_collection"),
            index=False,
        )


def process_rocprofv3_output(workload_dir: str, using_native_tool: bool) -> list[str]:
    """
    rocprofv3 specific output processing for csv format.
    """
    results_files_csv: list[str] = []

    if using_native_tool:
        try:
            convert_native_counter_collection_csv(workload_dir)
        except Exception:
            console_error(
                "Error converting native counter collection csv.\n"
                f"Stacktrace:\n{traceback.format_exc()}"
            )

    counter_info_csvs = glob.glob(
        f"{workload_dir}/out/pmc_1/*/*_counter_collection.csv"
    )
    existing_counter_files_csv = [f for f in counter_info_csvs if Path(f).is_file()]

    if existing_counter_files_csv:
        for counter_file in existing_counter_files_csv:
            counter_path = Path(counter_file)
            current_dir = counter_path.parent

            agent_info_filepath = current_dir / counter_path.name.replace(
                "_counter_collection", "_agent_info"
            )

            if not agent_info_filepath.is_file():
                raise ValueError(
                    f'{counter_file} has no corresponding "agent info" file'
                )

            converted_csv_file = current_dir / counter_path.name.replace(
                "_counter_collection", "_converted"
            )

            try:
                v3_counter_csv_to_v2_csv(
                    counter_file, str(agent_info_filepath), str(converted_csv_file)
                )
            except Exception as e:
                console_warning(
                    f"Error converting {counter_file} from v3 to v2 csv: {e}"
                )
                return []

        results_files_csv = glob.glob(f"{workload_dir}/out/pmc_1/*/*_converted.csv")
    else:
        return []

    return results_files_csv


@demarcate
def save_torch_trace_inputs(
    workload_dir: str,
    fbase: str,
    output_format: str = "rocpd",
) -> None:
    """
    Move counter_collection and marker_api_trace data to workload_dir,
    for creation of PyTorch operator trace in Analyze mode.
    """
    src_dir = Path(workload_dir) / "out" / "pmc_1"
    if output_format == "rocpd":
        # Only one pair expected
        src_counter = src_dir / f"{fbase}_counter_collection.csv"
        src_marker = src_dir / f"{fbase}_marker_api_trace.csv"
        dst_counter = Path(workload_dir) / f"torch_trace_{fbase}_counter_collection.csv"
        dst_marker = Path(workload_dir) / f"torch_trace_{fbase}_marker_api_trace.csv"
        # These files are expected to exist
        # Letting shutil.copyfile raise error if files not found
        shutil.copyfile(src_counter, dst_counter)
        shutil.copyfile(src_marker, dst_marker)
        console_log(
            "torch trace",
            "Moved counter collection and marker trace files"
            "to workload dir for PyTorch trace creation.",
        )
        console_log("Counter Collection: ", str(dst_counter))
        console_log("Marker API Trace: ", str(dst_marker))
    elif output_format == "csv":
        # Multiple pairs possible (one per PID/process)
        counter_files = glob.glob(str(src_dir / "*/*_counter_collection.csv"))
        marker_files = glob.glob(str(src_dir / "*/*_marker_api_trace.csv"))
        (Path(workload_dir) / f"{fbase}").mkdir(parents=True, exist_ok=True)
        # Expecting the files to be present
        # Letting shutil.copyfile raise error if files not found
        # Path: workload_dir/fbase/torch_trace_<src_basename> (discovered by
        # process_torch_trace_output via glob **/torch_trace*_marker_api_trace.csv)
        for src_counter in counter_files:
            dst_counter = str(
                Path(workload_dir)
                / f"{fbase}"
                / ("torch_trace_" + Path(src_counter).name)
            )
            shutil.copyfile(src_counter, dst_counter)
            console_log("torch trace", f"Copied Counter Collection: {dst_counter}")
        for src_marker in marker_files:
            dst_marker = str(
                Path(workload_dir)
                / f"{fbase}"
                / ("torch_trace_" + Path(src_marker).name)
            )
            shutil.copyfile(src_marker, dst_marker)
            console_log("torch trace", f"Copied Marker API Trace: {dst_marker}")
    else:
        console_warning(
            "torch trace",
            f"Unknown output_format: {output_format} in save_torch_trace_inputs",
        )


@demarcate
def process_torch_trace_output(
    workload_dir: str,
) -> None:
    """
    Joins counter_collection and marker_api_trace data for PyTorch operator listing.

    - Performs inner join on Correlation_ID, filtering out unmatched entries
    - Consolidates data across passes and groups by Operator_Name, saving one CSV
      per operator under workload_dir/torch_trace/
    - Removes the source marker_api_trace and counter_collection files after
      consolidation.
    """
    # Find all marker_api_trace CSV files
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    marker_api_trace_csvs = list(
        Path(workload_dir).glob("**/torch_trace*_marker_api_trace.csv")
    )
    counter_collection_csvs = [
        markers_file.parent
        / markers_file.name.replace("_marker_api_trace.", "_counter_collection.")
        for markers_file in marker_api_trace_csvs
    ]
    existing_csv_files = [
        [marker_api_trace_csvs[i], counter_collection_csvs[i]]
        for i in range(len(marker_api_trace_csvs))
        if counter_collection_csvs[i].is_file() and marker_api_trace_csvs[i].is_file()
    ]

    if not existing_csv_files:
        if Path(f"{workload_dir}/torch_trace").exists():
            console_log(
                "torch trace",
                "Torch data has already been processed and saved to "
                f"{workload_dir}/torch_trace",
            )
        else:
            console_warning(
                "torch trace",
                "No marker files with corresponding counter files found."
                "Ensure profiling was done with '--torch-trace'.",
            )
        return
    # Remove previous torch_trace output dir so we can regenerate; source
    # marker/counter files are removed after consolidation below.
    if Path(f"{workload_dir}/torch_trace").exists():
        shutil.rmtree(Path(f"{workload_dir}/torch_trace"))
        console_log(
            f"Removed previous torch_trace directory: {workload_dir}/torch_trace"
        )

    # Join marker and counter data
    def _merge_pair(
        marker_path: Path,
        counter_path: Path,
        join_keys: list = ("Correlation_ID"),
    ) -> pd.DataFrame:
        """Merge a pair of marker and counter csv files on specified keys,
        return the merged dataframe.
        """
        marker_df = pd.read_csv(marker_path)
        counter_df = pd.read_csv(counter_path)
        # Normalize column names to handle case inconsistencies
        marker_df.columns = marker_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )
        counter_df.columns = counter_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )

        return pd.merge(
            marker_df,
            counter_df,
            on=join_keys,
            how="inner",
            suffixes=("_function", "_kernel"),
        )

    # If rocpd format, pairs are present in workload_dir, one pair per fbase
    # If csv format, pairs are present in workload/{fbase}/ one pair per process
    # Extracting the output_format used in profiling from the path of a marker file
    if Path(workload_dir).resolve() == existing_csv_files[0][0].parent.resolve():
        join_keys = ("Correlation_ID", "GUID")  # output_format "rocpd"
    else:
        join_keys = ("Correlation_ID",)  # output_format "csv"
    consolidated_df = pd.concat(
        [_merge_pair(f[0], f[1], join_keys) for f in existing_csv_files],
        ignore_index=True,
    )
    required_columns = [
        "Function",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp_function",
        "End_Timestamp_function",
        "Start_Timestamp_kernel",
        "End_Timestamp_kernel",
    ]
    missing_columns = [
        col for col in required_columns if col not in consolidated_df.columns
    ]
    if missing_columns:
        console_error(
            f"Consolidated torch trace is missing required columns {missing_columns}"
        )
        return
    consolidated_df = consolidated_df[required_columns]
    if consolidated_df.isnull().values.any():
        console_warning("Consolidated torch trace contains missing values")
        return
    consolidated_df = consolidated_df.sort_values(by=["Function", "Counter_Name"])
    split_columns = consolidated_df["Function"].str.split(":#", expand=True)
    consolidated_df["Operator_Name"] = (
        split_columns[0] if len(split_columns.columns) > 0 else None
    )
    consolidated_df["Context_Id"] = (
        split_columns[1] if len(split_columns.columns) > 1 else None
    )
    consolidated_df.drop(columns=["Function"], inplace=True)
    consolidated_df = consolidated_df[
        [
            "Operator_Name",
            "Context_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp_function",
            "End_Timestamp_function",
            "Start_Timestamp_kernel",
            "End_Timestamp_kernel",
        ]
    ]
    if consolidated_df.isnull().values.any():
        console_error(
            "Missing values in consolidated torch trace after splitting ",
            "the Function name.",
        )
        return
    grouped = consolidated_df.groupby("Operator_Name")
    for operator_name, group in grouped:
        # Extract the operator name from hierarchy
        last_operator = operator_name.split("/")[-1]
        sanitized_operator_name = last_operator.replace("torch.", "").replace(".", "_")
        # Ensure output directory exists
        Path(f"{workload_dir}/torch_trace").mkdir(parents=True, exist_ok=True)
        output_file = f"{workload_dir}/torch_trace/{sanitized_operator_name}.csv"
        # If the file already exists, append to it, else create new file.
        if Path(output_file).is_file():
            group.to_csv(output_file, mode="a", header=False, index=False)
            console_log(f"Appended trace to existing file {output_file}")
        else:
            group.to_csv(output_file, index=False)
            console_log(f"Saved consolidated trace to {output_file}")
    for trace_file in marker_api_trace_csvs + counter_collection_csvs:
        try:
            Path(trace_file).unlink()
            console_debug(f"Removed temporary torch trace file: {trace_file}")
        except OSError as e:
            console_warning(f"Error removing temporary file {trace_file}: {e}")


@demarcate
def process_kokkos_trace_output(workload_dir: str, fbase: str) -> None:
    # marker api trace csv files are generated for each process
    marker_api_trace_csvs = glob.glob(
        f"{workload_dir}/out/pmc_1/*/*_marker_api_trace.csv"
    )
    existing_marker_files_csv = [f for f in marker_api_trace_csvs if Path(f).is_file()]

    # concate and output marker api trace info
    combined_results = pd.concat(
        [pd.read_csv(f) for f in existing_marker_files_csv], ignore_index=True
    )

    combined_results.to_csv(
        f"{workload_dir}/out/pmc_1/results_{fbase}_marker_api_trace.csv",
        index=False,
    )

    if Path(f"{workload_dir}/out").exists():
        shutil.copyfile(
            f"{workload_dir}/out/pmc_1/results_{fbase}_marker_api_trace.csv",
            f"{workload_dir}/{fbase}_marker_api_trace.csv",
        )


@demarcate
def process_hip_trace_output(workload_dir: str, fbase: str) -> None:
    # hip api trace csv files are generated for each process
    hip_api_trace_csvs = glob.glob(f"{workload_dir}/out/pmc_1/*/*_hip_api_trace.csv")
    existing_hip_files_csv = [f for f in hip_api_trace_csvs if Path(f).is_file()]

    # concate and output hip api trace info
    combined_results = pd.concat(
        [pd.read_csv(f) for f in existing_hip_files_csv], ignore_index=True
    )

    combined_results.to_csv(
        f"{workload_dir}/out/pmc_1/results_{fbase}_hip_api_trace.csv",
        index=False,
    )

    if Path(f"{workload_dir}/out").exists():
        shutil.copyfile(
            f"{workload_dir}/out/pmc_1/results_{fbase}_hip_api_trace.csv",
            f"{workload_dir}/{fbase}_hip_api_trace.csv",
        )


@demarcate
def gen_sysinfo(
    workload_name: str,
    workload_dir: str,
    app_cmd: str,
    skip_roof: bool,
    mspec: Any,  # noqa: ANN401
    soc: Any,  # noqa: ANN401
) -> None:
    df = mspec.get_class_members()

    # Append workload information to machine specs
    df["command"] = app_cmd
    df["workload_name"] = workload_name

    blocks = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
    if hasattr(soc, "roofline_obj") and (not skip_roof):
        blocks.append("roofline")
    df["ip_blocks"] = "|".join(blocks)

    df.to_csv(workload_dir + "/" + "sysinfo.csv", index=False)


def get_submodules(package_name: str) -> list[str]:
    """List all submodules for a target package"""
    import importlib
    import pkgutil

    submodules: list[str] = []

    # walk all submodules in target package
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__):
        pretty_name = name.split("_", 1)[1].replace("_", "")
        # ignore base submodule, add all other
        if pretty_name != "base":
            submodules.append(pretty_name)

    return submodules


def is_workload_empty(path: str) -> None:
    """Peek workload directory to verify valid profiling output"""
    pmc_perf_path = Path(path) / "pmc_perf.csv"
    if pmc_perf_path.is_file():
        temp_df = pd.read_csv(pmc_perf_path)
        if temp_df.dropna().empty:
            console_error(
                "profiling",
                f"Found empty cells in {pmc_perf_path}.\n"
                "Profiling data could be corrupt.",
            )
    else:
        console_error("analysis", "No profiling data found.")


def print_status(msg: str) -> None:
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def set_locale_encoding() -> None:
    try:
        # Attempt to set the locale to 'C.UTF-8'
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except locale.Error:
        # If 'C.UTF-8' is not available, check if the current locale is UTF-8 based
        current_locale = locale.getdefaultlocale()
        if current_locale and current_locale[1] and "UTF-8" in current_locale[1]:
            try:
                locale.setlocale(locale.LC_ALL, current_locale[0])
            except locale.Error as e:
                console_error(
                    f"Failed to set locale to the current UTF-8-based locale: {e}"
                )
        else:
            console_error(
                "Please ensure that a UTF-8-based locale is available on your system.",
                exit=False,
            )


def reverse_multi_index_df_pmc(
    final_df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[Any]]:
    """
    Util function to decompose multi-index dataframe.
    """
    # Check if the columns have more than one level
    if not isinstance(final_df.columns, pd.MultiIndex) or final_df.columns.nlevels < 2:
        raise ValueError("Input DataFrame does not have a multi-index column.")

    # Extract the first level of the MultiIndex columns (the file names)
    coll_levels = final_df.columns.get_level_values(0).unique().tolist()

    # Initialize the list of DataFrames
    dfs: list[pd.DataFrame] = []

    # Loop through each 'coll_level' and rebuild the DataFrames
    for level in coll_levels:
        # Select columns that belong to the current 'coll_level'
        columns_for_level = final_df.xs(level, axis=1, level=0)
        # Append the DataFrame for this level
        if isinstance(columns_for_level, pd.Series):
            columns_for_level = columns_for_level.to_frame()
        dfs.append(columns_for_level)

    # Return the list of DataFrames and the column levels
    return dfs, coll_levels


def impute_counters_iteration_multiplex(
    df_multi_index: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    """
    Perform data imputation for missing counter values due to iteration multiplexing.
    """
    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Kernel_ID",
    ]
    result_dfs: list[pd.DataFrame] = []
    dfs, coll_levels = reverse_multi_index_df_pmc(df_multi_index)

    for df in dfs:
        # Group by unique kernel configurations
        unique_occurences = (
            df.groupby("Kernel_Name")
            if policy == "kernel"
            else df.groupby(
                [
                    "Kernel_Name",
                    "Grid_Size",
                    "Workgroup_Size",
                    "LDS_Per_Workgroup",
                ],
                as_index=False,
            )
        )

        counter_columns = [
            col for col in df.columns if col not in non_counter_column_index
        ]
        # Collect imputed groups as dataframes
        group_dfs = []

        for _, group in unique_occurences:
            # Identify counter buckets
            counter_groups: set[frozenset[str]] = set()
            for _, row in group.iterrows():
                # Set of counter column names with non empty values
                cols_frozenset = frozenset(
                    row[counter_columns][row[counter_columns].notna()].index
                )
                # If no counters found for this dispatch, continue
                if not cols_frozenset:
                    continue
                # Since counter buckets are repeated in round robin fashion,
                # we can stop once we see a repeated bucket
                if cols_frozenset in counter_groups:
                    break
                counter_groups.add(cols_frozenset)

            # If no counters found for this group, continue
            if not counter_groups:
                continue

            # Iterate over subgroups of dispatches containing
            # all counters and impute missing values
            subgroup_size = len(counter_groups)
            all_counters = {
                counter for counter_group in counter_groups for counter in counter_group
            }
            # Collect imputed sub-groups as dataframes
            subgroup_dfs = []
            previous_fill_values = {}
            for i in range(0, len(group), subgroup_size):
                subgroup = group.iloc[i : i + subgroup_size]

                # Build imputation mapping once for all counters in this subgroup
                fill_values = {}
                for counter in all_counters:
                    valid_mask = subgroup[counter].notna()
                    if valid_mask.any():
                        # Get the first valid value for this counter
                        fill_values[counter] = subgroup.loc[valid_mask, counter].iloc[0]

                # Apply all fills at once using vectorized fillna
                if fill_values:
                    subgroup = subgroup.fillna(fill_values)

                # If this is the last subgroup and it still has missing values,
                # use previous subgroup's fill values
                # NOTE: This wont work if the first subgroup is itself incomplete
                is_last_subgroup = (i + subgroup_size) >= len(group)
                # First any() returns bool pd.Series for every column,
                # second any() returns single bool
                if (
                    is_last_subgroup
                    and previous_fill_values
                    and subgroup.isna().any().any()
                ):
                    # Use previous subgroup's fill values for remaining missing values
                    subgroup = subgroup.fillna(previous_fill_values)

                subgroup_dfs.append(subgroup)
                previous_fill_values = fill_values

            # Concatenate all subgroups for this group
            if subgroup_dfs:
                # Add the imputed group dataframe
                group_dfs.append(pd.concat(subgroup_dfs, ignore_index=True))

        # Create a new dataframe by concatenating all groups
        result_dfs.append(
            pd.concat(group_dfs, ignore_index=True)
            if group_dfs
            else pd.DataFrame(df.columns)
        )

    final_df = pd.concat(result_dfs, keys=coll_levels, axis=1, copy=False)
    return final_df


def merge_counters_spatial_multiplex(df_multi_index: pd.DataFrame) -> pd.DataFrame:
    """
    For spatial multiplexing, this merges counter values for the same kernel that
    runs on different devices. For time stamp, start time stamp will use median
    while for end time stamp, it will be equal to the summation between median
    start stamp and median delta time.
    """
    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Queue_ID",
        "PID",
        "TID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Wave_Size",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Correlation_ID",
        "Kernel_ID",
        "Node",
    ]

    expired_column_index = [
        "Node",
        "PID",
        "TID",
        "Queue_ID",
    ]

    result_dfs: list[pd.DataFrame] = []

    # TODO: will need to optimize to avoid this conversion to single index format
    # and do merge directly on multi-index dataframe
    dfs, coll_levels = reverse_multi_index_df_pmc(df_multi_index)

    for df in dfs:
        kernel_name_column_name = "Kernel_Name"
        if "Kernel_Name" not in df and "Name" in df:
            kernel_name_column_name = "Name"

        # Find the values in Kernel_Name that occur more than once
        kernel_single_occurances = df[kernel_name_column_name].value_counts().index

        # Define a list to store the merged rows
        result_data: list[dict[str, Any]] = []

        for kernel_name in kernel_single_occurances:
            # Get all rows for the current kernel_name
            group = df[df[kernel_name_column_name] == kernel_name]

            # Create a dictionary to store the merged row for the current group
            merged_row: dict[str, Any] = {}

            # Process non-counter columns
            for col in [
                col
                for col in non_counter_column_index
                if col not in expired_column_index
            ]:
                if col == "Start_Timestamp":
                    # For Start_Timestamp, take the median
                    merged_row[col] = group["Start_Timestamp"].median()
                elif col == "End_Timestamp":
                    # For End_Timestamp, calculate the median delta time
                    delta_time = group[col] - group["Start_Timestamp"]
                    merged_row[col] = group["Start_Timestamp"] + delta_time.median()
                else:
                    # For other non-counter columns, take the first occurrence (0th row)
                    merged_row[col] = group.iloc[0][col]

            # Process counter columns (assumed to be all columns not in
            # non_counter_column_index)
            counter_columns = [
                col for col in group.columns if col not in non_counter_column_index
            ]
            for counter_col in counter_columns:
                # for counter columns, take the first non-none (or non-nan) value
                current_valid_counter_group = group[group[counter_col].notna()]
                first_valid_value = (
                    current_valid_counter_group.iloc[0][counter_col]
                    if len(current_valid_counter_group) > 0
                    else None
                )
                merged_row[counter_col] = first_valid_value

            # Append the merged row to the result list
            result_data.append(merged_row)

        # Create a new DataFrame from the merged rows
        result_dfs.append(pd.DataFrame(result_data))

    final_df = pd.concat(result_dfs, keys=coll_levels, axis=1, copy=False)
    return final_df


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
    filename = (
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "profile_configs"
        / "sets"
        / f"{arch}_sets.yaml"
    )
    with open(filename) as file:
        content = file.read()
    data = yaml.safe_load(content)

    sets_data = data.get("sets", [])

    sets_info: dict[str, Any] = {}
    for set_item in sets_data:
        set_option = set_item.get("set_option", "")
        if set_option:
            sets_info[set_option] = set_item
    return sets_info


def get_uuid(length: int = 8) -> str:
    return uuid.uuid4().hex[:length]


def format_scientific_notation_if_needed(
    value: Union[int, float],
    align: str = ">",
    width_align: int = 6,
    precision: int = 2,
    fmt_type_align: str = "f",
    max_length: int = 6,
    sci_lower_bound: float = 1e-2,
    sci_upper_bound: float = 1e6,
) -> str:
    """
    Format a numeric value as normal or scientific notation string.

    Uses scientific notation if:
    - abs(value) < sci_lower_bound (but not zero)
    - abs(value) >= sci_upper_bound
    - formatted normal string length exceeds max_length

    Parameters:
    - value: numeric value to format
    - align: alignment character ('<', '>', '^', '=')
    - width_align: total width of formatted output
    - precision: number of digits after decimal point
    - fmt_type_align: format type, e.g., 'f', 'e', 'g'
    - max_length: max allowed length for normal format string (excluding padding)
    - sci_lower_bound: lower bound for scientific notation usage
    - sci_upper_bound: upper bound for scientific notation usage

    Returns:
    - formatted string according to the criteria, respecting alignment
    """

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
                if (
                    len(normal_str_strip) > len(sci_str_strip)
                    or len(normal_str_strip) > max_length
                ):
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


def load_yaml(filepath: str) -> dict[str, Any]:
    """Load YAML file and return as dictionary."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def get_panel_alias() -> dict[str, str]:
    panel_yaml = load_yaml(
        f"{config.rocprof_compute_home}/rocprof_compute_soc/analysis_configs/gfx9_config_template.yaml"
    )
    return {
        panel["panel_alias"]: str(panel["panel_id"]) for panel in panel_yaml["panels"]
    }


def get_rank() -> Optional[str]:
    rank_env_vars = [
        "SLURM_PROCID",
        "FLUX_TASK_RANK",
        "PMI_RANK",
        "PMIX_RANK",
        "PALS_RANKID",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
        "MPI_RANKID",
        "MPI_LOCALRANKID",
        "MPI_RANK",
    ]
    for env_var in rank_env_vars:
        value = os.environ.get(env_var)
        if value is not None:
            return value

    return None


def replace_rank(name: str) -> str:
    def rank(match: re.Match[str]) -> str:
        value = get_rank()
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
