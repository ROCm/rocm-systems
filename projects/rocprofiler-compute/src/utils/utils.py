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
        if not Path(args.rocprofiler_sdk_library_path).exists():
            console_error(
                "Could not find rocprofiler-sdk library at "
                f"{args.rocprofiler_sdk_library_path}"
            )
        rocprof_cmd = "rocprofiler-sdk"
        console_debug(f"rocprof_cmd is {rocprof_cmd}")
        console_debug(f"rocprofiler_sdk_path is {args.rocprofiler_sdk_library_path}")
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
    fname: str,
    profiler_options: Union[list[str], dict[str, Union[str, list[str]]]],
    workload_dir: str,
    mspec: Any,  # noqa: ANN401
    loglevel: int,
    format_rocprof_output: str,
    retain_rocpd_output: bool = False,
) -> None:
    fpath = Path(fname)
    fbase = fpath.stem
    console_debug(f"pmc file: {fpath.name}")

    is_mode_live_attach = (
        isinstance(profiler_options, list) and "--pid" in profiler_options
    ) or (
        isinstance(profiler_options, dict)
        and profiler_options.get("ROCPROF_ATTACH_PID") is not None
    )

    # standard rocprof options
    if rocprof_cmd == "rocprofiler-sdk":
        options = cast(dict[str, Union[str, list[str]]], profiler_options)
        options["ROCPROF_COUNTER_COLLECTION"] = "1"
        options["ROCPROF_COUNTERS"] = f"pmc: {' '.join(parse_text(fname))}"
    else:
        default_options = ["-i", fname]
        options = default_options + cast(list[str], profiler_options)

    if rocprof_cmd == "rocprofiler-sdk":
        options["ROCPROF_AGENT_INDEX"] = "absolute"
    else:
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
    if Path(fname).with_suffix(".yaml").exists():
        with open(Path(fname).with_suffix(".yaml")) as file:
            counter_defs["rocprofiler-sdk"]["counters"].extend(
                yaml.safe_load(file)["rocprofiler-sdk"]["counters"]
            )
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

    # set required env var for >= mi300
    if mspec.gpu_model.lower() not in (
        "mi50",
        "mi60",
        "mi100",
        "mi210",
        "mi250",
        "mi250x",
    ):
        new_env["ROCPROFILER_INDIVIDUAL_XCC_MODE"] = "1"

    is_timestamps = Path(fname).name == "timestamps.txt"
    time_1 = time.time()

    if rocprof_cmd == "rocprofiler-sdk":
        app_cmd = options.pop("APP_CMD") if "APP_CMD" in options else None
        for key, value in options.items():
            new_env[key] = value
        console_debug(f"rocprof sdk env vars: {new_env}")

        if is_mode_live_attach:

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
                libname = options["ROCPROF_ATTACH_TOOL_LIBRARY"]
                c_lib = ctypes.CDLL(libname)
                if c_lib is None:
                    console_error(f"Error opening {libname}")
                c_lib.attach.argtypes = [ctypes.c_uint]

                pid = options["ROCPROF_ATTACH_PID"]
                if pid is None:
                    console_error(
                        "Mode of attach/detach must have setup for process ID"
                    )

                c_lib.attach(int(pid))
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
                c_lib.detach()

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
        # Write results_fbase.csv
        rocpd_data.convert_db_to_csv(
            glob.glob(workload_dir + "/out/pmc_1/*/*.db")[0],
            workload_dir + f"/results_{fbase}.csv",
        )
        if retain_rocpd_output:
            shutil.copyfile(
                glob.glob(workload_dir + "/out/pmc_1/*/*.db")[0],
                workload_dir + "/" + fbase + ".db",
            )
            console_warning(
                f"Retaining large raw rocpd database: {workload_dir}/{fbase}.db"
            )
        # Remove temp directory
        shutil.rmtree(workload_dir + "/" + "out")
        return

    # rocprofv3 requires additional processing for each process
    results_files = process_rocprofv3_output(
        format_rocprof_output, workload_dir, is_timestamps
    )

    if rocprof_cmd == "rocprofiler-sdk":
        # TODO: as rocprofv3 --kokkos-trace feature improves,
        # rocprof-compute should make updates accordingly
        if "ROCPROF_HIP_RUNTIME_API_TRACE" in options:
            process_hip_trace_output(workload_dir, fbase)
    else:
        if "--kokkos-trace" in options:
            # TODO: as rocprofv3 --kokkos-trace feature improves,
            # rocprof-compute should make updates accordingly
            process_kokkos_trace_output(workload_dir, fbase)
        elif "--hip-trace" in options:
            process_hip_trace_output(workload_dir, fbase)

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


def pc_sampling_prof(
    method: str,
    interval: int,
    workload_dir: str,
    appcmd: list[str],
    rocprofiler_sdk_library_path: str,
) -> None:
    """
    Run rocprof with pc sampling. Current support v3 only.
    """
    # Todo:
    #   - precheck with rocprofv3 –-list-avail

    unit = "time" if method == "host_trap" else "cycles"

    if rocprof_cmd == "rocprofiler-sdk":
        rocm_libdir = str(Path(rocprofiler_sdk_library_path).parent)
        rocprofiler_sdk_tool_path = str(
            Path(rocm_libdir) / "rocprofiler-sdk/librocprofiler-sdk-tool.so"
        )
        ld_preload = [
            rocprofiler_sdk_tool_path,
            rocprofiler_sdk_library_path,
        ]
        options = {
            "ROCPROFILER_LIBRARY_CTOR": "1",
            "LD_PRELOAD": ":".join(ld_preload),
            "ROCP_TOOL_LIBRARIES": rocprofiler_sdk_tool_path,
            "LD_LIBRARY_PATH": rocm_libdir,
            "ROCPROF_OUTPUT_FORMAT": "csv,json",
            "ROCPROF_OUTPUT_PATH": workload_dir,
            "ROCPROF_OUTPUT_FILE_NAME": "ps_file",
            "ROCPROFILER_PC_SAMPLING_BETA_ENABLED": "1",
            "ROCPROF_PC_SAMPLING_UNIT": unit,
            "ROCPROF_PC_SAMPLING_INTERVAL": str(interval),
            "ROCPROF_PC_SAMPLING_METHOD": method,
            "ROCPROF_KERNEL_TRACE": "1",
        }
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        console_debug(f"pc sampling rocprof sdk env vars: {new_env}")
        console_debug(f"pc sampling rocprof sdk user provided command: {appcmd}")
        success, output = capture_subprocess_output(
            appcmd, new_env=new_env, profileMode=True
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
        ]
        options.extend(appcmd)

        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=os.environ.copy(), profileMode=True
        )

    if not success:
        console_error("PC sampling failed.")


def process_rocprofv3_output(
    rocprof_output: str, workload_dir: str, is_timestamps: bool
) -> list[str]:
    """
    rocprofv3 specific output processing.
    takes care of json or csv formats, for csv format,
    additional processing is performed.
    """
    results_files_csv: list[str] = []

    if rocprof_output == "json":
        results_files_json = glob.glob(f"{workload_dir}/out/pmc_1/*/*.json")

        for json_file in results_files_json:
            csv_file = str(Path(json_file).with_suffix(".csv"))
            v3_json_to_csv(json_file, csv_file)
        results_files_csv = glob.glob(f"{workload_dir}/out/pmc_1/*/*.csv")

    elif rocprof_output == "csv":
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
        elif is_timestamps:
            # when the input is timestamps, we know counter csv file
            # is not generated and will instead parse kernel trace file
            results_files_csv = glob.glob(
                f"{workload_dir}/out/pmc_1/*/*_kernel_trace.csv"
            )
        else:
            # when the input is not for timestamps, and counter csv file
            # is not generated, we assume failed rocprof run and will completely
            # bypass the file generation and merging for current pmc
            results_files_csv = []
    else:
        console_error("The output file of rocprofv3 can only support json or csv!!!")

    return results_files_csv


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


def detect_roofline(mspec: Any) -> dict[str, str]:  # noqa: ANN401
    from utils import specs

    rocm_ver = int(mspec.rocm_version[:1])

    target_binary: dict[str, Any] = {
        "rocm_ver": rocm_ver,
        "distro": "override",
        "path": None,
    }

    # Create distro ID list based off of ID (a string, containing a single distro)
    # and ID_LIKE (a string, listing at least one distro, separated by a single space)
    # from the system /etc/os-release file
    os_release = Path("/etc/os-release").read_text()
    id_list = specs.search(r'^ID_LIKE="?(.*?)"?$', os_release) or ""
    id = specs.search(r'^ID="?(.*?)"?$', os_release) or ""
    id_list = id_list.split() + [id]

    if "ROOFLINE_BIN" in os.environ.keys():
        rooflineBinary = os.environ["ROOFLINE_BIN"]
        if Path(rooflineBinary).exists():
            console_warning(
                "roofline",
                f"Detected user-supplied binary --> ROOFLINE_BIN = {rooflineBinary}\n",
            )
            # distro stays marked as override and path value is substituted in
            target_binary["path"] = rooflineBinary
            return target_binary
        else:
            console_error(
                "roofline",
                "user-supplied path to binary not accessible --> "
                f"ROOFLINE_BIN = {rooflineBinary}\n",
            )

    # check that the system OS is based off of one of the following distributions
    elif "azurelinux" in id_list:
        distro = "azurelinux"

    elif "debian" in id_list:
        distro = "22.04"

    elif "fedora" in id_list:
        distro = "platform:el8"

    elif "suse" in id_list:
        distro = "15.6"

    else:
        console_error(
            "roofline", "Cannot find a valid binary for your operating system"
        )

    # distro gets assigned, to follow default roofline bin location and nomenclature
    target_binary["distro"] = distro
    return target_binary


def mibench(args: argparse.Namespace, mspec: Any) -> None:  # noqa: ANN401
    """Run roofline microbenchmark to generate peek BW and FLOP measurements."""
    console_log("roofline", "No roofline data found. Generating...")

    distro_map = {
        "platform:el8": "rhel8",
        "15.6": "sles15sp6",
        "22.04": "ubuntu22_04",
        "azurelinux": "azurelinux3",
    }

    binary_paths: list[str] = []

    target_binary = detect_roofline(mspec)
    if target_binary["distro"] == "override":
        binary_paths.append(target_binary["path"])
    else:
        # check two potential locations for roofline binaries due to differences in
        # development usage vs formal install
        potential_paths = [
            config.rocprof_compute_home / "utils" / "rooflines" / "roofline",
            config.rocprof_compute_home.parent.parent / "bin" / "roofline",
        ]

        for directory in potential_paths:
            path_to_binary = (
                f"{directory}-{distro_map[target_binary['distro']]}"
                f"-rocm{target_binary['rocm_ver']}"
            )
            binary_paths.append(path_to_binary)

    # Distro is valid but cant find rocm ver
    found = False
    for binary_path in binary_paths:
        if Path(binary_path).exists():
            found = True
            path_to_binary = binary_path
            break

    if not found:
        console_error("roofline", f"Unable to locate expected binary ({binary_paths}).")

    my_args = [
        path_to_binary,
        "-o",
        f"{args.path}/roofline.csv",
        "-d",
        str(args.device),
    ]
    if args.quiet:
        my_args += "--quiet"

    subprocess.run(my_args, check=True)


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
                    delta_time = group["End_Timestamp"] - group["Start_Timestamp"]
                    median_delta_time = delta_time.median()
                    merged_row[col] = merged_row["Start_Timestamp"] + median_delta_time
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
