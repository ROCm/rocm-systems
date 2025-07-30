##############################################################################bl
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import glob
import io
import json
import locale
import logging
import os
import pathlib
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from itertools import product
from pathlib import Path as path
from typing import Optional

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
from utils.mi_gpu_spec import mi_gpu_specs

rocprof_cmd = ""
rocprof_args = ""
spi_pipe_counter_regexs = [r"SPI_CS\d+_(.*)", r"SPI_CSQ_P\d+_(.*)"]


def is_tcc_channel_counter(counter):
    return counter.startswith("TCC") and counter.endswith("]")


def add_counter_extra_config_input_yaml(
    data: dict,
    counter_name: str,
    description: str,
    expression: str,
    architectures: list,
    properties: list = None,
) -> dict:
    """
    Add a new counter to the rocprofiler-sdk dictionary.
    Initialize missing parts if data is empty or incomplete.
    Enforces that 'architectures' and 'properties' are lists for correct YAML list serialization.
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


def extract_counter_info_extra_config_input_yaml(
    data: dict, counter_name: str
) -> Optional[dict]:
    """
    Extract the full counter dictionary from 'data' for the given counter_name.

    Args:
        data (dict): The source YAML dict.
        counter_name (str): The counter to find.

    Returns:
        Optional[dict]: The full counter dict if found, else None.
    """
    counters = data.get("rocprofiler-sdk", {}).get("counters", [])
    for counter in counters:
        if counter.get("name") == counter_name:
            return counter
    return None


def using_v1():
    return "ROCPROF" in os.environ.keys() and os.environ["ROCPROF"].endswith("rocprof")


def using_v3():
    return "ROCPROF" not in os.environ.keys() or (
        "ROCPROF" in os.environ.keys()
        and (
            os.environ["ROCPROF"].endswith("rocprofv3")
            or os.environ["ROCPROF"] == "rocprofiler-sdk"
        )
    )


def get_version(rocprof_compute_home) -> dict:
    """Return ROCm Compute Profiler versioning info"""

    # symantic version info - note that version file(s) can reside in
    # two locations depending on development vs formal install
    searchDirs = [rocprof_compute_home, rocprof_compute_home.parent]
    found = False
    versionDir = None

    for dir in searchDirs:
        version = str(path(dir).joinpath("VERSION"))
        try:
            with open(version, "r") as file:
                VER = file.read().replace("\n", "")
                found = True
                versionDir = dir
                break
        except:
            pass
    if not found:
        console_error("Cannot find VERSION file at {}".format(searchDirs))

    # git version info
    try:
        success, output = capture_subprocess_output(
            ["git", "-C", versionDir, "log", "--pretty=format:%h", "-n", "1"],
        )
        if success:
            SHA = output
            MODE = "dev"
        else:
            raise Exception(output)
    except:
        try:
            shaFile = path(versionDir).joinpath("VERSION.sha").absolute().resolve()
            with open(shaFile, "r") as file:
                SHA = file.read().replace("\n", "")
                MODE = "release"
        except Exception:
            SHA = "unknown"
            MODE = "unknown"

    versionData = {"version": VER, "sha": SHA, "mode": MODE}
    return versionData


def get_version_display(version, sha, mode):
    """Pretty print versioning info"""
    buf = io.StringIO()
    print("-" * 40, file=buf)
    print("rocprofiler-compute version: %s (%s)" % (version, mode), file=buf)
    print("Git revision:     %s" % sha, file=buf)
    print("-" * 40, file=buf)
    return buf.getvalue()


def detect_rocprof(args):
    """Detect loaded rocprof version. Resolve path and set cmd globally."""
    global rocprof_cmd

    if os.environ.get("ROCPROF") == "rocprofiler-sdk":
        if not path(args.rocprofiler_sdk_library_path).exists():
            console_error(
                "Could not find rocprofiler-sdk library at "
                + args.rocprofiler_sdk_library_path
            )
        rocprof_cmd = "rocprofiler-sdk"
        console_debug("rocprof_cmd is {}".format(rocprof_cmd))
        console_debug(
            "rocprofiler_sdk_path is {}".format(args.rocprofiler_sdk_library_path)
        )
        return rocprof_cmd

    # detect rocprof
    if not "ROCPROF" in os.environ.keys():
        # default rocprof
        rocprof_cmd = "rocprofv3"
    else:
        rocprof_cmd = os.environ["ROCPROF"]

    # resolve rocprof path
    rocprof_path = shutil.which(rocprof_cmd)

    if not rocprof_path:
        rocprof_cmd = "rocprofv3"
        console_warning(
            "Unable to resolve path to %s binary. Reverting to default." % rocprof_cmd
        )
        rocprof_path = shutil.which(rocprof_cmd)
        if not rocprof_path:
            console_error(
                "Please verify installation or set ROCPROF environment variable with full path."
            )
    else:
        # Resolve any sym links in file path
        rocprof_path = str(path(rocprof_path.rstrip("\n")).resolve())
        console_debug("ROC Profiler: " + str(rocprof_path))

    console_debug("rocprof_cmd is {}".format(str(rocprof_cmd)))
    return rocprof_cmd  # TODO: Do we still need to return this? It's not being used in the function call


def store_app_cmd(args):
    global rocprof_args
    rocprof_args = args


def capture_subprocess_output(
    subprocess_args, new_env=None, profileMode=False, enable_logging=True
):
    console_debug("subprocess", "Running: " + " ".join(subprocess_args))
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if new_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=new_env,
        )
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream, mask):
        try:
            # Because the process' output is line buffered, there's only ever one
            # line to read when this function is called
            line = stream.readline()
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
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)


# Create a dictionary that maps agent ID to agent objects
def get_agent_dict(data):
    agents = data["rocprofiler-sdk-tool"][0]["agents"]

    agent_map = {}

    for agent in agents:
        agent_id = agent["id"]["handle"]
        agent_map[agent_id] = agent

    return agent_map


# Returns a dictionary that maps agent ID to GPU ID
# starting at 0.
def get_gpuid_dict(data):

    agents = data["rocprofiler-sdk-tool"][0]["agents"]

    agent_list = []

    # Get agent ID and node_id for GPU agents only
    for agent in agents:

        if agent["type"] == 2:
            agent_id = agent["id"]["handle"]
            node_id = agent["node_id"]
            agent_list.append((agent_id, node_id))

    # Sort by node ID
    agent_list.sort(key=lambda x: x[1])

    # Map agent ID to node id
    map = {}
    gpu_id = 0
    for agent in agent_list:
        map[agent[0]] = gpu_id
        gpu_id = gpu_id + 1

    return map


# Create a dictionary that maps counter ID to counter objects
def v3_json_get_counters(data):
    counters = data["rocprofiler-sdk-tool"][0]["counters"]

    counter_map = {}

    for counter in counters:
        counter_id = counter["id"]["handle"]
        agent_id = counter["agent_id"]["handle"]

        counter_map[(agent_id, counter_id)] = counter

    return counter_map


def v3_json_get_dispatches(data):
    records = data["rocprofiler-sdk-tool"][0]["buffer_records"]

    records_map = {}

    for rec in records["kernel_dispatch"]:
        id = rec["correlation_id"]["internal"]

        records_map[id] = rec

    return records_map


def v3_json_to_csv(json_file_path, csv_file_path):

    f = open(json_file_path, "rt")
    data = json.load(f)

    dispatch_records = v3_json_get_dispatches(data)
    dispatches = data["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"]
    kernel_symbols = data["rocprofiler-sdk-tool"][0]["kernel_symbols"]
    agents = get_agent_dict(data)
    pid = data["rocprofiler-sdk-tool"][0]["metadata"]["pid"]

    gpuid_map = get_gpuid_dict(data)

    counter_info = v3_json_get_counters(data)

    # CSV headers. If there are no dispatches we still end up with a valid CSV file.
    csv_data = dict.fromkeys(
        [
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
    )

    for key in csv_data:
        csv_data[key] = []

    for d in dispatches:

        dispatch_info = d["dispatch_data"]["dispatch_info"]

        agent_id = dispatch_info["agent_id"]["handle"]

        kernel_id = dispatch_info["kernel_id"]

        row = {}

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

        # TODO: Accum VGPR is missing from rocprofv3 output.
        row["Accum_VGPR"] = 0

        row["SGPR"] = d["sgpr_count"]
        row["Wave_Size"] = agents[agent_id]["wave_front_size"]

        row["Kernel_Name"] = kernel_symbols[kernel_id]["formatted_kernel_name"]

        id = d["dispatch_data"]["correlation_id"]["internal"]
        rec = dispatch_records[id]

        row["Start_Timestamp"] = rec["start_timestamp"]
        row["End_Timestamp"] = rec["end_timestamp"]
        row["Correlation_ID"] = d["dispatch_data"]["correlation_id"]["external"]

        # Get counters
        ctrs = {}

        records = d["records"]
        for r in records:
            ctr_id = r["counter_id"]["handle"]
            value = r["value"]

            name = counter_info[(agent_id, ctr_id)]["name"]

            if name.endswith("_ACCUM"):
                # It's an accumulate counter. Omniperf expects the accumulated value
                # to be in SQ_ACCUM_PREV_HIRES.
                name = "SQ_ACCUM_PREV_HIRES"

            # Some counters appear multiple times and need to be summed
            if name in ctrs:
                ctrs[name] += value
            else:
                ctrs[name] = value

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


def v3_counter_csv_to_v2_csv(counter_file, agent_info_filepath, converted_csv_file):
    """
    Convert the counter file of csv output for a certain csv from rocprofv3 format to rocprfv2 format.
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

    # NB: Agent_Id is int in older rocporfv3, now switched to string with prefix "Agent ". We need to make sure handle both cases.
    console_debug(
        "The type of Agent ID from counter csv file is {}".format(
            result["Agent_Id"].dtype
        )
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
                'Parsing rocprofv3 csv output: Error of getting "Agent_Id", the error message "{}"'.format(
                    e
                )
            )

    # Grab the Wave_Front_Size column from agent info
    result = result.merge(
        pd_agent_info[["Node_Id", "Wave_Front_Size"]],
        left_on="Agent_Id",
        right_on="Node_Id",
        how="left",
    )

    # Map agent ID (Node_Id) to GPU_ID
    gpu_id_map = {}
    gpu_id = 0
    for idx, row in pd_agent_info.iterrows():
        if row["Agent_Type"] == "GPU":
            agent_id = row["Node_Id"]
            gpu_id_map[agent_id] = gpu_id
            gpu_id = gpu_id + 1

    # Update Agent_Id for each record to match GPU ID
    for idx, row in result["Agent_Id"].items():
        agent_id = result.at[idx, "Agent_Id"]
        result.at[idx, "Agent_Id"] = gpu_id_map[agent_id]

    # Drop the 'Node_Id' column if you don't need it in the final DataFrame
    result.drop(columns="Node_Id", inplace=True)

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

    # Rename the accumulate counter to SQ_ACCUM_PREV_HIRES.
    for col in result.columns:
        if col.endswith("_ACCUM"):
            result.rename(columns={col: "SQ_ACCUM_PREV_HIRES"}, inplace=True)

    result.to_csv(converted_csv_file, index=False)


def parse_text(text_file):
    """
    Parse the text file to get the pmc counters.
    """

    def process_line(line):
        if "pmc:" not in line:
            return ""
        line = line.strip()
        pos = line.find("#")
        if pos >= 0:
            line = line[0:pos]

        def _dedup(_line, _sep):
            for itr in _sep:
                _line = " ".join(_line.split(itr))
            return _line.strip()

        # remove tabs and duplicate spaces
        return _dedup(line.replace("pmc:", ""), ["\n", "\t", " "]).split(" ")

    with open(text_file, "r") as file:
        return [
            counter
            for litr in [process_line(itr) for itr in file.readlines()]
            for counter in litr
        ]


def run_prof(
    fname,
    profiler_options,
    workload_dir,
    mspec,
    loglevel,
    format_rocprof_output,
    retain_rocpd_output=False,
):
    fbase = path(fname).stem

    console_debug("pmc file: %s" % path(fname).name)

    # standard rocprof options
    if rocprof_cmd == "rocprofiler-sdk":
        options = profiler_options
        options["ROCPROF_COUNTER_COLLECTION"] = "1"
        options["ROCPROF_COUNTERS"] = "pmc: " + " ".join(parse_text(fname))
    else:
        default_options = ["-i", fname]
        options = default_options + profiler_options

    if using_v3():
        if rocprof_cmd == "rocprofiler-sdk":
            options["ROCPROF_AGENT_INDEX"] = "absolute"
        else:
            options = ["-A", "absolute"] + options

    new_env = os.environ.copy()

    if using_v3():
        # Default counter definitions
        if rocprof_cmd == "rocprofiler-sdk":
            counter_defs_path = (
                path(options["ROCP_TOOL_LIBRARIES"])
                .resolve()
                .parent.parent.parent.joinpath(
                    "share", "rocprofiler-sdk", "counter_defs.yaml"
                )
            )
        else:
            counter_defs_path = (
                path(shutil.which(rocprof_cmd))
                .resolve()
                .parent.parent.joinpath("share", "rocprofiler-sdk", "counter_defs.yaml")
            )
        # Custom counter definitions for MI 100
        if mspec.gpu_model.lower() == "mi100":
            counter_defs_path = (
                config.rocprof_compute_home
                / "rocprof_compute_soc"
                / "profile_configs"
                / "gfx908_counter_defs.yaml"
            )
        # Read counter definitions
        with open(counter_defs_path, "r") as file:
            counter_defs = yaml.safe_load(file)
        # Get extra counter definitions
        path_counter_config_yaml = path(fname).with_suffix(".yaml")
        if path_counter_config_yaml.exists():
            with open(path_counter_config_yaml, "r") as file:
                extra_counter_defs = yaml.safe_load(file)
            # Merge extra counter definitions
            counter_defs["rocprofiler-sdk"]["counters"].extend(
                extra_counter_defs["rocprofiler-sdk"]["counters"]
            )
        # Write counter definitions to a temporary file
        tmpfile_path = (
            path(tempfile.mkdtemp(prefix="rocprof_counter_defs_", dir="/tmp"))
            / "counter_defs.yaml"
        )
        with open(tmpfile_path, "w") as tmpfile:
            yaml.dump(counter_defs, tmpfile, default_flow_style=False, sort_keys=False)
        # Set rocprofiler sdk counter definitions
        new_env["ROCPROFILER_METRICS_PATH"] = str(tmpfile_path.parent)
        console_debug(
            f"Adding env var for counter definitions: ROCPROFILER_METRICS_PATH={new_env['ROCPROFILER_METRICS_PATH']}"
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

    is_timestamps = False
    if path(fname).name == "timestamps.txt":
        is_timestamps = True
    time_1 = time.time()

    if rocprof_cmd == "rocprofiler-sdk":
        app_cmd = options.pop("APP_CMD")
        for key, value in options.items():
            new_env[key] = value
        console_debug("rocprof sdk env vars: {}".format(new_env))
        console_debug("rocprof sdk user provided command: {}".format(app_cmd))
        success, output = capture_subprocess_output(
            app_cmd, new_env=new_env, profileMode=True
        )
    else:
        console_debug("rocprof command: {}".format([rocprof_cmd] + options))
        # profile the app
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=new_env, profileMode=True
        )

    time_2 = time.time()
    console_debug(
        "Finishing subprocess of fname {}, the time it takes was {} m {} sec ".format(
            fname, int((time_2 - time_1) / 60), str((time_2 - time_1) % 60)
        )
    )

    # Delete counter definition temporary directory
    if new_env.get("ROCPROFILER_METRICS_PATH"):
        shutil.rmtree(new_env["ROCPROFILER_METRICS_PATH"], ignore_errors=True)

    if not success:
        if loglevel > logging.INFO:
            for line in output.splitlines():
                console_error(output, exit=False)
        console_error("Profiling execution failed.")

    results_files = []

    if format_rocprof_output == "rocpd":
        if rocprof_cmd == "rocprofiler-sdk" or rocprof_cmd.endswith("v3"):
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
        else:
            console_error(
                "rocpd output format is only supported with rocprofiler-sdk or rocprofv3."
            )
    elif rocprof_cmd.endswith("v2"):
        # rocprofv2 has separate csv files for each process
        results_files = glob.glob(workload_dir + "/out/pmc_1/results_*.csv")

        if len(results_files) == 0:
            return

        # Combine results into single CSV file
        combined_results = pd.concat(
            [pd.read_csv(f) for f in results_files], ignore_index=True
        )

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )
    elif rocprof_cmd.endswith("v3") or rocprof_cmd == "rocprofiler-sdk":
        # rocprofv3 requires additional processing for each process
        results_files = process_rocprofv3_output(
            format_rocprof_output, workload_dir, is_timestamps
        )

        if rocprof_cmd == "rocprofiler-sdk":
            # TODO: as rocprofv3 --kokkos-trace feature improves, rocprof-compute should make updates accordingly
            if "ROCPROF_HIP_RUNTIME_API_TRACE" in options:
                process_hip_trace_output(workload_dir, fbase)
        else:
            if "--kokkos-trace" in options:
                # TODO: as rocprofv3 --kokkos-trace feature improves, rocprof-compute should make updates accordingly
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
                f"Cannot write results for {fbase}.csv due to no counter csv files generated."
            )
            return

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )

    if not using_v3() and not using_v1():
        # flatten tcc for applicable mi300 input
        f = path(workload_dir + "/out/pmc_1/results_" + fbase + ".csv")
        xcds = mi_gpu_specs.get_num_xcds(
            mspec.gpu_arch, mspec.gpu_model, mspec.compute_partition
        )
        df = flatten_tcc_info_across_xcds(f, xcds, int(mspec._l2_banks))
        df.to_csv(f, index=False)

    if path(workload_dir + "/out").exists():
        # copy and remove out directory if needed
        shutil.copyfile(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv",
            workload_dir + "/" + fbase + ".csv",
        )
        # Remove temp directory
        shutil.rmtree(workload_dir + "/" + "out")

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
    df = pd.read_csv(workload_dir + "/" + fbase + ".csv")
    df.rename(columns=output_headers, inplace=True)
    df.to_csv(workload_dir + "/" + fbase + ".csv", index=False)


def pc_sampling_prof(
    method, interval, workload_dir, appcmd, rocprofiler_sdk_library_path
):
    """
    Run rocprof with pc sampling. Current support v3 only.
    """
    # Todo:
    #   - precheck with rocprofv3 –-list-avail

    unit = "time" if method == "host_trap" else "cycles"

    if rocprof_cmd == "rocprofiler-sdk":
        rocm_libdir = str(pathlib.Path(rocprofiler_sdk_library_path).parent)
        rocprofiler_sdk_tool_path = str(
            pathlib.Path(rocm_libdir).joinpath(
                "rocprofiler-sdk/librocprofiler-sdk-tool.so"
            )
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
        }
        new_env = os.environ.copy()
        for key, value in options.items():
            new_env[key] = value
        console_debug("pc sampling rocprof sdk env vars: {}".format(new_env))
        console_debug("pc sampling rocprof sdk user provided command: {}".format(appcmd))
        success, output = capture_subprocess_output(
            appcmd, new_env=new_env, profileMode=True
        )
    else:
        options = [
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
            "ps_file",  # todo: sync up with the name from source in 2100_.yaml
            "--",
            appcmd,
        ]
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=os.environ.copy(), profileMode=True
        )

    if not success:
        console_error("PC sampling failed.")


def process_rocprofv3_output(rocprof_output, workload_dir, is_timestamps):
    """
    rocprofv3 specific output processing.
    takes care of json or csv formats, for csv format, additional processing is performed.
    """
    results_files_csv = {}

    if rocprof_output == "json":
        results_files_json = glob.glob(workload_dir + "/out/pmc_1/*/*.json")

        for json_file in results_files_json:
            csv_file = pathlib.Path(json_file).with_suffix(".csv")
            v3_json_to_csv(json_file, csv_file)
        results_files_csv = glob.glob(workload_dir + "/out/pmc_1/*/*.csv")

    elif rocprof_output == "csv":
        counter_info_csvs = glob.glob(
            workload_dir + "/out/pmc_1/*/*_counter_collection.csv"
        )
        existing_counter_files_csv = [d for d in counter_info_csvs if path(d).is_file()]

        if existing_counter_files_csv:
            for counter_file in existing_counter_files_csv:
                counter_path = path(counter_file)
                current_dir = counter_path.parent

                agent_info_filepath = current_dir / counter_path.name.replace(
                    "_counter_collection", "_agent_info"
                )

                if not agent_info_filepath.is_file():
                    raise ValueError(
                        '{} has no coresponding "agent info" file'.format(counter_file)
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

            results_files_csv = glob.glob(workload_dir + "/out/pmc_1/*/*_converted.csv")
        elif is_timestamps:
            # when the input is timestamps, we know counter csv file is not generated and will instead parse kernel trace file
            results_files_csv = glob.glob(
                workload_dir + "/out/pmc_1/*/*_kernel_trace.csv"
            )
        else:
            # when the input is not for timestamps, and counter csv file is not generated, we assume failed rocprof run and will completely bypass the file generation and merging for current pmc
            results_files_csv = []
    else:
        console_error("The output file of rocprofv3 can only support json or csv!!!")

    return results_files_csv


@demarcate
def process_kokkos_trace_output(workload_dir, fbase):
    # marker api trace csv files are generated for each process
    marker_api_trace_csvs = glob.glob(
        workload_dir + "/out/pmc_1/*/*_marker_api_trace.csv"
    )
    existing_marker_files_csv = [d for d in marker_api_trace_csvs if path(d).is_file()]

    # concate and output marker api trace info
    combined_results = pd.concat(
        [pd.read_csv(f) for f in existing_marker_files_csv], ignore_index=True
    )

    combined_results.to_csv(
        workload_dir + "/out/pmc_1/results_" + fbase + "_marker_api_trace.csv",
        index=False,
    )

    if path(workload_dir + "/out").exists():
        shutil.copyfile(
            workload_dir + "/out/pmc_1/results_" + fbase + "_marker_api_trace.csv",
            workload_dir + "/" + fbase + "_marker_api_trace.csv",
        )


@demarcate
def process_hip_trace_output(workload_dir, fbase):
    # marker api trace csv files are generated for each process
    hip_api_trace_csvs = glob.glob(workload_dir + "/out/pmc_1/*/*_hip_api_trace.csv")
    existing_hip_files_csv = [d for d in hip_api_trace_csvs if path(d).is_file()]

    # concate and output marker api trace info
    combined_results = pd.concat(
        [pd.read_csv(f) for f in existing_hip_files_csv], ignore_index=True
    )

    combined_results.to_csv(
        workload_dir + "/out/pmc_1/results_" + fbase + "_hip_api_trace.csv",
        index=False,
    )

    if path(workload_dir + "/out").exists():
        shutil.copyfile(
            workload_dir + "/out/pmc_1/results_" + fbase + "_hip_api_trace.csv",
            workload_dir + "/" + fbase + "_hip_api_trace.csv",
        )


def replace_timestamps(workload_dir):

    if not path(workload_dir, "timestamps.csv").is_file():
        return

    df_stamps = pd.read_csv(workload_dir + "/timestamps.csv")
    if "Start_Timestamp" in df_stamps.columns and "End_Timestamp" in df_stamps.columns:
        # Update timestamps for all *.csv output files
        for fname in glob.glob(workload_dir + "/" + "*.csv"):
            if path(fname).name != "sysinfo.csv":
                df_pmc_perf = pd.read_csv(fname)

                df_pmc_perf["Start_Timestamp"] = df_stamps["Start_Timestamp"]
                df_pmc_perf["End_Timestamp"] = df_stamps["End_Timestamp"]
                df_pmc_perf.to_csv(fname, index=False)
    else:
        console_warning(
            "Incomplete profiling data detected. Unable to update timestamps.\n"
        )


def gen_sysinfo(workload_name, workload_dir, app_cmd, skip_roof, mspec, soc):
    console_debug("[gen_sysinfo]")
    df = mspec.get_class_members()

    # Append workload information to machine specs
    df["command"] = app_cmd
    df["workload_name"] = workload_name

    blocks = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
    if hasattr(soc, "roofline_obj") and (not skip_roof):
        blocks.append("roofline")
    df["ip_blocks"] = "|".join(blocks)

    # Save csv
    df.to_csv(workload_dir + "/" + "sysinfo.csv", index=False)


def detect_roofline(mspec):
    from utils import specs

    rocm_ver = int(mspec.rocm_version[:1])

    target_binary = {"rocm_ver": rocm_ver, "distro": "override", "path": None}

    os_release = path("/etc/os-release").read_text()
    ubuntu_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)
    rhel_distro = specs.search(r'PLATFORM_ID="(.*?)"', os_release)
    sles_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)

    if "ROOFLINE_BIN" in os.environ.keys():
        rooflineBinary = os.environ["ROOFLINE_BIN"]
        if path(rooflineBinary).exists():
            msg = "Detected user-supplied binary --> ROOFLINE_BIN = %s\n" % rooflineBinary
            console_warning("roofline", msg)
            # distro stays marked as override and path value is substituted in
            target_binary["path"] = rooflineBinary
            return target_binary
        else:
            msg = (
                "user-supplied path to binary not accessible --> ROOFLINE_BIN = %s\n"
                % rooflineBinary
            )
            console_error("roofline", msg)

    # Must be a valid RHEL machine
    elif rocm_ver == 6 and (
        rhel_distro == "platform:el8"
        or rhel_distro == "platform:al8"
        or rhel_distro == "platform:el9"
        or rhel_distro == "platform:el10"
    ):
        # RHEL8 supported up to ROCm6
        distro = "platform:el8"
    elif rocm_ver == 7 and (
        rhel_distro == "platform:el9" or rhel_distro == "platform:el10"
    ):
        # ROCm7 supports RHEL9 and above
        distro = "platform:el9"

    # Must be a valid SLES machine
    elif (
        (type(sles_distro) == str and len(sles_distro) >= 3)
        and sles_distro[:2] == "15"  # confirm string and len
        and int(sles_distro[3]) >= 6  # SLES15 and SP >= 6
    ):
        # Use SP6 binary for all forward compatible service pack versions
        distro = "15.6"

    # Must be a valid Ubuntu machine
    elif ubuntu_distro == "22.04" or ubuntu_distro == "24.04":
        distro = "22.04"

    else:
        console_error("roofline", "Cannot find a valid binary for your operating system")

    # distro gets assigned, to follow default roofline bin location and nomenclature
    target_binary["distro"] = distro
    return target_binary


def mibench(args, mspec):
    """Run roofline microbenchmark to generate peek BW and FLOP measurements."""
    console_log("roofline", "No roofline data found. Generating...")

    distro_map = {
        "platform:el8": "rhel8",
        "platform:el9": "rhel9",
        "15.6": "sles15sp6",
        "22.04": "ubuntu22_04",
    }

    binary_paths = []

    target_binary = detect_roofline(mspec)
    if target_binary["distro"] == "override":
        binary_paths.append(target_binary["path"])
    else:
        # check two potential locations for roofline binaries due to differences in
        # development usage vs formal install
        potential_paths = [
            "%s/utils/rooflines/roofline" % config.rocprof_compute_home,
            "%s/bin/roofline" % config.rocprof_compute_home.parent.parent,
        ]

        for dir in potential_paths:
            path_to_binary = (
                dir
                + "-"
                + distro_map[target_binary["distro"]]
                + "-rocm"
                + str(target_binary["rocm_ver"])
            )
            binary_paths.append(path_to_binary)

    # Distro is valid but cant find rocm ver
    found = False
    for path in binary_paths:
        if pathlib.Path(path).exists():
            found = True
            path_to_binary = path
            break

    if not found:
        console_error("roofline", "Unable to locate expected binary (%s)." % binary_paths)

    my_args = [
        path_to_binary,
        "-o",
        args.path + "/" + "roofline.csv",
        "-d",
        str(args.device),
    ]
    if args.quiet:
        my_args += "--quiet"
    subprocess.run(
        my_args,
        check=True,
    )


def flatten_tcc_info_across_xcds(file, xcds, tcc_channel_per_xcd):
    """
    Flatten TCC per channel counters across all XCDs in partition.
    NB: This func highly depends on the default behavior of rocprofv2 on MI300,
        which might be broken anytime in the future!
    """
    df_orig = pd.read_csv(file)
    # display(df_orig.info)

    ### prepare column headers
    tcc_cols_orig = []
    non_tcc_cols_orig = []
    for c in df_orig.columns.to_list():
        if "TCC" in c:
            tcc_cols_orig.append(c)
        else:
            non_tcc_cols_orig.append(c)
    # print(tcc_cols_orig)

    cols = non_tcc_cols_orig
    tcc_cols_in_group = {}
    for i in range(0, xcds):
        tcc_cols_in_group[i] = []

    for col in tcc_cols_orig:
        for i in range(0, xcds):
            # filter the channel index only
            p = re.compile(r"\[(\d+)\]")
            # pick up the 1st element only
            r = (
                lambda match: "["
                + str(int(match.group(1)) + i * tcc_channel_per_xcd)
                + "]"
            )
            tcc_cols_in_group[i].append(re.sub(pattern=p, repl=r, string=col))

    for i in range(0, xcds):
        # print(tcc_cols_in_group[i])
        cols += tcc_cols_in_group[i]
    # print(cols)
    df = pd.DataFrame(columns=cols)

    ### Rearrange data with extended column names

    # print(len(df_orig.index))
    for idx in range(0, len(df_orig.index), xcds):
        # assume the front none TCC columns are the same for all XCCs
        df_non_tcc = df_orig.iloc[idx].filter(regex=r"^(?!.*TCC).*$")
        # display(df_non_tcc)
        flatten_list = df_non_tcc.tolist()

        # extract all tcc from one dispatch
        # NB: assuming default contiguous order might not be safe!
        df_tcc_all = df_orig.iloc[idx : (idx + xcds)].filter(regex="TCC")
        # display(df_tcc_all)

        for idx, row in df_tcc_all.iterrows():
            flatten_list += row.tolist()
        # print(len(df.index), len(flatten_list), len(df.columns), flatten_list)
        # NB: It is not the best perf to append a row once a time
        df.loc[len(df.index)] = flatten_list

    return df


def get_submodules(package_name):
    """List all submodules for a target package"""
    import importlib
    import pkgutil

    submodules = []

    # walk all submodules in target package
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__):
        pretty_name = name.split("_", 1)[1].replace("_", "")
        # ignore base submodule, add all other
        if pretty_name != "base":
            submodules.append(pretty_name)

    return submodules


def is_workload_empty(path):
    """Peek workload directory to verify valid profiling output"""
    pmc_perf_path = path + "/pmc_perf.csv"
    if pathlib.Path(pmc_perf_path).is_file():
        temp_df = pd.read_csv(pmc_perf_path)
        if temp_df.dropna().empty:
            console_error(
                "profiling"
                "Found empty cells in %s.\nProfiling data could be corrupt."
                % pmc_perf_path
            )

    else:
        console_error("analysis", "No profiling data found.")


def print_status(msg):
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def set_locale_encoding():
    try:
        # Attempt to set the locale to 'C.UTF-8'
        locale.setlocale(locale.LC_ALL, "C.UTF-8")
    except locale.Error:
        # If 'C.UTF-8' is not available, check if the current locale is UTF-8 based
        current_locale = locale.getdefaultlocale()
        if current_locale and "UTF-8" in current_locale[1]:
            try:
                locale.setlocale(locale.LC_ALL, current_locale[0])
            except locale.Error as error:
                console_error(
                    "Failed to set locale to the current UTF-8-based locale.",
                    exit=False,
                )
                console_error(error)
        else:
            console_error(
                "Please ensure that a UTF-8-based locale is available on your system.",
                exit=False,
            )


def reverse_multi_index_df_pmc(final_df):
    """
    Util function to decompose multi-index dataframe.
    """
    # Check if the columns have more than one level
    if len(final_df.columns.levels) < 2:
        raise ValueError("Input DataFrame does not have a multi-index column.")

    # Extract the first level of the MultiIndex columns (the file names)
    coll_levels = final_df.columns.get_level_values(0).unique().tolist()

    # Initialize the list of DataFrames
    dfs = []

    # Loop through each 'coll_level' and rebuild the DataFrames
    for level in coll_levels:
        # Select columns that belong to the current 'coll_level'
        columns_for_level = final_df.xs(level, axis=1, level=0)

        # Append the DataFrame for this level
        dfs.append(columns_for_level)

    # Return the list of DataFrames and the column levels
    return dfs, coll_levels


def merge_counters_spatial_multiplex(df_multi_index):
    """
    For spatial multiplexing, this merges counter values for the same kernel that runs on different devices. For time stamp, start time stamp will use median while for end time stamp, it will be equal to the summation between median start stamp and median delta time.
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

    result_dfs = []

    # TODO: will need optimize to avoid this convertion to single index format and do merge directly on multi-index dataframe
    dfs, coll_levels = reverse_multi_index_df_pmc(df_multi_index)

    for df in dfs:
        kernel_name_column_name = "Kernel_Name"
        if not "Kernel_Name" in df and "Name" in df:
            kernel_name_column_name = "Name"

        # Find the values in Kernel_Name that occur more than once
        kernel_single_occurances = df[kernel_name_column_name].value_counts().index

        # Define a list to store the merged rows
        result_data = []

        for kernel_name in kernel_single_occurances:
            # Get all rows for the current kernel_name
            group = df[df[kernel_name_column_name] == kernel_name]

            # Create a dictionary to store the merged row for the current group
            merged_row = {}

            # Process non-counter columns
            for col in [
                col for col in non_counter_column_index if col not in expired_column_index
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

            # Process counter columns (assumed to be all columns not in non_counter_column_index)
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


def convert_metric_id_to_panel_info(metric_id):
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
    if 0 < len(tokens) < 4:
        # File id
        file_id = str(int(tokens[0]))
        # 4 -> 04
        if len(file_id) == 1:
            file_id = f"0{file_id}"
        # Multiply integer by 100
        file_id = f"{file_id}00"
        # Panel id
        if len(tokens) > 1:
            panel_id = int(tokens[0]) * 100
            panel_id += int(tokens[1])
        else:
            panel_id = None
        # Metric id
        if len(tokens) > 2:
            metric_id = int(tokens[2])
        else:
            metric_id = None
        return (file_id, panel_id, metric_id)
    else:
        raise Exception(f"Invalid metric id: {metric_id}")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
