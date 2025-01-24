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
import time
from collections import OrderedDict
from itertools import product
from pathlib import Path as path

import pandas as pd

import config

rocprof_cmd = ""
rocprof_args = ""


def demarcate(function):
    def wrap_function(*args, **kwargs):
        logging.trace("----- [entering function] -> %s()" % (function.__qualname__))
        result = function(*args, **kwargs)
        logging.trace("----- [exiting  function] -> %s()" % function.__qualname__)
        return result

    return wrap_function


def console_error(*argv, exit=True):
    if len(argv) > 1:
        logging.error(f"[{argv[0]}] {argv[1]}")
    else:
        logging.error(f"{argv[0]}")
    if exit:
        sys.exit(1)


def console_log(*argv, indent_level=0):
    indent = ""
    if indent_level >= 1:
        indent = " " * 3 * indent_level + "|-> "  # spaces per indent level

    if len(argv) > 1:
        logging.info(indent + f"[{argv[0]}] {argv[1]}")
    else:
        logging.info(indent + f"{argv[0]}")


def console_debug(*argv):
    if len(argv) > 1:
        logging.debug(f"[{argv[0]}] {argv[1]}")
    else:
        logging.debug(f"{argv[0]}")


def console_warning(*argv):
    if len(argv) > 1:
        logging.warning(f"[{argv[0]}] {argv[1]}")
    else:
        logging.warning(f"{argv[0]}")


def trace_logger(message, *args, **kwargs):
    logging.log(logging.TRACE, message, *args, **kwargs)


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
    gitDir = str(path(rocprof_compute_home.parent).joinpath(".git"))
    if (shutil.which("git") is not None) and path(gitDir).exists():
        gitQuery = subprocess.run(
            ["git", "log", "--pretty=format:%h", "-n", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if gitQuery.returncode != 0:
            SHA = "unknown"
            MODE = "unknown"
        else:
            SHA = gitQuery.stdout.decode("utf-8")
            MODE = "dev"
    else:
        shaFile = str(path(versionDir).joinpath("VERSION.sha"))
        try:
            with open(shaFile, "r") as file:
                SHA = file.read().replace("\n", "")
        except EnvironmentError:
            console_error("Cannot find VERSION.sha file at {}".format(shaFile))
            sys.exit(1)

        MODE = "release"

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


def detect_rocprof():
    """Detect loaded rocprof version. Resolve path and set cmd globally."""
    global rocprof_cmd
    # detect rocprof
    if not "ROCPROF" in os.environ.keys():
        rocprof_cmd = "rocprof"
    else:
        rocprof_cmd = os.environ["ROCPROF"]

    # resolve rocprof path
    rocprof_path = shutil.which(rocprof_cmd)

    if not rocprof_path:
        rocprof_cmd = "rocprof"
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


def capture_subprocess_output(subprocess_args, new_env=None, profileMode=False):
    global rocprof_args
    # Format command for debug messages, formatting for rocprofv1 and rocprofv2
    command = " ".join(rocprof_args)
    console_debug("subprocess", "Running: " + command)
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
            "SGPR_Count",
            "Start_Timestamp",
            "End_Timestamp",
        ],
        columns="Counter_Name",
        values="Counter_Value",
    ).reset_index()

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

    # Accum_VGPR is currently missing in rocprofv3 output
    result["Accum_VGPR"] = 0

    # Drop the 'Node_Id' column if you don't need it in the final DataFrame
    result.drop(columns="Node_Id", inplace=True)
    result["Accum_VGPR"] = 0

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
        # "":"Accum_VGPR",
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


def run_prof(
    fname, profiler_options, workload_dir, mspec, loglevel, format_rocprof_output
):
    time_0 = time.time()
    fbase = path(fname).stem

    console_debug("pmc file: %s" % path(fname).name)

    # standard rocprof options
    default_options = ["-i", fname]
    options = default_options + profiler_options

    # set required env var for mi300
    new_env = None
    if (
        mspec.gpu_model.lower() == "mi300x_a0"
        or mspec.gpu_model.lower() == "mi300x_a1"
        or mspec.gpu_model.lower() == "mi300a_a0"
        or mspec.gpu_model.lower() == "mi300a_a1"
    ) and (
        path(fname).name == "pmc_perf_13.txt"
        or path(fname).name == "pmc_perf_14.txt"
        or path(fname).name == "pmc_perf_15.txt"
        or path(fname).name == "pmc_perf_16.txt"
        or path(fname).name == "pmc_perf_17.txt"
    ):
        new_env = os.environ.copy()
        new_env["ROCPROFILER_INDIVIDUAL_XCC_MODE"] = "1"

    time_1 = time.time()

    # profile the app
    if new_env:
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, new_env=new_env, profileMode=True
        )
    else:
        success, output = capture_subprocess_output(
            [rocprof_cmd] + options, profileMode=True
        )

    time_2 = time.time()
    console_debug(
        "Finishing subprocess of fname {}, the time it takes was {} m {} sec ".format(
            fname, int((time_2 - time_1) / 60), str((time_2 - time_1) % 60)
        )
    )

    if not success:
        if loglevel > logging.INFO:
            for line in output.splitlines():
                console_error(output, exit=False)
        console_error("Profiling execution failed.")

    if rocprof_cmd.endswith("v2"):
        # rocprofv2 has separate csv files for each process
        results_files = glob.glob(workload_dir + "/out/pmc_1/results_*.csv")

        # Combine results into single CSV file
        combined_results = pd.concat(
            [pd.read_csv(f) for f in results_files], ignore_index=True
        )

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )

    if rocprof_cmd.endswith("v3"):
        results_files_csv = {}
        if format_rocprof_output == "json":
            results_files_json = glob.glob(workload_dir + "/out/pmc_1/*/*.json")

            for json_file in results_files_json:
                csv_file = pathlib.Path(json_file).with_suffix(".csv")
                v3_json_to_csv(json_file, csv_file)
            results_files_csv = glob.glob(workload_dir + "/out/pmc_1/*/*.csv")
        elif format_rocprof_output == "csv":
            counter_info_csvs = glob.glob(
                workload_dir + "/out/pmc_1/*/*_counter_collection.csv"
            )
            existing_counter_files_csv = [
                d for d in counter_info_csvs if path(d).is_file()
            ]

            if len(existing_counter_files_csv) > 0:
                for counter_file in existing_counter_files_csv:
                    current_dir = str(path(counter_file).parent)
                    agent_info_filepath = str(
                        path(current_dir).joinpath(
                            path(counter_file).name.replace(
                                "_counter_collection", "_agent_info"
                            )
                        )
                    )
                    if not path(agent_info_filepath).is_file():
                        raise ValueError(
                            '{} has no coresponding "agent info" file'.format(
                                counter_file
                            )
                        )

                    converted_csv_file = str(
                        path(current_dir).joinpath(
                            path(counter_file).name.replace(
                                "_counter_collection", "_converted"
                            )
                        )
                    )

                    v3_counter_csv_to_v2_csv(
                        counter_file, agent_info_filepath, converted_csv_file
                    )

                results_files_csv = glob.glob(
                    workload_dir + "/out/pmc_1/*/*_converted.csv"
                )
            else:
                results_files_csv = glob.glob(
                    workload_dir + "/out/pmc_1/*/*_kernel_trace.csv"
                )

        else:
            console_error("The output file of rocprofv3 can only support json or csv!!!")

        # Combine results into single CSV file
        combined_results = pd.concat(
            [pd.read_csv(f) for f in results_files_csv], ignore_index=True
        )

        # Overwrite column to ensure unique IDs.
        combined_results["Dispatch_ID"] = range(0, len(combined_results))

        combined_results.to_csv(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", index=False
        )

    if new_env:
        # flatten tcc for applicable mi300 input
        f = path(workload_dir + "/out/pmc_1/results_" + fbase + ".csv")
        xcds = total_xcds(mspec.gpu_model, mspec.compute_partition)
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


def replace_timestamps(workload_dir):
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


def gen_sysinfo(
    workload_name, workload_dir, ip_blocks, app_cmd, skip_roof, roof_only, mspec, soc
):
    df = mspec.get_class_members()

    # Append workload information to machine specs
    df["command"] = app_cmd
    df["workload_name"] = workload_name

    blocks = []
    if ip_blocks == None:
        t = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
        blocks += t
    else:
        blocks += ip_blocks
    if hasattr(soc, "roofline_obj") and (not skip_roof):
        blocks.append("roofline")
    df["ip_blocks"] = "|".join(blocks)

    # Save csv
    df.to_csv(workload_dir + "/" + "sysinfo.csv", index=False)


def detect_roofline(mspec):
    from utils import specs

    rocm_ver = mspec.rocm_version[:1]

    os_release = path("/etc/os-release").read_text()
    ubuntu_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)
    rhel_distro = specs.search(r'PLATFORM_ID="(.*?)"', os_release)
    sles_distro = specs.search(r'VERSION_ID="(.*?)"', os_release)

    if "ROOFLINE_BIN" in os.environ.keys():
        rooflineBinary = os.environ["ROOFLINE_BIN"]
        if path(rooflineBinary).exists():
            console_warning("roofline", "Detected user-supplied binary")
            return {
                "rocm_ver": "override",
                "distro": "override",
                "path": rooflineBinary,
            }
        else:
            msg = "user-supplied path to binary not accessible"
            msg += "--> ROOFLINE_BIN = %s\n" % target_binary
            console_error("roofline", msg)
    elif rhel_distro == "platform:el8" or rhel_distro == "platform:el9":
        # Must be a valid RHEL machine
        distro = "platform:el8"
    elif (
        (type(sles_distro) == str and len(sles_distro) >= 3)
        and sles_distro[:2] == "15"  # confirm string and len
        and int(sles_distro[3]) >= 3  # SLES15 and SP >= 3
    ):
        # Must be a valid SLES machine
        # Use SP3 binary for all forward compatible service pack versions
        distro = "15.3"
    elif ubuntu_distro == "20.04" or ubuntu_distro == "22.04" or ubuntu_distro == "24.04":
        # Must be a valid Ubuntu machine
        distro = ubuntu_distro
    else:
        console_error("roofline", "Cannot find a valid binary for your operating system")

    target_binary = {"rocm_ver": rocm_ver, "distro": distro}
    return target_binary


def run_rocscope(args, fname):
    # profile the app
    if args.use_rocscope == True:
        result = shutil.which("rocscope")
        if result:
            rs_cmd = [
                result.stdout.decode("ascii").strip(),
                "metrics",
                "-p",
                args.path,
                "-n",
                args.name,
                "-t",
                fname,
                "--",
            ]
            for i in args.remaining.split():
                rs_cmd.append(i)
            console_log(rs_cmd)
            success, output = capture_subprocess_output(rs_cmd)
            if not success:
                console_error(result.stderr.decode("ascii"))


def mibench(args, mspec):
    """Run roofline microbenchmark to generate peek BW and FLOP measurements."""
    console_log("roofline", "No roofline data found. Generating...")

    distro_map = {
        "platform:el8": "rhel8",
        "15.3": "sles15sp5",
        "20.04": "ubuntu20_04",
        "22.04": "ubuntu20_04",
        "24.04": "ubuntu20_04",
    }

    binary_paths = []

    target_binary = detect_roofline(mspec)
    if target_binary["rocm_ver"] == "override":
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
                + "-"
                + mspec.gpu_series.lower()
                + "-rocm"
                + target_binary["rocm_ver"]
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


def total_xcds(archname, compute_partition):
    # check MI300 has a valid compute partition
    mi300a_archs = ["mi300a_a0", "mi300a_a1"]
    mi300x_archs = ["mi300x_a0", "mi300x_a1"]
    mi308x_archs = ["mi308x"]
    if (
        archname.lower() in mi300a_archs + mi300x_archs + mi308x_archs
        and compute_partition == "NA"
    ):
        console_error("Invalid compute partition found for {}".format(archname))
    if archname.lower() not in mi300a_archs + mi300x_archs + mi308x_archs:
        return 1
    # from the whitepaper
    # https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
    if compute_partition.lower() == "spx":
        if archname.lower() in mi300a_archs:
            return 6
        if archname.lower() in mi300x_archs:
            return 8
        if archname.lower() in mi308x_archs:
            return 4
    if compute_partition.lower() == "tpx":
        if archname.lower() in mi300a_archs:
            return 2
    if compute_partition.lower() == "dpx":
        if archname.lower() in mi300x_archs:
            return 4
        if archname.lower() in mi308x_archs:
            return 2
    if compute_partition.lower() == "qpx":
        if archname.lower() in mi300x_archs:
            return 2
    if compute_partition.lower() == "cpx":
        if archname.lower() in mi300x_archs:
            return 2
        if archname.lower() in mi308x_archs:
            return 1
    # TODO implement other archs here as needed
    console_error(
        "Unknown compute partition / arch found for {} / {}".format(
            compute_partition, archname
        )
    )


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
        console_error("profiling", "Cannot find pmc_perf.csv in %s" % path)


def print_status(msg):
    msg_length = len(msg)

    console_log("")
    console_log("~" * (msg_length + 1))
    console_log(msg)
    console_log("~" * (msg_length + 1))
    console_log("")


def set_locale_encoding():
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error as error:
        console_error(
            "Please ensure that the 'en_US.UTF-8' locale is available on your system.",
            exit=False,
        )
        console_error(error)
