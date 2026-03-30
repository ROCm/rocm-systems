# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import glob
import importlib
import logging
import os
import pkgutil
import re
import shlex
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Union, cast

import yaml

import config
import utils.utils_profile_csv as csv_ops
from utils import rocpd_data
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils_common import (
    capture_subprocess_output,
    get_rocprof_cmd,
    parse_text,
    perform_attach_detach,
)


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
    if get_rocprof_cmd() == "rocprofiler-sdk":
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

    if get_rocprof_cmd() == "rocprofiler-sdk":
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
        console_debug(f"rocprof command: {shlex.join([get_rocprof_cmd()] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [get_rocprof_cmd()] + options, new_env=new_env, profileMode=True
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
            get_rocprof_cmd() == "rocprofiler-sdk"
            and options["ROCPROF_COUNTER_COLLECTION"] == "0"
        ):
            for db_name in glob.glob(workload_dir + "/out/pmc_1/*/*.db"):
                pid = Path(db_name).stem.split("_")[0]
                # Read CSV as list of dicts instead of pandas DataFrame
                counter_rows, _ = csv_ops.read_csv_as_dicts(
                    f"{workload_dir}/out/pmc_1/{pid}_native_counter_collection.csv"
                )
                rocpd_data.update_rocpd_pmc_events(
                    counter_rows,
                    db_name,
                )
                console_debug(f"Updated rocpd db {db_name} with native tool counters.")
        # Write results_fbase.csv
        rocpd_data.convert_dbs_to_csv(
            glob.glob(workload_dir + "/out/pmc_1/*/*.db"),
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv",
            workload_dir + f"/out/pmc_1/{fbase}_marker_api_trace.csv",
        )
        # Read CSV as list of dicts
        combined_rows, _ = csv_ops.read_csv_as_dicts(
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv"
        )
        # Reset Dispatch_ID based on PID, Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup, Start_Timestamp, End_Timestamp
        csv_ops.assign_group_ids(
            combined_rows,
            [
                "PID",
                "Kernel_Name",
                "Grid_Size",
                "Workgroup_Size",
                "LDS_Per_Workgroup",
                "Start_Timestamp",
                "End_Timestamp",
            ],
            "Dispatch_ID",
        )
        # Reset Kernel_ID based on Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup
        csv_ops.assign_group_ids(
            combined_rows,
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            "Kernel_ID",
        )
        # Drop PID since its not required
        csv_ops.drop_column_from_rows(combined_rows, "PID")
        # Write back to CSV
        csv_ops.write_csv_from_dicts(
            workload_dir + f"/out/pmc_1/{fbase}_counter_collection.csv", combined_rows
        )
        csv_ops.write_csv_from_dicts(
            workload_dir + f"/results_{fbase}.csv", combined_rows
        )
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
        if get_rocprof_cmd() == "rocprofiler-sdk":
            # rocprofv3 requires additional processing for each process
            results_files = process_rocprofv3_output(
                workload_dir,
                # counter data collected using native tool
                using_native_tool=options["ROCPROF_COUNTER_COLLECTION"] == "0",
            )
            # TODO: as rocprofv3 --kokkos-trace feature improves,
            # rocprof-compute should make updates accordingly
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
        # Add torch operator trace processing
        if torch_trace_enabled:
            # move counter collection and marker trace to workload dir
            save_torch_trace_inputs(workload_dir, fbase, format_rocprof_output)
        # Combine results into single CSV file
        if results_files:
            combined_results = csv_ops.concat_csv_files(results_files)
        else:
            console_warning(
                f"Cannot write results for {fbase}.csv due to no counter "
                "csv files generated."
            )
            return

        # Overwrite column to ensure unique IDs.
        csv_ops.add_column_to_rows(
            combined_results, "Dispatch_ID", list(range(0, len(combined_results)))
        )

        # Reset Kernel_ID based on Kernel_Name, Grid_Size,
        # Workgroup_Size, LDS_Per_Workgroup
        csv_ops.assign_group_ids(
            combined_results,
            ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
            "Kernel_ID",
        )

        csv_ops.write_csv_from_dicts(
            workload_dir + "/out/pmc_1/results_" + fbase + ".csv", combined_results
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
        rows, _ = csv_ops.read_csv_as_dicts(str(csv_path))
        csv_ops.rename_columns(rows, output_headers)
        csv_ops.write_csv_from_dicts(str(csv_path), rows)
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

    if get_rocprof_cmd() == "rocprofiler-sdk":
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

        console_debug(f"rocprof command: {shlex.join([get_rocprof_cmd()] + options)}")
        # profile the app
        success, output = capture_subprocess_output(
            [get_rocprof_cmd()] + options, new_env=os.environ.copy(), profileMode=True
        )

    if not success:
        console_error("PC sampling failed.")


@demarcate
def gen_sysinfo(
    workload_dir: str,
    app_cmd: str,
    skip_roof: bool,
    mspec: Any,  # noqa: ANN401
    soc: Any,  # noqa: ANN401
) -> None:
    df = mspec.get_class_members()

    # Append workload information to machine specs
    df["command"] = app_cmd
    df["workload_path"] = workload_dir

    blocks = ["SQ", "LDS", "SQC", "TA", "TD", "TCP", "TCC", "SPI", "CPC", "CPF"]
    if not skip_roof:
        blocks.append("roofline")
    df["ip_blocks"] = "|".join(blocks)

    df.to_csv(workload_dir + "/" + "sysinfo.csv", index=False)


def get_submodules(package_name: str) -> list[str]:
    """List all submodules for a target package"""

    submodules: list[str] = []

    # walk all submodules in target package
    package = importlib.import_module(package_name)
    for _, name, _ in pkgutil.walk_packages(package.__path__):
        pretty_name = name.split("_", 1)[1].replace("_", "")
        # ignore base submodule, add all other
        if pretty_name != "base":
            submodules.append(pretty_name)

    return submodules


def v3_counter_csv_to_v2_csv(
    counter_file: str, agent_info_filepath: str, converted_csv_file: str
) -> None:
    """
    Convert the counter file of csv output for a certain csv from rocprofv3 format
    to rocprfv2 format.
    This function is not for use of other csv out file such as kernel trace file.
    """
    counter_collections, _ = csv_ops.read_csv_as_dicts(counter_file)
    agent_info, _ = csv_ops.read_csv_as_dicts(agent_info_filepath)

    # For backwards compatability. Older rocprof versions do not provide this.
    if counter_collections and "Accum_VGPR_Count" not in counter_collections[0]:
        csv_ops.add_column_to_rows(
            counter_collections, "Accum_VGPR_Count", [0] * len(counter_collections)
        )

    result = csv_ops.pivot_table(
        counter_collections,
        index_columns=[
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
        pivot_column="Counter_Name",
        value_column="Counter_Value",
    )

    # NB: Agent_Id is int in older rocporfv3, now switched to string with prefix
    # "Agent ". We need to make sure handle both cases.
    if result and isinstance(result[0].get("Agent_Id"), str):
        console_debug("Agent ID is string type, converting to int")
        # Apply the function to the 'Agent_Id' column to extract numeric
        # part but keep it str to align with csv data
        try:
            for row in result:
                agent_id_str = row.get("Agent_Id", "")
                if isinstance(agent_id_str, str) and "Agent " in agent_id_str:
                    match = re.search(r"Agent (\d+)", agent_id_str)
                    if match:
                        row["Agent_Id"] = match.group(1)
        except Exception as e:
            console_error(
                "v3_counter_csv_to_v2_csv",
                f'Error getting "Agent_Id": {e}',
            )
    else:
        console_debug("Agent ID is already numeric type")

    # Grab the Wave_Front_Size column from agent info
    # Extract only needed columns from agent_info
    agent_info_subset = [
        {"Node_Id": row.get("Node_Id"), "Wave_Front_Size": row.get("Wave_Front_Size")}
        for row in agent_info
    ]
    result = csv_ops.merge_rows(
        result,
        agent_info_subset,
        left_on="Agent_Id",
        right_on="Node_Id",
        how="left",
    )

    # Create GPU ID mapping from agent info
    gpu_agents = [row for row in agent_info if row.get("Agent_Type") == "GPU"]
    gpu_id_map = {row.get("Node_Id"): idx for idx, row in enumerate(gpu_agents)}

    # Map Agent_Id to GPU_ID
    for row in result:
        agent_id = row.get("Agent_Id")
        if agent_id in gpu_id_map:
            row["Agent_Id"] = gpu_id_map[agent_id]

    # Drop the temporary Node_Id column
    csv_ops.drop_column_from_rows(result, "Node_Id")

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
    csv_ops.rename_columns(result, name_mapping)

    # Column reordering: extract fieldnames and reorder
    preferred_order = [
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

    # Get all columns from first row if result is not empty
    if result:
        all_columns = list(result[0].keys())
        remaining_columns = [col for col in all_columns if col not in preferred_order]
        ordered_fieldnames = preferred_order + remaining_columns
    else:
        ordered_fieldnames = preferred_order

    # Rename accumulate counters to standard format
    accum_mapping = {}
    if result:
        for col in result[0].keys():
            if col.endswith("_ACCUM"):
                accum_mapping[col] = "SQ_ACCUM_PREV_HIRES"

    if accum_mapping:
        csv_ops.rename_columns(result, accum_mapping)
        # Update fieldnames after rename
        if result:
            ordered_fieldnames = [
                accum_mapping.get(col, col) for col in ordered_fieldnames
            ]

    csv_ops.write_csv_from_dicts(
        converted_csv_file, result, fieldnames=ordered_fieldnames
    )


def convert_native_counter_collection_csv(workload_dir: str) -> None:
    """
    Use native counter collection csv and rocprofiler-sdk kernel
    trace to write counter collection csv in rocprofiler-sdk format
    for further processing to pmc_perf.csv file
    """
    for native_filename in glob.glob(
        f"{workload_dir}/out/pmc_1/*_native_counter_collection.csv"
    ):
        counter_data, _ = csv_ops.read_csv_as_dicts(native_filename)
        # Group by on dispatch_id and counter_id and sum the counter_value,
        # Other rows in group have the same value, so take the first one
        groupby_cols = ["dispatch_id", "counter_name"]
        if counter_data:
            agg_dict = {
                col: "first"
                for col in counter_data[0].keys()
                if col not in groupby_cols
            }
        else:
            agg_dict = {}
        # Overwrite counter_value aggregation to sum
        agg_dict["counter_value"] = "sum"
        counter_data = csv_ops.groupby_aggregate(counter_data, groupby_cols, agg_dict)

        pid = Path(native_filename).stem.split("_")[0]
        kernel_data_filename = glob.glob(
            f"{workload_dir}/out/pmc_1/*/{pid}_kernel_trace.csv"
        )[0]
        kernel_data, _ = csv_ops.read_csv_as_dicts(kernel_data_filename)

        # Merge counter_data with kernel_data on dispatch_id
        merged_data = csv_ops.merge_rows(
            counter_data,
            kernel_data,
            left_on="dispatch_id",
            right_on="Dispatch_Id",
            how="inner",
        )

        # Build new rows with calculated columns
        rocprofv3_counter_data = []
        for row in merged_data:
            new_row = {
                "Correlation_Id": row.get("Correlation_Id"),
                "Dispatch_Id": row.get("dispatch_id"),
                "Agent_Id": row.get("Agent_Id"),
                "Queue_Id": row.get("Queue_Id"),
                "Process_Id": row.get("Thread_Id"),
                "Thread_Id": row.get("Thread_Id"),
                "Grid_Size": (
                    int(row.get("Grid_Size_X", 1))
                    * int(row.get("Grid_Size_Y", 1))
                    * int(row.get("Grid_Size_Z", 1))
                ),
                "Kernel_Id": row.get("Kernel_Id"),
                "Kernel_Name": row.get("Kernel_Name"),
                "Workgroup_Size": (
                    int(row.get("Workgroup_Size_X", 1))
                    * int(row.get("Workgroup_Size_Y", 1))
                    * int(row.get("Workgroup_Size_Z", 1))
                ),
                "LDS_Block_Size": row.get("LDS_Block_Size"),
                "Scratch_Size": row.get("Scratch_Size"),
                "VGPR_Count": row.get("VGPR_Count"),
                "Accum_VGPR_Count": row.get("Accum_VGPR_Count"),
                "SGPR_Count": row.get("SGPR_Count"),
                "Counter_Name": row.get("counter_name"),
                "Counter_Value": row.get("counter_value"),
                "Start_Timestamp": row.get("Start_Timestamp"),
                "End_Timestamp": row.get("End_Timestamp"),
            }
            rocprofv3_counter_data.append(new_row)

        csv_ops.write_csv_from_dicts(
            kernel_data_filename.replace("kernel_trace", "counter_collection"),
            rocprofv3_counter_data,
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
            "Moved counter collection and marker trace files "
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
def process_kokkos_trace_output(workload_dir: str, fbase: str) -> None:
    # marker api trace csv files are generated for each process
    marker_api_trace_csvs = glob.glob(
        f"{workload_dir}/out/pmc_1/*/*_marker_api_trace.csv"
    )
    existing_marker_files_csv = [f for f in marker_api_trace_csvs if Path(f).is_file()]

    # concate and output marker api trace info
    combined_results = csv_ops.concat_csv_files(existing_marker_files_csv)

    csv_ops.write_csv_from_dicts(
        f"{workload_dir}/out/pmc_1/results_{fbase}_marker_api_trace.csv",
        combined_results,
    )

    if Path(f"{workload_dir}/out").exists():
        shutil.copyfile(
            f"{workload_dir}/out/pmc_1/results_{fbase}_marker_api_trace.csv",
            f"{workload_dir}/{fbase}_marker_api_trace.csv",
        )
