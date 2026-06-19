# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

import config
from utils import schema, utils_analysis
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils_common import canonical_config_arch, normalize_filter_to_str_list

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd


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


def load_profiling_config(config_dir: str) -> dict[str, Any]:
    """
    Load profiling config from yaml file.
    """
    config_path = Path(config_dir) / "profiling_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        console_log(f"Could not find profiling_config.yaml in {config_dir}")
    return {}


@demarcate
def create_df_kernel_top_stats(
    df_in: pd.DataFrame,
    raw_data_dir: str,
    filter_gpu_ids: Optional[list[str]],
    filter_dispatch_ids: Optional[list[str]],
    filter_nodes: Optional[str],
    time_unit: str,
    kernel_verbose: int,
    sortby: str = "sum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create top stats info by grouping kernels with user's filters.

    Returns:
        A tuple of (kernel_top_df, dispatch_info_df).
    """

    df = df_in.copy()

    # The logic below for filters are the same as in parser.apply_filters(),
    # which can be merged together if need it.

    if filter_nodes:
        df = df.loc[
            df["Node"].astype(str).isin(normalize_filter_to_str_list(filter_nodes))
        ]

    if filter_gpu_ids:
        df = df.loc[
            df["GPU_ID"].astype(str).isin(normalize_filter_to_str_list(filter_gpu_ids))
        ]

    if filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        first_filter = filter_dispatch_ids[0]

        if isinstance(first_filter, str) and first_filter.startswith(">"):
            match = re.match(r">\s*(\d+)", str(first_filter))
            if match:
                threshold = int(match.group(1))
                df = df[df["Dispatch_ID"] > threshold]
        else:
            filter_strings = [str(f) for f in filter_dispatch_ids]
            df = df.loc[df["Dispatch_ID"].astype(str).isin(filter_strings)]

    # First, create a dispatches file used to populate global vars
    dispatch_columns = ["Kernel_Name", "GPU_ID"]
    if "Dispatch_ID" in df.columns:
        dispatch_columns.insert(0, "Dispatch_ID")
    if "Node" in df.columns:
        dispatch_columns.insert(0, "Node")

    dispatch_info = df[dispatch_columns]
    dispatch_output_path = Path(raw_data_dir) / "pmc_dispatch_info.csv"
    dispatch_info.to_csv(dispatch_output_path, index=False)

    if "Dispatch_ID" in df.columns:
        # Calculate execution times
        execution_times = df["End_Timestamp"] - df["Start_Timestamp"]
        time_stats = pd.DataFrame({
            "Kernel_Name": df["Kernel_Name"],
            "ExeTime": execution_times,
        })

        grouped = time_stats.groupby("Kernel_Name")["ExeTime"].agg([
            "count",
            "sum",
            "mean",
            "median",
        ])
    else:
        time_stats = pd.DataFrame({
            "Kernel_Name": df["Kernel_Name"],
            "count": df["Count"],
            "sum": df["Mean_Time"] * df["Count"],
            "mean": df["Mean_Time"],
            "median": df["Median_Time"],
        })

        result_data: list[dict[str, Any]] = []
        for _, group in time_stats.groupby("Kernel_Name"):
            row: dict[str, Any] = {}

            row["Kernel_Name"] = group["Kernel_Name"].iloc[0]
            row["count"] = group["count"].sum()
            row["sum"] = group["sum"].sum()
            row["mean"] = row["sum"] / row["count"]

            sorted_data_by_mean = group.sort_values("mean")
            sorted_data_by_mean["count_cumsum"] = sorted_data_by_mean["count"].cumsum()
            median_threshold = row["count"] / 2
            median_value = sorted_data_by_mean.loc[
                sorted_data_by_mean["count_cumsum"] >= median_threshold, "median"
            ].iloc[0]
            row["median"] = median_value

            result_data.append(row)

        grouped = pd.DataFrame(result_data)

    # Rename columns with time unit
    time_unit_suffix = f"({time_unit})"
    column_mapping = {
        "count": "Count",
        "sum": f"Sum{time_unit_suffix}",
        "mean": f"Mean{time_unit_suffix}",
        "median": f"Median{time_unit_suffix}",
    }
    grouped = grouped.rename(columns=column_mapping)

    # Convert time units
    time_divisor = config.TIME_UNITS[time_unit]
    for col in [
        f"Sum{time_unit_suffix}",
        f"Mean{time_unit_suffix}",
        f"Median{time_unit_suffix}",
    ]:
        grouped[col] = grouped[col] / time_divisor

    if "Dispatch_ID" in df.columns:
        grouped = grouped.reset_index()

    # Calculate percent
    sum_column = f"Sum{time_unit_suffix}"
    grouped["Percent"] = grouped[sum_column] / grouped[sum_column].sum() * 100

    #   Sort by total time as default.
    if sortby == "sum":
        grouped = grouped.sort_values(sum_column, ascending=False)
        grouped.to_csv(str(Path(raw_data_dir) / "pmc_kernel_top.csv"), index=False)
    elif sortby == "kernel":
        grouped = grouped.sort_values("Kernel_Name")
        grouped.to_csv(str(Path(raw_data_dir) / "pmc_kernel_top.csv"), index=False)

    return grouped.reset_index(drop=True), dispatch_info.reset_index(drop=True)


def build_agent_to_gpu_map_from_json(
    agents: list[dict[str, Any]],
) -> dict[int, int]:
    """
    Map agent ``id.handle`` values to 0-indexed GPU IDs.

    GPU agents are identified by the rocprofiler-sdk agent ``type`` enum
    value 2 in the ``agents`` array of ``ps_file_results.json``.  They are
    sorted by ``node_id`` so that the first GPU agent maps to GPU 0,
    the second to GPU 1, etc.
    """
    rocprofiler_agent_type_gpu = 2
    gpu_agents = sorted(
        (agent for agent in agents if agent.get("type") == rocprofiler_agent_type_gpu),
        key=lambda agent: agent["node_id"],
    )
    return {agent["id"]["handle"]: index for index, agent in enumerate(gpu_agents)}


@demarcate
def load_pc_sampling_results(workload_path: str) -> Optional[dict[str, Any]]:
    """
    Parse ``ps_file_results.json`` and return its ``rocprofiler-sdk-tool[0]``
    record. Returns ``None`` if the file is absent or fails to parse (a
    warning is logged in the latter case).

    The json can be multiple GB: parse once here and pass the dict to every
    PC sampling consumer instead of re-reading the file.
    """
    json_path = Path(workload_path) / "ps_file_results.json"
    if not json_path.exists():
        return None
    try:
        with json_path.open(encoding="utf-8") as json_file:
            return json.load(json_file)["rocprofiler-sdk-tool"][0]
    except (json.JSONDecodeError, KeyError, IndexError) as error:
        console_warning(f"PC sampling: failed to parse {json_path}: {error}")
        return None


def process_pc_sampling_kernel_trace(
    tool_data: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build kernel and dispatch info from the kernel dispatch records.

    Used for PC-sampling-only runs where ``pmc_perf`` data is not
    available.  Consumes a parsed ``rocprofiler-sdk-tool[0]`` dict
    (see ``load_pc_sampling_results``): kernel dispatch buffer records for
    timestamps and dispatch info, ``kernel_symbols`` for kernel names, and
    ``agents`` for the GPU ID mapping.  Returns an empty frame when
    *tool_data* is ``None`` (results json absent).
    """
    columns = [
        "Dispatch_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]
    if tool_data is None:
        console_warning("PC sampling results not found. Cannot build dispatch data.")
        return pd.DataFrame(columns=columns)

    dispatches = tool_data["buffer_records"]["kernel_dispatch"]
    kernel_id_to_name = {
        symbol["kernel_id"]: symbol["formatted_kernel_name"]
        for symbol in tool_data["kernel_symbols"]
    }
    agent_to_gpu = build_agent_to_gpu_map_from_json(tool_data["agents"])

    rows = [
        {
            "Dispatch_Id": dispatch["dispatch_info"]["dispatch_id"],
            "Kernel_Name": kernel_id_to_name.get(
                dispatch["dispatch_info"]["kernel_id"]
            ),
            "Start_Timestamp": dispatch["start_timestamp"],
            "End_Timestamp": dispatch["end_timestamp"],
            "GPU_ID": agent_to_gpu.get(
                dispatch["dispatch_info"]["agent_id"]["handle"], 0
            ),
        }
        for dispatch in dispatches
    ]

    return pd.DataFrame(rows, columns=columns)


@demarcate
def create_df_pmc(
    raw_data_root_dir: str,
    nodes: Optional[list[str]],
    spatial_multiplexing: bool,
    kernel_verbose: int,
    verbose: int,
    config_dict: dict[str, Any],
) -> pd.DataFrame:
    """
    Load all raw pmc counters and join into one df.
    """

    def create_single_df_pmc(
        raw_data_dir: str, node_name: Optional[str], kernel_verbose: int, verbose: int
    ) -> pd.DataFrame:
        pmc_perf_path = Path(raw_data_dir) / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
        if not pmc_perf_path.is_file():
            return pd.DataFrame()

        df = pd.read_csv(pmc_perf_path)

        if config_dict.get("format_rocprof_output") == "rocpd":
            df = utils_analysis.process_rocpd_csv(df)

        # Demangle original KernelNames
        # Skip for Standalone Roofline with -1 to keep full kernel names
        if kernel_verbose >= 0:
            kernel_name_shortener(df, kernel_verbose)

        if node_name is not None:
            df.insert(0, "Node", node_name)

        if verbose >= 2:
            console_debug(f"pmc_raw_data final_single_df {df.info}")
        return df

    root_path = Path(raw_data_root_dir)

    # 1. spatial multiplexing case
    if spatial_multiplexing:
        dfs: list[pd.DataFrame] = []

        for subdir in root_path.iterdir():
            if subdir.is_dir():
                new_df = create_single_df_pmc(
                    str(subdir), str(subdir.name), kernel_verbose, verbose
                )
                if not new_df.empty:
                    dfs.append(new_df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 2. regular single node case (nodes=None)
    if nodes is None:
        return create_single_df_pmc(raw_data_root_dir, None, kernel_verbose, verbose)

    # 3. all nodes case (nodes=[])
    if not nodes:
        dfs: list[pd.DataFrame] = []

        for subdir in root_path.iterdir():
            if subdir.is_dir():
                new_df = create_single_df_pmc(
                    str(subdir), str(subdir.name), kernel_verbose, verbose
                )
                if not new_df.empty:
                    dfs.append(new_df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # 4. specified node list case (nodes=[...])
    dfs: list[pd.DataFrame] = []

    for node in nodes:
        node_path = root_path / node
        if node_path.exists():
            new_df = create_single_df_pmc(str(node_path), node, kernel_verbose, verbose)
            if not new_df.empty:
                dfs.append(new_df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def collect_wave_occu_per_cu(in_dir: str, out_dir: str, num_se: int) -> None:
    """
    Collect wave occupancy info from in_dir csv files
    and consolidate into out_dir/wave_occu_per_cu.csv.
    It depends highly on wave_occu_se*.csv format.
    """
    in_path = Path(in_dir)
    all_data = pd.DataFrame()

    for i in range(num_se):
        file_path = in_path / f"wave_occu_se{i}.csv"
        if not file_path.exists():
            continue

        tmp_df = pd.read_csv(file_path)
        if tmp_df.empty:
            continue

        se_idx = f"SE{tmp_df.loc[0, 'SE']}"
        tmp_df.rename(
            columns={
                "Dispatch": "Dispatch",
                "SE": "SE",
                "CU": "CU",
                "Occupancy": se_idx,
            }
        )

        # TODO: join instead of concat!
        if i == 0:
            all_data = tmp_df[{"CU", se_idx}]
            all_data.sort_index(axis=1, inplace=True)
        else:
            all_data = pd.concat([all_data, tmp_df[se_idx]], axis=1, copy=False)

    if not all_data.empty:
        all_data.to_csv(Path(out_dir) / "wave_occu_per_cu.csv", index=False)


def is_single_panel_config(
    root_dir: str, supported_archs: dict[str, str]
) -> Optional[bool]:
    """
    Check the root configs dir structure to decide using one config set for all
    archs, or one for each arch.
    """
    # If not single config, verify all supported archs have defined configs
    arch_names = {
        canonical_config_arch(arch) or arch for arch in supported_archs.keys()
    }
    root_path = Path(root_dir)
    arch_count = sum(1 for arch in arch_names if (root_path / arch).exists())

    if arch_count == 0:
        return True
    elif arch_count == len(arch_names):
        return False
    else:
        console_warning(
            "Found multiple panel config sets but incomplete for all archs."
        )


def find_1st_sub_dir(directory: str) -> Optional[str]:
    """
    Find the first sub dir in a directory
    """
    dir_path = Path(directory)
    try:
        # Iterate over entries in the directory
        for entry in dir_path.iterdir():
            if entry.is_dir():  # Check if it's a directory
                return str(entry)
        return None
    except FileNotFoundError:
        console_error(f'The directory "{directory}" does not exist.', exit=False)


def get_valid_nodes(directory: str) -> list[str]:
    """Return subdirectory names that contain sysinfo.csv"""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return []
    return [
        entry.name
        for entry in dir_path.iterdir()
        if entry.is_dir() and (entry / "sysinfo.csv").exists()
    ]
