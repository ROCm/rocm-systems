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

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

import config
from utils import rocpd_data, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_debug, console_error, console_log, demarcate

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd


def load_sys_info(f: str) -> pd.DataFrame:
    """
    Load sys running info from csv file to a df.
    """
    return pd.read_csv(f)


def load_panel_configs(
    dirs: list[str],
) -> OrderedDict[int, dict[str, Any]]:
    """
    Load all panel configs from yaml file.
    """
    configs: dict[int, dict[str, Any]] = {}
    for dir_path in dirs:
        for yaml_file in Path(dir_path).glob("*.yaml"):
            with open(yaml_file) as file:
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
        with open(config_path) as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        console_log(f"Could not find profiling_config.yaml in {config_dir}")
    return {}


@demarcate
def create_df_kernel_top_stats(
    df_in: dict[str, pd.DataFrame],
    raw_data_dir: str,
    filter_gpu_ids: Optional[list[str]],
    filter_dispatch_ids: Optional[list[str]],
    filter_nodes: Optional[str],
    time_unit: str,
    kernel_verbose: int,
    sortby: str = "sum",
) -> None:
    """
    Create top stats info by grouping kernels with user's filters.
    """

    df = df_in["pmc_perf"].copy()

    # Demangle original KernelNames
    kernel_name_shortener(df, kernel_verbose)

    # The logic below for filters are the same as in parser.apply_filters(),
    # which can be merged together if need it.

    if filter_nodes:
        df = df.loc[df["Node"].astype(str).isin([filter_nodes])]

    if filter_gpu_ids:
        df = df.loc[df["GPU_ID"].astype(str).isin([filter_gpu_ids])]

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
    dispatch_columns = (
        ["Node", "Dispatch_ID", "Kernel_Name", "GPU_ID"]
        if "Node" in df.columns
        else ["Dispatch_ID", "Kernel_Name", "GPU_ID"]
    )
    dispatch_info = df[dispatch_columns]
    dispatch_output_path = Path(raw_data_dir) / "pmc_dispatch_info.csv"
    dispatch_info.to_csv(dispatch_output_path, index=False)

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

    grouped = grouped.reset_index()

    # Calculate percentage
    sum_column = f"Sum{time_unit_suffix}"
    grouped["Pct"] = grouped[sum_column] / grouped[sum_column].sum() * 100

    #   Sort by total time as default.
    if sortby == "sum":
        grouped = grouped.sort_values(sum_column, ascending=False)
        grouped.to_csv(str(Path(raw_data_dir) / "pmc_kernel_top.csv"), index=False)
    elif sortby == "kernel":
        grouped = grouped.sort_values("Kernel_Name")
        grouped.to_csv(str(Path(raw_data_dir) / "pmc_kernel_top.csv"), index=False)


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
        dfs: list[pd.DataFrame] = []
        coll_levels: list[str] = []

        for csv_file in Path(raw_data_dir).rglob("*.csv"):
            file_name = csv_file.name

            is_sq_file = file_name.startswith("SQ")
            is_pmc_perf = file_name == f"{schema.PMC_PERF_FILE_PREFIX}.csv"

            if is_sq_file or is_pmc_perf:
                tmp_df = pd.read_csv(csv_file)

                if config_dict.get("format_rocprof_output") == "rocpd":
                    tmp_df = rocpd_data.process_rocpd_csv(tmp_df)

                # Demangle original KernelNames
                kernel_name_shortener(tmp_df, kernel_verbose)

                # NB:
                #   Idealy, the Node column should be added out of
                #   multiindexing level. Here, we add it into pmc_perf
                #   as it is the main sub-df which can be handled easily
                #   later.
                if file_name == "pmc_perf.csv" and node_name is not None:
                    tmp_df.insert(0, "Node", node_name)

                dfs.append(tmp_df)
                # Remove .csv extension for collection level
                coll_levels.append(csv_file.stem)

        if not dfs:
            return pd.DataFrame()

        # TODO: double check the case if all tmp_df.shape[0] are not on the same page
        final_df = pd.concat(dfs, keys=coll_levels, axis=1, join="inner", copy=False)
        if verbose >= 2:
            console_debug(f"pmc_raw_data final_single_df {final_df.info}")
        return final_df

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
    arch_names = list(supported_archs.keys())
    root_path = Path(root_dir)
    arch_count = sum(1 for arch in arch_names if (root_path / arch).exists())

    if arch_count == 0:
        return True
    elif arch_count == len(arch_names):
        return False
    else:
        console_error("Found multiple panel config sets but incomplete for all archs.")


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
