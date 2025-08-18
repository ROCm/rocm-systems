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

import os
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import yaml

import config
from utils import rocpd_data, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_debug, console_error, console_log, demarcate

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd

# the build-in config to list kernel names purpose only
top_stats_build_in_config = {
    0: {
        "id": 0,
        "title": "Top Kernels",
        "data source": [{"raw_csv_table": {"id": 1, "source": "pmc_kernel_top.csv"}}],
    },
    1: {
        "id": 1,
        "title": "Dispatch List",
        "data source": [
            {"raw_csv_table": {"id": 2, "source": "pmc_dispatch_info.csv"}}
        ],
    },
}


def load_sys_info(f):
    """
    Load sys running info from csv file to a df.
    """
    return pd.read_csv(f)


def load_panel_configs(dir):
    """
    Load all panel configs from yaml file.
    """
    d = {}
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(".yaml"):
                with open(str(Path(root).joinpath(f))) as file:
                    config = yaml.safe_load(file)
                    # metric key can be None due to some metric tables
                    # not having any metrics
                    # metric key should be empty dict instead of None
                    for data_source in config["Panel Config"]["data source"]:
                        metric_table = data_source.get("metric_table")
                        if metric_table and metric_table["metric"] is None:
                            metric_table["metric"] = {}
                    d[config["Panel Config"]["id"]] = config["Panel Config"]

    # TODO: sort metrics as the header order in case they-
    # are not defined in the same order

    od = OrderedDict(sorted(d.items()))
    # for key, value in od.items():
    #     print(key, value)
    return od


def load_profiling_config(config_dir):
    """
    Load profiling config from yaml file.
    """
    try:
        with open(Path(config_dir).joinpath("profiling_config.yaml")) as file:
            prof_config = yaml.safe_load(file)
            return prof_config
    except FileNotFoundError:
        console_log(f"Could not find profiling_config.yaml in {config_dir}")
    return dict()


@demarcate
def create_df_kernel_top_stats(
    df_in,
    raw_data_dir,
    filter_gpu_ids,
    filter_dispatch_ids,
    filter_nodes,
    time_unit,
    max_stat_num,
    kernel_verbose,
    sortby="sum",
):
    """
    Create top stats info by grouping kernels with user's filters.
    """

    # NB: think about df = pd.DataFrame(df_in["pmc_perf"].copy())
    df = df_in["pmc_perf"]

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
        if ">" in filter_dispatch_ids[0]:
            m = re.match(r"\> (\d+)", filter_dispatch_ids[0])
            df = df[df["Dispatch_ID"] > int(m.group(1))]
        else:
            df = df.loc[df["Dispatch_ID"].astype(str).isin(filter_dispatch_ids)]

    # First, create a dispatches file used to populate global vars
    dispatch_info = (
        df.loc[:, ["Node", "Dispatch_ID", "Kernel_Name", "GPU_ID"]]
        if "Node" in df.columns
        else df.loc[:, ["Dispatch_ID", "Kernel_Name", "GPU_ID"]]
    )
    dispatch_info.to_csv(
        str(Path(raw_data_dir).joinpath("pmc_dispatch_info.csv")), index=False
    )

    time_stats = pd.concat(
        [df["Kernel_Name"], (df["End_Timestamp"] - df["Start_Timestamp"])],
        keys=["Kernel_Name", "ExeTime"],
        axis=1,
    )

    grouped = time_stats.groupby(by=["Kernel_Name"]).agg({
        "ExeTime": ["count", "sum", "mean", "median"]
    })

    time_unit_str = "(" + time_unit + ")"
    grouped.columns = [
        x.capitalize() + time_unit_str if x != "count" else x.capitalize()
        for x in grouped.columns.get_level_values(1)
    ]

    key = "Sum" + time_unit_str
    grouped[key] = grouped[key].div(config.TIME_UNITS[time_unit])
    key = "Mean" + time_unit_str
    grouped[key] = grouped[key].div(config.TIME_UNITS[time_unit])
    key = "Median" + time_unit_str
    grouped[key] = grouped[key].div(config.TIME_UNITS[time_unit])

    grouped = grouped.reset_index()  # Remove special group indexing

    key = "Sum" + time_unit_str
    grouped["Pct"] = grouped[key] / grouped[key].sum() * 100

    # NB:
    #   Sort by total time as default.
    if sortby == "sum":
        grouped = grouped.sort_values(by=("Sum" + time_unit_str), ascending=False)
        grouped.to_csv(
            str(Path(raw_data_dir).joinpath("pmc_kernel_top.csv")), index=False
        )
    elif sortby == "kernel":
        grouped = grouped.sort_values("Kernel_Name")
        grouped.to_csv(
            str(Path(raw_data_dir).joinpath("pmc_kernel_top.csv")), index=False
        )


@demarcate
def create_df_pmc(
    raw_data_root_dir, nodes, spatial_multiplexing, kernel_verbose, verbose, config
):
    """
    Load all raw pmc counters and join into one df.
    """

    def create_single_df_pmc(raw_data_dir, node_name, kernel_verbose, verbose):
        dfs = []
        coll_levels = []

        df = pd.DataFrame()  # noqa: F841
        new_df = pd.DataFrame()  # noqa: F841
        for root, dirs, files in os.walk(raw_data_dir):
            for f in files:
                # print("file ", f)
                if (f.endswith(".csv") and f.startswith("SQ")) or (
                    f == schema.pmc_perf_file_prefix + ".csv"
                ):
                    tmp_df = pd.read_csv(str(Path(root).joinpath(f)))
                    if config.get("format_rocprof_output") == "rocpd":
                        tmp_df = rocpd_data.process_rocpd_csv(tmp_df)
                    # Demangle original KernelNames
                    kernel_name_shortener(tmp_df, kernel_verbose)

                    # NB:
                    #   Idealy, the Node column should be added out of
                    #   multiindexing level. Here, we add it into pmc_perf
                    #   as it is the main sub-df which can be handled easily
                    #   later.
                    if f == "pmc_perf.csv" and node_name != None:
                        tmp_df.insert(0, "Node", node_name)
                    dfs.append(tmp_df)
                    coll_levels.append(f[:-4])

        # TODO: double check the case if all tmp_df.shape[0] are not on the same page
        final_df = pd.concat(dfs, keys=coll_levels, axis=1, join="inner", copy=False)
        if verbose >= 2:
            console_debug("pmc_raw_data final_single_df %s" % final_df.info)
        return final_df

    if spatial_multiplexing:
        df = pd.DataFrame()
        # todo: more err check
        for subdir in Path(raw_data_root_dir).iterdir():
            if subdir.is_dir():
                new_df = create_single_df_pmc(
                    subdir, str(subdir.name), kernel_verbose, verbose
                )
                df = pd.concat([df, new_df])
        return df

    # specified node list
    else:
        # regular single node case
        if nodes is None:
            return create_single_df_pmc(
                raw_data_root_dir, None, kernel_verbose, verbose
            )

        # "empty list" means all nodes
        elif not nodes:
            df = pd.DataFrame()
            # todo: more err check
            for subdir in Path(raw_data_root_dir).iterdir():
                if subdir.is_dir():
                    new_df = create_single_df_pmc(
                        subdir, str(subdir.name), kernel_verbose, verbose
                    )
                    df = pd.concat([df, new_df])
            return df

        # specified node list
        else:
            df = pd.DataFrame()
            # todo: more err check
            for subdir in nodes:
                p = Path(raw_data_root_dir)
                new_df = create_single_df_pmc(
                    p.joinpath(subdir), subdir, kernel_verbose, verbose
                )
                df = pd.concat([df, new_df])
            return df


def collect_wave_occu_per_cu(in_dir, out_dir, numSE):
    """
    Collect wave occupancy info from in_dir csv files
    and consolidate into out_dir/wave_occu_per_cu.csv.
    It depends highly on wave_occu_se*.csv format.
    """

    all = pd.DataFrame()

    for i in range(numSE):
        p = Path(in_dir, "wave_occu_se" + str(i) + ".csv")
        if p.exists():
            tmp_df = pd.read_csv(p)
            SE_idx = "SE" + str(tmp_df.loc[0, "SE"])
            tmp_df.rename(
                columns={
                    "Dispatch": "Dispatch",
                    "SE": "SE",
                    "CU": "CU",
                    "Occupancy": SE_idx,
                },
                inplace=True,
            )

            # TODO: join instead of concat!
            if i == 0:
                all = tmp_df[{"CU", SE_idx}]
                all.sort_index(axis=1, inplace=True)
            else:
                all = pd.concat([all, tmp_df[SE_idx]], axis=1, copy=False)

    if not all.empty:
        # print(all.transpose())
        all.to_csv(Path(out_dir, "wave_occu_per_cu.csv"), index=False)


def is_single_panel_config(root_dir, supported_archs):
    """
    Check the root configs dir structure to decide using one config set for all
    archs, or one for each arch.
    """
    # If not single config, verify all supported archs have defined configs
    supported_archs = supported_archs.keys()
    counter = 0
    for arch in supported_archs:
        if root_dir.joinpath(arch).exists():
            counter += 1
    if counter == 0:
        return True
    elif counter == len(supported_archs):
        return False
    else:
        console_error("Found multiple panel config sets but incomplete for all archs.")


def find_1st_sub_dir(directory):
    """
    Find the first sub dir in a directory
    """
    dir_path = Path(directory)
    try:
        # Iterate over entries in the directory
        for entry in dir_path.iterdir():
            if entry.is_dir():  # Check if it's a directory
                return entry
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
    return None
