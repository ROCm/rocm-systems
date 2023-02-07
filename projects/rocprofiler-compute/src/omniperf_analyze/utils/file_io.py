################################################################################
# Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

import os
import pandas as pd
import re
import yaml

import collections
from collections import OrderedDict
from pathlib import Path
from omniperf_analyze.utils import schema

# TODO: use pandas chunksize or dask to read really large csv file
# from dask import dataframe as dd

# the build-in config to list kernel names purpose only
top_stats_build_in_config = {
    0: {
        "id": 0,
        "title": "Top Stat",
        "data source": [{"raw_csv_table": {"id": 1, "source": "pmc_kernel_top.csv"}}],
    }
}

supported_arch = {"gfx906": "mi50", "gfx908": "mi100", "gfx90a": "mi200"}
# TODO:
# it should be:
# supported_arch = {"gfx906": ["mi50", "mi60"],
#                   "gfx908": ["mi100"],
#                   "gfx90a": ["mi210", "mi250", "mi250x"]}

time_units = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}


def load_sys_info(f):
    """
    Load sys running info from csv file to a df.
    """
    return pd.read_csv(f)


def load_soc_params(dir):
    """
    Load soc params for all supported archs to a df.
    """
    df = pd.DataFrame()
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(".csv"):
                tmp_df = pd.read_csv(os.path.join(root, f))
                df = pd.concat([tmp_df, df])
    df.set_index("name", inplace=True)
    return df


def get_soc(gfx_string):
    return supported_arch[gfx_string]


def get_soc_params(df, gfx_string):
    """
    Get soc params of single arch with gfx name
    """
    return df.loc[supported_arch[gfx_string]]


def load_panel_configs(dir):
    """
    Load all panel configs from yaml file.
    """
    d = {}
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith(".yaml"):
                with open(os.path.join(root, f)) as file:
                    config = yaml.safe_load(file)
                    d[config["Panel Config"]["id"]] = config["Panel Config"]

    # TODO: sort metrics as the header order in case they are not defined in the same order

    od = OrderedDict(sorted(d.items()))
    # for key, value in od.items():
    #     print(key, value)
    return od


def create_df_kernel_top_stats(
    raw_data_dir,
    filter_gpu_ids,
    filter_dispatch_ids,
    time_unit,
    num_results,
    sortby="sum",
):
    """
    Create top stats info by grouping kernels with user's filters.
    """
    # NB:
    #   We even don't have to create pmc_kernel_top.csv explictly
    df = pd.read_csv(os.path.join(raw_data_dir, schema.pmc_perf_file_prefix + ".csv"))

    # The logic below for filters are the same as in parser.apply_filters(),
    # which can be merged together if need it.
    if filter_gpu_ids:
        df = df.loc[df["gpu-id"].astype(str).isin([filter_gpu_ids])]

    if filter_dispatch_ids:
        # NB: support ignoring the 1st n dispatched execution by '> n'
        #     The better way may be parsing python slice string
        if ">" in filter_dispatch_ids[0]:
            m = re.match("\> (\d+)", filter_dispatch_ids[0])
            df = df[df["Index"] > int(m.group(1))]
        else:
            df = df.loc[df["Index"].astype(str).isin(filter_dispatch_ids)]

    # First, create a dispatches file used to populate global vars
    dispatch_info = df.loc[:, ["Index", "KernelName", "gpu-id"]]
    dispatch_info.to_csv(os.path.join(raw_data_dir, "pmc_dispatch_info.csv"), index=False)

    time_stats = pd.concat(
        [df["KernelName"], (df["EndNs"] - df["BeginNs"])],
        keys=["KernelName", "ExeTime"],
        axis=1,
    )

    grouped = time_stats.groupby(by=["KernelName"]).agg(
        {"ExeTime": ["count", "sum", "mean", "median"]}
    )

    time_unit_str = "(" + time_unit + ")"
    grouped.columns = [
        x.capitalize() + time_unit_str if x != "count" else x.capitalize()
        for x in grouped.columns.get_level_values(1)
    ]

    key = "Sum" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])
    key = "Mean" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])
    key = "Median" + time_unit_str
    grouped[key] = grouped[key].div(time_units[time_unit])

    grouped = grouped.reset_index()  # Remove special group indexing

    key = "Sum" + time_unit_str
    grouped["Pct"] = grouped[key] / grouped[key].sum() * 100

    # NB:
    #   Sort by total time as default.
    if sortby == "sum":
        grouped = grouped.sort_values(by=("Sum" + time_unit_str), ascending=False)

        grouped = grouped.head(num_results)  # Display only the top n results

        grouped.to_csv(os.path.join(raw_data_dir, "pmc_kernel_top.csv"), index=False)
    elif sortby == "kernel":
        grouped = grouped.sort_values("KernelName")

        grouped = grouped.head(num_results)  # Display only the top n results
        grouped.to_csv(os.path.join(raw_data_dir, "pmc_kernel_top.csv"), index=False)


def create_df_pmc(raw_data_dir, verbose):
    """
    Load all raw pmc counters and join into one df.
    """
    dfs = []
    coll_levels = []

    df = pd.DataFrame()
    new_df = pd.DataFrame()
    for root, dirs, files in os.walk(raw_data_dir):
        for f in files:
            # print("file ", f)
            if (f.endswith(".csv") and f.startswith("SQ")) or (
                f == schema.pmc_perf_file_prefix + ".csv"
            ):
                tmp_df = pd.read_csv(os.path.join(root, f))
                dfs.append(tmp_df)
                coll_levels.append(f[:-4])
    final_df = pd.concat(dfs, keys=coll_levels, axis=1, copy=False)
    # TODO: join instead of concat!
    if verbose >= 2:
        print("pmc_raw_data final_df ", final_df.info())
    return final_df


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


def is_single_panel_config(root_dir):
    """
    Check the root configs dir structure to decide using one config set for all
    archs, or one for each arch.
    """
    ret = True
    counter = 0
    for arch in supported_arch.keys():
        if root_dir.joinpath(arch).exists():
            counter += 1

    if counter == 0:
        return True
    elif counter == len(supported_arch.keys()):
        return False
    else:
        raise Exception("Found multiple panel config sets but incomplete for all archs!")
