#!/usr/bin/env python3

##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
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

"""
    Quick run:
        analyze.py -d 1st_run_dir -d 2nd_run_dir -b 2

    Common abbreviations in the code:
        df - pandas.dataframe
        pmc - HW performance conuter
        metric - derived expression from pmc and soc spec
"""

import sys


import copy
import random
import sys
import argparse
import os.path
from pathlib import Path
from omniperf_analyze.utils import parser, file_io
from omniperf_analyze.utils.gui_components.roofline import get_roofline
from utils import csv_converter
import pandas as pd

archConfigs = {}


################################################
# Helper Functions
################################################
def generate_config(arch, config_dir, list_kernels, filter_metrics):
    from omniperf_analyze.utils import schema

    single_panel_config = file_io.is_single_panel_config(Path(config_dir))
    global archConfigs

    ac = schema.ArchConfig()
    if list_kernels:
        ac.panel_configs = file_io.top_stats_build_in_config
    else:
        arch_panel_config = (
            config_dir if single_panel_config else config_dir.joinpath(arch)
        )
        ac.panel_configs = file_io.load_panel_configs(arch_panel_config)

    # TODO: filter_metrics should/might be one per arch
    # print(ac)

    parser.build_dfs(ac, filter_metrics)

    archConfigs[arch] = ac

    return archConfigs  # Note: This return comes in handy for rocScope which borrows generate_configs() in its rocomni plugin


def list_metrics(args):
    import pandas as pd
    from tabulate import tabulate

    if args.list_metrics in file_io.supported_arch.keys():
        arch = args.list_metrics
        if arch not in archConfigs.keys():
            generate_config(arch, args.config_dir, args.list_kernels, args.filter_metrics)
        print(
            tabulate(
                pd.DataFrame.from_dict(
                    archConfigs[args.list_metrics].metric_list,
                    orient="index",
                    columns=["Metric"],
                ),
                headers="keys",
                tablefmt="fancy_grid",
            ),
            file=output,
        )
        sys.exit(0)
    else:
        print("Error: Unsupported arch")
        sys.exit(-1)


def load_options(args, normalization_filter):
    # Use original normalization or user input from GUI
    if not normalization_filter:
        for k, v in archConfigs.items():
            parser.build_metric_value_string(v.dfs, v.dfs_type, args.normal_unit)
    else:
        for k, v in archConfigs.items():
            parser.build_metric_value_string(v.dfs, v.dfs_type, normalization_filter)

    # err checking for multiple runs and multiple gpu_kernel filter
    if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
        if len(args.gpu_kernel) == 1:
            for i in range(len(args.path) - 1):
                args.gpu_kernel.extend(args.gpu_kernel)
        else:
            print(
                "Error: the number of --filter-kernels doesn't match the number of --dir.",
                file=output,
            )
            sys.exit(-1)


################################################
# Core Functions
################################################
def initialize_run(args, normalization_filter=None):
    from collections import OrderedDict
    from omniperf_analyze.utils import schema

    # Fixme: cur_root.parent.joinpath('soc_params')
    soc_params_dir = os.path.join(os.path.dirname(__file__), "..", "soc_params")
    soc_spec_df = file_io.load_soc_params(soc_params_dir)

    if args.list_metrics:
        list_metrics(args)

    # Load required configs
    for d in args.path:
        sys_info = file_io.load_sys_info(Path(d[0], "sysinfo.csv"))
        arch = sys_info.iloc[0]["gpu_soc"]
        generate_config(arch, args.config_dir, args.list_kernels, args.filter_metrics)

    load_options(args, normalization_filter)

    runs = OrderedDict()

    # Todo: warning single -d with multiple dirs
    for d in args.path:
        w = schema.Workload()
        w.sys_info = file_io.load_sys_info(Path(d[0], "sysinfo.csv"))
        w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
        arch = w.sys_info.iloc[0]["gpu_soc"]
        w.dfs = copy.deepcopy(archConfigs[arch].dfs)
        w.dfs_type = archConfigs[arch].dfs_type
        w.soc_spec = file_io.get_soc_params(soc_spec_df, arch)
        runs[d[0]] = w

    # Return rather than referencing 'runs' globally (since used outside of file scope)
    return runs


def run_gui(args, runs):
    import dash
    from omniperf_analyze.utils import gui
    import dash_bootstrap_components as dbc

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

    if len(runs) == 1:
        file_io.create_df_kernel_top_stats(
            args.path[0][0],
            runs[args.path[0][0]].filter_gpu_ids,
            runs[args.path[0][0]].filter_dispatch_ids,
            args.time_unit,
            args.max_kernel_num,
        )
        runs[args.path[0][0]].raw_pmc = file_io.create_df_pmc(
            args.path[0][0], args.verbose
        )  # create mega df
        parser.load_kernel_top(runs[args.path[0][0]], args.path[0][0])

        input_filters = {
            "kernel": runs[args.path[0][0]].filter_kernel_ids,
            "gpu": runs[args.path[0][0]].filter_gpu_ids,
            "dispatch": runs[args.path[0][0]].filter_dispatch_ids,
            "normalization": args.normal_unit,
            "top_n": args.max_kernel_num,
        }

        gui.build_layout(
            app,
            runs,
            archConfigs["gfx90a"],
            input_filters,
            args.decimal,
            args.time_unit,
            args.cols,
            str(args.path[0][0]),
            args.g,
            args.verbose,
            args,
        )
        if args.random_port:
            app.run_server(debug=False, host="0.0.0.0", port=random.randint(1024, 49151))
        else:
            app.run_server(debug=False, host="0.0.0.0", port=args.gui)
    else:
        print("Multiple runs not yet supported in GUI. Retry without --gui flag.")


def run_cli(args, runs):
    from omniperf_analyze.utils import tty

    # NB:
    # If we assume the panel layout for all archs are similar, it doesn't matter
    # which archConfig passed into show_all function.
    # After decide to how to manage kernels display patterns, we can revisit it.
    cache = dict()
    for d in args.path:
        # demangle
        for filename in os.listdir(d[0]):
            if filename.endswith(".csv"):
                df = pd.read_csv(os.path.join(d[0], filename))
                new_df = csv_converter.kernel_name_shortener(
                    df, cache, args.kernelVerbose
                )
                new_df.to_csv(os.path.join(d[0], filename), index=False)

        file_io.create_df_kernel_top_stats(
            d[0],
            runs[d[0]].filter_gpu_ids,
            runs[d[0]].filter_dispatch_ids,
            args.time_unit,
            args.max_kernel_num,
        )
        runs[d[0]].raw_pmc = file_io.create_df_pmc(
            d[0], args.verbose
        )  # creates mega dataframe
        is_gui = False
        parser.load_table_data(
            runs[d[0]], d[0], is_gui, args.g, args.verbose
        )  # create the loaded table
    # TODO: In show_* functions always assume newest architecture. This way newest configs/figures are loaded
    if args.list_kernels:
        tty.show_kernels(
            args,
            runs,
            archConfigs[runs[args.path[0][0]].sys_info.iloc[0]["gpu_soc"]],
            output,
        )
    else:
        tty.show_all(
            args,
            runs,
            archConfigs[runs[args.path[0][0]].sys_info.iloc[0]["gpu_soc"]],
            output,
        )


def roofline_only(path_to_dir, dev_id, sort_type, mem_level, kernel_names, verbose):
    import pandas as pd
    from collections import OrderedDict

    # Change vL1D to a interpretable str, if required
    if "vL1D" in mem_level:
        mem_level.remove("vL1D")
        mem_level.append("L1")

    app_path = path_to_dir + "/pmc_perf.csv"
    roofline_exists = os.path.isfile(app_path)
    if not roofline_exists:
        print("Error: {} does not exist")
        sys.exit(0)
    t_df = OrderedDict()
    t_df["pmc_perf"] = pd.read_csv(app_path)
    get_roofline(
        path_to_dir,
        t_df,
        verbose,
        dev_id,  # [Optional] Specify device id to collect roofline info from
        sort_type,  # [Optional] Sort AI by top kernels or dispatches
        mem_level,  # [Optional] Toggle particular level(s) of memory hierarchy
        kernel_names,  # [Optional] Toggle overlay of kernel names in plot
        True,  # [Optional] Generate a standalone roofline analysis
    )


def analyze(args):
    if args.dependency:
        print("pip3 install astunparse numpy tabulate pandas pyyaml")
        sys.exit(0)

    # NB: maybe create bak file for the old run before open it
    global output
    output = open(args.output_file, "w+") if args.output_file else sys.stdout

    # Initalize archConfigs and runs[]
    runs = initialize_run(args)

    # Filtering
    if args.gpu_kernel:
        for d, gk in zip(args.path, args.gpu_kernel):
            for k_idx in gk:
                if int(k_idx) >= 10:
                    print(
                        "{} is an invalid kernel filter. Must be between 0-9.".format(
                            k_idx
                        )
                    )
                    sys.exit(2)
            runs[d[0]].filter_kernel_ids = gk
    if args.gpu_id:
        if len(args.gpu_id) == 1 and len(args.path) != 1:
            for i in range(len(args.path) - 1):
                args.gpu_id.extend(args.gpu_id)
        for d, gi in zip(args.path, args.gpu_id):
            runs[d[0]].filter_gpu_ids = gi
    if args.gpu_dispatch_id:
        if len(args.gpu_dispatch_id) == 1 and len(args.path) != 1:
            for i in range(len(args.path) - 1):
                args.gpu_dispatch_id.extend(args.gpu_dispatch_id)
        for d, gd in zip(args.path, args.gpu_dispatch_id):
            runs[d[0]].filter_dispatch_ids = gd

    # Launch CLI analysis or GUI
    if args.gui:
        run_gui(args, runs)
    else:
        if args.random_port:
            print("ERROR: --gui flag required to enable --random-port")
            sys.exit(1)
        run_cli(args, runs)
