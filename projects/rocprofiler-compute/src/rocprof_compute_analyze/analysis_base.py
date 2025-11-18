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
import copy
import re
import sys
import textwrap
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, TextIO

import pandas as pd

import config
from rocprof_compute_soc.soc_base import OmniSoC_Base
from utils import file_io, parser, schema
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.utils import (
    get_panel_alias,
    get_uuid,
    is_workload_empty,
    merge_counters_spatial_multiplex,
)

# the build-in config to list kernel names purpose only
TOP_STATS_BUILD_IN_CONFIG: OrderedDict[int, dict[str, Any]] = OrderedDict([
    (
        0,
        {
            "id": 0,
            "title": "Top Kernels",
            "data source": [
                {"raw_csv_table": {"id": 1, "source": "pmc_kernel_top.csv"}}
            ],
        },
    ),
    (
        1,
        {
            "id": 1,
            "title": "Dispatch List",
            "data source": [
                {"raw_csv_table": {"id": 2, "source": "pmc_dispatch_info.csv"}}
            ],
        },
    ),
])


class OmniAnalyze_Base:
    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str]
    ) -> None:
        self.__args = args
        self._runs: OrderedDict[str, schema.Workload] = OrderedDict()
        self._arch_configs: dict[str, schema.ArchConfig] = {}
        self._profiling_config: dict[str, Any] = {}
        self.__supported_archs = supported_archs
        self._output: Optional[TextIO] = None
        self.__socs: Optional[dict[str, OmniSoC_Base]] = None

    def get_args(self) -> argparse.Namespace:
        return self.__args

    def set_soc(self, omni_socs: dict[str, OmniSoC_Base]) -> None:
        self.__socs = omni_socs

    def get_socs(self) -> Optional[dict[str, OmniSoC_Base]]:
        return self.__socs

    @demarcate
    def spatial_multiplex_merge_counters(self, df: pd.DataFrame) -> pd.DataFrame:
        return merge_counters_spatial_multiplex(df)

    @demarcate
    def generate_configs(
        self,
        arch: str,
        config_dir: str,
        list_stats: bool,
        filter_metrics: Optional[list[str]],
        sys_info: pd.Series,
    ) -> dict[str, schema.ArchConfig]:
        single_panel_config = file_io.is_single_panel_config(
            config_dir, self.__supported_archs
        )

        ac = schema.ArchConfig()
        if list_stats:
            ac.panel_configs = TOP_STATS_BUILD_IN_CONFIG
        else:
            arch_panel_config = [
                config_dir if single_panel_config else str(f"{config_dir}/{arch}")
            ]
            # Use restructured perf metrics in TUI analyze mode
            if self.get_args().tui and arch in ["gfx942", "gfx950"]:
                arch_panel_config.append(
                    str(
                        config.rocprof_compute_home
                        / "rocprof_compute_tui"
                        / "utils"
                        / arch
                    )
                )
            ac.panel_configs = file_io.load_panel_configs(arch_panel_config)

        # TODO: filter_metrics should/might be one per arch
        parser.build_dfs(
            arch_configs=ac, filter_metrics=filter_metrics, sys_info=sys_info
        )
        self._arch_configs[arch] = ac
        return self._arch_configs

    @demarcate
    def list_metrics(self) -> None:
        args = self.get_args()
        arch = args.list_metrics

        if arch not in self.__supported_archs:
            console_error("analysis", "Unsupported arch")
        if arch not in self._arch_configs:
            sys_info = file_io.load_sys_info(f"{args.path[0][0]}/sysinfo.csv")
            self.generate_configs(
                arch,
                args.config_dir,
                args.list_stats,
                args.filter_metrics,
                sys_info.iloc[0],
            )

        metric_descriptions = {
            k: v
            for dfs in self._arch_configs[arch].dfs.values()
            for k, v in dfs.to_dict().get("Description", {}).items()
        }
        for key, value in self._arch_configs[arch].metric_list.items():
            dot_count = str(key).count(".")
            indent = "\t" * min(dot_count, 2)

            print(f"{indent}{key} -> {value}\n")

            if dot_count > 1:
                description = metric_descriptions.get(key, "")
                if description:
                    wrapped = textwrap.wrap(description, width=40)
                    print(f"{indent}" + f"\n{indent}".join(wrapped) + "\n")

        sys.exit(0)

    @demarcate
    def list_blocks(self) -> None:
        args = self.get_args()
        arch = args.list_blocks

        if arch not in self.__supported_archs:
            console_error("analysis", "Unsupported arch")
        if arch not in self._arch_configs:
            sys_info = file_io.load_sys_info(f"{args.path[0][0]}/sysinfo.csv")
            self.generate_configs(
                arch,
                args.config_dir,
                args.list_stats,
                args.filter_metrics,
                sys_info.iloc[0],
            )

        print(f"{'INDEX':<8} {'BLOCK ALIAS':<16} {'BLOCK NAME'}")
        for key, value in self._arch_configs[arch].metric_list.items():
            panel_alias_dict = get_panel_alias()
            if key.count(".") > 0:
                continue
            print(f"{key:<8} {panel_alias_dict[value]:<16} {value}")

        sys.exit(0)

    @demarcate
    def load_options(self, normalization_filter: Optional[str]) -> None:
        args = self.get_args()
        target_filter = normalization_filter or args.normal_unit

        for arch_config in self._arch_configs.values():
            parser.build_metric_value_string(
                arch_config.dfs,
                arch_config.dfs_type,
                target_filter,
                self._profiling_config,
            )
        # Error checking for multiple runs and multiple kernel filters
        if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
            if len(args.gpu_kernel) == 1:
                args.gpu_kernel *= len(args.path)
            else:
                console_error(
                    "analysis"
                    "The number of -k/--kernel doesn't match the number of --dir."
                )

    @demarcate
    def initalize_runs(
        self, normalization_filter: Optional[str] = None
    ) -> OrderedDict[str, schema.Workload]:
        args = self.get_args()
        if args.list_metrics:
            self.list_metrics()

        if args.list_blocks:
            self.list_blocks()

        def get_sysinfo_path(data_path: str) -> Optional[str]:
            return (
                data_path
                if args.nodes is None and not args.spatial_multiplexing
                else file_io.find_1st_sub_dir(data_path)
            )

        # load required configs
        for path_info in args.path:
            sysinfo_path = get_sysinfo_path(path_info[0])
            if sysinfo_path:
                sys_info = file_io.load_sys_info(f"{sysinfo_path}/sysinfo.csv")
                arch = sys_info.iloc[0]["gpu_arch"]
                self.generate_configs(
                    arch,
                    args.config_dir,
                    args.list_stats,
                    args.filter_metrics,
                    sys_info.iloc[0],
                )

        self.load_options(normalization_filter)

        for path_info in args.path:
            # FIXME:
            #    For regular single node case, load sysinfo.csv directly
            #    For multi-node, either the default "all", or specified some,
            #    pick up the one in the 1st sub_dir. We could fix it properly later.
            w = schema.Workload()
            sysinfo_path = get_sysinfo_path(path_info[0])
            if sysinfo_path:
                w.sys_info = file_io.load_sys_info(f"{sysinfo_path}/sysinfo.csv")
                if not getattr(args, "no_roof", False):
                    try:
                        roofline_df = pd.read_csv(f"{sysinfo_path}/roofline.csv")
                        w.roofline_peaks = roofline_df
                    except FileNotFoundError:
                        console_warning("roofline.csv not found.")
                        w.roofline_peaks = pd.DataFrame()
                else:
                    w.roofline_peaks = pd.DataFrame()

                arch = w.sys_info.iloc[0]["gpu_arch"]
                socs = self.get_socs()
                if socs and arch in socs:
                    mspec = socs[arch]._mspec
                    if args.specs_correction:
                        w.sys_info = parser.correct_sys_info(
                            mspec, args.specs_correction
                        )
                w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
                w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
                w.dfs_type = self._arch_configs[arch].dfs_type
                self._runs[path_info[0]] = w

        return self._runs

    @demarcate
    def sanitize(self) -> None:
        """Perform sanitization of inputs"""
        args = self.get_args()
        if args.tui:
            return

        if not args.path:
            console_error("The following arguments are required: -p/--path")

        # verify not accessing parent directories
        if ".." in str(args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
            )

        # ensure absolute path
        seen_paths: set[str] = set()
        for dir_info in args.path:
            full_path = Path(dir_info[0]).absolute().resolve()
            dir_info[0] = str(full_path)

            if not full_path.is_dir():
                console_error(
                    "analysis", f"Invalid directory {full_path}\nPlease try again."
                )
            # validate profiling data

            if dir_info[0] in seen_paths:
                console_error("analysis", "You cannot provide the same path twice.")
            seen_paths.add(dir_info[0])

            if not any([
                args.nodes,
                args.list_nodes,
                args.spatial_multiplexing,
            ]):
                is_workload_empty(dir_info[0])

        # FIXME:
        #   The proper location of this func should be in pre_processing().
        #   However, because of reading soc depends on sys spec, and sys
        #   spec depends on sys_info. And we read sys_info too early so we
        # . can not do it now. There should be a way to make it simpler.
        if args.list_nodes:
            # NB:
            #   There are 2 ways to do it: one is doing like the below, checking
            #   sub dirs only as we assume the profiling stage generate sub dirs
            #   with node name. The 2nd way would be checkign host name in each
            #   sub dir and very those.
            nodes = [
                subdir.name
                for subdir in Path(args.path[0][0]).iterdir()
                if subdir.is_dir()
            ]
            print("Node list:", "  ".join(nodes))
            sys.exit(0)

        # Ensure analysis output does not overwrite existing files
        if not args.output_name:
            return

        if not re.match(r"^[A-Za-z0-9_-]+$", args.output_name):
            console_error(
                "analysis",
                "Analysis output file/folder name must "
                "contain only alphanumeric characters "
                "or underscores (_), hyphens (-).",
            )

        path_to_check = args.output_name
        if args.output_format in ("txt", "db"):
            path_to_check += f".{args.output_format}"

        if Path(path_to_check).exists():
            console_error(
                f"Analysis output file/folder {path_to_check} already exists. "
                "Please choose a different name."
            )

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self) -> None:
        """Perform initialization prior to analysis."""
        console_debug("analysis", "prepping to do some analysis")
        console_log("analysis", "deriving rocprofiler-compute metrics...")
        args = self.get_args()

        # initalize output file
        if args.output_format == "txt":
            output_filename = args.output_name or f"rocprof_compute_{get_uuid()}"
            output_filename += ".txt"
            self._output = open(output_filename, "w+")
            console_warning("analysis", f"Created file: {output_filename}")
        elif args.output_format == "stdout":
            self._output = sys.stdout

        # Read profiling config
        self._profiling_config = file_io.load_profiling_config(args.path[0][0])

        # initalize runs
        self._runs = self.initalize_runs()

        # set filters
        filter_configs = [
            (args.gpu_kernel, "filter_kernel_ids"),
            (args.gpu_id, "filter_gpu_ids"),
            (args.gpu_dispatch_id, "filter_dispatch_ids"),
            (args.nodes, "nodes"),
        ]

        for filter_list, attr_name in filter_configs:
            if not filter_list:
                continue

            # Extend single filter to match all paths
            if len(filter_list) == 1 and len(args.path) > 1:
                filter_list *= len(args.path)

            # Apply filters to workloads
            for path_info, filter_value in zip(args.path, filter_list):
                setattr(self._runs[path_info[0]], attr_name, filter_value)

    @abstractmethod
    def run_analysis(self) -> None:
        """Run analysis."""
        console_debug("analysis", "generating analysis")
