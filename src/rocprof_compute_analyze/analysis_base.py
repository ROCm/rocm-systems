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


import copy
import sys
import textwrap
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

from utils import file_io, parser, schema
from utils.logger import console_debug, console_error, console_log, demarcate
from utils.utils import is_workload_empty, merge_counters_spatial_multiplex


class OmniAnalyze_Base:
    def __init__(self, args, supported_archs):
        self.__args = args
        self._runs = OrderedDict()
        self._arch_configs = {}
        self._profiling_config = dict()
        self.__supported_archs = supported_archs
        self._output = None
        self.__socs: dict = None  # available OmniSoC objs

    def get_args(self):
        return self.__args

    def set_soc(self, omni_socs):
        self.__socs = omni_socs

    def get_socs(self):
        return self.__socs

    @demarcate
    def spatial_multiplex_merge_counters(self, df):
        return merge_counters_spatial_multiplex(df)

    @demarcate
    def generate_configs(self, arch, config_dir, list_stats, filter_metrics, sys_info):
        single_panel_config = file_io.is_single_panel_config(
            Path(config_dir), self.__supported_archs
        )

        ac = schema.ArchConfig()
        if list_stats:
            ac.panel_configs = file_io.top_stats_build_in_config
        else:
            arch_panel_config = (
                config_dir if single_panel_config else config_dir.joinpath(arch)
            )
            ac.panel_configs = file_io.load_panel_configs(arch_panel_config)

        # TODO: filter_metrics should/might be one per arch
        # print(ac)

        parser.build_dfs(
            archConfigs=ac, filter_metrics=filter_metrics, sys_info=sys_info
        )
        self._arch_configs[arch] = ac
        return self._arch_configs

    @demarcate
    def list_metrics(self):
        args = self.__args
        if args.list_metrics in self.__supported_archs.keys():
            arch = args.list_metrics
            if arch not in self._arch_configs.keys():
                sys_info = file_io.load_sys_info(
                    Path(self.__args.path[0][0], "sysinfo.csv")
                )
                self.generate_configs(
                    arch,
                    args.config_dir,
                    args.list_stats,
                    args.filter_metrics,
                    sys_info.iloc[0],
                )

            metric_descriptions = {
                k: v
                for dfs in self._arch_configs[args.list_metrics].dfs.values()
                for k, v in dfs.to_dict().get("Description", {}).items()
            }
            for key, value in self._arch_configs[args.list_metrics].metric_list.items():
                prefix = ""
                description = ""
                if "." not in str(key):
                    prefix = ""
                elif str(key).count(".") == 1:
                    prefix = "\t"
                else:
                    prefix = "\t\t"
                    description = metric_descriptions.get(key, "")
                print(prefix + key, "->", value + "\n")
                if description:
                    print(
                        prefix
                        + f"\n{prefix}".join(textwrap.wrap(description, width=40))
                        + "\n"
                    )
            sys.exit(0)
        else:
            console_error("Unsupported arch")

    @demarcate
    def load_options(self, normalization_filter):
        if not normalization_filter:
            for k, v in self._arch_configs.items():
                parser.build_metric_value_string(
                    v.dfs, v.dfs_type, self.__args.normal_unit, self._profiling_config
                )
        else:
            for k, v in self._arch_configs.items():
                parser.build_metric_value_string(
                    v.dfs, v.dfs_type, normalization_filter, self._profiling_config
                )

        args = self.__args
        # Error checking for multiple runs and multiple kernel filters
        if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
            if len(args.gpu_kernel) == 1:
                for i in range(len(args.path) - 1):
                    args.gpu_kernel.extend(args.gpu_kernel)
            else:
                console_error(
                    "analysis"
                    "The number of -k/--kernel doesn't match the number of --dir."
                )

    @demarcate
    def initalize_runs(self, normalization_filter=None):
        if self.__args.list_metrics:
            self.list_metrics()

        # load required configs
        for d in self.__args.path:
            sysinfo_path = (
                Path(d[0])
                if self.__args.nodes is None
                and self.__args.spatial_multiplexing is not True
                else file_io.find_1st_sub_dir(d[0])
            )
            sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
            arch = sys_info.iloc[0]["gpu_arch"]
            args = self.__args
            self.generate_configs(
                arch,
                args.config_dir,
                args.list_stats,
                args.filter_metrics,
                sys_info.iloc[0],
            )

        self.load_options(normalization_filter)

        for d in self.__args.path:
            w = schema.Workload()
            # FIXME:
            #    For regular single node case, load sysinfo.csv directly
            #    For multi-node, either the default "all", or specified some,
            #    pick up the one in the 1st sub_dir. We could fix it properly later.
            sysinfo_path = (
                Path(d[0])
                if self.__args.nodes is None
                and self.__args.spatial_multiplexing is not True
                else file_io.find_1st_sub_dir(d[0])
            )
            w.sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
            arch = w.sys_info.iloc[0]["gpu_arch"]
            mspec = self.get_socs()[arch]._mspec
            if self.__args.specs_correction:
                w.sys_info = parser.correct_sys_info(
                    mspec, self.__args.specs_correction
                )
            w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
            w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
            w.dfs_type = self._arch_configs[arch].dfs_type
            self._runs[d[0]] = w

        return self._runs

    @demarcate
    def sanitize(self):
        """Perform sanitization of inputs"""
        if self.__args.tui:
            return
        if not self.__args.path:
            console_error("The following arguments are required: -p/--path")
        # verify not accessing parent directories
        if ".." in str(self.__args.path):
            console_error(
                "Access denied. Cannot access parent directories in path (i.e. ../)"
            )
        # ensure absolute path
        for dir in self.__args.path:
            full_path = str(Path(dir[0]).absolute().resolve())
            dir[0] = full_path
            if not Path(dir[0]).is_dir():
                console_error("Invalid directory {}\nPlease try again.".format(dir[0]))
            # validate profiling data

            # Todo: more err check
            if not (
                self.__args.nodes != None
                or self.__args.list_nodes
                or self.__args.spatial_multiplexing
            ):
                is_workload_empty(dir[0])
            # else:

        # no using same paths
        occurances = set()
        for dir in self.__args.path:
            dir = dir[0]
            if dir in occurances:
                console_error("You cannot provide the same path twice.")
            else:
                occurances.add(dir)

        # FIXME:
        #   The proper location of this func should be in pre_processing().
        #   However, because of reading soc depends on sys spec, and sys
        #   spec depends on sys_info. And we read sys_info too early so we
        # . can not do it now. There should be a way to make it simpler.
        if self.__args.list_nodes:
            nodes = []
            # NB:
            #   There are 2 ways to do it: one is doing like the below, checking
            #   sub dirs only as we assume the profiling stage generate sub dirs
            #   with node name. The 2nd way would be checkign host name in each
            #   sub dir and very those.
            for subdir in Path(self.__args.path[0][0]).iterdir():
                if subdir.is_dir():
                    nodes.append(str(subdir.name))
            print("Node list:", "  ".join(nodes))
            sys.exit(0)

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def pre_processing(self):
        """Perform initialization prior to analysis."""
        console_debug("analysis", "prepping to do some analysis")
        console_log("analysis", "deriving rocprofiler-compute metrics...")
        # initalize output file
        self._output = (
            open(self.__args.output_file, "w+")
            if self.__args.output_file
            else sys.stdout
        )

        # Read profiling config
        self._profiling_config = file_io.load_profiling_config(self.__args.path[0][0])

        # initalize runs
        self._runs = self.initalize_runs()

        # set filters
        if self.__args.gpu_kernel:
            for d, gk in zip(self.__args.path, self.__args.gpu_kernel):
                self._runs[d[0]].filter_kernel_ids = gk
        if self.__args.gpu_id:
            if len(self.__args.gpu_id) == 1 and len(self.__args.path) != 1:
                for i in range(len(self.__args.path) - 1):
                    self.__args.gpu_id.extend(self.__args.gpu_id)
            for d, gi in zip(self.__args.path, self.__args.gpu_id):
                self._runs[d[0]].filter_gpu_ids = gi
        if self.__args.gpu_dispatch_id:
            if len(self.__args.gpu_dispatch_id) == 1 and len(self.__args.path) != 1:
                for i in range(len(self.__args.path) - 1):
                    self.__args.gpu_dispatch_id.extend(self.__args.gpu_dispatch_id)
            for d, gd in zip(self.__args.path, self.__args.gpu_dispatch_id):
                self._runs[d[0]].filter_dispatch_ids = gd
        if self.__args.nodes:
            if len(self.__args.nodes) == 1 and len(self.__args.path) != 1:
                for i in range(len(self.__args.path) - 1):
                    self.__args.nodes.extend(self.__args.nodes)
            for d, gd in zip(self.__args.path, self.__args.nodes):
                self._runs[d[0]].nodes = gd

    @abstractmethod
    def run_analysis(self):
        """Run analysis."""
        console_debug("analysis", "generating analysis")
