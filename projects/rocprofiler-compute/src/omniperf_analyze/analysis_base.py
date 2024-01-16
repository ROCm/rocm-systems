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

from abc import ABC, abstractmethod
import os
import logging
import sys
import copy
from collections import OrderedDict
from pathlib import Path
from utils.utils import demarcate, error
from utils import schema, file_io, parser
import pandas as pd
from tabulate import tabulate

class OmniAnalyze_Base():
    def __init__(self,args,supported_archs):
        self.__args = args
        self._runs = OrderedDict() 
        self._arch_configs = {} 
        self.__supported_archs = supported_archs
        self._output = None 
        self.__socs = None # available OmniSoC objs

    def get_args(self):
        return self.__args
    def set_soc(self, omni_socs):
        self.__socs = omni_socs
    def get_socs(self):
        return self.__socs
    
    @demarcate
    def generate_configs(self, arch, config_dir, list_kernels, filter_metrics, sys_info):
        single_panel_config = file_io.is_single_panel_config(Path(config_dir), self.__supported_archs)
        
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

        parser.build_dfs(
            archConfigs=ac,
            filter_metrics=filter_metrics,
            sys_info=sys_info
        )
        self._arch_configs[arch] = ac
        return self._arch_configs
    
    @demarcate
    def list_metrics(self):
        args = self.__args
        if args.list_metrics in file_io.supported_arch.keys():
            arch = args.list_metrics
            if arch not in self._arch_configs.keys():
                sys_info = file_io.load_sys_info(Path(self.__args.path[0][0], "sysinfo.csv"))
                self.generate_configs(arch, args.config_dir, args.list_kernels, args.filter_metrics, sys_info)
            print(
                tabulate(
                    pd.DataFrame.from_dict(
                        self._arch_configs[args.list_metrics].metric_list,
                        orient="index",
                        columns=["Metric"],
                    ),
                    headers="keys",
                    tablefmt="fancy_grid"
                ),
                file=self._output
            )
            sys.exit(0)
        else:
            error("Unsupported arch")

    @demarcate
    def load_options(self, normalization_filter):
        if not normalization_filter:
            for k, v in self._arch_configs.items():
                parser.build_metric_value_string(v.dfs, v.dfs_type, self.__args.normal_unit)
        else:
            for k, v in self._arch_configs.items():
                parser.build_metric_value_string(v.dfs, v.dfs_type, normalization_filter)
        
        args = self.__args
        # Error checking for multiple runs and multiple gpu_kernel filters
        if args.gpu_kernel and (len(args.path) != len(args.gpu_kernel)):
            if len(args.gpu_kernel) == 1:
                for i in range(len(args.path) - 1):
                    args.gpu_kernel.extend(args.gpu_kernel)
            else:
                error("Error: the number of --filter-kernels doesn't match the number of --dir.")
    
    @demarcate
    def initalize_runs(self, normalization_filter=None):
        if self.__args.list_metrics:
            self.list_metrics()
        
        # load required configs
        for d in self.__args.path:
            sys_info = file_io.load_sys_info(Path(d[0], "sysinfo.csv"))
            arch = sys_info.iloc[0]["gpu_soc"]
            args = self.__args
            self.generate_configs(arch, args.config_dir, args.list_kernels, args.filter_metrics, sys_info)

        self.load_options(normalization_filter)
        
        for d in self.__args.path:
            w = schema.Workload()
            w.sys_info = file_io.load_sys_info(Path(d[0], "sysinfo.csv"))
            if self.__args.specs_correction:
                w.sys_info = parser.correct_sys_info(w.sys_info, self.__args.specs_correction)
            w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
            arch = w.sys_info.iloc[0]["gpu_soc"]
            w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
            w.dfs_type = self._arch_configs[arch].dfs_type
            w.soc_spec = self.get_socs()[arch].get_soc_param()
            self._runs[d[0]] = w

        return self._runs

    
    @demarcate
    def sanitize(self):
        """Perform sanitization of inputs
        """
        if not self.__args.list_metrics and not self.__args.path:
            error("The following arguments are required: -p/--path")
        # verify not accessing parent directories
        if ".." in str(self.__args.path):
            error("Access denied. Cannot access parent directories in path (i.e. ../)")
        # ensure absolute path
        for dir in self.__args.path:
            full_path = os.path.abspath(dir[0])
            dir[0] = full_path
            if not os.path.isdir(dir[0]):
                error("Invalid directory {}\nPlease try again.".format(dir[0]))
    
    #----------------------------------------------------
    # Required methods to be implemented by child classes
    #----------------------------------------------------
    @abstractmethod
    def pre_processing(self):
        """Perform initialization prior to analysis.
        """
        logging.debug("[analysis] prepping to do some analysis")
        logging.info("[analysis] deriving Omniperf metrics...")
        # initalize output file
        self._output = open(self.__args.output_file, "w+") if self.__args.output_file else sys.stdout
        
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

    @abstractmethod
    def run_analysis(self):
        """Run analysis.
        """
        logging.debug("[analysis] generating analysis")
