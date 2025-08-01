##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

import copy
from pathlib import Path

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_tui.utils.tui_utils import (
    get_top_kernels_and_dispatch_ids,
    process_panels_to_dataframes,
)
from utils import file_io, parser, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, demarcate


class tui_analysis(OmniAnalyze_Base):
    def __init__(self, args, supported_archs, path):
        super().__init__(args, supported_archs)
        self.path = str(path)
        self.arch = None
        self.raw_dfs = {}
        self.kernel_dfs = {}

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        self._profiling_config = file_io.load_profiling_config(self.path)

        self._runs = self.initalize_runs()

        if self.get_args().random_port:
            console_error("--gui flag is required to enable --random-port")

        self._runs[self.path].raw_pmc = file_io.create_df_pmc(
            self.path,
            self.get_args().nodes,
            self.get_args().spatial_multiplexing,
            self.get_args().kernel_verbose,
            self.get_args().verbose,
            self._profiling_config,
        )

        if self.get_args().spatial_multiplexing:
            self._runs[self.path].raw_pmc = self.spatial_multiplex_merge_counters(
                self._runs[self.path].raw_pmc
            )

        file_io.create_df_kernel_top_stats(
            df_in=self._runs[self.path].raw_pmc,
            raw_data_dir=self.path,
            filter_gpu_ids=self._runs[self.path].filter_gpu_ids,
            filter_dispatch_ids=self._runs[self.path].filter_dispatch_ids,
            filter_nodes=self._runs[self.path].filter_nodes,
            time_unit=self.get_args().time_unit,
            max_stat_num=self.get_args().max_stat_num,
            kernel_verbose=self.get_args().kernel_verbose,
        )

        kernel_name_shortener(
            self._runs[self.path].raw_pmc, self.get_args().kernel_verbose
        )

        # 1. load top kernel
        parser.load_kernel_top(
            workload=self._runs[self.path], dir=self.path, args=self.get_args()
        )

        # 2. load table data for each kernel
        self.raw_dfs.clear()
        for idx in self._runs[self.path].raw_pmc.index:
            kernel_df = self._runs[self.path].raw_pmc.loc[[idx]]
            kernel_name = kernel_df.pmc_perf["Kernel_Name"].loc[idx]
            this_dfs = copy.deepcopy(self._runs[self.path].dfs)
            parser.eval_metric(
                this_dfs,
                self._runs[self.path].dfs_type,
                self._runs[self.path].sys_info.iloc[0],
                kernel_df,
                self.get_args().debug,
                self._profiling_config,
            )

            self.raw_dfs[kernel_name] = this_dfs

    def initalize_runs(self, normalization_filter=None):
        sysinfo_path = Path(self.path)
        sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
        self.arch = sys_info.iloc[0]["gpu_arch"]
        args = self.get_args()
        self.generate_configs(
            self.arch,
            args.config_dir,
            args.list_stats,
            args.filter_metrics,
            sys_info.iloc[0],
        )

        self.load_options(normalization_filter)

        w = schema.Workload()
        w.sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
        mspec = self.get_socs()[self.arch]._mspec
        if args.specs_correction:
            w.sys_info = parser.correct_sys_info(mspec, args.specs_correction)
        w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
        w.dfs = copy.deepcopy(self._arch_configs[self.arch].dfs)
        w.dfs_type = self._arch_configs[self.arch].dfs_type
        self._runs[self.path] = w

        return self._runs

    @demarcate
    def run_kernel_analysis(self):
        self.kernel_dfs.clear()
        for kernel_name, df in self.raw_dfs.items():
            self.kernel_dfs[kernel_name] = process_panels_to_dataframes(
                self.get_args(), df, self._arch_configs[self.arch], roof_plot=None
            )
        return self.kernel_dfs

    @demarcate
    def run_top_kernel(self):
        return get_top_kernels_and_dispatch_ids(self._runs)
