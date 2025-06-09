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
import sys
from pathlib import Path

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_tui.utils.tui_utils import process_panels_to_dataframes
from utils import file_io, parser, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, demarcate


class tui_analysis(OmniAnalyze_Base):
    def __init__(self, args, supported_archs, path):
        super().__init__(args, supported_archs)
        self.path = str(path)

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self):
        """Perform any pre-processing steps prior to analysis."""
        # Read profiling config
        self._profiling_config = file_io.load_profiling_config(self.path)

        # initalize runs
        self._runs = self.initalize_runs()

        if self.get_args().random_port:
            console_error("--gui flag is required to enable --random-port")

        # create 'mega dataframe'
        self._runs[self.path].raw_pmc = file_io.create_df_pmc(
            self.path,
            self.get_args().nodes,
            self.get_args().spatial_multiplexing,
            self.get_args().kernel_verbose,
            self.get_args().verbose,
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

        # demangle and overwrite original 'Kernel_Name'
        kernel_name_shortener(
            self._runs[self.path].raw_pmc, self.get_args().kernel_verbose
        )

        # create the loaded table
        parser.load_table_data(
            workload=self._runs[self.path],
            dir=self.path,
            is_gui=False,
            args=self.get_args(),
        )

    def initalize_runs(self, normalization_filter=None):
        # load required configs
        sysinfo_path = Path(self.path)
        sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
        arch = sys_info.iloc[0]["gpu_arch"]
        args = self.get_args()
        self.generate_configs(
            arch,
            args.config_dir,
            args.list_stats,
            args.filter_metrics,
            sys_info.iloc[0],
        )

        self.load_options(normalization_filter)

        w = schema.Workload()
        # FIXME:
        #    For regular single node case, load sysinfo.csv directly
        #    For multi-node, either the default "all", or specified some,
        #    pick up the one in the 1st sub_dir. We could fix it properly later.
        sysinfo_path = Path(self.path)
        w.sys_info = file_io.load_sys_info(sysinfo_path.joinpath("sysinfo.csv"))
        arch = w.sys_info.iloc[0]["gpu_arch"]
        mspec = self.get_socs()[arch]._mspec
        if args.specs_correction:
            w.sys_info = parser.correct_sys_info(mspec, args.specs_correction)
        w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
        w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
        w.dfs_type = self._arch_configs[arch].dfs_type
        self._runs[self.path] = w

        return self._runs

    @demarcate
    def run_analysis(self):
        """Run TUI analysis."""
        super().run_analysis()

        results = process_panels_to_dataframes(
            self.get_args(),
            self._runs,
            self._arch_configs[self._runs[self.path].sys_info.iloc[0]["gpu_arch"]],
            self._profiling_config,
        )
        return results
