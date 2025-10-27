##############################################################################
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
from collections import OrderedDict
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_tui.utils.tui_utils import (
    get_top_kernels_and_dispatch_ids,
    process_panels_to_dataframes,
)
from utils import file_io, parser, schema
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, demarcate


class tui_analysis(OmniAnalyze_Base):
    def __init__(
        self, args: argparse.Namespace, supported_archs: dict[str, str], path: str
    ) -> None:
        super().__init__(args, supported_archs)
        self.path = path
        self.args = self.get_args()
        self.raw_dfs: dict[str, dict] = {}

    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        self._profiling_config = file_io.load_profiling_config(self.path)
        self._runs = self.initalize_runs()

        if self.args.random_port:
            console_error("--gui flag is required to enable --random-port")

        # Process PMC data
        workload = self._runs[self.path]

        workload.raw_pmc = file_io.create_df_pmc(
            self.path,
            self.args.nodes,
            self.args.spatial_multiplexing,
            self.args.kernel_verbose,
            self.args.verbose,
            self._profiling_config,
        )

        if self.args.spatial_multiplexing:
            workload.raw_pmc = self.spatial_multiplex_merge_counters(workload.raw_pmc)

        file_io.create_df_kernel_top_stats(
            df_in=workload.raw_pmc,
            raw_data_dir=self.path,
            filter_gpu_ids=workload.filter_gpu_ids,
            filter_dispatch_ids=workload.filter_dispatch_ids,
            filter_nodes=workload.filter_nodes,
            time_unit=self.args.time_unit,
            kernel_verbose=self.args.kernel_verbose,
        )
        kernel_name_shortener(self._runs[self.path].raw_pmc, self.args.kernel_verbose)

        # 1. load top kernel
        parser.load_non_mertrics_table(
            workload=self._runs[self.path], dir_path=self.path, args=self.args
        )

        # 2. Generate kernel-specific dataframes
        self.raw_dfs = {}
        for idx in workload.raw_pmc.index:
            kernel_df = workload.raw_pmc.loc[[idx]]
            kernel_name = str(kernel_df.pmc_perf["Kernel_Name"].loc[idx])
            kernel_dfs = copy.deepcopy(workload.dfs)

            parser.eval_metric(
                kernel_dfs,
                workload.dfs_type,
                workload.sys_info.iloc[0],
                workload.roofline_peaks,
                kernel_df,
                self.args.debug,
                self._profiling_config,
            )

            self.raw_dfs[kernel_name] = kernel_dfs

    def initalize_runs(
        self, normalization_filter: Optional[str] = None
    ) -> OrderedDict[str, schema.Workload]:
        # Load system info and configure
        sys_info = file_io.load_sys_info(str(Path(self.path) / "sysinfo.csv"))
        arch = sys_info.iloc[0]["gpu_arch"]

        self.generate_configs(
            arch,
            self.args.config_dir,
            self.args.list_stats,
            self.args.filter_metrics,
            sys_info.iloc[0],
        )
        self.load_options(normalization_filter)

        # Create workload with system and roofline data
        w = schema.Workload()
        w.sys_info = (
            parser.correct_sys_info(
                self.get_socs()[arch]._mspec, self.args.specs_correction
            )
            if self.args.specs_correction
            else sys_info
        )

        roofline_path = Path(self.path) / "roofline.csv"
        if roofline_path.is_file() and not getattr(self.args, "no_roof", False):
            w.roofline_peaks = pd.read_csv(roofline_path)

        w.avail_ips = w.sys_info["ip_blocks"].item().split("|")
        w.dfs = copy.deepcopy(self._arch_configs[arch].dfs)
        w.dfs_type = self._arch_configs[arch].dfs_type

        self._runs[self.path] = w
        return self._runs

    def run_kernel_analysis(self) -> dict[str, Any]:
        arch = list(self._arch_configs.keys())[0]
        return {
            kernel_name: process_panels_to_dataframes(
                self.args, df, self._arch_configs[arch], roof_plot=None
            )
            for kernel_name, df in self.raw_dfs.items()
        }

    def run_top_kernel(self) -> Optional[list[dict[Hashable, Any]]]:
        return get_top_kernels_and_dispatch_ids(self._runs)
