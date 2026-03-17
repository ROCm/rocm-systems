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

import sys
from pathlib import Path

import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser, tty
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, console_log, demarcate
from utils.utils import process_torch_trace_output


class cli_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------
    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        args = self.get_args()

        if args.random_port:
            console_error("--gui flag is required to enable --random-port")

        for path_info in args.path:
            workload = self._runs[path_info[0]]

            # create 'mega dataframe'
            workload.raw_pmc = file_io.create_df_pmc(
                path_info[0],
                args.nodes,
                args.spatial_multiplexing,
                args.kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            if args.spatial_multiplexing:
                workload.raw_pmc = self.spatial_multiplex_merge_counters(
                    workload.raw_pmc
                )

            if self._profiling_config.get("iteration_multiplexing") is not None:
                workload.raw_pmc = self.iteration_multiplex_impute_counters(
                    workload.raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                )

            file_io.create_df_kernel_top_stats(
                df_in=workload.raw_pmc,
                raw_data_dir=path_info[0],
                filter_gpu_ids=workload.filter_gpu_ids,
                filter_dispatch_ids=workload.filter_dispatch_ids,
                filter_nodes=workload.filter_nodes,
                time_unit=args.time_unit,
                kernel_verbose=args.kernel_verbose,
            )

            if getattr(args, "list_torch_operators", False):
                kernel_top_df = pd.read_csv(Path(path_info[0]) / "pmc_kernel_top.csv")
                file_data = process_torch_trace_output(
                    path_info[0],
                    kernel_top_df=kernel_top_df,
                    kernel_verbose=args.kernel_verbose,
                )
                tty.list_torch_operators(path_info[0], file_data)
                sys.exit(0)

            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(workload.raw_pmc, args.kernel_verbose)

            # create the loaded table
            parser.load_table_data(
                workload=workload,
                dir_path=path_info[0],
                is_gui=False,
                args=args,
                config=self._profiling_config,
            )

    @demarcate
    def run_analysis(self) -> None:
        """Run CLI analysis."""
        super().run_analysis()

        args = self.get_args()

        workload_path = args.path[0][0]
        workload = self._runs[workload_path]
        gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
        arch_config = self._arch_configs[gpu_arch]

        if getattr(args, "torch_operator", False):
            # Check whether any torch operator data was actually loaded
            torch_ops = getattr(workload, "torch_operators", None)
            if not torch_ops:
                console_error(
                    "No torch operators found in the profiling data. "
                    'Please ensure that workload is profiled with "--torch-trace" '
                    'and analyze is run with "--list-torch-operators" before '
                    'using "--torch-operator".'
                )
                # Abort analysis since the requested torch operator data is unavailable.
                return

            operator_args = args.torch_operator
            operator_list = []
            for op in operator_args:
                operator_list.extend([
                    o.strip() for o in str(op).split(",") if o.strip()
                ])
            operator_list = [o for o in operator_list if o]

            for op in operator_list:
                is_hierarchy = "/" in op
                lookup = op.split("/")[-1] if is_hierarchy else op
                op_key = lookup.replace("torch.", "").replace(".", "_")
                df = torch_ops.get(op_key)
                if df is None:
                    console_log(f"No data for operator: {op}")
                    continue
                if is_hierarchy:
                    df = df[df["Operator_Name"] == op]
                    if df.empty:
                        console_log(f"No rows for operator: {op}")
                        continue
                tty.show_torch_operator_table(op, df)

        if args.list_stats:
            tty.show_kernel_stats(
                args,
                self._runs,
                arch_config,
                self._output,
            )
        else:
            roof_plot = None

            # Generate roofline plot for single-path, compatible architectures
            if (len(args.path)) == 1:
                if gpu_arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx950"]:
                    soc = self.get_socs()
                    if soc and gpu_arch in soc:
                        roof_obj = soc[gpu_arch].roofline_obj

                        if roof_obj:
                            # store path in workload for calc_ai_analyze
                            workload.path = workload_path

                            # NOTE: using default data type
                            roof_plot = roof_obj.cli_generate_plot(
                                dtype=roof_obj.get_dtype()[0],
                                workload=workload,
                                config=self._profiling_config,
                                arch_config=arch_config,
                            )

            tty.show_all(
                args,
                self._runs,
                arch_config,
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )
