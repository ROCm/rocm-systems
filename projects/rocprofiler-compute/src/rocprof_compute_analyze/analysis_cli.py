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

import pathlib

import pandas as pd

import sys

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import file_io, parser, tty
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, console_log, console_warning, demarcate


def parse_torch_operator_patterns(args) -> list[str]:
    """Extract and flatten --torch-operator patterns from args."""
    operator_args = getattr(args, "torch_operator", None) or []
    pattern_list: list[str] = []
    for op in operator_args:
        pattern_list.extend(o.strip() for o in str(op).split(",") if o.strip())
    return pattern_list


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

            # demangle and overwrite original 'Kernel_Name'
            kernel_name_shortener(workload.raw_pmc, args.kernel_verbose)

            if getattr(args, "torch_operator", None) is not None:
                self.apply_torch_operator_filter(args, workload, path_info[0])

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

        if getattr(args, "list_torch_operators", False):
            kernel_name_to_id = None
            kernel_top_df = workload.dfs.get(parser.PMC_KERNEL_TOP_TABLE_ID)
            if (
                kernel_top_df is not None
                and not kernel_top_df.empty
                and "Kernel_Name" in kernel_top_df.columns
            ):
                kernel_name_to_id = {
                    str(row["Kernel_Name"]).strip(): idx
                    for idx, row in kernel_top_df.iterrows()
                }
            self.list_torch_operators(kernel_name_to_id=kernel_name_to_id)
            sys.exit(0)

        if getattr(args, "torch_operator", None) is not None:
            self.handle_torch_operator(args, workload)

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

    def apply_torch_operator_filter(self, args, workload, workload_path: str) -> None:
        """Set workload.filter_kernel_ids based on --torch-operator patterns.

        Called in pre_processing *before* load_table_data so that metric
        evaluation runs once with the correct kernel filter — the same
        approach used by -k/--kernel.
        """
        parser.load_torch_trace_data(workload, workload_path, args.kernel_verbose)

        if not workload.torch_operators:
            console_error(
                "torch trace",
                "No per-operator data found. "
                'Please run "--list-torch-operators" first to generate '
                "per-operator CSVs before using --torch-operator.",
            )

        pattern_list = parse_torch_operator_patterns(args)
        matched = parser.get_matched_torch_operators_for_display(
            workload.torch_operators, pattern_list
        )

        if not matched:
            console_warning(
                "torch trace",
                f"No operators matched the pattern(s): {pattern_list}",
            )
            return

        kernel_top_csv = pd.read_csv(
            str(pathlib.Path(workload_path) / "pmc_kernel_top.csv")
        )
        name_to_id: dict[str, int] = {
            str(row["Kernel_Name"]).strip(): idx
            for idx, row in kernel_top_csv.iterrows()
        }

        kernel_names: set[str] = set()
        for _, op_df in matched:
            if "Kernel_Name" in op_df.columns:
                kernel_names.update(
                    str(k).strip() for k in op_df["Kernel_Name"].dropna().unique()
                )

        kernel_ids = sorted(name_to_id[n] for n in kernel_names if n in name_to_id)

        if workload.filter_kernel_ids:
            existing_ids = set(workload.filter_kernel_ids)
            kernel_ids = [kid for kid in kernel_ids if kid in existing_ids]

        if kernel_ids:
            workload.filter_kernel_ids = kernel_ids
            console_log(
                "torch trace",
                f"Torch operator filter selected {len(kernel_ids)} kernel(s) "
                "for metric analysis.",
            )
        else:
            if workload.filter_kernel_ids:
                console_error(
                    "torch trace",
                    "No torch-operator kernels overlap with the -k filter "
                    f"{workload.filter_kernel_ids}. No kernels to analyze.",
                )
            else:
                console_error(
                    "torch trace",
                    "No kernels found for matched operators. "
                    "No kernels to analyze.",
                )

    def handle_torch_operator(self, args, workload) -> None:
        """Display matched torch operator hierarchies and filtered kernels."""
        pattern_list = parse_torch_operator_patterns(args)
        matched = parser.get_matched_torch_operators_for_display(
            workload.torch_operators, pattern_list
        )

        if not matched:
            return

        kernel_top_df = workload.dfs.get(parser.PMC_KERNEL_TOP_TABLE_ID)
        name_to_id: dict[str, int] = {}
        if (
            kernel_top_df is not None
            and not kernel_top_df.empty
            and "Kernel_Name" in kernel_top_df.columns
        ):
            name_to_id = {
                str(row["Kernel_Name"]).strip(): idx
                for idx, row in kernel_top_df.iterrows()
            }

        kernel_names: set[str] = set()
        for _, op_df in matched:
            if "Kernel_Name" in op_df.columns:
                kernel_names.update(
                    str(k).strip() for k in op_df["Kernel_Name"].dropna().unique()
                )

        for hierarchy, df in matched:
            tty.show_torch_operator_hierarchy(hierarchy, df)

        console_log(
            "torch trace",
            f"Matched hierarchies: {[h for h, _ in matched]}",
        )
        if workload.filter_kernel_ids:
            selected_ids = set(workload.filter_kernel_ids)
            kernel_names = {
                n for n in kernel_names if name_to_id.get(n, -1) in selected_ids
            }

        console_log("torch trace", "Filtered kernels:")
        for name in sorted(kernel_names):
            kid = name_to_id.get(name, "?")
            short = tty.extract_kernel_name(name)
            console_log("torch trace", f"  [{kid}] {short}")
