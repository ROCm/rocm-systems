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
import sys
from pathlib import Path

import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline import Roofline
from utils import file_io, parser, schema, tty
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.roofline_calc import calc_ai_analyze, validate_roofline_csv
from utils.utils import (
    build_call_trees,
    build_call_trees_with_kernel_ids,
    process_torch_trace_output,
    write_torch_trace_consolidated_csv,
)


def parse_torch_operator_patterns(args: argparse.Namespace) -> list[str]:
    """Extract and flatten --torch-operator patterns from args.

    Returns ``["**"]`` when ``--torch-operator`` is given with no arguments,
    which matches all operators.  Returns ``[]`` when the flag is absent.
    """
    raw = getattr(args, "torch_operator", None)
    if raw is None:
        return []
    pattern_list: list[str] = []
    for operator_arg in raw:
        pattern_list.extend(
            pattern.strip()
            for pattern in str(operator_arg).split(",")
            if pattern.strip()
        )
    if not pattern_list:
        pattern_list = ["**"]
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

            if getattr(args, "list_torch_operators", False):
                consolidated_df, torch_trace_path = process_torch_trace_output(
                    path_info[0]
                )
                if consolidated_df.empty:
                    tty.list_torch_operators(path_info[0], {})
                    sys.exit(0)
                kernel_top_df = pd.read_csv(Path(path_info[0]) / "pmc_kernel_top.csv")
                write_torch_trace_consolidated_csv(consolidated_df, torch_trace_path)
                call_trees = build_call_trees_with_kernel_ids(
                    consolidated_df,
                    kernel_top_df=kernel_top_df,
                )
                tty.list_torch_operators(path_info[0], call_trees)
                sys.exit(0)

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
                    is_roofline_valid, roofline_error_msg = validate_roofline_csv(
                        Path(workload_path)
                    )
                    soc = self.get_socs()
                    if not soc or gpu_arch not in soc:
                        console_warning(
                            "roofline",
                            "Skipping roofline charting: "
                            f"gpu arch {gpu_arch} not in soc {soc}",
                        )
                    if is_roofline_valid:
                        soc_obj = soc[gpu_arch]
                        # Normalize user-facing "vL1D" to CSV column name "L1"
                        mem_level = (
                            args.mem_level
                            if isinstance(args.mem_level, list)
                            else [args.mem_level]
                        )
                        mem_level = [("L1" if m == "vL1D" else m) for m in mem_level]

                        roof_obj = Roofline(
                            args=soc_obj.get_args(),
                            mspec=soc_obj._mspec,
                            run_parameters={
                                "workload_dir": workload_path,
                                "device_id": 0,
                                "sort_type": str(args.sort),
                                "mem_level": mem_level,
                                "is_standalone": True,
                                "roofline_data_type": args.roofline_data_type,
                                "kernel_filter": bool(args.gpu_kernel),
                                "iteration_multiplexing": self._profiling_config.get(
                                    "iteration_multiplexing"
                                ),
                            },
                        )
                        workload.path = workload_path

                        pmc_df = parser.apply_filters(
                            workload, workload_path, is_gui=False, debug=args.debug
                        )
                        ai_data = calc_ai_analyze(
                            workload=workload,
                            pmc_df=pmc_df,
                            mspec=soc_obj._mspec,
                            sort_type=str(args.sort),
                            config=self._profiling_config,
                            arch_config=arch_config,
                        )

                        # NOTE: using default data type
                        roof_plot = roof_obj.cli_generate_plot(
                            dtype=roof_obj.get_dtype()[0],
                            ai_data=ai_data,
                        )

                        ops_fig, flops_fig, ops_dt, flops_dt = (
                            roof_obj.construct_plotly_figures(ai_data=ai_data)
                        )
                        roof_obj.save_html_files(ops_fig, flops_fig, ops_dt, flops_dt)
                    else:
                        console_warning(
                            "roofline",
                            "Skipping roofline charting: "
                            f"Invalid roofline.csv: {roofline_error_msg}",
                        )

            tty.show_all(
                args,
                self._runs,
                arch_config,
                self._output,
                self._profiling_config,
                roof_plot=roof_plot,
            )

    def apply_torch_operator_filter(
        self, args: argparse.Namespace, workload: schema.Workload, workload_path: str
    ) -> None:
        """Set workload.filter_kernel_ids based on --torch-operator patterns.

        Called in pre_processing *before* load_table_data so that metric
        evaluation runs once with the correct kernel filter — the same
        approach used by -k/--kernel.
        """
        torch_trace_dir = Path(workload_path) / "torch_trace"
        consolidated_path = torch_trace_dir / "consolidated.csv"

        if consolidated_path.exists():
            consolidated_df = pd.read_csv(consolidated_path)
            console_log(
                "torch trace",
                f"Loaded cached {consolidated_path}. "
                "Delete torch_trace/ directory to force regeneration from raw traces.",
            )
        else:
            consolidated_df, torch_trace_path = process_torch_trace_output(
                workload_path
            )
            if consolidated_df.empty:
                console_warning(
                    "torch trace",
                    "No torch operator data found in this workload. "
                    "Proceeding without torch operator filter.",
                )
                return
            write_torch_trace_consolidated_csv(consolidated_df, torch_trace_path)

        pattern_list = parse_torch_operator_patterns(args)
        all_operators = consolidated_df["Operator_Name"].dropna().unique()
        matched_names = [
            str(op).strip()
            for op in all_operators
            if any(
                parser.torch_operator_pattern_matches(p.strip(), str(op).strip())
                for p in pattern_list
            )
        ]

        if not matched_names:
            console_warning(
                "torch trace",
                f"No operators matched the pattern(s): {pattern_list}",
            )
            return

        matched_df = consolidated_df[
            consolidated_df["Operator_Name"].isin(matched_names)
        ].copy()

        kernel_top_df = pd.read_csv(str(Path(workload_path) / "pmc_kernel_top.csv"))
        name_to_id: dict[str, int] = {
            str(kernel_name).strip(): idx
            for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
        }

        matched_df["Kernel_ID"] = matched_df["Kernel_Name"].str.strip().map(name_to_id)
        workload.matched_torch_trace_df = matched_df

        kernel_names = set(matched_df["Kernel_Name"].dropna().str.strip().unique())
        kernel_ids = sorted(
            name_to_id[kernel_name]
            for kernel_name in kernel_names
            if kernel_name in name_to_id
        )

        if workload.filter_kernel_ids:
            existing_ids = set(workload.filter_kernel_ids)
            kernel_ids = [
                kernel_id for kernel_id in kernel_ids if kernel_id in existing_ids
            ]

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
                    "No kernels found for matched operators. No kernels to analyze.",
                )

    def handle_torch_operator(
        self, args: argparse.Namespace, workload: schema.Workload
    ) -> None:
        """Display matched torch operator call tree."""
        matched_df = workload.matched_torch_trace_df
        if matched_df.empty:
            return

        call_trees = build_call_trees(matched_df)

        pattern_list = parse_torch_operator_patterns(args)
        matched_operators = matched_df["Operator_Name"].dropna().unique()
        print(f"\n{'=' * 80}")
        print(f"Matched PyTorch Operators: {', '.join(pattern_list)}")
        print("Grouped by source location, sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        tty.show_call_tree(call_trees)
        print(f"{'=' * 80}")

        console_log(
            "torch trace",
            f"Matched {len(matched_operators)} operator(s): {list(matched_operators)}",
        )
