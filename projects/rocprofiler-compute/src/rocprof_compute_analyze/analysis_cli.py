# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from membw_analysis.engine import run_membw_analysis
from ml_api_tracing.analysis import (
    active_operator_backends,
    apply_operator_filter,
    apply_sliced_kernel_top,
    process_ml_api_trace_output,
    select_operator_dispatches,
)
from ml_api_tracing.display import list_operators, show_matched_operator_report
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.logger import console_log, console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import calc_ai_analyze
from utils.utils_common import validate_roofline_csv


class cli_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------

    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        args = self.get_args()

        active_operator_filters, active_operator_lists = active_operator_backends(args)

        for path_info in args.path:
            workload = self._runs[path_info[0]]

            pc_sampling_data = self.load_pc_sampling_tool_data(path_info[0])

            # No counters collected -- derive scaffolding from the PC sampling
            # kernel trace and skip metrics calculation.
            if self.pc_sampling_only():
                console_log(
                    "analysis",
                    "Only PC sampling and kernel tracing data"
                    " available, metrics calculation will be"
                    " skipped",
                )
                self.build_pc_sampling_only_workload(
                    workload, path_info[0], args, pc_sampling_data
                )
                continue

            if active_operator_lists or active_operator_filters:
                process_ml_api_trace_output(workload, path_info[0])

            if active_operator_lists:
                workload.raw_pmc = file_io.create_df_pmc(
                    path_info[0],
                    args.verbose,
                )
                if self._profiling_config.get("iteration_multiplexing") is not None:
                    workload.raw_pmc = self.iteration_multiplex_impute_counters(
                        workload.raw_pmc,
                        policy=self._profiling_config["iteration_multiplexing"],
                        workload_dir=Path(path_info[0]),
                    )
                kernel_top_df, dispatch_info_df = file_io.create_df_kernel_top_stats(
                    df_in=workload.raw_pmc,
                    raw_data_dir=path_info[0],
                    filter_gpu_ids=workload.filter_gpu_ids,
                    filter_dispatch_ids=workload.filter_dispatch_ids,
                    time_unit=args.time_unit,
                )
                workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = kernel_top_df
                workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] = dispatch_info_df
                self.list_operators(path_info[0], kernel_top_df, active_operator_lists)
                continue

            if active_operator_filters:
                pmc_group_keys = self.apply_operator_filter(
                    args, workload, path_info[0], active_operator_filters
                )
                workload.raw_pmc, slot_to_id = file_io.create_df_pmc_with_slot_ids(
                    path_info[0],
                    args.verbose,
                )
                select_operator_dispatches(
                    workload, pmc_group_keys, slot_to_id, active_operator_filters
                )
            else:
                workload.raw_pmc = file_io.create_df_pmc(
                    path_info[0],
                    args.verbose,
                )

            if self._profiling_config.get("iteration_multiplexing") is not None:
                workload.raw_pmc = self.iteration_multiplex_impute_counters(
                    workload.raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                    workload_dir=Path(path_info[0]),
                )

            kernel_top_df, dispatch_info_df = file_io.create_df_kernel_top_stats(
                df_in=workload.raw_pmc,
                raw_data_dir=path_info[0],
                filter_gpu_ids=workload.filter_gpu_ids,
                filter_dispatch_ids=workload.filter_dispatch_ids,
                time_unit=args.time_unit,
            )
            workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = kernel_top_df
            workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] = dispatch_info_df

            if active_operator_filters:
                apply_sliced_kernel_top(
                    workload, kernel_top_df, args, active_operator_filters
                )

            # create the loaded table
            gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
            parser.load_table_data(
                workload=workload,
                dir_path=path_info[0],
                args=args,
                dfs_expressions=self._arch_configs[gpu_arch].dfs_expressions,
                pc_sampling_tool_data=pc_sampling_data,
            )

            if self.membw_analysis_collected():
                workload.membw_result = run_membw_analysis(workload.dfs, gpu_arch)

        if active_operator_lists:
            sys.exit(0)

    @demarcate
    def run_analysis(self) -> None:
        """Run CLI analysis."""
        super().run_analysis()

        args = self.get_args()

        workload_path = args.path[0][0]
        workload = self._runs[workload_path]
        gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
        arch_config = self._arch_configs[gpu_arch]

        active_operator_filters, _ = active_operator_backends(args)
        if active_operator_filters:
            for path_info in args.path:
                self.handle_operator(
                    args, self._runs[path_info[0]], active_operator_filters
                )

        if args.list_stats:
            tty.show_kernel_stats(
                args,
                self._runs,
                arch_config,
                self._output,
            )
        else:
            # Generate roofline plot for single-path, compatible architectures
            if (len(args.path)) == 1:
                if gpu_arch in ROOFLINE_SUPPORTED:
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

                        roof_obj = Roofline(
                            args=soc_obj.get_args(),
                            mspec=soc_obj._mspec,
                            run_parameters={
                                "workload_dir": workload_path,
                                "device_id": 0,
                                "gpu_arch": gpu_arch,
                                "kernel_filter": bool(args.gpu_kernel),
                                "iteration_multiplexing": self._profiling_config.get(
                                    "iteration_multiplexing"
                                ),
                                "matrix_ops_type": (
                                    mi_gpu_specs.get_roofline_matrix_type(gpu_arch)
                                ),
                            },
                        )
                        workload.path = workload_path

                        pmc_df = parser.apply_filters(workload, debug=args.debug)
                        ai_data = calc_ai_analyze(
                            workload=workload,
                            pmc_df=pmc_df,
                            arch_config=arch_config,
                        )

                        (
                            ops_fig,
                            flops_fig,
                            ops_dt,
                            flops_dt,
                        ) = roof_obj.construct_plotly_figures(ai_data=ai_data)
                        workload.roofline_html_path = roof_obj.save_html_files(
                            ops_fig, flops_fig, ops_dt, flops_dt
                        )
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
            )

    def list_operators(
        self, workload_path: str, kernel_top_df: pd.DataFrame, backends: List[str]
    ) -> None:
        """Coordinate the feature listing for this workload."""
        list_operators(
            workload_path, self._runs[workload_path], kernel_top_df, backends
        )

    def apply_operator_filter(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        backends: List[str],
    ) -> List[Tuple]:
        """Request staged selection; retain workload_path for existing CLI callers."""
        return apply_operator_filter(args, workload, backends)

    def handle_operator(
        self, args: argparse.Namespace, workload: schema.Workload, backends: List[str]
    ) -> None:
        """Coordinate the feature's matched operator report."""
        show_matched_operator_report(args, workload, backends)
