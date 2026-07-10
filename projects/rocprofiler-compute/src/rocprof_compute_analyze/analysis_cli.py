# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.kernel_filter import (
    ML_API_ANALYSIS_CLI_OPTIONS,
    KernelFilter,
    build_operator_filter,
    filter_by_backend,
    load_consolidated_ml_api_trace,
    parse_operator_patterns,
    resolve_kernel_filter,
)
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.roofline_calc import calc_ai_analyze
from utils.utils_analysis import (
    build_call_trees_with_kernel_ids,
    build_operator_summary,
    get_matrix_ops_type,
)
from utils.utils_common import validate_roofline_csv

# Maps each ML API trace backend to its analyze CLI attributes and display label.
_ML_API_ANALYSIS_CLI_OPTIONS = ML_API_ANALYSIS_CLI_OPTIONS


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

        active_operator_filters = [
            cli["filter_attr"]
            for cli in _ML_API_ANALYSIS_CLI_OPTIONS.values()
            if getattr(args, cli["filter_attr"], None) is not None
        ]
        if len(active_operator_filters) > 1:
            console_error(
                "analysis",
                "Only one operator filter may be used per analysis run. "
                "Run the analysis separately for each framework.",
            )

        active_operator_lists = [
            cli["list_attr"]
            for cli in _ML_API_ANALYSIS_CLI_OPTIONS.values()
            if getattr(args, cli["list_attr"], False)
        ]
        if len(active_operator_lists) > 1:
            console_error(
                "analysis",
                "Only one operator listing may be used per analysis run. "
                "Run the analysis separately for each framework.",
            )

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

            # create 'mega dataframe'
            workload.raw_pmc = file_io.create_df_pmc(
                path_info[0],
                args.kernel_verbose,
                args.verbose,
                self._profiling_config,
            )

            if self._profiling_config.get("iteration_multiplexing") is not None:
                workload.raw_pmc = self.iteration_multiplex_impute_counters(
                    workload.raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                    workload_dir=Path(path_info[0]),
                )

            # --list-operators needs the full Top Stats table, then exits.
            list_backend = next(
                (
                    backend
                    for backend, cli in _ML_API_ANALYSIS_CLI_OPTIONS.items()
                    if getattr(args, cli["list_attr"], False)
                ),
                None,
            )
            if list_backend is not None:
                kernel_top_df, _ = self._build_and_store_kernel_top_stats(
                    workload=workload,
                    workload_path=path_info[0],
                    args=args,
                )
                self.list_operators(path_info[0], kernel_top_df, list_backend)
                sys.exit(0)

            operator_backend = next(
                (
                    backend
                    for backend, cli in _ML_API_ANALYSIS_CLI_OPTIONS.items()
                    if getattr(args, cli["filter_attr"], None) is not None
                ),
                None,
            )

            # Resolve the -k selection. Index-based selections need the full Top
            # Stats table, but the workload should only store the final table.
            if workload.kernel_selection is not None:
                base_kernel_top_df, _ = file_io.create_df_kernel_top_stats(
                    df_in=workload.raw_pmc,
                    raw_data_dir=path_info[0],
                    filter_gpu_ids=workload.filter_gpu_ids,
                    filter_dispatch_ids=workload.filter_dispatch_ids,
                    time_unit=args.time_unit,
                    kernel_verbose=args.kernel_verbose,
                )
                workload.kernel_filter = resolve_kernel_filter(
                    workload.kernel_selection, base_kernel_top_df
                )

            if operator_backend is not None:
                # Operator filter narrows the selection and scopes Top Stats to
                # the selected kernels.
                self.apply_operator_filter(
                    args, workload, path_info[0], operator_backend
                )

            self._build_and_store_kernel_top_stats(
                workload=workload,
                workload_path=path_info[0],
                args=args,
                kernel_filter=workload.kernel_filter,
            )

            # create the loaded table
            gpu_arch = workload.sys_info.iloc[0]["gpu_arch"]
            parser.load_table_data(
                workload=workload,
                dir_path=path_info[0],
                is_gui=False,
                args=args,
                dfs_expressions=self._arch_configs[gpu_arch].dfs_expressions,
                pc_sampling_tool_data=pc_sampling_data,
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

        for backend, cli in _ML_API_ANALYSIS_CLI_OPTIONS.items():
            if getattr(args, cli["filter_attr"], None) is not None:
                self.handle_operator(args, workload, backend)

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
                                "sort_type": str(args.sort),
                                "mem_level": args.mem_level,
                                "roofline_data_type": args.roofline_data_type,
                                "kernel_filter": workload.kernel_filter.describe(),
                                "iteration_multiplexing": self._profiling_config.get(
                                    "iteration_multiplexing"
                                ),
                                "matrix_ops_type": get_matrix_ops_type(
                                    workload.sys_info.iloc[0]["gpu_series"]
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
                            arch_config=arch_config,
                        )

                        # NOTE: using default data type
                        roof_plot = roof_obj.cli_generate_plot(
                            dtype=roof_obj.get_dtype()[0],
                            ai_data=ai_data,
                        )

                        (
                            ops_fig,
                            flops_fig,
                            ops_dt,
                            flops_dt,
                        ) = roof_obj.construct_plotly_figures(ai_data=ai_data)
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

    @staticmethod
    def _filter_by_backend(consolidated_df: pd.DataFrame, backend: str) -> pd.DataFrame:
        """Return the rows attributed to ``backend``.

        When the Backend column is absent, rows are treated as the torch
        backend.
        """
        if "Backend" in consolidated_df.columns:
            return filter_by_backend(consolidated_df, backend)
        return filter_by_backend(consolidated_df, backend)

    def list_operators(
        self,
        workload_path: str,
        kernel_top_df: pd.DataFrame,
        backend: str,
    ) -> None:
        """Render the operator call tree for a single backend."""
        label = _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"]
        consolidated_df = load_consolidated_ml_api_trace(workload_path)
        if consolidated_df.empty:
            tty.list_ml_operators(workload_path, {}, framework_label=label)
            return

        backend_df = self._filter_by_backend(consolidated_df, backend)
        if backend_df.empty:
            tty.list_ml_operators(workload_path, {}, framework_label=label)
            return

        call_trees = build_call_trees_with_kernel_ids(
            consolidated_df=backend_df,
            kernel_top_df=kernel_top_df,
        )
        tty.list_ml_operators(workload_path, call_trees, framework_label=label)

    @staticmethod
    def _build_and_store_kernel_top_stats(
        workload: schema.Workload,
        workload_path: str,
        args: argparse.Namespace,
        filter_kernel_names: Optional[list[str]] = None,
        kernel_filter: Optional[KernelFilter] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create and store Top Stats tables for a workload."""
        kernel_top_df, dispatch_info_df = file_io.create_df_kernel_top_stats(
            df_in=workload.raw_pmc,
            raw_data_dir=workload_path,
            filter_gpu_ids=workload.filter_gpu_ids,
            filter_dispatch_ids=workload.filter_dispatch_ids,
            time_unit=args.time_unit,
            kernel_verbose=args.kernel_verbose,
            filter_kernel_names=filter_kernel_names,
            kernel_filter=kernel_filter,
        )
        workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = kernel_top_df
        workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] = dispatch_info_df
        return kernel_top_df, dispatch_info_df

    def apply_operator_filter(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        backend: str,
    ) -> None:
        """Narrow the workload's kernel filter to a backend's operator.

        Operator matches are intersected with any existing -k selection
        (workload.kernel_filter). Matched trace rows are stored in
        workload.matched_ml_api_trace_dfs[backend] and the resulting selection in
        workload.kernel_filter. Returns without narrowing when no operator data
        is available; exits when a set filter matches nothing.
        """
        cli = _ML_API_ANALYSIS_CLI_OPTIONS[backend]
        label = cli["label"]
        base_filter = workload.kernel_filter
        result = build_operator_filter(args, workload, workload_path, backend)
        if result is None:
            return

        operator_filter, matched_df = result
        if operator_filter.is_active:
            workload.kernel_filter = operator_filter
            workload.matched_ml_api_trace_dfs[backend] = matched_df
            console_log(
                "ml api trace",
                f"{label} operator filter selected {len(operator_filter.names)} "
                "kernel(s) for metric analysis.",
            )
            return

        if base_filter.is_active:
            console_error(
                "ml api trace",
                f"No {label}-operator kernels overlap with the -k filter. "
                "No kernels to analyze.",
            )
        else:
            console_error(
                "ml api trace",
                "No kernels found for matched operators. No kernels to analyze.",
            )

    def handle_operator(
        self, args: argparse.Namespace, workload: schema.Workload, backend: str
    ) -> None:
        """Display the matched operator call tree for a single backend."""
        cli = _ML_API_ANALYSIS_CLI_OPTIONS[backend]
        label = cli["label"]
        matched_df = workload.matched_ml_api_trace_dfs.get(backend)
        if matched_df is None or matched_df.empty:
            return

        # Build ids from the scoped Top Stats table so the call tree and Top
        # Stats reference the same kernel-id space.
        call_trees = build_call_trees_with_kernel_ids(
            consolidated_df=matched_df,
            kernel_top_df=workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID],
        )

        pattern_list = parse_operator_patterns(args, cli["filter_attr"])
        matched_operators = matched_df["Operator_Name"].dropna().unique()
        print(f"\n{'=' * 80}")
        print(f"Matched {label} Operators: {', '.join(pattern_list)}")
        print("Grouped by source location, sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        tty.show_call_tree(call_trees)
        tty.show_operator_summary(build_operator_summary(call_trees))
        print(f"{'=' * 80}")

        console_log(
            "ml api trace",
            f"Matched {len(matched_operators)} operator(s): {list(matched_operators)}",
        )
