# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path

import pandas as pd

from membw_analysis.engine import run_membw_analysis
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.roofline_calc import calc_ai_analyze
from utils.utils_analysis import (
    CallTreeNode,
    build_operator_summary,
    copy_matched_operator_subtree,
    filter_forest_by_backend,
    get_matrix_ops_type,
    process_ml_api_trace_output,
)
from utils.utils_common import validate_roofline_csv

# Maps each ML API trace backend to its analyze CLI attributes and display label.
_ML_API_ANALYSIS_CLI_OPTIONS = {
    "torch": {
        "filter_attr": "torch_operator",
        "list_attr": "list_torch_operators",
        "label": "PyTorch",
    },
    "triton": {
        "filter_attr": "triton_operator",
        "list_attr": "list_triton_operators",
        "label": "Triton",
    },
}


def parse_operator_patterns(args: argparse.Namespace, attr: str) -> list[str]:
    """Extract and flatten operator glob patterns from ``args.<attr>``.

    Returns ``["**"]`` when the flag is given with no arguments (match all),
    and ``[]`` when the flag is absent.
    """
    raw = getattr(args, attr, None)
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


def _collect_glob_matched_nodes(
    forest: dict[str, list[CallTreeNode]],
    patterns: list[str],
    backend: str,
) -> list[CallTreeNode]:
    """Return nodes whose backend matches and whose path or name glob-matches."""
    matched_nodes: list[CallTreeNode] = []

    def walk(node: CallTreeNode, ancestors: list[str]) -> None:
        path = "/".join(ancestors + [node.name])
        if node.backend == backend and any(
            parser.torch_operator_pattern_matches(pattern, path)
            or parser.torch_operator_pattern_matches(pattern, node.name)
            for pattern in patterns
        ):
            matched_nodes.append(node)
        for child in node.children:
            walk(child, ancestors + [node.name])

    for roots in forest.values():
        for root in roots:
            walk(root, [])
    return matched_nodes


def _assign_kernel_ids_from_top(
    node: CallTreeNode, name_to_id: dict[str, int]
) -> set[int]:
    """Set KernelStats.kernel_id on this node and descendants; return those ids."""
    kernel_ids: set[int] = set()
    for kernel_name, stats in node.kernels.items():
        kernel_id = name_to_id.get(str(kernel_name).strip())
        if kernel_id is not None:
            stats.kernel_id = kernel_id
            kernel_ids.add(kernel_id)
    for child in node.children:
        kernel_ids.update(_assign_kernel_ids_from_top(child, name_to_id))
    return kernel_ids


def _warn_ml_api_trace_errors(workload: schema.Workload) -> None:
    """Print accumulated ML API trace errors after the call tree."""
    errors = workload.ml_api_trace_errors
    if not errors:
        return
    console_warning("analysis", f"{len(errors)} ML API trace error(s):")
    for exc in errors:
        console_warning("analysis", str(exc))


def _ml_api_operator_cli_requested(args: argparse.Namespace) -> bool:
    """Return True when a list-operators or operator-filter flag is set."""
    for cli in _ML_API_ANALYSIS_CLI_OPTIONS.values():
        if getattr(args, cli["list_attr"], False):
            return True
        if getattr(args, cli["filter_attr"], None) is not None:
            return True
    return False


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

            if _ml_api_operator_cli_requested(args):
                process_ml_api_trace_output(workload, path_info[0])

            for backend, cli in _ML_API_ANALYSIS_CLI_OPTIONS.items():
                if getattr(args, cli["list_attr"], False):
                    self.list_operators(path_info[0], kernel_top_df, backend)
                    _warn_ml_api_trace_errors(workload)
                    sys.exit(0)

            for backend, cli in _ML_API_ANALYSIS_CLI_OPTIONS.items():
                if getattr(args, cli["filter_attr"], None) is not None:
                    self.apply_operator_filter(args, workload, path_info[0], backend)

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

            if getattr(args, "membw_analysis", False):
                workload.membw_result = run_membw_analysis(workload.dfs, gpu_arch)

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
        if any(
            getattr(args, cli["filter_attr"], None) is not None
            for cli in _ML_API_ANALYSIS_CLI_OPTIONS.values()
        ):
            _warn_ml_api_trace_errors(workload)

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
                                "kernel_filter": bool(args.gpu_kernel),
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

    def list_operators(
        self,
        workload_path: str,
        kernel_top_df: pd.DataFrame,
        backend: str,
    ) -> None:
        """Render the operator call tree for a single backend."""
        label = _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"]
        forest_view = filter_forest_by_backend(
            self._runs[workload_path].ml_api_call_trees, backend
        )
        tty.list_ml_operators(workload_path, forest_view, framework_label=label)

    def apply_operator_filter(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        backend: str,
    ) -> None:
        """Set workload.filter_kernel_ids from the backend's operator filter.

        Operator matches are intersected with the -k/--kernel filter when set;
        matched nodes are stored in workload.ml_api_glob_matches.
        """
        cli = _ML_API_ANALYSIS_CLI_OPTIONS[backend]
        label = cli["label"]
        pattern_list = parse_operator_patterns(args, cli["filter_attr"])
        matched_nodes = _collect_glob_matched_nodes(
            workload.ml_api_call_trees, pattern_list, backend
        )
        if not matched_nodes:
            console_warning(
                "ml api trace",
                f"No {label} operators matched the pattern(s): {pattern_list}",
            )
            _warn_ml_api_trace_errors(workload)
            sys.exit(0)

        kernel_top_df = workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID]
        name_to_id: dict[str, int] = {
            str(kernel_name).strip(): idx
            for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
        }
        kernel_ids: set[int] = set()
        for node in matched_nodes:
            kernel_ids.update(_assign_kernel_ids_from_top(node, name_to_id))
        selected_ids = sorted(kernel_ids)
        workload.ml_api_glob_matches = matched_nodes

        if workload.filter_kernel_ids:
            existing_ids = set(workload.filter_kernel_ids)
            selected_ids = [
                kernel_id for kernel_id in selected_ids if kernel_id in existing_ids
            ]

        if selected_ids:
            workload.filter_kernel_ids = selected_ids
            console_log(
                "ml api trace",
                f"{label} operator filter selected {len(selected_ids)} kernel(s) "
                "for metric analysis.",
            )
        elif workload.filter_kernel_ids:
            _warn_ml_api_trace_errors(workload)
            console_error(
                "ml api trace",
                f"No {label}-operator kernels overlap with the -k filter "
                f"{workload.filter_kernel_ids}. No kernels to analyze.",
            )
        else:
            _warn_ml_api_trace_errors(workload)
            console_error(
                "ml api trace",
                "No kernels found for matched operators. No kernels to analyze.",
            )

    def handle_operator(
        self, args: argparse.Namespace, workload: schema.Workload, backend: str
    ) -> None:
        """Display the matched operator call tree for a single backend."""
        if not workload.ml_api_glob_matches:
            return
        cli = _ML_API_ANALYSIS_CLI_OPTIONS[backend]
        label = cli["label"]
        pattern_list = parse_operator_patterns(args, cli["filter_attr"])
        subtree = copy_matched_operator_subtree(
            workload.ml_api_call_trees, workload.ml_api_glob_matches
        )
        print(f"\n{'=' * 80}")
        print(f"Matched {label} Operators: {', '.join(pattern_list)}")
        print("Sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        tty.show_call_tree(subtree)
        tty.show_operator_summary(build_operator_summary(subtree))
        print(f"{'=' * 80}")
