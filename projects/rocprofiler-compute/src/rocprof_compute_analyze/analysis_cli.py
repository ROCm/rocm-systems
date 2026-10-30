# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from membw_analysis.engine import run_membw_analysis
from ml_api_tracing.analysis import process_ml_api_trace_output
from ml_api_tracing.call_tree import iter_call_tree_paths, rollup_node_stats
from ml_api_tracing.models import CallTreeForest, CallTreeNode
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import calc_ai_analyze
from utils.utils_analysis import (
    build_operator_summary,
    copy_matched_operator_subtree,
    filter_forest_by_backends,
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


def parse_operator_patterns(
    args: argparse.Namespace, backends: list[str]
) -> Optional[dict[str, list[str]]]:
    """Parse operator glob patterns from args for each backend in backends.

    Returns a {backend: patterns} dict, or None when no filter flag is
    set for any backend. An empty flag value maps to ["**"].
    """
    framewise_patterns: dict[str, list[str]] = {}
    for backend in backends:
        raw = getattr(args, _ML_API_ANALYSIS_CLI_OPTIONS[backend]["filter_attr"], None)
        if raw is None:
            continue
        pattern_list: list[str] = []
        for operator_arg in raw:
            pattern_list.extend(
                pattern.strip()
                for pattern in str(operator_arg).split(",")
                if pattern.strip()
            )
        if not pattern_list:
            pattern_list = ["**"]
        framewise_patterns[backend] = pattern_list
    if not framewise_patterns:
        return None
    return framewise_patterns


def _collect_glob_matched_nodes(
    forest: CallTreeForest,
    framewise_patterns: dict[str, list[str]],
) -> list[CallTreeNode]:
    """Return nodes whose backend matches and whose path or name glob-matches."""
    return [
        node
        for roots in forest.values()
        for node, path in iter_call_tree_paths(roots)
        if any(
            parser.torch_operator_pattern_matches(pattern, "/".join(path))
            or parser.torch_operator_pattern_matches(pattern, node.name)
            for pattern in framewise_patterns.get(node.backend or "", [])
        )
    ]


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


def _pmc_group_keys_from_node(node: CallTreeNode) -> list[tuple]:
    pmc_group_keys: list[tuple] = []
    for stats in node.kernels.values():
        pmc_group_keys.extend(stats.pmc_group_keys)
    for child in node.children:
        pmc_group_keys.extend(_pmc_group_keys_from_node(child))
    return pmc_group_keys


def _analyze_ids_from_pmc_group_keys(
    pmc_group_keys: list[tuple],
    slot_to_id: dict[tuple, int],
    labels: str,
) -> list[int]:
    analyze_ids: list[int] = []
    for group_key in pmc_group_keys:
        if group_key not in slot_to_id:
            console_error(
                "ml api trace",
                f"No {labels} analyze dispatch id for pmc group key {group_key}",
            )
        analyze_ids.append(slot_to_id[group_key])
    return analyze_ids


def _intersect_with_user_dispatch_filter(
    operator_dispatch_ids: list[int],
    user_dispatch_filter: list,
    labels: str,
) -> list[int]:
    operator_ids = set(operator_dispatch_ids)
    if not user_dispatch_filter:
        selected_ids = operator_ids
    else:
        first_filter = str(user_dispatch_filter[0]).strip()
        if first_filter.startswith(">"):
            threshold = int(first_filter[1:].strip())
            selected_ids = {
                dispatch_id for dispatch_id in operator_ids if dispatch_id > threshold
            }
        else:
            selected_ids = operator_ids & {
                int(dispatch_id) for dispatch_id in user_dispatch_filter
            }
    if not selected_ids:
        console_error(
            "ml api trace",
            f"No {labels} dispatches matched the -d filter: {user_dispatch_filter}",
        )
    return sorted(selected_ids)


def _apply_sliced_kernel_top(
    workload: schema.Workload,
    kernel_top_df: pd.DataFrame,
    args: argparse.Namespace,
    labels: str,
) -> None:
    name_to_id = {
        str(kernel_name).strip(): idx
        for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
    }
    for roots in workload.ml_api_call_trees.values():
        for root in roots:
            _assign_kernel_ids_from_top(root, name_to_id)

    if not workload.filter_kernel_ids:
        return

    file_io.validate_kernel_filter_ids(
        workload.filter_kernel_ids,
        len(kernel_top_df["Kernel_Name"]),
    )
    selected_names = {
        str(kernel_top_df.loc[kernel_id, "Kernel_Name"]).strip()
        for kernel_id in workload.filter_kernel_ids
    }
    dispatch_names = {
        int(dispatch_id): str(kernel_name).strip()
        for dispatch_id, kernel_name in zip(
            workload.raw_pmc["Dispatch_ID"].tolist(),
            workload.raw_pmc["Kernel_Name"].tolist(),
        )
    }
    workload.filter_dispatch_ids = [
        dispatch_id
        for dispatch_id in workload.filter_dispatch_ids
        if dispatch_names.get(int(dispatch_id)) in selected_names
    ]
    if not workload.filter_dispatch_ids:
        console_error(
            "ml api trace",
            f"No {labels} operators matched the -k filter: {args.gpu_kernel}",
        )


class cli_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------

    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        args = self.get_args()

        active_operator_filters = [
            framework_name
            for framework_name, framework_flags in _ML_API_ANALYSIS_CLI_OPTIONS.items()
            if getattr(args, framework_flags["filter_attr"], None) is not None
        ]

        active_operator_lists = [
            framework_name
            for framework_name, framework_flags in _ML_API_ANALYSIS_CLI_OPTIONS.items()
            if getattr(args, framework_flags["list_attr"], False)
        ]

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
                labels = ", ".join(
                    _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"]
                    for backend in active_operator_filters
                )
                workload.raw_pmc, slot_to_id = file_io.create_df_pmc_with_slot_ids(
                    path_info[0],
                    args.verbose,
                )
                operator_dispatch_ids = _analyze_ids_from_pmc_group_keys(
                    pmc_group_keys, slot_to_id, labels
                )
                workload.filter_dispatch_ids = _intersect_with_user_dispatch_filter(
                    operator_dispatch_ids,
                    workload.filter_dispatch_ids,
                    labels,
                )
                console_log(
                    "ml api trace",
                    f"{labels} operator filter selected "
                    f"{len(workload.filter_dispatch_ids)} dispatch(es) "
                    "for metric analysis.",
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
                _apply_sliced_kernel_top(workload, kernel_top_df, args, labels)

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

        active_operator_filters = [
            framework_name
            for framework_name, framework_flags in _ML_API_ANALYSIS_CLI_OPTIONS.items()
            if getattr(args, framework_flags["filter_attr"], None) is not None
        ]
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
        self,
        workload_path: str,
        kernel_top_df: pd.DataFrame,
        backends: list[str],
    ) -> None:
        """Render the operator call tree for the given backends."""
        forest_view = filter_forest_by_backends(
            self._runs[workload_path].ml_api_call_trees, backends
        )
        name_to_id = {
            str(kernel_name).strip(): idx
            for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
        }
        for roots in forest_view.values():
            for root in roots:
                _assign_kernel_ids_from_top(root, name_to_id)
        framework_labels = [
            _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"] for backend in backends
        ]
        tty.list_ml_operators(
            workload_path, forest_view, framework_labels=framework_labels
        )

    def apply_operator_filter(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        backends: list[str],
    ) -> list[tuple]:
        """Glob-filter the operator forest and return pmc group keys on the subtree."""
        framewise_patterns = parse_operator_patterns(args, backends)
        if framewise_patterns is None:
            return []
        labels = ", ".join(
            _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"] for backend in backends
        )
        confined_forest = filter_forest_by_backends(
            workload.ml_api_call_trees, list(framewise_patterns)
        )
        matched_nodes = _collect_glob_matched_nodes(confined_forest, framewise_patterns)
        if not matched_nodes:
            all_patterns = ", ".join(
                pattern
                for patterns in framewise_patterns.values()
                for pattern in patterns
            )
            console_error(
                "ml api trace",
                f"No {labels} operators matched the pattern(s): {all_patterns}",
            )
        workload.ml_api_call_trees = copy_matched_operator_subtree(
            confined_forest, matched_nodes
        )

        pmc_group_keys: list[tuple] = []
        for roots in workload.ml_api_call_trees.values():
            for root in roots:
                pmc_group_keys.extend(_pmc_group_keys_from_node(root))
        if not pmc_group_keys:
            console_error(
                "ml api trace",
                f"No {labels} kernel dispatches on the matched operators",
            )
        return pmc_group_keys

    def handle_operator(
        self, args: argparse.Namespace, workload: schema.Workload, backends: list[str]
    ) -> None:
        """Display the matched operator call tree."""
        if not workload.ml_api_call_trees:
            return
        framewise_patterns = parse_operator_patterns(args, backends)
        patterns_str = ", ".join(
            pattern
            for patterns in (framewise_patterns or {}).values()
            for pattern in patterns
        )
        labels = ", ".join(
            _ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"] for backend in backends
        )
        subtree = workload.ml_api_call_trees
        print(f"\n{'=' * 80}")
        print(f"Matched {labels} Operators: {patterns_str}")
        print("Sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        for roots in subtree.values():
            for root in roots:
                rollup_node_stats(root)
        tty.show_call_tree(subtree)
        tty.show_operator_summary(build_operator_summary(subtree))
        print(f"{'=' * 80}")
