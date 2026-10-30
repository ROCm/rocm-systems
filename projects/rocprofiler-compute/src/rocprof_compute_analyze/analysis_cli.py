# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from membw_analysis.engine import run_membw_analysis
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.ml_api_selectors import (
    OPERATOR_LABELS,
    OPERATOR_SELECTORS,
    operator_patterns_for_traced_frameworks,
)
from utils.roofline_calc import calc_ai_analyze
from utils.utils_analysis import (
    CallTreeNode,
    build_operator_summary,
    copy_matched_operator_subtree,
    iter_call_tree_paths,
    process_ml_api_trace_output,
    rollup_node_stats,
)
from utils.utils_common import validate_roofline_csv


def _labels_for_backends(backends: List[str]) -> str:
    """Join backend display names."""
    if not backends:
        return "ML API"
    return ", ".join(OPERATOR_LABELS.get(backend, backend) for backend in backends)


def _restrict_patterns_to_operator_backends(
    selections: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Restrict pattern lists to keys in OPERATOR_SELECTORS."""
    return {
        backend: list(patterns)
        for backend, patterns in selections.items()
        if backend in OPERATOR_SELECTORS
    }


def _bind_operator_patterns_for_workload(
    operator_patterns: Dict[str, List[str]],
    workload_path: str,
) -> Dict[str, List[str]]:
    """Bind unscoped --operator patterns using this workload's yaml."""
    profiling_config = file_io.load_profiling_config(workload_path)
    trace_ml_api = profiling_config.get("trace_ml_api")
    traced_frameworks = [str(name) for name in trace_ml_api] if trace_ml_api else []
    return _restrict_patterns_to_operator_backends(
        operator_patterns_for_traced_frameworks(operator_patterns, traced_frameworks)
    )


def _join_unique_operator_patterns(
    operator_patterns_by_backend: Dict[str, List[str]],
) -> str:
    """Join unique operator patterns."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for patterns in operator_patterns_by_backend.values():
        for pattern in patterns:
            if pattern in seen:
                continue
            seen.add(pattern)
            ordered.append(pattern)
    return ", ".join(ordered)


def _collect_nodes_matching_operator_patterns(
    forest: Dict[str, List[CallTreeNode]],
    operator_patterns_by_backend: Dict[str, List[str]],
) -> List[CallTreeNode]:
    """Collect nodes whose name or path matches a pattern for that node's backend."""
    return [
        node
        for roots in forest.values()
        for node, path in iter_call_tree_paths(roots)
        if any(
            parser.torch_operator_pattern_matches(pattern, "/".join(path))
            or parser.torch_operator_pattern_matches(pattern, node.name)
            for pattern in operator_patterns_by_backend.get(node.backend or "", [])
        )
    ]


def _assign_kernel_ids_from_top(
    node: CallTreeNode, name_to_id: dict[str, int]
) -> set[int]:
    """Set kernel ids on this subtree and return them."""
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

        list_selectors = getattr(args, "list_operators", None)
        operator_patterns = getattr(args, "operator_patterns", None)
        listing_requested = list_selectors is not None
        filter_requested = operator_patterns is not None and not listing_requested

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

            if listing_requested or filter_requested:
                process_ml_api_trace_output(workload, path_info[0])

            if listing_requested:
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
                self.list_operators(path_info[0], kernel_top_df, list_selectors)
                continue

            if filter_requested:
                pmc_group_keys, operator_patterns_by_backend = (
                    self.apply_operator_filter(args, workload, path_info[0])
                )
                labels = _labels_for_backends(list(operator_patterns_by_backend))
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

            if filter_requested:
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

        if listing_requested:
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

        list_selectors = getattr(args, "list_operators", None)
        operator_patterns = getattr(args, "operator_patterns", None)
        if operator_patterns is not None and list_selectors is None:
            for path_info in args.path:
                self.handle_operator(args, self._runs[path_info[0]], path_info[0])

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
        operator_forest = self._runs[workload_path].ml_api_call_trees
        if backends:
            nodes_for_listed_backends = [
                node
                for roots in operator_forest.values()
                for node, _path in iter_call_tree_paths(roots)
                if node.backend in backends
            ]
            operator_forest = copy_matched_operator_subtree(
                operator_forest, nodes_for_listed_backends
            )
        name_to_id = {
            str(kernel_name).strip(): idx
            for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
        }
        for roots in operator_forest.values():
            for root in roots:
                _assign_kernel_ids_from_top(root, name_to_id)
        framework_labels = (
            [OPERATOR_LABELS.get(backend, backend) for backend in backends]
            if backends
            else ["ML API"]
        )
        tty.list_ml_operators(
            workload_path, operator_forest, framework_labels=framework_labels
        )

    def apply_operator_filter(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
    ) -> Tuple[List[Tuple], Dict[str, List[str]]]:
        """Filter the operator forest and return pmc keys plus the bound map."""
        operator_patterns = getattr(args, "operator_patterns", None)
        if operator_patterns is None:
            return [], {}
        try:
            operator_patterns_by_backend = _bind_operator_patterns_for_workload(
                operator_patterns, workload_path
            )
        except ValueError as error:
            console_error("ml api trace", f"{workload_path}: {error}")
        labels = _labels_for_backends(list(operator_patterns_by_backend))
        if not operator_patterns_by_backend:
            console_error(
                "ml api trace",
                f"{workload_path}: No {labels} operators matched the pattern(s).",
            )
        nodes_matching_operator_patterns = _collect_nodes_matching_operator_patterns(
            workload.ml_api_call_trees, operator_patterns_by_backend
        )
        if not nodes_matching_operator_patterns:
            operator_pattern_text = _join_unique_operator_patterns(
                operator_patterns_by_backend
            )
            console_error(
                "ml api trace",
                f"{workload_path}: No {labels} operators matched the "
                f"pattern(s): {operator_pattern_text}",
            )
        workload.ml_api_call_trees = copy_matched_operator_subtree(
            workload.ml_api_call_trees, nodes_matching_operator_patterns
        )

        pmc_group_keys: list[tuple] = list(
            dict.fromkeys(
                key
                for roots in workload.ml_api_call_trees.values()
                for root in roots
                for key in _pmc_group_keys_from_node(root)
            )
        )
        if not pmc_group_keys:
            console_error(
                "ml api trace",
                f"{workload_path}: No {labels} kernel dispatches "
                "on the matched operators",
            )
        return pmc_group_keys, operator_patterns_by_backend

    def handle_operator(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
    ) -> None:
        """Display the matched operator call tree."""
        if not workload.ml_api_call_trees:
            return
        operator_patterns = getattr(args, "operator_patterns", None)
        if operator_patterns is None:
            return
        try:
            operator_patterns_by_backend = _bind_operator_patterns_for_workload(
                operator_patterns, workload_path
            )
        except ValueError as error:
            console_error("ml api trace", f"{workload_path}: {error}")
        operator_pattern_text = _join_unique_operator_patterns(
            operator_patterns_by_backend
        )
        labels = _labels_for_backends(list(operator_patterns_by_backend))
        print(f"\n{'=' * 80}")
        print(f"Matched {labels} Operators: {operator_pattern_text}")
        print("Sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        for roots in workload.ml_api_call_trees.values():
            for root in roots:
                rollup_node_stats(root)
        tty.show_call_tree(workload.ml_api_call_trees)
        tty.show_operator_summary(build_operator_summary(workload.ml_api_call_trees))
        print(f"{'=' * 80}")
