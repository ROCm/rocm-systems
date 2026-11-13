# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from membw_analysis.engine import run_membw_analysis
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import calc_ai_analyze
from utils.utils_analysis import (
    CallTreeNode,
    bind_operator_patterns,
    build_operator_summary,
    check_backends_traced,
    copy_matched_operator_subtree,
    filter_forest_by_backends,
    iter_call_tree_paths,
    process_ml_api_trace_output,
    rollup_node_stats,
    traced_frameworks_from_config,
)
from utils.utils_common import selects_roofline, validate_roofline_csv

# Display names for the backends in operator headings and messages.
_OPERATOR_LABELS = {"torch": "PyTorch", "triton": "Triton", "user": "User"}


def _labels_for_backends(backends: List[str]) -> str:
    """Join backend display names."""
    if not backends:
        return "ML API"
    return ", ".join(_OPERATOR_LABELS.get(backend, backend) for backend in backends)


def _operator_pattern_text(operator_patterns: Dict[str, List[str]]) -> str:
    """Join the patterns for messages, listing each pattern once."""
    return ", ".join(
        dict.fromkeys(
            pattern for patterns in operator_patterns.values() for pattern in patterns
        )
    )


def _traced_frameworks(workload_path: str) -> List[str]:
    """Read the frameworks that the workload traced."""
    return traced_frameworks_from_config(file_io.load_profiling_config(workload_path))


def _check_listed_backends(backends: List[str], workload_path: str) -> None:
    """Exit with an error when the workload did not trace a listed backend."""
    try:
        check_backends_traced(backends, _traced_frameworks(workload_path))
    except ValueError as error:
        console_error("ml api trace", f"{workload_path}: {error}")


def _operator_patterns_for_workload(
    patterns: List[str], workload_path: str
) -> Dict[str, List[str]]:
    """Group --operator patterns by backend for this workload, or exit."""
    try:
        return bind_operator_patterns(patterns, _traced_frameworks(workload_path))
    except ValueError as error:
        console_error("ml api trace", f"{workload_path}: {error}")
        raise


def _collect_glob_matched_nodes(
    forest: dict[str, list[CallTreeNode]],
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

            iteration_multiplexing = self._profiling_config.get(
                "iteration_multiplexing"
            )

            # --list-operators wins over --operator, so listing ends this path.
            if args.list_operators is not None:
                _check_listed_backends(args.list_operators, path_info[0])
                process_ml_api_trace_output(workload, path_info[0])
                workload.raw_pmc = file_io.create_df_pmc(
                    path_info[0],
                    args.verbose,
                )
                if iteration_multiplexing is not None:
                    workload.raw_pmc = self.iteration_multiplex_impute_counters(
                        workload.raw_pmc,
                        policy=iteration_multiplexing,
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
                self.list_operators(path_info[0], kernel_top_df, args.list_operators)
                continue

            if args.operator is not None:
                workload.operator_patterns = _operator_patterns_for_workload(
                    args.operator, path_info[0]
                )
                process_ml_api_trace_output(workload, path_info[0])
                pmc_group_keys = self.apply_operator_filter(workload, path_info[0])
                labels = _labels_for_backends(list(workload.operator_patterns))
                (
                    workload.raw_pmc,
                    slot_to_id,
                    workload.pmc_pass_layout,
                ) = file_io.create_df_pmc_with_slot_ids(
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
                # create 'mega dataframe' (+ PassLayout for same-pass binding)
                workload.raw_pmc, workload.pmc_pass_layout = file_io.load_df_pmc(
                    path_info[0],
                    args.verbose,
                    preserve_pass_provenance=iteration_multiplexing is None,
                )

            if iteration_multiplexing is not None:
                workload.raw_pmc = self.iteration_multiplex_impute_counters(
                    workload.raw_pmc,
                    policy=iteration_multiplexing,
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

            if args.operator is not None:
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

        if args.list_operators is not None:
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

        # Listing exits in pre_processing, so this runs only for a filter.
        if args.operator is not None:
            for path_info in args.path:
                self.handle_operator(self._runs[path_info[0]])

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

                        pmc_df = parser.apply_filters(
                            workload, debug=args.debug_metric_expression
                        )
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
                elif selects_roofline(args.filter_metrics):
                    console_warning(
                        "roofline",
                        f"{gpu_arch} does not support the roofline benchmark, so "
                        "analyze shows no roofline data.",
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
        tty.list_ml_operators(
            workload_path,
            forest_view,
            framework_labels=[_labels_for_backends(backends)],
        )

    def apply_operator_filter(
        self, workload: schema.Workload, workload_path: str
    ) -> List[Tuple]:
        """Keep the operators matching workload.operator_patterns; return pmc keys."""
        labels = _labels_for_backends(list(workload.operator_patterns))
        matched_nodes = _collect_glob_matched_nodes(
            workload.ml_api_call_trees, workload.operator_patterns
        )
        if not matched_nodes:
            console_error(
                "ml api trace",
                f"{workload_path}: No {labels} operators matched the "
                f"pattern(s): {_operator_pattern_text(workload.operator_patterns)}",
            )
        workload.ml_api_call_trees = copy_matched_operator_subtree(
            workload.ml_api_call_trees, matched_nodes
        )

        pmc_group_keys: List[Tuple] = list(
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
        return pmc_group_keys

    def handle_operator(self, workload: schema.Workload) -> None:
        """Display the matched operator call tree."""
        if not workload.ml_api_call_trees:
            return
        labels = _labels_for_backends(list(workload.operator_patterns))
        subtree = workload.ml_api_call_trees
        print(f"\n{'=' * 80}")
        print(
            f"Matched {labels} Operators: "
            f"{_operator_pattern_text(workload.operator_patterns)}"
        )
        print("Sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        for roots in subtree.values():
            for root in roots:
                rollup_node_stats(root)
        tty.show_call_tree(subtree)
        tty.show_operator_summary(build_operator_summary(subtree))
        print(f"{'=' * 80}")
