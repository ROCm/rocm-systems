# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, TextIO

import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_analyze.analysis_db import pc_sampling_dispatch_trace
from rocprof_compute_analyze.layout_writer import write_ml_api_trace_csv
from rocprof_compute_analyze.output_location import (
    REPORT_FILE_NAME,
    workload_folder_name,
)
from rocprof_compute_analyze.terminal_summary import print_terminal_summary
from utils import file_io, parser, schema, tty
from utils.file_io import load_pc_sampling_results
from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.logger import console_error, demarcate
from utils.terminal_selection import TerminalSelection, build_terminal_selection
from utils.utils_analysis import (
    CallTreeNode,
    build_operator_summary,
    copy_matched_operator_subtree,
    filter_forest_by_backends,
    iter_call_tree_paths,
    process_ml_api_trace_output,
    rollup_node_stats,
)

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


def _kernel_ids_by_name(kernel_names: list[str]) -> dict[str, int]:
    """Map each name of a full kernel ranking to its position, the -k id."""
    return {
        str(kernel_name).strip(): idx for idx, kernel_name in enumerate(kernel_names)
    }


class cli_analysis(OmniAnalyze_Base):
    """The report view: operator trees, the terminal summary and report.txt.

    Every table and value comes from the analysis DB. This class reads the ML API
    trace, builds the operator forest, and renders the frames the DB built.
    """

    # -----------------------
    # Required child methods
    # -----------------------

    @demarcate
    def pre_processing(self) -> None:
        """Load the operator view inputs, or the listing's kernel stats."""
        super().pre_processing()
        args = self.get_args()
        self._report_runs: dict[str, schema.Workload] = {}
        self._kernel_rankings: dict[str, list[str]] = {}
        self._terminal_selection = TerminalSelection()

        listing_backends = [
            backend
            for backend, flags in _ML_API_ANALYSIS_CLI_OPTIONS.items()
            if getattr(args, flags["list_attr"], False)
        ]
        if args.list_stats or listing_backends:
            self._prepare_listing(listing_backends)
            return

        self._terminal_selection = build_terminal_selection(
            args.filter_metrics, self._first_gpu_arch()
        )
        backends = self._operator_filter_backends()
        for path_info in args.path:
            workload_path = path_info[0]
            workload = self._runs[workload_path]
            if self._ml_api_trace_profiled():
                process_ml_api_trace_output(workload, workload_path)
            if backends:
                self.apply_operator_filter(args, workload, workload_path, backends)

    @demarcate
    def show_listing(self) -> None:
        """Print the --list-stats listing to the terminal; write no files."""
        args = self.get_args()
        tty.show_kernel_stats(
            args,
            self._runs,
            self._first_arch_config(),
            sys.stdout,
        )

    @demarcate
    def write_outputs(self, run_folder: Path) -> None:
        """Write the ML API traces and report files, then print the terminal summary.

        Every table comes from the report frames the analysis DB built. The
        terminal shows the standard sections and the -b selection; the report
        files have every section.
        """
        super().write_outputs(run_folder)
        self._write_ml_api_traces(run_folder)
        self._write_reports(run_folder)
        self._show_operator_view(sys.stdout, self._workload_paths())
        tty.show_all(
            self.get_args(),
            self._report_runs,
            self._first_arch_config(),
            sys.stdout,
            self._profiling_config,
            terminal_selection=self._terminal_selection,
        )
        tty.print_roofline_html_path(self._report_runs, sys.stdout)
        print_terminal_summary(run_folder, self._workload_paths(), sys.stdout)

    def set_report_runs(
        self,
        report_runs: dict[str, schema.Workload],
        kernel_rankings: dict[str, list[str]],
    ) -> None:
        """Take the report frames the analysis DB built, one per -p workload.

        kernel_rankings holds each workload's kernel names before the -k filter,
        in the order the -k ids index them.
        """
        self._report_runs = report_runs
        self._kernel_rankings = kernel_rankings

    def _prepare_listing(self, listing_backends: list[str]) -> None:
        """Build the Top Kernels and Dispatch List of each workload for a listing.

        A listing reads the same counters as a run, but writes nothing. A list
        of operators ends the process after printing its tree.
        """
        args = self.get_args()
        for path_info in args.path:
            workload_path = path_info[0]
            workload = self._runs[workload_path]
            if listing_backends:
                process_ml_api_trace_output(workload, workload_path)
            workload.raw_pmc = self._listing_counters(workload_path)
            if self._profiling_config.get("iteration_multiplexing") is not None:
                workload.raw_pmc = self.iteration_multiplex_impute_counters(
                    workload.raw_pmc,
                    policy=self._profiling_config["iteration_multiplexing"],
                    workload_dir=Path(workload_path),
                )
            kernel_top_df, dispatch_info_df = file_io.create_df_kernel_top_stats(
                df_in=workload.raw_pmc,
                filter_gpu_ids=workload.filter_gpu_ids,
                filter_dispatch_ids=workload.filter_dispatch_ids,
                time_unit=args.time_unit,
            )
            workload.dfs[parser.PMC_KERNEL_TOP_TABLE_ID] = kernel_top_df
            workload.dfs[parser.PMC_DISPATCH_INFO_TABLE_ID] = dispatch_info_df
            if listing_backends:
                self.list_operators(workload_path, kernel_top_df, listing_backends)
        if listing_backends:
            sys.exit(0)

    def _listing_counters(self, workload_path: str) -> pd.DataFrame:
        """Return the dispatch rows a listing shows: the counters, or the PC trace
        when the profile has no counters."""
        if self.pc_sampling_only():
            return pc_sampling_dispatch_trace(load_pc_sampling_results(workload_path))
        return file_io.create_df_pmc(workload_path, self.get_args().verbose)

    def _first_gpu_arch(self) -> str:
        """Return the GPU arch of the first -p workload, which drives rendering."""
        workload = self._runs[self.get_args().path[0][0]]
        return workload.sys_info.iloc[0]["gpu_arch"]

    def _first_arch_config(self) -> schema.ArchConfig:
        """Return the arch config of the first -p workload, which drives rendering."""
        return self._arch_config_of(self._workload_paths()[0])

    def _arch_config_of(self, workload_path: str) -> schema.ArchConfig:
        """Return the arch config of one -p workload's GPU."""
        gpu_arch = self._runs[workload_path].sys_info.iloc[0]["gpu_arch"]
        return self._arch_configs[gpu_arch]

    def _workload_paths(self) -> List[str]:
        """Return the -p workload paths, in the order they were given."""
        return [path_info[0] for path_info in self.get_args().path]

    def _ml_api_trace_profiled(self) -> bool:
        """True when the profile recorded an ML API trace for any backend."""
        config = self._profiling_config
        return bool(config.get("ml_api_trace")) or any(
            config.get(f"{backend}_trace", False) for backend in KNOWN_ML_API_BACKENDS
        )

    def _operator_filter_backends(self) -> list[str]:
        """Return the backends that have an operator filter flag."""
        args = self.get_args()
        return [
            backend
            for backend, flags in _ML_API_ANALYSIS_CLI_OPTIONS.items()
            if getattr(args, flags["filter_attr"], None) is not None
        ]

    def _write_ml_api_traces(self, run_folder: Path) -> None:
        """Save each traced workload's ML API trace in its run-folder subfolder."""
        for position, workload_path in enumerate(self._workload_paths(), start=1):
            workload = self._runs[workload_path]
            if not workload.ml_api_trace_pairs:
                continue
            workload_folder = run_folder / workload_folder_name(
                position, Path(workload_path)
            )
            write_ml_api_trace_csv(workload_folder, workload.ml_api_trace_df)

    def _write_reports(self, run_folder: Path) -> None:
        """Write report.txt at the run root, and per workload with several -p.

        The root report covers every -p workload, as a comparison when there are
        several. Each workload folder's report covers that workload alone.
        """
        workload_paths = self._workload_paths()
        # A comparison narrows the report frames in place to the metrics every
        # run shares, so each workload's own report is written before it.
        if len(workload_paths) > 1:
            for position, workload_path in enumerate(workload_paths, start=1):
                self._write_report_file(
                    run_folder
                    / workload_folder_name(position, Path(workload_path))
                    / REPORT_FILE_NAME,
                    {workload_path: self._report_runs[workload_path]},
                    self._arch_config_of(workload_path),
                    [workload_path],
                )
        self._write_report_file(
            run_folder / REPORT_FILE_NAME,
            self._report_runs,
            self._first_arch_config(),
            workload_paths,
        )

    def _write_report_file(
        self,
        report_path: Path,
        report_runs: Dict[str, schema.Workload],
        arch_config: schema.ArchConfig,
        workload_paths: List[str],
    ) -> None:
        """Write the given runs' sections and their operator view to report_path."""
        with report_path.open("w", encoding="utf-8") as report:
            tty.show_all(
                self.get_args(),
                report_runs,
                arch_config,
                report,
                self._profiling_config,
            )
            self._show_operator_view(report, workload_paths)

    def _show_operator_view(self, output: TextIO, workload_paths: List[str]) -> None:
        """Print the matched operator trees of the workloads, with their kernel rows."""
        backends = self._operator_filter_backends()
        if not backends:
            return
        args = self.get_args()
        with contextlib.redirect_stdout(output):
            for workload_path in workload_paths:
                report_run = self._report_runs[workload_path]
                self.handle_operator(
                    args,
                    self._runs[workload_path],
                    backends,
                    self._kernel_rankings[workload_path],
                    report_run.dfs[parser.PMC_KERNEL_TOP_TABLE_ID],
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
        name_to_id = _kernel_ids_by_name(kernel_top_df["Kernel_Name"].tolist())
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
    ) -> None:
        """Glob-filter the operator forest to the operator view.

        The filter selects what the report shows. It does not change the
        calculation, so every kernel and dispatch stays in the analysis DB.
        """
        framewise_patterns = parse_operator_patterns(args, backends)
        if framewise_patterns is None:
            return
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

    def handle_operator(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        backends: list[str],
        kernel_ranking: list[str],
        report_kernel_df: pd.DataFrame,
    ) -> None:
        """Display the matched operator call tree, then the rows of the kernels it
        launched. The ids are the full ranking's, the ids -k takes. The kernel rows
        are labelled and come from the run's Top Kernels; no other table is narrowed.
        """
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
        name_to_id = _kernel_ids_by_name(kernel_ranking)
        kernel_ids: set[int] = set()
        for roots in subtree.values():
            for root in roots:
                kernel_ids.update(_assign_kernel_ids_from_top(root, name_to_id))
                rollup_node_stats(root)
        print(f"\n{'=' * 80}")
        print(f"Matched {labels} Operators: {patterns_str}")
        print("Sorted by total GPU kernel duration.")
        print(f"{'=' * 80}")
        tty.show_call_tree(subtree)
        tty.show_operator_summary(build_operator_summary(subtree))
        print(f"{'=' * 80}")
        print(f"Kernels launched by the matched {labels} operators (labelled rows)")
        names_by_id = {kernel_id: name for name, kernel_id in name_to_id.items()}
        report_names = report_kernel_df["Kernel_Name"].astype(str).str.strip()
        matched_rows = report_kernel_df[
            report_names.isin({names_by_id[kernel_id] for kernel_id in kernel_ids})
        ]
        if matched_rows.empty:
            print("No kernels launched by the matched operators are in this run.")
        else:
            print(
                tty.get_table_string(
                    matched_rows, transpose=False, decimal=args.decimal
                )
            )
        shown_names = set(report_names)
        left_out_ids = sorted(
            kernel_id
            for kernel_id in kernel_ids
            if names_by_id[kernel_id] not in shown_names
        )
        if left_out_ids:
            print(
                "Not in this run, left out by -k: kernel id(s) "
                + ", ".join(str(kernel_id) for kernel_id in left_out_ids)
            )
        print(f"{'=' * 80}")
