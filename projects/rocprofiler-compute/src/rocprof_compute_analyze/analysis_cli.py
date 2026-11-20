# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from membw_analysis.engine import (
    evaluate_membw_dfs,
    load_membw_spec,
    run_membw_analysis,
)
from membw_analysis.models import MEMBW_TABLE_IDS
from memory_chart.mem_chart_html import chart_view, mem_chart_document
from memory_chart.render import format_mem_chart_heading
from memory_chart.units import panel_units
from ml_api_tracing.analyze.orchestrator import (
    apply_operator_filter,
    apply_sliced_kernel_top,
    process_ml_api_trace_output,
    select_operator_dispatches,
    validate_operator_trace_options,
)
from ml_api_tracing.analyze.output import list_operators, show_matched_operator_report
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from roofline.roofline_main import ROOFLINE_SUPPORTED, Roofline
from utils import file_io, parser, schema, tty
from utils.html_report.report_shell import write_report
from utils.logger import console_log, console_warning, demarcate
from utils.mi_gpu_spec import mi_gpu_specs
from utils.roofline_calc import ROOFLINE_TABLE_IDS, calc_ai_analyze
from utils.utils_analysis import impute_counters_iteration_multiplex
from utils.utils_common import selects_roofline, validate_roofline_csv

# The device the HTML reports describe, which their file names show.
_REPORT_DEVICE_ID = 0


class _MemChart(NamedTuple):
    """The memory chart the terminal draws: its panel, tables and values."""

    panel_id: int
    panel: Dict[str, Any]
    table_ids: List[int]
    values: Dict[str, Any]


class cli_analysis(OmniAnalyze_Base):
    # -----------------------
    # Required child methods
    # -----------------------

    @demarcate
    def pre_processing(self) -> None:
        """Perform any pre-processing steps prior to analysis."""
        super().pre_processing()
        args = self.get_args()

        listing_requested = args.list_operators is not None
        filter_requested = args.operator is not None and not listing_requested

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
                profiling_config = (
                    self.get_profiling_config()
                    if path_info[0] == args.path[0][0]
                    else file_io.load_profiling_config(path_info[0])
                )
                bound_patterns = validate_operator_trace_options(
                    path_info[0], args.list_operators, args.operator, profiling_config
                )
                if filter_requested:
                    workload.operator_patterns = bound_patterns
                process_ml_api_trace_output(workload, path_info[0])

            if filter_requested:
                pmc_group_keys = apply_operator_filter(workload, path_info[0])
                workload.raw_pmc, slot_to_id, workload.pass_tables = (
                    file_io.create_df_pmc_with_slot_ids(path_info[0], args.verbose)
                )
                select_operator_dispatches(
                    workload,
                    pmc_group_keys,
                    slot_to_id,
                    list(workload.operator_patterns),
                )
            else:
                workload.raw_pmc, workload.pass_tables = file_io.create_df_pmc(
                    path_info[0], args.verbose
                )

            if self._profiling_config.get("iteration_multiplexing"):
                workload.raw_pmc, workload.pass_tables = (
                    impute_counters_iteration_multiplex(
                        workload.raw_pmc,
                        workload_dir=Path(path_info[0]),
                    )
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

            if listing_requested:
                list_operators(
                    path_info[0], workload, kernel_top_df, args.list_operators
                )
                continue

            if filter_requested:
                apply_sliced_kernel_top(
                    workload, kernel_top_df, args, list(workload.operator_patterns)
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

        if listing_requested:
            sys.exit(0)

    def _roofline(
        self, args: argparse.Namespace, workload_path: str, gpu_arch: str
    ) -> Optional[Roofline]:
        """The roofline report to write for the workload, or None (with the reason
        logged)."""
        if gpu_arch not in ROOFLINE_SUPPORTED:
            if selects_roofline(args.filter_metrics):
                console_warning(
                    "roofline",
                    f"{gpu_arch} does not support the roofline benchmark, so "
                    "analyze shows no roofline data.",
                )
            return None
        is_roofline_valid, roofline_error_msg = validate_roofline_csv(
            Path(workload_path)
        )
        soc = self.get_socs()
        if not soc or gpu_arch not in soc:
            console_warning(
                "roofline",
                f"Skipping roofline charting: gpu arch {gpu_arch} not in soc {soc}",
            )
            return None
        if not is_roofline_valid:
            console_warning(
                "roofline",
                "Skipping roofline charting: Invalid roofline.csv: "
                f"{roofline_error_msg}",
            )
            return None
        soc_obj = soc[gpu_arch]
        return Roofline(
            args=soc_obj.get_args(),
            mspec=soc_obj._mspec,
            run_parameters={
                "workload_dir": workload_path,
                "device_id": _REPORT_DEVICE_ID,
                "gpu_arch": gpu_arch,
                "iteration_multiplexing": self._profiling_config.get(
                    "iteration_multiplexing"
                ),
                "matrix_ops_type": mi_gpu_specs.get_roofline_matrix_type(gpu_arch),
            },
        )

    def _mem_chart(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        arch_config: schema.ArchConfig,
    ) -> Optional[_MemChart]:
        """The memory chart the terminal draws, or None when it draws none."""
        for panel_id, panel in arch_config.panel_configs.items():
            table_ids = tty.mem_chart_tables(
                args, self._runs, panel_id, panel, self._profiling_config
            )
            values = tty.mem_chart_values(workload.dfs, table_ids)
            if values:
                return _MemChart(panel_id, panel, table_ids, values)
        return None

    def _write_reports(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        gpu_arch: str,
        arch_config: schema.ArchConfig,
    ) -> None:
        """Write the roofline and memory chart pages, evaluating each kernel's tables
        once for both."""
        roofline = self._roofline(args, workload_path, gpu_arch)
        mem_chart = self._mem_chart(args, workload, arch_config)
        table_ids = list(ROOFLINE_TABLE_IDS) if roofline else []
        if mem_chart:
            table_ids += mem_chart.table_ids
            if workload.membw_result is not None:
                table_ids += [t for t in MEMBW_TABLE_IDS if t in arch_config.dfs]
        kernel_tables: Dict[int, parser.KernelTables] = {}
        if table_ids:
            workload.path = workload_path
            pmc_df = parser.apply_filters(workload, debug=args.debug_metric_expression)
            kernel_tables = parser.evaluate_kernel_tables(
                arch_config, workload, pmc_df, table_ids
            )
        if roofline:
            ai_data = calc_ai_analyze(workload=workload, kernel_tables=kernel_tables)
            ops_fig, flops_fig, ops_dt, flops_dt = roofline.construct_plotly_figures(
                ai_data=ai_data
            )
            workload.roofline_html_path = roofline.save_html_files(
                ops_fig, flops_fig, ops_dt, flops_dt
            )
        if mem_chart:
            self._write_mem_chart_html(
                args, workload, workload_path, gpu_arch, mem_chart, kernel_tables
            )

    def _write_mem_chart_html(
        self,
        args: argparse.Namespace,
        workload: schema.Workload,
        workload_path: str,
        gpu_arch: str,
        mem_chart: _MemChart,
        kernel_tables: Dict[int, parser.KernelTables],
    ) -> None:
        """Write the memory chart page: the chart the terminal shows, and the same
        chart for each kernel."""
        units = panel_units(mem_chart.panel)
        spec = load_membw_spec(gpu_arch) if workload.membw_result is not None else None
        kernels = [
            (
                name,
                chart_view(
                    gpu_arch,
                    tty.mem_chart_values(tables, mem_chart.table_ids),
                    units,
                    evaluate_membw_dfs(tables, spec, gpu_arch) if spec else None,
                ),
            )
            for name, tables in kernel_tables.values()
        ]
        aggregate = chart_view(gpu_arch, mem_chart.values, units, workload.membw_result)
        heading = format_mem_chart_heading(
            args.normal_unit, panel_id=mem_chart.panel_id
        )
        workload.mem_chart_html_path = write_report(
            workload_path,
            "mem_chart",
            _REPORT_DEVICE_ID,
            args.gpu_kernel,
            mem_chart_document(gpu_arch, heading, mem_chart.panel, aggregate, kernels),
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

        if args.operator is not None and args.list_operators is None:
            for path_info in args.path:
                show_matched_operator_report(self._runs[path_info[0]])

        if args.list_stats:
            tty.show_kernel_stats(
                args,
                self._runs,
                arch_config,
                self._output,
            )
        else:
            if len(args.path) == 1:
                self._write_reports(
                    args, workload, workload_path, gpu_arch, arch_config
                )

            tty.show_all(
                args,
                self._runs,
                arch_config,
                self._output,
                self._profiling_config,
            )
