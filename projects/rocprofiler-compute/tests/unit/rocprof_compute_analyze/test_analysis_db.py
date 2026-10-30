# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_db.py static methods."""

import copy
import csv
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import common
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from membw_analysis.engine import run_membw_analysis
from membw_analysis.models import BottleneckNode, MemBwAnalysisResult, SupportingMetric
from memory_chart.loader import Layouts
from pc_sampling import (
    per_kernel_isa_export,
    per_source_line_export,
    source_snapshot_analysis,
)
from pc_sampling.code_object_analysis import CodeObjectInstruction, CodeObjectSymbol
from pc_sampling.pc_sampling_analysis import SOURCE_LINE_MISSING, InstructionLineRecord
from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from rocprof_compute_analyze.analysis_db import (
    SourceFrameCollector,
    db_analysis,
    filter_dispatch_frame,
    membw_metric_tables,
    pc_sampling_dispatch_trace,
    report_evaluation_diagnostics,
)
from utils import analysis_orm as orm
from utils import schema
from utils.file_io import create_df_kernel_top_stats
from utils.metrics.expression import build_eval_string
from utils.metrics.noise_clamper import (
    clear_noise_clamp_warnings,
    get_noise_clamp_warnings,
)

ISA_WORKLOAD_NAME = "vector_copy"
ISA_WORKLOAD_SUB_NAME = "run"

VIEW_CSV_FILENAMES = frozenset({
    "kernel.csv",
    "kernel_metric.csv",
    "membw_node.csv",
    "pc_sampling_summary.csv",
    "roofline_ceiling.csv",
    "source_lines.csv",
    "workload_metric.csv",
})


def drain_evaluation_diagnostics():
    """Discard messages left over from earlier evaluate() calls."""
    with patch("rocprof_compute_analyze.analysis_db.console_warning"), patch(
        "rocprof_compute_analyze.analysis_db.console_debug"
    ):
        report_evaluation_diagnostics()


def make_dual_issue_arch_config(metric_name: str, peak_col: str = "Peak"):
    """Build an arch_config with a metric_table carrying one VALU row."""
    metric_df = pd.DataFrame(
        {
            "Metric": [metric_name],
            "Value": ["unused_expression"],
            peak_col: ["unused_peak_expression"],
        },
        index=pd.Index(["1.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {201: metric_df}
    arch_config.dfs_type = {201: "metric_table"}
    return arch_config


def make_roofline_calc_analyzer(workload_path, pmc_df, roofline_df):
    """Build a SimpleNamespace analyzer for calc_roofline_data tests."""
    sys_info_df = pd.DataFrame([{"gpu_arch": "gfx90a"}])
    arch_config = SimpleNamespace(dfs={402: roofline_df})
    return SimpleNamespace(
        _runs={workload_path: SimpleNamespace(sys_info=sys_info_df)},
        _typed_sys_info_per_workload={
            workload_path: sys_info_df.iloc[0].to_dict(),
        },
        _pmc_df_per_workload={workload_path: pmc_df},
        _arch_configs={"gfx90a": arch_config},
        get_args=lambda: SimpleNamespace(),
    )


def write_roofline_csv(workload_dir: Path, csv_text: str) -> None:
    """Write a roofline.csv into a workload directory."""
    (workload_dir / "roofline.csv").write_text(csv_text, encoding="utf-8")


def make_ceiling_analyzer(workload_path: str, gpu_arch: str):
    """Build an analyzer whose only input is the workload's gpu_arch."""
    analyzer = db_analysis(SimpleNamespace(output_name=None), {})
    analyzer._runs = {
        workload_path: schema.Workload(sys_info=pd.DataFrame([{"gpu_arch": gpu_arch}]))
    }
    return analyzer


def make_pct_of_peak_arch_config():
    """Build a metric table with one flagged and one unflagged Percent of Peak row.

    The expressions are constants and one peak variable, so the expected
    percentages are easy to derive.
    """
    metric_df = pd.DataFrame(
        {
            "Metric": ["Flagged metric", "Plain metric"],
            "Avg": ["100.0 / 3", "100.0 / 3"],
            "Peak": ["$FP32Flops_empirical_peak", "$FP32Flops_empirical_peak"],
            "Percent of Peak": [True, False],
        },
        index=pd.Index(["7.1.0", "7.1.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {701: metric_df}
    arch_config.dfs_type = {701: "metric_table"}
    arch_config.panel_configs = {
        700: {
            "id": 700,
            "title": "Wavefront",
            "data source": [
                {"metric_table": {"id": 701, "title": "Wavefront Launch Stats"}}
            ],
        }
    }
    return arch_config


def make_pct_of_peak_analyzer(workload_path: str, sys_info: dict):
    """Build an analyzer whose metric config went through the real load_options path.

    load_options runs build_metric_value_string, which stringifies the
    Percent of Peak flag before calc_metrics_data reads it.
    """
    analyzer = db_analysis(
        SimpleNamespace(normal_unit="per_kernel", gpu_kernel=None, path=[]),
        {},
    )
    analyzer._arch_configs = {"gfx942": make_pct_of_peak_arch_config()}
    analyzer.load_options()
    analyzer._runs = {workload_path: schema.Workload(sys_info=pd.DataFrame([sys_info]))}
    analyzer._pmc_df_per_workload = {
        workload_path: pd.DataFrame({
            "Kernel_Name": ["vecCopy"],
            "SQ_WAVES": [1.0],
            "GRBM_GUI_ACTIVE": [1000.0],
        })
    }
    metrics_info, expressions = analyzer.calc_metrics_data()
    analyzer._metrics_info_data_per_workload = metrics_info
    analyzer._metric_expression_data_per_workload = expressions
    return analyzer


def calc_values_with_stubbed_dual_issue(analyzer):
    """Run calc_expressions with the dual-issue check, which is not under test."""
    with patch.object(db_analysis, "validate_dual_issue_metrics"):
        return analyzer.calc_expressions()


def select_metric_value(values_df, metric_id, value_name):
    """Return the one stored value of a metric, or fail if it is not unique."""
    rows = values_df[
        (values_df["metric_id"] == metric_id) & (values_df["value_name"] == value_name)
    ]
    assert len(rows) == 1
    return rows["value"].iloc[0]


def store_instruction_lines(code_object_store):
    """Every instruction line a code object owns, across all its symbols."""
    return [
        line
        for symbol in code_object_store.kernel_symbols
        for line in symbol.instruction_lines
    ]


def kernel_instruction_lines(kernel):
    """Every instruction line attributed to a kernel, across all its symbols."""
    return [
        line for symbol in kernel.kernel_symbols for line in symbol.instruction_lines
    ]


def make_pc_sampling_dispatch(dispatch_id, kernel_id):
    """Build one PC-sampling kernel dispatch record."""
    return {
        "start_timestamp": 0,
        "end_timestamp": 0,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": 1},
        },
    }


def make_colliding_pc_sampling_tool_data(process_id: int, sample_count: int):
    """Build process-local sampling data that reuses shared display identities."""
    tool_data = make_pc_sampling_tool_data()
    shared_sample = tool_data["buffer_records"]["pc_sample_stochastic"][0]
    tool_data["metadata"]["pid"] = process_id
    tool_data["buffer_records"]["pc_sample_stochastic"] = [
        copy.deepcopy(shared_sample) for _ in range(sample_count)
    ]
    tool_data["buffer_records"]["kernel_dispatch"] = [make_pc_sampling_dispatch(0, 100)]
    tool_data["strings"] = {
        "pc_sample_instructions": ["v_mov"],
        "pc_sample_comments": ["/s/shared.cpp:7"],
    }
    tool_data["kernel_symbols"] = [tool_data["kernel_symbols"][0]]
    return tool_data


def make_source_frame_collector(workload, workload_path="/fake/workload"):
    """Build a collector for tests that call the insert paths directly."""
    return SourceFrameCollector(Path(workload_path), workload)


def make_pc_sampling_database_analyzer(
    tool_data_per_workload,
    filter_gpu_ids=(),
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
    sys_info_row=None,
):
    """Build a database analyzer configured for sampling-only workloads."""
    analyzer = db_analysis(
        SimpleNamespace(),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([
                (
                    dict(sys_info_row)
                    if sys_info_row is not None
                    else {"gpu_arch": "gfx942"}
                )
            ]),
            filter_gpu_ids=list(filter_gpu_ids),
            filter_kernel_ids=list(filter_kernel_ids),
            filter_dispatch_ids=list(filter_dispatch_ids),
        )
        for workload_path in tool_data_per_workload
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._roofline_ceiling_rows_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = tool_data_per_workload
    traces = {
        workload_path: pc_sampling_dispatch_trace(tool_data_records)
        for workload_path, tool_data_records in tool_data_per_workload.items()
    }
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(
            trace, analyzer._runs[workload_path]
        )
        for workload_path, trace in traces.items()
    }
    analyzer._workload_total_duration_per_workload = (
        analyzer.calc_workload_total_durations(traces)
    )
    analyzer._roofline_data_per_kernel = {}
    analyzer._membw_result_per_workload = {}
    return analyzer


def make_pc_sampling_only_database_analyzer(
    workload_path,
    tool_data_records,
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
):
    """Build a database analyzer configured for one sampling-only workload."""
    return make_pc_sampling_database_analyzer(
        {workload_path: tool_data_records},
        filter_kernel_ids=filter_kernel_ids,
        filter_dispatch_ids=filter_dispatch_ids,
    )


def make_counter_backed_database_analyzer(
    workload_path,
    filter_blocks,
    tool_data_records,
):
    """Build a counter-backed analyzer with optional sampling records."""
    analyzer = db_analysis(
        SimpleNamespace(),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
        )
    }
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._roofline_ceiling_rows_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": filter_blocks}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    analyzer._dispatch_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "dispatch_id": 7,
                "kernel_name": "vecCopy",
                "gpu_id": 0,
                "start_timestamp": 10,
                "end_timestamp": 20,
            }
        ])
    }
    # The one dispatch above runs for 10 ns, and no filter drops it.
    analyzer._workload_total_duration_per_workload = {workload_path: 10}
    analyzer._roofline_data_per_kernel = {workload_path: pd.DataFrame()}
    analyzer._membw_result_per_workload = {}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._kernel_values_data_per_workload = {}
    analyzer._workload_values_data_per_workload = {}
    return analyzer


def run_memory_chart_workload(tmp_path, gpu_arch):
    """Run analysis of one counter-backed workload; return its workload row."""
    analyzer = make_counter_backed_database_analyzer(str(tmp_path), ["3"], [])
    analyzer._runs[str(tmp_path)].sys_info = pd.DataFrame([{"gpu_arch": gpu_arch}])
    run_analysis_with_existing_database(analyzer)
    return orm.Database.get_session().query(orm.Workload).one()


def run_analysis_with_existing_database(analyzer):
    """Run analysis while preserving the test's existing database session."""
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.build_database(Path("analysis.db"))


def query_instruction_sample_states(session):
    """Query the sample states of instruction lines, leaving out source lines."""
    return session.query(orm.PCSampleState).filter(
        orm.PCSampleState.instruction_uuid.is_not(None)
    )


def query_instruction_stall_reasons(session):
    """Query the stall-reason counts of instruction lines, leaving out source lines."""
    return (
        session
        .query(orm.PCSampleStallReason)
        .join(orm.PCSampleStallReason.pc_sample_state)
        .filter(orm.PCSampleState.instruction_uuid.is_not(None))
    )


def run_analysis_with_materialized_views(analyzer):
    """Run analysis while materializing views in the existing test database."""
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.build_database(Path("analysis.db"))
    orm.Database.create_views()
    orm.Database.commit()


def cleanup_database() -> None:
    """Close and clear the process-wide test database state."""
    current_session = orm.Database._session
    if current_session is not None:
        current_session.close()
    current_engine = orm.Database._engine
    if current_engine is not None:
        current_engine.dispose()
    orm.Database._session = None
    orm.Database._engine = None


def make_metric_value_analyzer(workload_path):
    """Build an analyzer with one metric per base-unit case and stored values."""
    analyzer = make_counter_backed_database_analyzer(workload_path, ["3"], [])
    analyzer._metrics_info_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "name": "HBM Bandwidth",
                "metric_id": "4.1.9",
                "description": "Scaled bandwidth",
                "unit": "GB/s",
                "table_name": "Roofline",
                "sub_table_name": "Bandwidth",
            },
            {
                "name": "VALU FLOPs",
                "metric_id": "2.1.0",
                "description": "Scaled compute rate",
                "unit": "GFLOP/s",
                "table_name": "System Speed-of-Light",
                "sub_table_name": "System Speed-of-Light",
            },
            {
                "name": "Read BW (Bytes)",
                "metric_id": "17.1.1",
                "description": "Already in base units",
                "unit": "Bytes/s",
                "table_name": "L2",
                "sub_table_name": "Bandwidth",
            },
            {
                "name": "Utilization",
                "metric_id": "3.1.0",
                "description": "Percent",
                "unit": "Percent",
                "table_name": "Memory",
                "sub_table_name": "Utilization",
            },
            {
                "name": "Kernel Time",
                "metric_id": "7.2.0",
                "description": "Duration",
                "unit": "Nanoseconds",
                "table_name": "Wavefront",
                "sub_table_name": "Timing",
            },
            {
                "name": "Scratch Allocation",
                "metric_id": "6.1.0",
                "description": "Scratch per wave",
                "unit": "KB per Wave",
                "table_name": "Memory Chart",
                "sub_table_name": "Scratch",
            },
        ])
    }
    analyzer._kernel_values_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "metric_id": "4.1.9",
                "kernel_name": "vecCopy",
                "value_name": "Value",
                "value": 1 / 3,
            },
            {
                "metric_id": "4.1.9",
                "kernel_name": "vecCopy",
                "value_name": "Peak (Empirical)",
                "value": 5324.8,
            },
            {
                "metric_id": "4.1.9",
                "kernel_name": "vecCopy",
                "value_name": "Percent of Peak",
                "value": 6.25,
            },
            {
                "metric_id": "2.1.0",
                "kernel_name": "vecCopy",
                "value_name": "Avg",
                "value": 2.25,
            },
            {
                "metric_id": "17.1.1",
                "kernel_name": "vecCopy",
                "value_name": "Avg",
                "value": 1 / 3,
            },
            {
                "metric_id": "3.1.0",
                "kernel_name": "vecCopy",
                "value_name": "Value",
                "value": 99.5,
            },
            {
                "metric_id": "7.2.0",
                "kernel_name": "vecCopy",
                "value_name": "Avg",
                "value": 12.5,
            },
            {
                "metric_id": "6.1.0",
                "kernel_name": "vecCopy",
                "value_name": "Value",
                "value": 2.5,
            },
        ])
    }
    analyzer._workload_values_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "metric_id": "4.1.9",
                "value_name": "Value",
                "value": 0.1,
            },
        ])
    }
    return analyzer


def store_metric_values(tmp_path, decimal, time_unit):
    """Store the metric values under one set of display options.

    Returns the stored metric definitions and values, in a comparable form.
    """
    cleanup_database()
    orm.Database.init(":memory:")
    try:
        analyzer = make_metric_value_analyzer(str(tmp_path))
        analyzer.get_args().decimal = decimal
        analyzer.get_args().time_unit = time_unit
        run_analysis_with_existing_database(analyzer)

        session = orm.Database.get_session()
        kernel_values = sorted(
            (
                value.metric.metric_id,
                value.metric.unit,
                value.value_name,
                value.value,
            )
            for value in session.query(orm.KernelMetricValue).all()
        )
        workload_values = sorted(
            (
                value.metric.metric_id,
                value.metric.unit,
                value.value_name,
                value.value,
            )
            for value in session.query(orm.WorkloadMetricValue).all()
        )
        dispatches = sorted(
            (dispatch.dispatch_id, dispatch.start_timestamp, dispatch.end_timestamp)
            for dispatch in session.query(orm.Dispatch).all()
        )
        return kernel_values, workload_values, dispatches
    finally:
        cleanup_database()


# =============================================================================
# Filter validation helpers
# =============================================================================


def make_gpu_split_dispatch_frame():
    """Frame with two GPUs, where kernel "a" is the longest kernel on GPU 0.

    Every dispatch lasts 100ns. GPU 0 runs a, a, b and a; GPU 1 runs b once.
    """
    return pd.DataFrame({
        "Kernel_Name": ["a", "a", "b", "a", "b"],
        "GPU_ID": [0, 0, 1, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "Start_Timestamp": [0, 200, 400, 600, 800],
        "End_Timestamp": [100, 300, 500, 700, 900],
    })


@pytest.fixture
def record_and_exit_filter_errors(monkeypatch):
    """Record each filter_validation error, then exit like console_error does."""
    calls, record_and_exit = common.record_and_exit_stub()
    common.patch_console(
        monkeypatch, "utils.filter_validation", "error", error=record_and_exit
    )
    return calls


# Metric names of the gfx950 membw tree, by the panel 3000 table that holds them.
MEMBW_TABLE_METRICS = {
    3001: [
        "L1 Cache - TA stalled by TCP (aggregated)",
        "L1 Cache - TCP stalled by UTCL1",
        "L1 Cache - TCP stalled by UTCL2",
        "L1 Cache - TCP stalled by TD",
        "L1 Cache - TCP stalled by L2",
        "L1 Cache - VMEM stalled by L1 Cache",
    ],
    3012: [
        "L2 Back Pressure Indicator",
        "L2 Memory BW Bound - Combined Credit Pressure",
        "L2 Memory BW Bound - Read Credit Pressure",
        "L2 Memory BW Bound - Write Credit Pressure",
        "L2 Internal Resource Pressure - Latency FIFO",
        "L2 Internal Resource Pressure - Source FIFO",
        "L2 Cache Efficiency",
        "L2 Remote Access Pressure (GMI)",
    ],
    3018: [
        "EA IO BW Bound - Combined",
        "EA Write Backpressure",
        "EA HBM Atomic Pressure",
    ],
}

# Avg values that activate the UTCL1 stall and HBM bandwidth branches.
MEMBW_UTCL1_HBM_AVG = {
    "L1 Cache - TA stalled by TCP (aggregated)": 22.4,
    "L1 Cache - TCP stalled by UTCL1": 18.7,
    "L1 Cache - TCP stalled by UTCL2": 1.2,
    "L1 Cache - TCP stalled by TD": 0.5,
    "L1 Cache - TCP stalled by L2": 2.1,
    "L1 Cache - VMEM stalled by L1 Cache": 3.0,
    "L2 Back Pressure Indicator": 5.0,
    "L2 Memory BW Bound - Combined Credit Pressure": 15.0,
    "L2 Memory BW Bound - Read Credit Pressure": 12.0,
    "L2 Memory BW Bound - Write Credit Pressure": 3.0,
    "L2 Internal Resource Pressure - Latency FIFO": 2.0,
    "L2 Internal Resource Pressure - Source FIFO": 1.0,
    "L2 Cache Efficiency": 80.0,
    "L2 Remote Access Pressure (GMI)": 0.5,
    "EA IO BW Bound - Combined": 0.0,
    "EA Write Backpressure": 1.0,
    "EA HBM Atomic Pressure": 0.5,
}


def make_membw_arch_dfs() -> Dict[int, pd.DataFrame]:
    """Panel 3000 metric tables as the arch config holds them, before values."""
    return {
        table_id: pd.DataFrame(
            {"Metric": metric_names, "Unit": ["Percent"] * len(metric_names)},
            index=pd.Index(
                [
                    f"{table_id // 100}.{table_id % 100}.{index}"
                    for index in range(len(metric_names))
                ],
                name="Metric_ID",
            ),
        )
        for table_id, metric_names in MEMBW_TABLE_METRICS.items()
    }


def make_membw_workload_values(
    avg_by_metric_name: Dict[str, float],
    arch_dfs: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Workload value rows for the panel 3000 metrics, with a Max row to ignore."""
    rows = []
    for template in arch_dfs.values():
        for metric_id, metric_name in template["Metric"].items():
            rows.append({"metric_id": metric_id, "value_name": "Max", "value": 999.0})
            if metric_name in avg_by_metric_name:
                rows.append({
                    "metric_id": metric_id,
                    "value_name": "Avg",
                    "value": avg_by_metric_name[metric_name],
                })
    return pd.DataFrame(rows, columns=["metric_id", "value_name", "value"])


def rebuild_membw_node(node_row, children_by_parent) -> BottleneckNode:
    """Rebuild one tree node, and its subtree, from its stored rows."""
    return BottleneckNode(
        id=node_row.node_id,
        label=node_row.label,
        level=node_row.level,
        state=node_row.state,
        supporting=tuple(
            SupportingMetric(
                key=metric_row.metric_key,
                value=metric_row.value,
                unit=metric_row.unit,
                display=metric_row.display,
            )
            for metric_row in node_row.supporting_metrics
        ),
        children=tuple(
            rebuild_membw_node(child_row, children_by_parent)
            for child_row in children_by_parent.get(node_row.membw_node_uuid, [])
        ),
        guidance_id=node_row.guidance_id,
    )


def rebuild_membw_result(session, workload) -> MemBwAnalysisResult:
    """Rebuild a workload's membw result from the analysis rows alone."""
    analysis_row = (
        session
        .query(orm.MembwAnalysis)
        .filter_by(workload_id=workload.workload_id)
        .one()
    )
    node_rows = (
        session
        .query(orm.MembwNode)
        .filter_by(membw_analysis_uuid=analysis_row.membw_analysis_uuid)
        .order_by(orm.MembwNode.node_order)
        .all()
    )
    children_by_parent: Dict[Optional[int], List[orm.MembwNode]] = {}
    for node_row in node_rows:
        children_by_parent.setdefault(node_row.parent_node_uuid, []).append(node_row)
    block_rows = (
        session
        .query(orm.MembwGuidanceBlock)
        .filter_by(membw_analysis_uuid=analysis_row.membw_analysis_uuid)
        .order_by(orm.MembwGuidanceBlock.block_order)
        .all()
    )
    return MemBwAnalysisResult(
        arch=analysis_row.arch,
        availability=analysis_row.availability,
        availability_reason=analysis_row.availability_reason,
        nodes=tuple(
            rebuild_membw_node(root_row, children_by_parent)
            for root_row in children_by_parent.get(None, [])
        ),
        guidance_blocks=tuple(block_row.text for block_row in block_rows),
    )


def make_membw_analyzer(workload_path, gpu_arch, membw_analysis):
    """Build a counter-backed analyzer whose workload has panel 3000 values."""
    analyzer = make_counter_backed_database_analyzer(workload_path, ["3"], [])
    analyzer._runs[workload_path].sys_info = pd.DataFrame([{"gpu_arch": gpu_arch}])
    analyzer._profiling_config = {
        "filter_blocks": ["3"],
        "membw_analysis": membw_analysis,
    }
    arch_dfs = make_membw_arch_dfs()
    arch_config = schema.ArchConfig()
    arch_config.dfs = arch_dfs
    analyzer._arch_configs = {gpu_arch: arch_config}
    analyzer._workload_values_data_per_workload = {
        workload_path: make_membw_workload_values(MEMBW_UTCL1_HBM_AVG, arch_dfs)
    }
    analyzer._membw_result_per_workload = analyzer.calc_membw_analysis()
    return analyzer


# =============================================================================
# db_analysis.evaluate() tests
# =============================================================================


def test_evaluate_parse_false_basic_expressions():
    """Test parse=False mode with basic expressions and substitutions."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"numCUs": 64, "clock_speed": 1500}

    # Test raw_pmc_df -> pmc_df substitution on flat single-index columns
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert list(result) == [10, 20, 30]

    # Test ammolite__ substitution for sys_info access
    result = db_analysis.evaluate(
        "test_metric",
        "ammolite__numCUs * 2",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 128

    # Test expression with helper function
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 60


def test_evaluate_parse_true_basic_expressions():
    """Test parse=True mode with $ substitution and AST transformation."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [2, 4, 6],
    })
    sys_info = {"numCUs": 64, "multiplier": 2}

    # Test $variable substitution
    result = db_analysis.evaluate(
        "test_metric",
        "$numCUs * $multiplier",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 128

    # Test AST transformation with SUPPORTED_CALL functions (SUM -> to_sum)
    # and bare identifiers (Counter1 -> raw_pmc_df["Counter1"])
    result = db_analysis.evaluate(
        "test_metric",
        "SUM(Counter1)",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 60

    # Test combined $ substitution and column access with AVG
    result = db_analysis.evaluate(
        "test_metric",
        "AVG(Counter1) + $numCUs",
        pmc_df,
        sys_info,
        parse=True,
    )
    assert result == 84  # avg(10,20,30)=20 + 64


def test_evaluate_none_and_na_handling():
    """Test evaluate() handling of None and NA values."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    # Explicit None in expression result returns None without warning
    result = db_analysis.evaluate(
        "test_metric",
        "None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None

    # Scalar NA values (NaN) return None
    pmc_df_nan = pd.DataFrame({"Counter1": [np.nan, np.nan, np.nan]})
    result = db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Counter1'])",
        pmc_df_nan,
        sys_info,
        parse=False,
    )
    assert result is None

    # Series with NA values are preserved (not converted to None)
    pmc_df_mixed = pd.DataFrame({"Counter1": [10, np.nan, 30]})
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['Counter1']",
        pmc_df_mixed,
        sys_info,
        parse=False,
    )
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == 10
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 30

    # Exceptions return None gracefully
    result = db_analysis.evaluate(
        "test_metric",
        "raw_pmc_df['NonExistent']",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result is None


def test_evaluate_with_none_in_formula_does_not_nullify_valid_result():
    """
    Test that expressions containing 'None' in formula string
    still return valid results when evaluation produces a value.

    This is a regression test for the bugfix where expressions like
    .where(..., None) were incorrectly returning None even when
    the actual result was valid.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 0, 3],  # Has a zero for conditional
    })
    sys_info = {}

    # Expression with None as fallback in .where() - should return valid result
    # when condition is met for at least some values
    result = db_analysis.evaluate(
        "test_metric",
        "(raw_pmc_df['Counter1'] / "
        "raw_pmc_df['Counter2'].where("
        "raw_pmc_df['Counter2'] != 0, None))",
        pmc_df,
        sys_info,
        parse=False,
    )
    # Result should be a Series, not None
    assert result is not None
    assert isinstance(result, pd.Series)

    # Expression that literally has "None" string but evaluates to a number
    result = db_analysis.evaluate(
        "test_metric",
        "10 if True else None",
        pmc_df,
        sys_info,
        parse=False,
    )
    assert result == 10


def test_evaluate_divide_by_zero_silenced_and_logged_at_debug():
    """
    Divide-by-zero (x/0 -> inf, 0/0 -> NaN) emits a numpy RuntimeWarning
    that is captured and logged via console_debug. The "evaluated to N/A"
    console_warning must not fire when a RuntimeWarning was caught.
    """
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {}

    cases = [
        # x/0 yields scalar inf; evaluate() collapses to None
        "to_sum(raw_pmc_df['Counter1']) / 0",
        # 0/0 yields scalar NaN; evaluate() collapses to None
        "(to_sum(raw_pmc_df['Counter1']) * 0) / 0",
    ]

    for expr in cases:
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as mock_warning:
            with patch(
                "rocprof_compute_analyze.analysis_db.console_debug"
            ) as mock_debug:
                result = db_analysis.evaluate(
                    "test_metric",
                    expr,
                    pmc_df,
                    sys_info,
                    parse=False,
                )

        assert result is None, f"Expected None for '{expr}', got {result}"

        mock_warning.assert_not_called()
        debug_msgs = [str(call) for call in mock_debug.call_args_list]
        assert any("RuntimeWarning" in m for m in debug_msgs), (
            f"Expected RuntimeWarning in console_debug output for '{expr}', "
            f"got {debug_msgs}"
        )


# =============================================================================
# Evaluation diagnostics tests
# =============================================================================


def test_evaluate_reports_na_at_debug_level():
    """An expression that evaluates to N/A is reported as debug, not warning."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": []})

    db_analysis.evaluate(
        "test_metric",
        "to_max(raw_pmc_df['Counter1'])",
        pmc_df,
        {},
        parse=False,
    )

    with patch("rocprof_compute_analyze.analysis_db.console_warning") as mock_warning:
        with patch("rocprof_compute_analyze.analysis_db.console_debug") as mock_debug:
            report_evaluation_diagnostics()

    mock_warning.assert_not_called()
    debug_msgs = [call.args[0] for call in mock_debug.call_args_list]
    assert any("evaluated to N/A" in msg for msg in debug_msgs), (
        f"Expected an N/A message at debug level, got {debug_msgs}"
    )


def test_evaluate_failure_message_names_the_exception_type():
    """A missing counter is reported as a warning naming the exception type."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['TCC_TAG_STALL_sum'])",
        pmc_df,
        {},
        parse=False,
    )

    with patch("rocprof_compute_analyze.analysis_db.console_warning") as mock_warning:
        report_evaluation_diagnostics()

    warning_msgs = [call.args[0] for call in mock_warning.call_args_list]
    assert len(warning_msgs) == 1, f"Expected one warning, got {warning_msgs}"
    assert "KeyError: 'TCC_TAG_STALL_sum'" in warning_msgs[0]


def test_evaluate_reports_a_repeated_failure_once():
    """The same failure across kernels is reported once, not once per kernel."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    for _ in range(5):
        db_analysis.evaluate(
            "test_metric",
            "to_sum(raw_pmc_df['Missing_Counter'])",
            pmc_df,
            {},
            parse=False,
        )

    with patch("rocprof_compute_analyze.analysis_db.console_warning") as mock_warning:
        report_evaluation_diagnostics()

    assert mock_warning.call_count == 1, (
        f"Expected one warning, got {mock_warning.call_args_list}"
    )


def test_report_evaluation_diagnostics_clears_collected_messages():
    """A second report emits nothing, so a later run starts clean."""
    drain_evaluation_diagnostics()
    pmc_df = pd.DataFrame({"Counter1": [1, 2, 3]})

    db_analysis.evaluate(
        "test_metric",
        "to_sum(raw_pmc_df['Missing_Counter'])",
        pmc_df,
        {},
        parse=False,
    )
    drain_evaluation_diagnostics()

    with patch("rocprof_compute_analyze.analysis_db.console_warning") as mock_warning:
        with patch("rocprof_compute_analyze.analysis_db.console_debug") as mock_debug:
            report_evaluation_diagnostics()

    mock_warning.assert_not_called()
    mock_debug.assert_not_called()


# =============================================================================
# db_analysis.calc_builtin_vars() tests
# =============================================================================


def test_calc_builtin_vars_processes_per_xcd_first():
    """
    Test that PER_XCD variables are processed before non-PER_XCD variables,
    allowing non-PER_XCD vars to reference PER_XCD vars via $placeholder.
    """
    pmc_df = pd.DataFrame({
        "Counter1": [100, 200],
    })
    sys_info = {"base_value": 10, "gpu_arch": "gfx942"}

    # Mock BUILD_IN_VARS with dependency chain:
    # - PER_XCD_VAR: computed from base_value
    # - DERIVED_VAR: depends on PER_XCD_VAR via $PER_XCD_VAR
    mock_builtin_vars = {
        "PER_XCD_VAR": "$base_value * 2",  # Should be processed first -> 20
        "DERIVED_VAR": "$PER_XCD_VAR + 5",  # Depends on PER_XCD_VAR -> 25
    }

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                db_analysis.calc_builtin_vars(
                    pmc_df, sys_info, ["$PER_XCD_VAR", "$DERIVED_VAR"]
                )

    # Verify PER_XCD var was computed
    assert sys_info["PER_XCD_VAR"] == 20

    # Verify DERIVED_VAR used the computed PER_XCD_VAR value
    assert sys_info["DERIVED_VAR"] == 25


def test_calc_builtin_vars_with_dataframe_expressions():
    """Test builtin vars that operate on DataFrame columns."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
    })
    sys_info = {"multiplier": 2, "gpu_arch": "gfx942"}

    # Use SUPPORTED_CALL function names (SUM -> to_sum via CodeTransformer)
    mock_builtin_vars = {
        "TOTAL_COUNT": "SUM(Counter1)",  # 60
        "SCALED_TOTAL": "$TOTAL_COUNT * $multiplier",  # 120
    }

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                db_analysis.calc_builtin_vars(
                    pmc_df, sys_info, ["$TOTAL_COUNT", "$SCALED_TOTAL"]
                )

    assert sys_info["TOTAL_COUNT"] == 60
    assert sys_info["SCALED_TOTAL"] == 120


def test_calc_builtin_vars_uses_gfx1250_gui_active_sum():
    """gfx1250 computes per-dispatch GUI-active cycles from the XCD sum."""
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE_sum": [800, 1600]})
    sys_info = {"num_xcd": 8, "gpu_arch": "gfx1250"}

    db_analysis.calc_builtin_vars(
        pmc_df,
        sys_info,
        ["$GRBM_GUI_ACTIVE_PER_XCD"],
    )

    pd.testing.assert_series_equal(
        sys_info["GRBM_GUI_ACTIVE_PER_XCD"],
        pd.Series([100.0, 200.0]),
        check_names=False,
    )


# =============================================================================
# db_analysis.calc_dataframe_expressions() tests
# =============================================================================


def test_calc_dataframe_expressions_applies_evaluate_to_rows():
    """Test that expressions are evaluated for each row of expression_df."""
    pmc_df = pd.DataFrame({
        "Counter1": [10, 20, 30],
        "Counter2": [1, 2, 3],
    })
    sys_info = {"scale": 100, "gpu_arch": "gfx942"}

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["sum", "scaled"],
        "value": [
            "to_sum(raw_pmc_df['Counter1'])",
            "ammolite__scale * 2",
        ],
    })

    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        result = db_analysis.calc_dataframe_expressions(pmc_df, sys_info, expression_df)

    assert isinstance(result, pd.Series)
    assert len(result) == 2
    assert result.iloc[0] == 60  # sum of Counter1
    assert result.iloc[1] == 200  # 100 * 2


def test_calc_dataframe_expressions_with_builtin_vars():
    """Test that calc_dataframe_expressions calls calc_builtin_vars first."""
    pmc_df = pd.DataFrame({"Counter1": [10, 20, 30]})
    sys_info = {"base": 5, "gpu_arch": "gfx942"}

    # Expression references a builtin var that gets computed
    mock_builtin_vars = {
        "COMPUTED_VAR": "$base * 10",  # 50
    }

    expression_df = pd.DataFrame({
        "metric_id": ["1.1", "1.2"],
        "value_name": ["test", "none_result"],
        "value": [
            "ammolite__COMPUTED_VAR + 1",  # Should be 51
            "None",
        ],
    })

    with patch(
        "rocprof_compute_analyze.analysis_db.mi_gpu_specs.get_gpu_series",
        return_value="MI300",
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.get_build_in_vars",
            return_value=mock_builtin_vars,
        ):
            with patch(
                "utils.utils_counter_defs.get_build_in_vars",
                return_value=mock_builtin_vars,
            ):
                result = db_analysis.calc_dataframe_expressions(
                    pmc_df, sys_info, expression_df
                )

    assert result.iloc[0] == 51
    # None from evaluate becomes NaN in pandas Series
    assert pd.isna(result.iloc[1])


def test_calc_dataframe_expressions_empty_returns_assignable_series():
    """An empty expression_df returns an empty Series, not a DataFrame."""
    expression_df = pd.DataFrame(columns=["metric_id", "value_name", "value"])

    result = db_analysis.calc_dataframe_expressions(
        pd.DataFrame({"Counter1": [1, 2, 3]}),
        {"gpu_arch": "gfx942"},
        expression_df,
    )

    assert isinstance(result, pd.Series)
    assert result.empty
    # Reproduces the call site: assigning the result as a single column must
    # not raise "Columns must be same length as key".
    expression_df["value"] = result


# =============================================================================
# calc_metrics_data tests
# =============================================================================


def test_calc_metrics_data_builds_rows_and_preserves_schema():
    """Metric tables expand into rows with table-level fields resolved once;
    non-metric tables are skipped and the output frames keep their columns."""
    workload_path = "/fake/workload"
    metric_df = pd.DataFrame(
        {
            "Metric": ["Grid Size"],
            "Avg": [" 10 "],
            "Min": [" 5 "],
            "Max": [" 20 "],
            "Unit": ["Work items"],
            "Description": ["Grid size desc"],
        },
        index=pd.Index(["7.1.0"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    # Table 1 is the in-memory Top Kernels frame: it has no Metric column and is
    # skipped; table 701 maps to panel 700 (table_name) and sub-table 701
    # (sub_table_name).
    arch_config.dfs = {
        1: pd.DataFrame({"Kernel_Name": ["kernel_a"]}),
        701: metric_df,
    }
    arch_config.panel_configs = {
        700: {
            "id": 700,
            "title": "Wavefront",
            "data source": [
                {"metric_table": {"id": 701, "title": "Wavefront Launch Stats"}}
            ],
        }
    }

    analyzer = db_analysis(MagicMock(verbose=0), {})
    analyzer._pmc_df_per_workload = {workload_path: pd.DataFrame({"Counter1": [1]})}
    analyzer._runs = {
        workload_path: MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]))
    }
    analyzer._arch_configs = {"gfx942": arch_config}

    metrics_info, expressions = analyzer.calc_metrics_data()

    info = metrics_info[workload_path]
    assert "pct_of_peak" in info.columns
    assert list(info["metric_id"]) == ["7.1.0"]
    assert list(info["name"]) == ["Grid Size"]
    assert list(info["table_name"]) == ["Wavefront"]
    assert list(info["sub_table_name"]) == ["Wavefront Launch Stats"]
    assert not bool(info["pct_of_peak"].iloc[0])

    exprs = expressions[workload_path]
    assert list(exprs.columns) == ["metric_id", "value_name", "value"]
    # Metric/Unit/Description are non-expression columns, so only Avg/Min/Max
    # expand into expression rows, stripped and in dataframe-column order.
    assert list(exprs["value_name"]) == ["Avg", "Min", "Max"]
    assert list(exprs["value"]) == ["10", "5", "20"]
    assert set(exprs["metric_id"]) == {"7.1.0"}


LONG_FORM_PMC_PERF = (
    "GPU_ID,Dispatch_ID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,"
    "Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Kernel_Name,"
    "Start_Timestamp,End_Timestamp,Kernel_ID,Counter_Name,Counter_Value\n"
    "0,0,256,64,0,0,8,0,16,kernel_a,10,20,0,SQ_WAVES,4\n"
)


def test_calc_pmc_df_data_reads_compressed_results(tmp_path):
    """The reader consumes compressed rocpd result artifacts."""
    common.write_result_csv(tmp_path, LONG_FORM_PMC_PERF)

    analyzer = db_analysis(MagicMock(verbose=0), {})
    analyzer._runs = {str(tmp_path): MagicMock()}
    analyzer._profiling_config = {}

    pmc_df_per_workload = analyzer.calc_pmc_df_data()

    assert pmc_df_per_workload[str(tmp_path)]["SQ_WAVES"].tolist() == [4]


def test_calc_pmc_df_data_skips_workload_without_results(tmp_path):
    """A workload with no counter artifacts, such as a PC-sampling-only run."""
    analyzer = db_analysis(MagicMock(verbose=0), {})
    analyzer._runs = {str(tmp_path): MagicMock()}
    analyzer._profiling_config = {}

    assert analyzer.calc_pmc_df_data() == {}


def test_calc_roofline_ceilings_skips_an_invalid_roofline_csv(tmp_path, monkeypatch):
    """A malformed roofline.csv gives no ceilings and a warning, not a crash."""
    mock_warning = common.patch_console(
        monkeypatch, "rocprof_compute_analyze.analysis_db", "warning"
    )["warning"]
    (tmp_path / "roofline.csv").write_text("this,is,bad,csv", encoding="utf-8")
    analyzer = db_analysis(SimpleNamespace(), {})
    analyzer._runs = {
        str(tmp_path): schema.Workload(sys_info=pd.DataFrame([{"gpu_arch": "gfx90a"}]))
    }

    assert analyzer.calc_roofline_ceilings() == {}
    assert "Roofline peaks are NULL" in mock_warning.call_args.args[0]


def test_calc_metrics_data_exports_qualified_per_channel_names():
    """
    Panel 18's per-channel rows reach the database as "Channel N", not as a
    bare index.
    """
    workload_path = "/fake/workload"
    metric_df = pd.DataFrame(
        {
            "Metric": ["Channel 0", "Channel 1"],
            "Min": [" 33.29 ", " 33.33 "],
            "Max": [" 33.51 ", " 33.33 "],
        },
        index=pd.Index(["18.2.0", "18.2.1"], name="Metric_ID"),
    )
    arch_config = schema.ArchConfig()
    arch_config.dfs = {1802: metric_df}
    arch_config.panel_configs = {
        1800: {
            "id": 1800,
            "title": "L2 Cache (per Channel)",
            "data source": [
                {"metric_table": {"id": 1802, "title": "L2 Cache Hit Rate (Percent)"}}
            ],
        }
    }

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pd.DataFrame({"Counter1": [1]})}
    analyzer._runs = {
        workload_path: MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]))
    }
    analyzer._arch_configs = {"gfx942": arch_config}

    metrics_info, expressions = analyzer.calc_metrics_data()

    info = metrics_info[workload_path]
    assert list(info["name"]) == ["Channel 0", "Channel 1"]
    assert list(info["table_name"]) == ["L2 Cache (per Channel)"] * 2
    assert list(info["sub_table_name"]) == ["L2 Cache Hit Rate (Percent)"] * 2

    # The label column is non-expression, so only Min/Max become expressions.
    exprs = expressions[workload_path]
    assert set(exprs["value_name"]) == {"Min", "Max"}


# =============================================================================
# filter_dispatch_frame tests
# =============================================================================


def make_repeated_dispatch_frame():
    """Frame where the longest kernel by total time is not the longest dispatch.

    ``kernel_frequent`` runs three times for 500ns each, ``kernel_long`` once
    for 1000ns.
    """
    return pd.DataFrame({
        "Kernel_Name": [
            "kernel_long",
            "kernel_frequent",
            "kernel_frequent",
            "kernel_frequent",
        ],
        "GPU_ID": [0, 0, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [0, 2000, 3000, 4000],
        "End_Timestamp": [1000, 2500, 3500, 4500],
    })


def test_filter_dispatch_frame_kernel_ids_match_the_cli_top_stats():
    """-k selects the same kernel here as it does in the cli top stats table."""
    dispatch_frame = make_repeated_dispatch_frame()
    kernel_top_df, _ = create_df_kernel_top_stats(
        df_in=dispatch_frame,
        filter_gpu_ids=None,
        filter_dispatch_ids=None,
        time_unit="ns",
    )

    filtered_df = filter_dispatch_frame(dispatch_frame, None, [0], None)

    assert filtered_df["Kernel_Name"].unique().tolist() == [
        kernel_top_df.loc[0, "Kernel_Name"]
    ]


@pytest.mark.parametrize("kernel_id", [99, -1])
def test_filter_dispatch_frame_rejects_out_of_range_kernel_id(kernel_id):
    """An out-of-range -k exits instead of raising IndexError."""
    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_repeated_dispatch_frame(), None, [kernel_id], None)


def test_filter_dispatch_frame_bounds_kernel_ids_by_the_filtered_frame():
    """Kernel ids are bounded by the frame left after the gpu and dispatch filters.

    Dispatch 1 leaves one kernel, so id 1 is out of range here even though the
    unfiltered frame has two kernels.
    """
    assert not filter_dispatch_frame(
        make_repeated_dispatch_frame(), None, [1], None
    ).empty

    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_repeated_dispatch_frame(), None, [1], ["1"])


def test_filter_dispatch_frame_rejects_an_unknown_gpu_id(
    record_and_exit_filter_errors,
):
    """A --gpu-id absent from the frame exits before any rows are kept."""
    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_gpu_split_dispatch_frame(), ["7"], None, None)

    assert record_and_exit_filter_errors == [("analysis", "['7'] is an invalid gpu-id")]


def test_filter_dispatch_frame_rejects_a_partly_invalid_gpu_list(
    record_and_exit_filter_errors,
):
    """A valid GPU beside an unknown one still exits, naming only the unknown id."""
    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_gpu_split_dispatch_frame(), ["0", "7"], None, None)

    assert record_and_exit_filter_errors == [("analysis", "['7'] is an invalid gpu-id")]


def test_filter_dispatch_frame_rejects_an_unknown_dispatch_id(
    record_and_exit_filter_errors,
):
    """A --kernel-iteration-range id absent from the frame exits with the range hint."""
    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_gpu_split_dispatch_frame(), None, None, ["9"])

    assert record_and_exit_filter_errors == [
        ("analysis", "9 is an invalid dispatch id. Dispatch ids run from 1 to 5.")
    ]


def test_filter_dispatch_frame_checks_gpu_before_kernel_and_dispatch(
    record_and_exit_filter_errors,
):
    """An unknown GPU is reported first, as the cli reports it."""
    with pytest.raises(SystemExit):
        filter_dispatch_frame(make_gpu_split_dispatch_frame(), ["7"], [9], ["9"])

    assert record_and_exit_filter_errors == [("analysis", "['7'] is an invalid gpu-id")]


def test_filter_dispatch_frame_keeps_rows_surviving_every_filter():
    """Only rows that pass the gpu, kernel and dispatch filters remain.

    Kernel id 0 on GPU 0 is kernel a, which dispatches 1, 2 and 4. Dispatch 2
    then narrows that to one row.
    """
    filtered_df = filter_dispatch_frame(
        make_gpu_split_dispatch_frame(), ["0"], [0], ["2"]
    )

    assert filtered_df["Dispatch_ID"].tolist() == [2]


def test_filter_dispatch_frame_accepts_a_threshold_with_a_leading_space():
    """A '> n' with a leading space selects the same rows as '> n' without one."""
    filtered_df = filter_dispatch_frame(
        make_gpu_split_dispatch_frame(), None, None, [" >2"]
    )

    assert filtered_df["Dispatch_ID"].tolist() == [3, 4, 5]


def test_pc_sampling_dispatch_data_keeps_gpu_and_dispatch_narrowing(monkeypatch):
    """A PC-sampling-only run keeps the --gpu-id and --kernel-iteration-range narrowing.

    The trace is narrowed the same way as a counter frame, so only dispatch 2 on
    GPU 0 remains.
    """
    trace = make_gpu_split_dispatch_frame().rename(
        columns={"Dispatch_ID": "Dispatch_Id"}
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.process_pc_sampling_kernel_traces",
        lambda _tool_data_records: trace,
    )
    workload = schema.Workload(filter_gpu_ids=["0"], filter_dispatch_ids=["2"])

    dispatch_df = db_analysis._build_pc_sampling_dispatch_data([], workload)

    assert dispatch_df["dispatch_id"].tolist() == [2]
    assert dispatch_df["kernel_name"].tolist() == ["a"]


def test_calc_dispatch_data_covers_only_rows_surviving_every_filter():
    """The dispatch rows come from the filtered frame, not the raw counter frame."""
    workload_path = "workload"
    analyzer = db_analysis(SimpleNamespace(), {})
    analyzer._runs = {
        workload_path: schema.Workload(
            filter_gpu_ids=["0"],
            filter_kernel_ids=[0],
            filter_dispatch_ids=["2"],
        )
    }
    analyzer._pmc_df_per_workload = {workload_path: make_gpu_split_dispatch_frame()}

    analyzer._pmc_df_per_workload = analyzer.apply_pmc_filters()
    dispatch_df = analyzer.calc_dispatch_data({})[workload_path]

    assert dispatch_df["dispatch_id"].tolist() == [2]
    assert dispatch_df["kernel_name"].tolist() == ["a"]
    assert dispatch_df["gpu_id"].tolist() == [0]


# =============================================================================
# Workload total duration: the Pct denominator, taken before -k
# =============================================================================


def make_workload_total_analyzer(
    workload_path,
    pmc_df,
    filter_gpu_ids=(),
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
):
    """Build a counter-backed analyzer holding one unfiltered counter frame."""
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"),
        {},
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
            filter_gpu_ids=list(filter_gpu_ids),
            filter_kernel_ids=list(filter_kernel_ids),
            filter_dispatch_ids=list(filter_dispatch_ids),
        )
    }
    analyzer._profiling_config = {}
    analyzer._pmc_df_per_workload = {workload_path: pmc_df}
    return analyzer


def make_pmc_database_analyzer(
    workload_path,
    pmc_df,
    filter_gpu_ids=(),
    filter_kernel_ids=(),
    filter_dispatch_ids=(),
):
    """Build a counter-backed analyzer that filters its frame as pre_processing does."""
    analyzer = make_workload_total_analyzer(
        workload_path,
        pmc_df,
        filter_gpu_ids=filter_gpu_ids,
        filter_kernel_ids=filter_kernel_ids,
        filter_dispatch_ids=filter_dispatch_ids,
    )
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._pc_sampling_tool_data_per_workload = {}
    analyzer._workload_total_duration_per_workload = (
        analyzer.calc_workload_total_durations({})
    )
    analyzer._pmc_df_per_workload = analyzer.apply_pmc_filters()
    analyzer._dispatch_data_per_workload = analyzer.calc_dispatch_data({})
    analyzer._roofline_data_per_kernel = {workload_path: pd.DataFrame()}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._kernel_values_data_per_workload = {}
    analyzer._workload_values_data_per_workload = {}
    return analyzer


@pytest.mark.parametrize(
    ("filter_gpu_ids", "filter_kernel_ids", "filter_dispatch_ids", "expected_total"),
    [
        ([], [], [], 2500),
        (["0"], [], [], 2500),
        ([], [], ["1"], 1000),
        ([], [], ["> 1"], 1500),
        ([], [1], [], 2500),
        ([], [0], ["1"], 1000),
    ],
    ids=[
        "unfiltered",
        "gpu",
        "dispatch_id",
        "dispatch_bound",
        "kernel_filter_only",
        "kernel_filter_with_dispatch",
    ],
)
def test_calc_workload_total_durations_ignores_the_kernel_filter(
    filter_gpu_ids, filter_kernel_ids, filter_dispatch_ids, expected_total
):
    """The total follows the gpu and dispatch filters, and -k never changes it."""
    analyzer = make_workload_total_analyzer(
        "/fake/workload",
        make_repeated_dispatch_frame(),
        filter_gpu_ids=filter_gpu_ids,
        filter_kernel_ids=filter_kernel_ids,
        filter_dispatch_ids=filter_dispatch_ids,
    )

    totals = analyzer.calc_workload_total_durations({})

    assert totals == {"/fake/workload": expected_total}


def test_calc_workload_total_durations_totals_sampling_trace_before_kernel_filter(
    tmp_path,
):
    """A sampling-only workload's total covers every kernel the trace holds."""

    def make_timed_tool_data():
        # Dispatch 0 runs for 300 ns and dispatch 1 for 500 ns.
        tool_data = make_pc_sampling_tool_data()
        dispatch_records = tool_data["buffer_records"]["kernel_dispatch"]
        for record, end_timestamp in zip(dispatch_records, [300, 500]):
            record["end_timestamp"] = end_timestamp
        return tool_data

    workload_path = str(tmp_path / "workloads" / ISA_WORKLOAD_NAME)
    unfiltered = make_pc_sampling_only_database_analyzer(
        workload_path, [make_timed_tool_data()]
    )
    kernel_filtered = make_pc_sampling_only_database_analyzer(
        workload_path, [make_timed_tool_data()], filter_kernel_ids=[0]
    )

    assert len(kernel_filtered._dispatch_data_per_workload[workload_path]) < len(
        unfiltered._dispatch_data_per_workload[workload_path]
    )
    assert unfiltered._workload_total_duration_per_workload[workload_path] == 800
    assert kernel_filtered._workload_total_duration_per_workload[workload_path] == 800


def test_run_analysis_stores_the_pct_inputs_for_a_kernel_filtered_workload(
    db_session, tmp_path
):
    """Pct from the stored rows equals the top-stats Percent of the kept kernel."""
    raw_pmc = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c", "kernel_b"],
        "GPU_ID": [0, 0, 1, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "Start_Timestamp": [0, 0, 0, 0, 0],
        "End_Timestamp": [500, 300, 100, 200, 300],
    })
    analyzer = make_pmc_database_analyzer(
        str(tmp_path / "workload"),
        raw_pmc,
        filter_gpu_ids=["0"],
        filter_kernel_ids=[0],
    )

    run_analysis_with_materialized_views(analyzer)

    stored_workload = db_session.query(orm.Workload).one()
    assert stored_workload.total_duration_ns == 1300
    rows = db_session.execute(
        text(
            "SELECT kernel_name, duration_ns_sum, workload_total_duration_ns "
            "FROM compute_kernel_view"
        )
    ).fetchall()
    assert [row[0] for row in rows] == ["kernel_b"]
    kernel_top_df, _ = create_df_kernel_top_stats(
        df_in=raw_pmc,
        raw_data_dir=str(tmp_path),
        filter_gpu_ids=["0"],
        filter_dispatch_ids=None,
        time_unit="ns",
    )
    expected_percent = kernel_top_df.set_index("Kernel_Name").loc["kernel_b", "Percent"]
    _, duration_sum, total_duration = rows[0]
    assert duration_sum / total_duration * 100 == pytest.approx(expected_percent)


# =============================================================================
# Noise-clamp warning + summary tests
# =============================================================================


def test_calc_expressions_noise_clamp():
    """Variance warnings fire only at workload level, summary once per workload.

    - evaluate(emit_variance_warnings=True) emits the per-metric warning when
      to_noise_clamp advances the global counter; the False kwarg stays silent.
    - calc_expressions emits exactly one variance warning per workload
      (kernel-level pass is silent) and calls print_noise_clamp_summary once.
    """
    workload_path = "/fake/workload"
    noise_clamp_expression = (
        "to_noise_clamp(to_min(raw_pmc_df['DIFF']), to_max(raw_pmc_df['REF']))"
    )
    # Two distinct kernels so groupby yields two kernel-level evaluate calls
    # in addition to one workload-level call. Without the kwarg gate the
    # unguarded code would emit three warnings; with the gate, exactly one.
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b"],
        "DIFF": [-100.0, -100.0],
        "REF": [1000.0, 1000.0],
    })
    expression_template = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["clamped"],
        "value": [noise_clamp_expression],
    })
    sys_info_df = pd.DataFrame([{"placeholder": 1, "gpu_arch": "gfx942"}])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pmc_df_per_workload = {workload_path: pmc_df}
    analyzer._metric_expression_data_per_workload = {workload_path: expression_template}
    analyzer._metrics_info_data_per_workload = {}
    analyzer._roofline_ceilings_per_workload = {workload_path: {}}
    analyzer._typed_sys_info_per_workload = {
        workload_path: {"placeholder": 1, "gpu_arch": "gfx942"}
    }
    analyzer._runs = {workload_path: MagicMock(sys_info=sys_info_df)}
    analyzer._arch_configs = MagicMock()

    # Direct evaluate kwarg behavior.
    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=True,
        )
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric: direct_test" in warning_call.args[0]
        ]
        assert len(variance_warning_calls) == 1
        assert get_noise_clamp_warnings()["count"] >= 1

    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        db_analysis.evaluate(
            "direct_test_off",
            noise_clamp_expression,
            pmc_df,
            {},
            emit_variance_warnings=False,
        )
        assert get_noise_clamp_warnings()["count"] >= 1
        variance_warning_calls = [
            warning_call
            for warning_call in console_warning_mock.call_args_list
            if "Variance corrected for metric:" in warning_call.args[0]
        ]
        assert variance_warning_calls == []

    # calc_expressions per-workload bracket.
    clear_noise_clamp_warnings()
    with patch(
        "rocprof_compute_analyze.analysis_db.get_build_in_vars", return_value={}
    ):
        with patch(
            "rocprof_compute_analyze.analysis_db.console_warning"
        ) as console_warning_mock:
            with patch(
                "rocprof_compute_analyze.analysis_db.print_noise_clamp_summary"
            ) as print_noise_clamp_summary_mock:
                with patch.object(db_analysis, "validate_dual_issue_metrics"):
                    analyzer.calc_expressions()

    variance_warning_calls = [
        warning_call
        for warning_call in console_warning_mock.call_args_list
        if "Variance corrected for metric:" in warning_call.args[0]
    ]
    assert len(variance_warning_calls) == 1
    assert "1.1 - clamped" in variance_warning_calls[0].args[0]
    print_noise_clamp_summary_mock.assert_called_once()
    assert get_noise_clamp_warnings()["count"] >= 1


# =============================================================================
# _derive_pct_of_peak_values tests
# =============================================================================


class TestDerivePctOfPeakValues:
    """Tests for db_analysis._derive_pct_of_peak_values."""

    def _make_values_df(
        self,
        metric_ids: list[str],
        value_names: list[str],
        values: list[float],
        kernel_names: Optional[list[str]] = None,
    ):
        """Build a long-format values DataFrame as produced by calc_expressions."""
        data = {
            "metric_id": metric_ids,
            "value_name": value_names,
            "value": values,
        }
        if kernel_names is not None:
            data["kernel_name"] = kernel_names
        return pd.DataFrame(data)

    def test_pct_of_peak_true_metric_appends_percent_of_peak_row(self):
        """A pct_of_peak-enabled metric produces one new Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 200.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values({"1.1"}, values_df)
        assert len(new_rows) == 1
        assert new_rows[0]["value_name"] == "Percent of Peak"
        assert new_rows[0]["value"] == pytest.approx(25.0)

    def test_multi_kernel_produces_one_row_per_kernel(self):
        """Calling once per kernel produces one Percent of Peak row per kernel."""
        kernel_a_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[100.0, 200.0],
            kernel_names=["kernel_a", "kernel_a"],
        )
        kernel_b_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[60.0, 300.0],
            kernel_names=["kernel_b", "kernel_b"],
        )
        rows_a = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_a_df)
        rows_b = db_analysis._derive_pct_of_peak_values({"1.1"}, kernel_b_df)
        assert len(rows_a) == 1
        assert rows_a[0]["value"] == pytest.approx(50.0)  # 100/200*100
        assert len(rows_b) == 1
        assert rows_b[0]["value"] == pytest.approx(20.0)  # 60/300*100

    def test_pct_of_peak_false_metric_produces_no_pct_row(self):
        """A metric not in pct_of_peak_metric_ids produces no Percent of Peak row."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, 100.0],
        )
        new_rows = db_analysis._derive_pct_of_peak_values(set(), values_df)
        assert new_rows == []

    def test_incomplete_data_skips_metric(self):
        """A metric missing Peak or Avg/Value must be skipped gracefully."""
        incomplete_cases = [
            # Only "Avg" present -- no "Peak" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Avg"], values=[50.0]
            ),
            # Only "Peak" present -- no "Avg" or "Value" row
            self._make_values_df(
                metric_ids=["1.1"], value_names=["Peak"], values=[100.0]
            ),
        ]
        for incomplete_values in incomplete_cases:
            new_rows = db_analysis._derive_pct_of_peak_values(
                {"1.1"}, incomplete_values
            )
            assert new_rows == []

    def test_uncalculable_percentage_is_stored_as_null(self):
        """A flagged metric with a value and a NaN peak keeps a row whose value
        is NULL, never 0 (FR-15)."""
        values_df = self._make_values_df(
            metric_ids=["1.1", "1.1"],
            value_names=["Avg", "Peak"],
            values=[50.0, np.nan],
        )
        new_rows = db_analysis._derive_pct_of_peak_values({"1.1"}, values_df)
        assert len(new_rows) == 1
        assert new_rows[0]["value_name"] == "Percent of Peak"
        assert pd.isna(new_rows[0]["value"])


# =============================================================================
# Dual-issue VALU validation tests
# =============================================================================


def test_validate_dual_issue_metrics_emits_warning_above_peak():
    """Long-format VALU Utilization above peak triggers the dual-issue warning."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU Utilization can go up to 200%" in msg


def test_validate_dual_issue_metrics_silent_below_peak():
    """Below-peak VALU Utilization stays silent."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [80.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()


def test_validate_dual_issue_metrics_uses_peak_empirical_fallback():
    """Peak (Empirical) wins when present; falls back to Peak otherwise."""
    arch_config = make_dual_issue_arch_config(
        "VALU FLOPs (F64)", peak_col="Peak (Empirical)"
    )
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak (Empirical)"],
        "value": [600.0, 400.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_called_once()
    msg = console_warning_mock.call_args.args[0]
    assert "VALU FLOPs can exceed the peak value" in msg


def test_validate_dual_issue_metrics_appends_valu2_suffix_on_gfx950():
    """gfx950 with non-zero SQ_ACTIVE_INST_VALU2 appends the confirmation."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"SQ_ACTIVE_INST_VALU2": [1, 2, 3]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx950"},
            workload_values_df,
            arch_config,
        )

    msg = console_warning_mock.call_args.args[0]
    assert "Dual-issue activity detected via SQ_ACTIVE_INST_VALU2 counter" in msg


def test_validate_dual_issue_metrics_skips_non_metric_table_dfs():
    """dfs entries whose dfs_type is not metric_table are ignored."""
    arch_config = make_dual_issue_arch_config("VALU Utilization")
    arch_config.dfs_type = {201: "raw_csv_table"}
    workload_values_df = pd.DataFrame({
        "metric_id": ["1.1", "1.1"],
        "value_name": ["Value", "Peak"],
        "value": [150.0, 100.0],
    })
    pmc_df = pd.DataFrame({"GRBM_GUI_ACTIVE": [1000]})

    with patch("utils.metrics.common.console_warning") as console_warning_mock:
        db_analysis.validate_dual_issue_metrics(
            pmc_df,
            {"gpu_arch": "gfx942"},
            workload_values_df,
            arch_config,
        )

    console_warning_mock.assert_not_called()


# =============================================================================
# PC-sampling population
# =============================================================================


def make_pc_sampling_tool_data():
    """Two offsets under two kernels sharing one code object, with counts."""
    stall = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT"
    inst_type = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU"
    return {
        "metadata": {"pid": 42},
        "buffer_records": {
            "pc_sample_host_trap": [],
            "pc_sample_stochastic": [
                {
                    "inst_index": 0,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x10},
                        "dispatch_id": 0,
                        "wave_issued": False,
                        "snapshot": {"stall_reason": stall},
                        "inst_type": inst_type,
                    },
                },
                {
                    "inst_index": 1,
                    "record": {
                        "pc": {"code_object_id": 5, "code_object_offset": 0x20},
                        "dispatch_id": 1,
                        "wave_issued": True,
                        "snapshot": {},
                        "inst_type": inst_type,
                    },
                },
            ],
            "kernel_dispatch": [
                make_pc_sampling_dispatch(0, 100),
                make_pc_sampling_dispatch(1, 101),
            ],
        },
        "strings": {
            "pc_sample_instructions": ["v_mov", "v_add"],
            "pc_sample_comments": ["/s/a.cpp:1", "/s/a.cpp:2"],
        },
        "kernel_symbols": [
            {
                "kernel_id": 100,
                "code_object_id": 5,
                "kernel_name": "_Z7vecCopyv.kd",
                "formatted_kernel_name": "vecCopy",
                "truncated_kernel_name": "vecCopy",
            },
            {
                "kernel_id": 101,
                "code_object_id": 5,
                "kernel_name": "vecAdd.kd",
                "formatted_kernel_name": "vecAdd",
                "truncated_kernel_name": "vecAdd",
            },
        ],
        "code_objects": [{"code_object_id": 5, "load_base": 0x1000}],
        "agents": [],
    }


def test_add_pc_sampling_data_no_tool_data_is_noop(db_session):
    """A workload without tool data inserts no rows."""
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {"/fake/workload": []}

    code_object_stores = analyzer.add_pc_sampling_data(
        "/fake/workload",
        workload,
        {},
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    assert code_object_stores == {}
    assert db_session.query(orm.CodeObjectStore).count() == 0
    assert db_session.query(orm.InstructionLine).count() == 0


def test_add_pc_sampling_data_populates_and_attributes_kernels(db_session):
    """Instruction lines use the workload kernels and transient store registry."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
        "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    code_object = db_session.query(orm.CodeObjectStore).one()
    assert set(code_object_stores) == {(42, 5)}
    assert code_object_stores[(42, 5)] is code_object
    assert code_object.code_object_uuid is not None
    assert code_object.pid == 42
    assert code_object.load_base == 0x1000

    lines = db_session.query(orm.InstructionLine).all()
    kernel_by_offset = {
        line.code_object_offset: line.kernel_symbol.kernel.kernel_name for line in lines
    }
    assert kernel_by_offset == {0x10: "vecCopy", 0x20: "vecAdd"}

    # The stalled sample carries a stall-reason count; both carry an inst type.
    stalled = next(line for line in lines if line.code_object_offset == 0x10)
    assert stalled.pc_sample_state.stall_count == 1
    assert {
        r.stall_reason_lookup.text for r in stalled.pc_sample_state.stall_reasons
    } == {"WAITCNT"}


def test_add_pc_sampling_data_inserts_wave_measurements_from_sys_info(
    db_session,
):
    """Configured denominators populate percentages; absent ones stay null."""
    tool_data = make_pc_sampling_tool_data()
    samples = tool_data["buffer_records"]["pc_sample_stochastic"]
    samples[0]["record"].update({"exec_mask": 0b1111, "wave_cnt": 8})
    samples[1]["record"].update({"exec_mask": 0b11, "wave_cnt": 16})
    cases = [
        (
            "/fake/workload/missing-wave-denominators",
            {"gpu_arch": "gfx942"},
            {0x10: (None, None), 0x20: (None, None)},
        ),
        (
            "/fake/workload/configured-wave-denominators",
            {
                "gpu_arch": "gfx942",
                "wave_size": "64",
                "max_waves_per_cu": "32",
            },
            {0x10: (6.25, 25.0), 0x20: (3.125, 50.0)},
        ),
    ]

    for workload_path, sys_info_row, expected_by_offset in cases:
        workload = orm.Workload(name=workload_path, sub_name="sampling")
        db_session.add(workload)
        kernel_objs = {
            kernel_name: orm.Kernel(kernel_name=kernel_name, workload=workload)
            for kernel_name in ("vecCopy", "vecAdd")
        }
        db_session.add_all(kernel_objs.values())
        analyzer = make_pc_sampling_database_analyzer(
            {workload_path: [copy.deepcopy(tool_data)]},
            sys_info_row=sys_info_row,
        )

        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            {},
            make_source_frame_collector(workload, workload_path),
            sys_info=sys_info_row,
        )
        db_session.commit()

        sample_states_by_offset = {
            line.code_object_offset: line.pc_sample_state
            for code_object_store in code_object_stores.values()
            for line in store_instruction_lines(code_object_store)
        }
        actual_by_offset = {
            offset: (
                sample_state.active_thread_percent,
                sample_state.wave_occupancy_percent,
            )
            for offset, sample_state in sample_states_by_offset.items()
        }
        assert actual_by_offset == expected_by_offset


def test_add_pc_sampling_data_separates_shared_code_object_ids_across_pids(
    db_session,
):
    """Keep colliding code-object IDs isolated by process ID."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    kernel_objs = {
        kernel_name: orm.Kernel(
            kernel_name=kernel_name,
            workload=workload,
        )
        for kernel_name in ("vecCopy", "vecAdd")
    }
    for kernel in kernel_objs.values():
        db_session.add(kernel)

    first_tool_data = make_pc_sampling_tool_data()
    second_tool_data = copy.deepcopy(first_tool_data)
    second_tool_data["metadata"]["pid"] = 99
    second_tool_data["code_objects"][0]["load_base"] = 0x3000

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [first_tool_data, second_tool_data]
    }
    code_object_stores = analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    assert set(code_object_stores) == {(42, 5), (99, 5)}
    first_store = code_object_stores[(42, 5)]
    second_store = code_object_stores[(99, 5)]
    assert first_store is not second_store
    assert first_store.code_object_uuid != second_store.code_object_uuid
    assert (first_store.pid, second_store.pid) == (42, 99)
    # Both stores attribute to the same workload kernels; the stores stay
    # distinct because code_object_id is process-local.
    expected_kernels = {kernel_objs["vecCopy"], kernel_objs["vecAdd"]}
    assert {
        line.kernel_symbol.kernel for line in store_instruction_lines(first_store)
    } == expected_kernels
    assert {
        line.kernel_symbol.kernel for line in store_instruction_lines(second_store)
    } == expected_kernels
    total_vec_copy_samples = sum(
        line.pc_sample_state.total_count
        for line in db_session.query(orm.InstructionLine).all()
        if line.kernel_symbol.kernel.kernel_name == "vecCopy"
    )
    assert total_vec_copy_samples == 2


def test_run_analysis_scopes_pc_sampling_uuids_by_process(db_session):
    """Assign distinct database identities to each process's sampling data."""
    workload_path = "/fake/workload"
    tool_data_records = [
        make_colliding_pc_sampling_tool_data(42, 1),
        make_colliding_pc_sampling_tool_data(99, 2),
    ]
    tool_data_records[0]["buffer_records"]["kernel_dispatch"].append(
        make_pc_sampling_dispatch(1, 100)
    )
    workload = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx942"}]),
    )
    analyzer = db_analysis(
        SimpleNamespace(),
        {},
    )
    analyzer._runs = {workload_path: workload}
    analyzer._roofline_ceilings_per_workload = {}
    analyzer._roofline_ceiling_rows_per_workload = {}
    analyzer._profiling_config = {"filter_blocks": ["pc_sampling"]}
    analyzer._pc_sampling_tool_data_per_workload = {workload_path: tool_data_records}
    trace_df = pc_sampling_dispatch_trace(tool_data_records)
    analyzer._dispatch_data_per_workload = {
        workload_path: analyzer._build_pc_sampling_dispatch_data(trace_df, workload)
    }
    analyzer._workload_total_duration_per_workload = (
        analyzer.calc_workload_total_durations({workload_path: trace_df})
    )
    analyzer._roofline_data_per_kernel = {}
    analyzer._membw_result_per_workload = {}

    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(orm.Database, "init"))
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.build_database(Path("analysis.db"))

    dispatches = (
        db_session
        .query(orm.Dispatch)
        .order_by(orm.Dispatch.kernel_uuid, orm.Dispatch.dispatch_id)
        .all()
    )
    code_object_stores = (
        db_session.query(orm.CodeObjectStore).order_by(orm.CodeObjectStore.pid).all()
    )
    assert [(store.pid, store.code_object_id) for store in code_object_stores] == [
        (42, 5),
        (99, 5),
    ]
    # A kernel is identified by name across the workload, so both processes
    # share one kernel row. Renumbering keeps their dispatch ids distinct.
    assert [
        (dispatch.dispatch_id, dispatch.kernel.kernel_name) for dispatch in dispatches
    ] == [
        (1, "vecCopy"),
        (2, "vecCopy"),
        (3, "vecCopy"),
    ]
    assert len({dispatch.kernel_uuid for dispatch in dispatches}) == 1
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 3

    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.kernel_symbol.code_object_uuid for line in instruction_lines}) == 2
    assert {
        (line.code_object_offset, line.instruction) for line in instruction_lines
    } == {(0x10, "v_mov")}
    assert {
        (frame.source_line.source_file.file_path, frame.source_line.line_number)
        for line in instruction_lines
        for frame in line.source_lines
    } == {("/s/shared.cpp", 7)}

    instruction_by_process_id = {
        line.kernel_symbol.code_object_store.pid: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    # Each process's line resolves to the one shared kernel, which is reachable
    # from the kernel view because it owns dispatches.
    shared_kernel_uuid = dispatches[0].kernel_uuid
    for instruction_line in instruction_by_process_id.values():
        assert instruction_line.kernel_symbol.kernel_uuid == shared_kernel_uuid
        assert instruction_line.kernel_symbol.kernel.dispatches
        assert instruction_line.kernel_symbol.code_object_store.code_object_id == 5


def test_run_analysis_materialized_views_keep_pc_sampling_origins(
    db_session,
):
    """Preserve process-scoped code objects through materialized views."""
    workload_path = "/fake/workload"
    analyzer = make_pc_sampling_only_database_analyzer(
        workload_path,
        [
            make_colliding_pc_sampling_tool_data(42, 1),
            make_colliding_pc_sampling_tool_data(99, 2),
        ],
    )

    run_analysis_with_materialized_views(analyzer)

    code_object_stores = (
        db_session.query(orm.CodeObjectStore).order_by(orm.CodeObjectStore.pid).all()
    )
    assert [(store.pid, store.code_object_id) for store in code_object_stores] == [
        (42, 5),
        (99, 5),
    ]
    dispatches = db_session.query(orm.Dispatch).order_by(orm.Dispatch.dispatch_id).all()
    assert [
        (dispatch.dispatch_id, dispatch.kernel.kernel_name) for dispatch in dispatches
    ] == [
        (1, "vecCopy"),
        (2, "vecCopy"),
    ]
    assert len({dispatch.dispatch_uuid for dispatch in dispatches}) == 2
    assert len({dispatch.kernel_uuid for dispatch in dispatches}) == 1

    assert len(code_object_stores) == 2
    assert {store.code_object_id for store in code_object_stores} == {5}
    assert len({store.code_object_uuid for store in code_object_stores}) == 2

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert len(instruction_lines) == 2
    assert len({line.instruction_uuid for line in instruction_lines}) == 2
    assert len({line.kernel_symbol.code_object_uuid for line in instruction_lines}) == 2
    assert {line.code_object_offset for line in instruction_lines} == {0x10}
    assert {line.instruction for line in instruction_lines} == {"v_mov"}
    assert {
        (frame.source_line.source_file.file_path, frame.source_line.line_number)
        for line in instruction_lines
        for frame in line.source_lines
    } == {("/s/shared.cpp", 7)}

    instruction_by_process_id = {
        line.kernel_symbol.code_object_store.pid: line for line in instruction_lines
    }
    assert set(instruction_by_process_id) == {42, 99}
    assert {
        process_id: line.pc_sample_state.total_count
        for process_id, line in instruction_by_process_id.items()
    } == {42: 1, 99: 2}
    assert sum(line.pc_sample_state.total_count for line in instruction_lines) == 3

    shared_kernel_uuid = dispatches[0].kernel_uuid
    for instruction_line in instruction_by_process_id.values():
        assert instruction_line.kernel_symbol.kernel_uuid == shared_kernel_uuid
        assert (
            instruction_line.kernel_symbol.code_object_uuid
            == instruction_line.kernel_symbol.code_object_store.code_object_uuid
        )
        assert instruction_line.kernel_symbol.code_object_store.code_object_id == 5

    sample_states = query_instruction_sample_states(db_session).all()
    stall_reasons = query_instruction_stall_reasons(db_session).all()
    instruction_samples = db_session.query(orm.InstructionSample).all()
    assert len(sample_states) == 2
    assert {state.total_count for state in sample_states} == {1, 2}
    assert len(stall_reasons) == 2
    assert {reason.count for reason in stall_reasons} == {1, 2}
    assert len(instruction_samples) == 2
    assert {sample.count for sample in instruction_samples} == {1, 2}

    orphan_queries = {
        "instruction_line_to_kernel_symbol": """
            SELECT COUNT(*)
            FROM compute_instruction_line AS instruction_line
            LEFT JOIN compute_kernel_symbol AS kernel_symbol
                ON instruction_line.kernel_symbol_uuid
                    = kernel_symbol.kernel_symbol_uuid
            WHERE kernel_symbol.kernel_symbol_uuid IS NULL
        """,
        "kernel_symbol_to_kernel": """
            SELECT COUNT(*)
            FROM compute_kernel_symbol AS kernel_symbol
            LEFT JOIN compute_kernel AS kernel
                ON kernel_symbol.kernel_uuid = kernel.kernel_uuid
            WHERE kernel.kernel_uuid IS NULL
        """,
        "kernel_symbol_to_code_object": """
            SELECT COUNT(*)
            FROM compute_kernel_symbol AS kernel_symbol
            LEFT JOIN compute_code_object_store AS code_object
                ON kernel_symbol.code_object_uuid = code_object.code_object_uuid
            WHERE code_object.code_object_uuid IS NULL
        """,
        "sample_state_to_instruction_or_source_line": """
            SELECT COUNT(*)
            FROM compute_pc_sample_state AS sample_state
            LEFT JOIN compute_instruction_line AS instruction_line
                ON sample_state.instruction_uuid = instruction_line.instruction_uuid
            LEFT JOIN compute_source_line AS source_line
                ON sample_state.source_line_uuid = source_line.source_line_uuid
            WHERE instruction_line.instruction_uuid IS NULL
                AND source_line.source_line_uuid IS NULL
        """,
        "stall_reason_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_pc_sample_stall_reason AS stall_reason
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON stall_reason.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_pc_sample_stall_reason_lookup AS stall_lookup
                ON stall_reason.pc_sample_stall_reason_lookup_uuid =
                    stall_lookup.pc_sample_stall_reason_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR stall_lookup.pc_sample_stall_reason_lookup_uuid IS NULL
        """,
        "instruction_sample_to_state_and_lookup": """
            SELECT COUNT(*)
            FROM compute_instruction_sample AS instruction_sample
            LEFT JOIN compute_pc_sample_state AS sample_state
                ON instruction_sample.pc_sample_state_uuid =
                    sample_state.pc_sample_state_uuid
            LEFT JOIN compute_instruction_sample_lookup AS sample_lookup
                ON instruction_sample.instruction_sample_lookup_uuid =
                    sample_lookup.instruction_sample_lookup_uuid
            WHERE sample_state.pc_sample_state_uuid IS NULL
                OR sample_lookup.instruction_sample_lookup_uuid IS NULL
        """,
    }
    orphan_counts = {
        relationship: db_session.execute(text(query)).scalar_one()
        for relationship, query in orphan_queries.items()
    }
    assert orphan_counts == dict.fromkeys(orphan_queries, 0)

    pc_sampling_rows = (
        db_session
        .execute(
            text(
                "SELECT pid, kernel_uuid, kernel_name, count "
                "FROM compute_pc_sampling_summary_view ORDER BY pid"
            )
        )
        .mappings()
        .all()
    )
    assert [row["kernel_name"] for row in pc_sampling_rows] == [
        "vecCopy",
        "vecCopy",
    ]
    # Both rows name the same kernel; pid is what keeps them apart.
    assert len({row["kernel_uuid"] for row in pc_sampling_rows}) == 1
    pc_sampling_mapping = {row["pid"]: row["count"] for row in pc_sampling_rows}
    expected_pc_sampling_mapping = {
        line.kernel_symbol.code_object_store.pid: line.pc_sample_state.total_count
        for line in instruction_lines
    }
    assert pc_sampling_mapping == expected_pc_sampling_mapping
    assert sum(pc_sampling_mapping.values()) == 3

    kernel_rows = (
        db_session
        .execute(
            text(
                "SELECT kernel_uuid, kernel_name, dispatch_count "
                "FROM compute_kernel_view ORDER BY kernel_uuid"
            )
        )
        .mappings()
        .all()
    )
    # The kernel view inner-joins dispatches, so the shared kernel appears once
    # and carries both processes' dispatches.
    assert [row["kernel_name"] for row in kernel_rows] == ["vecCopy"]
    assert kernel_rows[0]["dispatch_count"] == 2
    # Every sampling row resolves to a kernel the kernel view exposes.
    assert {row["kernel_uuid"] for row in pc_sampling_rows} == {
        row["kernel_uuid"] for row in kernel_rows
    }


def test_run_analysis_exports_process_scoped_pc_sampling_csv(
    tmp_path,
):
    """Export one PC sampling CSV row per process, resolvable in kernel.csv."""
    try:
        orm.Database.init(":memory:")
        database_session = orm.Database.get_session()
        assert database_session is not None
        analyzer = make_pc_sampling_only_database_analyzer(
            str(tmp_path),
            [
                make_colliding_pc_sampling_tool_data(42, 1),
                make_colliding_pc_sampling_tool_data(99, 2),
            ],
        )

        run_analysis_with_materialized_views(analyzer)

        expected_pc_sampling_mapping = {
            pid: count
            for pid, count in database_session.execute(
                text(
                    "SELECT pid, count "
                    "FROM compute_pc_sampling_summary_view ORDER BY pid"
                )
            )
        }
        expected_kernel_mapping = {
            kernel_uuid: dispatch_count
            for kernel_uuid, dispatch_count in database_session.execute(
                text(
                    "SELECT kernel_uuid, dispatch_count "
                    "FROM compute_kernel_view ORDER BY kernel_uuid"
                )
            )
        }

        csv_directory = tmp_path / "csv"
        orm.Database.write_csv_dir(csv_directory)

        output_filenames = {path.name for path in csv_directory.iterdir()}
        assert output_filenames == VIEW_CSV_FILENAMES
        assert "dispatch.csv" not in output_filenames

        pc_sampling_frame = pd.read_csv(csv_directory / "pc_sampling_summary.csv")
        kernel_frame = pd.read_csv(csv_directory / "kernel.csv")
        assert "pid" in pc_sampling_frame.columns
        assert "code_object_id" in pc_sampling_frame.columns
        assert "pid" not in kernel_frame.columns
        assert list(pc_sampling_frame["kernel_name"]) == ["vecCopy", "vecCopy"]
        assert list(kernel_frame["kernel_name"]) == ["vecCopy"]

        pc_sampling_csv_mapping = {
            row.pid: row.count for row in pc_sampling_frame.itertuples(index=False)
        }
        assert pc_sampling_csv_mapping == {42: 1, 99: 2}
        assert pc_sampling_csv_mapping == expected_pc_sampling_mapping

        kernel_csv_mapping = {
            row.kernel_uuid: row.dispatch_count
            for row in kernel_frame.itertuples(index=False)
        }
        assert kernel_csv_mapping == expected_kernel_mapping
        assert set(kernel_csv_mapping.values()) == {2}

        # T1 regression guard: every sampling row must resolve in kernel.csv.
        assert set(pc_sampling_frame["kernel_uuid"]) <= set(kernel_frame["kernel_uuid"])
    finally:
        cleanup_database()


def test_run_analysis_keeps_mixed_counter_and_pc_sampling_ownership(
    db_session,
    tmp_path,
):
    """Attribute every process's sampling lines to the dispatched kernel."""
    workload_path = str(tmp_path)
    first_tool_data = make_colliding_pc_sampling_tool_data(42, 1)
    second_tool_data = make_colliding_pc_sampling_tool_data(99, 2)
    second_tool_data["code_objects"][0]["load_base"] = 0x3000
    tool_data_records = [first_tool_data, second_tool_data]

    process_id_by_load_base = {0x1000: 42, 0x3000: 99}
    sample_count_by_process_id = {42: 1, 99: 2}
    for load_base, process_id in process_id_by_load_base.items():
        (tmp_path / f"{process_id}_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "retained ISA",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        tool_data_records,
    )
    analyzer._roofline_data_per_kernel = {
        workload_path: pd.DataFrame([{"kernel_name": "vecCopy", "total_flops": 64.0}])
    }
    analyzer._metrics_info_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "name": "Counter metric",
                "metric_id": "1.1",
                "description": "Counter-derived value",
                "unit": "cycles",
                "table_name": "Counter",
                "sub_table_name": "Counter values",
            }
        ])
    }
    analyzer._kernel_values_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "metric_id": "1.1",
                "kernel_name": "vecCopy",
                "value_name": "avg",
                "value": 12.5,
            }
        ])
    }

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()
    aggregate_kernel = dispatch.kernel
    assert dispatch.dispatch_id == 7
    assert aggregate_kernel.kernel_name == "vecCopy"

    metric_value = db_session.query(orm.KernelMetricValue).one()
    roofline_data = db_session.query(orm.KernelRooflineData).one()
    assert metric_value.kernel_uuid == aggregate_kernel.kernel_uuid
    assert metric_value.value == 12.5
    assert roofline_data.kernel_uuid == aggregate_kernel.kernel_uuid
    assert roofline_data.total_flops == 64.0

    # One kernel row per name, so the sampling lines attribute to the same
    # kernel the counter dispatch created and stay reachable from kernel.csv.
    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 1
    assert kernels == [aggregate_kernel]

    stores = (
        db_session
        .query(orm.CodeObjectStore)
        .order_by(orm.CodeObjectStore.load_base)
        .all()
    )
    assert [(store.pid, store.code_object_id, store.load_base) for store in stores] == [
        (42, 5, 0x1000),
        (99, 5, 0x3000),
    ]

    chain_by_process_id = {}
    for store in stores:
        instruction_by_offset = {
            line.code_object_offset: line for line in store_instruction_lines(store)
        }
        assert set(instruction_by_offset) == {0x10, 0x30}

        sampled_line = instruction_by_offset[0x10]
        isa_line = instruction_by_offset[0x30]
        assert (
            sampled_line.kernel_symbol.kernel_uuid == isa_line.kernel_symbol.kernel_uuid
        )
        assert isa_line.pc_sample_state is None

        process_id = store.pid
        assert process_id_by_load_base[store.load_base] == process_id
        assert (
            sampled_line.pc_sample_state.total_count
            == sample_count_by_process_id[process_id]
        )
        chain_by_process_id[process_id] = (
            sampled_line.kernel_symbol.kernel_uuid,
            store.code_object_uuid,
            sampled_line.instruction_uuid,
            isa_line.instruction_uuid,
        )

    assert set(chain_by_process_id) == {42, 99}
    (
        kernel_uuid_chain,
        code_object_uuid_chain,
        sampled_instruction_uuid_chain,
        isa_instruction_uuid_chain,
    ) = zip(*chain_by_process_id.values())
    # Process identity lives on the code object, not the kernel.
    assert set(kernel_uuid_chain) == {aggregate_kernel.kernel_uuid}
    assert len(set(code_object_uuid_chain)) == 2
    assert len(set(sampled_instruction_uuid_chain)) == 2
    assert len(set(isa_instruction_uuid_chain)) == 2

    assert len(kernel_instruction_lines(aggregate_kernel)) == 4
    assert aggregate_kernel.dispatches
    assert aggregate_kernel.metric_values
    assert aggregate_kernel.roofline_data_points

    sample_states = query_instruction_sample_states(db_session).all()
    assert len(sample_states) == 2
    assert all(
        state.instruction_line.kernel_symbol.kernel_uuid == aggregate_kernel.kernel_uuid
        for state in sample_states
    )
    assert query_instruction_stall_reasons(db_session).count() == 2
    assert db_session.query(orm.InstructionSample).count() == 2
    assert db_session.query(orm.InstructionLine).count() == 4


def test_run_analysis_does_not_register_filtered_pc_sampling_symbols(
    db_session,
    tmp_path,
):
    """Exclude sampled symbols whose kernels were filtered from dispatches."""
    workload_path = str(tmp_path)
    analyzer = make_counter_backed_database_analyzer(
        workload_path,
        ["1", "pc_sampling"],
        [make_pc_sampling_tool_data()],
    )

    run_analysis_with_existing_database(analyzer)

    dispatch = db_session.query(orm.Dispatch).one()

    kernels = db_session.query(orm.Kernel).all()
    assert len(kernels) == 1
    assert {kernel.kernel_name for kernel in kernels} == {"vecCopy"}

    instruction_lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in instruction_lines] == [0x10]

    # vecAdd was filtered out of the dispatches, so only vecCopy's line survives
    # and it attributes to the dispatched kernel.
    sampled_line = instruction_lines[0]
    assert sampled_line.kernel_symbol.code_object_store.pid == 42
    assert sampled_line.kernel_symbol.kernel_uuid == dispatch.kernel_uuid
    assert sampled_line.pc_sample_state.total_count == 1
    assert query_instruction_sample_states(db_session).count() == 1
    assert query_instruction_stall_reasons(db_session).count() == 1
    assert db_session.query(orm.InstructionSample).count() == 1


# =============================================================================
# Code-object ISA ingestion (add_code_object_isa)
# =============================================================================


def make_disasm_code_object(code_object_id, instructions, symbol_name="sym"):
    """Build one code_obj_info code object with a single symbol."""
    return {
        "id": code_object_id,
        "symbols": [
            {
                "name": symbol_name,
                "virtual_address": instructions[0]["virtual_address"],
                "instructions": instructions,
            }
        ],
    }


def make_source_workload_tool_data_records(
    workload_path,
    snapshot_sources,
    sampled_sources,
    source_path_map=None,
):
    """Create PC-sampling inputs whose comments point at snapshot sources.

    A path named only in sampled_sources is absent from the snapshot.
    source_path_map pairs raw DWARF paths with canonical ones.
    """
    workload_path.mkdir(parents=True, exist_ok=True)
    for original_source_path, content in snapshot_sources.items():
        snapshot_path = (
            workload_path / "src" / Path(original_source_path).relative_to("/")
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(content, encoding="utf-8")

    if source_path_map:
        map_path = workload_path / "src" / "42_source_map.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        map_path.write_text(
            json.dumps({"source_paths": source_path_map}), encoding="utf-8"
        )

    tool_data = make_pc_sampling_tool_data()
    tool_data["strings"]["pc_sample_comments"] = list(sampled_sources)
    return [tool_data]


def fetch_source_lines_by_workload(session):
    """Group (file path, digest, line, content) tuples under each workload."""
    source_lines_by_workload = {}
    for source_file in session.query(orm.SourceFile).all():
        source_lines_by_workload.setdefault(source_file.workload.sub_name, []).extend(
            (
                source_file.file_path,
                source_file.md5_checksum,
                source_line.line_number,
                source_line.content,
            )
            for source_line in source_file.source_lines
        )
    return {
        workload_name: sorted(
            rows, key=lambda row: (row[0], row[2] is not None, row[2])
        )
        for workload_name, rows in source_lines_by_workload.items()
    }


def make_source_export_analyzer(tmp_path):
    """Build an analyzer and the source export its result folder should hold.

    Each workload holds one file its samples name and one it does not, so the
    export is pinned to the files the database records.
    """
    unreferenced_source_path = "/workspace/include/unreferenced.hpp"
    tool_data_per_workload = {}
    expected_exported_files = {}
    for workload_name, referenced_source_path, contents in (
        ("first", "/workspace/src/first.cpp", "int first;\n"),
        ("second", "/workspace/src/second.cpp", "int second;\n"),
    ):
        workload_path = tmp_path / "workloads" / workload_name / "run"
        tool_data_per_workload[str(workload_path)] = (
            make_source_workload_tool_data_records(
                workload_path,
                {
                    referenced_source_path: contents,
                    unreferenced_source_path: "int unreferenced;\n",
                },
                [f"{referenced_source_path}:1"],
            )
        )
        expected_exported_files[
            Path(workload_name)
            / "run"
            / "source"
            / Path(referenced_source_path).relative_to("/")
        ] = contents.encode()

    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload)
    result_path = tmp_path / "run"
    return analyzer, result_path, expected_exported_files


def run_source_export_analysis(analyzer, run_folder):
    """Build the database, write it into run_folder, and clear the database state."""
    run_folder.mkdir()
    try:
        with ExitStack() as patch_stack:
            patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
            patch_stack.enter_context(
                patch(
                    "rocprof_compute_analyze.analysis_db.get_version",
                    return_value={"version": "test", "sha": "test"},
                )
            )
            analyzer.build_database(run_folder / "analysis.db")
            analyzer.write_outputs(run_folder)
    finally:
        cleanup_database()


def read_exported_source_files(result_path):
    """Return the exported source files under a CSV result folder."""
    exported_files = common.read_binary_file_tree(
        result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
    )
    return {
        exported_path: contents
        for exported_path, contents in exported_files.items()
        if source_snapshot_analysis.SOURCE_EXPORT_DIRECTORY_NAME in exported_path.parts
        # The per-source-line files beside each copy have their own test.
        and not exported_path.name.endswith(
            per_source_line_export.SOURCE_LINE_FILE_SUFFIX
        )
    }


def read_view_csv_files(result_path):
    """Return view CSV contents keyed by filename."""
    return {
        csv_path.name: csv_path.read_bytes() for csv_path in result_path.glob("*.csv")
    }


def write_view_csv_files_and_capture(
    original_write_csv_dir,
    captured_view_csv_files,
    csv_directory,
):
    """Write view CSVs and capture their contents before source export."""
    original_write_csv_dir(csv_directory)
    captured_view_csv_files.update(read_view_csv_files(csv_directory))


def test_write_outputs_writes_the_database_and_the_view_csvs_together(tmp_path):
    """One run writes analysis.db, every view CSV, and the per-kernel export."""
    analyzer, result_path, _expected_exported_files = make_source_export_analyzer(
        tmp_path
    )

    run_source_export_analysis(analyzer, result_path)

    assert (result_path / "analysis.db").is_file()
    assert (result_path / "kernel.csv").is_file()
    assert (result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME).is_dir()


def test_write_outputs_closes_the_database_session(tmp_path):
    """The session is closed once every artifact is written."""
    analyzer, result_path, _expected_exported_files = make_source_export_analyzer(
        tmp_path
    )
    result_path.mkdir()
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(analyzer, "run_analysis_metrics"))
        patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.get_version",
                return_value={"version": "test", "sha": "test"},
            )
        )
        analyzer.build_database(result_path / "analysis.db")
    try:
        analyzer.write_outputs(result_path)

        assert orm.Database.get_session() is None
    finally:
        cleanup_database()


def test_run_analysis_source_export_scopes_workloads_and_preserves_view_csvs(
    tmp_path,
):
    """Isolate workload exports without changing the generated view CSVs."""
    analyzer, result_path, expected_exported_files = make_source_export_analyzer(
        tmp_path
    )
    view_csvs_before_source_export = {}
    original_write_csv_dir = orm.Database.write_csv_dir

    with ExitStack() as patch_stack:
        patch_stack.enter_context(
            patch.object(
                orm.Database,
                "write_csv_dir",
                side_effect=partial(
                    write_view_csv_files_and_capture,
                    original_write_csv_dir,
                    view_csvs_before_source_export,
                ),
            )
        )
        export_source_snapshot_files = patch_stack.enter_context(
            patch(
                "rocprof_compute_analyze.analysis_db.export_source_snapshot_files",
                wraps=source_snapshot_analysis.export_source_snapshot_files,
            )
        )
        run_source_export_analysis(analyzer, result_path)

    # The analyzer names each export folder, rather than the export rederiving
    # it, so a folder cannot drift from the workload row it belongs to.
    export_arguments = export_source_snapshot_files.call_args.kwargs
    assert (
        export_arguments["export_directory"] == result_path / "per_kernel_pc_sampling"
    )
    assert [
        (snapshot.workload_name, snapshot.workload_sub_name)
        for snapshot in export_arguments["workload_source_snapshots"]
    ] == [("first", "run"), ("second", "run")]

    view_csvs_after_analysis = read_view_csv_files(result_path)
    assert set(view_csvs_before_source_export) == VIEW_CSV_FILENAMES
    assert set(view_csvs_after_analysis) == VIEW_CSV_FILENAMES
    assert view_csvs_after_analysis == view_csvs_before_source_export

    exported_source_files = read_exported_source_files(result_path)
    assert exported_source_files == expected_exported_files

    # Every path the CSV records locates its copy by dropping the leading "/".
    source_lines_frame = pd.read_csv(result_path / "source_lines.csv")
    assert {
        Path(file_path).relative_to("/")
        for file_path in source_lines_frame["file_path"]
    } == {
        # The first three components name the workload and the source folder.
        Path(*exported_path.parts[3:])
        for exported_path in exported_source_files
    }


def test_run_analysis_keeps_each_workloads_source_files_apart(db_session, tmp_path):
    """Two workloads sharing a basename each keep their own absolute path."""
    tool_data_per_workload = {}
    for workload_name, source_path in (
        ("first", "/home/u/first/src/a.cpp"),
        ("second", "/opt/projects/second/src/a.cpp"),
    ):
        workload_path = tmp_path / workload_name
        tool_data_per_workload[str(workload_path)] = (
            make_source_workload_tool_data_records(
                workload_path,
                {source_path: "int first;\nint second;\n"},
                [f"{source_path}:1", f"{source_path}:2"],
            )
        )
    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload)

    run_analysis_with_materialized_views(analyzer)

    file_paths_by_workload = {
        workload_name: sorted({row[0] for row in rows})
        for workload_name, rows in fetch_source_lines_by_workload(db_session).items()
    }
    assert file_paths_by_workload == {
        "first": ["/home/u/first/src/a.cpp"],
        "second": ["/opt/projects/second/src/a.cpp"],
    }
    assert db_session.query(orm.SourceFile).count() == 2


def test_run_analysis_stores_lines_no_instruction_references(db_session, tmp_path):
    """A referenced file is stored whole, not just the lines a comment names."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int first;\nint second;\nint third;\n"},
        ["/home/u/app/vcopy.cpp:2", "/home/u/app/vcopy.cpp:2"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    digest = "86812da3721fe4b16ee110ad3dbcbbfa"
    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/vcopy.cpp", digest, 1, "int first;"),
        ("/home/u/app/vcopy.cpp", digest, 2, "int second;"),
        ("/home/u/app/vcopy.cpp", digest, 3, "int third;"),
    ]


def test_run_analysis_records_source_file_missing_from_snapshot(db_session, tmp_path):
    """A referenced file absent from the snapshot still gets its rows."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/present.cpp": "int present;\n"},
        ["/home/u/app/present.cpp:1", "/home/u/app/absent.cpp:7"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/absent.cpp", None, 7, None),
        (
            "/home/u/app/present.cpp",
            "d4aeed8f3a65c2f28866bd51dff2ebfd",
            1,
            "int present;",
        ),
    ]


def test_run_analysis_resolves_raw_dwarf_path_to_canonical_path(db_session, tmp_path):
    """A file whose raw DWARF path is not canonical is read from its copy."""
    workload_path = tmp_path / "workload"
    raw_dwarf_path = "/home/u/app/build/../include/vcopy.hpp"
    canonical_path = "/home/u/app/include/vcopy.hpp"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {canonical_path: "int first;\nint second;\n"},
        [f"{raw_dwarf_path}:2"],
        source_path_map={raw_dwarf_path: canonical_path},
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    source_file = db_session.query(orm.SourceFile).one()
    assert source_file.file_path == canonical_path
    # A checksum at all means the snapshot copy was found.
    assert source_file.md5_checksum is not None
    assert [
        (source_line.line_number, source_line.content)
        for source_line in source_file.source_lines
    ] == [(1, "int first;"), (2, "int second;")]


def test_run_analysis_records_line_past_end_of_source_file(db_session, tmp_path):
    """A frame naming a line the snapshot copy lacks gets a contentless row."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int only;\n"},
        ["/home/u/app/vcopy.cpp:1", "/home/u/app/vcopy.cpp:99"],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    digest = "ac72447959bb8fac84d23eea9b103598"
    assert fetch_source_lines_by_workload(db_session)["workload"] == [
        ("/home/u/app/vcopy.cpp", digest, 1, "int only;"),
        ("/home/u/app/vcopy.cpp", digest, 99, None),
    ]


def test_run_analysis_links_frames_innermost_first(db_session, tmp_path):
    """An inline stack is stored in order and rebuilt by the summary view."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {
            "/opt/rocm/hip.h": "line one\nline two\n",
            "/home/u/app/vcopy.cpp": "int a;\nint b;\n",
        },
        [
            "/opt/rocm/hip.h:? -> /opt/rocm/hip.h:2 -> /home/u/app/vcopy.cpp:1",
            "/home/u/app/vcopy.cpp:2",
        ],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    chained_line = (
        db_session.query(orm.InstructionLine).filter_by(code_object_offset=0x10).one()
    )
    assert [
        (
            frame.frame_index,
            frame.source_line.source_file.file_path,
            frame.source_line.line_number,
        )
        for frame in chained_line.source_lines
    ] == [
        (0, "/opt/rocm/hip.h", None),
        (1, "/opt/rocm/hip.h", 2),
        (2, "/home/u/app/vcopy.cpp", 1),
    ]

    rebuilt_sources = {
        offset: source
        for offset, source in db_session.execute(
            text(
                "SELECT offset, source FROM compute_pc_sampling_summary_view "
                "ORDER BY offset"
            )
        )
    }
    assert rebuilt_sources == {
        0x10: ("/opt/rocm/hip.h:? -> /opt/rocm/hip.h:2 -> /home/u/app/vcopy.cpp:1"),
        0x20: "/home/u/app/vcopy.cpp:2",
    }


def test_add_code_object_isa_adds_unsampled_lines(db_session):
    """Un-sampled instructions of a dispatched kernel are added and attributed
    via the mangled-name join; a disassembly offset that matches an already-
    sampled offset inserts no duplicate row; ISA of an un-dispatched symbol is
    not stored at all; a non-surviving pid's disassembly is skipped."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    load_base = 0x1000
    try:
        # 0x1010 -> offset 0x10 is already sampled; 0x1030 -> offset 0x30 is new
        # and joins vecCopy through its mangled ELF name; 0x1040 -> offset 0x40
        # belongs to a symbol never dispatched, so it is dropped entirely.
        code_objects = [
            {
                "id": 5,
                "symbols": [
                    {
                        "name": "_Z7vecCopyv",
                        "virtual_address": load_base + 0x10,
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x10,
                                "name": "v_mov",
                                "comment": "",
                            },
                            {
                                "virtual_address": load_base + 0x30,
                                "name": "s_nop",
                                "comment": "c",
                            },
                        ],
                    },
                    {
                        "name": "_Z7unknownv",
                        "virtual_address": load_base + 0x40,
                        "instructions": [
                            {
                                "virtual_address": load_base + 0x40,
                                "name": "s_endpgm",
                                "comment": "",
                            }
                        ],
                    },
                ],
            }
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )
        # A non-surviving pid (metadata pid is 42) reuses code_object_id 5; its
        # 0x50 offset must never be stored since only pid 42's load_base is known.
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": load_base + 0x50,
                                "name": "s_nop",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
            "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [make_pc_sampling_tool_data()]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
            {"gpu_arch": "gfx950"},
        )
        db_session.commit()

        lines = db_session.query(orm.InstructionLine).all()
        by_offset = {line.code_object_offset: line for line in lines}
        # Two sampled offsets + one new un-sampled offset; the un-dispatched
        # symbol's 0x40 line and the non-surviving pid's 0x50 line are dropped,
        # and 0x10 is not duplicated.
        assert set(by_offset) == {0x10, 0x20, 0x30}
        assert set(code_object_stores) == {(42, 5)}
        code_object_store = code_object_stores[(42, 5)]
        assert code_object_store.pid == 42
        assert db_session.query(orm.CodeObjectStore).one() is code_object_store
        assert code_object_store is by_offset[0x10].kernel_symbol.code_object_store
        # The disassembly-only line joins its kernel and carries no sample state.
        isa_line = by_offset[0x30]
        assert isa_line.kernel_symbol.kernel.kernel_name == "vecCopy"
        assert isa_line.pc_sample_state is None
        assert isa_line.instruction == "s_nop"
        # The sampled line at 0x10 kept its kernel attribution and sample state.
        assert by_offset[0x10].kernel_symbol.kernel.kernel_name == "vecCopy"
        assert by_offset[0x10].pc_sample_state is not None
        # Both belong to the same (reused) code object store.
        assert isa_line.kernel_symbol.code_object_store is code_object_store
        assert {line.kernel_symbol.code_object_uuid for line in by_offset.values()} == {
            code_object_store.code_object_uuid
        }
        # The sampling and ISA passes resolved to one shared symbol, whose own
        # offset the ISA pass backfilled from the symbol's virtual address.
        vec_copy_symbol = by_offset[0x10].kernel_symbol
        assert isa_line.kernel_symbol is vec_copy_symbol
        assert vec_copy_symbol.code_object_offset == 0x10
        # vecAdd's sampled line at 0x20 gets its own symbol in the same object.
        assert by_offset[0x20].kernel_symbol is not vec_copy_symbol
        assert db_session.query(orm.KernelSymbol).count() == 2
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_unsampled_code_objects_by_process(db_session):
    """Matching ISA-only code objects retain distinct process-local UUID chains."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["pc_sample_stochastic"] = []
        # code object 9 has a load_base and a dispatched kernel, but no samples.
        first_tool_data["code_objects"].append({
            "code_object_id": 9,
            "load_base": 0x2000,
        })
        first_tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        first_tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][-1]["load_base"] = 0x4000

        first_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x2000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        second_code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x4000 + 0x8, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": first_code_objects}), encoding="utf-8"
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({"code_objects": second_code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "helper": orm.Kernel(kernel_name="helper", workload=workload),
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload, workload_path)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        assert code_object_stores == {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
            {"gpu_arch": "gfx950"},
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 9), (99, 9)}
        first_store = code_object_stores[(42, 9)]
        second_store = code_object_stores[(99, 9)]
        assert (first_store.pid, second_store.pid) == (42, 99)
        assert first_store.load_base == 0x2000
        assert second_store.load_base == 0x4000
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = store_instruction_lines(first_store)[0]
        second_line = store_instruction_lines(second_store)[0]
        assert first_line.code_object_offset == second_line.code_object_offset == 0x8
        assert first_line.instruction == second_line.instruction == "s_endpgm"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert first_line.kernel_symbol.kernel is kernel_objs["helper"]
        assert second_line.kernel_symbol.kernel is kernel_objs["helper"]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_skips_code_object_without_load_base(db_session):
    """A code object with no known load_base cannot be offset-mapped, so it is
    skipped rather than stored with an inconsistent offset."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        tool_data = make_pc_sampling_tool_data()
        tool_data["code_objects"].append({"code_object_id": 9, "load_base": None})
        tool_data["kernel_symbols"].append({
            "kernel_id": 102,
            "code_object_id": 9,
            "kernel_name": "_Z6helperv.kd",
            "formatted_kernel_name": "helper",
        })
        tool_data["buffer_records"]["kernel_dispatch"].append(
            make_pc_sampling_dispatch(2, 102)
        )
        code_objects = [
            make_disasm_code_object(
                9,
                [{"virtual_address": 0x500, "name": "s_endpgm", "comment": ""}],
                symbol_name="_Z6helperv",
            )
        ]
        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({"code_objects": code_objects}), encoding="utf-8"
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        helper = orm.Kernel(kernel_name="helper", workload=workload)
        db_session.add(helper)
        kernel_objs = {"helper": helper}

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {workload_path: [tool_data]}
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
            {"gpu_arch": "gfx950"},
        )
        db_session.commit()

        # The store exists (its kernel was dispatched) but no ISA line was added.
        store = db_session.query(orm.CodeObjectStore).filter_by(code_object_id=9).one()
        assert code_object_stores[(42, 9)] is store
        assert store.pid == 42
        assert store_instruction_lines(store) == []
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_scopes_duplicate_offsets_by_process(db_session):
    """One process's ISA offset cannot suppress the matching offset in another."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000

        (Path(workload_path) / "42_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x1000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )
        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    make_disasm_code_object(
                        5,
                        [
                            {
                                "virtual_address": 0x3000 + 0x30,
                                "name": "s_shared",
                                "comment": "",
                            }
                        ],
                        symbol_name="_Z7vecCopyv",
                    )
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            kernel_name: orm.Kernel(
                kernel_name=kernel_name,
                workload=workload,
            )
            for kernel_name in ("vecCopy", "vecAdd")
        }
        for kernel in kernel_objs.values():
            db_session.add(kernel)

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        kernel_symbols = {}
        source_frames = make_source_frame_collector(workload)
        code_object_stores = analyzer.add_pc_sampling_data(
            workload_path,
            workload,
            kernel_objs,
            kernel_symbols,
            source_frames,
            sys_info={},
        )
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            kernel_symbols,
            source_frames,
            {"gpu_arch": "gfx950"},
        )
        db_session.commit()

        assert set(code_object_stores) == {(42, 5), (99, 5)}
        first_store = code_object_stores[(42, 5)]
        second_store = code_object_stores[(99, 5)]
        assert (first_store.pid, second_store.pid) == (42, 99)
        assert first_store.code_object_uuid != second_store.code_object_uuid

        first_line = next(
            line
            for line in store_instruction_lines(first_store)
            if line.code_object_offset == 0x30
        )
        second_line = next(
            line
            for line in store_instruction_lines(second_store)
            if line.code_object_offset == 0x30
        )
        assert first_line.instruction == second_line.instruction == "s_shared"
        assert first_line.instruction_uuid != second_line.instruction_uuid
        assert (
            first_line.kernel_symbol.code_object_uuid
            != second_line.kernel_symbol.code_object_uuid
        )
        assert first_line.kernel_symbol.kernel is kernel_objs["vecCopy"]
        assert second_line.kernel_symbol.kernel is kernel_objs["vecCopy"]
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_code_object_isa_requires_process_local_dispatch(db_session):
    """Store only symbols dispatched by the process that loaded the code object."""
    workload_path = common.get_output_dir()
    Path(workload_path).mkdir(parents=True, exist_ok=True)
    try:
        first_tool_data = make_pc_sampling_tool_data()
        first_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 100)
        ]

        second_tool_data = copy.deepcopy(first_tool_data)
        second_tool_data["metadata"]["pid"] = 99
        second_tool_data["code_objects"][0]["load_base"] = 0x3000
        second_tool_data["buffer_records"]["kernel_dispatch"] = [
            make_pc_sampling_dispatch(0, 101)
        ]

        (Path(workload_path) / "99_code_obj_info.json").write_text(
            json.dumps({
                "code_objects": [
                    {
                        "id": 5,
                        "symbols": [
                            {
                                "name": "_Z7vecCopyv",
                                "virtual_address": 0x3000 + 0x30,
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x30,
                                        "name": "not_dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                            {
                                "name": "vecAdd",
                                "virtual_address": 0x3000 + 0x40,
                                "instructions": [
                                    {
                                        "virtual_address": 0x3000 + 0x40,
                                        "name": "dispatched",
                                        "comment": "",
                                    }
                                ],
                            },
                        ],
                    }
                ]
            }),
            encoding="utf-8",
        )

        workload = orm.Workload(name="w", sub_name="s")
        db_session.add(workload)
        kernel_objs = {
            "vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload),
            "vecAdd": orm.Kernel(kernel_name="vecAdd", workload=workload),
        }
        db_session.add_all(kernel_objs.values())

        analyzer = db_analysis(MagicMock(), {})
        analyzer._pc_sampling_tool_data_per_workload = {
            workload_path: [first_tool_data, second_tool_data]
        }
        code_object_stores = {}
        analyzer.add_code_object_isa(
            workload_path,
            workload,
            kernel_objs,
            code_object_stores,
            {},
            make_source_frame_collector(workload),
            {"gpu_arch": "gfx950"},
        )
        db_session.commit()

        assert set(code_object_stores) == {(99, 5)}
        second_store = code_object_stores[(99, 5)]
        assert second_store.pid == 99
        assert second_store.code_object_uuid is not None
        stored_isa = {
            (line.instruction, line.kernel_symbol.kernel.kernel_name)
            for line in store_instruction_lines(second_store)
        }
        assert stored_isa == {("dispatched", "vecAdd")}
    finally:
        common.clean_output_dir(True, workload_path)


def test_add_pc_sampling_data_drops_lines_without_kernel(db_session):
    """Lines whose kernel is absent from kernel_objs (filtered out) are dropped
    along with their sample state and child counts, not attributed to no kernel."""
    workload_path = "/fake/workload"
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    # Only vecCopy survives filtering; vecAdd's line must be dropped.
    kernel_objs = {"vecCopy": orm.Kernel(kernel_name="vecCopy", workload=workload)}
    db_session.add(kernel_objs["vecCopy"])

    analyzer = db_analysis(MagicMock(), {})
    analyzer._pc_sampling_tool_data_per_workload = {
        workload_path: [make_pc_sampling_tool_data()]
    }
    analyzer.add_pc_sampling_data(
        workload_path,
        workload,
        kernel_objs,
        {},
        make_source_frame_collector(workload),
        sys_info={},
    )
    db_session.commit()

    lines = db_session.query(orm.InstructionLine).all()
    assert [line.code_object_offset for line in lines] == [0x10]
    retained_line = lines[0]
    assert retained_line.kernel_symbol.kernel is kernel_objs["vecCopy"]

    retained_sample_state = query_instruction_sample_states(db_session).one()
    assert retained_line.pc_sample_state is retained_sample_state

    stall_reason_count = query_instruction_stall_reasons(db_session).one()
    assert stall_reason_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.stall_reasons == [stall_reason_count]
    assert stall_reason_count.stall_reason_lookup.text == "WAITCNT"
    assert stall_reason_count.count == 1

    instruction_type_count = db_session.query(orm.InstructionSample).one()
    assert instruction_type_count.pc_sample_state is retained_sample_state
    assert retained_sample_state.instruction_samples == [instruction_type_count]
    assert instruction_type_count.instruction_sample_lookup.text == "VALU"
    assert instruction_type_count.count == 1


def test_run_analysis_keeps_sampled_row_without_a_comment(db_session, tmp_path):
    """An instruction with no comment keeps its row with a null source."""
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/home/u/app/vcopy.cpp": "int a;\n"},
        # The sampling path turns the empty comment into the "N/A" sentinel.
        ["/home/u/app/vcopy.cpp:1", ""],
    )
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): tool_data_records
    })

    run_analysis_with_materialized_views(analyzer)

    assert [
        source_file.file_path for source_file in db_session.query(orm.SourceFile)
    ] == ["/home/u/app/vcopy.cpp"]
    rebuilt_sources = {
        offset: source
        for offset, source in db_session.execute(
            text(
                "SELECT offset, source FROM compute_pc_sampling_summary_view "
                "ORDER BY offset"
            )
        )
    }
    assert rebuilt_sources == {0x10: "/home/u/app/vcopy.cpp:1", 0x20: None}


def test_run_analysis_aggregates_each_source_line_over_all_processes(
    db_session, tmp_path
):
    """Each source line aggregates the instructions of all processes.

    An instruction counts once toward each distinct source line of its inline
    stack, even if a one-line lambda puts a source line twice in that stack.
    """
    workload_path = tmp_path / "workload"
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/s/a.cpp": "int first;\nint second;\n"},
        ["/s/a.cpp:1 -> /s/a.cpp:2", "/s/a.cpp:2 -> /s/a.cpp:2"],
    )
    second_process_tool_data = copy.deepcopy(tool_data_records[0])
    second_process_tool_data["metadata"]["pid"] = 43
    analyzer = make_pc_sampling_database_analyzer({
        str(workload_path): [*tool_data_records, second_process_tool_data]
    })

    run_analysis_with_materialized_views(analyzer)

    line_states = {
        sample_state.source_line.line_number: (
            sample_state.total_count,
            sample_state.issue_count,
            sample_state.stall_count,
            {
                stall_reason.stall_reason_lookup.text: stall_reason.count
                for stall_reason in sample_state.stall_reasons
            },
        )
        for sample_state in db_session.query(orm.PCSampleState).filter(
            orm.PCSampleState.source_line_uuid.is_not(None)
        )
    }
    assert line_states == {
        1: (2, 0, 2, {"WAITCNT": 2}),
        2: (4, 2, 2, {"WAITCNT": 2}),
    }


def make_csv_run_analyzer(tmp_path, tool_data_per_workload, **filters):
    """Build a CSV-output analyzer over sampling-only workloads."""
    analyzer = make_pc_sampling_database_analyzer(tool_data_per_workload, **filters)
    result_path = tmp_path / "run"
    return analyzer, result_path


def read_per_kernel_isa_file(result_path, kernel_row, code_object_id=5, pid=42):
    """Return one exported ISA file as its header and its rows.

    The folder is named after the kernel's row in kernel.csv.
    """
    export_path = (
        result_path
        / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
        / ISA_WORKLOAD_NAME
        / ISA_WORKLOAD_SUB_NAME
        / f"{kernel_row['short_name']}_uuid_{kernel_row['kernel_uuid']}"
        / f"isa_code_object_id_{code_object_id}_pid_{pid}.csv"
    )
    with export_path.open(newline="", encoding="utf-8") as export_file:
        header, *rows = list(csv.reader(export_file))
    return header, rows


def stall_reason_columns(header):
    """Return the pivoted stall-reason columns of one ISA header."""
    return header[header.index("Active thread percent") + 1 : header.index("Source")]


def per_kernel_isa_paths(result_path):
    """Return the exported ISA files, relative to the per-kernel folder."""
    per_kernel_directory = result_path / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
    return sorted(
        str(export_path.relative_to(per_kernel_directory))
        for export_path in per_kernel_directory.rglob("isa_*.csv")
    )


def test_run_analysis_writes_one_isa_file_per_kernel_code_object_and_process(
    tmp_path,
):
    """A kernel compiled twice, and a code object in two processes, split up."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    first_process_tool_data = make_pc_sampling_tool_data()
    second_process_tool_data = make_pc_sampling_tool_data()
    second_process_tool_data["metadata"]["pid"] = 43
    # The same kernel, sampled a second time out of another code object.
    other_code_object_tool_data = make_pc_sampling_tool_data()
    other_code_object_tool_data["metadata"]["pid"] = 44
    other_code_object_tool_data["code_objects"] = [
        {"code_object_id": 6, "load_base": 0x2000}
    ]
    for sample in other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"]:
        sample["record"]["pc"]["code_object_id"] = 6
    for kernel_symbol in other_code_object_tool_data["kernel_symbols"]:
        kernel_symbol["code_object_id"] = 6

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {
            str(workload_path): [
                first_process_tool_data,
                second_process_tool_data,
                other_code_object_tool_data,
            ]
        },
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    assert per_kernel_isa_paths(result_path) == sorted(
        f"vector_copy/run/{kernel_row.short_name}_uuid_{kernel_row.kernel_uuid}"
        f"/isa_code_object_id_{code_object_id}_pid_{pid}.csv"
        for kernel_row in kernel_frame.itertuples()
        for code_object_id, pid in ((5, 42), (5, 43), (6, 44))
    )


@pytest.mark.parametrize("filter_gpu_ids", [(), ["0"]])
def test_run_analysis_isa_file_carries_the_kernels_sampled_lines(
    tmp_path, filter_gpu_ids
):
    """One kernel's file holds its own offsets, counts and source.

    Filtering on the workload's own GPU keeps every row it already had.
    """
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [make_pc_sampling_tool_data()]},
        filter_gpu_ids=filter_gpu_ids,
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_frame = pd.read_csv(result_path / "kernel.csv").set_index("kernel_name")
    header, rows = read_per_kernel_isa_file(result_path, kernel_frame.loc["vecCopy"])

    assert header == [
        "Instruction line number",
        "Code object offset",
        "Instruction line",
        "Instruction type",
        "Total count",
        "Active count",
        "Stall count",
        "Wave occupancy percent",
        "Active thread percent",
        "Stall WAITCNT",
        "Source",
        "Code object id",
        "Pid",
    ]
    assert rows == [
        [
            "1",
            str(0x10),
            "v_mov",
            "",
            "1",
            "0",
            "1",
            "",
            "",
            "1",
            "/s/a.cpp:1",
            "5",
            "42",
        ]
    ]


def test_run_analysis_isa_file_carries_the_static_instruction_type(tmp_path):
    """Real mnemonics get a pipeline; an unclassified one leaves the cell empty."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    tool_data["strings"]["pc_sample_instructions"] = [
        "v_mov_b32_e32 v1, 0",
        "not_an_instruction",
    ]
    # Both offsets belong to one kernel, so one file holds both rows.
    tool_data["kernel_symbols"][1]["kernel_id"] = 100
    tool_data["buffer_records"]["kernel_dispatch"][1]["dispatch_info"]["kernel_id"] = (
        100
    )

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data]},
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_row = pd.read_csv(result_path / "kernel.csv").iloc[0]
    header, rows = read_per_kernel_isa_file(result_path, kernel_row)

    type_column = header.index("Instruction type")
    assert [row[type_column] for row in rows] == ["VALU", ""]


def test_run_analysis_isa_stall_columns_follow_the_workloads_reasons(tmp_path):
    """Each observed reason gets a column, empty where it was not seen."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    tool_data["buffer_records"]["pc_sample_stochastic"][1]["record"]["snapshot"] = {
        "stall_reason": (
            "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT"
        )
    }
    tool_data["buffer_records"]["pc_sample_stochastic"][1]["record"]["wave_issued"] = (
        False
    )
    # Both offsets belong to one kernel, so one file holds both reasons.
    tool_data["kernel_symbols"][1]["kernel_id"] = 100
    tool_data["buffer_records"]["kernel_dispatch"][1]["dispatch_info"]["kernel_id"] = (
        100
    )

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data]},
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_row = pd.read_csv(result_path / "kernel.csv").iloc[0]
    header, rows = read_per_kernel_isa_file(result_path, kernel_row)

    assert stall_reason_columns(header) == ["Stall SLEEP_WAIT", "Stall WAITCNT"]
    sleep_index, waitcnt_index = (
        header.index("Stall SLEEP_WAIT"),
        header.index("Stall WAITCNT"),
    )
    assert [(row[sleep_index], row[waitcnt_index]) for row in rows] == [
        ("", "1"),
        ("1", ""),
    ]


def test_run_analysis_isa_carries_no_stall_columns_for_host_trap(tmp_path):
    """host_trap records no stall reason, so its files carry no stall column."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    tool_data["buffer_records"]["pc_sample_host_trap"] = [
        {
            "inst_index": 0,
            "record": {
                "pc": {"code_object_id": 5, "code_object_offset": 0x10},
                "dispatch_id": 0,
            },
        }
    ]
    tool_data["buffer_records"]["pc_sample_stochastic"] = []

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data]},
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_row = pd.read_csv(result_path / "kernel.csv").iloc[0]
    header, rows = read_per_kernel_isa_file(result_path, kernel_row)

    assert stall_reason_columns(header) == []
    # host_trap knows the sample landed, but not whether the wave issued.
    assert rows[0][4:7] == ["1", "", ""]


def test_run_analysis_writes_source_line_counts_beside_the_source_copy(tmp_path):
    """The per-source-line file sits beside the source copy and has ISA references.

    The instruction without a source line number goes to the unattributable
    row, which comes first.
    """
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data_records = make_source_workload_tool_data_records(
        workload_path,
        {"/s/a.cpp": "int first;\nint second;\nint third;\n"},
        ["/s/a.cpp:1", "/s/a.cpp:?"],
    )
    # Both offsets belong to one kernel, so both ISA references point into the
    # ISA file of that kernel.
    tool_data = tool_data_records[0]
    tool_data["kernel_symbols"][1]["kernel_id"] = 100
    tool_data["buffer_records"]["kernel_dispatch"][1]["dispatch_info"]["kernel_id"] = (
        100
    )
    analyzer, result_path = make_csv_run_analyzer(
        tmp_path, {str(workload_path): tool_data_records}
    )
    run_source_export_analysis(analyzer, result_path)

    source_directory = (
        result_path
        / per_kernel_isa_export.PER_KERNEL_DIRECTORY_NAME
        / ISA_WORKLOAD_NAME
        / ISA_WORKLOAD_SUB_NAME
        / source_snapshot_analysis.SOURCE_EXPORT_DIRECTORY_NAME
        / "s"
    )
    assert sorted(path.name for path in source_directory.iterdir()) == [
        "a.cpp",
        "a.cpp_pc_sampling.csv",
    ]
    kernel_row = pd.read_csv(result_path / "kernel.csv").iloc[0]
    isa_file = (
        f"{kernel_row.short_name}_uuid_{kernel_row.kernel_uuid}"
        "/isa_code_object_id_5_pid_42.csv"
    )
    with (source_directory / "a.cpp_pc_sampling.csv").open(
        newline="", encoding="utf-8"
    ) as source_line_file:
        header, *rows = list(csv.reader(source_line_file))

    assert header == [
        "Source line number",
        "Source line",
        "Total count",
        "Active count",
        "Stall count",
        "Wave occupancy percent",
        "Active thread percent",
        "Stall WAITCNT",
        "ISA references",
    ]
    assert rows == [
        ["", "Unattributable instructions", "1", "", "", "", "", "", f"{isa_file}:2"],
        [
            "1",
            "int first;",
            "1",
            "0",
            "1",
            "",
            "",
            "1",
            f"{isa_file}:1",
        ],
        ["2", "int second;", "", "", "", "", "", "", ""],
        ["3", "int third;", "", "", "", "", "", "", ""],
    ]


def test_run_analysis_kernel_filter_reaches_a_sampling_only_workload(tmp_path):
    """-k keeps one kernel's rows, and no code object left with nothing in it."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    tool_data = make_pc_sampling_tool_data()
    # The filter indexes kernels by descending dispatch duration.
    tool_data["buffer_records"]["kernel_dispatch"][0]["end_timestamp"] = 10
    other_code_object_tool_data = make_pc_sampling_tool_data()
    other_code_object_tool_data["metadata"]["pid"] = 43
    other_code_object_tool_data["code_objects"] = [
        {"code_object_id": 6, "load_base": 0x2000}
    ]
    # Only the kernel the filter drops was sampled out of this code object.
    other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"] = [
        other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"][1]
    ]
    other_code_object_tool_data["buffer_records"]["pc_sample_stochastic"][0]["record"][
        "pc"
    ]["code_object_id"] = 6
    for kernel_symbol in other_code_object_tool_data["kernel_symbols"]:
        kernel_symbol["code_object_id"] = 6

    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [tool_data, other_code_object_tool_data]},
        filter_kernel_ids=[0],
    )
    run_source_export_analysis(analyzer, result_path)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    assert list(kernel_frame["kernel_name"]) == ["vecCopy"]
    summary_frame = pd.read_csv(result_path / "pc_sampling_summary.csv")
    assert list(summary_frame["offset"]) == [0x10]
    # The second code object held only the kernel the filter dropped.
    assert set(summary_frame["code_object_id"]) == {5}
    assert per_kernel_isa_paths(result_path) == [
        f"vector_copy/run/{kernel_frame['short_name'].iloc[0]}"
        f"_uuid_{kernel_frame['kernel_uuid'].iloc[0]}"
        "/isa_code_object_id_5_pid_42.csv"
    ]


@pytest.mark.parametrize("filter_dispatch_ids", [["2"], [">1"], ["> 1"]])
def test_run_analysis_dispatch_filter_reaches_a_sampling_only_workload(
    tmp_path, filter_dispatch_ids
):
    """-d drops a dispatch, and the kernel that keeps none of its own."""
    workload_path = tmp_path / "workloads" / ISA_WORKLOAD_NAME / ISA_WORKLOAD_SUB_NAME
    analyzer, result_path = make_csv_run_analyzer(
        tmp_path,
        {str(workload_path): [make_pc_sampling_tool_data()]},
        filter_dispatch_ids=filter_dispatch_ids,
    )

    run_source_export_analysis(analyzer, result_path)

    kernel_frame = pd.read_csv(result_path / "kernel.csv")
    assert list(kernel_frame["kernel_name"]) == ["vecAdd"]
    assert per_kernel_isa_paths(result_path) == [
        f"vector_copy/run/{kernel_frame['short_name'].iloc[0]}"
        f"_uuid_{kernel_frame['kernel_uuid'].iloc[0]}"
        "/isa_code_object_id_5_pid_42.csv"
    ]


# =============================================================================
# TESTS FOR Analysis DB mode: Analysis DB mode code path
# =============================================================================


def test_calc_roofline_data_early_exit_on_empty_roofline_df(monkeypatch):
    """Test calc_roofline_data exits early when roofline data is empty.

    This test verifies that when the roofline dataframe (ID 402) is empty
    or filtered out, the function logs a warning and skips that workload
    without adding it to the result dictionary.
    """
    workload_path = "/mock/workload/path"
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel1", "kernel2"],
        "Start_Timestamp": [100, 200],
        "End_Timestamp": [150, 300],
    })
    roofline_df = pd.DataFrame()  # Empty roofline dataframe triggers early exit

    analyzer = make_roofline_calc_analyzer(workload_path, pmc_df, roofline_df)

    warning_messages = []

    def mock_warning(msg):
        warning_messages.append(msg)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_warning", mock_warning
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_debug", lambda msg: None
    )

    # Call the actual function
    result = db_analysis.calc_roofline_data(analyzer)

    # Verify early exit behavior
    assert result == {}, "Should return no kernel data when roofline data is empty"
    assert len(warning_messages) == 1, "Should log one warning message"
    assert "Roofline data is filtered out or not found" in warning_messages[0]
    assert workload_path in warning_messages[0]


# =============================================================================
# Static instruction type tests
# =============================================================================


def test_both_instruction_line_paths_share_one_instruction_type(db_session):
    """The sampling and disassembly paths classify lines the same way, and a
    pipeline seen twice is stored once."""
    workload = orm.Workload(name="w", sub_name="s")
    kernel = orm.Kernel(kernel_name="vecCopy", workload=workload)
    code_object_store = orm.CodeObjectStore(
        workload=workload, pid=42, code_object_id=5, load_base=0x1000
    )
    db_session.add_all([workload, kernel, code_object_store])
    source_frames = make_source_frame_collector(workload)
    kernel_symbols = {}

    db_analysis._add_instruction_line(
        InstructionLineRecord(
            code_object_offset=0x10,
            kernel_name="vecCopy",
            instruction="v_mov_b32_e32 v1, 0",
            source=SOURCE_LINE_MISSING,
            total_count=1,
            issue_count=1,
            stall_count=0,
            stall_reasons={},
            inst_types={},
            active_thread_percent=None,
            wave_occupancy_percent=None,
        ),
        code_object_store,
        kernel,
        kernel_symbols,
        source_frames,
    )
    db_analysis._add_symbol_isa(
        kernel_symbols[(42, 5, "vecCopy")],
        CodeObjectSymbol(
            name="vecCopy",
            virtual_address=0x1020,
            instructions=[
                CodeObjectInstruction(
                    virtual_address=0x1020,
                    instruction="v_add_f32_e32 v0, v1, v2",
                    source=None,
                ),
                CodeObjectInstruction(
                    virtual_address=0x1028,
                    instruction="s_waitcnt lgkmcnt(0)",
                    source=None,
                ),
                CodeObjectInstruction(
                    virtual_address=0x1030,
                    instruction="not_an_instruction v0",
                    source=None,
                ),
            ],
        ),
        source_frames,
    )
    db_session.commit()

    pipeline_by_offset = {
        line.code_object_offset: (
            line.instruction_type_lookup.text
            if line.instruction_type_lookup is not None
            else None
        )
        for line in db_session.query(orm.InstructionLine).all()
    }
    assert pipeline_by_offset == {
        0x10: "VALU",
        0x20: "VALU",
        0x28: "INTERNAL",
        0x30: None,
    }
    # The two VALU lines share one lookup row.
    assert db_session.query(orm.InstructionTypeLookup).count() == 2


def test_calc_roofline_data_includes_all_kernels(monkeypatch):
    """calc_roofline_data computes roofline for all kernels, not just top N."""
    NUM_KERNELS = 15
    kernel_names = [f"kernel_{i:02d}" for i in range(NUM_KERNELS)]

    # Two dispatches per kernel: a unique long dispatch and a fixed short
    # dispatch. The long duration decreases with index so the sort in
    # calc_roofline_data produces a deterministic descending order.
    long_durations = [500 - i * 30 for i in range(NUM_KERNELS)]
    short_duration = 50

    rows = []
    t = 0
    for i in range(NUM_KERNELS):
        rows.append((kernel_names[i], t, t + long_durations[i]))
        t += long_durations[i] + 10
        rows.append((kernel_names[i], t, t + short_duration))
        t += short_duration + 10

    pmc_df = pd.DataFrame(
        rows, columns=["Kernel_Name", "Start_Timestamp", "End_Timestamp"]
    )

    roofline_metrics = [
        "Performance (GFLOPs)",
        "AI HBM",
        "AI L2",
        "AI L1",
        "AI L0",
        "AI LDS",
    ]
    roofline_df = pd.DataFrame({
        "Metric": roofline_metrics,
        "Value": ["expr_" + m.lower().replace(" ", "_") for m in roofline_metrics],
    })

    workload_path = "/mock/workload/path"
    analyzer = make_roofline_calc_analyzer(workload_path, pmc_df, roofline_df)
    evaluated_row_counts = []

    def evaluate_roofline_metric(name, value, kernel_pmc_df, sys_info):
        evaluated_row_counts.append(len(kernel_pmc_df))
        return 42.0

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.db_analysis.evaluate",
        evaluate_roofline_metric,
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_warning", lambda msg: None
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_debug", lambda msg: None
    )

    kernel_data = db_analysis.calc_roofline_data(analyzer)

    assert len(kernel_data) == 1
    df = kernel_data[workload_path]
    assert len(df) == NUM_KERNELS, f"Expected {NUM_KERNELS} kernels, got {len(df)}"
    assert list(df["kernel_name"]) == kernel_names, (
        f"Expected kernels sorted by duration descending, got {list(df['kernel_name'])}"
    )

    expected_columns = [
        "total_flops",
        "l0_cache_data",
        "l1_cache_data",
        "l2_cache_data",
        "hbm_cache_data",
        "lds_cache_data",
    ]
    for col in expected_columns:
        assert col in df.columns
        assert (df[col] == 42.0).all()

    assert evaluated_row_counts == [2] * (NUM_KERNELS * len(roofline_metrics))


# =============================================================================
# Memory chart layout (Workload.memory_chart_layout_extdata)
# =============================================================================


def test_run_analysis_stores_the_memory_chart_layout_file_unchanged(
    db_session, tmp_path
):
    workload = run_memory_chart_workload(tmp_path, "gfx942")
    layout_file = Layouts.for_arch("gfx942").path
    assert workload.memory_chart_layout_extdata == json.loads(
        layout_file.read_text(encoding="utf-8")
    )


def test_run_analysis_stores_null_for_an_arch_without_a_layout(db_session, tmp_path):
    # Also covers a layout file skipped as broken
    workload = run_memory_chart_workload(tmp_path, "gfx1030")
    assert workload.memory_chart_layout_extdata is None


# =============================================================================
# Metric coverage: every metric table is stored per kernel and per workload
# =============================================================================


def test_run_analysis_stores_plot_points_and_memory_chart_per_kernel_and_workload(
    db_session, tmp_path
):
    workload_path = str(tmp_path)
    arch_config = schema.ArchConfig()
    arch_config.panel_configs = {
        300: {
            "id": 300,
            "title": "Memory Chart",
            "data source": [{"metric_table": {"id": 301, "title": "Memory Chart"}}],
        },
        400: {
            "id": 400,
            "title": "Roofline",
            "data source": [
                {"metric_table": {"id": 402, "title": "Roofline Plot Points"}}
            ],
        },
        3000: {
            "id": 3000,
            "title": "Memory Bandwidth Analysis",
            "data source": [{"metric_table": {"id": 3001, "title": "L1 Cache"}}],
        },
    }
    arch_config.dfs = {
        301: pd.DataFrame(
            {
                "Metric": ["VGPR"],
                "Value": ["to_avg(raw_pmc_df['Arch_VGPR'])"],
                "Unit": ["Registers"],
            },
            index=pd.Index(["3.1.0"], name="Metric_ID"),
        ),
        3001: pd.DataFrame(
            {
                "Metric": ["L1 Cache - Peak VGPR"],
                "Value": ["to_max(raw_pmc_df['Arch_VGPR'])"],
                "Unit": ["Registers"],
            },
            index=pd.Index(["30.1.0"], name="Metric_ID"),
        ),
        402: pd.DataFrame(
            {
                "Metric": ["Performance (GFLOPs)"],
                "Value": ["to_sum(raw_pmc_df['SQ_FLOPS'])"],
                "Unit": ["GFLOP/s"],
            },
            index=pd.Index(["4.2.0"], name="Metric_ID"),
        ),
    }
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_b"],
        "Start_Timestamp": [0, 10, 30],
        "End_Timestamp": [10, 30, 40],
        "Arch_VGPR": [4.0, 8.0, 16.0],
        "SQ_FLOPS": [1.0, 2.0, 4.0],
    })

    analyzer = make_counter_backed_database_analyzer(workload_path, ["3", "4"], [])
    analyzer._arch_configs = {"gfx942": arch_config}
    analyzer._pmc_df_per_workload = {workload_path: pmc_df}
    analyzer._dispatch_data_per_workload = {
        workload_path: pd.DataFrame([
            {
                "dispatch_id": dispatch_id,
                "kernel_name": kernel_name,
                "gpu_id": 0,
                "start_timestamp": start,
                "end_timestamp": end,
            }
            for dispatch_id, (kernel_name, start, end) in enumerate(
                [("kernel_a", 0, 10), ("kernel_a", 10, 30), ("kernel_b", 30, 40)],
                start=1,
            )
        ])
    }
    (
        analyzer._metrics_info_data_per_workload,
        analyzer._metric_expression_data_per_workload,
    ) = analyzer.calc_metrics_data()
    analyzer._typed_sys_info_per_workload = analyzer.calc_typed_sys_info()
    (
        analyzer._kernel_values_data_per_workload,
        analyzer._workload_values_data_per_workload,
    ) = analyzer.calc_expressions()

    run_analysis_with_existing_database(analyzer)

    session = orm.Database.get_session()
    kernel_values = {
        (row.kernel.kernel_name, row.metric.metric_id): row.value
        for row in session.query(orm.KernelMetricValue).filter(
            orm.KernelMetricValue.value_name == "Value"
        )
    }
    assert kernel_values[("kernel_a", "4.2.0")] == pytest.approx(3.0e9)
    assert kernel_values[("kernel_b", "4.2.0")] == pytest.approx(4.0e9)
    assert kernel_values[("kernel_a", "3.1.0")] == pytest.approx(6.0)
    assert kernel_values[("kernel_b", "3.1.0")] == pytest.approx(16.0)
    assert kernel_values[("kernel_a", "30.1.0")] == pytest.approx(8.0)
    assert kernel_values[("kernel_b", "30.1.0")] == pytest.approx(16.0)

    workload_values = {
        row.metric.metric_id: row.value
        for row in session.query(orm.WorkloadMetricValue).filter(
            orm.WorkloadMetricValue.value_name == "Value"
        )
    }
    assert workload_values["4.2.0"] == pytest.approx(7.0e9)
    assert workload_values["3.1.0"] == pytest.approx(28.0 / 3.0)
    assert workload_values["30.1.0"] == pytest.approx(16.0)

    plot_point = (
        session
        .query(orm.MetricDefinition)
        .filter(orm.MetricDefinition.metric_id == "4.2.0")
        .one()
    )
    assert plot_point.sub_table_name == "Roofline Plot Points"


# =============================================================================

# Roofline ceilings stored per workload (FR-23)
# =============================================================================


@pytest.mark.parametrize(
    "gpu_arch, roofline_text",
    [
        ("gfx942", None),
        ("gfx942", "device,FP32Flops\n0,2000.0,extra\n1,9999.0\n"),
        ("gfx908", "device,HBMBw\n0,5300\n1,6300\n"),
    ],
    ids=["missing", "invalid", "arch-without-roofline"],
)
def test_roofline_ceiling_rows_need_a_supported_arch_and_a_valid_roofline(
    tmp_path, gpu_arch, roofline_text
):
    """No ceiling rows without a roofline benchmark or a validated roofline.csv."""
    if roofline_text is not None:
        write_roofline_csv(tmp_path, roofline_text)
    analyzer = make_ceiling_analyzer(str(tmp_path), gpu_arch)

    assert analyzer.calc_roofline_ceiling_rows() == {}


@pytest.mark.parametrize(
    "roofline_text, expected_message",
    [
        ("device,HBMBw\n0,5300\n0,6300\n", "duplicate device id"),
        ("device,HBMBw\n1,5300\n2,6300\n", "device id 0 occurs 0 times"),
    ],
    ids=["duplicate-device-id", "no-device-zero"],
)
def test_roofline_ceiling_rows_warn_when_the_device_zero_row_is_unusable(
    tmp_path, roofline_text, expected_message
):
    """validate_roofline_csv passes these files; the device-0 reader rejects them."""
    write_roofline_csv(tmp_path, roofline_text)
    analyzer = make_ceiling_analyzer(str(tmp_path), "gfx942")

    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        rows = analyzer.calc_roofline_ceiling_rows()

    assert rows == {}
    console_warning_mock.assert_called_once()
    message = console_warning_mock.call_args.args[0]
    assert "not stored" in message
    assert expected_message in message


def test_roofline_ceiling_rows_take_device_zero_in_base_units(tmp_path):
    """Device 0 gives the ceilings, MALL included, in base units. A zero mean is not
    a ceiling, and the device-1 row is ignored."""
    write_roofline_csv(
        tmp_path,
        "device,HBMBw,MALLBw,FP32Flops,I8Ops\n0,5300,9000,81000,0\n"
        "1,6300,9100,91000,50000\n",
    )
    analyzer = make_ceiling_analyzer(str(tmp_path), "gfx942")

    rows = analyzer.calc_roofline_ceiling_rows()

    assert {
        (row.kind, row.memory_level, row.data_type, row.pipe): (row.unit, row.value)
        for row in rows[str(tmp_path)]
    } == {
        ("bandwidth", "HBM", None, None): ("Bytes/s", 5.3e12),
        ("bandwidth", "MALL", None, None): ("Bytes/s", 9e12),
        ("compute", None, "FP32", "VALU"): ("FLOP/s", 8.1e13),
    }


def test_roofline_ceiling_rows_read_the_mi200_fixture():
    """The MI200 mem_levels fixture gives 4 bandwidth and 7 compute ceilings."""
    fixture_dir = (
        Path(__file__).resolve().parents[2] / "workloads" / "mem_levels_HBM" / "MI200"
    )
    analyzer = make_ceiling_analyzer(str(fixture_dir), "gfx90a")

    rows = analyzer.calc_roofline_ceiling_rows()[str(fixture_dir)]

    assert len(rows) == 11
    hbm = next(row for row in rows if row.memory_level == "HBM")
    assert hbm.value == pytest.approx(1388.7961e9)


def test_run_analysis_stores_the_roofline_ceilings_of_each_workload(
    db_session, tmp_path
):
    """Each workload's ceilings reach the database with their roof parts and unit."""
    write_roofline_csv(tmp_path, "device,HBMBw,MALLBw,FP32Flops\n0,5300,9000,81000\n")
    analyzer = make_counter_backed_database_analyzer(str(tmp_path), ["3"], [])
    analyzer._runs[str(tmp_path)].sys_info = pd.DataFrame([{"gpu_arch": "gfx942"}])
    analyzer._roofline_ceiling_rows_per_workload = analyzer.calc_roofline_ceiling_rows()

    run_analysis_with_existing_database(analyzer)

    workload = db_session.query(orm.Workload).one()
    assert {
        (ceiling.kind, ceiling.memory_level, ceiling.data_type, ceiling.pipe): (
            ceiling.unit,
            ceiling.value,
        )
        for ceiling in workload.roofline_ceilings
    } == {
        ("bandwidth", "HBM", None, None): ("Bytes/s", 5.3e12),
        ("bandwidth", "MALL", None, None): ("Bytes/s", 9e12),
        ("compute", None, "FP32", "VALU"): ("FLOP/s", 8.1e13),
    }


# =============================================================================
# Calculation parity: Percent of Peak, roofline peaks, typed sys vars, NULL, guard
# =============================================================================


def test_percent_of_peak_is_stored_for_every_flagged_metric(tmp_path, gfx942_sys_info):
    """The flag read through load_options still yields Percent of Peak rows.

    The percentage is 100 * unrounded value / roofline peak (FR-12), and an
    unflagged metric gets no such row.
    """
    drain_evaluation_diagnostics()
    workload_path = str(tmp_path)
    write_roofline_csv(tmp_path, "device,FP32Flops\n0,2000.0\n1,9999.0\n")
    analyzer = make_pct_of_peak_analyzer(workload_path, gfx942_sys_info)
    analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()
    analyzer._typed_sys_info_per_workload = analyzer.calc_typed_sys_info()

    kernel_values, workload_values = calc_values_with_stubbed_dual_issue(analyzer)

    expected_pct = 100 * (100 / 3) / 2000.0
    assert select_metric_value(
        workload_values[workload_path], "7.1.0", "Percent of Peak"
    ) == pytest.approx(expected_pct)
    assert select_metric_value(
        kernel_values[workload_path], "7.1.0", "Percent of Peak"
    ) == pytest.approx(expected_pct)
    assert not (
        (workload_values[workload_path]["metric_id"] == "7.1.1")
        & (workload_values[workload_path]["value_name"] == "Percent of Peak")
    ).any()


@pytest.mark.parametrize(
    "gpu_arch, column",
    [
        ("gfx950", "MFMAF6F4Flops"),
        ("gfx1250", "WMMAF6F4Flops"),
        ("gfx1250", "WMMAF64Flops"),
    ],
    ids=["gfx950-mfma-f6f4", "gfx1250-wmma-f6f4", "gfx1250-wmma-fp64"],
)
def test_roofline_peaks_cover_every_column_from_the_first_row(
    tmp_path, gpu_arch, column
):
    """Columns the old subset dropped (F6F4, WMMA FP64) become peak variables."""
    write_roofline_csv(tmp_path, f"device,{column}\n0,1234.5\n1,99.0\n")
    analyzer = db_analysis(SimpleNamespace(output_name=None), {})
    analyzer._runs = {
        str(tmp_path): schema.Workload(sys_info=pd.DataFrame([{"gpu_arch": gpu_arch}]))
    }

    ceilings = analyzer.calc_roofline_ceilings()

    assert ceilings[str(tmp_path)][column] == pytest.approx(1234.5)


@pytest.mark.parametrize(
    "roofline_text",
    [None, "device,FP32Flops\n0,2000.0,extra\n1,9999.0\n"],
    ids=["missing", "invalid"],
)
def test_missing_or_invalid_roofline_gives_null_peaks_and_one_warning(
    tmp_path, roofline_text, gfx942_sys_info
):
    """A roofline.csv that is missing or invalid: NULL peaks, one warning, no
    failure warnings from the expressions that use the peaks (FR-13, FR-15)."""
    drain_evaluation_diagnostics()
    workload_path = str(tmp_path)
    if roofline_text is not None:
        write_roofline_csv(tmp_path, roofline_text)
    analyzer = make_pct_of_peak_analyzer(workload_path, gfx942_sys_info)

    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as console_warning_mock:
        analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()
    assert workload_path not in analyzer._roofline_ceilings_per_workload
    console_warning_mock.assert_called_once()
    assert "NULL" in console_warning_mock.call_args.args[0]

    analyzer._typed_sys_info_per_workload = analyzer.calc_typed_sys_info()
    _, workload_values = calc_values_with_stubbed_dual_issue(analyzer)

    assert pd.isna(select_metric_value(workload_values[workload_path], "7.1.0", "Peak"))
    assert pd.isna(
        select_metric_value(workload_values[workload_path], "7.1.0", "Percent of Peak")
    )
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as failure_warning_mock, patch(
        "rocprof_compute_analyze.analysis_db.console_debug"
    ):
        report_evaluation_diagnostics()
    failure_warning_mock.assert_not_called()


@pytest.mark.parametrize(
    "cu_per_gpu",
    [np.nan, 0],
    ids=["missing", "zero"],
)
def test_expression_on_missing_or_zero_sys_var_is_null_not_zero(
    cu_per_gpu, gfx942_sys_info
):
    """A sys var that cannot be used makes the expression NULL, not 0 (FR-14, FR-15)."""
    workload_path = "/fake/workload"
    analyzer = db_analysis(SimpleNamespace(), {})
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{**gfx942_sys_info, "cu_per_gpu": cu_per_gpu}])
        )
    }
    with patch("utils.metrics.evaluation_pipeline.console_warning"):
        typed_sys_info = analyzer.calc_typed_sys_info()[workload_path]
    expression_df = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["Avg"],
        "value": ["to_sum(raw_pmc_df['SQ_WAVES']) / ammolite__cu_per_gpu"],
    })
    pmc_df = pd.DataFrame({"SQ_WAVES": [1.0, 2.0]})

    result = db_analysis.calc_dataframe_expressions(
        pmc_df, typed_sys_info, expression_df
    )

    assert pd.isna(result.iloc[0])
    drain_evaluation_diagnostics()


def test_builtin_capped_at_missing_cu_per_gpu_is_null(gfx942_sys_info):
    """numActiveCUs caps at cu_per_gpu, so a missing cu_per_gpu makes it NULL, not
    an uncapped number (FR-14, FR-15)."""
    workload_path = "/fake/workload"
    analyzer = db_analysis(SimpleNamespace(), {})
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{**gfx942_sys_info, "cu_per_gpu": np.nan}])
        )
    }
    with patch("utils.metrics.evaluation_pipeline.console_warning"):
        typed_sys_info = analyzer.calc_typed_sys_info()[workload_path]
    expression_df = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["Avg"],
        "value": ["ammolite__numActiveCUs"],
    })
    pmc_df = pd.DataFrame({
        "SQ_BUSY_CU_CYCLES": [1.0e6],
        "GRBM_GUI_ACTIVE": [1000.0],
    })

    result = db_analysis.calc_dataframe_expressions(
        pmc_df, typed_sys_info, expression_df
    )

    assert pd.isna(result.iloc[0])
    drain_evaluation_diagnostics()


def test_specs_correction_string_values_still_evaluate(gfx942_sys_info):
    """A --specs-correction value is a string in sys_info; it is typed first (FR-14)."""
    workload_path = "/fake/workload"
    analyzer = db_analysis(SimpleNamespace(), {})
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{**gfx942_sys_info, "cu_per_gpu": "304"}])
        )
    }
    typed_sys_info = analyzer.calc_typed_sys_info()[workload_path]
    expression_df = pd.DataFrame({
        "metric_id": ["1.1"],
        "value_name": ["Avg"],
        "value": ["to_sum(raw_pmc_df['SQ_WAVES']) / ammolite__cu_per_gpu"],
    })
    pmc_df = pd.DataFrame({"SQ_WAVES": [1.0, 2.0]})

    result = db_analysis.calc_dataframe_expressions(
        pmc_df, typed_sys_info, expression_df
    )

    assert result.iloc[0] == pytest.approx(3.0 / 304)


@pytest.mark.parametrize(
    "ip_blocks, should_halt",
    [("all", True), ("roofline", False)],
    ids=["counter-run-halts", "roofline-only-exempt"],
)
def test_pre_processing_halts_on_idle_gpu_before_any_write(
    ip_blocks, should_halt, gfx942_sys_info
):
    """FR-16: zero GPU activity halts pre-processing before any dispatch is built."""
    workload_path = "/fake/workload"
    pmc_df = pd.DataFrame({
        "Kernel_Name": ["vecCopy", "vecCopy"],
        "GRBM_GUI_ACTIVE": [1000, 0],
    })
    analyzer = db_analysis(
        SimpleNamespace(output_name=None, output_format="database"), {}
    )
    analyzer._runs = {
        workload_path: schema.Workload(
            sys_info=pd.DataFrame([{**gfx942_sys_info, "ip_blocks": ip_blocks}])
        )
    }
    with ExitStack() as patch_stack:
        patch_stack.enter_context(patch.object(OmniAnalyze_Base, "pre_processing"))
        patch_stack.enter_context(
            patch.object(db_analysis, "pc_sampling_collected", return_value=False)
        )
        patch_stack.enter_context(
            patch.object(db_analysis, "calc_roofline_ceilings", return_value={})
        )
        patch_stack.enter_context(
            patch.object(
                db_analysis,
                "calc_pmc_df_data",
                return_value={workload_path: pmc_df},
            )
        )
        patch_stack.enter_context(
            patch.object(
                db_analysis,
                "apply_pmc_filters",
                return_value={workload_path: pmc_df},
            )
        )
        dispatch_mock = patch_stack.enter_context(
            patch.object(db_analysis, "calc_dispatch_data", return_value={})
        )
        patch_stack.enter_context(
            patch.object(db_analysis, "calc_metrics_data", return_value=({}, {}))
        )
        patch_stack.enter_context(
            patch.object(db_analysis, "calc_expressions", return_value=({}, {}))
        )
        patch_stack.enter_context(
            patch.object(db_analysis, "calc_roofline_data", return_value={})
        )
        patch_stack.enter_context(
            patch("utils.metrics.evaluation_pipeline.console_warning")
        )
        error_mock = patch_stack.enter_context(
            patch(
                "utils.metrics.evaluation_pipeline.console_error",
                side_effect=SystemExit,
            )
        )

        if should_halt:
            with pytest.raises(SystemExit):
                analyzer.pre_processing()
            error_mock.assert_called_once_with("Halting execution for warning above.")
            dispatch_mock.assert_not_called()
        else:
            analyzer.pre_processing()
            error_mock.assert_not_called()
            dispatch_mock.assert_called_once()


def test_peak_absent_from_a_valid_roofline_is_null_without_failure_warning(
    tmp_path, gfx942_sys_info
):
    """A valid roofline.csv that does not measure a peak leaves it NULL. Its
    expression gives no KeyError failure warning (FR-13, FR-15)."""
    drain_evaluation_diagnostics()
    workload_path = str(tmp_path)
    write_roofline_csv(tmp_path, "device,FP32Flops\n0,2000.0\n1,9999.0\n")
    analyzer = make_pct_of_peak_analyzer(workload_path, gfx942_sys_info)
    analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()
    analyzer._typed_sys_info_per_workload = analyzer.calc_typed_sys_info()
    # Only FP32Flops is measured, so FP64Flops has no peak in roofline.csv.
    analyzer._metric_expression_data_per_workload[workload_path].loc[
        lambda df: df["value_name"] == "Peak", "value"
    ] = "ammolite__FP64Flops_empirical_peak"

    _, workload_values = calc_values_with_stubbed_dual_issue(analyzer)

    assert pd.isna(select_metric_value(workload_values[workload_path], "7.1.0", "Peak"))
    with patch(
        "rocprof_compute_analyze.analysis_db.console_warning"
    ) as failure_warning_mock, patch(
        "rocprof_compute_analyze.analysis_db.console_debug"
    ):
        report_evaluation_diagnostics()
    failure_warning_mock.assert_not_called()


def test_workload_stores_every_roofline_row_with_its_device_id(db_session, tmp_path):
    """The workload keeps the first roofline.csv row as its benchmark data. The
    benchmark table keeps every row with its device id, so a later device-0 row
    is not lost. The device id stays an integer, and NaN or Inf cells are stored
    as NULL (FR-13)."""
    write_roofline_csv(
        tmp_path,
        "device,FP32Flops,FP64Flops,HBMBw,WMMAF64Flops\n"
        "1,2000.0,,inf,512.5\n"
        "0,9999.0,3.0,4.0,5.0\n",
    )
    analyzer = make_counter_backed_database_analyzer(str(tmp_path), ["3"], [])
    analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()

    run_analysis_with_existing_database(analyzer)

    workload = db_session.query(orm.Workload).one()
    assert workload.roofline_bench_extdata == {
        "device": 1,
        "FP32Flops": 2000.0,
        "FP64Flops": None,
        "HBMBw": None,
        "WMMAF64Flops": 512.5,
    }
    assert type(workload.roofline_bench_extdata["device"]) is int
    benchmark_rows = (
        db_session
        .query(orm.WorkloadBenchmark)
        .order_by(orm.WorkloadBenchmark.workload_benchmark_uuid)
        .all()
    )
    assert [row.device_id for row in benchmark_rows] == [1, 0]
    assert benchmark_rows[1].benchmark_extdata == {
        "device": 0,
        "FP32Flops": 9999.0,
        "FP64Flops": 3.0,
        "HBMBw": 4.0,
        "WMMAF64Flops": 5.0,
    }


def test_unparseable_roofline_csv_warns_and_stores_null_benchmark_data(
    db_session, tmp_path
):
    """A roofline.csv that validates but pandas cannot parse warns once and stores
    NULL benchmark data, without failing the analysis (FR-13)."""
    # The quote opens a field that never closes, so the csv module reads to EOF.
    write_roofline_csv(tmp_path, 'device,FP32Flops\n0,"2000\n1,3\n')
    analyzer = make_counter_backed_database_analyzer(str(tmp_path), ["3"], [])

    with patch("rocprof_compute_analyze.analysis_db.console_warning") as warning_mock:
        analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()
    run_analysis_with_existing_database(analyzer)

    warning_mock.assert_called_once()
    assert "Roofline peaks are NULL for" in warning_mock.call_args.args[0]
    assert db_session.query(orm.Workload).one().roofline_bench_extdata is None
    assert db_session.query(orm.WorkloadBenchmark).count() == 0


@pytest.mark.parametrize(
    "roofline_text",
    [None, "device,FP32Flops\n0,2000.0,extra\n1,9999.0\n"],
    ids=["missing", "invalid"],
)
def test_workload_benchmark_data_is_null_without_a_valid_roofline_csv(
    db_session, tmp_path, roofline_text
):
    """A missing or invalid roofline.csv stores NULL benchmark data (FR-13).

    This guards the NULL path of the stored column and the empty benchmark table,
    so it also passes on the base code.
    """
    if roofline_text is not None:
        write_roofline_csv(tmp_path, roofline_text)
    analyzer = make_counter_backed_database_analyzer(str(tmp_path), ["3"], [])

    with patch("rocprof_compute_analyze.analysis_db.console_warning"):
        analyzer._roofline_ceilings_per_workload = analyzer.calc_roofline_ceilings()
    run_analysis_with_existing_database(analyzer)

    assert db_session.query(orm.Workload).one().roofline_bench_extdata is None
    assert db_session.query(orm.WorkloadBenchmark).count() == 0


# =============================================================================
# Stored values: unrounded, base units, independent of display options
# =============================================================================


def test_evaluate_round_keeps_every_digit():
    """ROUND in a config expression is for display, so the DB keeps full precision."""
    pmc_df = pd.DataFrame({"Counter1": [1, 2]})
    expression = build_eval_string("ROUND(AVG(Counter1), 0)")

    result = db_analysis.evaluate(
        "test_metric",
        expression,
        pmc_df,
        {},
    )

    assert result == 1.5


def test_builtin_num_active_cus_keeps_integer_rounding():
    """numActiveCUs uses ROUND as integer arithmetic, so it still rounds to an integer.

    Its formula feeds ROUND into MOD and TO_INT. Keeping every digit would
    change the count, so this guards against applying the DB-only unrounded
    ROUND to built-in variables.
    """
    # 4 * 75 / 8 = 37.5 active waves. ROUND gives 38, and the formula gives
    # 38 / 32 * 8 + MIN(MOD(38, 32), 8) = 15.5, which TO_INT turns into 15.
    pmc_df = pd.DataFrame({"SQ_BUSY_CU_CYCLES": [75], "GRBM_GUI_ACTIVE": [8]})
    sys_info = {
        "gpu_arch": "gfx942",
        "num_xcd": 1,
        "max_waves_per_cu": 32,
        "cu_per_gpu": 304,
    }

    db_analysis.calc_builtin_vars(
        pmc_df, sys_info, ["$numActiveCUs", "$GRBM_GUI_ACTIVE_PER_XCD"]
    )

    assert sys_info["numActiveCUs"] == 15


def test_run_analysis_stores_values_in_base_units(tmp_path):
    """Scaled config units are stored as base units, and only their values change."""
    kernel_values, workload_values, _ = store_metric_values(tmp_path, 2, "ns")

    stored = {
        (metric_id, value_name): (unit, value)
        for metric_id, unit, value_name, value in kernel_values
    }
    # GB/s is bytes per ns, so the stored Bytes/s value is 1e9 times larger.
    assert stored[("4.1.9", "Value")] == ("Bytes/s", (1 / 3) * 1e9)
    assert stored[("4.1.9", "Peak (Empirical)")] == ("Bytes/s", 5324.8 * 1e9)
    assert stored[("2.1.0", "Avg")] == ("FLOP/s", 2.25 * 1e9)
    # Percent of Peak is a ratio of two values in the same unit, so it keeps its scale.
    assert stored[("4.1.9", "Percent of Peak")] == ("Bytes/s", 6.25)
    # Units that are already base units are stored as they are, unrounded.
    assert stored[("17.1.1", "Avg")] == ("Bytes/s", 1 / 3)
    assert stored[("3.1.0", "Value")] == ("Percent", 99.5)
    assert stored[("7.2.0", "Avg")] == ("Nanoseconds", 12.5)
    # KB per Wave is stored as Bytes per Wave, so the value is 1024 times larger.
    assert stored[("6.1.0", "Value")] == ("Bytes per Wave", 2.5 * 1024)

    assert workload_values == [("4.1.9", "Bytes/s", "Value", 0.1 * 1e9)]


def test_run_analysis_stored_values_ignore_decimal_and_time_unit(tmp_path):
    """The database holds the same values whatever --decimal and --time-unit say.

    This is a regression guard. The DB path never reads those options, so the
    test passes on base code too. It fails if a later change makes the stored
    values depend on them.
    """
    default_options = store_metric_values(tmp_path, 2, "ns")
    other_options = store_metric_values(tmp_path, 6, "us")

    assert other_options == default_options


# Memory bandwidth analysis rows
# =============================================================================


def test_membw_metric_tables_reads_the_avg_value_of_each_metric():
    """Each panel 3000 table gets its metrics' Avg values, and no other value."""
    arch_dfs = make_membw_arch_dfs()
    workload_values = make_membw_workload_values(
        {"L2 Cache Efficiency": 80.0},
        arch_dfs,
    )

    membw_tables = membw_metric_tables(arch_dfs, workload_values)

    assert sorted(membw_tables) == [3001, 3012, 3018]
    l2_table = membw_tables[3012]
    assert list(l2_table.columns) == ["Metric", "Avg", "Unit"]
    avg_by_metric = dict(zip(l2_table["Metric"], l2_table["Avg"]))
    assert avg_by_metric["L2 Cache Efficiency"] == 80.0
    assert pd.isna(avg_by_metric["L2 Back Pressure Indicator"])
    assert set(l2_table["Unit"]) == {"Percent"}


def test_membw_metric_tables_skips_tables_the_arch_config_lacks():
    """Without panel 3000 in the arch config, there is no membw input."""
    workload_values = make_membw_workload_values(MEMBW_UTCL1_HBM_AVG, {})

    assert membw_metric_tables({}, workload_values) == {}


@pytest.mark.parametrize(
    ("avg_by_metric_name", "expected_availability"),
    [
        (MEMBW_UTCL1_HBM_AVG, "full"),
        ({"L2 Cache Efficiency": 80.0}, "partial"),
        ({}, "unavailable"),
    ],
    ids=["full", "partial", "unavailable"],
)
def test_membw_analysis_rows_rebuild_the_result_they_came_from(
    db_session,
    avg_by_metric_name,
    expected_availability,
):
    """The stored rows rebuild the membw result they were written from, unchanged."""
    arch_dfs = make_membw_arch_dfs()
    result = run_membw_analysis(
        membw_metric_tables(
            arch_dfs,
            make_membw_workload_values(avg_by_metric_name, arch_dfs),
        ),
        "gfx950",
    )
    assert result.availability == expected_availability
    workload = orm.Workload(name="w", sub_name="s")
    db_session.add(workload)
    db_analysis.add_membw_analysis(workload, result)
    db_session.commit()

    assert rebuild_membw_result(db_session, workload) == result


def test_calc_membw_analysis_skips_a_workload_without_block_30_values(tmp_path):
    """A later workload with no block 30 values gets no membw result, even when
    the first workload was profiled with block 30.
    """
    membw_path = str(tmp_path / "membw")
    plain_path = str(tmp_path / "plain")
    analyzer = make_membw_analyzer(membw_path, "gfx950", True)
    analyzer._runs[plain_path] = schema.Workload(
        sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
    )
    analyzer._workload_values_data_per_workload[plain_path] = pd.DataFrame(
        columns=["metric_id", "value_name", "value"]
    )

    membw_results = analyzer.calc_membw_analysis()

    assert list(membw_results) == [membw_path]


@pytest.mark.parametrize(
    ("gpu_arch", "membw_analysis", "expects_analysis"),
    [
        ("gfx950", True, True),
        ("gfx950", False, False),
        ("gfx942", True, False),
    ],
    ids=["membw_profiled", "membw_not_profiled", "arch_without_tree_spec"],
)
def test_run_analysis_stores_membw_analysis_only_for_profiled_workloads(
    db_session,
    tmp_path,
    gpu_arch,
    membw_analysis,
    expects_analysis,
):
    """Only a membw-profiled workload on an arch with a tree spec gets rows."""
    analyzer = make_membw_analyzer(str(tmp_path), gpu_arch, membw_analysis)

    run_analysis_with_materialized_views(analyzer)

    analysis_row = db_session.query(orm.MembwAnalysis).one_or_none()
    if not expects_analysis:
        assert analysis_row is None
        assert db_session.query(orm.MembwNode).count() == 0
        assert db_session.query(orm.MembwGuidanceBlock).count() == 0
        return
    assert analysis_row.arch == gpu_arch
    assert analysis_row.availability == "full"
    assert analysis_row.nodes
    assert analysis_row.guidance_blocks
    # Each node gives one view row per supporting metric, or one row when it has none
    expected_view_row_count = sum(
        max(1, len(node.supporting_metrics)) for node in analysis_row.nodes
    )
    view_row_count = db_session.execute(
        text("SELECT COUNT(*) FROM compute_membw_node_view")
    ).scalar()
    assert view_row_count == expected_view_row_count
