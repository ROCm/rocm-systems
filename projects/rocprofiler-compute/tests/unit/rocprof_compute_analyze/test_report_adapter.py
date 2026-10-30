# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/report_adapter.py.

The parity tests run the CLI and DB pipelines on a workload and compare the
frames the adapter rebuilds with the CLI's own frames. The only cells allowed to
differ are the ones NFR-2 lists (FR-15 NULL values and Percent of Peak). The
synthetic tests build small analysis DBs by hand, to cover the rules the
workloads do not exercise.
"""

import argparse
import re
from io import StringIO
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import common
import pandas as pd
import pytest

from argparser import omniarg_parser
from config import rocprof_compute_home
from rocprof_compute_analyze.analysis_cli import cli_analysis
from rocprof_compute_analyze.analysis_db import db_analysis
from rocprof_compute_analyze.report_adapter import (
    NULL_TEXT,
    SYS_INFO_TABLE_ID,
    build_report_runs,
    build_report_workload,
)
from utils import analysis_orm as orm
from utils import parser, schema, tty
from utils.roofline_calc import calc_ai_analyze

WORKLOADS_DIR = Path(__file__).resolve().parents[2] / "workloads"
VCOPY_WORKLOAD = WORKLOADS_DIR / "vcopy/MI300X_A1"
ROOFLINE_WORKLOAD = WORKLOADS_DIR / "mem_levels_HBM/MI200"

KERNEL_TOP_TABLE_ID = 1
DISPATCH_TABLE_ID = 2
STANDARD_TABLE_IDS = (KERNEL_TOP_TABLE_ID, DISPATCH_TABLE_ID, SYS_INFO_TABLE_ID)
PERCENT_OF_PEAK = "Percent of Peak"
SYNTHETIC_METRIC_TABLE_ID = 1001
SYNTHETIC_SYS_INFO = {"gpu_arch": "gfx942", "workload_name": "app", "cu_per_gpu": 304}


class PipelineRuns(NamedTuple):
    """The CLI pipeline, the DB pipeline and the adapter's frames of one workload."""

    cli: cli_analysis
    db: db_analysis
    runs: Dict[str, schema.Workload]


def parse_analyze_args(workload_path: Path) -> argparse.Namespace:
    """Parse analyze options for one workload with the real option definitions."""
    parser_under_test = argparse.ArgumentParser(prog="tool")
    omniarg_parser(
        parser_under_test,
        rocprof_compute_home,
        common.SUPPORTED_ARCHS,
        {"ver_pretty": "rocprof-compute (unit test)"},
    )
    return parser_under_test.parse_args(["analyze", "-p", str(workload_path)])


def cleanup_database() -> None:
    """Close and clear the process-wide analysis database state."""
    current_session = orm.Database._session
    if current_session is not None:
        current_session.close()
    current_engine = orm.Database._engine
    if current_engine is not None:
        current_engine.dispose()
    orm.Database._session = None
    orm.Database._engine = None


def seal_database() -> None:
    """Commit the synthetic rows and create the views, as the DB pipeline does."""
    orm.Database.commit()
    orm.Database.create_views()


def run_both_pipelines(workload_path: Path) -> PipelineRuns:
    """Run the CLI pipeline and the DB pipeline, then build the adapter's runs."""
    args = parse_analyze_args(workload_path)

    cli = cli_analysis(args, common.SUPPORTED_ARCHS)
    cli.sanitize()
    cli.pre_processing()

    db = db_analysis(args, common.SUPPORTED_ARCHS)
    db.sanitize()
    db.pre_processing()
    db.build_database(Path("analysis.db"))
    # write_outputs creates the views before it writes them, so the adapter sees
    # the same views here.
    seal_database()
    workload_ids = {
        run_path: workload_obj.workload_id
        for run_path, workload_obj in zip(db._runs.keys(), db._workload_objs)
    }
    runs = build_report_runs(
        orm.Database.get_session(),
        workload_ids,
        db._arch_configs,
        args.time_unit,
    )
    return PipelineRuns(cli=cli, db=db, runs=runs)


@pytest.fixture(scope="module")
def vcopy_pipelines():
    pipelines = run_both_pipelines(VCOPY_WORKLOAD)
    yield pipelines
    cleanup_database()


@pytest.fixture(scope="module")
def roofline_pipelines():
    pipelines = run_both_pipelines(ROOFLINE_WORKLOAD)
    yield pipelines
    cleanup_database()


@pytest.fixture(params=["vcopy_pipelines", "roofline_pipelines"])
def parity_pipelines(request):
    """Each workload whose CLI and adapter frames are compared."""
    return request.getfixturevalue(request.param)


def only_workload(runs: Dict[str, schema.Workload]) -> schema.Workload:
    return next(iter(runs.values()))


def cell_matches(cli_value, adapter_value) -> bool:
    """Whether one cell of the adapter equals the CLI's, to float precision."""
    if isinstance(cli_value, str) or isinstance(adapter_value, str):
        return cli_value == adapter_value
    if pd.isna(cli_value) or pd.isna(adapter_value):
        return pd.isna(cli_value) and pd.isna(adapter_value)
    return float(adapter_value) == pytest.approx(float(cli_value), rel=1e-9)


def synthetic_arch_config() -> schema.ArchConfig:
    """A metric table with a bandwidth row, a NULL-able row and two ROUND shapes.

    The expressions are written the way the expression parser translates them.
    """
    template = pd.DataFrame(
        {
            "Metric": ["Bandwidth", "Rate", "Two rounds"],
            "Avg": [
                "to_round(to_avg(Bytes), 1)",
                "to_avg(Rate)",
                "to_round(to_avg(a), 1) + to_round(to_avg(b), 2)",
            ],
            "Unit": ["GB/s", "Count", "GB/s"],
            "Description": ["Memory bandwidth", "Count of rates", "Two rounds"],
        },
        index=pd.Index(["10.1.0", "10.1.1", "10.1.2"], name="Metric_ID"),
    )
    return schema.ArchConfig(
        panel_configs={
            0: {
                "id": 0,
                "title": "Top Stats",
                "data source": [
                    {
                        "raw_csv_table": {
                            "id": 1,
                            "title": "Top Kernels",
                            "source": "kernel_top",
                        }
                    },
                    {
                        "raw_csv_table": {
                            "id": 2,
                            "title": "Dispatch List",
                            "source": "dispatch_info",
                        }
                    },
                ],
            },
            10: {
                "id": 10,
                "title": "Synthetic",
                "data source": [
                    {
                        "metric_table": {
                            "id": SYNTHETIC_METRIC_TABLE_ID,
                            "title": "Synthetic Table",
                            "header": {
                                "metric": "Metric",
                                "value": "Avg",
                                "unit": "Unit",
                            },
                        }
                    }
                ],
            },
        },
        dfs={SYNTHETIC_METRIC_TABLE_ID: template},
        dfs_type={SYNTHETIC_METRIC_TABLE_ID: "metric_table"},
    )


def synthetic_args(time_unit: str = "ns") -> argparse.Namespace:
    return argparse.Namespace(
        decimal=2,
        view=None,
        normal_unit="per_kernel",
        time_unit=time_unit,
        include_cols=None,
        cols=None,
        verbose=0,
        max_stat_num=20,
    )


def add_synthetic_workload(session, total_duration_ns: Optional[int]) -> orm.Workload:
    workload = orm.Workload(
        name="app",
        sub_name="MI300X",
        sys_info_extdata=dict(SYNTHETIC_SYS_INFO),
        profiling_config_extdata={},
        total_duration_ns=total_duration_ns,
    )
    session.add(workload)
    session.flush()
    return workload


def add_synthetic_kernel(session, workload, name: str, durations_ns: List[int]):
    """Add a kernel whose dispatches last the given durations, in dispatch order."""
    kernel = orm.Kernel(kernel_name=name, workload=workload)
    session.add(kernel)
    for dispatch_id, duration_ns in enumerate(durations_ns, start=1):
        session.add(
            orm.Dispatch(
                dispatch_id=dispatch_id,
                gpu_id=0,
                start_timestamp=0,
                end_timestamp=duration_ns,
                kernel=kernel,
            )
        )
    return kernel


def add_synthetic_metric(
    session, workload, metric_id: str, unit: str, value_name: str, value
):
    """Store one workload metric value in the base unit, as the DB writer does."""
    metric = orm.MetricDefinition(
        name=f"metric {metric_id}",
        metric_id=metric_id,
        description="synthetic",
        table_name="table",
        sub_table_name="sub table",
        unit=unit,
        workload=workload,
    )
    session.add(metric)
    session.add(
        orm.WorkloadMetricValue(
            metric=metric,
            workload=workload,
            value_name=value_name,
            value=value,
        )
    )


def test_dispatch_and_sys_info_match_the_cli_frames(
    parity_pipelines,
):
    """Dispatch List and System Info match the CLI frames exactly."""
    cli_workload = only_workload(parity_pipelines.cli._runs)
    adapter_workload = only_workload(parity_pipelines.runs)

    for table_id in (DISPATCH_TABLE_ID, SYS_INFO_TABLE_ID):
        pd.testing.assert_frame_equal(
            adapter_workload.dfs[table_id],
            cli_workload.dfs[table_id],
            check_dtype=False,
        )


def test_top_kernels_match_the_cli_frame_except_an_unknown_percent(
    parity_pipelines,
):
    """Top Kernels match the CLI, except a Percent the CLI shows as NaN is N/A.

    The CLI shows NaN where the filtered total is zero (FR-15). Every other
    cell, including the times, is the CLI's value.
    """
    cli_frame = only_workload(parity_pipelines.cli._runs).dfs[KERNEL_TOP_TABLE_ID]
    adapter_frame = only_workload(parity_pipelines.runs).dfs[KERNEL_TOP_TABLE_ID]
    expected = cli_frame.astype({"Percent": object})
    expected.loc[cli_frame["Percent"].isna(), "Percent"] = NULL_TEXT

    pd.testing.assert_frame_equal(adapter_frame, expected, check_dtype=False)


def test_metric_tables_match_the_cli_frames_except_null_percent_of_peak(
    parity_pipelines,
):
    """Every metric table has the CLI's rows, columns and values.

    The one allowed difference is a NULL DB value of Percent of Peak: the CLI
    leaves that cell blank, and the adapter shows N/A (FR-15).
    """
    cli_workload = only_workload(parity_pipelines.cli._runs)
    adapter_workload = only_workload(parity_pipelines.runs)
    metric_table_ids = [
        table_id
        for table_id in adapter_workload.dfs
        if table_id not in STANDARD_TABLE_IDS
    ]
    assert metric_table_ids

    for table_id in metric_table_ids:
        cli_frame = cli_workload.dfs[table_id]
        adapter_frame = adapter_workload.dfs[table_id]
        assert list(adapter_frame.columns) == list(cli_frame.columns), table_id
        assert list(adapter_frame.index) == list(cli_frame.index), table_id
        for column in cli_frame.columns:
            for metric_id, cli_value, adapter_value in zip(
                cli_frame.index, cli_frame[column], adapter_frame[column]
            ):
                where = (table_id, metric_id, column)
                if cli_value == "" and adapter_value == NULL_TEXT:
                    assert column == PERCENT_OF_PEAK, where
                    continue
                assert cell_matches(cli_value, adapter_value), (
                    where,
                    cli_value,
                    adapter_value,
                )


def test_roofline_is_not_valid_for_a_workload_without_roofline_csv(vcopy_pipelines):
    """No valid roofline.csv means no peaks and no per-kernel roofline tables."""
    adapter_workload = only_workload(vcopy_pipelines.runs)

    assert adapter_workload.roofline_peaks.empty
    assert adapter_workload.roofline_metrics == {}


def test_roofline_peaks_and_per_kernel_tables_match_the_cli(roofline_pipelines):
    """The peaks and each kernel's rate and plot-point tables match the CLI's."""
    cli_workload = only_workload(roofline_pipelines.cli._runs)
    adapter_workload = only_workload(roofline_pipelines.runs)
    cli_arch_config = roofline_pipelines.cli._arch_configs[
        cli_workload.sys_info.iloc[0]["gpu_arch"]
    ]
    pmc_df = parser.apply_filters(cli_workload, debug=False)
    calc_ai_analyze(workload=cli_workload, pmc_df=pmc_df, arch_config=cli_arch_config)

    pd.testing.assert_frame_equal(
        adapter_workload.roofline_peaks,
        cli_workload.roofline_peaks,
        check_dtype=False,
    )
    assert adapter_workload.roofline_metrics
    assert list(adapter_workload.roofline_metrics) == list(
        cli_workload.roofline_metrics
    )
    for kernel_id, cli_metrics in cli_workload.roofline_metrics.items():
        adapter_metrics = adapter_workload.roofline_metrics[kernel_id]
        assert adapter_metrics["name"] == cli_metrics["name"]
        for table_key in ("ai_table", "calc_table"):
            pd.testing.assert_frame_equal(
                adapter_metrics[table_key],
                cli_metrics[table_key],
                check_dtype=False,
                rtol=1e-9,
            )


def test_top_stats_percent_is_each_kernel_share_of_the_stored_total(db_session):
    """Percent divides each kernel's time by the total the database stores."""
    workload = add_synthetic_workload(db_session, total_duration_ns=1000)
    add_synthetic_kernel(db_session, workload, "kernel_a", [100, 200])
    add_synthetic_kernel(db_session, workload, "kernel_b", [100])
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    top_kernels = frames.dfs[KERNEL_TOP_TABLE_ID]
    assert top_kernels["Kernel_Name"].tolist() == ["kernel_a", "kernel_b"]
    assert top_kernels["Percent"].tolist() == pytest.approx([30.0, 10.0])
    assert list(top_kernels.columns) == [
        "Kernel_Name",
        "Count",
        "Sum(ns)",
        "Mean(ns)",
        "Median(ns)",
        "Percent",
    ]


def test_top_stats_have_no_row_for_a_kernel_the_database_does_not_store(db_session):
    """A kernel the -k filter leaves out is not stored, so it has no row.

    The stored total is the filtered total, so the kept kernel is 100 percent.
    """
    workload = add_synthetic_workload(db_session, total_duration_ns=300)
    add_synthetic_kernel(db_session, workload, "kernel_a", [100, 200])
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    top_kernels = frames.dfs[KERNEL_TOP_TABLE_ID]
    assert top_kernels["Kernel_Name"].tolist() == ["kernel_a"]
    assert top_kernels["Percent"].tolist() == pytest.approx([100.0])


@pytest.mark.parametrize("total_duration_ns", [0, None])
def test_top_stats_percent_is_N_A_without_a_usable_total(db_session, total_duration_ns):
    """A zero or NULL total leaves each kernel's share unknown (FR-15)."""
    workload = add_synthetic_workload(db_session, total_duration_ns=total_duration_ns)
    add_synthetic_kernel(db_session, workload, "kernel_a", [0])
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    assert frames.dfs[KERNEL_TOP_TABLE_ID]["Percent"].tolist() == [NULL_TEXT]


def test_top_stats_times_follow_the_time_unit(db_session):
    """The time columns are named and scaled for the chosen --time-unit."""
    workload = add_synthetic_workload(db_session, total_duration_ns=1000)
    add_synthetic_kernel(db_session, workload, "kernel_a", [100, 200])
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "us")

    top_kernels = frames.dfs[KERNEL_TOP_TABLE_ID]
    assert "Sum(us)" in top_kernels.columns
    assert top_kernels["Sum(us)"].tolist() == pytest.approx([0.3])
    assert top_kernels["Mean(us)"].tolist() == pytest.approx([0.15])


def test_dispatch_list_holds_the_stored_dispatches_in_order(db_session):
    workload = add_synthetic_workload(db_session, total_duration_ns=300)
    add_synthetic_kernel(db_session, workload, "kernel_a", [100, 200])
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    dispatch_list = frames.dfs[DISPATCH_TABLE_ID]
    assert list(dispatch_list.columns) == ["Dispatch_ID", "Kernel_Name", "GPU_ID"]
    assert dispatch_list["Dispatch_ID"].tolist() == [1, 2]


def test_metric_values_are_shown_in_their_config_unit_and_rounded(db_session):
    """A Bytes/s value of a GB/s metric is shown in GB/s, rounded as its config does.

    A NULL value shows as N/A. A config expression with more than one ROUND call
    is not rounded, since the adapter cannot rebuild it from one digit count.
    """
    workload = add_synthetic_workload(db_session, total_duration_ns=0)
    add_synthetic_metric(db_session, workload, "10.1.0", "Bytes/s", "Avg", 2.56e9)
    add_synthetic_metric(db_session, workload, "10.1.1", "Count", "Avg", None)
    add_synthetic_metric(db_session, workload, "10.1.2", "Bytes/s", "Avg", 2.56e9)
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    metric_frame = frames.dfs[SYNTHETIC_METRIC_TABLE_ID]
    assert metric_frame.index.tolist() == ["10.1.0", "10.1.1", "10.1.2"]
    assert metric_frame.loc["10.1.0", "Avg"] == pytest.approx(2.6)
    assert metric_frame.loc["10.1.1", "Avg"] == NULL_TEXT
    assert metric_frame.loc["10.1.2", "Avg"] == pytest.approx(2.56)
    assert metric_frame.loc["10.1.0", "Unit"] == "GB/s"
    assert metric_frame.loc["10.1.1", "Description"] == "Count of rates"


def test_sys_info_table_and_run_sys_info_come_from_the_database(db_session):
    workload = add_synthetic_workload(db_session, total_duration_ns=0)
    seal_database()

    frames = build_report_workload(db_session, workload, synthetic_arch_config(), "ns")

    assert frames.sys_info.iloc[0]["gpu_arch"] == "gfx942"
    assert frames.dfs[SYS_INFO_TABLE_ID].loc["gpu_arch", "Info"] == "gfx942"


def test_show_all_renders_the_adapter_frames_without_a_selection_marker(
    db_session,
):
    """show_all renders the adapter's frames: only stored kernels, no * marker."""
    workload = add_synthetic_workload(db_session, total_duration_ns=1000)
    add_synthetic_kernel(db_session, workload, "kernel_a", [100, 200])
    add_synthetic_kernel(db_session, workload, "kernel_b", [100])
    add_synthetic_metric(db_session, workload, "10.1.0", "Bytes/s", "Avg", 2.5e9)
    add_synthetic_metric(db_session, workload, "10.1.1", "Count", "Avg", None)
    seal_database()
    arch_config = synthetic_arch_config()
    frames = build_report_workload(db_session, workload, arch_config, "ns")
    output = StringIO()

    tty.show_all(
        synthetic_args(),
        {"/runs/app/MI300X": frames},
        arch_config,
        output,
        profiling_config={"filter_blocks": []},
    )

    rendered = output.getvalue()
    assert "kernel_a" in rendered
    assert "Synthetic Table" in rendered
    assert re.search(r"Bandwidth\s+│\s+2\.5\s+│\s+GB/s", rendered)
    assert "N/A" in rendered
    assert "Selected" not in rendered
    assert "*" not in rendered
