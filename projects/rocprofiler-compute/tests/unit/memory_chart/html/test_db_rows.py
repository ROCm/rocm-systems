# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for reading and rendering memory chart rows from the analysis DB."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from memory_chart.html import report
from memory_chart.html.db_rows import read_workload_chart_rows
from memory_chart.html.report import write_mem_chart_report_from_db
from memory_chart.html.views import ChartScope, stored_chart_tables
from tests.unit.memory_chart.html.test_report import make_views
from utils import analysis_orm as orm
from utils import schema


@pytest.fixture
def database() -> Iterator[Session]:
    """An empty in-memory analysis database session."""
    engine = create_engine("sqlite:///:memory:")
    orm.Base.metadata.create_all(engine)
    session = Session(engine)

    yield session

    session.close()
    engine.dispose()


def add_workload(session: Session, layout: object = None) -> orm.Workload:
    """Add one workload of a gfx950 GPU, with an optional stored layout."""
    workload = orm.Workload(
        name="vcopy",
        sub_name="MI350",
        sys_info_extdata={"gpu_arch": "gfx950"},
        memory_chart_layout_extdata=layout,
    )
    session.add(workload)
    session.flush()
    return workload


def add_dispatch(
    session: Session, kernel: orm.Kernel, dispatch_id: int, duration: int
) -> None:
    """Add one dispatch that runs for the given number of nanoseconds."""
    session.add(
        orm.Dispatch(
            kernel=kernel,
            dispatch_id=dispatch_id,
            gpu_id=0,
            start_timestamp=0,
            end_timestamp=duration,
        )
    )


def test_rows_rank_kernels_by_total_duration(database: Session) -> None:
    """Order kernels longest first and keep a kernel that has no metric rows."""
    workload = add_workload(database)
    short = orm.Kernel(kernel_name="short", workload=workload)
    long = orm.Kernel(kernel_name="long", workload=workload)
    add_dispatch(database, short, 1, 100)
    add_dispatch(database, long, 2, 500)

    metric = orm.MetricDefinition(workload=workload, metric_id="3.1.1", name="Hit")
    database.add(
        orm.WorkloadMetricValue(
            metric=metric, workload=workload, value_name="Value", value=1.5
        )
    )
    database.add(
        orm.KernelMetricValue(
            metric=metric, kernel=short, value_name="Value", value=2.0
        )
    )
    database.flush()

    rows = read_workload_chart_rows(workload, ["3.1.1"])

    assert rows.arch == "gfx950"
    assert rows.aggregate == {"3.1.1": {"Value": 1.5}}
    assert [kernel.name for kernel in rows.kernels] == ["long", "short"]
    assert rows.kernels[0].values == {}
    assert rows.kernels[1].values == {"3.1.1": {"Value": 2.0}}


def test_rows_of_workload_without_dispatches_have_no_kernels(database: Session) -> None:
    """Read the workload-level rows and no kernels when nothing was dispatched."""
    workload = add_workload(database)
    metric = orm.MetricDefinition(workload=workload, metric_id="3.1.1", name="Hit")
    database.add(
        orm.WorkloadMetricValue(
            metric=metric, workload=workload, value_name="Value", value=1.5
        )
    )
    database.flush()

    rows = read_workload_chart_rows(workload, ["3.1.1", "9.9.9"])

    assert rows.aggregate == {"3.1.1": {"Value": 1.5}}
    assert rows.kernels == ()


def test_stored_tables_are_those_with_a_stored_metric_row(database: Session) -> None:
    """Count a configured table as loaded only when one of its metrics is stored."""
    workload = add_workload(database)
    metric = orm.MetricDefinition(workload=workload, metric_id="3.1.1", name="Hit")
    database.add(
        orm.WorkloadMetricValue(
            metric=metric, workload=workload, value_name="Value", value=1.5
        )
    )
    database.flush()

    arch_config = schema.ArchConfig(
        dfs={
            301: pd.DataFrame(index=["3.1.1"]),
            302: pd.DataFrame(index=["3.2.1"]),
        }
    )
    rows = read_workload_chart_rows(workload, ["3.1.1", "3.2.1"])

    assert stored_chart_tables(rows, arch_config) == {301}


def test_db_writer_skips_kernel_filtered_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Write nothing for -k, whose rows hold only the kernels the filter keeps."""
    caplog.set_level(logging.INFO)
    output = tmp_path / "1-vcopy" / report.memory_chart_html_name()

    path = write_mem_chart_report_from_db(
        orm.Workload(),
        schema.ArchConfig(),
        Namespace(normal_unit="per_kernel"),
        ChartScope(kernel_ids=[1]),
        output,
    )

    assert path is None
    assert "-k runs store only the selected kernels" in caplog.text
    assert not output.exists()


def test_db_writer_skips_without_stored_panel_3(
    database: Session, tmp_path: Path
) -> None:
    """Write nothing when no panel 3 table has stored rows, as the counters do."""
    workload = add_workload(database, layout={"archs": ["gfx950"]})
    output = tmp_path / "1-vcopy" / report.memory_chart_html_name()

    path = write_mem_chart_report_from_db(
        workload,
        schema.ArchConfig(),
        Namespace(normal_unit="per_kernel"),
        ChartScope(),
        output,
    )

    assert path is None
    assert not output.exists()


def test_db_writer_skips_without_stored_layout(
    database: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Write nothing and log the reason when the workload stores no layout."""
    workload = add_workload(database, layout=None)
    metric = orm.MetricDefinition(workload=workload, metric_id="3.1.1", name="Hit")
    database.add(
        orm.WorkloadMetricValue(
            metric=metric, workload=workload, value_name="Value", value=1.5
        )
    )
    database.flush()
    monkeypatch.setattr(report, "chart_metric_ids", lambda _config: ["3.1.1"])
    caplog.set_level(logging.INFO)
    output = tmp_path / "1-vcopy" / report.memory_chart_html_name()

    path = write_mem_chart_report_from_db(
        workload,
        schema.ArchConfig(dfs={301: pd.DataFrame(index=["3.1.1"])}),
        Namespace(normal_unit="per_kernel"),
        ChartScope(),
        output,
    )

    assert path is None
    assert "no layout for gfx950" in caplog.text
    assert not output.exists()


def test_db_writer_creates_the_workload_folder(
    database: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the page into a workload folder that does not exist yet."""
    workload = add_workload(database, layout={"archs": ["gfx950"]})
    metric = orm.MetricDefinition(workload=workload, metric_id="3.1.1", name="Hit")
    database.add(
        orm.WorkloadMetricValue(
            metric=metric, workload=workload, value_name="Value", value=1.5
        )
    )
    database.flush()
    monkeypatch.setattr(report, "chart_metric_ids", lambda _config: ["3.1.1"])
    monkeypatch.setattr(
        report, "build_chart_views_from_db", lambda *_: make_views("kernel zero")
    )
    output = tmp_path / "1-vcopy" / report.memory_chart_html_name()

    path = write_mem_chart_report_from_db(
        workload,
        schema.ArchConfig(dfs={301: pd.DataFrame(index=["3.1.1"])}),
        Namespace(normal_unit="per_kernel"),
        ChartScope(),
        output,
    )

    assert path == output
    assert output.exists()
