# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the PC-sampling section built from the analysis database."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from pc_sampling.pc_sampling_analysis import SOURCE_LINE_MISSING
from rocprof_compute_analyze.analysis_db import SourceFrameCollector, db_analysis
from rocprof_compute_analyze.pc_sampling_section import (
    PC_SAMPLING_FILTER_NOTE,
    build_pc_sampling_section,
    load_pc_sampling_summary_rows,
    normalize_summary_rows,
)
from utils import analysis_orm as orm
from utils import schema
from utils.analysis_orm import Database
from utils.file_io import load_pc_sampling_results
from utils.parser import PMC_KERNEL_TOP_TABLE_ID, load_pc_sampling_data

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI350"
DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "instruction_type",
    "code_object_id",
    "offset",
    "count",
    "active_thread_percent",
    "Kernel_Name",
]
STOCHASTIC_DISPLAY_COLUMNS = [
    "pid",
    "source_line",
    "instruction",
    "instruction_type",
    "code_object_id",
    "offset",
    "count",
    "active_thread_percent",
    "issue_count",
    "stall_count",
    "wave_occupancy_percent",
    "stall_reason",
    "Kernel_Name",
]
GPU_ARCH = "gfx950"


def store_fixture_samples(session: Session) -> Dict[str, Any]:
    """Store the MI350 fixture's samples in the database, as analyze does.

    Returns the workload row, the parsed tool records and the sysinfo row, so a
    test can build the parsed-record frame from the same inputs.
    """
    tool_records = load_pc_sampling_results(PC_SAMPLING_WORKLOAD)
    sys_info = pd.read_csv(Path(PC_SAMPLING_WORKLOAD) / "sysinfo.csv").iloc[0]
    sys_info = sys_info.to_dict()

    workload = orm.Workload(name="vcopy", sub_name="MI350")
    session.add(workload)
    kernel_objs = {}
    for tool_data in tool_records:
        for symbol in tool_data["kernel_symbols"]:
            name = symbol["formatted_kernel_name"]
            if name not in kernel_objs:
                kernel_objs[name] = orm.Kernel(kernel_name=name, workload=workload)
                session.add(kernel_objs[name])

    analyzer = db_analysis(
        SimpleNamespace(normal_unit="per_kernel", gpu_kernel=None, path=[]), {}
    )
    analyzer._pc_sampling_tool_data_per_workload = {PC_SAMPLING_WORKLOAD: tool_records}
    analyzer.add_pc_sampling_data(
        PC_SAMPLING_WORKLOAD,
        workload,
        kernel_objs,
        {},
        SourceFrameCollector(Path(PC_SAMPLING_WORKLOAD), workload),
        sys_info,
    )
    session.commit()
    # analyze creates the views before it writes the database.
    Database.create_views()
    Database.commit()
    return {
        "workload": workload,
        "tool_records": tool_records,
        "sys_info": sys_info,
    }


def parsed_records_frame(
    tool_records: List[Dict[str, Any]],
    sys_info: Dict[str, Any],
    kernel_name: str,
    sorting_type: str,
    num_rows: Optional[int],
) -> pd.DataFrame:
    """Return today's terminal frame for a single -k kernel."""
    workload = schema.Workload(
        sys_info=pd.DataFrame([sys_info]),
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: pd.DataFrame({"Kernel_Name": [kernel_name]})},
    )
    return load_pc_sampling_data(
        workload, sorting_type, tool_records, num_rows=num_rows
    )


def summary_row(
    kernel_name: str,
    offset: int,
    count: int,
    pid: int = 42,
    code_object_id: int = 5,
    issue_count: Optional[int] = None,
    stall_reason: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build one row of the summary view, under the view's column names."""
    stochastic = issue_count is not None
    return {
        "pid": pid,
        "code_object_id": code_object_id,
        "kernel_name": kernel_name,
        "offset": offset,
        "instruction": "v_mov_b32",
        "source": "/app/src/vecCopy.cpp:12",
        "count": count,
        "issue_count": issue_count,
        "stall_count": count - issue_count if stochastic else None,
        "wave_occupancy_percent": 25.0 if stochastic else None,
        "active_thread_percent": 50.0,
        "stall_reason": (
            json.dumps(stall_reason) if stall_reason is not None else None
        ),
    }


def summary_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a summary frame through the same normalizer the loader uses."""
    return normalize_summary_rows(pd.DataFrame(rows))


def test_filter_note_names_the_options_pc_sampling_ignores() -> None:
    """The note tells the reader that only -k narrows PC samples (FR-25)."""
    assert "-k" in PC_SAMPLING_FILTER_NOTE
    assert "--gpu-id" in PC_SAMPLING_FILTER_NOTE
    assert "--kernel-iteration-range" in PC_SAMPLING_FILTER_NOTE
    assert "narrowed by -k only" in PC_SAMPLING_FILTER_NOTE


@pytest.mark.parametrize("sorting_type", ["offset", "count"])
@pytest.mark.parametrize("num_rows", [0, 3, 10])
def test_section_matches_parsed_frame_for_each_kernel(
    db_session: Session, sorting_type: str, num_rows: int
) -> None:
    """For one -k kernel, the database section equals today's terminal frame."""
    stored = store_fixture_samples(db_session)
    summary_rows = load_pc_sampling_summary_rows(
        db_session, stored["workload"].workload_id
    )
    kernel_names = sorted(summary_rows["Kernel_Name"].unique())
    assert kernel_names

    for kernel_name in kernel_names:
        section = build_pc_sampling_section(
            summary_rows,
            gpu_arch=GPU_ARCH,
            sorting_type=sorting_type,
            num_rows=num_rows,
            kernel_names=[kernel_name],
        )
        parsed = parsed_records_frame(
            stored["tool_records"],
            stored["sys_info"],
            kernel_name,
            sorting_type,
            num_rows,
        )
        assert not section.empty
        assert list(section.columns) == list(parsed.columns)
        # The parsed frame keeps these float columns as nullable Float64.
        for column in ["active_thread_percent", "wave_occupancy_percent"]:
            parsed[column] = parsed[column].astype("float64")
        pd.testing.assert_frame_equal(section, parsed)


def test_section_has_the_stochastic_columns_for_the_fixture(
    db_session: Session,
) -> None:
    """The fixture is stochastic, so the section carries the stochastic columns."""
    stored = store_fixture_samples(db_session)
    summary_rows = load_pc_sampling_summary_rows(
        db_session, stored["workload"].workload_id
    )
    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="count", num_rows=5
    )
    assert list(section.columns) == STOCHASTIC_DISPLAY_COLUMNS


def test_section_keeps_the_top_rows_of_every_kernel() -> None:
    """num_rows applies to each kernel separately, not to the whole table."""
    summary_rows = summary_frame([
        summary_row("vecCopy", 0x10, count=9),
        summary_row("vecCopy", 0x20, count=4),
        summary_row("vecCopy", 0x30, count=1),
        summary_row("vecAdd", 0x10, count=7),
        summary_row("vecAdd", 0x20, count=2),
    ])

    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="count", num_rows=2
    )

    assert list(section["Kernel_Name"]) == ["vecAdd", "vecAdd", "vecCopy", "vecCopy"]
    assert list(section["count"]) == [7, 2, 9, 4]
    assert list(section.index) == [0, 1, 2, 3]


def test_section_sorts_offsets_numerically_within_each_kernel() -> None:
    """Offset sorting is numeric, so 0x9 precedes 0x10 and rows show as hex."""
    summary_rows = summary_frame([
        summary_row("vecCopy", 0x10, count=1),
        summary_row("vecCopy", 0x9, count=1),
        summary_row("vecAdd", 0x100, count=1),
        summary_row("vecAdd", 0x80, count=1),
    ])

    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="offset", num_rows=0
    )

    assert list(section["Kernel_Name"]) == ["vecAdd", "vecAdd", "vecCopy", "vecCopy"]
    assert list(section["offset"]) == ["0x80", "0x100", "0x9", "0x10"]


def test_section_kernel_names_narrow_the_rows() -> None:
    """Only the kernels the -k filter names are shown; None keeps every kernel."""
    summary_rows = summary_frame([
        summary_row("vecCopy", 0x10, count=3),
        summary_row("vecAdd", 0x10, count=5),
    ])

    everything = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="count"
    )
    narrowed = build_pc_sampling_section(
        summary_rows,
        gpu_arch=GPU_ARCH,
        sorting_type="count",
        kernel_names=["vecAdd"],
    )

    assert sorted(everything["Kernel_Name"]) == ["vecAdd", "vecCopy"]
    assert list(narrowed["Kernel_Name"]) == ["vecAdd"]
    assert list(narrowed["count"]) == [5]


def test_host_trap_rows_drop_the_stochastic_columns() -> None:
    """A host_trap sample has no issue counts, so those columns are left out."""
    summary_rows = summary_frame([
        summary_row("vecCopy", 0x10, count=4),
        summary_row("vecCopy", 0x20, count=2),
    ])

    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="count"
    )

    assert list(section.columns) == DISPLAY_COLUMNS


def test_stall_reasons_list_in_descending_count_order() -> None:
    """Each row lists its stall reasons as (reason, count), largest first."""
    summary_rows = summary_frame([
        summary_row(
            "vecCopy",
            0x10,
            count=6,
            issue_count=2,
            stall_reason={"WAITCNT": 1, "BARRIER_WAIT": 3},
        ),
        summary_row("vecCopy", 0x20, count=2, issue_count=2),
    ])

    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="offset"
    )

    stalled = section[section["offset"] == "0x10"].iloc[0]
    assert stalled["stall_reason"] == [("BARRIER_WAIT", 3), ("WAITCNT", 1)]
    assert section[section["offset"] == "0x20"].iloc[0]["stall_reason"] == []


def test_section_is_empty_without_rows() -> None:
    """No samples, or no sample of the selected kernel, gives an empty frame."""
    summary_rows = summary_frame([summary_row("vecCopy", 0x10, count=3)])

    assert build_pc_sampling_section(
        summary_rows.iloc[0:0], gpu_arch=GPU_ARCH, sorting_type="count"
    ).empty
    assert build_pc_sampling_section(
        summary_rows,
        gpu_arch=GPU_ARCH,
        sorting_type="count",
        kernel_names=["vecMissing"],
    ).empty


def test_summary_rows_are_read_for_one_workload_only(db_session: Session) -> None:
    """A workload's section never includes the samples of another workload."""
    stored = store_fixture_samples(db_session)
    workload_id = stored["workload"].workload_id

    own_rows = load_pc_sampling_summary_rows(db_session, workload_id)
    other_rows = load_pc_sampling_summary_rows(db_session, workload_id + 1000)

    assert not own_rows.empty
    assert other_rows.empty
    assert "source_line" in own_rows.columns
    assert "Kernel_Name" in own_rows.columns
    assert all(
        value is None or isinstance(value, dict) for value in own_rows["stall_reason"]
    )
    assert np.issubdtype(own_rows["count"].dtype, np.integer)


def test_missing_source_and_stall_reasons_load_as_n_a_and_empty(
    db_session: Session,
) -> None:
    """Samples with no inline frames or stall reasons load as N/A and None.

    The section still builds from them: the source reads N/A and the stall
    reason list is empty.
    """
    stored = store_fixture_samples(db_session)
    workload_id = stored["workload"].workload_id
    db_session.query(orm.InstructionSourceLine).delete()
    db_session.query(orm.PCSampleStallReason).delete()
    db_session.commit()

    summary_rows = load_pc_sampling_summary_rows(db_session, workload_id)
    assert not summary_rows.empty
    assert summary_rows["source_line"].eq(SOURCE_LINE_MISSING).all()
    assert summary_rows["stall_reason"].isna().all()

    section = build_pc_sampling_section(
        summary_rows, gpu_arch=GPU_ARCH, sorting_type="count", num_rows=0
    )
    assert list(section.columns) == STOCHASTIC_DISPLAY_COLUMNS
    assert section["source_line"].eq(SOURCE_LINE_MISSING).all()
    assert all(stall_reasons == [] for stall_reasons in section["stall_reason"])
