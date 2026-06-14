# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_db import db_analysis
from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map_from_json,
    load_pc_sampling_results,
    process_pc_sampling_kernel_trace,
)
from utils.parser import (
    PMC_KERNEL_TOP_TABLE_ID,
    load_pc_sampling_data,
    load_pc_sampling_data_per_kernel,
    nullify_unevaluated_metric_values,
    search_pc_sampling_record,
)
from utils.utils_common import is_only_pc_sampling

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI300A_A1"

PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"


# ── Helpers for building synthetic JSON / records ────────────


def make_record(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int,
    wave_issued: bool = True,
    stall_reason: str | None = None,
) -> dict:
    snapshot = {}
    if stall_reason is not None:
        snapshot["stall_reason"] = stall_reason
    return {
        "inst_index": inst_index,
        "record": {
            "pc": {
                "code_object_id": code_object_id,
                "code_object_offset": offset,
            },
            "dispatch_id": dispatch_id,
            "wave_issued": wave_issued,
            "snapshot": snapshot,
        },
    }


def make_dispatch(
    dispatch_id: int,
    kernel_id: int,
    agent_handle: int = 1,
    start: int = 0,
    end: int = 0,
) -> dict:
    """A kernel_dispatch buffer record mapping a dispatch to a kernel."""
    return {
        "start_timestamp": start,
        "end_timestamp": end,
        "dispatch_info": {
            "dispatch_id": dispatch_id,
            "kernel_id": kernel_id,
            "agent_id": {"handle": agent_handle},
        },
    }


def make_kernel_symbol(
    kernel_id: int,
    code_object_id: int,
    formatted_kernel_name: str,
) -> dict:
    """A kernel_symbols entry mapping kernel/code-object ids to a name."""
    return {
        "kernel_id": kernel_id,
        "code_object_id": code_object_id,
        "formatted_kernel_name": formatted_kernel_name,
    }


def make_agent(handle: int, node_id: int, agent_type: int) -> dict:
    """An agents entry (type 1 == CPU, type 2 == GPU)."""
    return {"id": {"handle": handle}, "type": agent_type, "node_id": node_id}


def make_tool_data(
    stochastic: list | None = None,
    host_trap: list | None = None,
    instructions: list | None = None,
    comments: list | None = None,
    kernel_symbols: list | None = None,
    kernel_dispatch: list | None = None,
    agents: list | None = None,
) -> dict:
    """Build a single ``rocprofiler-sdk-tool[0]`` dict for the analyze paths."""
    return {
        "buffer_records": {
            "pc_sample_stochastic": stochastic if stochastic is not None else [],
            "pc_sample_host_trap": host_trap if host_trap is not None else [],
            "kernel_dispatch": kernel_dispatch if kernel_dispatch is not None else [],
        },
        "strings": {
            "pc_sample_instructions": (
                instructions if instructions is not None else []
            ),
            "pc_sample_comments": comments if comments is not None else [],
        },
        "kernel_symbols": kernel_symbols if kernel_symbols is not None else [],
        "agents": agents if agents is not None else [],
    }


def write_results_json(path: Path, **kwargs) -> Path:
    """Write a results json wrapping a single tool record built from kwargs."""
    path.write_text(json.dumps({"rocprofiler-sdk-tool": [make_tool_data(**kwargs)]}))
    return path


# ═══════════════════════════════════════════════════════════════
# is_only_pc_sampling
# ═══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "filter_blocks, expected",
    [
        ([], False),
        (["21"], True),
        (["pc_sampling"], True),
        (["21", "pc_sampling"], True),
        (["21", "2"], False),
        (["2"], False),
    ],
)
def test_is_only_pc_sampling(filter_blocks: list[str], expected: bool) -> None:
    """True only when every requested block is PC sampling (21 / pc_sampling)."""
    assert is_only_pc_sampling(filter_blocks) is expected


# ═══════════════════════════════════════════════════════════════
# search_pc_sampling_record
# ═══════════════════════════════════════════════════════════════


def test_search_pc_sampling_record_empty_list_returns_none() -> None:
    """Return None when the input record list is empty."""
    assert search_pc_sampling_record([]) is None


def test_search_pc_sampling_record_single_dict_input_issued() -> None:
    """Accept a single dict (not a list) and count it as one issued sample."""
    record = make_record(
        code_object_id=1,
        offset=0x10,
        inst_index=0,
        dispatch_id=0,
        wave_issued=True,
    )
    result = search_pc_sampling_record(record)
    assert result is not None
    assert len(result) == 1
    co_id, off, idx, total, issued, stalled, _, _ = result[0]
    assert (co_id, off, idx) == (1, 0x10, 0)
    assert total == 1
    assert issued == 1
    assert stalled == 0


def test_search_pc_sampling_record_groups_by_key() -> None:
    """
    Group records by (code_object_id, offset, inst_index)
    and sum counts per group.
    """
    records = [
        make_record(1, 0x10, 0, dispatch_id=0),
        make_record(1, 0x10, 0, dispatch_id=1),
        make_record(1, 0x20, 1, dispatch_id=2),
    ]
    result = search_pc_sampling_record(records)
    assert result is not None
    assert len(result) == 2
    assert result[0][3] == 2  # total_count for first key
    assert result[1][3] == 1


def test_search_pc_sampling_record_dispatch_id_collection() -> None:
    """Collect unique dispatch IDs across duplicate records for the same key."""
    records = [
        make_record(1, 0x10, 0, dispatch_id=0),
        make_record(1, 0x10, 0, dispatch_id=0),
        make_record(1, 0x10, 0, dispatch_id=1),
    ]
    result = search_pc_sampling_record(records)
    assert result is not None
    dispatch_ids = result[0][7]
    assert dispatch_ids == [0, 1]


def test_search_pc_sampling_record_skips_none_fields() -> None:
    """Skip records whose code_object_id or offset is None."""
    valid = make_record(1, 0x10, 0, dispatch_id=0)
    invalid = {
        "inst_index": 0,
        "record": {
            "pc": {
                "code_object_id": None,
                "code_object_offset": 0x10,
            },
            "dispatch_id": 1,
            "wave_issued": True,
            "snapshot": {},
        },
    }
    result = search_pc_sampling_record([valid, invalid])
    assert result is not None
    assert len(result) == 1


def make_record_without_snapshot(
    code_object_id: int,
    offset: int,
    inst_index: int,
    dispatch_id: int | None,
    wave_issued: bool,
) -> dict:
    """A record missing the ``snapshot`` key entirely."""
    return {
        "inst_index": inst_index,
        "record": {
            "pc": {
                "code_object_id": code_object_id,
                "code_object_offset": offset,
            },
            "dispatch_id": dispatch_id,
            "wave_issued": wave_issued,
        },
    }


@pytest.mark.parametrize(
    "records, expected",
    [
        pytest.param(
            [
                make_record(1, 0x10, 0, dispatch_id=0, wave_issued=True),
                make_record(1, 0x10, 0, dispatch_id=1, wave_issued=True),
            ],
            {
                "total": 2,
                "issued": 2,
                "stalled": 0,
                "reasons": set(),
                "dispatch_ids": [0, 1],
            },
            id="all_issued",
        ),
        pytest.param(
            [
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}WAITCNT",
                ),
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=1,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}ALU_DEPENDENCY",
                ),
            ],
            {
                "total": 2,
                "issued": 0,
                "stalled": 2,
                "reasons": {"WAITCNT", "ALU_DEPENDENCY"},
                "dispatch_ids": [0, 1],
            },
            id="all_stalled",
        ),
        pytest.param(
            [
                make_record_without_snapshot(
                    1, 0x10, 0, dispatch_id=0, wave_issued=False
                )
            ],
            {
                "total": 1,
                "issued": 0,
                "stalled": 1,
                "reasons": set(),
                "dispatch_ids": [0],
            },
            id="missing_snapshot",
        ),
        pytest.param(
            [
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=False,
                    stall_reason="SHORT",
                )
            ],
            {
                "total": 1,
                "issued": 0,
                "stalled": 1,
                "reasons": set(),
                "dispatch_ids": [0],
            },
            id="short_stall_reason",
        ),
        pytest.param(
            [
                make_record(
                    1,
                    0x10,
                    0,
                    dispatch_id=0,
                    wave_issued=False,
                    stall_reason=f"{PREFIX}NOT_A_REAL_KEY",
                )
            ],
            {
                "total": 1,
                "issued": 0,
                "stalled": 1,
                "reasons": set(),
                "dispatch_ids": [0],
            },
            id="unknown_stall_key",
        ),
        pytest.param(
            [make_record(1, 0x10, 0, dispatch_id=None, wave_issued=True)],
            {
                "total": 1,
                "issued": 1,
                "stalled": 0,
                "reasons": set(),
                "dispatch_ids": [],
            },
            id="missing_dispatch_id",
        ),
    ],
)
def test_search_pc_sampling_record_field_edges(
    records: list[dict], expected: dict
) -> None:
    """Exercise snapshot/stall-reason/dispatch-id edge cases for a single group."""
    result = search_pc_sampling_record(records)
    assert result is not None
    assert len(result) == 1
    _, _, _, total, issued, stalled, reasons, dispatch_ids = result[0]
    assert total == expected["total"]
    assert issued == expected["issued"]
    assert stalled == expected["stalled"]
    assert {reason for reason, _ in reasons} == expected["reasons"]
    assert dispatch_ids == expected["dispatch_ids"]


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_data_per_kernel
# ═══════════════════════════════════════════════════════════════


def setup_per_kernel_data(
    method: str = "host_trap",
) -> dict:
    """Build tool_data for per-kernel tests.

    vecCopy (kernel_id 100) and vecAdd (kernel_id 101) share code object 5,
    so kernels are distinguished only by kernel_id via dispatch attribution.
    """
    samples = [
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(
            5,
            0x20,
            1,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        make_record(5, 0x30, 2, dispatch_id=2),
    ]

    key = "host_trap" if method == "host_trap" else "stochastic"
    kwargs = {key: samples}
    return make_tool_data(
        instructions=["v_mov_b32", "s_waitcnt", "v_add_f32"],
        comments=[
            "/src/vcopy.cpp:42",
            "/src/vcopy.cpp:43",
            "/src/vadd.cpp:30",
        ],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        kernel_dispatch=[
            make_dispatch(0, 100),
            make_dispatch(1, 100),
            make_dispatch(2, 101),
        ],
        **kwargs,
    )


@pytest.mark.parametrize(
    "method, sorting_type",
    [
        ("host_trap", "offset"),
        ("host_trap", "count"),
        ("stochastic", "offset"),
        ("stochastic", "count"),
    ],
)
def test_load_per_kernel_schema_and_sort(
    tmp_path: Path,
    method: str,
    sorting_type: str,
) -> None:
    """Column projection follows method; row order follows sorting_type."""
    host_trap_cols = [
        "source_line",
        "instruction",
        "code_object_id",
        "offset",
        "count",
    ]
    stochastic_cols = host_trap_cols + [
        "count_issued",
        "count_stalled",
        "stall_reason",
    ]
    expected_columns = host_trap_cols if method == "host_trap" else stochastic_cols
    tool_data = setup_per_kernel_data(method=method)
    df = load_pc_sampling_data_per_kernel(
        method=method,
        tool_data=tool_data,
        kernel_name="vecCopy",
        sorting_type=sorting_type,
    )
    assert not df.empty
    assert list(df.columns) == expected_columns
    for _, row in df.iterrows():
        assert row["code_object_id"] == 5
    if sorting_type == "count":
        counts = df["count"].tolist()
        assert counts == sorted(counts, reverse=True)
    else:
        offsets = df["offset"].tolist()
        assert offsets == sorted(offsets)


def test_load_per_kernel_offset_sort_is_numeric() -> None:
    """Offset sort orders by numeric value, not lexicographic hex string."""
    tool_data = make_tool_data(
        host_trap=[
            make_record(5, 0x100, 0, dispatch_id=0),
            make_record(5, 0x20, 1, dispatch_id=1),
        ],
        instructions=["a", "b"],
        comments=["/src/f.cpp:1", "/src/f.cpp:2"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )
    df = load_pc_sampling_data_per_kernel(
        method="host_trap",
        tool_data=tool_data,
        kernel_name="vecCopy",
        sorting_type="offset",
    )
    # 0x20 (32) must precede 0x100 (256); lexicographic order would invert them.
    assert df["offset"].tolist() == ["0x20", "0x100"]


def make_per_kernel_guard_data(
    instructions: list | None,
    comments: list | None,
    indices: tuple[int, int] = (0, 1),
) -> dict:
    """Per-kernel tool_data with caller-controlled instruction/comment tables."""
    samples = [
        make_record(5, 0x10, indices[0], dispatch_id=0),
        make_record(5, 0x20, indices[1], dispatch_id=1),
    ]
    return make_tool_data(
        host_trap=samples,
        instructions=instructions,
        comments=comments,
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )


@pytest.mark.parametrize(
    "instructions, comments, instruction_none, source_line_none, indices",
    [
        pytest.param(
            ["v_mov"],
            ["/s/a.cpp:1", "/s/a.cpp:2"],
            "some",
            "none",
            (0, 1),
            id="instructions_short",
        ),
        pytest.param(
            ["v_mov", "v_add"],
            ["/s/a.cpp:1"],
            "none",
            "some",
            (0, 1),
            id="comments_short",
        ),
        pytest.param(
            ["v_mov"],
            ["/s/a.cpp:1", "/s/a.cpp:2", "/s/a.cpp:3", "/s/a.cpp:4", "/s/a.cpp:5"],
            "all",
            "none",
            (3, 4),
            id="instructions_all_out_of_range",
        ),
    ],
)
def test_load_per_kernel_out_of_range_index_guards(
    instructions: list | None,
    comments: list | None,
    instruction_none: str,
    source_line_none: str,
    indices: tuple[int, int],
) -> None:
    """An inst_index past the end of a string table yields None, not an error."""
    tool_data = make_per_kernel_guard_data(instructions, comments, indices)
    df = load_pc_sampling_data_per_kernel(
        method="host_trap",
        tool_data=tool_data,
        kernel_name="vecCopy",
        sorting_type="offset",
    )
    assert not df.empty
    assert_none_kind(df["instruction"], instruction_none)
    assert_none_kind(df["source_line"], source_line_none)


@pytest.mark.parametrize(
    "instructions, comments",
    [
        pytest.param(None, ["/s/a.cpp:1", "/s/a.cpp:2"], id="empty_instructions"),
        pytest.param(["v_mov", "v_add"], None, id="empty_comments"),
    ],
)
def test_load_per_kernel_empty_string_table_exits(
    instructions: list | None,
    comments: list | None,
) -> None:
    """An empty instruction/comment table is treated as missing and exits."""
    tool_data = make_per_kernel_guard_data(instructions, comments)
    with pytest.raises(SystemExit):
        load_pc_sampling_data_per_kernel(
            method="host_trap",
            tool_data=tool_data,
            kernel_name="vecCopy",
            sorting_type="offset",
        )


def assert_none_kind(column: pd.Series, kind: str) -> None:
    """Assert how many entries in *column* are None: all / some / none."""
    if kind == "all":
        assert column.isna().all()
    elif kind == "some":
        assert column.isna().any() and not column.isna().all()
    else:
        assert not column.isna().any()


def test_load_per_kernel_multi_dispatch_groupby() -> None:
    """Two dispatch IDs at one (code_object_id, offset) collapse to a summed row."""
    samples = [
        make_record(
            5,
            0x10,
            0,
            dispatch_id=0,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        make_record(
            5,
            0x10,
            1,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}ALU_DEPENDENCY",
        ),
    ]
    tool_data = make_tool_data(
        stochastic=samples,
        instructions=["v_mov", "v_add"],
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 100)],
    )
    df = load_pc_sampling_data_per_kernel(
        method="stochastic",
        tool_data=tool_data,
        kernel_name="vecCopy",
        sorting_type="count",
    )
    assert len(df) == 1
    row = df.iloc[0]
    assert row["count"] == 2
    assert row["count_stalled"] == 2
    assert row["count_issued"] == 0
    assert {reason for reason, _ in row["stall_reason"]} == {
        "WAITCNT",
        "ALU_DEPENDENCY",
    }


def test_load_per_kernel_kernel_not_found() -> None:
    """
    Return an empty DataFrame when the requested kernel name
    is absent from the kernel symbols.
    """
    tool_data = setup_per_kernel_data()
    df = load_pc_sampling_data_per_kernel(
        method="host_trap",
        tool_data=tool_data,
        kernel_name="nonexistent",
        sorting_type="offset",
    )
    assert df.empty


def test_load_per_kernel_no_pc_sample_key() -> None:
    """
    When the tool data has no populated pc_sample array,
    the per-kernel loader calls console_error which exits.
    """
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    with pytest.raises(SystemExit):
        load_pc_sampling_data_per_kernel(
            method="host_trap",
            tool_data=tool_data,
            kernel_name="vecCopy",
            sorting_type="offset",
        )


def test_load_per_kernel_invalid_sorting_type() -> None:
    """Return an empty DataFrame and log an error for an unrecognized sorting type."""
    tool_data = setup_per_kernel_data()
    with patch("utils.parser.console_error"):
        df = load_pc_sampling_data_per_kernel(
            method="host_trap",
            tool_data=tool_data,
            kernel_name="vecCopy",
            sorting_type="invalid",
        )
    assert df.empty


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_data
# ═══════════════════════════════════════════════════════════════


def test_load_pc_sampling_data_empty_prefix(
    tmp_path: Path,
) -> None:
    """Return an empty DataFrame when the file prefix is an empty string."""
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "", "count")
    assert df.empty


def test_load_pc_sampling_data_none_prefix(
    tmp_path: Path,
) -> None:
    """Return an empty DataFrame when the file prefix is the literal string 'none'."""
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "none", "count")
    assert df.empty


def test_load_pc_sampling_data_missing_results_json(
    tmp_path: Path,
) -> None:
    """Return an empty DataFrame when ps_file_results.json does not exist."""
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert df.empty


@pytest.mark.parametrize("method", ["stochastic", "host_trap"])
def test_load_pc_sampling_data_no_filter(
    tmp_path: Path,
    method: str,
) -> None:
    """No-filter aggregates by source line; kernel name resolved per dispatch.

    vecCopy (kernel_id 100) and vecAdd (kernel_id 101) share code object 5,
    so each sample's kernel must be resolved via its dispatch_id, not the
    shared code_object_id.
    """
    samples = [
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(5, 0x10, 0, dispatch_id=0),
        make_record(5, 0x20, 1, dispatch_id=1),
    ]
    write_results_json(
        tmp_path / "ps_file_results.json",
        instructions=["v_mov_b32 v0 v1", "s_waitcnt vmcnt(0)"],
        comments=["/src/vcopy.cpp:42", "/src/vadd.cpp:99"],
        kernel_symbols=[
            make_kernel_symbol(100, 5, "vecCopy"),
            make_kernel_symbol(101, 5, "vecAdd"),
        ],
        kernel_dispatch=[make_dispatch(0, 100), make_dispatch(1, 101)],
        **{method: samples},
    )
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert not df.empty
    assert list(df.columns) == [
        "source_line",
        "Kernel_Name",
        "instruction",
        "count",
    ]
    by_kernel = dict(zip(df["source_line"], df["Kernel_Name"]))
    assert df.iloc[0]["source_line"].startswith("...")
    assert df.iloc[0]["count"] == 2
    # Samples sharing code object 5 still resolve to their own kernels.
    assert by_kernel[".../vcopy.cpp:42"] == "vecCopy"
    assert by_kernel[".../vadd.cpp:99"] == "vecAdd"


def test_load_pc_sampling_data_multiple_kernels_error(
    tmp_path: Path,
) -> None:
    """
    Return an empty DataFrame and log an error when more
    than one kernel ID is filtered.
    """
    write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=[make_record(100, 0x10, 0, dispatch_id=0)],
    )
    workload = schema.Workload(filter_kernel_ids=[0, 1])
    with patch("utils.parser.console_error"):
        df = load_pc_sampling_data(
            workload,
            str(tmp_path),
            "ps_file",
            "count",
        )
    assert df.empty


def test_load_pc_sampling_data_single_kernel_valid(
    tmp_path: Path,
) -> None:
    """Return per-kernel data when exactly one valid kernel ID is filtered."""
    write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=[make_record(100, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov_b32"],
        comments=["/src/vcopy.cpp:42"],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy"]})
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df},
    )
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert not df.empty


def test_load_pc_sampling_data_single_kernel_out_of_bounds(
    tmp_path: Path,
) -> None:
    """
    Return an empty DataFrame when the filtered kernel ID
    exceeds the kernel-top table range.
    """
    write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=[make_record(100, 0x10, 0, dispatch_id=0)],
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy", "vecAdd"]})
    workload = schema.Workload(
        filter_kernel_ids=[99],
        dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df},
    )
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert df.empty


def test_load_pc_sampling_data_method_not_detected(
    tmp_path: Path,
) -> None:
    """
    Return an empty DataFrame for a single-kernel filter when neither
    pc_sample array is populated (no detectable method).
    """
    write_results_json(
        tmp_path / "ps_file_results.json",
        kernel_symbols=[make_kernel_symbol(100, 100, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy"]})
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df},
    )
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert df.empty


@pytest.mark.parametrize(
    "populated, expected_column_count",
    [
        ("host_trap", 5),  # host_trap-only is detected
        ("both", 8),  # stochastic wins when both arrays are populated
    ],
)
def test_load_pc_sampling_data_method_detection(
    tmp_path: Path,
    populated: str,
    expected_column_count: int,
) -> None:
    """Single-kernel method detection: host_trap-only and stochastic-priority."""
    host_trap = [make_record(5, 0x10, 0, dispatch_id=0)]
    kwargs = {"host_trap": host_trap}
    if populated == "both":
        kwargs["stochastic"] = [make_record(5, 0x10, 0, dispatch_id=0)]
    write_results_json(
        tmp_path / "ps_file_results.json",
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
        **kwargs,
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy"]})
    workload = schema.Workload(
        filter_kernel_ids=[0],
        dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df},
    )
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert len(df.columns) == expected_column_count


def test_load_pc_sampling_data_no_filter_instruction_out_of_range(
    tmp_path: Path,
) -> None:
    """No-filter: an inst_index past the instruction table yields a None entry."""
    write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=[make_record(5, 0x10, 1, dispatch_id=0)],
        instructions=["v_mov"],  # len 1; inst_index 1 is out of range
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],  # len 2; inst_index 1 in range
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert not df.empty
    assert df.iloc[0]["instruction"] is None
    assert df.iloc[0]["source_line"] == ".../a.cpp:2"


def test_load_pc_sampling_data_uses_provided_tool_data(tmp_path: Path) -> None:
    """When tool_data is passed, use it without reading the results json.

    No ps_file_results.json is written, so a non-empty result proves the
    provided dict was used (the file-parse fallback would warn and return
    empty).
    """
    tool_data = make_tool_data(
        stochastic=[make_record(5, 0x10, 0, dispatch_id=0)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100)],
    )
    workload = schema.Workload()  # no kernel filter -> no-filter path
    df = load_pc_sampling_data(
        workload, str(tmp_path), "ps_file", "count", tool_data=tool_data
    )
    assert not df.empty
    assert df.iloc[0]["Kernel_Name"] == "vecCopy"


# ═══════════════════════════════════════════════════════════════
# nullify_unevaluated_metric_values
# ═══════════════════════════════════════════════════════════════


def test_nullify_unevaluated_metrics_metric_table_nullified() -> None:
    """
    Replace Value/Avg/Min/Max with 'N/A' in metric tables
    while preserving Metric_ID and Metric.
    """
    df = pd.DataFrame({
        "Metric_ID": ["1.1.0", "1.1.1"],
        "Metric": ["Wavefronts", "VALU Insts"],
        "Value": [
            "AVG(SQ_WAVES)",
            "AVG(SQ_INSTS_VALU)",
        ],
        "Avg": ["formula1", "formula2"],
        "Min": ["formula3", "formula4"],
        "Max": ["formula5", "formula6"],
    })
    workload = schema.Workload(
        dfs={10: df},
        dfs_type={10: "metric_table"},
    )
    nullify_unevaluated_metric_values(workload)
    for col in ["Value", "Avg", "Min", "Max"]:
        assert (workload.dfs[10][col] == "N/A").all()
    assert workload.dfs[10]["Metric_ID"].iloc[0] == "1.1.0"
    assert workload.dfs[10]["Metric"].iloc[0] == "Wavefronts"


def test_nullify_unevaluated_metrics_non_metric_table_untouched() -> None:
    """Leave non-metric-table DataFrames unchanged."""
    df = pd.DataFrame({"Value": [42, 99]})
    workload = schema.Workload(
        dfs={20: df},
        dfs_type={20: "raw_csv_table"},
    )
    nullify_unevaluated_metric_values(workload)
    assert workload.dfs[20]["Value"].tolist() == [42, 99]


def test_nullify_unevaluated_metrics_empty_df_skipped() -> None:
    """Skip empty DataFrames without error even when typed as metric_table."""
    df = pd.DataFrame()
    workload = schema.Workload(
        dfs={30: df},
        dfs_type={30: "metric_table"},
    )
    nullify_unevaluated_metric_values(workload)
    assert workload.dfs[30].empty


# ═══════════════════════════════════════════════════════════════
# process_pc_sampling_kernel_trace
# ═══════════════════════════════════════════════════════════════


def test_load_pc_sampling_results_missing_returns_none(tmp_path: Path) -> None:
    """Return None when the results json is absent."""
    assert load_pc_sampling_results(str(tmp_path)) is None


def test_load_pc_sampling_results_parses_tool_record(tmp_path: Path) -> None:
    """Return the rocprofiler-sdk-tool[0] dict when the results json exists."""
    write_results_json(
        tmp_path / "ps_file_results.json",
        kernel_symbols=[make_kernel_symbol(12, 2, "vecCopy")],
    )
    tool_data = load_pc_sampling_results(str(tmp_path))
    assert tool_data is not None
    assert tool_data["kernel_symbols"][0]["formatted_kernel_name"] == "vecCopy"


def test_process_pc_sampling_none_returns_empty() -> None:
    """Return empty DataFrame with expected columns when tool_data is None."""
    df = process_pc_sampling_kernel_trace(None)
    assert df.empty
    assert list(df.columns) == [
        "Dispatch_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]


def test_process_pc_sampling_with_agent_info() -> None:
    """Verify column selection, GPU mapping, kernel names, and timestamps."""
    tool_data = make_tool_data(
        kernel_symbols=[
            make_kernel_symbol(12, 2, "vecCopy"),
            make_kernel_symbol(13, 2, "vecAdd"),
            make_kernel_symbol(14, 2, "vecMul"),
        ],
        kernel_dispatch=[
            make_dispatch(
                1, 12, agent_handle=20, start=1981199661678356, end=1981199662835032
            ),
            make_dispatch(2, 13, agent_handle=30, start=2000, end=3000),
            make_dispatch(3, 14, agent_handle=99, start=4000, end=5000),
        ],
        agents=[
            make_agent(handle=10, node_id=1, agent_type=1),
            make_agent(handle=20, node_id=2, agent_type=2),
            make_agent(handle=30, node_id=3, agent_type=2),
        ],
    )

    df = process_pc_sampling_kernel_trace(tool_data)

    # Correct shape and columns
    assert len(df) == 3
    assert list(df.columns) == [
        "Dispatch_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]

    # Multi-GPU mapping: handle 20 -> GPU 0, handle 30 -> GPU 1, unknown -> 0
    assert df["GPU_ID"].tolist() == [0, 1, 0]
    assert df["Kernel_Name"].tolist() == ["vecCopy", "vecAdd", "vecMul"]

    # Timestamps passed through unchanged
    assert df["Start_Timestamp"].iloc[0] == 1981199661678356
    assert df["End_Timestamp"].iloc[0] == 1981199662835032


def test_process_pc_sampling_no_gpu_agents() -> None:
    """Default GPU_ID to 0 when no GPU agents are present."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(12, 2, "vecCopy")],
        kernel_dispatch=[make_dispatch(1, 12, agent_handle=99, start=1000, end=2000)],
        agents=[make_agent(handle=10, node_id=1, agent_type=1)],
    )
    df = process_pc_sampling_kernel_trace(tool_data)
    assert len(df) == 1
    assert df["GPU_ID"].iloc[0] == 0


def test_process_pc_sampling_unmapped_kernel_id() -> None:
    """A dispatch whose kernel_id is absent from kernel_symbols maps to None."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(12, 2, "vecCopy")],
        kernel_dispatch=[make_dispatch(1, 999, agent_handle=20, start=1, end=2)],
        agents=[make_agent(handle=20, node_id=2, agent_type=2)],
    )
    df = process_pc_sampling_kernel_trace(tool_data)
    assert len(df) == 1
    assert df.iloc[0]["Kernel_Name"] is None


# ═══════════════════════════════════════════════════════════════
# build_agent_to_gpu_map_from_json
# ═══════════════════════════════════════════════════════════════


def test_build_agent_to_gpu_map_single_gpu() -> None:
    """Map one GPU agent to GPU index 0, ignoring CPU agents."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=20, node_id=2, agent_type=2),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {20: 0}


def test_build_agent_to_gpu_map_two_gpus() -> None:
    """Assign sequential GPU indices to multiple GPU agents sorted by node_id."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=30, node_id=3, agent_type=2),
        make_agent(handle=20, node_id=2, agent_type=2),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {20: 0, 30: 1}


def test_build_agent_to_gpu_map_no_gpu_agents() -> None:
    """Return an empty map when no GPU agents are present."""
    agents = [
        make_agent(handle=10, node_id=1, agent_type=1),
        make_agent(handle=11, node_id=2, agent_type=1),
    ]
    assert build_agent_to_gpu_map_from_json(agents) == {}


def test_build_agent_to_gpu_map_empty() -> None:
    """Return an empty map when the agents list is empty."""
    assert build_agent_to_gpu_map_from_json([]) == {}


# ═══════════════════════════════════════════════════════════════
# calc_pc_sampling_data
# ═══════════════════════════════════════════════════════════════


def make_db_analysis(workload_path: str) -> db_analysis:
    """Construct a db_analysis whose only populated state is _runs."""
    instance = db_analysis.__new__(db_analysis)
    instance._runs = {workload_path: None}
    return instance


def test_calc_pc_sampling_data_missing_file(
    tmp_path: Path,
) -> None:
    """Workloads without ps_file_results.json are skipped, returning an empty map."""
    instance = make_db_analysis(str(tmp_path))
    assert instance.calc_pc_sampling_data({str(tmp_path): None}) == {}


@pytest.mark.parametrize("placement", ["stochastic", "host_trap", "mixed"])
def test_calc_pc_sampling_data_aggregation(
    tmp_path: Path,
    placement: str,
) -> None:
    """Records group by (code_object_id, offset) with the documented schema."""
    s0 = make_record(100, 0x10, 0, dispatch_id=0, wave_issued=True)
    s1 = make_record(
        100,
        0x10,
        0,
        dispatch_id=1,
        wave_issued=False,
        stall_reason=f"{PREFIX}WAITCNT",
    )
    s2 = make_record(101, 0x20, 1, dispatch_id=2, wave_issued=True)
    if placement == "stochastic":
        kwargs = {"stochastic": [s0, s1, s2]}
    elif placement == "host_trap":
        kwargs = {"host_trap": [s0, s1, s2]}
    else:
        kwargs = {"stochastic": [s0, s1], "host_trap": [s2]}
    write_results_json(
        tmp_path / "ps_file_results.json",
        instructions=["v_mov", "v_add"],
        comments=["/s/a.cpp:1", "/s/a.cpp:2"],
        kernel_symbols=[
            {"code_object_id": 100, "formatted_kernel_name": "vecCopy"},
            {"code_object_id": 101, "formatted_kernel_name": "vecAdd"},
        ],
        **kwargs,
    )
    instance = make_db_analysis(str(tmp_path))
    result = instance.calc_pc_sampling_data({
        str(tmp_path): load_pc_sampling_results(str(tmp_path))
    })

    expected_columns = [
        "offset",
        "count",
        "count_issued",
        "count_stalled",
        "stall_reason",
        "instruction",
        "source_line",
        "kernel_name",
    ]
    assert str(tmp_path) in result
    df = result[str(tmp_path)]
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    row_copy = df[df["offset"] == 0x10].iloc[0]
    assert row_copy["count"] == 2
    assert row_copy["kernel_name"] == "vecCopy"
    row_add = df[df["offset"] == 0x20].iloc[0]
    assert row_add["count"] == 1
    assert row_add["kernel_name"] == "vecAdd"


def test_calc_pc_sampling_data_unmapped_kernel(
    tmp_path: Path,
) -> None:
    """A code_object_id absent from kernel_symbols maps to a None kernel_name."""
    write_results_json(
        tmp_path / "ps_file_results.json",
        stochastic=[make_record(999, 0x10, 0, dispatch_id=0, wave_issued=True)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[{"code_object_id": 100, "formatted_kernel_name": "vecCopy"}],
    )
    instance = make_db_analysis(str(tmp_path))
    df = instance.calc_pc_sampling_data({
        str(tmp_path): load_pc_sampling_results(str(tmp_path))
    })[str(tmp_path)]
    assert df.iloc[0]["kernel_name"] is None


def test_calc_pc_sampling_data_uses_provided_tool_data(tmp_path: Path) -> None:
    """calc uses the provided tool_data map without reading the results json."""
    tool_data = make_tool_data(
        stochastic=[make_record(5, 0x10, 0, dispatch_id=0, wave_issued=True)],
        instructions=["v_mov"],
        comments=["/s/a.cpp:1"],
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
    )
    instance = make_db_analysis(str(tmp_path))
    # No ps_file_results.json on disk: a populated result proves the map was used.
    result = instance.calc_pc_sampling_data({str(tmp_path): tool_data})
    assert str(tmp_path) in result
    assert not result[str(tmp_path)].empty
    assert result[str(tmp_path)].iloc[0]["kernel_name"] == "vecCopy"


def test_calc_dispatch_data_uses_provided_tool_data(tmp_path: Path) -> None:
    """calc_dispatch_data builds PC-sampling dispatch rows from the provided map."""
    tool_data = make_tool_data(
        kernel_symbols=[make_kernel_symbol(100, 5, "vecCopy")],
        kernel_dispatch=[make_dispatch(0, 100, agent_handle=20, start=10, end=20)],
        agents=[make_agent(handle=20, node_id=2, agent_type=2)],
    )
    instance = make_db_analysis(str(tmp_path))
    instance._profiling_config = {"filter_blocks": ["21"]}  # pc_sampling_only -> True
    result = instance.calc_dispatch_data({str(tmp_path): tool_data})
    df = result[str(tmp_path)]
    assert list(df.columns) == [
        "dispatch_id",
        "kernel_name",
        "gpu_id",
        "start_timestamp",
        "end_timestamp",
    ]
    assert df.iloc[0]["kernel_name"] == "vecCopy"
    assert df.iloc[0]["gpu_id"] == 0


# ═══════════════════════════════════════════════════════════════
# PC sampling analyze integration tests
# ═══════════════════════════════════════════════════════════════


def test_pc_sampling_analyze_basic(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze on block 21 with default options and verify exit code 0."""
    workload_dir = common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    common.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_kernel_filter(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze on block 21 with a single-kernel filter and verify exit code 0."""
    workload_dir = common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--kernel",
        "0",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out
    assert "21. PC Sampling" in captured.out

    common.clean_output_dir(True, workload_dir)


@pytest.mark.parametrize("sorting_type", ["offset", "count"])
def test_pc_sampling_analyze_sorting_type(
    binary_handler_analyze_rocprof_compute,
    capsys,
    sorting_type,
) -> None:
    """Run analyze with each --pc-sampling-sorting-type and verify exit code 0."""
    workload_dir = common.setup_workload_dir(
        PC_SAMPLING_WORKLOAD, param_id=sorting_type
    )
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--pc-sampling-sorting-type",
        sorting_type,
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    common.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_db_output(
    binary_handler_analyze_rocprof_compute,
    monkeypatch,
) -> None:
    """Analyze in db mode produces a populated pcsampling table."""
    workload_dir = Path(common.setup_workload_dir(PC_SAMPLING_WORKLOAD)).resolve()
    db_name = "pc_sampling_db_test"
    db_path = workload_dir / f"{db_name}.db"
    # --output-name rejects path separators, so run from inside the workload
    # dir to keep the db there; clean_output_dir then removes it with the dir.
    monkeypatch.chdir(workload_dir)
    try:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            str(workload_dir),
            "--block",
            "21",
            "--output-format",
            "db",
            "--output-name",
            db_name,
        ])
        assert code == 0
        assert db_path.is_file()
        conn = sqlite3.connect(str(db_path))
        try:
            row_count = conn.execute(
                "SELECT COUNT(*) FROM compute_pcsampling"
            ).fetchone()[0]
        finally:
            conn.close()
        assert row_count > 0
    finally:
        common.clean_output_dir(True, str(workload_dir))


def test_pc_sampling_analyze_list_stats(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """
    Run analyze with --list-stats on a PC sampling workload
    and verify exit code 0.
    """
    workload_dir = common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    try:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--list-stats",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Detected Kernels" in captured.out
        assert "Dispatch list" in captured.out
    finally:
        common.clean_output_dir(True, workload_dir)
