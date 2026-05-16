# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.parser PC-sampling helpers."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from utils import schema
from utils.parser import (
    PMC_KERNEL_TOP_TABLE_ID,
    load_pc_sampling_data,
    load_pc_sampling_data_per_kernel,
    search_pc_sampling_record,
)

PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"


# ── Helpers for building synthetic JSON / records ────────────


def _make_record(
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


def _write_json(
    path: Path,
    stochastic: list | None = None,
    host_trap: list | None = None,
    instructions: list | None = None,
    comments: list | None = None,
    kernel_symbols: list | None = None,
) -> Path:
    data = {
        "rocprofiler-sdk-tool": [
            {
                "buffer_records": {
                    "pc_sample_stochastic": (
                        stochastic if stochastic is not None else []
                    ),
                    "pc_sample_host_trap": (host_trap if host_trap is not None else []),
                },
                "strings": {
                    "pc_sample_instructions": (
                        instructions if instructions is not None else []
                    ),
                    "pc_sample_comments": (comments if comments is not None else []),
                },
                "kernel_symbols": (
                    kernel_symbols if kernel_symbols is not None else []
                ),
            }
        ]
    }
    path.write_text(json.dumps(data))
    return path


def _write_kernel_trace(
    path: Path,
    rows: list[tuple],
) -> Path:
    lines = ["Dispatch_Id,Kernel_Id,Kernel_Name"]
    for dispatch_id, kernel_id, kernel_name in rows:
        lines.append(f"{dispatch_id},{kernel_id},{kernel_name}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_stochastic_csv(
    path: Path,
    rows: list[tuple],
) -> Path:
    lines = ["Correlation_Id,Instruction,Instruction_Comment"]
    for corr_id, instruction, comment in rows:
        lines.append(f"{corr_id},{instruction},{comment}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════
# search_pc_sampling_record
# ═══════════════════════════════════════════════════════════════


def test_search_pc_sampling_record_empty_list_returns_none() -> None:
    """Return None when the input record list is empty."""
    assert search_pc_sampling_record([]) is None


def test_search_pc_sampling_record_single_dict_input_issued() -> None:
    """Accept a single dict (not a list) and count it as one issued sample."""
    record = _make_record(
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
        _make_record(1, 0x10, 0, dispatch_id=0),
        _make_record(1, 0x10, 0, dispatch_id=1),
        _make_record(1, 0x20, 1, dispatch_id=2),
    ]
    result = search_pc_sampling_record(records)
    assert result is not None
    assert len(result) == 2
    assert result[0][3] == 2  # total_count for first key
    assert result[1][3] == 1


def test_search_pc_sampling_record_stall_reason_aggregation() -> None:
    """Aggregate distinct stall reasons and track stalled vs issued counts."""
    records = [
        _make_record(
            1,
            0x10,
            0,
            dispatch_id=0,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        _make_record(
            1,
            0x10,
            0,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}ALU_DEPENDENCY",
        ),
    ]
    result = search_pc_sampling_record(records)
    assert result is not None
    stall_reasons = result[0][6]
    reason_names = [r[0] for r in stall_reasons]
    assert "WAITCNT" in reason_names
    assert "ALU_DEPENDENCY" in reason_names
    assert result[0][4] == 0  # count_issued
    assert result[0][5] == 2  # count_stalled


def test_search_pc_sampling_record_dispatch_id_collection() -> None:
    """Collect unique dispatch IDs across duplicate records for the same key."""
    records = [
        _make_record(1, 0x10, 0, dispatch_id=0),
        _make_record(1, 0x10, 0, dispatch_id=0),
        _make_record(1, 0x10, 0, dispatch_id=1),
    ]
    result = search_pc_sampling_record(records)
    assert result is not None
    dispatch_ids = result[0][7]
    assert dispatch_ids == [0, 1]


def test_search_pc_sampling_record_skips_none_fields() -> None:
    """Skip records whose code_object_id or offset is None."""
    valid = _make_record(1, 0x10, 0, dispatch_id=0)
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


# ═══════════════════════════════════════════════════════════════
# load_pc_sampling_data_per_kernel
# ═══════════════════════════════════════════════════════════════


def _setup_per_kernel_files(
    tmp_path: Path,
    method: str = "host_trap",
) -> tuple[Path, Path]:
    """Create JSON + kernel trace CSV for per-kernel tests."""
    kernel_trace = tmp_path / "kt.csv"
    _write_kernel_trace(
        kernel_trace,
        [
            (0, 100, "vecCopy"),
            (1, 100, "vecCopy"),
            (2, 101, "vecAdd"),
        ],
    )

    samples = [
        _make_record(100, 0x10, 0, dispatch_id=0),
        _make_record(
            100,
            0x20,
            1,
            dispatch_id=1,
            wave_issued=False,
            stall_reason=f"{PREFIX}WAITCNT",
        ),
        _make_record(101, 0x10, 2, dispatch_id=2),
    ]

    key = "host_trap" if method == "host_trap" else "stochastic"
    kwargs = {key: samples}
    json_path = _write_json(
        tmp_path / "r.json",
        instructions=["v_mov_b32", "s_waitcnt", "v_add_f32"],
        comments=[
            "/src/vcopy.cpp:42",
            "/src/vcopy.cpp:43",
            "/src/vadd.cpp:30",
        ],
        kernel_symbols=[
            {
                "code_object_id": 100,
                "formatted_kernel_name": "vecCopy",
            },
            {
                "code_object_id": 101,
                "formatted_kernel_name": "vecAdd",
            },
        ],
        **kwargs,
    )
    return json_path, kernel_trace


def test_load_per_kernel_host_trap_offset_sort(
    tmp_path: Path,
) -> None:
    """
    Host-trap method returns offset-sorted rows without
    stall columns, filtered to the requested kernel.
    """
    json_path, kt = _setup_per_kernel_files(tmp_path, method="host_trap")
    df = load_pc_sampling_data_per_kernel(
        method="host_trap",
        file_name=json_path,
        csv_file_name=kt,
        kernel_name="vecCopy",
        sorting_type="offset",
    )
    assert not df.empty
    expected_cols = [
        "source_line",
        "instruction",
        "code_object_id",
        "offset",
        "count",
    ]
    assert list(df.columns) == expected_cols
    assert "count_issued" not in df.columns
    for _, row in df.iterrows():
        assert row["code_object_id"] == 100


def test_load_per_kernel_stochastic_count_sort(
    tmp_path: Path,
) -> None:
    """Stochastic method includes stall columns and sorts rows by descending count."""
    json_path, kt = _setup_per_kernel_files(tmp_path, method="stochastic")
    df = load_pc_sampling_data_per_kernel(
        method="stochastic",
        file_name=json_path,
        csv_file_name=kt,
        kernel_name="vecCopy",
        sorting_type="count",
    )
    assert not df.empty
    expected_cols = [
        "source_line",
        "instruction",
        "code_object_id",
        "offset",
        "count",
        "count_issued",
        "count_stalled",
        "stall_reason",
    ]
    assert list(df.columns) == expected_cols
    counts = df["count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_load_per_kernel_kernel_not_in_trace(
    tmp_path: Path,
) -> None:
    """
    Return an empty DataFrame when the requested kernel name
    is absent from the trace.
    """
    json_path, kt = _setup_per_kernel_files(tmp_path)
    df = load_pc_sampling_data_per_kernel(
        method="host_trap",
        file_name=json_path,
        csv_file_name=kt,
        kernel_name="nonexistent",
        sorting_type="offset",
    )
    assert df.empty


def test_load_per_kernel_no_pc_sample_key(
    tmp_path: Path,
) -> None:
    """
    When the JSON has no matching pc_sample key,
    search_key_in_json calls console_error which exits.
    """
    kt = tmp_path / "kt.csv"
    _write_kernel_trace(kt, [(0, 100, "vecCopy")])
    json_path = _write_json(tmp_path / "r.json")
    with pytest.raises(SystemExit):
        load_pc_sampling_data_per_kernel(
            method="host_trap",
            file_name=json_path,
            csv_file_name=kt,
            kernel_name="vecCopy",
            sorting_type="offset",
        )


def test_load_per_kernel_invalid_sorting_type(
    tmp_path: Path,
) -> None:
    """Return an empty DataFrame and log an error for an unrecognized sorting type."""
    json_path, kt = _setup_per_kernel_files(tmp_path)
    with patch("utils.parser.console_error"):
        df = load_pc_sampling_data_per_kernel(
            method="host_trap",
            file_name=json_path,
            csv_file_name=kt,
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


def test_load_pc_sampling_data_missing_kernel_trace(
    tmp_path: Path,
) -> None:
    """Return an empty DataFrame when the kernel trace CSV does not exist."""
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert df.empty


def test_load_pc_sampling_data_no_filter_stochastic_csv(
    tmp_path: Path,
) -> None:
    """Load grouped data from a stochastic CSV when no kernel filter is applied."""
    _write_stochastic_csv(
        tmp_path / "ps_file_pc_sampling_stochastic.csv",
        [
            (0, "v_mov_b32 v0 v1", "/src/vcopy.cpp:42"),
            (0, "s_waitcnt vmcnt(0)", "/src/vcopy.cpp:43"),
            (1, "v_mov_b32 v0 v1", "/src/vcopy.cpp:42"),
        ],
    )
    kt = tmp_path / "ps_file_kernel_trace.csv"
    kt.write_text("Dispatch_Id,Kernel_Id,Kernel_Name\n0,100,vecCopy\n1,100,vecCopy\n")
    workload = schema.Workload()
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert not df.empty
    assert list(df.columns) == [
        "source_line",
        "Kernel_Name",
        "instruction",
        "count",
    ]
    assert df.iloc[0]["source_line"].startswith("...")


def test_load_pc_sampling_data_multiple_kernels_error(
    tmp_path: Path,
) -> None:
    """
    Return an empty DataFrame and log an error when more
    than one kernel ID is filtered.
    """
    kt = tmp_path / "ps_file_kernel_trace.csv"
    kt.write_text("Dispatch_Id,Kernel_Id,Kernel_Name\n0,100,vecCopy\n")
    _write_stochastic_csv(
        tmp_path / "ps_file_pc_sampling_stochastic.csv",
        [(0, "v_mov", "/src/v.cpp:1")],
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
    kt = tmp_path / "ps_file_kernel_trace.csv"
    kt.write_text("Dispatch_Id,Kernel_Id,Kernel_Name\n0,100,vecCopy\n")
    _write_stochastic_csv(
        tmp_path / "ps_file_pc_sampling_stochastic.csv",
        [(0, "v_mov", "/src/v.cpp:1")],
    )
    samples = [_make_record(100, 0x10, 0, dispatch_id=0)]
    _write_json(
        tmp_path / "ps_file_results.json",
        stochastic=samples,
        instructions=["v_mov_b32"],
        comments=["/src/vcopy.cpp:42"],
        kernel_symbols=[
            {
                "code_object_id": 100,
                "formatted_kernel_name": "vecCopy",
            }
        ],
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
    kt = tmp_path / "ps_file_kernel_trace.csv"
    kt.write_text("Dispatch_Id,Kernel_Id,Kernel_Name\n0,100,vecCopy\n")
    _write_stochastic_csv(
        tmp_path / "ps_file_pc_sampling_stochastic.csv",
        [(0, "v_mov", "/src/v.cpp:1")],
    )
    _write_json(
        tmp_path / "ps_file_results.json",
        stochastic=[_make_record(100, 0x10, 0, dispatch_id=0)],
    )
    kernel_top_df = pd.DataFrame({"Kernel_Name": ["vecCopy", "vecAdd"]})
    workload = schema.Workload(
        filter_kernel_ids=[99],
        dfs={PMC_KERNEL_TOP_TABLE_ID: kernel_top_df},
    )
    df = load_pc_sampling_data(workload, str(tmp_path), "ps_file", "count")
    assert df.empty
