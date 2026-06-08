# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path
from unittest.mock import patch

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_db import (
    _attribute_pc_samples_native,
    db_analysis,
)
from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map,
    process_pc_sampling_kernel_trace,
)
from utils.parser import (
    PMC_KERNEL_TOP_TABLE_ID,
    load_code_obj_info,
    load_pc_sampling_data,
    load_pc_sampling_data_per_kernel,
    match_instruction_for_offset,
    nullify_unevaluated_metric_values,
    resolve_snapshot_source_path,
    resolve_source_file,
    search_pc_sampling_record,
    split_instruction_comment,
)

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI300A_A1"

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


def _write_pc_kernel_trace(path: Path, rows: list[tuple]) -> Path:
    """Write a minimal ps_file_kernel_trace.csv.

    Each *row* is ``(agent_id, dispatch_id, kernel_name, start_ts, end_ts)``.
    Only the columns actually read by ``process_pc_sampling_kernel_trace``
    are written (the existing ``_write_kernel_trace`` uses a different
    schema without Agent_Id or timestamps, so it cannot be reused here).
    """
    lines = ["Agent_Id,Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp"]
    for agent_id, dispatch_id, kernel_name, start_ts, end_ts in rows:
        lines.append(f"{agent_id},{dispatch_id},{kernel_name},{start_ts},{end_ts}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_process_pc_sampling_missing_trace_returns_empty(
    tmp_path: Path,
) -> None:
    """Return empty DataFrame with expected columns when trace is absent."""
    df = process_pc_sampling_kernel_trace(str(tmp_path))
    assert df.empty
    assert list(df.columns) == [
        "Dispatch_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]


def test_process_pc_sampling_with_agent_info(tmp_path: Path) -> None:
    """Verify column selection, GPU mapping, timestamps, and extra column dropping."""
    _write_pc_kernel_trace(
        tmp_path / "ps_file_kernel_trace.csv",
        [
            ("Agent 2", 1, "vecCopy", 1981199661678356, 1981199662835032),
            ("Agent 3", 2, "vecAdd", 2000, 3000),
            ("Agent 99", 3, "vecMul", 4000, 5000),
        ],
    )
    agent_csv = tmp_path / "ps_file_agent_info.csv"
    agent_csv.write_text("Node_Id,Agent_Type\n1,CPU\n2,GPU\n3,GPU\n")

    df = process_pc_sampling_kernel_trace(str(tmp_path))

    # Correct shape and columns
    assert len(df) == 3
    assert list(df.columns) == [
        "Dispatch_Id",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "GPU_ID",
    ]

    # Multi-GPU mapping: Agent 2 -> GPU 0, Agent 3 -> GPU 1, unknown -> 0
    assert df["GPU_ID"].tolist() == [0, 1, 0]
    assert df["Kernel_Name"].tolist() == ["vecCopy", "vecAdd", "vecMul"]

    # Timestamps passed through unchanged
    assert df["Start_Timestamp"].iloc[0] == 1981199661678356
    assert df["End_Timestamp"].iloc[0] == 1981199662835032


def test_process_pc_sampling_no_agent_info(tmp_path: Path) -> None:
    """Default GPU_ID to 0 when ps_file_agent_info.csv is missing."""
    _write_pc_kernel_trace(
        tmp_path / "ps_file_kernel_trace.csv",
        [("Agent 99", 1, "vecCopy", 1000, 2000)],
    )
    df = process_pc_sampling_kernel_trace(str(tmp_path))
    assert len(df) == 1
    assert df["GPU_ID"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════
# build_agent_to_gpu_map
# ═══════════════════════════════════════════════════════════════


def test_build_agent_to_gpu_map_single_gpu(
    tmp_path: Path,
) -> None:
    """Map one GPU agent to GPU index 0, ignoring CPU agents."""
    csv_path = tmp_path / "agent_info.csv"
    csv_path.write_text("Node_Id,Agent_Type\n1,CPU\n2,GPU\n")
    result = build_agent_to_gpu_map(csv_path)
    assert result == {"Agent 2": 0}


def test_build_agent_to_gpu_map_two_gpus(
    tmp_path: Path,
) -> None:
    """Assign sequential GPU indices to multiple GPU agents sorted by Node_Id."""
    csv_path = tmp_path / "agent_info.csv"
    csv_path.write_text("Node_Id,Agent_Type\n1,CPU\n3,GPU\n2,GPU\n")
    result = build_agent_to_gpu_map(csv_path)
    assert result == {"Agent 2": 0, "Agent 3": 1}


def test_build_agent_to_gpu_map_no_gpu_agents(
    tmp_path: Path,
) -> None:
    """Return an empty map when no GPU agents are present in the CSV."""
    csv_path = tmp_path / "agent_info.csv"
    csv_path.write_text("Node_Id,Agent_Type\n1,CPU\n2,CPU\n")
    result = build_agent_to_gpu_map(csv_path)
    assert result == {}


def test_build_agent_to_gpu_map_missing_file(
    tmp_path: Path,
) -> None:
    """Return an empty map when the agent info CSV file does not exist."""
    result = build_agent_to_gpu_map(tmp_path / "nonexistent.csv")
    assert result == {}


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


def test_pc_sampling_analyze_sorting_type_offset(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze with --pc-sampling-sorting-type offset and verify exit code 0."""
    workload_dir = common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--pc-sampling-sorting-type",
        "offset",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    common.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_sorting_type_count(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze with --pc-sampling-sorting-type count and verify exit code 0."""
    workload_dir = common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--block",
        "21",
        "--pc-sampling-sorting-type",
        "count",
    ])
    assert code == 0
    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    common.clean_output_dir(True, workload_dir)


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


# ═══════════════════════════════════════════════════════════════
# split_instruction_comment
# ═══════════════════════════════════════════════════════════════


def test_split_instruction_comment_basic() -> None:
    """Split a simple comment into (source_file, line) at the ':'."""
    assert split_instruction_comment("/a/b.cpp:42") == ("/a/b.cpp", "42")


def test_split_instruction_comment_splits_on_last_colon() -> None:
    """Split on the LAST ':' so Windows-style drive prefixes survive."""
    assert split_instruction_comment("C:/x.cpp:5") == ("C:/x.cpp", "5")


def test_split_instruction_comment_line_is_str() -> None:
    """The returned line is always a string instance, never an int."""
    _, line = split_instruction_comment("/a/b.cpp:42")
    assert isinstance(line, str)


def test_split_instruction_comment_empty_string() -> None:
    """An empty comment yields (None, None)."""
    assert split_instruction_comment("") == (None, None)


def test_split_instruction_comment_none() -> None:
    """A None comment yields (None, None)."""
    assert split_instruction_comment(None) == (None, None)


def test_split_instruction_comment_no_colon() -> None:
    """A comment with no ':' yields (None, None)."""
    assert split_instruction_comment("noColon") == (None, None)


def test_split_instruction_comment_empty_side() -> None:
    """An empty file or line side yields (None, None)."""
    assert split_instruction_comment(":42") == (None, None)
    assert split_instruction_comment("/a/b.cpp:") == (None, None)


# ═══════════════════════════════════════════════════════════════
# resolve_snapshot_source_path
# ═══════════════════════════════════════════════════════════════


def test_resolve_snapshot_source_path_original_present(tmp_path: Path) -> None:
    """When the original source path exists, keep the default display (None)."""
    src = tmp_path / "kernel.hip"
    src.write_text("// src\n")
    comment = f"{src}:42"
    assert resolve_snapshot_source_path(comment, tmp_path) is None


def test_resolve_snapshot_source_path_falls_back_to_snapshot(tmp_path: Path) -> None:
    """A missing original resolves to the snapshot copy under code_obj_sources/."""
    missing = "/nonexistent/build/kernel.hip"
    snapshot = tmp_path / "code_obj_sources" / missing.lstrip("/")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("// snapshot\n")

    resolved = resolve_snapshot_source_path(f"{missing}:42", tmp_path)
    assert resolved == f"{snapshot}:42"


def test_resolve_snapshot_source_path_no_snapshot(tmp_path: Path) -> None:
    """A missing original with no snapshot copy returns None (default display)."""
    assert resolve_snapshot_source_path("/gone/kernel.hip:42", tmp_path) is None


def test_resolve_snapshot_source_path_unparsable_comment(tmp_path: Path) -> None:
    """A comment without a parsable path returns None."""
    assert resolve_snapshot_source_path("noColon", tmp_path) is None


def test_resolve_snapshot_source_path_rejects_traversal(tmp_path: Path) -> None:
    """A '..' path that would escape code_obj_sources/ resolves to None."""
    assert resolve_snapshot_source_path("../../etc/passwd:1", tmp_path) is None


# ═══════════════════════════════════════════════════════════════
# resolve_source_file (bare path, native attribution path)
# ═══════════════════════════════════════════════════════════════


def test_resolve_source_file_keeps_present_original(tmp_path: Path) -> None:
    """An existing capture-host path is returned unchanged."""
    src = tmp_path / "kernel.hip"
    src.write_text("// src\n")
    assert resolve_source_file(str(src), tmp_path) == str(src)


def test_resolve_source_file_redirects_to_snapshot(tmp_path: Path) -> None:
    """A missing original redirects to the code_obj_sources/ snapshot copy."""
    missing = "/nonexistent/build/kernel.hip"
    snapshot = tmp_path / "code_obj_sources" / missing.lstrip("/")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("// snapshot\n")
    assert resolve_source_file(missing, tmp_path) == str(snapshot)


def test_resolve_source_file_keeps_original_when_no_snapshot(tmp_path: Path) -> None:
    """A missing original with no snapshot copy is returned unchanged."""
    assert resolve_source_file("/gone/kernel.hip", tmp_path) == "/gone/kernel.hip"


def test_resolve_source_file_none_passthrough(tmp_path: Path) -> None:
    """A None source_file yields None."""
    assert resolve_source_file(None, tmp_path) is None


# ═══════════════════════════════════════════════════════════════
# Helpers for native code-object JSON
# ═══════════════════════════════════════════════════════════════


def _write_code_obj_info(
    path: Path,
    code_object_id: int,
    symbol_name: str,
    instructions: list[dict],
) -> Path:
    """Write a minimal ``*_code_obj_info.json`` file for one code object."""
    data = {
        "code_objects": [
            {
                "id": code_object_id,
                "symbols": [
                    {
                        "name": symbol_name,
                        "code_object_offset": 0,
                        "virtual_address": 1000,
                        "size": sum(i.get("size", 0) for i in instructions),
                        "instructions": instructions,
                    }
                ],
            }
        ]
    }
    path.write_text(json.dumps(data))
    return path


def _vec_copy_instructions() -> list[dict]:
    """Two synthetic instructions for a vecCopy symbol."""
    return [
        {
            "name": "s_load_b64",
            "comment": "/home/u/kernel.hip:42",
            "virtual_address": 1000,
            "code_obj_offset": 0,
            "size": 4,
        },
        {
            "name": "v_add_u32",
            "comment": "/home/u/kernel.hip:43",
            "virtual_address": 1004,
            "code_obj_offset": 4,
            "size": 4,
        },
    ]


# ═══════════════════════════════════════════════════════════════
# load_code_obj_info
# ═══════════════════════════════════════════════════════════════


def test_load_code_obj_info_none_when_absent(tmp_path: Path) -> None:
    """Return None when the directory has no ``*_code_obj_info.json``."""
    assert load_code_obj_info(tmp_path) is None


def test_load_code_obj_info_sorted_intervals(tmp_path: Path) -> None:
    """Return per-id instruction lists sorted ascending by code_obj_offset."""
    # Write instructions out of offset order to verify sorting.
    _write_code_obj_info(
        tmp_path / "12345_code_obj_info.json",
        code_object_id=2,
        symbol_name="vecCopy",
        instructions=[
            {
                "name": "v_add_u32",
                "comment": "/home/u/kernel.hip:43",
                "code_obj_offset": 4,
                "size": 4,
            },
            {
                "name": "s_load_b64",
                "comment": "/home/u/kernel.hip:42",
                "code_obj_offset": 0,
                "size": 4,
            },
        ],
    )
    merged = load_code_obj_info(tmp_path)
    assert merged is not None
    assert set(merged.keys()) == {2}
    offsets = [inst["code_obj_offset"] for inst in merged[2]]
    assert offsets == [0, 4]
    assert merged[2][0]["name"] == "s_load_b64"
    assert merged[2][0]["comment"] == "/home/u/kernel.hip:42"


def test_load_code_obj_info_multi_pid_merge(tmp_path: Path) -> None:
    """Merge two per-PID files so both code object ids are present."""
    _write_code_obj_info(
        tmp_path / "111_code_obj_info.json",
        code_object_id=2,
        symbol_name="vecCopy",
        instructions=_vec_copy_instructions(),
    )
    _write_code_obj_info(
        tmp_path / "222_code_obj_info.json",
        code_object_id=3,
        symbol_name="vecAdd",
        instructions=[
            {
                "name": "v_mul_f32",
                "comment": "/home/u/add.hip:7",
                "code_obj_offset": 0,
                "size": 8,
            }
        ],
    )
    merged = load_code_obj_info(tmp_path)
    assert merged is not None
    assert set(merged.keys()) == {2, 3}
    assert len(merged[2]) == 2
    assert merged[3][0]["name"] == "v_mul_f32"


def test_load_code_obj_info_skips_malformed_entries(tmp_path: Path) -> None:
    """Instructions missing offset or size are skipped; valid ones survive."""
    _write_code_obj_info(
        tmp_path / "111_code_obj_info.json",
        code_object_id=2,
        symbol_name="vecCopy",
        instructions=[
            {  # missing "size"
                "name": "s_load_b64",
                "comment": "/home/u/kernel.hip:42",
                "code_obj_offset": 0,
            },
            {  # missing "code_obj_offset"
                "name": "s_nop",
                "comment": "/home/u/kernel.hip:43",
                "size": 4,
            },
            {  # valid
                "name": "v_add_u32",
                "comment": "/home/u/kernel.hip:44",
                "code_obj_offset": 4,
                "size": 4,
            },
        ],
    )
    merged = load_code_obj_info(tmp_path)
    assert merged is not None
    assert len(merged[2]) == 1
    assert merged[2][0]["name"] == "v_add_u32"


def test_load_code_obj_info_skips_code_object_without_id(tmp_path: Path) -> None:
    """A code_objects entry missing 'id' is skipped; a valid sibling survives."""
    data = {
        "code_objects": [
            {  # no "id" -> skipped
                "symbols": [
                    {
                        "name": "orphan",
                        "instructions": [
                            {
                                "name": "v_nop",
                                "comment": "/home/u/orphan.hip:1",
                                "code_obj_offset": 0,
                                "size": 4,
                            }
                        ],
                    }
                ],
            },
            {
                "id": 7,
                "symbols": [
                    {
                        "name": "vecAdd",
                        "instructions": [
                            {
                                "name": "v_mul_f32",
                                "comment": "/home/u/add.hip:7",
                                "code_obj_offset": 0,
                                "size": 8,
                            }
                        ],
                    }
                ],
            },
        ]
    }
    (tmp_path / "111_code_obj_info.json").write_text(json.dumps(data))
    merged = load_code_obj_info(tmp_path)
    assert merged is not None
    assert set(merged.keys()) == {7}
    assert merged[7][0]["name"] == "v_mul_f32"


def test_load_code_obj_info_skips_unreadable_file(tmp_path: Path) -> None:
    """A truncated/invalid-JSON file is skipped; a valid sibling still loads."""
    bad = tmp_path / "111_code_obj_info.json"
    bad.write_text('{"code_objects": [  truncated')
    _write_code_obj_info(
        tmp_path / "222_code_obj_info.json",
        code_object_id=3,
        symbol_name="vecAdd",
        instructions=[
            {
                "name": "v_mul_f32",
                "comment": "/home/u/add.hip:7",
                "code_obj_offset": 0,
                "size": 8,
            }
        ],
    )
    merged = load_code_obj_info(tmp_path)
    assert merged is not None
    assert set(merged.keys()) == {3}
    assert merged[3][0]["name"] == "v_mul_f32"


def test_load_code_obj_info_all_unreadable_returns_none(tmp_path: Path) -> None:
    """When every native file fails to parse, return None for the SDK fallback."""
    (tmp_path / "111_code_obj_info.json").write_text("{ not json")
    assert load_code_obj_info(tmp_path) is None


# ═══════════════════════════════════════════════════════════════
# match_instruction_for_offset
# ═══════════════════════════════════════════════════════════════


def _intervals() -> list[dict]:
    """Two adjacent 4-byte intervals: [0,4) and [4,8)."""
    return [
        {"code_obj_offset": 0, "size": 4, "name": "s_load_b64"},
        {"code_obj_offset": 4, "size": 4, "name": "v_add_u32"},
    ]


def test_match_instruction_for_offset_inside_interval() -> None:
    """An offset inside an interval returns that instruction."""
    intervals = _intervals()
    assert match_instruction_for_offset(intervals, 0)["name"] == "s_load_b64"
    assert match_instruction_for_offset(intervals, 3)["name"] == "s_load_b64"
    assert match_instruction_for_offset(intervals, 4)["name"] == "v_add_u32"
    assert match_instruction_for_offset(intervals, 7)["name"] == "v_add_u32"


def test_match_instruction_for_offset_past_all_ranges() -> None:
    """An offset past every interval returns None."""
    assert match_instruction_for_offset(_intervals(), 8) is None
    assert match_instruction_for_offset(_intervals(), 100) is None


def test_match_instruction_for_offset_in_gap_or_before_first() -> None:
    """An offset before the first interval (in a gap) returns None."""
    gapped = [
        {"code_obj_offset": 4, "size": 4, "name": "v_add_u32"},
        {"code_obj_offset": 16, "size": 4, "name": "v_mul_f32"},
    ]
    # Before the first interval.
    assert match_instruction_for_offset(gapped, 0) is None
    # In the gap between [4,8) and [16,20).
    assert match_instruction_for_offset(gapped, 10) is None


def test_match_instruction_for_offset_empty_list() -> None:
    """An empty interval list returns None."""
    assert match_instruction_for_offset([], 0) is None


def test_match_instruction_for_offset_unknown_id_via_get() -> None:
    """Simulate an unknown code object id by passing .get(99, []) -> None."""
    code_obj_info = {2: _intervals()}
    assert match_instruction_for_offset(code_obj_info.get(99, []), 0) is None


# ═══════════════════════════════════════════════════════════════
# calc_pc_sampling_data (db_analysis method)
# ═══════════════════════════════════════════════════════════════


class _StubRuns:
    """Minimal stand-in for db_analysis exposing only ``_runs``.

    ``calc_pc_sampling_data`` iterates ``self._runs.keys()`` (workload-path
    strings) and reads files from disk, so the only attribute it needs is
    ``_runs``. We invoke the method unbound on this stub to avoid the heavy
    real ``db_analysis`` constructor.
    """

    def __init__(self, workload_paths: list[str]) -> None:
        self._runs = {path: None for path in workload_paths}


_PC_SAMPLING_EXPECTED_COLS = {
    "offset",
    "count",
    "count_issued",
    "count_stalled",
    "stall_reason",
    "kernel_name",
    "instruction",
    "source_line",
    "source_file",
    "line",
}


def test_calc_pc_sampling_data_native_branch(tmp_path: Path) -> None:
    """NATIVE branch: range-match samples against synthetic disassembly."""
    # Samples landing on offset 0 (s_load_b64) and offset 4 (v_add_u32),
    # both in code object id 2.
    samples = [
        _make_record(2, 0, inst_index=0, dispatch_id=0),
        _make_record(2, 4, inst_index=1, dispatch_id=1),
    ]
    _write_json(
        tmp_path / "ps_file_results.json",
        stochastic=samples,
        instructions=["UNUSED_INST_0", "UNUSED_INST_1"],
        comments=["unused:0", "unused:1"],
        kernel_symbols=[{"code_object_id": 2, "formatted_kernel_name": "vecCopy"}],
    )
    _write_code_obj_info(
        tmp_path / "12345_code_obj_info.json",
        code_object_id=2,
        symbol_name="vecCopy",
        instructions=_vec_copy_instructions(),
    )

    stub = _StubRuns([str(tmp_path)])
    result = db_analysis.calc_pc_sampling_data(stub)

    df = result[str(tmp_path)]
    assert _PC_SAMPLING_EXPECTED_COLS.issubset(set(df.columns))

    # Native attribution: row for offset 0 -> s_load_b64 / kernel.hip:42.
    row0 = df[df["offset"] == 0].iloc[0]
    assert row0["instruction"] == "s_load_b64"
    assert row0["source_line"] == "/home/u/kernel.hip:42"
    assert row0["source_file"] == "/home/u/kernel.hip"
    assert row0["line"] == "42"
    assert row0["kernel_name"] == "vecCopy"

    row4 = df[df["offset"] == 4].iloc[0]
    assert row4["instruction"] == "v_add_u32"
    assert row4["source_file"] == "/home/u/kernel.hip"
    assert row4["line"] == "43"


def test_attribute_pc_samples_native_no_match_offset(tmp_path: Path) -> None:
    """An offset that falls in a gap yields None for every native column."""
    code_obj_info = {
        2: [
            {
                "code_obj_offset": 0,
                "size": 4,
                "name": "s_load_b64",
                "comment": "/home/u/kernel.hip:42",
            },
            {
                "code_obj_offset": 16,
                "size": 4,
                "name": "v_add_u32",
                "comment": "/home/u/kernel.hip:43",
            },
        ]
    }
    # Offset 8 falls in the [4, 16) gap; offset 16 matches the second interval.
    grouped_df = pd.DataFrame({"code_object_id": [2, 2], "code_object_offset": [8, 16]})

    columns = _attribute_pc_samples_native(grouped_df, code_obj_info, tmp_path)

    # Gap offset -> all native columns None.
    assert columns["instruction"][0] is None
    assert columns["source_line"][0] is None
    assert columns["source_file"][0] is None
    assert columns["line"][0] is None

    # Matched offset still resolves.
    assert columns["instruction"][1] == "v_add_u32"
    assert columns["source_file"][1] == "/home/u/kernel.hip"
    assert columns["line"][1] == "43"


def test_calc_pc_sampling_data_fallback_branch(tmp_path: Path) -> None:
    """FALLBACK branch: no native JSON -> use SDK strings by inst_index."""
    samples = [
        _make_record(100, 0x10, inst_index=0, dispatch_id=0),
        _make_record(100, 0x20, inst_index=1, dispatch_id=1),
    ]
    _write_json(
        tmp_path / "ps_file_results.json",
        stochastic=samples,
        instructions=["v_mov_b32", "s_waitcnt"],
        comments=["/src/vcopy.cpp:42", "/src/vcopy.cpp:43"],
        kernel_symbols=[{"code_object_id": 100, "formatted_kernel_name": "vecCopy"}],
    )

    stub = _StubRuns([str(tmp_path)])
    result = db_analysis.calc_pc_sampling_data(stub)

    df = result[str(tmp_path)]
    assert _PC_SAMPLING_EXPECTED_COLS.issubset(set(df.columns))

    # inst_index 0 maps to the first SDK string/comment.
    row0 = df[df["offset"] == 0x10].iloc[0]
    assert row0["instruction"] == "v_mov_b32"
    assert row0["source_line"] == "/src/vcopy.cpp:42"
    # Fallback path leaves source_file / line unset.
    assert row0["source_file"] is None
    assert row0["line"] is None
    assert row0["kernel_name"] == "vecCopy"
