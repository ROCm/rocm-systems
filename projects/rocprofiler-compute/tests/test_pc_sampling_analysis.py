# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import test_utils

from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map,
    process_pc_sampling_kernel_trace,
)
from utils.parser import (
    PMC_KERNEL_TOP_TABLE_ID,
    load_pc_sampling_data,
    load_pc_sampling_data_per_kernel,
    nullify_unevaluated_metric_values,
    search_pc_sampling_record,
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
    workload_dir = test_utils.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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

    test_utils.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_kernel_filter(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze on block 21 with a single-kernel filter and verify exit code 0."""
    workload_dir = test_utils.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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

    test_utils.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_sorting_type_offset(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze with --pc-sampling-sorting-type offset and verify exit code 0."""
    workload_dir = test_utils.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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

    test_utils.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_sorting_type_count(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze with --pc-sampling-sorting-type count and verify exit code 0."""
    workload_dir = test_utils.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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

    test_utils.clean_output_dir(True, workload_dir)


def test_pc_sampling_analyze_list_stats(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """
    Run analyze with --list-stats on a PC sampling workload
    and verify exit code 0.
    """
    workload_dir = test_utils.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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
        test_utils.clean_output_dir(True, workload_dir)
