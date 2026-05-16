# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.file_io PC-sampling helpers."""

from pathlib import Path

import pandas as pd

from utils import schema
from utils.file_io import (
    build_agent_to_gpu_map,
    process_pc_sampling_kernel_trace,
)
from utils.parser import nullify_unevaluated_metric_values


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
