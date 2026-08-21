# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for the analyze CLI on PC sampling workloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI350"

# Source paths recorded in the fixture workload's code object metadata.
VCOPY_SOURCE = "/app/projects/rocprofiler-compute/sample/vcopy.cpp"
HIP_RUNTIME_SOURCE = (
    "/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/include/hip/"
    "amd_detail/amd_hip_runtime.h"
)


# ═══════════════════════════════════════════════════════════════
# PC sampling analyze integration tests
# ═══════════════════════════════════════════════════════════════


def test_pc_sampling_analyze_basic(
    binary_handler_analyze_rocprof_compute,
    capsys,
) -> None:
    """Run analyze on block 21 with default options and verify exit code 0."""
    workload_dir = integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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
    workload_dir = integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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
    workload_dir = integration_common.setup_workload_dir(
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


def test_pc_sampling_analyze_database_output(
    binary_handler_analyze_rocprof_compute,
    monkeypatch,
) -> None:
    """Preserve sampled rows, ISA attribution, and dispatches in database output."""
    workload_dir = Path(
        integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    ).resolve()
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
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "compute_code_object_store",
                    "compute_instruction_line",
                    "compute_pc_sample_state",
                    "compute_instruction_sample",
                )
            }
            line_count = counts["compute_instruction_line"]
            state_count = counts["compute_pc_sample_state"]
            state_total = conn.execute(
                "SELECT SUM(total_count) FROM compute_pc_sample_state"
            ).fetchone()[0]
            inst_sample_total = conn.execute(
                "SELECT SUM(count) FROM compute_instruction_sample"
            ).fetchone()[0]
            # Only dispatched kernels' ISA is stored, so every line is attributed.
            attributed = conn.execute(
                "SELECT COUNT(*) FROM compute_instruction_line il "
                "JOIN compute_kernel_symbol ks "
                "ON il.kernel_symbol_uuid = ks.kernel_symbol_uuid "
                "JOIN compute_kernel k ON ks.kernel_uuid = k.kernel_uuid"
            ).fetchone()[0]
            db_pc_sampling = pd.read_sql_query(
                "SELECT kernel_name, offset, instruction, source, count, "
                "count_issue, count_stall, wave_occupancy_percent, "
                "active_thread_percent, stall_reason "
                "FROM compute_pc_sampling_summary_view "
                "ORDER BY kernel_name, offset",
                conn,
            )
            # Derived wave metrics live on the state table; read them straight
            # from it so the summary view's copy is checked independently.
            db_wave_metrics = pd.read_sql_query(
                "SELECT active_thread_percent, wave_occupancy_percent "
                "FROM compute_pc_sample_state",
                conn,
            )
            pc_sampling_views = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'view' AND name LIKE 'compute_pc_sampling%' "
                "ORDER BY name"
            ).fetchall()
            db_dispatch_count = conn.execute(
                "SELECT dispatch_count FROM compute_kernel_view"
            ).fetchone()[0]
            db_code_object_process_ids = conn.execute(
                "SELECT DISTINCT pid FROM compute_code_object_store"
            ).fetchall()
            db_source_line_counts = conn.execute(
                "SELECT file_path, COUNT(*) FROM compute_source_lines_view "
                "GROUP BY file_path ORDER BY file_path"
            ).fetchall()
            # Content is clipped: these are long header lines.
            db_referenced_source_lines = conn.execute(
                "SELECT f.file_path, l.line_number, substr(l.content, 1, 45) "
                "FROM compute_instruction_source_line isl "
                "JOIN compute_source_line l "
                "ON l.source_line_uuid = isl.source_line_uuid "
                "JOIN compute_source_file f "
                "ON f.source_file_uuid = l.source_file_uuid "
                "GROUP BY f.file_path, l.line_number "
                "ORDER BY f.file_path, l.line_number"
            ).fetchall()
        finally:
            conn.close()
        assert counts["compute_code_object_store"] > 0
        # Only sampled offsets carry a sample state; the dispatched kernels' full
        # disassembly is added as extra lines, so lines outnumber states.
        assert state_count == 19
        assert line_count == 20
        # Un-dispatched ISA is never stored, so no line is left un-attributed.
        assert attributed == line_count
        # inst_type is a per-sample class, so its counts sum to the sample total.
        assert state_total == 857
        assert inst_sample_total == state_total
        assert len(db_pc_sampling) == 19
        assert db_pc_sampling["count"].sum() == 857
        assert len(db_wave_metrics) == state_count
        # Every sample in this workload carries a full wave64 mask
        # (0xFFFFFFFFFFFFFFFF) against sysinfo wave_size 64, so each line is
        # exactly 100% active. This pins the popcount against mask precision loss.
        assert (db_wave_metrics["active_thread_percent"] == 100.0).all()
        # wave_cnt varies per sample (4..32) against max_waves_per_cu 32, so
        # occupancy differs per line; assert only that it is populated and sane.
        assert db_wave_metrics["wave_occupancy_percent"].notna().all()
        assert (
            db_wave_metrics["wave_occupancy_percent"]
            .between(0, 100, inclusive="right")
            .all()
        )
        # vcopy is fully converged: every wave64 lane is active.
        assert db_pc_sampling["active_thread_percent"].eq(100.0).all()
        assert db_pc_sampling["wave_occupancy_percent"].gt(0).all()
        assert db_pc_sampling["wave_occupancy_percent"].le(100).all()
        assert pc_sampling_views == [("compute_pc_sampling_summary_view",)]
        assert db_dispatch_count == 3
        assert db_code_object_process_ids == [(698961,)]
        # The chain is rebuilt innermost first.
        assert (
            db_pc_sampling.loc[db_pc_sampling["offset"] == 8192, "source"].item()
            == f"{HIP_RUNTIME_SOURCE}:258 -> {HIP_RUNTIME_SOURCE}:317 "
            f"-> {VCOPY_SOURCE}:36"
        )
        assert (
            db_pc_sampling.loc[db_pc_sampling["offset"] == 8256, "source"].item()
            == f"{HIP_RUNTIME_SOURCE}:? -> {HIP_RUNTIME_SOURCE}:308 "
            f"-> {VCOPY_SOURCE}:36"
        )
        # Each file is stored whole, plus one null-line row for the ":?" frame.
        assert db_source_line_counts == [
            (VCOPY_SOURCE, 227),
            (HIP_RUNTIME_SOURCE, 414),
        ]
        assert db_referenced_source_lines == [
            (VCOPY_SOURCE, 36, "    int id = blockIdx.x*blockDim.x+threadIdx."),
            (VCOPY_SOURCE, 37, "    if (id < n)"),
            (VCOPY_SOURCE, 38, "      c[id] = a[id];"),
            (VCOPY_SOURCE, 39, "}"),
            (HIP_RUNTIME_SOURCE, None, None),
            (
                HIP_RUNTIME_SOURCE,
                253,
                "__DEVICE__ unsigned int __hip_get_block_idx_x",
            ),
            (
                HIP_RUNTIME_SOURCE,
                258,
                "__DEVICE__ unsigned int __hip_get_block_dim_x",
            ),
            (
                HIP_RUNTIME_SOURCE,
                308,
                "  __HIP_DEVICE_BUILTIN(x, __hip_get_block_idx",
            ),
            (
                HIP_RUNTIME_SOURCE,
                317,
                "  __HIP_DEVICE_BUILTIN_INTERNAL(x, __hip_get_",
            ),
        ]
    finally:
        common.clean_output_dir(True, str(workload_dir))


def test_pc_sampling_analyze_csv_output(
    binary_handler_analyze_rocprof_compute,
    monkeypatch,
) -> None:
    """Preserve PC sampling totals and dispatch counts in CSV output."""
    workload_dir = Path(
        integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    ).resolve()
    csv_name = "pc_sampling_csv_test"
    monkeypatch.chdir(workload_dir)
    try:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            str(workload_dir),
            "--block",
            "21",
            "--output-format",
            "csv",
            "--output-name",
            csv_name,
        ])
        assert code == 0

        csv_dir = workload_dir / csv_name
        summary_csv = csv_dir / "pc_sampling_summary.csv"
        assert summary_csv.is_file()
        csv_pc_sampling = pd.read_csv(summary_csv)
        csv_kernel = pd.read_csv(csv_dir / "kernel.csv")
        assert len(csv_pc_sampling) == 19
        assert csv_pc_sampling["count"].sum() == 857
        assert set(csv_pc_sampling["pid"]) == {698961}
        percent_columns = [
            "wave_occupancy_percent",
            "active_thread_percent",
        ]
        assert csv_pc_sampling[percent_columns].notna().all().all()
        assert csv_kernel.iloc[0]["dispatch_count"] == 3
        csv_source_lines = pd.read_csv(csv_dir / "source_lines.csv")
        assert set(csv_source_lines["file_path"]) == {
            VCOPY_SOURCE,
            HIP_RUNTIME_SOURCE,
        }
        # Every sampling row must resolve to a kernel the kernel view exposes.
        assert set(csv_pc_sampling["kernel_uuid"]) <= set(csv_kernel["kernel_uuid"])

        isa_percent_columns = [
            "Wave occupancy percent",
            "Active thread percent",
        ]
        isa_csv_paths = sorted((csv_dir / "per_kernel_pc_sampling").rglob("*.csv"))
        assert isa_csv_paths
        has_populated_sample_metrics = False
        for isa_csv_path in isa_csv_paths:
            isa_rows = pd.read_csv(isa_csv_path)
            if not set(isa_percent_columns) <= set(isa_rows.columns):
                continue
            sampled_isa_rows = isa_rows.loc[isa_rows["Total count"].notna()]
            if (
                not sampled_isa_rows.empty
                and sampled_isa_rows[isa_percent_columns].notna().all().all()
            ):
                has_populated_sample_metrics = True
                break
        assert has_populated_sample_metrics
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
    workload_dir = integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
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
