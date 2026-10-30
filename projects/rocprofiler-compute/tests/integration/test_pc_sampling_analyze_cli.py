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
    pc_sampling_output = captured.out.split("21. PC Sampling", maxsplit=1)[1]
    assert "issue_count" in pc_sampling_output
    assert "stall_count" in pc_sampling_output

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
    tmp_path,
) -> None:
    """Preserve sampled rows, ISA attribution, and dispatches in database output."""
    workload_dir = Path(
        integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    ).resolve()
    run_folder = tmp_path / "pc_sampling_db_test"
    db_path = run_folder / "analysis.db"
    try:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            str(workload_dir),
            "--block",
            "21",
            "--output-directory",
            str(tmp_path),
            "--output-name",
            run_folder.name,
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
            # The state table also holds source line states; count instruction
            # states only.
            state_count = conn.execute(
                "SELECT COUNT(*) FROM compute_pc_sample_state "
                "WHERE instruction_uuid IS NOT NULL"
            ).fetchone()[0]
            state_total = conn.execute(
                "SELECT SUM(total_count) FROM compute_pc_sample_state "
                "WHERE instruction_uuid IS NOT NULL"
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
                "SELECT kernel_name, offset, instruction, instruction_type, "
                "source, count, "
                "issue_count, stall_count, "
                "wave_occupancy_percent, active_thread_percent, stall_reason "
                "FROM compute_pc_sampling_summary_view "
                "ORDER BY kernel_name, offset",
                conn,
            )
            # Derived wave metrics live on the state table; the summary view
            # does not carry them.
            db_wave_metrics = pd.read_sql_query(
                "SELECT active_thread_percent, wave_occupancy_percent "
                "FROM compute_pc_sample_state WHERE instruction_uuid IS NOT NULL",
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
        # The static type is filled from the disassembly, so every sampled line
        # of a real workload carries one.
        assert db_pc_sampling["instruction_type"].notna().all()
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
    tmp_path,
) -> None:
    """Preserve PC sampling totals and dispatch counts in the workload folder."""
    workload_dir = Path(
        integration_common.setup_workload_dir(PC_SAMPLING_WORKLOAD)
    ).resolve()
    csv_dir = tmp_path / "pc_sampling_csv_test"
    try:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            str(workload_dir),
            "--block",
            "21",
            "--output-directory",
            str(tmp_path),
            "--output-name",
            csv_dir.name,
        ])
        assert code == 0

        workload_folder = next(csv_dir.glob("1-*"))
        summary = pd.read_csv(workload_folder / "pc-sampling-summary.csv", comment="#")
        kernels = pd.read_csv(workload_folder / "kernels.csv")
        dispatches = pd.read_csv(workload_folder / "dispatches.csv")
        isa_frames = {
            path.parent.name: pd.read_csv(path)
            for path in workload_folder.glob("kernel-*/*-pc-sampling-isa.csv")
        }
        sampled_lines = pd.concat(
            frame.dropna(subset=["Total count"]) for frame in isa_frames.values()
        )
        source_frames = [
            pd.read_csv(path)
            for path in workload_folder.glob("kernel-*/*-pc-sampling-src.csv")
        ]

        # One summary row per kernel with samples, and one ISA and source file each
        assert len(summary) == len(isa_frames) == len(source_frames) > 0
        assert summary["Total count"].sum() == 857
        assert set(sampled_lines["Pid"]) == {698961}
        assert sampled_lines["Active thread percent"].eq(100.0).all()
        assert sampled_lines["Wave occupancy percent"].gt(0).all()
        assert sampled_lines["Wave occupancy percent"].le(100).all()
        assert (dispatches["Kernel_Index"] == 0).sum() == 3
        assert set(pd.concat(source_frames)["Source file"].dropna()) == {
            VCOPY_SOURCE,
            HIP_RUNTIME_SOURCE,
        }
        # Every sampled kernel is listed in kernels.csv under its folder
        assert set(summary["Folder"]) <= set(kernels["Folder"])
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
