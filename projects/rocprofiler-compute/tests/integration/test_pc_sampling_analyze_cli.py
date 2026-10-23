# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for the analyze CLI on PC sampling workloads."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common

PC_SAMPLING_WORKLOAD = "tests/workloads/vcopy_pc_sampling_only/MI350"


@pytest.fixture
def capture_expectations():
    """Read capture-dependent counts and source attribution from the raw fixture."""
    workload = Path(PC_SAMPLING_WORKLOAD)
    sampling_path = next(workload.glob("*_ps_file_results.json"))
    process_id = int(sampling_path.name.split("_", 1)[0])
    tool = json.loads(sampling_path.read_text(encoding="utf-8"))[
        "rocprofiler-sdk-tool"
    ][0]
    code_objects = json.loads(
        (workload / f"{process_id}_code_obj_info.json").read_text(encoding="utf-8")
    )["code_objects"]
    assert len(code_objects) == 1
    assert [symbol["name"] for symbol in code_objects[0]["symbols"]] == [
        "_Z7vecCopyPdS_S_ii"
    ]
    source_paths = json.loads(
        (workload / "src" / f"{process_id}_source_map.json").read_text(encoding="utf-8")
    )["source_paths"]
    snapshots = {
        canonical: (workload / "src" / canonical.lstrip("/"))
        .read_text(encoding="utf-8")
        .splitlines()
        for canonical in source_paths.values()
    }
    samples = tool["buffer_records"]["pc_sample_stochastic"]
    sample_counts = Counter(
        sample["record"]["pc"]["code_object_offset"] for sample in samples
    )
    sources_by_offset = {
        sample["record"]["pc"]["code_object_offset"]: tool["strings"][
            "pc_sample_comments"
        ][sample["inst_index"]]
        for sample in samples
    }
    load_base = next(
        code_object["load_base"]
        for code_object in tool["code_objects"]
        if code_object["code_object_id"] == code_objects[0]["id"]
    )
    instructions = code_objects[0]["symbols"][0]["instructions"]
    all_sources = {
        instruction["virtual_address"] - load_base: instruction["comment"]
        for instruction in instructions
    }
    all_sources.update(sources_by_offset)
    assert set(source_paths) == {
        frame.rsplit(":", 1)[0]
        for instruction in instructions
        for frame in instruction["comment"].split(" -> ")
        if frame
    }
    snapshot_files = {
        path.relative_to(workload / "src").as_posix()
        for path in (workload / "src").rglob("*")
        if path.is_file() and not path.name.endswith("_source_map.json")
    }
    assert snapshot_files == {path.lstrip("/") for path in snapshots}
    source_line_counts, referenced_lines, canonical_sources = source_expectations(
        source_paths, snapshots, all_sources, sources_by_offset
    )
    return {
        "pid": process_id,
        "sample_counts": sample_counts,
        "sample_total": len(samples),
        "instruction_count": len(all_sources),
        "sources": canonical_sources,
        "source_line_counts": source_line_counts,
        "referenced_lines": referenced_lines,
        "source_paths": set(snapshots),
        "dispatch_count": len(tool["buffer_records"]["kernel_dispatch"]),
    }


def source_expectations(source_paths, snapshots, all_sources, sources_by_offset):
    """Compute canonical frame chains and referenced snapshot lines."""
    frames = {
        (source_paths[path], None if line == "?" else int(line))
        for comment in all_sources.values()
        for frame in comment.split(" -> ")
        if frame
        for path, line in [frame.rsplit(":", 1)]
    }
    source_line_counts = sorted(
        (path, len(lines) + ((path, None) in frames))
        for path, lines in snapshots.items()
    )
    referenced_lines = sorted(
        [
            (
                path,
                line,
                snapshots[path][line - 1][:45] if line is not None else None,
            )
            for path, line in frames
        ],
        key=lambda row: (row[0], row[1] or 0),
    )
    canonical_sources = {
        offset: " -> ".join(
            f"{source_paths[path]}:{line}"
            for frame in comment.split(" -> ")
            if frame
            for path, line in [frame.rsplit(":", 1)]
        )
        for offset, comment in sources_by_offset.items()
    }
    return source_line_counts, referenced_lines, canonical_sources


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
    capture_expectations,
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
        assert state_count == len(capture_expectations["sample_counts"])
        assert line_count == capture_expectations["instruction_count"]
        # Un-dispatched ISA is never stored, so no line is left un-attributed.
        assert attributed == line_count
        # inst_type is a per-sample class, so its counts sum to the sample total.
        assert state_total == capture_expectations["sample_total"]
        assert inst_sample_total == state_total
        assert len(db_pc_sampling) == state_count
        assert db_pc_sampling["count"].sum() == state_total
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
        assert db_dispatch_count == capture_expectations["dispatch_count"] == 3
        assert db_code_object_process_ids == [(capture_expectations["pid"],)]
        assert set(db_pc_sampling["kernel_name"]) == {
            "vecCopy(double*, double*, double*, int, int)"
        }
        assert (
            dict(zip(db_pc_sampling["offset"], db_pc_sampling["count"]))
            == (capture_expectations["sample_counts"])
        )
        assert (
            (db_pc_sampling["issue_count"] + db_pc_sampling["stall_count"])
            .eq(db_pc_sampling["count"])
            .all()
        )
        # Canonical paths and frame order must survive database reconstruction.
        assert (
            dict(zip(db_pc_sampling["offset"], db_pc_sampling["source"]))
            == (capture_expectations["sources"])
        )
        assert db_source_line_counts == capture_expectations["source_line_counts"]
        assert db_referenced_source_lines == capture_expectations["referenced_lines"]
    finally:
        common.clean_output_dir(True, str(workload_dir))


def test_pc_sampling_analyze_csv_output(
    capture_expectations,
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
        assert len(csv_pc_sampling) == len(capture_expectations["sample_counts"])
        assert csv_pc_sampling["count"].sum() == capture_expectations["sample_total"]
        assert set(csv_pc_sampling["pid"]) == {capture_expectations["pid"]}
        assert csv_pc_sampling["active_thread_percent"].eq(100.0).all()
        assert csv_pc_sampling["wave_occupancy_percent"].gt(0).all()
        assert csv_pc_sampling["wave_occupancy_percent"].le(100).all()
        assert csv_kernel.iloc[0]["dispatch_count"] == 3
        csv_source_lines = pd.read_csv(csv_dir / "source_lines.csv")
        assert (
            set(csv_source_lines["file_path"]) == capture_expectations["source_paths"]
        )
        # Every sampling row must resolve to a kernel the kernel view exposes.
        assert set(csv_pc_sampling["kernel_uuid"]) <= set(csv_kernel["kernel_uuid"])
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
