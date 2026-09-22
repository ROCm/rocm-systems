# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for native_data module.

Covers building the analyze counter frame from the native tool's per-pid
artifacts: discovering the files, joining counters to dispatches and dispatches
to kernel symbols, summing counter instances, and reassigning dispatch and
kernel ids per counter set.
"""

import csv
from pathlib import Path

import pytest

from utils import csv_compression, native_data

COUNTER_COLUMNS = [
    "dispatch_id",
    "gpu_id",
    "kernel_id",
    "lds_per_workgroup",
    "counter_id",
    "counter_name",
    "counter_value",
]
DISPATCH_COLUMNS = [
    "dispatch_id",
    "gpu_id",
    "kernel_id",
    "grid_size",
    "workgroup_size",
    "lds_per_workgroup",
    "scratch_per_workitem",
    "start_timestamp",
    "end_timestamp",
    "correlation_id",
]
SYMBOL_COLUMNS = [
    "kernel_id",
    "kernel_name",
    "kernel_short_name",
    "arch_vgpr",
    "accum_vgpr",
    "sgpr",
]


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with csv_compression.open_gzip_csv_write(path) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def counter_row(dispatch_id, counter_name, counter_value, kernel_id=7):
    return {
        "dispatch_id": dispatch_id,
        "gpu_id": 2,
        "kernel_id": kernel_id,
        "lds_per_workgroup": 0,
        "counter_id": 5,
        "counter_name": counter_name,
        "counter_value": counter_value,
    }


def dispatch_row(dispatch_id, kernel_id=7, start=1000, end=2000, correlation_id=None):
    # A dispatch carries its own correlation id, which is not its dispatch id.
    # They are only equal by accident, so the default keeps them apart.
    return {
        "dispatch_id": dispatch_id,
        "gpu_id": 2,
        "kernel_id": kernel_id,
        "grid_size": 1048576,
        "workgroup_size": 256,
        "lds_per_workgroup": 0,
        "scratch_per_workitem": 0,
        "start_timestamp": start,
        "end_timestamp": end,
        "correlation_id": dispatch_id + 500
        if correlation_id is None
        else correlation_id,
    }


def symbol_row(kernel_id=7, kernel_name="vecCopy(double*, int)"):
    return {
        "kernel_id": kernel_id,
        "kernel_name": kernel_name,
        "kernel_short_name": "vecCopy",
        "arch_vgpr": 4,
        "accum_vgpr": 4,
        "sgpr": 16,
    }


def write_process(
    workload_dir: Path,
    fbase: str,
    pid: int,
    counters: list[dict],
    dispatches: list[dict],
    symbols: list[dict],
) -> None:
    """Write one process's three artifacts into a workload directory."""
    for prefix, columns, rows in (
        (native_data.COUNTERS_PREFIX, COUNTER_COLUMNS, counters),
        (native_data.DISPATCH_PREFIX, DISPATCH_COLUMNS, dispatches),
        (native_data.KERNEL_SYMBOLS_PREFIX, SYMBOL_COLUMNS, symbols),
    ):
        write_csv(
            csv_compression.compressed_name(
                workload_dir / f"{prefix}_{fbase}_{pid}.csv"
            ),
            columns,
            rows,
        )


def read_rows(path: Path) -> list[dict]:
    with csv_compression.open_gzip_csv_read(path) as infile:
        return list(csv.DictReader(infile))


@pytest.fixture
def workload_dir(tmp_path):
    return tmp_path


# =============================================================================
# Artifact discovery
# =============================================================================


def test_find_native_artifacts_groups_by_counter_set_and_pid(workload_dir):
    write_process(
        workload_dir,
        "pmc_perf_0",
        200,
        [counter_row(1, "SQ_WAVES", 1)],
        [dispatch_row(1)],
        [symbol_row()],
    )
    write_process(
        workload_dir,
        "pmc_perf_0",
        100,
        [counter_row(1, "SQ_WAVES", 1)],
        [dispatch_row(1)],
        [symbol_row()],
    )
    write_process(
        workload_dir,
        "pmc_perf_1",
        300,
        [counter_row(1, "SQ_WAVES", 1)],
        [dispatch_row(1)],
        [symbol_row()],
    )

    artifacts = native_data.find_native_artifacts(workload_dir)

    assert [(a.fbase, a.pid) for a in artifacts] == [
        ("pmc_perf_0", 100),
        ("pmc_perf_0", 200),
        ("pmc_perf_1", 300),
    ]


def test_find_native_artifacts_ignores_other_csvs(workload_dir):
    write_csv(
        csv_compression.compressed_name(workload_dir / "results_pmc_perf_0.csv"),
        ["Counter_Name"],
        [],
    )
    write_csv(
        csv_compression.compressed_name(workload_dir / "kernel_symbols_pmc_perf_0.csv"),
        ["Kernel_Name"],
        [],
    )

    assert native_data.find_native_artifacts(workload_dir) == []


def test_find_native_artifacts_skips_an_incomplete_set(workload_dir, monkeypatch):
    warnings = []
    monkeypatch.setattr(native_data, "console_warning", warnings.append)
    write_csv(
        csv_compression.compressed_name(workload_dir / "counters_pmc_perf_0_100.csv"),
        COUNTER_COLUMNS,
        [counter_row(1, "SQ_WAVES", 1)],
    )

    assert native_data.find_native_artifacts(workload_dir) == []
    assert len(warnings) == 1
    assert "dispatch" in warnings[0]


# =============================================================================
# Joining
# =============================================================================


def test_counter_instances_are_summed_per_dispatch_and_counter(workload_dir):
    write_process(
        workload_dir,
        "pmc_perf_0",
        100,
        [
            counter_row(1, "SQ_WAVES", 10),
            counter_row(1, "SQ_WAVES", 32),
            counter_row(1, "SQ_BUSY", 5),
        ],
        [dispatch_row(1)],
        [symbol_row()],
    )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    assert native_data.write_counter_frame(workload_dir, output) == 2

    rows = read_rows(output)
    values = {row["Counter_Name"]: float(row["Counter_Value"]) for row in rows}
    assert values == {"SQ_WAVES": 42.0, "SQ_BUSY": 5.0}


def test_dispatch_and_symbol_columns_reach_the_output(workload_dir):
    write_process(
        workload_dir,
        "pmc_perf_0",
        100,
        [counter_row(1, "SQ_WAVES", 10)],
        [dispatch_row(1)],
        [symbol_row()],
    )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    native_data.write_counter_frame(workload_dir, output)

    row = read_rows(output)[0]
    # The correlation id is the dispatch's own, not the reassigned dispatch id.
    assert row["Correlation_Id"] == "501"
    assert row["Dispatch_ID"] == "1"
    assert row["Grid_Size"] == "1048576"
    assert row["Workgroup_Size"] == "256"
    assert row["Kernel_Name"] == "vecCopy(double*, int)"
    assert row["Arch_VGPR"] == "4"
    assert row["SGPR"] == "16"
    assert row["Start_Timestamp"] == "1000"
    # The pid stands in for the per-process identifier the rocpd path calls GUID.
    assert row["GUID"] == "100"
    # PID only groups dispatches; it is not part of the analyze frame.
    assert "PID" not in row


def test_counter_without_a_dispatch_row_is_dropped(workload_dir):
    write_process(
        workload_dir,
        "pmc_perf_0",
        100,
        [counter_row(1, "SQ_WAVES", 10), counter_row(2, "SQ_WAVES", 20)],
        [dispatch_row(1)],
        [symbol_row()],
    )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    assert native_data.write_counter_frame(workload_dir, output) == 1


def test_dispatch_without_a_kernel_symbol_is_dropped(workload_dir):
    write_process(
        workload_dir,
        "pmc_perf_0",
        100,
        [counter_row(1, "SQ_WAVES", 10, kernel_id=9)],
        [dispatch_row(1, kernel_id=9)],
        [symbol_row(kernel_id=7)],
    )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    assert native_data.write_counter_frame(workload_dir, output) == 0


# =============================================================================
# Id reassignment
# =============================================================================


def test_dispatch_ids_are_unique_across_processes(workload_dir):
    # Both processes number their own dispatches from 1.
    for pid in (100, 200):
        write_process(
            workload_dir,
            "pmc_perf_0",
            pid,
            [counter_row(1, "SQ_WAVES", 10)],
            [dispatch_row(1, start=1000 + pid)],
            [symbol_row()],
        )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    native_data.write_counter_frame(workload_dir, output)

    rows = read_rows(output)
    assert [row["Dispatch_ID"] for row in rows] == ["1", "2"]
    # Same kernel in both processes, so one kernel id.
    assert {row["Kernel_ID"] for row in rows} == {"0"}


def test_dispatch_ids_restart_for_each_counter_set(workload_dir):
    # Separate runs of the same application, so the same dispatch gets the same
    # id in both sets and the sets can be lined up afterwards.
    for fbase, start in (("pmc_perf_0", 1000), ("pmc_perf_1", 5000)):
        write_process(
            workload_dir,
            fbase,
            100,
            [counter_row(1, "SQ_WAVES", 10)],
            [dispatch_row(1, start=start)],
            [symbol_row()],
        )
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    native_data.write_counter_frame(workload_dir, output)

    assert [row["Dispatch_ID"] for row in read_rows(output)] == ["1", "1"]


def test_no_artifacts_writes_nothing(workload_dir):
    output = csv_compression.compressed_name(workload_dir / "pmc_perf.csv")

    assert native_data.write_counter_frame(workload_dir, output) == 0
    assert not output.exists()
