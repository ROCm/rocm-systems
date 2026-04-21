# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for the rocpd_data module's CSV export.

Covers ID assignment semantics, multi-DB ID continuity, header
behavior across empty and populated DBs, and the PID drop in the
output schema.
"""

import csv
import sqlite3
from pathlib import Path

import pytest

from utils.rocpd_data import export_rocpd_csvs

COUNTERS_TABLE_DDL = """
CREATE TABLE counters_collection (
    agent_id INTEGER, guid TEXT, stack_id INTEGER, dispatch_id INTEGER,
    pid INTEGER, grid_size INTEGER, workgroup_size INTEGER,
    lds_block_size INTEGER, scratch_size INTEGER, vgpr_count INTEGER,
    accum_vgpr_count INTEGER, sgpr_count INTEGER, kernel_name TEXT,
    start INTEGER, end INTEGER, kernel_id INTEGER,
    counter_name TEXT, value REAL
)
"""

REGIONS_TABLE_DDL = """
CREATE TABLE regions (
    category TEXT, extdata TEXT, pid INTEGER, tid INTEGER,
    stack_id INTEGER, guid TEXT, start INTEGER, end INTEGER
)
"""

COUNTERS_INSERT = (
    "INSERT INTO counters_collection VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

REGIONS_INSERT = "INSERT INTO regions VALUES (?,?,?,?,?,?,?,?)"


def _build_counter_row(
    *,
    pid: int,
    kernel_name: str,
    start: int,
    end: int,
    counter_name: str = "SQ_WAVES",
    value: float = 1.0,
    grid_size: int = 64,
    workgroup_size: int = 256,
    lds: int = 0,
    dispatch_id: int = 0,
    kernel_id: int = 0,
    stack_id: int = 0,
) -> tuple:
    """Build a counters_collection row matching the production schema."""
    return (
        0,  # agent_id (GPU_ID)
        "abc-1234",  # guid
        stack_id,
        dispatch_id,
        pid,
        grid_size,
        workgroup_size,
        lds,
        0,  # scratch_size
        32,  # vgpr_count
        0,  # accum_vgpr_count
        16,  # sgpr_count
        kernel_name,
        start,
        end,
        kernel_id,
        counter_name,
        value,
    )


def _create_db(db_path: Path, counter_rows: list, region_rows: list) -> None:
    """Create a minimal rocpd-style SQLite DB with the given rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(COUNTERS_TABLE_DDL)
    if counter_rows:
        conn.executemany(COUNTERS_INSERT, counter_rows)
    conn.execute(REGIONS_TABLE_DDL)
    if region_rows:
        conn.executemany(REGIONS_INSERT, region_rows)
    conn.commit()
    conn.close()


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header, data_rows) from a CSV file."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


@pytest.fixture
def output_paths(tmp_path: Path) -> dict[str, Path]:
    """Provide standard output CSV paths under a tmp dir."""
    return {
        "counter": tmp_path / "counter_collection.csv",
        "marker": tmp_path / "marker_api_trace.csv",
        "results": tmp_path / "results.csv",
    }


def test_pid_dropped_from_counter_header(tmp_path, output_paths):
    """PID is used for grouping but must not appear in the output CSV."""
    db_path = tmp_path / "single.db"
    rows = [
        _build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20),
    ]
    _create_db(db_path, rows, [])

    export_rocpd_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, _ = _read_csv(output_paths["counter"])
    assert "PID" not in header
    assert "Dispatch_ID" in header
    assert "Kernel_ID" in header


def test_dispatch_and_kernel_ids_assigned_sequentially(tmp_path, output_paths):
    """First-seen dispatch/kernel signatures are numbered from zero."""
    db_path = tmp_path / "ids.db"
    rows = [
        _build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20),
        _build_counter_row(pid=100, kernel_name="kernel_b", start=30, end=40),
        _build_counter_row(pid=100, kernel_name="kernel_a", start=50, end=60),
    ]
    _create_db(db_path, rows, [])

    export_rocpd_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, data = _read_csv(output_paths["counter"])
    dispatch_col = header.index("Dispatch_ID")
    kernel_col = header.index("Kernel_ID")

    assert [row[dispatch_col] for row in data] == ["0", "1", "2"]
    # kernel_a (rows 0 and 2) shares Kernel_ID; kernel_b is distinct.
    assert [row[kernel_col] for row in data] == ["0", "1", "0"]


def test_kernel_ids_continuous_across_dbs(tmp_path, output_paths):
    """A kernel signature seen in DB1 keeps its ID when re-seen in DB2."""
    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"
    _create_db(
        db1,
        [_build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20)],
        [],
    )
    _create_db(
        db2,
        [
            _build_counter_row(pid=200, kernel_name="kernel_a", start=30, end=40),
            _build_counter_row(pid=200, kernel_name="kernel_b", start=50, end=60),
        ],
        [],
    )

    export_rocpd_csvs(
        [str(db1), str(db2)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, data = _read_csv(output_paths["counter"])
    kernel_col = header.index("Kernel_ID")
    dispatch_col = header.index("Dispatch_ID")

    # kernel_a appears in both DBs and must keep Kernel_ID 0; kernel_b is new.
    assert [row[kernel_col] for row in data] == ["0", "0", "1"]
    # Each row is a unique dispatch (different PID/timestamps).
    assert [row[dispatch_col] for row in data] == ["0", "1", "2"]


def test_returns_total_row_count(tmp_path, output_paths):
    """Return value is the total number of counter rows written."""
    db1 = tmp_path / "db1.db"
    db2 = tmp_path / "db2.db"
    _create_db(
        db1,
        [_build_counter_row(pid=100, kernel_name="k1", start=10, end=20)],
        [],
    )
    _create_db(
        db2,
        [
            _build_counter_row(pid=100, kernel_name="k2", start=30, end=40),
            _build_counter_row(pid=100, kernel_name="k3", start=50, end=60),
        ],
        [],
    )

    total = export_rocpd_csvs(
        [str(db1), str(db2)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )
    assert total == 3


def test_results_csv_matches_counter_csv(tmp_path, output_paths):
    """Both writers emit identical content (same header and data rows)."""
    db_path = tmp_path / "parity.db"
    rows = [
        _build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20),
        _build_counter_row(pid=100, kernel_name="kernel_b", start=30, end=40),
    ]
    _create_db(db_path, rows, [])

    export_rocpd_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    counter_header, counter_data = _read_csv(output_paths["counter"])
    results_header, results_data = _read_csv(output_paths["results"])
    assert counter_header == results_header
    assert counter_data == results_data


def test_header_written_when_first_db_is_empty(tmp_path, output_paths):
    """An empty first DB must not suppress the header for later DBs."""
    empty_db = tmp_path / "empty.db"
    populated_db = tmp_path / "populated.db"
    _create_db(empty_db, [], [])
    _create_db(
        populated_db,
        [_build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20)],
        [],
    )

    total = export_rocpd_csvs(
        [str(empty_db), str(populated_db)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    assert total == 1
    header, data = _read_csv(output_paths["counter"])
    # Header must be present even though the first DB had no rows.
    assert "Dispatch_ID" in header
    assert "PID" not in header
    assert len(data) == 1


def test_no_dbs_produces_empty_outputs(tmp_path, output_paths):
    """Calling with no DBs produces empty (header-less) output files."""
    total = export_rocpd_csvs(
        [],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    assert total == 0
    for path in output_paths.values():
        assert path.exists()
        assert path.read_text() == ""


def test_same_kernel_different_pids_yields_distinct_dispatches(tmp_path, output_paths):
    """Dispatch grouping uses PID + timestamps so duplicates split by PID."""
    db_path = tmp_path / "dispatch.db"
    rows = [
        _build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20),
        _build_counter_row(pid=200, kernel_name="kernel_a", start=10, end=20),
    ]
    _create_db(db_path, rows, [])

    export_rocpd_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, data = _read_csv(output_paths["counter"])
    dispatch_col = header.index("Dispatch_ID")
    kernel_col = header.index("Kernel_ID")

    # Same kernel signature → same Kernel_ID; different PIDs → distinct
    # Dispatch_IDs.
    assert [row[kernel_col] for row in data] == ["0", "0"]
    assert [row[dispatch_col] for row in data] == ["0", "1"]
