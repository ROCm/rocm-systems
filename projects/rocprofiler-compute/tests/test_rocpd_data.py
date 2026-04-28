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
from contextlib import closing
from pathlib import Path

import pytest

from utils.rocpd_data import convert_dbs_to_csvs, update_rocpd_pmc_events

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

    convert_dbs_to_csvs(
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

    convert_dbs_to_csvs(
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

    convert_dbs_to_csvs(
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

    total = convert_dbs_to_csvs(
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

    convert_dbs_to_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    counter_header, counter_data = _read_csv(output_paths["counter"])
    results_header, results_data = _read_csv(output_paths["results"])
    assert counter_header == results_header
    assert counter_data == results_data


def test_header_written_for_empty_db(tmp_path, output_paths):
    """An empty DB still writes the header (cursor.description is set)."""
    empty_db = tmp_path / "empty.db"
    _create_db(empty_db, [], [])

    total = convert_dbs_to_csvs(
        [str(empty_db)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    assert total == 0
    header, data = _read_csv(output_paths["counter"])
    assert "Dispatch_ID" in header
    assert "PID" not in header
    assert data == []


def test_header_written_when_first_db_query_fails(tmp_path, output_paths, monkeypatch):
    """A failed first DB must not suppress the header for later DBs.

    Guards the cross-DB ``header_written`` tracking: the first DB's
    ``conn.execute`` raises (no ``counters_collection`` table), the
    except branch returns without writing a header, and a subsequent
    well-formed DB must still emit the header before its data rows.
    """
    failing_db = tmp_path / "failing.db"
    # Build a DB without a counters_collection table so the SELECT raises.
    conn = sqlite3.connect(str(failing_db))
    conn.execute(REGIONS_TABLE_DDL)
    conn.commit()
    conn.close()

    populated_db = tmp_path / "populated.db"
    _create_db(
        populated_db,
        [_build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20)],
        [],
    )

    errors: list[str] = []
    monkeypatch.setattr(
        "utils.rocpd_data.console_error", lambda msg: errors.append(msg)
    )

    total = convert_dbs_to_csvs(
        [str(failing_db), str(populated_db)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    assert total == 1
    header, data = _read_csv(output_paths["counter"])
    assert "Dispatch_ID" in header
    assert len(data) == 1
    # The first DB's failure was logged but did not abort the run.
    assert any("counters" in msg.lower() for msg in errors)


def test_no_dbs_produces_empty_outputs(tmp_path, output_paths):
    """Calling with no DBs produces empty (header-less) output files."""
    total = convert_dbs_to_csvs(
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

    convert_dbs_to_csvs(
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


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_convert_is_deterministic_across_runs(tmp_path):
    """The same input DB must produce byte-identical CSVs across runs.

    Guards the `COUNTERS_COLLECTION_QUERY` (no `ORDER BY`) against
    silently producing different first-seen ID assignments between runs.
    """
    db_path = tmp_path / "det.db"
    rows = [
        _build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20),
        _build_counter_row(pid=100, kernel_name="kernel_b", start=30, end=40),
        _build_counter_row(pid=200, kernel_name="kernel_a", start=50, end=60),
        _build_counter_row(
            pid=200,
            kernel_name="kernel_a",
            start=50,
            end=60,
            counter_name="GRBM_GUI_ACTIVE",
            value=2.0,
        ),
    ]
    _create_db(db_path, rows, [])

    def _run(suffix: str) -> tuple[str, str]:
        counter = tmp_path / f"counter_{suffix}.csv"
        marker = tmp_path / f"marker_{suffix}.csv"
        results = tmp_path / f"results_{suffix}.csv"
        convert_dbs_to_csvs([str(db_path)], str(counter), str(marker), str(results))
        return counter.read_text(), results.read_text()

    counter_a, results_a = _run("a")
    counter_b, results_b = _run("b")
    assert counter_a == counter_b
    assert results_a == results_b


# ---------------------------------------------------------------------------
# Marker trace
# ---------------------------------------------------------------------------


def _build_marker_row(
    *,
    pid: int,
    tid: int,
    function: str,
    start: int,
    end: int,
    category: str = "marker_api",
    stack_id: int = 0,
    guid: str = "abc-1234",
) -> tuple:
    """Build a regions row matching the rocpd schema (extdata is JSON)."""
    return (
        category,
        f'{{"message": "{function}"}}',
        pid,
        tid,
        stack_id,
        guid,
        start,
        end,
    )


def test_marker_trace_header_and_rows_written(tmp_path, output_paths):
    """Marker CSV gets the expected header and one data row per region."""
    db_path = tmp_path / "marker.db"
    region_rows = [
        _build_marker_row(pid=100, tid=101, function="hipMalloc", start=5, end=8),
        _build_marker_row(pid=100, tid=101, function="hipFree", start=12, end=14),
    ]
    _create_db(db_path, [], region_rows)

    convert_dbs_to_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, data = _read_csv(output_paths["marker"])
    assert header == [
        "Domain",
        "Function",
        "Process_Id",
        "Thread_Id",
        "Correlation_Id",
        "GUID",
        "Start_Timestamp",
        "End_Timestamp",
    ]
    function_col = header.index("Function")
    assert [row[function_col] for row in data] == ["hipMalloc", "hipFree"]


def test_marker_trace_ordered_by_start_within_db(tmp_path, output_paths):
    """`MARKER_API_TRACE_QUERY` has ORDER BY start; output must be time-ordered."""
    db_path = tmp_path / "ordered.db"
    region_rows = [
        _build_marker_row(pid=100, tid=101, function="late", start=100, end=110),
        _build_marker_row(pid=100, tid=101, function="early", start=10, end=20),
        _build_marker_row(pid=100, tid=101, function="middle", start=50, end=60),
    ]
    _create_db(db_path, [], region_rows)

    convert_dbs_to_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    header, data = _read_csv(output_paths["marker"])
    function_col = header.index("Function")
    assert [row[function_col] for row in data] == ["early", "middle", "late"]


def test_marker_trace_header_written_once_across_dbs(tmp_path, output_paths):
    """Header is emitted exactly once when streaming multiple DBs."""
    db1 = tmp_path / "m1.db"
    db2 = tmp_path / "m2.db"
    _create_db(
        db1, [], [_build_marker_row(pid=100, tid=1, function="a", start=1, end=2)]
    )
    _create_db(
        db2, [], [_build_marker_row(pid=200, tid=2, function="b", start=3, end=4)]
    )

    convert_dbs_to_csvs(
        [str(db1), str(db2)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    with open(output_paths["marker"], newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3
    assert rows[0][0] == "Domain"
    assert rows[1][0] != "Domain"
    assert rows[2][0] != "Domain"


def test_marker_trace_failure_does_not_abort_counter_export(
    tmp_path, output_paths, monkeypatch
):
    """A DB missing the regions table is logged and counter export proceeds."""
    db_path = tmp_path / "no_regions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(COUNTERS_TABLE_DDL)
    conn.executemany(
        COUNTERS_INSERT,
        [_build_counter_row(pid=100, kernel_name="kernel_a", start=10, end=20)],
    )
    conn.commit()
    conn.close()

    errors: list[str] = []
    monkeypatch.setattr(
        "utils.rocpd_data.console_error", lambda msg: errors.append(msg)
    )

    total = convert_dbs_to_csvs(
        [str(db_path)],
        str(output_paths["counter"]),
        str(output_paths["marker"]),
        str(output_paths["results"]),
    )

    assert total == 1
    counter_header, counter_data = _read_csv(output_paths["counter"])
    assert "Dispatch_ID" in counter_header
    assert len(counter_data) == 1
    # Marker file is empty (no header) because the query failed before headers.
    assert output_paths["marker"].read_text() == ""
    assert any("marker trace" in msg for msg in errors)


# ---------------------------------------------------------------------------
# update_rocpd_pmc_events
# ---------------------------------------------------------------------------

PMC_EVENT_TABLE_DDL = (
    "CREATE TABLE {name} (guid TEXT, event_id INTEGER, pmc_id INTEGER, value REAL)"
)
KERNEL_DISPATCH_TABLE_DDL = (
    "CREATE TABLE rocpd_kernel_dispatch ("
    "dispatch_id INTEGER, event_id INTEGER, guid TEXT"
    ")"
)


def _build_pmc_event_db(
    db_path: Path,
    *,
    pmc_event_table: str | None = "rocpd_pmc_event_abc_1234",
    kernel_dispatch_rows: list[tuple] | None = None,
) -> None:
    """Build a minimal rocpd DB exercising the pmc_event update path."""
    conn = sqlite3.connect(str(db_path))
    if pmc_event_table is not None:
        conn.execute(PMC_EVENT_TABLE_DDL.format(name=pmc_event_table))
    conn.execute(KERNEL_DISPATCH_TABLE_DDL)
    if kernel_dispatch_rows:
        conn.executemany(
            "INSERT INTO rocpd_kernel_dispatch VALUES (?,?,?)",
            kernel_dispatch_rows,
        )
    conn.commit()
    conn.close()


def test_update_rocpd_pmc_events_inserts_rows(tmp_path):
    """Maps dispatch_id → event_id and inserts into the rocpd_pmc_event table."""
    db_path = tmp_path / "pmc.db"
    # Table name "rocpd_pmc_event_abc_1234" → guid "abc-1234" (underscores
    # in the suffix become dashes).
    _build_pmc_event_db(
        db_path,
        kernel_dispatch_rows=[
            (10, 100, "abc-1234"),
            (11, 101, "abc-1234"),
        ],
    )

    counter_info = [
        {"dispatch_id": "10", "counter_id": 1, "counter_value": 1.5},
        {"dispatch_id": "11", "counter_id": 2, "counter_value": 2.5},
    ]
    update_rocpd_pmc_events(counter_info, str(db_path))

    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = list(
            conn.execute(
                "SELECT guid, event_id, pmc_id, value "
                "FROM rocpd_pmc_event_abc_1234 ORDER BY pmc_id"
            )
        )
    assert rows == [
        ("abc-1234", 100, 1, 1.5),
        ("abc-1234", 101, 2, 2.5),
    ]


def test_update_rocpd_pmc_events_empty_dispatch_table_logs_and_returns(
    tmp_path, monkeypatch
):
    """No kernel dispatch rows → console_error logged and call returns early."""
    db_path = tmp_path / "pmc_empty.db"
    _build_pmc_event_db(db_path, kernel_dispatch_rows=[])

    errors: list[str] = []
    monkeypatch.setattr(
        "utils.rocpd_data.console_error", lambda msg: errors.append(msg)
    )

    update_rocpd_pmc_events(
        [{"dispatch_id": "10", "counter_id": 1, "counter_value": 1.5}],
        str(db_path),
    )

    assert any("kernel dispatch" in msg.lower() for msg in errors)
    # Nothing was inserted.
    with closing(sqlite3.connect(str(db_path))) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM rocpd_pmc_event_abc_1234"
        ).fetchone()[0]
    assert count == 0


def test_update_rocpd_pmc_events_invalid_db_logs_error(tmp_path, monkeypatch):
    """A non-sqlite file is caught by the broad except and surfaced."""
    bad_db = tmp_path / "not_a_db.txt"
    bad_db.write_text("not a sqlite database")

    errors: list[str] = []
    monkeypatch.setattr(
        "utils.rocpd_data.console_error", lambda msg: errors.append(msg)
    )

    update_rocpd_pmc_events([], str(bad_db))

    assert errors
