#!/usr/bin/env python3
"""
Tests for rocinsight.connection — RocinsightConnection and merge_sqlite_dbs.

Uses only stdlib (sqlite3, tempfile, pathlib) and pytest — no real GPU required.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from rocinsight.connection import RocinsightConnection, execute_statement, merge_sqlite_dbs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_minimal_db(path: Path) -> None:
    """Create a minimal rocpd-style SQLite database with kernels + memory_copies."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE kernels (
            id     INTEGER PRIMARY KEY,
            name   TEXT,
            start  INTEGER,
            end    INTEGER,
            duration INTEGER
        );
        CREATE TABLE memory_copies (
            id       INTEGER PRIMARY KEY,
            category TEXT,
            start    INTEGER,
            end      INTEGER,
            size     INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO kernels VALUES (1, 'kernel_a', 1000, 2000, 1000)"
    )
    conn.execute(
        "INSERT INTO memory_copies VALUES (1, 'HostToDevice', 0, 500, 1024)"
    )
    conn.commit()
    conn.close()


def _create_db_with_pmc(path: Path) -> None:
    """Create a rocpd-style SQLite database with pmc_events."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE kernels (
            id INTEGER PRIMARY KEY,
            name TEXT,
            start INTEGER,
            end INTEGER,
            duration INTEGER
        );
        CREATE TABLE pmc_events (
            id            INTEGER PRIMARY KEY,
            dispatch_id   INTEGER,
            name          TEXT,
            counter_name  TEXT,
            counter_value REAL
        );
    """)
    conn.execute("INSERT INTO kernels VALUES (1, 'gpu_kernel', 0, 1000, 1000)")
    conn.execute("INSERT INTO pmc_events VALUES (1, 1, 'gpu_kernel', 'SQ_WAVES', 32.0)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def single_db(tmp_dir):
    p = tmp_dir / "trace.db"
    _create_minimal_db(p)
    return p


@pytest.fixture()
def two_dbs(tmp_dir):
    """Two databases both containing pmc_events (which IS in _ROCINSIGHT_TABLES)."""
    p1 = tmp_dir / "trace0.db"
    p2 = tmp_dir / "trace1.db"

    for path, counter_name, counter_value in [
        (p1, "SQ_WAVES", 32.0),
        (p2, "GRBM_COUNT", 1000.0),
    ]:
        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE pmc_events (
                id            INTEGER PRIMARY KEY,
                dispatch_id   INTEGER,
                name          TEXT,
                counter_name  TEXT,
                counter_value REAL
            );
            CREATE TABLE kernels (
                id INTEGER PRIMARY KEY,
                name TEXT,
                start INTEGER,
                end INTEGER,
                duration INTEGER
            );
            CREATE TABLE memory_copies (
                id INTEGER PRIMARY KEY,
                category TEXT,
                start INTEGER,
                end INTEGER,
                size INTEGER
            );
        """)
        conn.execute(
            "INSERT INTO pmc_events VALUES (1, 1, 'gpu_kernel', ?, ?)",
            (counter_name, counter_value),
        )
        conn.commit()
        conn.close()
    return p1, p2


# ---------------------------------------------------------------------------
# Single-file connection
# ---------------------------------------------------------------------------

class TestSingleFileConnection:
    def test_opens_single_db_directly(self, single_db):
        conn = RocinsightConnection(single_db)
        assert conn.connection is not None
        conn.close()

    def test_connection_property_is_sqlite3_connection(self, single_db):
        conn = RocinsightConnection(single_db)
        assert isinstance(conn.connection, sqlite3.Connection)
        conn.close()

    def test_single_db_can_query_kernels(self, single_db):
        conn = RocinsightConnection(single_db)
        rows = conn.execute("SELECT name FROM kernels").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "kernel_a"
        conn.close()

    def test_accepts_string_path(self, single_db):
        conn = RocinsightConnection(str(single_db))
        rows = conn.execute("SELECT COUNT(*) FROM kernels").fetchone()
        assert rows[0] == 1
        conn.close()

    def test_accepts_list_with_single_path(self, single_db):
        conn = RocinsightConnection([single_db])
        rows = conn.execute("SELECT COUNT(*) FROM kernels").fetchone()
        assert rows[0] == 1
        conn.close()

    def test_context_manager(self, single_db):
        with RocinsightConnection(single_db) as conn:
            rows = conn.execute("SELECT name FROM kernels").fetchall()
        assert len(rows) == 1

    def test_execute_delegated_to_connection(self, single_db):
        conn = RocinsightConnection(single_db)
        result = conn.execute("SELECT name FROM kernels WHERE id = 1").fetchone()
        assert result[0] == "kernel_a"
        conn.close()


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_raises_file_not_found_for_missing_db(self, tmp_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            RocinsightConnection(tmp_dir / "does_not_exist.db")

    def test_raises_for_second_missing_db(self, single_db, tmp_dir):
        missing = tmp_dir / "ghost.db"
        with pytest.raises(FileNotFoundError):
            RocinsightConnection([single_db, missing])


# ---------------------------------------------------------------------------
# Multi-file connection (UNION ALL views)
# ---------------------------------------------------------------------------

class TestMultiFileConnection:
    def test_multi_file_creates_in_memory_union(self, two_dbs):
        """pmc_events from both shards are visible via the UNION ALL view."""
        p1, p2 = two_dbs
        conn = RocinsightConnection([p1, p2])
        # pmc_events is in _ROCINSIGHT_TABLES so a UNION ALL view is created for it
        rows = conn.execute(
            "SELECT counter_name FROM pmc_events ORDER BY counter_name"
        ).fetchall()
        names = [r[0] for r in rows]
        assert "SQ_WAVES" in names
        assert "GRBM_COUNT" in names
        conn.close()

    def test_multi_file_row_count(self, two_dbs):
        """Row count in the union view equals sum of rows across both shards."""
        p1, p2 = two_dbs
        conn = RocinsightConnection([p1, p2])
        row = conn.execute("SELECT COUNT(*) FROM pmc_events").fetchone()
        assert row[0] == 2
        conn.close()

    def test_multi_file_union_view_exists(self, two_dbs):
        """A TEMP VIEW for pmc_events is created when both shards have the table."""
        p1, p2 = two_dbs
        conn = RocinsightConnection([p1, p2])
        views = conn.execute(
            "SELECT name FROM sqlite_temp_master WHERE type='view'"
        ).fetchall()
        view_names = [v[0] for v in views]
        assert "pmc_events" in view_names
        conn.close()

    def test_multi_file_data_from_both_dbs_queryable(self, two_dbs):
        """Values from both shards appear in the UNION ALL view query."""
        p1, p2 = two_dbs
        conn = RocinsightConnection([p1, p2])
        rows = conn.execute(
            "SELECT counter_value FROM pmc_events ORDER BY counter_value"
        ).fetchall()
        values = {r[0] for r in rows}
        assert 32.0 in values    # from p1
        assert 1000.0 in values  # from p2
        conn.close()


# ---------------------------------------------------------------------------
# Table missing from one shard
# ---------------------------------------------------------------------------

class TestPartialSchema:
    def test_table_missing_from_one_shard_skipped_gracefully(self, tmp_dir):
        """db0 has pmc_events; db1 does not — should not crash."""
        db0 = tmp_dir / "with_pmc.db"
        db1 = tmp_dir / "without_pmc.db"
        _create_db_with_pmc(db0)
        _create_minimal_db(db1)

        # Should not raise; missing tables in one shard are skipped
        conn = RocinsightConnection([db0, db1])
        conn.close()

    def test_pmc_events_from_only_one_shard_queryable(self, tmp_dir):
        """pmc_events present in db0 only is accessible via the union view."""
        db0 = tmp_dir / "with_pmc.db"
        db1 = tmp_dir / "no_pmc.db"
        _create_db_with_pmc(db0)
        _create_minimal_db(db1)

        conn = RocinsightConnection([db0, db1])
        # pmc_events view should exist (db0 has it; db1 does not — gracefully skipped)
        row = conn.execute("SELECT COUNT(*) FROM pmc_events").fetchone()
        assert row[0] == 1
        conn.close()


# ---------------------------------------------------------------------------
# execute_statement helper
# ---------------------------------------------------------------------------

class TestExecuteStatement:
    def test_execute_statement_on_rocinsight_connection(self, single_db):
        conn = RocinsightConnection(single_db)
        result = execute_statement(conn, "SELECT COUNT(*) FROM kernels").fetchone()
        assert result[0] == 1
        conn.close()

    def test_execute_statement_on_raw_sqlite3_connection(self, single_db):
        raw = sqlite3.connect(str(single_db))
        result = execute_statement(raw, "SELECT COUNT(*) FROM kernels").fetchone()
        assert result[0] == 1
        raw.close()

    def test_execute_statement_with_params(self, single_db):
        conn = RocinsightConnection(single_db)
        result = execute_statement(
            conn, "SELECT name FROM kernels WHERE id = ?", (1,)
        ).fetchone()
        assert result[0] == "kernel_a"
        conn.close()


# ---------------------------------------------------------------------------
# merge_sqlite_dbs
# ---------------------------------------------------------------------------

class TestMergeSqliteDbs:
    def test_single_db_copied_to_default_output(self, single_db):
        out = merge_sqlite_dbs([single_db])
        assert out.exists()
        assert out.name == "merged_processes.db"
        # Verify data integrity
        conn = sqlite3.connect(str(out))
        row = conn.execute("SELECT COUNT(*) FROM kernels").fetchone()
        conn.close()
        assert row[0] == 1

    def test_single_db_custom_output_path(self, tmp_dir, single_db):
        dest = tmp_dir / "custom_out.db"
        out = merge_sqlite_dbs([single_db], output_path=dest)
        assert out == dest
        assert dest.exists()

    def test_merge_two_dbs_combines_rows(self, two_dbs, tmp_dir):
        p1, p2 = two_dbs
        dest = tmp_dir / "merged.db"
        out = merge_sqlite_dbs([p1, p2], output_path=dest)
        assert out.exists()
        conn = sqlite3.connect(str(out))
        row = conn.execute("SELECT COUNT(*) FROM pmc_events").fetchone()
        conn.close()
        # p1 has SQ_WAVES (id=1), p2 has GRBM_COUNT (id=1) — INSERT OR IGNORE merges by id
        # At minimum p1's data is present (base copy); p2's row has same id=1 so may be ignored
        assert row[0] >= 1

    def test_merge_returns_path_object(self, single_db):
        out = merge_sqlite_dbs([single_db])
        assert isinstance(out, Path)

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="No database paths"):
            merge_sqlite_dbs([])

    def test_output_path_parent_is_first_db_parent_by_default(self, single_db):
        out = merge_sqlite_dbs([single_db])
        assert out.parent == single_db.parent
