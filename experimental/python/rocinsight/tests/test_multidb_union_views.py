"""Tests for multi-database analysis view unioning across attached shards."""
import sqlite3
import pytest


def _create_test_db(path, kernel_name, kernel_duration):
    """Create a minimal DB with a kernels table (mimicking the rocpd view)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE kernels (name TEXT, start INTEGER, end INTEGER, duration INTEGER)"
    )
    conn.execute(
        "INSERT INTO kernels VALUES (?, 100, ?, ?)",
        (kernel_name, 100 + kernel_duration, kernel_duration),
    )
    conn.execute(
        "CREATE TABLE memory_copies "
        "(name TEXT, start INTEGER, end INTEGER, duration INTEGER, size INTEGER)"
    )
    conn.execute(
        "CREATE TABLE regions "
        "(name TEXT, category TEXT, start INTEGER, end INTEGER, duration INTEGER)"
    )
    conn.execute(
        "CREATE TABLE pmc_events "
        "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, counter_name TEXT, counter_value REAL)"
    )
    conn.commit()
    conn.close()


class TestMultiFileUnionViews:
    def test_kernels_visible_from_both_dbs(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 1000)
        _create_test_db(db1, "kernel_B", 2000)

        from rocinsight.connection import RocinsightConnection, execute_statement
        conn = RocinsightConnection([str(db0), str(db1)])
        rows = execute_statement(conn, "SELECT name FROM kernels ORDER BY name").fetchall()
        names = [r[0] for r in rows]
        assert "kernel_A" in names
        assert "kernel_B" in names
        assert len(rows) == 2

    def test_memory_copies_from_both_dbs(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "k", 100)
        _create_test_db(db1, "k", 100)
        for path, mc_name in [(db0, "h2d_0"), (db1, "h2d_1")]:
            c = sqlite3.connect(str(path))
            c.execute("INSERT INTO memory_copies VALUES (?, 0, 100, 100, 1024)", (mc_name,))
            c.commit()
            c.close()

        from rocinsight.connection import RocinsightConnection, execute_statement
        conn = RocinsightConnection([str(db0), str(db1)])
        rows = execute_statement(conn, "SELECT name FROM memory_copies").fetchall()
        assert len(rows) == 2

    def test_single_db_still_works(self, tmp_path):
        db0 = tmp_path / "single.db"
        _create_test_db(db0, "kernel_only", 500)
        from rocinsight.connection import RocinsightConnection, execute_statement
        conn = RocinsightConnection([str(db0)])
        rows = execute_statement(conn, "SELECT name FROM kernels").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "kernel_only"
