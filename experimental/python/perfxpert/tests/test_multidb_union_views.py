#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################

"""Tests for multi-database analysis view unioning across ATTACHed shards.

Ports rocm-systems#4982. Real rocpd DB files store `kernels`, `memory_copies`,
`regions` as SQL VIEWs — not base tables. When PerfxpertConnection ATTACHes
multiple shards, analysis code does `SELECT * FROM kernels` against the
unified connection and expects to see rows from ALL shards.  Before the
fix the UNION ALL logic only handled base tables in ``_PERFXPERT_TABLES``,
so queries against the analysis views silently returned rows from the
first shard only.
"""

import sqlite3


def _create_test_db(path, kernel_name, kernel_duration):
    """Create a minimal DB with real analysis VIEWs over backing tables."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE kernels_data (name TEXT, start INTEGER, end INTEGER, "
        "duration INTEGER)"
    )
    conn.execute(
        "INSERT INTO kernels_data VALUES (?, 100, ?, ?)",
        (kernel_name, 100 + kernel_duration, kernel_duration),
    )
    conn.execute(
        "CREATE VIEW kernels AS "
        "SELECT name, start, end, duration FROM kernels_data"
    )
    conn.execute(
        "CREATE TABLE memory_copies_data "
        "(name TEXT, start INTEGER, end INTEGER, duration INTEGER, "
        "size INTEGER)"
    )
    conn.execute(
        "CREATE VIEW memory_copies AS "
        "SELECT name, start, end, duration, size FROM memory_copies_data"
    )
    conn.execute(
        "CREATE TABLE regions_data "
        "(name TEXT, category TEXT, start INTEGER, end INTEGER, "
        "duration INTEGER)"
    )
    conn.execute(
        "CREATE VIEW regions AS "
        "SELECT name, category, start, end, duration FROM regions_data"
    )
    conn.execute(
        "CREATE TABLE pmc_events "
        "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, "
        "counter_name TEXT, counter_value REAL)"
    )
    conn.commit()
    conn.close()


class TestMultiFileUnionViews:
    def test_time_breakdown_percentages_do_not_exceed_100_for_multidb_overlap(
        self, tmp_path
    ):
        """Multi-DB breakdown must preserve wall-clock runtime and expose a
        separate normalization runtime for overlap-safe percentages."""
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 80)
        _create_test_db(db1, "kernel_B", 80)

        from perfxpert.analyze import compute_time_breakdown
        from perfxpert.connection import PerfxpertConnection

        breakdown = compute_time_breakdown(PerfxpertConnection([str(db0), str(db1)]))

        assert breakdown["total_kernel_time"] == 160
        assert breakdown["total_runtime"] == 80
        assert breakdown["normalized_runtime"] == 160
        assert breakdown["kernel_percent"] == 100.0
        assert breakdown["memcpy_percent"] == 0.0
        assert breakdown["overhead_percent"] == 0.0

    def test_time_breakdown_uses_memcpy_runtime_for_multidb_overlap(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 0)
        _create_test_db(db1, "kernel_B", 0)

        for path, name in [(db0, "copy0"), (db1, "copy1")]:
            conn = sqlite3.connect(str(path))
            conn.execute("DELETE FROM kernels_data")
            conn.execute(
                "INSERT INTO memory_copies_data VALUES (?, 100, 180, 80, 1024)",
                (name,),
            )
            conn.commit()
            conn.close()

        from perfxpert.analyze import compute_time_breakdown
        from perfxpert.connection import PerfxpertConnection

        breakdown = compute_time_breakdown(PerfxpertConnection([str(db0), str(db1)]))

        assert breakdown["total_kernel_time"] == 0
        assert breakdown["total_memcpy_time"] == 160
        assert breakdown["total_runtime"] == 80
        assert breakdown["normalized_runtime"] == 160
        assert breakdown["kernel_percent"] == 0.0
        assert breakdown["memcpy_percent"] == 100.0
        assert breakdown["overhead_percent"] == 0.0

    def test_time_breakdown_counts_shards_with_only_one_activity_source(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 80)
        _create_test_db(db1, "kernel_B", 0)

        conn = sqlite3.connect(str(db1))
        conn.execute("DELETE FROM kernels_data")
        conn.execute(
            "INSERT INTO memory_copies_data VALUES ('copy_B', 100, 120, 20, 1024)"
        )
        conn.commit()
        conn.close()

        from perfxpert.analyze import compute_time_breakdown
        from perfxpert.connection import PerfxpertConnection

        breakdown = compute_time_breakdown(PerfxpertConnection([str(db0), str(db1)]))

        assert breakdown["total_kernel_time"] == 80
        assert breakdown["total_memcpy_time"] == 20
        assert breakdown["total_runtime"] == 80
        assert breakdown["normalized_runtime"] == 100
        assert breakdown["kernel_percent"] == 80.0
        assert breakdown["memcpy_percent"] == 20.0
        assert breakdown["overhead_percent"] == 0.0

    def test_time_breakdown_uses_one_envelope_per_shard_across_sources(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 80)
        _create_test_db(db1, "kernel_B", 20)

        conn = sqlite3.connect(str(db0))
        conn.execute(
            "INSERT INTO memory_copies_data VALUES ('copy_A', 300, 340, 40, 1024)"
        )
        conn.commit()
        conn.close()

        from perfxpert.analyze import compute_time_breakdown
        from perfxpert.connection import PerfxpertConnection

        breakdown = compute_time_breakdown(PerfxpertConnection([str(db0), str(db1)]))

        assert breakdown["total_runtime"] == 240
        assert breakdown["normalized_runtime"] == 260
        assert breakdown["total_kernel_time"] == 100
        assert breakdown["total_memcpy_time"] == 40

    def test_kernels_visible_from_both_dbs(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 1000)
        _create_test_db(db1, "kernel_B", 2000)

        for path in (db0, db1):
            conn = sqlite3.connect(str(path))
            obj_type = conn.execute(
                "SELECT type FROM sqlite_master WHERE name='kernels'"
            ).fetchone()[0]
            conn.close()
            assert obj_type == "view"

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        conn = PerfxpertConnection([str(db0), str(db1)])
        rows = execute_statement(
            conn, "SELECT name FROM kernels ORDER BY name"
        ).fetchall()
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
            c.execute(
                "INSERT INTO memory_copies_data VALUES (?, 0, 100, 100, 1024)",
                (mc_name,),
            )
            c.commit()
            c.close()

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        conn = PerfxpertConnection([str(db0), str(db1)])
        rows = execute_statement(
            conn, "SELECT name FROM memory_copies"
        ).fetchall()
        assert len(rows) == 2

    def test_regions_from_both_dbs(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "k", 100)
        _create_test_db(db1, "k", 100)
        for path, rname in [(db0, "region_0"), (db1, "region_1")]:
            c = sqlite3.connect(str(path))
            c.execute(
                "INSERT INTO regions_data VALUES (?, 'api', 0, 100, 100)",
                (rname,),
            )
            c.commit()
            c.close()

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        conn = PerfxpertConnection([str(db0), str(db1)])
        rows = execute_statement(conn, "SELECT name FROM regions").fetchall()
        assert len(rows) == 2

    def test_single_db_still_works(self, tmp_path):
        """Single-DB path must not regress — only the multi-DB union path is special."""
        db0 = tmp_path / "single.db"
        _create_test_db(db0, "kernel_only", 500)

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        conn = PerfxpertConnection([str(db0)])
        rows = execute_statement(conn, "SELECT name FROM kernels").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "kernel_only"

    def test_mixed_schema_shards_use_column_intersection(self, tmp_path):
        """Cycle-2 I-2 regression: shard[1] with an extra column must not
        raise at query time. The unified view must SELECT only the
        intersection of columns, and a RuntimeWarning must be emitted so
        the caller knows a column was dropped."""
        import sqlite3
        import warnings

        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"

        # shard0: baseline schema (no "extra_col")
        _create_test_db(db0, "kernel_A", 1000)

        # shard1: identical table but with an extra "extra_col" column
        conn1 = sqlite3.connect(str(db1))
        conn1.execute(
            "CREATE TABLE kernels (name TEXT, start INTEGER, end INTEGER, "
            "duration INTEGER, extra_col TEXT)"
        )
        conn1.execute(
            "INSERT INTO kernels VALUES ('kernel_B', 100, 200, 100, 'x')"
        )
        conn1.execute(
            "CREATE TABLE pmc_events "
            "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, "
            "counter_name TEXT, counter_value REAL)"
        )
        conn1.commit()
        conn1.close()

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            conn = PerfxpertConnection([str(db0), str(db1)])

        # Rows from both shards must be visible — no OperationalError at
        # query time.
        rows = execute_statement(
            conn, "SELECT name FROM kernels ORDER BY name"
        ).fetchall()
        assert {r[0] for r in rows} == {"kernel_A", "kernel_B"}

        # The dropped "extra_col" must NOT be addressable through the
        # unified view (confirming intersection semantics).
        with __import__("pytest").raises(sqlite3.OperationalError):
            execute_statement(conn, "SELECT extra_col FROM kernels").fetchall()

        # A RuntimeWarning must have surfaced, naming the dropped column.
        matches = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "extra_col" in str(w.message)
        ]
        assert matches, (
            f"expected RuntimeWarning mentioning 'extra_col'; got: "
            f"{[str(w.message) for w in caught]}"
        )

    def test_mixed_schema_shards_extra_col_on_shard0(self, tmp_path):
        """Symmetric case: extra column on shard[0] instead of shard[1]
        must not leak through the unified view (intersection is
        order-independent)."""
        import sqlite3
        import warnings

        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"

        # shard0: extra column
        conn0 = sqlite3.connect(str(db0))
        conn0.execute(
            "CREATE TABLE kernels (name TEXT, start INTEGER, end INTEGER, "
            "duration INTEGER, extra_col TEXT)"
        )
        conn0.execute(
            "INSERT INTO kernels VALUES ('kernel_A', 0, 100, 100, 'y')"
        )
        conn0.execute(
            "CREATE TABLE pmc_events "
            "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, "
            "counter_name TEXT, counter_value REAL)"
        )
        conn0.commit()
        conn0.close()

        # shard1: baseline schema
        _create_test_db(db1, "kernel_B", 2000)

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            conn = PerfxpertConnection([str(db0), str(db1)])

        rows = execute_statement(
            conn, "SELECT name FROM kernels ORDER BY name"
        ).fetchall()
        assert {r[0] for r in rows} == {"kernel_A", "kernel_B"}

        with __import__("pytest").raises(sqlite3.OperationalError):
            execute_statement(conn, "SELECT extra_col FROM kernels").fetchall()

        assert any(
            issubclass(w.category, RuntimeWarning)
            and "extra_col" in str(w.message)
            for w in caught
        )

    def test_view_missing_from_one_shard_does_not_break(self, tmp_path):
        """If a shard lacks one of the analysis views, union the rest."""
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_test_db(db0, "kernel_A", 1000)

        # shard1 has kernels but NO memory_copies / regions
        conn1 = sqlite3.connect(str(db1))
        conn1.execute(
            "CREATE TABLE kernels (name TEXT, start INTEGER, end INTEGER, "
            "duration INTEGER)"
        )
        conn1.execute(
            "INSERT INTO kernels VALUES ('kernel_B', 100, 200, 100)"
        )
        conn1.execute(
            "CREATE TABLE pmc_events "
            "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, "
            "counter_name TEXT, counter_value REAL)"
        )
        conn1.commit()
        conn1.close()

        from perfxpert.connection import (
            PerfxpertConnection,
            execute_statement,
        )

        conn = PerfxpertConnection([str(db0), str(db1)])
        # kernels exists in both shards → both names visible
        rows = execute_statement(
            conn, "SELECT name FROM kernels ORDER BY name"
        ).fetchall()
        assert {r[0] for r in rows} == {"kernel_A", "kernel_B"}
        # memory_copies exists only in shard0 → still queryable
        rows_mc = execute_statement(
            conn, "SELECT COUNT(*) FROM memory_copies"
        ).fetchone()
        assert rows_mc[0] == 0
