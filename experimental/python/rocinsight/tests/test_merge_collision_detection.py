"""Tests for merge_sqlite_dbs primary key collision detection."""
import sqlite3
import pytest


def _create_db_with_pmc(path, row_id, counter_name, counter_value):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE pmc_events "
        "(id INTEGER PRIMARY KEY, dispatch_id INTEGER, counter_name TEXT, counter_value REAL)"
    )
    conn.execute(
        "INSERT INTO pmc_events VALUES (?, 1, ?, ?)",
        (row_id, counter_name, counter_value),
    )
    conn.commit()
    conn.close()


class TestMergeCollisionDetection:
    def test_raises_on_id_collision(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_db_with_pmc(db0, 1, "SQ_WAVES", 32.0)
        _create_db_with_pmc(db1, 1, "GRBM_COUNT", 1000.0)  # same id=1

        from rocinsight.connection import merge_sqlite_dbs
        merged = tmp_path / "merged.db"
        with pytest.raises(ValueError, match="collision|overlap"):
            merge_sqlite_dbs([str(db0), str(db1)], str(merged))

    def test_merge_succeeds_no_collision(self, tmp_path):
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        _create_db_with_pmc(db0, 1, "SQ_WAVES", 32.0)
        _create_db_with_pmc(db1, 2, "GRBM_COUNT", 1000.0)  # different id

        from rocinsight.connection import merge_sqlite_dbs
        merged = tmp_path / "merged.db"
        merge_sqlite_dbs([str(db0), str(db1)], str(merged))
        conn = sqlite3.connect(str(merged))
        count = conn.execute("SELECT COUNT(*) FROM pmc_events").fetchone()[0]
        assert count == 2

    def test_no_pk_table_merges_without_error(self, tmp_path):
        """Tables without primary keys should merge normally."""
        db0 = tmp_path / "shard0.db"
        db1 = tmp_path / "shard1.db"
        for path, val in [(db0, "a"), (db1, "b")]:
            conn = sqlite3.connect(str(path))
            conn.execute("CREATE TABLE nopk (val TEXT)")
            conn.execute("INSERT INTO nopk VALUES (?)", (val,))
            # Also need pmc_events for the merge to work on shared tables
            conn.execute("CREATE TABLE pmc_events (id INTEGER PRIMARY KEY, counter_name TEXT)")
            conn.commit()
            conn.close()

        from rocinsight.connection import merge_sqlite_dbs
        merged = tmp_path / "merged.db"
        merge_sqlite_dbs([str(db0), str(db1)], str(merged))
