#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
Pure-sqlite3 database connection for rocinsight.

Replaces rocprofiler-sdk's RocpdImportData (which requires libpyrocpd, a
compiled C++ extension) with a stdlib-only implementation. Supports single
and multiple .db files; multi-file databases are merged via SQLite ATTACH.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Union


__all__ = ["RocinsightConnection", "execute_statement"]


# ---------------------------------------------------------------------------
# Compatibility alias used throughout analyze.py (same name as rocpd class)
# ---------------------------------------------------------------------------

def execute_statement(conn, statement, params=()):
    """Execute a SQL statement on a RocinsightConnection or raw sqlite3.Connection."""
    if isinstance(conn, RocinsightConnection):
        raw = conn.connection
    else:
        raw = conn
    if params:
        return raw.execute(statement, params)
    return raw.execute(statement)


class RocinsightConnection:
    """
    Pure-Python replacement for rocpd's RocpdImportData.

    Opens one or more rocprofiler-sdk .db (SQLite) trace files and exposes
    a unified sqlite3.Connection that analysis functions can query directly.

    Single file
    -----------
    Opens the file directly — zero overhead, no ATTACH needed.

    Multiple files
    --------------
    Creates an in-memory database and ATTACHes each file as a named schema
    (db0, db1, …).  Unified views named after the standard rocpd tables
    (rocpd_kernel_dispatch, rocpd_memory_copy, …) are created via UNION ALL
    so that all analysis queries work unchanged.
    """

    # Standard rocpd table names that analysis functions query
    _ROCINSIGHT_TABLES = [
        "rocpd_kernel_dispatch",
        "rocpd_memory_copy",
        "rocpd_api",
        "rocpd_agent",
        "rocpd_metadata",
        "pmc_events",
    ]

    def __init__(
        self,
        db_paths: Union[str, Path, List[Union[str, Path]]],
        *,
        timeout: float = 30.0,
    ):
        if isinstance(db_paths, (str, Path)):
            db_paths = [db_paths]
        self._paths = [Path(p) for p in db_paths]
        for p in self._paths:
            if not p.exists():
                raise FileNotFoundError(f"Database not found: {p}")
        self.connection = self._open(timeout)

    # ------------------------------------------------------------------
    # Public interface (mirrors the subset used by analyze.py)
    # ------------------------------------------------------------------

    def execute(self, sql: str, params=()):
        return self.connection.execute(sql, params)

    def cursor(self):
        return self.connection.cursor()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self, timeout: float) -> sqlite3.Connection:
        if len(self._paths) == 1:
            conn = sqlite3.connect(str(self._paths[0]), timeout=timeout)
            conn.row_factory = sqlite3.Row
            return conn

        # Multi-file: merge via in-memory DB + ATTACH
        conn = sqlite3.connect(":memory:", timeout=timeout)
        conn.row_factory = sqlite3.Row

        for idx, path in enumerate(self._paths):
            conn.execute(f"ATTACH DATABASE ? AS db{idx}", (str(path),))

        self._create_union_views(conn)
        return conn

    def _create_union_views(self, conn: sqlite3.Connection) -> None:
        """Create UNION ALL views so multi-file queries are transparent."""
        n = len(self._paths)
        for table in self._ROCINSIGHT_TABLES:
            # Check which schemas actually have this table
            schemas_with_table = []
            for idx in range(n):
                try:
                    conn.execute(
                        f"SELECT 1 FROM db{idx}.{table} LIMIT 1"
                    )
                    schemas_with_table.append(f"db{idx}")
                except sqlite3.OperationalError:
                    pass

            if not schemas_with_table:
                continue

            # Get column list from first available schema
            schema = schemas_with_table[0]
            cursor = conn.execute(f"PRAGMA {schema}.table_info({table})")
            cols = [row[1] for row in cursor.fetchall()]
            if not cols:
                continue

            col_list = ", ".join(cols)
            parts = [
                f"SELECT {col_list} FROM {s}.{table}"
                for s in schemas_with_table
            ]
            union_sql = " UNION ALL ".join(parts)
            conn.execute(
                f"CREATE TEMP VIEW IF NOT EXISTS {table} AS {union_sql}"
            )


# ---------------------------------------------------------------------------
# Convenience: merge multiple .db files into a single output file (pure SQL)
# ---------------------------------------------------------------------------

def merge_sqlite_dbs(
    db_paths: List[Union[str, Path]],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Merge multiple rocpd .db files into one SQLite file using ATTACH + INSERT.

    This is the rocinsight equivalent of rocpd.merge.merge_sqlite_dbs,
    implemented without any C++ dependency.

    Parameters
    ----------
    db_paths:
        List of .db file paths to merge (must share the same schema).
    output_path:
        Destination path for merged database.  Defaults to a sibling
        file called ``merged_processes.db`` next to the first input file.

    Returns
    -------
    Path to the merged database file.
    """
    if not db_paths:
        raise ValueError("No database paths provided")

    db_paths = [Path(p) for p in db_paths]
    if output_path is None:
        output_path = db_paths[0].parent / "merged_processes.db"
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)

    import shutil

    # Start from a copy of the first DB (preserves schema + data)
    shutil.copy2(db_paths[0], output_path)

    if len(db_paths) == 1:
        return output_path

    conn = sqlite3.connect(str(output_path))
    try:
        for idx, src in enumerate(db_paths[1:], start=1):
            alias = f"src{idx}"
            conn.execute(f"ATTACH DATABASE ? AS {alias}", (str(src),))
            cursor = conn.execute(f"PRAGMA {alias}.sqlite_master")
            tables = [
                row[1]
                for row in cursor.fetchall()
                if row[0] == "table" and not row[1].startswith("sqlite_")
            ]
            for table in tables:
                try:
                    conn.execute(
                        f"INSERT OR IGNORE INTO main.{table} "
                        f"SELECT * FROM {alias}.{table}"
                    )
                except sqlite3.OperationalError:
                    pass  # table may have different schema in this shard
            conn.execute(f"DETACH DATABASE {alias}")
        conn.commit()
    finally:
        conn.close()

    return output_path
