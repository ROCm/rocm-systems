#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""
Pure-sqlite3 database connection for perfxpert.

Replaces rocprofiler-sdk's RocpdImportData (which requires libpyrocpd, a
compiled C++ extension) with a stdlib-only implementation. Supports single
and multiple .db files; multi-file databases are merged via SQLite ATTACH.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Union


__all__ = ["PerfxpertConnection", "execute_statement"]


# ---------------------------------------------------------------------------
# Compatibility alias used throughout analyze.py (same name as rocpd class)
# ---------------------------------------------------------------------------

def execute_statement(conn, statement, params=()):
    """Execute a SQL statement on a PerfxpertConnection or raw sqlite3.Connection."""
    if isinstance(conn, PerfxpertConnection):
        raw = conn.connection
    else:
        raw = conn
    if params:
        return raw.execute(statement, params)
    return raw.execute(statement)


class PerfxpertConnection:
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
    _PERFXPERT_TABLES = [
        "rocpd_kernel_dispatch",
        "rocpd_memory_copy",
        "rocpd_api",
        "rocpd_agent",
        "rocpd_metadata",
        "pmc_events",
    ]

    # SQL views defined inside each rocpd database that analysis code queries
    # directly (e.g. analysis/core.py uses "kernels", "memory_copies",
    # "regions").  These live as VIEWs in each shard's sqlite_master, so
    # ATTACHed multi-DB sessions must create a TEMP VIEW that UNIONs every
    # shard — otherwise only db0's rows are visible (silent data loss).
    # Ports rocm-systems#4982.
    _ANALYSIS_VIEWS = ["kernels", "memory_copies", "regions"]

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

    @property
    def db_count(self) -> int:
        return len(self._paths)

    def sum_shard_runtime_envelopes(self, sources: List[str]) -> int:
        """Sum per-shard runtime envelopes across the requested sources.

        This keeps multi-DB aliasing details inside the connection layer so
        analysis code can normalize percentages without depending on ATTACH
        schema internals.
        """
        if len(self._paths) <= 1:
            return 0

        total_runtime = 0
        for idx in range(len(self._paths)):
            alias = f"db{idx}"
            shard_sources = []
            for source in sources:
                check = self.connection.execute(
                    f"SELECT COUNT(*) FROM {alias}.sqlite_master "
                    f"WHERE (type='table' OR type='view') AND name=?",
                    (source,),
                ).fetchone()
                if check and check[0] > 0:
                    shard_sources.append(f"SELECT start, end FROM {alias}.[{source}]")

            if not shard_sources:
                continue

            shard_query = (
                "SELECT COALESCE(MAX(end) - MIN(start), 0) FROM ("
                + " UNION ALL ".join(shard_sources)
                + ")"
            )
            shard_runtime = self.connection.execute(shard_query).fetchone()
            total_runtime += shard_runtime[0] or 0

        return total_runtime

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
        """Create UNION ALL views so multi-file queries are transparent.

        Mixed-schema shards (e.g. mismatched rocpd versions) are handled
        by taking the **intersection** of column names across every
        shard that has the table. This means:
          - Extra columns present on one shard but not others are
            silently dropped at view-build time (explicit, intentional)
            instead of raising ``OperationalError`` at query time.
          - If the intersection is empty (no shared columns across
            shards), the view is skipped and a warning is emitted so
            the caller can surface a diagnostic instead of hitting a
            confusing SQL error downstream.
        """
        import warnings

        n = len(self._paths)
        for table in self._PERFXPERT_TABLES:
            # Check which schemas actually have this table AND collect
            # their column sets for the intersection.
            schemas_with_table: list[str] = []
            per_shard_cols: list[list[str]] = []
            for idx in range(n):
                try:
                    conn.execute(f"SELECT 1 FROM db{idx}.{table} LIMIT 1")
                except sqlite3.OperationalError:
                    continue
                cursor = conn.execute(f"PRAGMA db{idx}.table_info({table})")
                shard_cols = [row[1] for row in cursor.fetchall()]
                if not shard_cols:
                    continue
                schemas_with_table.append(f"db{idx}")
                per_shard_cols.append(shard_cols)

            if not schemas_with_table:
                continue

            # Intersect preserving the order of shard-0's column list so
            # the view is stable across restarts.
            first_cols = per_shard_cols[0]
            common: set = set(first_cols)
            for shard_cols in per_shard_cols[1:]:
                common &= set(shard_cols)
            cols = [c for c in first_cols if c in common]
            if not cols:
                warnings.warn(
                    f"perfxpert.connection: no common columns across shards "
                    f"for table {table!r}; skipping unified view. "
                    f"Per-shard column sets: "
                    f"{[sorted(s) for s in per_shard_cols]}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue

            # If any shard had extra columns beyond the intersection,
            # surface a warning once so the operator knows data is
            # being trimmed. This is quiet by default (single warning
            # per table) to avoid noise on happy-path multi-DB.
            dropped_per_shard = [
                sorted(set(s) - common) for s in per_shard_cols
            ]
            if any(dropped_per_shard):
                warnings.warn(
                    f"perfxpert.connection: schema-mixed shards for table "
                    f"{table!r}; unified view uses the column intersection. "
                    f"Dropped columns per shard: {dropped_per_shard}",
                    RuntimeWarning,
                    stacklevel=3,
                )

            col_list = ", ".join(cols)
            parts = [
                f"SELECT {col_list} FROM {s}.{table}"
                for s in schemas_with_table
            ]
            union_sql = " UNION ALL ".join(parts)
            conn.execute(
                f"CREATE TEMP VIEW IF NOT EXISTS {table} AS {union_sql}"
            )

        # Union analysis-facing views (kernels, memory_copies, regions)
        # that are defined as SQL VIEWs inside each rocpd database
        # file.  Without this, analysis code that queries `SELECT *
        # FROM kernels` against a multi-DB session would see only the
        # first shard's view.  Ports rocm-systems#4982.
        #
        # Mixed-schema shards: the analysis views (kernels, regions,
        # memory_copies) expose a stable column vocabulary across
        # rocpd versions, but if a shard diverges the bare ``SELECT *``
        # UNION would fail at CREATE-VIEW time with "SELECTs to the
        # left and right of UNION ALL do not have the same number of
        # result columns". We intersect column names here too so
        # schema-mixed shards are handled identically to base tables.
        for view_name in self._ANALYSIS_VIEWS:
            shards_with_view: list[str] = []
            per_shard_cols: list[list[str]] = []
            for idx in range(n):
                alias = f"db{idx}"
                try:
                    check = conn.execute(
                        f"SELECT COUNT(*) FROM {alias}.sqlite_master "
                        f"WHERE (type='table' OR type='view') "
                        f"AND name='{view_name}'"
                    ).fetchone()
                except sqlite3.OperationalError:
                    continue
                if not (check and check[0] > 0):
                    continue
                # PRAGMA table_info works on views as well as tables.
                try:
                    cursor = conn.execute(
                        f"PRAGMA {alias}.table_info({view_name})"
                    )
                    shard_cols = [row[1] for row in cursor.fetchall()]
                except sqlite3.OperationalError:
                    shard_cols = []
                if not shard_cols:
                    continue
                shards_with_view.append(alias)
                per_shard_cols.append(shard_cols)

            if not shards_with_view:
                continue

            first_cols = per_shard_cols[0]
            common: set = set(first_cols)
            for shard_cols in per_shard_cols[1:]:
                common &= set(shard_cols)
            cols = [c for c in first_cols if c in common]
            if not cols:
                warnings.warn(
                    f"perfxpert.connection: no common columns across shards "
                    f"for analysis view {view_name!r}; skipping unified "
                    f"view. Per-shard column sets: "
                    f"{[sorted(s) for s in per_shard_cols]}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue

            dropped_per_shard = [
                sorted(set(s) - common) for s in per_shard_cols
            ]
            if any(dropped_per_shard):
                warnings.warn(
                    f"perfxpert.connection: schema-mixed shards for analysis "
                    f"view {view_name!r}; unified view uses the column "
                    f"intersection. Dropped columns per shard: "
                    f"{dropped_per_shard}",
                    RuntimeWarning,
                    stacklevel=3,
                )

            col_list = ", ".join(cols)
            parts = [
                f"SELECT {col_list} FROM {alias}.{view_name}"
                for alias in shards_with_view
            ]
            union_sql = " UNION ALL ".join(parts)
            conn.execute(
                f"CREATE TEMP VIEW IF NOT EXISTS [{view_name}] "
                f"AS {union_sql}"
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

    This is the perfxpert equivalent of rocpd.merge.merge_sqlite_dbs,
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
            cursor = conn.execute(
                f"SELECT type, name FROM {alias}.sqlite_master "
                f"WHERE type = 'table'"
            )
            tables = [
                row[1]
                for row in cursor.fetchall()
                if not row[1].startswith("sqlite_")
            ]
            for table in tables:
                try:
                    # Check for primary key collisions before inserting
                    pk_cols = [
                        row[1] for row in conn.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                        if row[5] > 0  # row[5] is the 'pk' column
                    ]
                    if pk_cols:
                        pk_join = " AND ".join(
                            f"src.{c} = dst.{c}" for c in pk_cols
                        )
                        collision = conn.execute(
                            f"SELECT COUNT(*) FROM {alias}.{table} src "
                            f"INNER JOIN main.{table} dst ON {pk_join}"
                        ).fetchone()
                        if collision and collision[0] > 0:
                            raise ValueError(
                                f"Primary key collision in table '{table}': "
                                f"{collision[0]} overlapping row(s) between "
                                f"shards. Cannot merge losslessly. Use "
                                f"PerfxpertConnection([db1, db2]) for "
                                f"multi-DB analysis instead of physical merge."
                            )
                    # No collision (or no PK) — safe to insert
                    conn.execute(
                        f"INSERT INTO main.{table} "
                        f"SELECT * FROM {alias}.{table}"
                    )
                except sqlite3.OperationalError:
                    pass  # table may have different schema in this shard
            conn.commit()
            conn.execute(f"DETACH DATABASE {alias}")
    finally:
        conn.close()

    return output_path
