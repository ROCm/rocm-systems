#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
Pure-Python sqlite3 database connection for AI analysis.

Drop-in replacement for rocpd's ``importer.RocpdImportData`` that uses only
the stdlib ``sqlite3`` module (no C extension required).  The class
:class:`AnalysisConnection` attaches one or more profiling databases into
an in-memory connection, creates UNION ALL views that unify the
``rocpd_*`` tables from each database (stripping per-process UUID suffixes),
and exposes the resulting connection for analysis queries.

Usage:
    from rocm_ai_analysis.db import AnalysisConnection

    with AnalysisConnection(["trace_0.db", "trace_1.db"]) as conn:
        rows = conn.execute("SELECT * FROM rocpd_api").fetchall()
"""

import os
import sqlite3
import sys
from typing import Dict, List, Optional, Union

__all__ = ["AnalysisConnection", "RocpdImportData", "execute_statement"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_sqlite_db(file_path: str) -> bool:
    """Return True if *file_path* starts with the SQLite magic header."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
        return header == b"SQLite format 3\x00"
    except (OSError, IOError):
        return False


def _check_for_valid_dbs(input_files: List[str]) -> bool:
    """Validate that every path in *input_files* is a valid SQLite3 database."""
    for file_path in input_files:
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} does not exist.", file=sys.stderr)
            return False
        if not _is_sqlite_db(file_path):
            print(
                f"Error: {file_path} is not an SQLite3 database. File not supported.",
                file=sys.stderr,
            )
            return False
    return True


def execute_statement(conn, statement: str, is_script: bool = False):
    """Execute a SQL statement on *conn* (AnalysisConnection or sqlite3.Connection).

    Parameters
    ----------
    conn : AnalysisConnection or sqlite3.Connection
        Database connection (or wrapper) to execute against.
    statement : str
        SQL statement or script to execute.
    is_script : bool
        If ``True``, use ``executescript`` (for multi-statement SQL).

    Returns
    -------
    sqlite3.Cursor
        The cursor resulting from the execution.

    Raises
    ------
    sqlite3.Error
        Re-raised after printing a diagnostic message to stderr.
    """
    if isinstance(conn, AnalysisConnection):
        _conn = conn.connection
    else:
        _conn = conn

    assert isinstance(_conn, sqlite3.Connection), (
        f"Expected sqlite3.Connection, got {type(_conn).__name__}"
    )

    try:
        if is_script:
            return _conn.executescript(statement)
        return _conn.execute(f"{statement}")
    except sqlite3.Error as err:
        sys.stderr.write(f"SQLite3 error: {err}\nStatement:\n\t{statement}\n")
        sys.stderr.flush()
        raise err


# ---------------------------------------------------------------------------
# Temp view creation — mirrors importer.py _create_temp_views exactly
# ---------------------------------------------------------------------------

def _create_temp_views(
    connection: sqlite3.Connection, input_files: List[str]
) -> Dict[str, List[str]]:
    """Create temporary unified views from multiple database files.

    This mirrors the logic in ``rocpd/importer.py::_create_temp_views``:

    1. ATTACH each database as ``db0``, ``db1``, ...
    2. Read UUIDs from ``rocpd_metadata WHERE tag='uuid'`` in each db.
    3. Enumerate all ``rocpd_*`` tables in each attached db.
    4. For each table that contains a known UUID substring, strip the UUID
       to derive the *base* table name.
    5. Create ``CREATE TEMPORARY VIEW <base> AS SELECT * FROM db0.<base>
       UNION ALL SELECT * FROM db1.<base> ...``.

    Returns
    -------
    dict
        Mapping of base table name to list of ``SELECT`` statements
        that were UNION'd together.
    """
    assert isinstance(connection, sqlite3.Connection)
    assert isinstance(input_files, list)

    # Attach each database and extract the uuid from each database
    dbinfo: List[str] = []
    uuids: List[str] = []
    for i, inp in enumerate(input_files):
        execute_statement(connection, f"ATTACH DATABASE '{inp}' AS db{i}")
        try:
            _uuids = [
                itr[0]
                for itr in execute_statement(
                    connection,
                    f"SELECT value FROM db{i}.rocpd_metadata WHERE tag='uuid'",
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            # Database may not have rocpd_metadata table (e.g. older format)
            _uuids = []
        dbinfo.append(f"db{i}")
        uuids += [itr for itr in _uuids if itr not in uuids]

    # unique set of universal process identifiers
    uuids = list(set(uuids))

    all_tables: Dict[str, List[str]] = {}
    for ditr in dbinfo:
        # get the tables for the given attached database
        tables = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT name FROM {ditr}.sqlite_master WHERE type='table' AND name LIKE 'rocpd_%'",
            ).fetchall()
        ]

        # loop over the tables
        for itr in tables:
            # loop over the UUIDs
            for uitr in uuids:
                # skip the tables without the UUID suffix
                if f"{uitr}" not in itr:
                    continue

                # strip the UUID suffix to create a base table name,
                # e.g. 'rocpd_string_03daf93' -> 'rocpd_string'
                base = itr.replace(f"{uitr}", "")

                # create a list of attached databases which have the base table name
                if base not in all_tables:
                    all_tables[base] = []

                # create the SELECT statement from this database
                select = f"SELECT * FROM {ditr}.{base}"

                # make sure that we don't duplicate SELECT statements of same
                # table from same attached database
                if select in all_tables[base]:
                    continue

                # add this to list
                all_tables[base].append(select)

    # create the temporary view that is a union of all the attached databases
    for key, itr in all_tables.items():
        stmt = "CREATE TEMPORARY VIEW {} AS {}".format(
            key, " UNION ALL ".join(itr)
        )
        execute_statement(connection, stmt)

    return all_tables


# ---------------------------------------------------------------------------
# AnalysisConnection — drop-in for RocpdImportData
# ---------------------------------------------------------------------------

class AnalysisConnection:
    """Pure-Python drop-in replacement for ``RocpdImportData``.

    Opens an in-memory SQLite connection, ATTACHes one or more profiling
    databases, and creates UNION ALL views that unify the ``rocpd_*``
    tables (with UUID suffixes stripped).

    Parameters
    ----------
    input : str or list[str]
        Path(s) to ``.rpd`` / ``.db`` database files.

    Attributes
    ----------
    connection : sqlite3.Connection
        The underlying in-memory SQLite connection.
    table_info : dict
        Mapping of base table name to list of SELECT statements.
    _paths : list[str]
        The resolved input file paths.

    Example
    -------
    ::

        with AnalysisConnection("output.db") as db:
            for row in db.execute("SELECT * FROM rocpd_api LIMIT 5"):
                print(row)
    """

    def __init__(
        self,
        input: Union[str, List[str]],
        skip_auto_merge: bool = False,
        automerge_limit: Optional[int] = None,
        dbname: str = ":memory:",
    ):
        # Normalise input to a list of paths
        if isinstance(input, str):
            self._paths: List[str] = [input]
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            self._paths = list(input)
        else:
            raise ValueError(
                f"input must be a str or non-empty list of str, got {type(input).__name__}"
            )

        # Validate
        if not _check_for_valid_dbs(self._paths):
            raise ValueError("One or more input files are not valid SQLite3 databases")

        # Open in-memory connection
        self.connection: sqlite3.Connection = sqlite3.connect(dbname)
        self.connection.execute("PRAGMA foreign_keys = ON")

        # Create unified views from all attached databases
        self.table_info = _create_temp_views(self.connection, self._paths)

    # -- Proxy attribute access to self.connection --------------------------

    def __getattr__(self, name):
        """Proxy attribute lookups to ``self.connection``."""
        # Avoid infinite recursion during __init__ before self.connection exists
        if name in ("connection", "_paths", "table_info"):
            raise AttributeError(name)
        return getattr(self.connection, name)

    # -- Context manager ----------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.connection.__exit__(exc_type, exc, tb)


# Alias so code that does ``from rocm_ai_analysis.db import RocpdImportData``
# continues to work.
RocpdImportData = AnalysisConnection
