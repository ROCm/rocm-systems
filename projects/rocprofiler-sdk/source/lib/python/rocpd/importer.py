###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

#
# Utility classes to simplify generating rpd files
#
#

import sys
import os
import sqlite3

from .schema import RocpdSchema
from . import libpyrocpd

__all__ = ["RocpdImportData", "execute_statement", "get_schema_version"]


def internal_init(_input, _output, skip_auto_merge, automerge_limit):
    from . import package

    _input = package.flatten_rocpd_yaml_input_file(
        _input, skip_auto_merge=skip_auto_merge, automerge_limit=automerge_limit
    )
    assert not os.path.isdir(_output), "Output database name must not be a directory"
    assert _check_for_valid_dbs(
        _input
    ), "RocpdImportData error, invalid SQLite3 database provided"
    _connection = libpyrocpd.connect(_output)
    _connection.execute("PRAGMA foreign_keys = ON")
    _table_info, _schema_versions = _create_temp_views(_connection, _input)
    _schema_version = _schema_versions[0] if _schema_versions else "0.0.0"
    _create_meta_views(_connection, _schema_version)
    return (_connection, _input, _table_info, _schema_versions)


class RocpdImportData(libpyrocpd.RocpdImportData):

    def __init__(
        self, input, skip_auto_merge=False, automerge_limit=None, dbname=":memory:"
    ):
        from . import package

        if automerge_limit is None:
            automerge_limit = package.IDEAL_NUMBER_OF_DATABASE_FILES

        if isinstance(input, RocpdImportData):
            super(RocpdImportData, self).__init__(input)
            self.table_info = input.table_info
            self.schema_version = input.schema_version
        else:

            if isinstance(input, sqlite3.Connection):
                raise ValueError(
                    "RocpdImportData does not accept existing sqlite3 connections"
                )
            elif isinstance(input, str) or (
                isinstance(input, list) and len(input) > 0 and isinstance(input[0], str)
            ):
                _connection, _filenames, _table_info, _schema_versions = internal_init(
                    input, dbname, skip_auto_merge, automerge_limit
                )
                self.table_info = _table_info
                unique_versions = list(set(_schema_versions))
                if len(unique_versions) > 1:
                    raise RuntimeError(
                        f"Multiple schema versions found across input databases: {unique_versions}"
                    )
                self.schema_version = unique_versions[0] if unique_versions else "0.0.0"
            else:
                raise ValueError(
                    f"input is unsupported type. Expected sqlite3.Connection, string, or (non-empty) list of strings. type={type(input).__name__}"
                )
            super(RocpdImportData, self).__init__(_connection, _filenames)

    def __getattr__(self, name):
        # any attribute or method not found in RocpdImportData will be looked up on self.connection
        return getattr(self.connection, name)

    def __enter__(self):
        # support "with RocpdImportData(...) as db:":
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.connection.__exit__(exc_type, exc, tb)


def _is_sqlite_db(file_path):
    with open(file_path, "rb") as f:
        header = f.read(16)
    return header == b"SQLite format 3\x00"


def _check_for_valid_dbs(input_files) -> bool:
    # check the list of .db files to confirm they are SQLite3 DBs
    for file in input_files:
        sqlite_db = _is_sqlite_db(file)
        if not sqlite_db:
            print(f"Error: {file} is not an SQLite3 database. File not supported.")
            return False
    return True


def execute_statement(conn, statement, is_script=False):
    if isinstance(conn, RocpdImportData):
        _conn = conn.connection
    else:
        _conn = conn

    assert isinstance(_conn, sqlite3.Connection)
    try:
        if is_script:
            return _conn.executescript(statement)
        return _conn.execute(f"{statement}")
    except sqlite3.Error as err:
        sys.stderr.write(f"SQLite3 error: {err}\nStatement:\n\t{statement}\n")
        sys.stderr.flush()
        raise err


def _create_temp_views(connection, input):
    """Create temporary unified views from multiple database files."""

    assert isinstance(connection, sqlite3.Connection)
    assert isinstance(input, list)

    # Attach each database and extract the uuid and schema_version from each database
    dbinfo = []
    uuids = []
    schema_versions = []
    for i, inp in enumerate(input):
        execute_statement(connection, f"ATTACH DATABASE '{inp}' AS db{i}")
        _uuids = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT value FROM db{i}.rocpd_metadata WHERE tag='uuid'",
            ).fetchall()
        ]
        _schema_versions = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT value FROM db{i}.rocpd_metadata WHERE tag='schema_version'",
            ).fetchall()
        ]
        dbinfo += [f"db{i}"]
        uuids += [itr for itr in _uuids if itr not in uuids]
        schema_versions += [v for v in _schema_versions if v not in schema_versions]

    # unique set of universal process identifiers
    uuids = list(set(uuids))

    all_tables = {}
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

                # strip the UUID suffix to create a base table name, e.g. 'rocpd_string_03daf93' -> 'rocpd_string'
                base = itr.replace(f"{uitr}", "")

                # create a list of attached databases which have the base table name
                if base not in all_tables.keys():
                    all_tables[base] = []

                # create the SELECT statement from this database
                select = f"SELECT * FROM {ditr}.{base}"

                # make sure that we don't duplicate SELECT statements of same table from same attached database
                if select in all_tables[base]:
                    continue

                # add this to list
                all_tables[base] += [select]

    # create the temporary view that is a union of all the attached databases
    for key, itr in all_tables.items():
        stmt = "CREATE TEMPORARY VIEW {} AS {}".format(key, " UNION ALL ".join(itr))
        execute_statement(connection, stmt)

    return all_tables, schema_versions


def _create_meta_views(connection, schema_version="0.0.0"):
    schema = RocpdSchema(version=schema_version)
    sql_script = schema.views.replace("CREATE VIEW", "CREATE TEMPORARY VIEW")
    execute_statement(connection, sql_script, is_script=True)


def get_schema_version(importData) -> tuple:
    """Return the schema version of importData as a comparable integer tuple.

    Example usage:
        if get_schema_version(importData) >= (3, 0, 1):
            # include fields added in schema 3.0.1

    Falls back to (0, 0, 0) if the version string is absent or malformed.
    """
    ver_str = getattr(importData, "schema_version", "0.0.0")
    try:
        return tuple(int(x) for x in ver_str.split("."))
    except (ValueError, AttributeError):
        return (0, 0, 0)
