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

from .database import (
    RocpdSourceSet,
    attach_readonly,
    configure_untrusted_schema,
    create_union_views,
    inspect_attached_rocpd,
    select_table,
)
from .schema import RocpdSchema
from . import libpyrocpd
from .features import get_supported_features_from_version

__all__ = ["RocpdImportData", "execute_statement"]


def internal_init(_input, _output, skip_auto_merge, automerge_limit):
    from . import package

    if isinstance(_input, str):
        _input = [_input]
    _input = package.flatten_rocpd_yaml_input_file(
        _input, skip_auto_merge=skip_auto_merge, automerge_limit=automerge_limit
    )
    if os.path.isdir(_output):
        raise ValueError("Output database name must not be a directory")
    if not _check_for_valid_dbs(_input):
        raise ValueError("RocpdImportData error, invalid SQLite3 database provided")
    _connection = libpyrocpd.connect(_output, uri=True)
    try:
        configure_untrusted_schema(_connection)
        _table_info, _version = _create_temp_views(_connection, _input)
        _schema_version = libpyrocpd.schema_version(_version)
        _create_meta_views(_connection, _schema_version)
        return (_connection, _input, _table_info, _schema_version)
    except BaseException:
        _connection.close()
        raise


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
        else:

            if isinstance(input, sqlite3.Connection):
                raise ValueError(
                    "RocpdImportData does not accept existing sqlite3 connections"
                )
            elif isinstance(input, str) or (
                isinstance(input, list) and len(input) > 0 and isinstance(input[0], str)
            ):
                _connection, _filenames, _table_info, _schema_version = internal_init(
                    input, dbname, skip_auto_merge, automerge_limit
                )
                self.table_info = _table_info
            else:
                raise ValueError(
                    f"input is unsupported type. Expected sqlite3.Connection, string, or (non-empty) list of strings. type={type(input).__name__}"
                )
            _supported_features = list(
                get_supported_features_from_version(_schema_version)
            )
            super(RocpdImportData, self).__init__(
                _connection, _filenames, _schema_version, _supported_features
            )

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
    """Create temporary unified views from multiple database files.

    Returns the per-table SELECT statements and the validated schema version.
    """

    assert isinstance(connection, sqlite3.Connection)
    assert isinstance(input, list)

    # Attach and validate each database. Source paths are bound parameters and
    # source sqlite_master SQL is never executed.
    sources = []
    source_set = RocpdSourceSet()
    for i, inp in enumerate(input):
        alias = f"db{i}"
        resolved_source = attach_readonly(connection, inp, alias)
        source = inspect_attached_rocpd(connection, alias, resolved_source)
        source_set.add(source)
        sources.append(source)

    if source_set.version is None:
        raise ValueError("No source databases provided")

    all_tables = {}
    union_tables = {}
    union_columns = {}
    for source in sources:
        for base, tables in source.tables.items():
            union_columns[base] = source.columns[base]
            for table in tables:
                select = select_table(source.alias, table, source.columns[base])
                all_tables.setdefault(base, []).append(select)
                union_tables.setdefault(base, []).append((source.alias, table))

    # create the temporary view that is a union of all the attached databases
    create_union_views(connection, union_tables, union_columns, temporary=True)

    return all_tables, source_set.version


def _create_meta_views(connection, schema_version):
    schema = RocpdSchema(version=str(schema_version))
    sql_script = schema.views.replace("CREATE VIEW", "CREATE TEMPORARY VIEW")
    execute_statement(connection, sql_script, is_script=True)
