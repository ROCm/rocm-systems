#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

"""Validation boundary for attaching and inspecting rocPD databases.

Input databases are data, not SQL sources. This module is the only place where
the merge and import paths inspect their schemas. It validates object names and
physical table layouts against a supported :class:`RocpdSchema`. Canonical
views and standalone indexes may be absent from the input. Input view bodies
are discarded, and SQL stored in an input database's ``sqlite_master`` table
is never replayed.
"""

from pathlib import Path
import os
import re
import sqlite3
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from .schema import RocpdSchema, query_supported_schema_versions

SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ROCPD_UUID = re.compile(r"^_[A-Za-z0-9_]+$")
ROCPD_GUID = re.compile(r"^[A-Za-z0-9_-]+$")
SCHEMA_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
REGULAR_TABLE_SQL = re.compile(
    r"^\s*CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s",
    flags=re.IGNORECASE,
)
METADATA_TABLE = re.compile(r"^rocpd_metadata(?P<uuid>_[A-Za-z0-9_]+)$")
_SQL_TOKEN = re.compile(
    r"--[^\r\n]*|/\*[\s\S]*?\*/|"
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`(?:[^`]|``)*`|\[[^\]]*\]|"
    r"[A-Za-z_][A-Za-z0-9_$]*|[^\s]"
)

_METADATA_COLUMNS = (
    (0, "id", "INTEGER", 1, None, 1, 0),
    (1, "tag", "TEXT", 1, None, 0, 0),
    (2, "value", "TEXT", 1, None, 0, 0),
)


class RocpdSourceSchema(NamedTuple):
    """Validated description of one attached rocPD database."""

    path: str
    alias: str
    version: str
    uuids: Tuple[str, ...]
    tables: Mapping[str, Tuple[str, ...]]
    schemas: Tuple[RocpdSchema, ...]


def quote_identifier(name: str) -> str:
    """Validate and quote a SQLite identifier.

    SQLite does not support binding identifiers. Keeping the accepted alphabet
    deliberately small makes every interpolated identifier auditable.
    """

    if not isinstance(name, str) or not SQL_IDENTIFIER.fullmatch(name):
        raise ValueError(f"Refusing invalid database identifier: {name!r}")
    return f'"{name}"'


def qualified_identifier(database: str, name: str) -> str:
    return f"{quote_identifier(database)}.{quote_identifier(name)}"


def validate_merge_destination(path: str) -> None:
    """Refuse to replace a database while SQLite sidecars are present.

    Publication replaces only the main file. Even empty or orphaned sidecars
    are rejected, since they can belong to another SQLite connection. The
    destination must remain offline throughout the merge; this check is not
    synchronization with concurrent readers or writers.
    """

    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = f"{os.fspath(path)}{suffix}"
        if os.path.lexists(sidecar):
            raise ValueError(
                f"Destination database has SQLite sidecar: {sidecar!s}. "
                "Recover/checkpoint and close the destination before merging."
            )


def readonly_database_uri(path: str) -> str:
    """Return an absolute, read-only and immutable SQLite URI."""

    source = Path(os.fspath(path)).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Input database is not a regular file: {path!r}")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{source}{suffix}")
        if sidecar.is_file() and sidecar.stat().st_size > 0:
            raise ValueError(
                f"Input database has an uncheckpointed SQLite journal: {sidecar!s}"
            )
    return f"{source.as_uri()}?mode=ro&immutable=1"


def attach_readonly(connection: sqlite3.Connection, path: str, alias: str) -> str:
    """Attach *path* without interpolating it into SQL."""

    quoted_alias = quote_identifier(alias)
    uri = readonly_database_uri(path)
    connection.execute(
        f"ATTACH DATABASE ? AS {quoted_alias}",
        (uri,),
    )
    return str(Path(os.fspath(path)).expanduser().resolve(strict=True))


def detach_database(connection: sqlite3.Connection, alias: str) -> None:
    connection.execute(f"DETACH DATABASE {quote_identifier(alias)}")


def configure_untrusted_schema(connection: sqlite3.Connection) -> None:
    """Disable SQLite's use of application-defined functions in schemas."""

    # Unknown pragmas are ignored by older SQLite versions, while read-only URI
    # attachment and strict object validation still provide the core boundary.
    connection.execute("PRAGMA trusted_schema = OFF")


def _master_objects(
    connection: sqlite3.Connection, alias: str
) -> List[Tuple[str, str, str, Optional[str]]]:
    quoted_alias = quote_identifier(alias)
    rows = connection.execute(
        f"""
        SELECT type, name, tbl_name, sql
        FROM {quoted_alias}.sqlite_master
        WHERE name NOT GLOB 'sqlite_*'
        ORDER BY type, name
        """
    ).fetchall()
    result = []
    for kind, name, table_name, sql in rows:
        quote_identifier(name)
        quote_identifier(table_name)
        result.append((kind, name, table_name, sql))
    return result


def _pragma(
    connection: sqlite3.Connection,
    alias: str,
    pragma: str,
    name: str,
) -> List[Tuple]:
    if pragma not in {
        "foreign_key_list",
        "index_list",
        "index_xinfo",
        "table_info",
        "table_xinfo",
    }:
        raise ValueError(f"Unsupported schema pragma: {pragma!r}")
    return connection.execute(
        f"PRAGMA {quote_identifier(alias)}.{pragma}({quote_identifier(name)})"
    ).fetchall()


def _table_columns(
    connection: sqlite3.Connection, alias: str, table: str
) -> Tuple[Tuple, ...]:
    rows = _pragma(connection, alias, "table_xinfo", table)
    if not rows:
        # table_xinfo was added after the oldest SQLite versions supported by
        # Python 3.6. Those versions cannot contain generated/hidden columns,
        # so table_info plus a zero hidden-column flag is equivalent.
        rows = [
            tuple(row) + (0,) for row in _pragma(connection, alias, "table_info", table)
        ]
    return tuple(tuple(row) for row in rows)


def _foreign_keys(
    connection: sqlite3.Connection, alias: str, table: str
) -> Tuple[Tuple, ...]:
    return tuple(
        tuple(row) for row in _pragma(connection, alias, "foreign_key_list", table)
    )


def _index_signatures(
    connection: sqlite3.Connection, alias: str, table: str
) -> Tuple[Tuple, ...]:
    signatures = []
    for _, index_name, unique, origin, partial in _pragma(
        connection, alias, "index_list", table
    ):
        quote_identifier(index_name)
        columns = tuple(
            tuple(row[1:])
            for row in _pragma(connection, alias, "index_xinfo", index_name)
        )
        signatures.append((unique, origin, partial, columns))
    return tuple(sorted(signatures, key=repr))


def _check_constraints(sql: str) -> Tuple[Tuple[str, ...], ...]:
    """Extract CHECK expression tokens without evaluating source SQL.

    PRAGMAs do not expose CHECK clauses. Compare their tokens to the trusted
    schema, ignoring comments, whitespace, keyword case and constraint order.
    Quoted tokens are preserved so literal contents and parentheses inside
    strings cannot change or disguise a constraint.
    """

    checks = []
    expression = None
    depth = 0
    for match in _SQL_TOKEN.finditer(sql):
        token = match.group()
        if token.startswith(("--", "/*")):
            continue
        if token[0] not in "'\"`[":
            token = token.lower()
        if expression is None:
            if token == "check":
                expression = []
            continue
        if not expression and token != "(":
            raise ValueError("Invalid CHECK constraint")
        expression.append(token)
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                checks.append(tuple(expression))
                expression = None
    if expression is not None:
        raise ValueError("Unterminated CHECK constraint")
    return tuple(sorted(checks))


def _table_signature(
    connection: sqlite3.Connection, alias: str, table: str
) -> Tuple[Tuple, Tuple, Tuple]:
    return (
        _table_columns(connection, alias, table),
        _foreign_keys(connection, alias, table),
        _index_signatures(connection, alias, table),
    )


def _metadata_candidates(
    objects: Sequence[Tuple[str, str, str, Optional[str]]],
    source_path: str,
) -> List[Tuple[str, str]]:
    candidates = []
    for kind, name, _, sql in objects:
        match = METADATA_TABLE.fullmatch(name)
        if kind == "table" and match:
            if not isinstance(sql, str) or not REGULAR_TABLE_SQL.match(sql):
                raise ValueError(
                    f"Refusing non-table rocPD metadata object in {source_path!r}"
                )
            candidates.append((name, match.group("uuid")))

    if not candidates:
        raise ValueError(
            f"Expected at least one physical rocPD metadata table in {source_path!r}"
        )
    return candidates


def _metadata_values(
    connection: sqlite3.Connection,
    alias: str,
    metadata_table: str,
    table_uuid: str,
    source_path: str,
) -> Tuple[str, str, str]:
    columns = _table_columns(connection, alias, metadata_table)
    if columns != _METADATA_COLUMNS:
        raise ValueError(
            f"Invalid rocPD metadata table schema in {source_path!r}: {columns!r}"
        )

    required = (
        "uuid",
        "guid",
        "schema_version",
        "schema_version_major",
        "schema_version_minor",
        "schema_version_patch",
    )
    placeholders = ", ".join("?" for _ in required)
    # Read only required metadata, with one extra row to detect duplicates.
    # Additional metadata remains in the source and is copied by merge.
    rows = connection.execute(
        f"SELECT tag, value FROM {qualified_identifier(alias, metadata_table)} "
        f"WHERE tag COLLATE BINARY IN ({placeholders})",
        required,
    ).fetchmany(len(required) + 1)
    values: Dict[str, List[str]] = {}
    for tag, value in rows:
        if isinstance(tag, str):
            values.setdefault(tag, []).append(value)

    if any(len(values.get(key, [])) != 1 for key in required):
        raise ValueError(
            f"Invalid or duplicate required rocPD metadata in {source_path!r}"
        )

    uuid = values["uuid"][0]
    guid = values["guid"][0]
    version = values["schema_version"][0]
    if not isinstance(uuid, str) or not ROCPD_UUID.fullmatch(uuid):
        raise ValueError(f"Invalid rocPD UUID metadata in {source_path!r}: {uuid!r}")
    if uuid != table_uuid:
        raise ValueError(
            f"rocPD UUID metadata does not match its table suffix in {source_path!r}"
        )
    if not isinstance(guid, str) or not ROCPD_GUID.fullmatch(guid):
        raise ValueError(f"Invalid rocPD GUID metadata in {source_path!r}: {guid!r}")
    if not isinstance(version, str) or not SCHEMA_VERSION.fullmatch(version):
        raise ValueError(f"Invalid rocPD schema version in {source_path!r}: {version!r}")

    version_parts = version.split(".")
    for index, key in enumerate(
        ("schema_version_major", "schema_version_minor", "schema_version_patch")
    ):
        if len(values.get(key, [])) != 1 or str(values[key][0]) != version_parts[index]:
            raise ValueError(
                f"Inconsistent {key} metadata in input database {source_path!r}"
            )

    supported = {str(item) for item in query_supported_schema_versions()}
    if version not in supported:
        raise ValueError(
            f"Unsupported rocPD schema version {version!r} in {source_path!r}; "
            f"supported versions: {sorted(supported)!r}"
        )
    return uuid, guid, version


def _reference_schema(
    uuid: str, guid: str, version: str
) -> Tuple[RocpdSchema, sqlite3.Connection]:
    schema = RocpdSchema(uuid=uuid, guid=guid, version=version)
    reference = sqlite3.connect(":memory:")
    try:
        configure_untrusted_schema(reference)
        reference.executescript(schema.tables)
        reference.executescript(schema.indexes)
        reference.executescript(schema.views)
    except BaseException:
        reference.close()
        raise
    return schema, reference


def inspect_attached_rocpd(
    connection: sqlite3.Connection,
    alias: str,
    source_path: str,
) -> RocpdSourceSchema:
    """Validate an attached database against its trusted versioned schema."""

    quote_identifier(alias)
    objects = _master_objects(connection, alias)
    metadata = _metadata_candidates(objects, source_path)
    actual_by_kind = {
        kind: {name for obj_kind, name, _, _ in objects if obj_kind == kind}
        for kind in ("table", "index", "view", "trigger")
    }
    unsupported_kinds = {kind for kind, _, _, _ in objects} - {
        "table",
        "index",
        "view",
        "trigger",
    }
    if unsupported_kinds:
        raise ValueError(
            f"Unsupported database objects in {source_path!r}: "
            f"{sorted(unsupported_kinds)!r}"
        )

    object_sql = {name: sql for kind, name, _, sql in objects if kind == "table"}
    expected_by_kind = {kind: set() for kind in ("table", "index", "view", "trigger")}
    tables: Dict[str, List[str]] = {}
    schemas = []
    uuids = []
    version = None

    for metadata_table, table_uuid in metadata:
        uuid, guid, partition_version = _metadata_values(
            connection, alias, metadata_table, table_uuid, source_path
        )
        if version is None:
            version = partition_version
        elif partition_version != version:
            raise ValueError(
                f"Multiple schema versions found inside {source_path!r}: "
                f"{sorted({version, partition_version})!r}"
            )

        schema, reference = _reference_schema(uuid, guid, partition_version)
        try:
            expected_objects = _master_objects(reference, "main")
            expected_table_sql = {
                name: sql for kind, name, _, sql in expected_objects if kind == "table"
            }
            partition_expected = {
                kind: {
                    name for obj_kind, name, _, _ in expected_objects if obj_kind == kind
                }
                for kind in ("table", "index", "view", "trigger")
            }
            expected_by_kind["table"].update(partition_expected["table"])
            expected_by_kind["index"].update(partition_expected["index"])
            expected_by_kind["view"].update(partition_expected["view"])
            expected_by_kind["trigger"].update(partition_expected["trigger"])

            # Standalone indexes are rebuilt from trusted DDL and need not be
            # present in the source. Drop only absent canonical indexes from
            # this private reference before comparing table signatures. Keep
            # implicit PRIMARY KEY/UNIQUE indexes and validate present indexes.
            for index in partition_expected["index"] - actual_by_kind["index"]:
                reference.execute(f"DROP INDEX {quote_identifier(index)}")

            for table in sorted(partition_expected["table"]):
                sql = object_sql.get(table)
                if not isinstance(sql, str) or not REGULAR_TABLE_SQL.match(sql):
                    raise ValueError(
                        f"Refusing non-table object {table!r} in {source_path!r}"
                    )
                if _check_constraints(sql) != _check_constraints(
                    expected_table_sql[table]
                ):
                    raise ValueError(
                        f"rocPD table {table!r} CHECK constraints do not match schema "
                        f"version {partition_version} in {source_path!r}"
                    )
                if _table_signature(connection, alias, table) != _table_signature(
                    reference, "main", table
                ):
                    raise ValueError(
                        f"rocPD table {table!r} does not match schema version "
                        f"{partition_version} in {source_path!r}"
                    )
                if not table.endswith(uuid):
                    raise ValueError(
                        f"rocPD table {table!r} has an invalid UUID suffix in "
                        f"{source_path!r}"
                    )
                base = table[: -len(uuid)]
                quote_identifier(base)
                tables.setdefault(base, []).append(table)
        finally:
            reference.close()

        uuids.append(uuid)
        schemas.append(schema)

    for kind in ("table", "index", "view", "trigger"):
        unexpected = actual_by_kind[kind] - expected_by_kind[kind]
        # Views and standalone indexes are optional on input, but unfamiliar
        # persistent objects are still rejected rather than silently discarded.
        missing = (
            expected_by_kind[kind] - actual_by_kind[kind]
            if kind not in ("view", "index")
            else set()
        )
        if unexpected or missing:
            raise ValueError(
                f"Invalid rocPD {kind} objects in {source_path!r}; "
                f"unexpected={sorted(unexpected)!r}, missing={sorted(missing)!r}"
            )

    return RocpdSourceSchema(
        path=source_path,
        alias=alias,
        version=version,
        uuids=tuple(uuids),
        tables={base: tuple(names) for base, names in tables.items()},
        schemas=tuple(schemas),
    )


def create_union_views(
    connection: sqlite3.Connection,
    tables: Mapping[str, Iterable[Tuple[str, str]]],
    *,
    temporary: bool = False,
) -> None:
    """Create base-table UNION views from validated table mappings."""

    prefix = "CREATE TEMPORARY VIEW" if temporary else "CREATE VIEW"
    for base, sources in sorted(tables.items()):
        selects = [
            (
                f"SELECT * FROM {qualified_identifier(alias, table)}"
                if alias
                else f"SELECT * FROM {quote_identifier(table)}"
            )
            for alias, table in sources
        ]
        if not selects:
            raise ValueError(f"No tables available for rocPD view {base!r}")
        connection.execute(
            f"{prefix} {quote_identifier(base)} AS " + " UNION ALL ".join(selects)
        )
