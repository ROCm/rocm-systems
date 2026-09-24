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

Input databases are data, not SQL sources. SQL stored in an input database's
``sqlite_master`` table is never replayed. Validation establishes only what the
merge and import paths rely on: every table of the trusted, versioned
:class:`RocpdSchema` is a regular table with the expected columns, and all
reads name those columns explicitly. Row values remain untrusted; merged output
enforces constraints through the trusted DDL. Other schema objects in an input
database are never read and are skipped with a warning.
"""

from pathlib import Path
import os
import re
import sqlite3
import sys
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

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
METADATA_COLUMNS = ("id", "tag", "value")
REQUIRED_METADATA = (
    "uuid",
    "guid",
    "schema_version",
    "schema_version_major",
    "schema_version_minor",
    "schema_version_patch",
)

# name -> (type, sql) for the user objects of one database
SchemaObjects = Mapping[str, Tuple[str, Optional[str]]]


class RocpdSourceSchema(NamedTuple):
    """Validated description of one attached rocPD database."""

    path: str
    alias: str
    version: str
    uuids: Tuple[str, ...]
    # base table name -> physical (UUID-suffixed) table names
    tables: Mapping[str, Tuple[str, ...]]
    # base table name -> trusted column names, in schema order
    columns: Mapping[str, Tuple[str, ...]]
    schemas: Tuple[RocpdSchema, ...]
    # input objects that are not part of the trusted schema and are not read
    ignored: Tuple[str, ...]


class RocpdSourceSet:
    """Check that validated sources can be combined into one data set."""

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.uuids: Set[str] = set()

    def add(self, source: RocpdSourceSchema) -> None:
        if self.version is None:
            self.version = source.version
        elif source.version != self.version:
            raise ValueError(
                "Multiple schema versions found: "
                f"{sorted({self.version, source.version})}"
            )
        duplicates = self.uuids.intersection(source.uuids)
        if duplicates:
            raise ValueError(
                f"Duplicate rocPD UUID across inputs: {sorted(duplicates)!r}"
            )
        self.uuids.update(source.uuids)


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


def column_list(columns: Sequence[str]) -> str:
    return ", ".join(quote_identifier(column) for column in columns)


def select_table(alias: Optional[str], table: str, columns: Sequence[str]) -> str:
    """Build a SELECT of the trusted *columns* of *table* in database *alias*."""

    source = qualified_identifier(alias, table) if alias else quote_identifier(table)
    return f"SELECT {column_list(columns)} FROM {source}"


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


def attach_readonly(connection: sqlite3.Connection, path: str, alias: str) -> str:
    """Attach *path* read-only and immutable, returning its resolved path.

    The path is bound as a parameter rather than interpolated into SQL.
    Immutable attachment ignores journals, so inputs with uncheckpointed
    journal content are rejected instead of silently losing committed rows.
    """

    quoted_alias = quote_identifier(alias)
    source = Path(os.fspath(path)).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Input database is not a regular file: {path!r}")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(f"{source}{suffix}")
        if sidecar.is_file() and sidecar.stat().st_size > 0:
            raise ValueError(
                f"Input database has an uncheckpointed SQLite journal: {sidecar!s}"
            )
    connection.execute(
        f"ATTACH DATABASE ? AS {quoted_alias}",
        (f"{source.as_uri()}?mode=ro&immutable=1",),
    )
    return str(source)


def detach_database(connection: sqlite3.Connection, alias: str) -> None:
    connection.execute(f"DETACH DATABASE {quote_identifier(alias)}")


def configure_untrusted_schema(connection: sqlite3.Connection) -> None:
    """Disable SQLite's use of application-defined functions in schemas."""

    # Unknown pragmas are ignored by older SQLite versions, while read-only URI
    # attachment and explicit table reads still provide the core boundary.
    connection.execute("PRAGMA trusted_schema = OFF")


def _schema_objects(connection: sqlite3.Connection, alias: str) -> SchemaObjects:
    rows = connection.execute(f"""
        SELECT type, name, sql
        FROM {quote_identifier(alias)}.sqlite_master
        WHERE name NOT GLOB 'sqlite_*'
        """).fetchall()
    return {name: (kind, sql) for kind, name, sql in rows}


def _table_columns(
    connection: sqlite3.Connection, alias: str, table: str
) -> Tuple[Tuple[str, int], ...]:
    """Return ``(name, hidden)`` per column; *hidden* is non-zero for
    generated and hidden columns."""

    target = f"{quote_identifier(alias)}.%s({quote_identifier(table)})"
    rows = connection.execute(f"PRAGMA {target % 'table_xinfo'}").fetchall()
    if rows:
        return tuple((row[1], row[6]) for row in rows)
    # table_xinfo was added after the oldest SQLite versions supported by
    # Python 3.6. Those versions cannot contain generated/hidden columns.
    rows = connection.execute(f"PRAGMA {target % 'table_info'}").fetchall()
    return tuple((row[1], 0) for row in rows)


def _check_table(
    connection: sqlite3.Connection,
    alias: str,
    objects: SchemaObjects,
    table: str,
    expected_columns: Sequence[str],
    source_path: str,
) -> None:
    """Require *table* to be a regular table with exactly the expected columns."""

    kind, sql = objects.get(table, (None, None))
    if kind is None:
        raise ValueError(f"Missing rocPD table {table!r} in {source_path!r}")
    if kind != "table" or not isinstance(sql, str) or not REGULAR_TABLE_SQL.match(sql):
        raise ValueError(
            f"rocPD object {table!r} is not a regular table in {source_path!r}"
        )
    columns = _table_columns(connection, alias, table)
    names = [name for name, _ in columns]
    if any(hidden for _, hidden in columns) or sorted(names) != sorted(expected_columns):
        raise ValueError(
            f"rocPD table {table!r} columns do not match the trusted schema in "
            f"{source_path!r}: {names!r}"
        )


def _metadata_tables(
    connection: sqlite3.Connection,
    alias: str,
    objects: SchemaObjects,
    source_path: str,
) -> List[Tuple[str, str]]:
    """Return ``(table, uuid)`` for each partition's metadata table.

    A table named like a metadata table but lacking the metadata columns is a
    custom object and is ignored with the other unsupported objects.
    """

    candidates = []
    for name in objects:
        match = METADATA_TABLE.fullmatch(name) if isinstance(name, str) else None
        if not match:
            continue
        try:
            _check_table(connection, alias, objects, name, METADATA_COLUMNS, source_path)
        except ValueError:
            continue
        candidates.append((name, match.group("uuid")))
    if not candidates:
        raise ValueError(f"Expected at least one rocPD metadata table in {source_path!r}")
    return sorted(candidates)


def _metadata_values(
    connection: sqlite3.Connection,
    alias: str,
    metadata_table: str,
    table_uuid: str,
    source_path: str,
) -> Tuple[str, str, str]:
    placeholders = ", ".join("?" for _ in REQUIRED_METADATA)
    # Read only required metadata, with one extra row to detect duplicates.
    # Additional metadata remains in the source and is copied by merge.
    rows = connection.execute(
        f"SELECT tag, value FROM {qualified_identifier(alias, metadata_table)} "
        f"WHERE tag COLLATE BINARY IN ({placeholders})",
        REQUIRED_METADATA,
    ).fetchmany(len(REQUIRED_METADATA) + 1)
    values: Dict[str, List[str]] = {}
    for tag, value in rows:
        values.setdefault(tag, []).append(value)

    if any(len(values.get(key, [])) != 1 for key in REQUIRED_METADATA):
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
    # Writers derive the table UUID from the GUID: rocprofv3 uses a UUIDv7 GUID
    # with hyphens turned into underscores, rocprofiler-systems an MD5 hex GUID.
    if uuid != "_" + guid.replace("-", "_"):
        raise ValueError(
            f"rocPD UUID metadata {uuid!r} does not match GUID {guid!r} in "
            f"{source_path!r}"
        )
    if not isinstance(version, str) or not SCHEMA_VERSION.fullmatch(version):
        raise ValueError(f"Invalid rocPD schema version in {source_path!r}: {version!r}")

    for key, part in zip(REQUIRED_METADATA[3:], version.split(".")):
        if str(values[key][0]) != part:
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


def _trusted_objects(
    schema: RocpdSchema,
) -> Tuple[Dict[str, Tuple[str, ...]], Set[str]]:
    """Return the trusted table columns and all object names of *schema*."""

    reference = sqlite3.connect(":memory:")
    try:
        configure_untrusted_schema(reference)
        reference.executescript(schema.tables)
        reference.executescript(schema.indexes)
        reference.executescript(schema.views)
        objects = _schema_objects(reference, "main")
        tables = {
            name: tuple(column for column, _ in _table_columns(reference, "main", name))
            for name, (kind, _) in objects.items()
            if kind == "table"
        }
        return tables, set(objects)
    finally:
        reference.close()


def inspect_attached_rocpd(
    connection: sqlite3.Connection,
    alias: str,
    source_path: str,
) -> RocpdSourceSchema:
    """Validate an attached database against its trusted versioned schema."""

    quote_identifier(alias)
    objects = _schema_objects(connection, alias)
    known: Set[str] = set()
    tables: Dict[str, List[str]] = {}
    columns: Dict[str, Tuple[str, ...]] = {}
    schemas = []
    uuids = []
    version = None

    for metadata_table, table_uuid in _metadata_tables(
        connection, alias, objects, source_path
    ):
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

        schema = RocpdSchema(uuid=uuid, guid=guid, version=partition_version)
        trusted_tables, trusted_names = _trusted_objects(schema)
        known.update(trusted_names)
        for table, expected in sorted(trusted_tables.items()):
            if not table.endswith(uuid):
                raise ValueError(f"Trusted rocPD table {table!r} lacks UUID {uuid!r}")
            _check_table(connection, alias, objects, table, expected, source_path)
            base = table[: -len(uuid)]
            tables.setdefault(base, []).append(table)
            columns[base] = expected

        uuids.append(uuid)
        schemas.append(schema)

    # Custom objects are never read or replayed. Canonical views and indexes
    # are rebuilt from trusted DDL, so their source definitions are ignored too.
    ignored = tuple(
        f"{objects[name][0]} {name!r}"
        for name in sorted(objects, key=repr)
        if name not in known
    )
    if ignored:
        print(
            f"Warning: ignoring unsupported objects in {source_path!r}: "
            f"{', '.join(ignored)}",
            file=sys.stderr,
        )

    return RocpdSourceSchema(
        path=source_path,
        alias=alias,
        version=version,
        uuids=tuple(uuids),
        tables={base: tuple(names) for base, names in tables.items()},
        columns=columns,
        schemas=tuple(schemas),
        ignored=ignored,
    )


def create_union_views(
    connection: sqlite3.Connection,
    tables: Mapping[str, Iterable[Tuple[Optional[str], str]]],
    columns: Mapping[str, Sequence[str]],
    *,
    temporary: bool = False,
) -> None:
    """Create base-table UNION views over validated ``(alias, table)`` pairs.

    An alias of ``None`` refers to a table in the connection's own database.
    """

    prefix = "CREATE TEMPORARY VIEW" if temporary else "CREATE VIEW"
    for base, sources in sorted(tables.items()):
        selects = [select_table(alias, table, columns[base]) for alias, table in sources]
        if not selects:
            raise ValueError(f"No tables available for rocPD view {base!r}")
        connection.execute(
            f"{prefix} {quote_identifier(base)} AS " + " UNION ALL ".join(selects)
        )
