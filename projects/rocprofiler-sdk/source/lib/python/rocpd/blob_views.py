###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""
rocpd.blob_views
================
Generic decoded-view machinery for the self-describing blob schema.

This module is intentionally dependency-free within the rocpd package so that
it can be imported directly (without the compiled libpyrocpd extension) by
standalone tools and prototypes.

Public API
----------
setup_blob_views(conn)
    Create one ``{source_table}_decoded`` TEMP VIEW per row in
    rocpd_info_blob_schema.  Safe to call on any sqlite3.Connection that has
    the three blob-schema tables present.
"""

import sqlite3
import struct as _struct
from typing import Dict, Tuple

__all__ = ["setup_blob_views"]


def _blob_struct_fmt(size: int, data_type: str, is_signed: int) -> str:
    """Return a ``struct.unpack_from`` format character for one blob field.

    Handles both the C++ writer convention (``"uint32_t"``, ``"uint8_t"``) and
    the shorter form (``"uint32"``, ``"UINT8"``).
    """
    dt = data_type.lower().replace("_t", "").replace(" ", "")
    if dt in ("float", "f32", "fp32"):
        return "f"
    if dt in ("double", "f64", "fp64"):
        return "d"
    signed_map   = {1: "b", 2: "h", 4: "i", 8: "q"}
    unsigned_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
    return (signed_map if is_signed else unsigned_map).get(size, "B")


def setup_blob_views(conn: sqlite3.Connection) -> None:
    """Create a ``{source_table}_decoded`` TEMP VIEW for every blob schema
    registered in the database.

    How it works
    ------------
    **Step 1 – Load metadata into a closure cache.**

    Reads ``rocpd_info_blob_schema`` (id, source_table, byte_order) and for
    each schema reads all rows from ``rocpd_info_blob_field``
    (name, offset, size, data_type, is_signed).  Every field is stored in a
    dict keyed by ``(schema_id, field_name)`` so that the scalar function
    registered in Step 2 never queries the database at decode time.

    **Step 2 – Register ``rocpd_blob_field(blob, schema_id, field_name)``.**

    A single connection-scoped SQLite scalar function backed by
    ``struct.unpack_from``.  It is registered once and reused by every decoded
    view.  The closure over the field cache means each call is a dict lookup +
    a single ``struct.unpack_from`` call — no SQL, no I/O.

    **Step 3 – CREATE TEMP VIEW ``{source_table}_decoded``.**

    For each schema, the view is a ``LEFT JOIN`` of the domain table
    (``source_table``) with ``rocpd_blob_event`` on ``event_id``.
    Domain columns are projected directly; each
    blob field becomes ``rocpd_blob_field(e.blob, schema_id, 'field') AS field``.

    The view is created with ``IF NOT EXISTS`` so calling this function more
    than once on the same connection is safe.  If the blob-schema tables are
    absent (older database) the function returns silently.

    Parameters
    ----------
    conn : open ``sqlite3.Connection``
        Any connection that has ``rocpd_info_blob_schema``,
        ``rocpd_info_blob_field``, and ``rocpd_blob_event`` present as tables
        or views (with or without the rocpd UUID suffix).
    """
    # -----------------------------------------------------------------------
    # Guard: older databases that pre-date the blob-schema feature have none
    # of these tables; return without error.
    # -----------------------------------------------------------------------
    try:
        schemas = conn.execute(
            "SELECT id, source_table, byte_order FROM rocpd_info_blob_schema"
        ).fetchall()
    except sqlite3.OperationalError:
        return

    if not schemas:
        return

    # -----------------------------------------------------------------------
    # Step 1: build field_cache
    # field_cache[(schema_id, field_name)] = (byte_offset, endian_char, fmt_char)
    # -----------------------------------------------------------------------
    field_cache: Dict[Tuple[int, str], Tuple[int, str, str]] = {}

    for schema_id, source_table, byte_order in schemas:
        endian = "<" if (byte_order or "little").startswith("l") else ">"
        try:
            fields = conn.execute(
                "SELECT name, offset, size, data_type, is_signed "
                "FROM rocpd_info_blob_field "
                "WHERE schema_id = ? ORDER BY offset, id",
                (schema_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            fields = []

        for name, offset, size, data_type, is_signed in fields:
            fmt = _blob_struct_fmt(size, data_type or "uint8_t", is_signed or 0)
            field_cache[(schema_id, name)] = (offset, endian, fmt)

    # -----------------------------------------------------------------------
    # Step 2: register rocpd_blob_field(blob, schema_id, field_name)
    # -----------------------------------------------------------------------
    def _rocpd_blob_field(blob: bytes, schema_id: int, field_name: str):
        if blob is None:
            return None
        entry = field_cache.get((schema_id, field_name))
        if entry is None:
            return None
        offset, endian, fmt = entry
        try:
            return _struct.unpack_from(endian + fmt, blob, offset)[0]
        except _struct.error:
            return None

    conn.create_function("rocpd_blob_field", 3, _rocpd_blob_field, deterministic=True)

    # -----------------------------------------------------------------------
    # Step 3: CREATE TEMP VIEW {source_table}_decoded for each schema
    # -----------------------------------------------------------------------
    for schema_id, source_table, _byte_order in schemas:
        try:
            domain_cols = [
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({source_table})"
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            domain_cols = []

        domain_select = ",\n    ".join(f"s.{col}" for col in domain_cols)

        try:
            field_names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM rocpd_info_blob_field "
                    "WHERE schema_id = ? ORDER BY offset, id",
                    (schema_id,),
                ).fetchall()
            ]
        except sqlite3.OperationalError:
            field_names = []

        blob_select = ",\n    ".join(
            f"rocpd_blob_field(e.blob, {schema_id}, '{name}') AS {name}"
            for name in field_names
        )

        separator = ",\n    " if domain_select and blob_select else ""
        view_name = f"{source_table}_decoded"
        view_sql = (
            f"CREATE TEMP VIEW IF NOT EXISTS {view_name} AS\n"
            f"SELECT\n"
            f"    {domain_select}{separator}\n"
            f"    {blob_select}\n"
            f"FROM {source_table} s\n"
            f"LEFT JOIN rocpd_blob_event e ON "
            f"e.event_id = s.event_id AND e.schema_id = {schema_id}"
        )
        try:
            conn.execute(view_sql)
        except sqlite3.OperationalError as exc:
            import sys
            sys.stderr.write(
                f"setup_blob_views: could not create {view_name}: {exc}\n"
            )
