#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""
rocpd.blob_prototype
====================
A self-contained prototype that demonstrates the complete self-describing blob
schema lifecycle end-to-end using only the Python standard library (sqlite3 +
ctypes).

Lifecycle
---------
1.  Create a fresh SQLite database file.
2.  Define the three blob-schema metadata tables and three domain tables in
    that database using plain SQL (no UUID suffix — the real rocpd system adds
    the suffix at write time and _create_temp_views creates the no-suffix alias).
3.  Define three C structs with ctypes (PcSampleExtdataV1,
    MemoryAllocExtdataV1, CounterSnapshotExtdataV1) using _pack_ = 1, exactly
    matching what a C++ profiler writer would do with #pragma pack(push, 1).
4.  Introspect each struct's _fields_ list to automatically insert rows into
    rocpd_info_blob_schema (one row per struct) and rocpd_info_blob_field (one
    row per field) — no hand-written byte offsets needed.
5.  Generate rows of deterministic fake data, serialise each ctypes struct
    instance to raw bytes via bytes(instance), and insert one rocpd_blob_event
    row per sample. Blob rows carry event_id and domain rows use the same
    event_id for correlation.
6.  Commit and close the connection.
7.  Re-open the same database file.
8.  Call setup_blob_views(conn) — defined in this file — which reads the three
    metadata tables, registers the rocpd_blob_field() SQLite scalar function
    backed by struct.unpack_from, and creates one TEMP VIEW per registered
    schema: gpu_pc_sample_decoded, memory_alloc_decoded, hw_counter_snap_decoded.
9.  Query each decoded view and print a formatted table showing actual field
    values (hw_id_chiplet, arb_state_issue_valu, …), not raw schema metadata.

Run:
    python3 -m rocpd.blob_prototype
or:
    python3 -c "from rocpd import blob_prototype; blob_prototype.main()"
"""

import ctypes
import os
import random
import sqlite3
import struct as _struct
import sys
import tempfile

# ---------------------------------------------------------------------------
# Blob-view machinery (inlined so the script runs with no external imports)
# ---------------------------------------------------------------------------

def _blob_struct_fmt(size: int, data_type: str, is_signed: int) -> str:
    """Return a struct.unpack_from format character for one blob field."""
    dt = data_type.lower().replace("_t", "").replace(" ", "")
    if dt in ("float", "f32", "fp32"):
        return "f"
    if dt in ("double", "f64", "fp64"):
        return "d"
    signed_map   = {1: "b", 2: "h", 4: "i", 8: "q"}
    unsigned_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
    return (signed_map if is_signed else unsigned_map).get(size, "B")


def setup_blob_views(conn: sqlite3.Connection) -> None:
    """Create a {source_table}_decoded TEMP VIEW for every blob schema
    registered in the database.

    Step 1 — read rocpd_info_blob_schema + rocpd_info_blob_field into a
             closure dict keyed by (schema_id, field_name).
    Step 2 — register rocpd_blob_field(blob, schema_id, field_name) as a
             connection-scoped SQLite scalar function backed by struct.unpack_from.
    Step 3 — CREATE TEMP VIEW {source_table}_decoded as a LEFT JOIN of the
             domain table with rocpd_blob_event on event_id, with every blob
             field projected as a decoded column.
    """
    try:
        schemas = conn.execute(
            "SELECT id, source_table, byte_order FROM rocpd_info_blob_schema"
        ).fetchall()
    except sqlite3.OperationalError:
        return
    if not schemas:
        return

    # Step 1: build field cache
    field_cache = {}
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

    # Step 2: register scalar function
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

    # Step 3: CREATE TEMP VIEW {source_table}_decoded
    for schema_id, source_table, _byte_order in schemas:
        try:
            domain_cols = [row[1] for row in conn.execute(
                f"PRAGMA table_info({source_table})"
            ).fetchall()]
        except sqlite3.OperationalError:
            domain_cols = []

        domain_select = ",\n    ".join(f"s.{col}" for col in domain_cols)

        try:
            field_names = [row[0] for row in conn.execute(
                "SELECT name FROM rocpd_info_blob_field "
                "WHERE schema_id = ? ORDER BY offset, id",
                (schema_id,),
            ).fetchall()]
        except sqlite3.OperationalError:
            field_names = []

        blob_select = ",\n    ".join(
            f"rocpd_blob_field(e.blob, {schema_id}, '{name}') AS {name}"
            for name in field_names
        )
        separator = ",\n    " if domain_select and blob_select else ""
        view_sql = (
            f"CREATE TEMP VIEW IF NOT EXISTS {source_table}_decoded AS\n"
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
            sys.stderr.write(
                f"setup_blob_views: could not create {source_table}_decoded: {exc}\n"
            )

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_DDL = """
PRAGMA foreign_keys = ON;

-- One row per struct version; source_table drives the TEMP VIEW name.
CREATE TABLE IF NOT EXISTS rocpd_info_blob_schema (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    source_table TEXT    NOT NULL,
    description  TEXT,
    byte_order   TEXT    NOT NULL DEFAULT 'little',
    alignment    INTEGER NOT NULL DEFAULT 1,
    struct_size  INTEGER NOT NULL,
    version      INTEGER NOT NULL DEFAULT 1
);

-- One row per field within a struct.
CREATE TABLE IF NOT EXISTS rocpd_info_blob_field (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_id   INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    offset      INTEGER NOT NULL,
    size        INTEGER NOT NULL,
    data_type   TEXT    NOT NULL,
    is_signed   INTEGER NOT NULL DEFAULT 0,
    description TEXT,
    FOREIGN KEY (schema_id) REFERENCES rocpd_info_blob_schema(id)
);

-- Shared event table (mirrors rocpd_event usage in full rocpd schema).
CREATE TABLE IF NOT EXISTS rocpd_event (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id INTEGER
);

-- One row per blob instance (the actual packed binary data).
CREATE TABLE IF NOT EXISTS rocpd_blob_event (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id  INTEGER NOT NULL,
    schema_id INTEGER NOT NULL,
    blob      BLOB    NOT NULL,
    FOREIGN KEY (event_id) REFERENCES rocpd_event(id),
    FOREIGN KEY (schema_id) REFERENCES rocpd_info_blob_schema(id)
);

-- Domain table used to correlate dispatches with samples through event_id.
CREATE TABLE IF NOT EXISTS kernel_dispatch (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dispatch_id INTEGER NOT NULL,
    event_id    INTEGER NOT NULL,
    kernel_name TEXT,
    FOREIGN KEY (event_id) REFERENCES rocpd_event(id)
);

-- Domain table 1: GPU PC sample
--   Architecture-independent columns live here; arch-specific data is in the blob.
CREATE TABLE IF NOT EXISTS gpu_pc_sample (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id           INTEGER NOT NULL,
    timestamp          INTEGER NOT NULL,
    dispatch_id        INTEGER NOT NULL,
    code_object_offset INTEGER,
    FOREIGN KEY (event_id) REFERENCES rocpd_event(id)
);

-- Domain table 2: Memory allocation event
CREATE TABLE IF NOT EXISTS memory_alloc (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   INTEGER NOT NULL,
    timestamp  INTEGER NOT NULL,
    alloc_size INTEGER NOT NULL,
    FOREIGN KEY (event_id) REFERENCES rocpd_event(id)
);

-- Domain table 3: HW performance counter snapshot
CREATE TABLE IF NOT EXISTS hw_counter_snap (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id  INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    kernel_id INTEGER NOT NULL,
    FOREIGN KEY (event_id) REFERENCES rocpd_event(id)
);
"""

# ---------------------------------------------------------------------------
# ctypes struct definitions — _pack_ = 1 matches #pragma pack(push, 1)
# ---------------------------------------------------------------------------

class PcSampleExtdataV1(ctypes.LittleEndianStructure):
    """Architecture-specific fields for a GPU PC sample.

    hw_id fields are present for both host-trap and stochastic sampling.
    arb_state fields are populated only for stochastic samples (zero otherwise).
    """
    _pack_ = 1
    _fields_ = [
        # Hardware component identifiers decoded from the HW_ID register
        ("hw_id_chiplet",          ctypes.c_uint8),
        ("hw_id_wave_id",          ctypes.c_uint8),
        ("hw_id_simd_id",          ctypes.c_uint8),
        ("hw_id_pipe_id",          ctypes.c_uint8),
        ("hw_id_cu_or_wgp_id",     ctypes.c_uint8),
        ("hw_id_shader_array_id",  ctypes.c_uint8),
        ("hw_id_shader_engine_id", ctypes.c_uint8),
        ("hw_id_vm_id",            ctypes.c_uint8),
        # Arbiter state snapshot (1 = active this cycle, 0 = inactive)
        ("dual_issue_valu",            ctypes.c_uint8),
        ("arb_state_issue_valu",       ctypes.c_uint8),
        ("arb_state_issue_matrix",     ctypes.c_uint8),
        ("arb_state_issue_lds",        ctypes.c_uint8),
        ("arb_state_issue_lds_direct", ctypes.c_uint8),
        ("arb_state_issue_scalar",     ctypes.c_uint8),
        ("arb_state_issue_vmem_tex",   ctypes.c_uint8),
        ("arb_state_issue_flat",       ctypes.c_uint8),
        ("arb_state_issue_exp",        ctypes.c_uint8),
        ("arb_state_issue_misc",       ctypes.c_uint8),
        ("arb_state_issue_brmsg",      ctypes.c_uint8),
        ("arb_state_stall_valu",       ctypes.c_uint8),
        ("arb_state_stall_matrix",     ctypes.c_uint8),
        ("arb_state_stall_lds",        ctypes.c_uint8),
        ("arb_state_stall_lds_direct", ctypes.c_uint8),
        ("arb_state_stall_scalar",     ctypes.c_uint8),
        ("arb_state_stall_vmem_tex",   ctypes.c_uint8),
        ("arb_state_stall_flat",       ctypes.c_uint8),
        ("arb_state_stall_exp",        ctypes.c_uint8),
        ("arb_state_stall_misc",       ctypes.c_uint8),
        ("arb_state_stall_brmsg",      ctypes.c_uint8),
    ]


class MemoryAllocExtdataV1(ctypes.LittleEndianStructure):
    """Extended fields for a memory allocation event."""
    _pack_ = 1
    _fields_ = [
        ("virtual_address",  ctypes.c_uint64),
        ("physical_address", ctypes.c_uint64),
        ("numa_node",        ctypes.c_uint32),
        ("agent_flags",      ctypes.c_uint32),
        ("page_size",        ctypes.c_uint32),
        ("pool_type",        ctypes.c_uint8),
        ("is_managed",       ctypes.c_uint8),
        ("is_coherent",      ctypes.c_uint8),
    ]


class CounterSnapshotExtdataV1(ctypes.LittleEndianStructure):
    """Raw HW performance counter values captured alongside a kernel dispatch."""
    _pack_ = 1
    _fields_ = [
        ("sq_waves",          ctypes.c_uint64),
        ("sq_insts_valu",     ctypes.c_uint64),
        ("sq_insts_salu",     ctypes.c_uint64),
        ("sq_insts_vmem",     ctypes.c_uint64),
        ("sq_insts_lds",      ctypes.c_uint64),
        ("tcp_utcl1_request", ctypes.c_uint64),
        ("l2_rd_req",         ctypes.c_uint64),
        ("l2_wr_req",         ctypes.c_uint64),
        ("grbm_count",        ctypes.c_uint32),
        ("grbm_gui_active",   ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Helpers shared by the write phase
# ---------------------------------------------------------------------------

def _ctype_meta(ct) -> tuple:
    """Return (size_bytes, data_type_str, is_signed) for a ctypes scalar type.

    Handles the naming conventions used by ctypes for integer and float types
    so that _blob_struct_fmt() in importer.py can round-trip correctly.
    """
    name = ct.__name__           # e.g. 'c_uint32', 'c_int8', 'c_float'
    size = ctypes.sizeof(ct)
    if "float" in name:
        return size, "float", 0
    if "double" in name:
        return size, "double", 0
    signed = not (
        "uint" in name or name in ("c_ubyte", "c_ushort", "c_ulong", "c_ulonglong")
    )
    tag = {
        (1, False): "uint8_t",  (2, False): "uint16_t",
        (4, False): "uint32_t", (8, False): "uint64_t",
        (1, True):  "int8_t",   (2, True):  "int16_t",
        (4, True):  "int32_t",  (8, True):  "int64_t",
    }.get((size, signed), f"uint{size * 8}_t")
    return size, tag, int(signed)


def register_struct(conn: sqlite3.Connection,
                    struct_cls,
                    source_table: str,
                    description: str = "") -> int:
    """Introspect a ctypes Structure and write its layout into the blob metadata
    tables.  Returns the new schema_id.

    This is the Python equivalent of the C++ register_blob_schema() lambda in
    generateRocpd.cpp.  A real profiler writer calls this once at collection
    start, then calls insert_blob_event() once per sample.

    Parameters
    ----------
    conn         : open sqlite3 connection to the rocpd database
    struct_cls   : a ctypes.Structure subclass with _pack_ = 1
    source_table : name of the domain table whose rows correlate by event_id
                   (drives the TEMP VIEW name: {source_table}_decoded)
    description  : optional human-readable description of the struct
    """
    schema_size = ctypes.sizeof(struct_cls)
    cur = conn.execute(
        """
        INSERT INTO rocpd_info_blob_schema
               (name, source_table, description, byte_order, alignment, struct_size, version)
        VALUES (?, ?, ?, 'little', 1, ?, 1)
        """,
        (struct_cls.__name__, source_table,
         description or (struct_cls.__doc__ or "").split("\n")[0].strip(),
         schema_size),
    )
    schema_id = cur.lastrowid

    for field_name, field_type in struct_cls._fields_:
        # ctypes exposes offset and size via the descriptor on the class
        offset = getattr(struct_cls, field_name).offset
        size, data_type, is_signed = _ctype_meta(field_type)
        conn.execute(
            """
            INSERT INTO rocpd_info_blob_field
                   (schema_id, name, offset, size, data_type, is_signed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (schema_id, field_name, offset, size, data_type, is_signed),
        )
    conn.commit()
    return schema_id


def insert_blob_event(conn: sqlite3.Connection,
                      event_id: int,
                      schema_id: int,
                      struct_instance) -> int:
    """Serialise a ctypes struct instance to raw bytes and insert one
    rocpd_blob_event row.  Returns the new blob_event row id.

    Correlation is done via event_id (same pattern as rocpd_arg and
    rocpd_pmc_event in rocpd): the domain row and blob row both share event_id.
    """
    blob = bytes(struct_instance)
    cur = conn.execute(
        "INSERT INTO rocpd_blob_event (event_id, schema_id, blob) VALUES (?, ?, ?)",
        (event_id, schema_id, sqlite3.Binary(blob)),
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_table(title: str, rows: list, columns: list) -> None:
    if not rows:
        print(f"\n{title}\n  (no rows)\n")
        return
    widths = [
        max(len(str(c)), max(len(str(r[i])) for r in rows))
        for i, c in enumerate(columns)
    ]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(f"\n{title}")
    print(sep)
    print(fmt.format(*columns))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Phase 1 — write:  create DB, register schemas, insert fake sample data
# ---------------------------------------------------------------------------

def _write_database(db_path: str) -> None:
    """Create the database, register three structs, insert fake rows."""
    rng = random.Random(42)          # fixed seed → deterministic output

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)

    # ------------------------------------------------------------------
    # Step A: register each struct's layout in the metadata tables.
    # register_struct() introspects _fields_ — no hand-written offsets.
    # ------------------------------------------------------------------
    pc_schema_id  = register_struct(conn, PcSampleExtdataV1,       "gpu_pc_sample")
    mem_schema_id = register_struct(conn, MemoryAllocExtdataV1,    "memory_alloc")
    ctr_schema_id = register_struct(conn, CounterSnapshotExtdataV1,"hw_counter_snap")

    print(f"  Struct sizes  :"
          f"  PcSampleExtdataV1={ctypes.sizeof(PcSampleExtdataV1)} B"
          f"  MemoryAllocExtdataV1={ctypes.sizeof(MemoryAllocExtdataV1)} B"
          f"  CounterSnapshotExtdataV1={ctypes.sizeof(CounterSnapshotExtdataV1)} B")
    print(f"  Schema IDs    :  pc={pc_schema_id}  mem={mem_schema_id}  ctr={ctr_schema_id}")
    print(f"  Fields stored :  pc={len(PcSampleExtdataV1._fields_)}"
          f"  mem={len(MemoryAllocExtdataV1._fields_)}"
          f"  ctr={len(CounterSnapshotExtdataV1._fields_)}")

    # ------------------------------------------------------------------
    # Step B: create dispatches that are correlated through event_id.
    # ------------------------------------------------------------------
    dispatch_event_ids = {}
    for dispatch_id in range(3):
        cur = conn.execute(
            "INSERT INTO rocpd_event (correlation_id) VALUES (?)",
            (10_000 + dispatch_id,),
        )
        event_id = cur.lastrowid
        dispatch_event_ids[dispatch_id] = event_id
        conn.execute(
            "INSERT INTO kernel_dispatch (dispatch_id, event_id, kernel_name) "
            "VALUES (?, ?, ?)",
            (dispatch_id, event_id, f"vector_add_{dispatch_id}"),
        )

    # Step C: generate fake GPU PC sample rows.
    # Each sample gets its OWN event_id for blob correlation (1:1 with
    # rocpd_blob_event).  Samples are linked to their parent dispatch via
    # dispatch_id, not via a shared event_id.
    # ------------------------------------------------------------------
    N_PC = 9  # 3 samples per dispatch
    for i in range(N_PC):
        # Simulate a stochastic sample: hw_id always set, arb_state random
        ext = PcSampleExtdataV1(
            hw_id_chiplet          = rng.randint(0, 1),
            hw_id_wave_id          = rng.randint(0, 7),
            hw_id_simd_id          = rng.randint(0, 3),
            hw_id_pipe_id          = rng.randint(0, 1),
            hw_id_cu_or_wgp_id     = rng.randint(0, 15),
            hw_id_shader_array_id  = rng.randint(0, 3),
            hw_id_shader_engine_id = rng.randint(0, 3),
            hw_id_vm_id            = rng.randint(0, 15),
            arb_state_issue_valu   = rng.randint(0, 1),
            arb_state_issue_matrix = rng.randint(0, 1),
            arb_state_issue_lds    = rng.randint(0, 1),
            arb_state_issue_scalar = rng.randint(0, 1),
            arb_state_stall_valu   = rng.randint(0, 1),
            arb_state_stall_matrix = rng.randint(0, 1),
            arb_state_stall_lds    = rng.randint(0, 1),
            arb_state_stall_scalar = rng.randint(0, 1),
        )
        dispatch_id = i // 3  # 3 samples per dispatch
        pc_event_id = conn.execute(
            "INSERT INTO rocpd_event (correlation_id) VALUES (?)",
            (20_000 + i,),
        ).lastrowid
        insert_blob_event(conn, pc_event_id, pc_schema_id, ext)
        conn.execute(
            "INSERT INTO gpu_pc_sample"
            "    (event_id, timestamp, dispatch_id, code_object_offset)"
            " VALUES (?, ?, ?, ?)",
            (pc_event_id, 1_000_000 * (i + 1), dispatch_id, 0x1A00 + i * 4),
        )

    # ------------------------------------------------------------------
    # Step D: generate fake memory allocation events.
    # ------------------------------------------------------------------
    for i in range(4):
        mem_event_id = conn.execute(
            "INSERT INTO rocpd_event (correlation_id) VALUES (?)",
            (20_000 + i,),
        ).lastrowid
        ext = MemoryAllocExtdataV1(
            virtual_address  = 0x7F00_0000_0000 + i * 0x10000,
            physical_address = 0x8000_0000 + i * 0x10000,
            numa_node        = rng.randint(0, 3),
            agent_flags      = rng.choice([0x1, 0x3, 0x7]),
            page_size        = 4096,
            pool_type        = rng.randint(0, 2),
            is_managed       = rng.randint(0, 1),
            is_coherent      = rng.randint(0, 1),
        )
        insert_blob_event(conn, mem_event_id, mem_schema_id, ext)
        conn.execute(
            "INSERT INTO memory_alloc (event_id, timestamp, alloc_size)"
            " VALUES (?, ?, ?)",
            (mem_event_id, 2_000_000 * (i + 1), rng.choice([4096, 65536, 1 << 20])),
        )

    # ------------------------------------------------------------------
    # Step E: generate fake HW counter snapshots.
    # ------------------------------------------------------------------
    for i in range(5):
        ctr_event_id = conn.execute(
            "INSERT INTO rocpd_event (correlation_id) VALUES (?)",
            (30_000 + i,),
        ).lastrowid
        ext = CounterSnapshotExtdataV1(
            sq_waves          = rng.randint(1_000,   65_535),
            sq_insts_valu     = rng.randint(100_000, 10_000_000),
            sq_insts_salu     = rng.randint(10_000,  1_000_000),
            sq_insts_vmem     = rng.randint(5_000,   500_000),
            sq_insts_lds      = rng.randint(1_000,   100_000),
            tcp_utcl1_request = rng.randint(1_000,   50_000),
            l2_rd_req         = rng.randint(500,     20_000),
            l2_wr_req         = rng.randint(200,     10_000),
            grbm_count        = rng.randint(100_000, 1_000_000),
            grbm_gui_active   = rng.randint(80_000,  900_000),
        )
        insert_blob_event(conn, ctr_event_id, ctr_schema_id, ext)
        conn.execute(
            "INSERT INTO hw_counter_snap (event_id, timestamp, kernel_id)"
            " VALUES (?, ?, ?)",
            (ctr_event_id, 3_000_000 * (i + 1), i + 1),
        )

    conn.commit()
    conn.close()
    print(f"  Rows inserted :  3 kernel_dispatch  {N_PC} gpu_pc_sample  4 memory_alloc  5 hw_counter_snap")
    print(f"  event rows    :  {3 + N_PC + 4 + 5} rocpd_event")
    print(f"  blob_event rows: {N_PC + 4 + 5}  (one per sample, all in rocpd_blob_event)")


# ---------------------------------------------------------------------------
# Phase 2 — read:  reopen, call setup_blob_views, query decoded TEMP VIEWs
# ---------------------------------------------------------------------------

def _read_database(db_path: str) -> None:
    """Re-open the database, create the decoded views, query and display them."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row

    # setup_blob_views does three things (see rocpd/importer.py):
    #   1. Reads rocpd_info_blob_schema → one entry per struct
    #   2. Reads rocpd_info_blob_field  → field dict, cached in a closure
    #   3. Registers rocpd_blob_field(blob, schema_id, field_name) scalar fn
    #   4. CREATE TEMP VIEW {source_table}_decoded for each schema
    setup_blob_views(conn)

    # ------------------------------------------------------------------
    # Query gpu_pc_sample_decoded
    # Equivalent SQL (generated dynamically by setup_blob_views):
    #
    #   SELECT s.id, s.event_id, s.timestamp, s.dispatch_id, s.code_object_offset,
    #          rocpd_blob_field(e.blob, 1, 'hw_id_chiplet')    AS hw_id_chiplet,
    #          rocpd_blob_field(e.blob, 1, 'hw_id_wave_id')    AS hw_id_wave_id,
    #          ...
    #   FROM gpu_pc_sample s
    #   LEFT JOIN rocpd_blob_event e ON e.event_id = s.event_id
    #                               AND e.schema_id = 1
    # ------------------------------------------------------------------
    cols_pc = [
        "id", "event_id", "dispatch_id", "code_object_offset",
        "hw_id_chiplet", "hw_id_wave_id", "hw_id_simd_id",
        "hw_id_cu_or_wgp_id", "hw_id_shader_engine_id",
        "arb_state_issue_valu", "arb_state_stall_valu",
        "arb_state_issue_matrix", "arb_state_stall_matrix",
        "arb_state_issue_scalar", "arb_state_stall_scalar",
    ]
    rows_pc = conn.execute(
        f"SELECT {', '.join(cols_pc)} FROM gpu_pc_sample_decoded"
    ).fetchall()
    _print_table(
        "gpu_pc_sample_decoded  —  PC sampling: decoded arch-specific fields",
        [tuple(r) for r in rows_pc], cols_pc,
    )

    # ------------------------------------------------------------------
    # Demonstrate kernel_dispatch -> pc samples correlation by event_id.
    # ------------------------------------------------------------------
    target_event_id = conn.execute(
        "SELECT event_id FROM kernel_dispatch WHERE dispatch_id = 1"
    ).fetchone()[0]
    cols_link = [
        "kernel_dispatch_id", "dispatch_id", "event_id",
        "pc_sample_id", "code_object_offset",
        "hw_id_chiplet", "hw_id_wave_id", "arb_state_issue_valu",
    ]
    rows_link = conn.execute(
        "SELECT "
        "    kd.id AS kernel_dispatch_id, kd.dispatch_id, kd.event_id, "
        "    psd.id AS pc_sample_id, psd.code_object_offset, "
        "    psd.hw_id_chiplet, psd.hw_id_wave_id, psd.arb_state_issue_valu "
        "FROM kernel_dispatch kd "
        "JOIN gpu_pc_sample_decoded psd ON psd.dispatch_id = kd.dispatch_id "
        "WHERE kd.event_id = ? "
        "ORDER BY psd.id",
        (target_event_id,),
    ).fetchall()
    _print_table(
        f"kernel_dispatch (event_id={target_event_id}) -> gpu_pc_sample_decoded via dispatch_id",
        [tuple(r) for r in rows_link],
        cols_link,
    )

    # ------------------------------------------------------------------
    # Query memory_alloc_decoded
    # ------------------------------------------------------------------
    cols_mem = [
        "id", "alloc_size",
        "virtual_address", "physical_address",
        "numa_node", "agent_flags", "page_size",
        "pool_type", "is_managed", "is_coherent",
    ]
    rows_mem = conn.execute(
        f"SELECT {', '.join(cols_mem)} FROM memory_alloc_decoded"
    ).fetchall()
    _print_table(
        "memory_alloc_decoded  —  memory allocation: decoded extended fields",
        [tuple(r) for r in rows_mem], cols_mem,
    )

    # ------------------------------------------------------------------
    # Query hw_counter_snap_decoded
    # ------------------------------------------------------------------
    cols_ctr = [
        "id", "kernel_id",
        "sq_waves", "sq_insts_valu", "sq_insts_salu",
        "sq_insts_vmem", "sq_insts_lds",
        "l2_rd_req", "l2_wr_req",
        "grbm_count", "grbm_gui_active",
    ]
    rows_ctr = conn.execute(
        f"SELECT {', '.join(cols_ctr)} FROM hw_counter_snap_decoded"
    ).fetchall()
    _print_table(
        "hw_counter_snap_decoded  —  HW counter snapshot: decoded counter values",
        [tuple(r) for r in rows_ctr], cols_ctr,
    )

    conn.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    db_path = os.path.join(tempfile.gettempdir(), "rocpd_blob_prototype.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    print("=" * 68)
    print("  rocpd blob-schema prototype")
    print("=" * 68)

    print(f"\n[1/4]  Creating database: {db_path}")

    print(f"\n[2/4]  Registering 3 structs and inserting fake sample data ...")
    print(f"       (register_struct introspects ctypes._fields_ → no hand-written offsets)")
    _write_database(db_path)
    print(f"       Database written and closed.\n")

    print(f"[3/4]  Re-opening database and calling setup_blob_views() ...")
    print(f"       setup_blob_views reads rocpd_info_blob_schema + rocpd_info_blob_field,")
    print(f"       registers rocpd_blob_field() scalar function, and creates:")
    print(f"         •  gpu_pc_sample_decoded")
    print(f"         •  memory_alloc_decoded")
    print(f"         •  hw_counter_snap_decoded")

    print(f"\n[4/4]  Querying decoded views  (actual field values, not schema metadata) ...")
    _read_database(db_path)

    print(f"Database saved at: {db_path}")
    print(f"Inspect with:  sqlite3 {db_path}")


if __name__ == "__main__":
    main()
