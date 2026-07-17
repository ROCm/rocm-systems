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
import stat
from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .schema import RocpdSchema
from . import libpyrocpd

__all__ = [
    "RocpdImportData",
    "execute_statement",
    "setup_blob_views",
    "setup_isa_decode_views",
]


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
    _table_info = _create_temp_views(_connection, _input)
    # When _input is empty (e.g. a missing file caused package.py to return []),
    # _create_temp_views creates no TEMP VIEWs.  Calling _create_meta_views
    # with uuid="" would then produce circularly-defined TEMP VIEWs because
    # {{uuid}} renders as "" making every view reference its own name.  Skip it
    # entirely; there is no data to set up views for anyway.
    if _input:
        _create_meta_views(_connection)
    return (_connection, _input, _table_info)


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
                _connection, _filenames, _table_info = internal_init(
                    input, dbname, skip_auto_merge, automerge_limit
                )
                self.table_info = _table_info
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


def _quote_identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def _quote_sql_literal(value: str) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(f"Invalid SQL text value: {value!r}")
    return "'" + value.replace("'", "''") + "'"


def _missing_object_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "no such table" in message or "no such view" in message


def _table_columns(conn, table_name: str):
    try:
        return [
            row[1]
            for row in conn.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()
        ]
    except sqlite3.OperationalError as error:
        if _missing_object_error(error):
            return []
        raise


def _blob_struct_fmt(size: int, data_type: str, is_signed: int) -> str:
    """Validate a blob field descriptor and return its ``struct`` format."""
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Blob field size must be positive, got {size!r}")
    if is_signed not in (0, 1, False, True):
        raise ValueError(f"Blob field signedness must be 0 or 1, got {is_signed!r}")
    if not isinstance(data_type, str) or not data_type.strip():
        raise ValueError("Blob field data_type must be a non-empty string")

    data_type = data_type.lower().replace("_t", "").replace(" ", "")
    if data_type in ("float", "f32", "fp32"):
        if size != 4:
            raise ValueError(f"{data_type} fields must be 4 bytes, got {size}")
        return "f"
    if data_type in ("double", "f64", "fp64"):
        if size != 8:
            raise ValueError(f"{data_type} fields must be 8 bytes, got {size}")
        return "d"
    if data_type in ("bool", "boolean"):
        if size != 1 or bool(is_signed):
            raise ValueError("Boolean blob fields must be one unsigned byte")
        return "?"

    signed_map = {1: "b", 2: "h", 4: "i", 8: "q"}
    unsigned_map = {1: "B", 2: "H", 4: "I", 8: "Q"}
    integer_types = {
        "char",
        "byte",
        "int",
        "integer",
        "short",
        "long",
        "longlong",
        "uint",
        "unsigned",
        "unsignedint",
        "unsignedshort",
        "unsignedlong",
        "unsignedlonglong",
        f"int{size * 8}",
        f"i{size * 8}",
        f"uint{size * 8}",
        f"u{size * 8}",
    }
    if data_type not in integer_types or size not in signed_map:
        raise ValueError(
            f"Unsupported blob field descriptor data_type={data_type!r}, size={size}"
        )

    explicitly_unsigned = data_type.startswith("u") or data_type.startswith("unsigned")
    explicitly_signed = data_type.startswith("int") or data_type.startswith("i")
    if explicitly_unsigned and bool(is_signed):
        raise ValueError(f"Unsigned type {data_type!r} cannot be marked signed")
    if explicitly_signed and not bool(is_signed):
        raise ValueError(f"Signed type {data_type!r} must be marked signed")
    return (signed_map if bool(is_signed) else unsigned_map)[size]


def setup_isa_decode_views(conn):
    """Register lazy ISA decode functions and enrich PC-sample decoded TEMP VIEWs.

    ``setup_blob_views`` remains the generic self-describing blob decoder. This
    helper layers PC-sampling-specific ISA disassembly and enum-name expansion on
    top of its generic ``rocpd_gpu_pc_sample_decoded`` output when available.
    """
    try:
        code_objects = conn.execute("""
            SELECT
            guid,
                id,
                uri,
                load_delta,
                load_size,
                COALESCE(
                    storage_type,
                    CASE JSON_EXTRACT(extdata, '$.storage_type')
                        WHEN 1 THEN 'FILE'
                        WHEN 2 THEN 'MEMORY'
                        ELSE NULL
                    END
                ) AS storage_type,
                JSON_EXTRACT(extdata, '$.memory_size') AS memory_size
            FROM rocpd_info_code_object
            """).fetchall()
    except sqlite3.OperationalError as error:
        if _missing_object_error(error) or "no such column" in str(error).lower():
            code_objects = []
        else:
            raise

    code_object_metadata = {}
    for row in code_objects:
        guid, code_object_id, uri, load_delta, load_size, storage_type, memory_size = row
        key = (str(guid), int(code_object_id))
        metadata = {
            "uri": uri,
            "load_delta": int(load_delta or 0),
            "load_size": int(load_size or 0),
            "storage_type": storage_type,
            "memory_size": int(memory_size or 0),
        }
        previous = code_object_metadata.setdefault(key, metadata)
        if previous != metadata:
            raise ValueError(f"Conflicting code-object metadata for {key!r}")

    def _int_param(values, name):
        raw_values = values.get(name, [])
        if len(raw_values) > 1:
            raise ValueError(f"Duplicate {name!r} parameter in code-object URI")
        value = raw_values[0] if raw_values else None
        if value is None:
            return None
        parsed = int(value, 0)
        if parsed < 0:
            raise ValueError(f"Negative {name!r} in code-object URI")
        return parsed

    def _existing_path(path):
        if os.path.exists(path):
            return path
        home_prefix = os.path.expanduser("~").rstrip(os.sep) + os.sep
        for source_prefix, replacement_prefix in (
            ("/dockerx/", home_prefix),
            (home_prefix, "/dockerx/"),
        ):
            if path.startswith(source_prefix):
                candidate = replacement_prefix + path[len(source_prefix) :]
                if os.path.exists(candidate):
                    return candidate
        return None

    max_code_object_bytes = 512 * 1024 * 1024

    def _validated_file_uri(uri, read_bytes=False, expected_size=0):
        parsed = urlparse(uri)
        if parsed.scheme.lower() != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("Only local file:// code-object URIs are supported")
        if parsed.query:
            raise ValueError("Code-object URI query parameters are unsupported")

        params = parse_qs(parsed.fragment, keep_blank_values=True, strict_parsing=True)
        if set(params) - {"offset", "size"}:
            raise ValueError("Unsupported code-object URI fragment parameter")
        offset = _int_param(params, "offset") or 0
        size = _int_param(params, "size")
        path = _existing_path(unquote(parsed.path))
        if not path:
            raise FileNotFoundError(unquote(parsed.path))

        file_stat = os.stat(path)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Code-object path is not a regular file: {path}")
        if offset > file_stat.st_size:
            raise ValueError("Code-object URI offset is beyond end of file")
        if size is None:
            size = file_stat.st_size - offset
        if size <= 0 or size > max_code_object_bytes:
            raise ValueError(f"Invalid code-object byte count: {size}")
        if offset + size > file_stat.st_size:
            raise ValueError("Code-object URI range is beyond end of file")
        if expected_size and size != expected_size:
            raise ValueError(
                f"Code-object URI size {size} does not match metadata size {expected_size}"
            )

        if read_bytes:
            with open(path, "rb") as file:
                file.seek(offset)
                data = file.read(size)
            if len(data) != size:
                raise ValueError("Code-object file was truncated while being read")
            return data

        normalized = Path(path).resolve().as_uri()
        return f"{normalized}#offset={offset}&size={size}"

    decoder_cache = OrderedDict()
    decode_cache = OrderedDict()
    max_decoders = 64
    max_decoded_instructions = 4096

    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _get_decoder(key):
        decoder = decoder_cache.get(key)
        if decoder is not None:
            decoder_cache.move_to_end(key)
            return decoder

        metadata = code_object_metadata.get(key)
        if not metadata or not metadata["uri"]:
            return None
        try:
            decoder = libpyrocpd.isa_decoder()
            if metadata["storage_type"] == "FILE":
                uri = _validated_file_uri(metadata["uri"])
                decoder.add_code_object_file(
                    uri,
                    key[1],
                    metadata["load_delta"],
                    metadata["load_size"],
                )
            elif metadata["storage_type"] == "MEMORY":
                # Limitation: lazy post-processing decode can only read a
                # memory-backed code object when its URI resolves to a real
                # on-disk file. A code object loaded purely from process
                # memory has a non-file URI (e.g. ``memory://...``) and is
                # not snapshotted to disk during PC sampling (unlike ATT),
                # so its bytes are gone once the profiled process exits and
                # _validated_file_uri raises below -> the decoded view then
                # reports "Decode unavailable" for those program counters.
                # Pass --complete-isa-decode at collection time to bake in
                # the disassembly for such code objects (it disassembles
                # them in-process from live memory).
                data = _validated_file_uri(
                    metadata["uri"],
                    read_bytes=True,
                    expected_size=metadata["memory_size"],
                )
                decoder.add_code_object_memory(
                    data,
                    key[1],
                    metadata["load_delta"],
                    metadata["load_size"],
                )
            else:
                return None
        except (OSError, ValueError, TypeError, RuntimeError):
            return None
        return _remember(decoder_cache, key, decoder, max_decoders)

    def _decode(guid, code_object_id, code_object_offset):
        if guid is None or code_object_id is None or code_object_offset is None:
            return None

        key = (str(guid), int(code_object_id), int(code_object_offset))
        if key in decode_cache:
            decode_cache.move_to_end(key)
            return decode_cache[key]
        decoder = _get_decoder(key[:2])
        result = None
        if decoder is not None:
            try:
                result = decoder.decode(key[1], key[2])
            except (ValueError, TypeError, RuntimeError):
                result = None
        return _remember(decode_cache, key, result, max_decoded_instructions)

    def _instruction(guid, code_object_id, code_object_offset):
        result = _decode(guid, code_object_id, code_object_offset)
        if not result:
            return None
        return result.get("instruction")

    def _comment(guid, code_object_id, code_object_offset):
        result = _decode(guid, code_object_id, code_object_offset)
        if not result:
            return (
                "Decode unavailable for code object id "
                f"{code_object_id} at offset {code_object_offset}"
            )
        return result.get("comment") or None

    conn.create_function("rocpd_isa_instruction", 3, _instruction)
    conn.create_function("rocpd_isa_comment", 3, _comment)

    blob_source = "rocpd_gpu_pc_sample_blob_decoded"
    if not _table_columns(conn, blob_source):
        blob_source = "rocpd_gpu_pc_sample"

    source_columns = _table_columns(conn, blob_source)
    if not source_columns:
        return
    if "guid" not in source_columns:
        raise ValueError("PC-sampling ISA decoding requires a guid column")

    computed_columns = {
        "instruction",
        "instruction_comment",
        "inst_type_name",
        "stall_reason_name",
    }
    source_select = ",\n                ".join(
        f"s.{_quote_identifier(column)}"
        for column in source_columns
        if column not in computed_columns
    )

    # Prefer disassembly stored at finalization (opt-in
    # --complete-isa-decode) when the rocpd_disassembly_data table is
    # present and populated; otherwise fall back to on-demand disassembly via the
    # scalar UDFs.  Either path yields the same instruction/instruction_comment
    # output columns, so downstream consumers (e.g. csv.py) are unaffected.
    disasm_columns = _table_columns(conn, "rocpd_disassembly_data")
    if disasm_columns:
        if "guid" not in disasm_columns:
            raise ValueError("Persisted disassembly data requires a guid column")
        instruction_expr = (
            "COALESCE(d.instruction, "
            "rocpd_isa_instruction(s.guid, s.code_object_id, s.code_object_offset))"
        )
        comment_expr = (
            "COALESCE(d.comment, "
            "rocpd_isa_comment(s.guid, s.code_object_id, s.code_object_offset))"
        )
        disasm_join = (
            "LEFT JOIN rocpd_disassembly_data d "
            "ON d.guid = s.guid "
            "AND d.code_object_id = s.code_object_id "
            "AND d.code_object_offset = s.code_object_offset"
        )
    else:
        instruction_expr = (
            "rocpd_isa_instruction(s.guid, s.code_object_id, s.code_object_offset)"
        )
        comment_expr = "rocpd_isa_comment(s.guid, s.code_object_id, s.code_object_offset)"
        disasm_join = ""

    conn.execute("DROP VIEW IF EXISTS temp.rocpd_gpu_pc_sample_decoded")
    conn.execute(f"""
            CREATE TEMP VIEW rocpd_gpu_pc_sample_decoded AS
            SELECT
                {source_select},
                {instruction_expr} AS instruction,
                {comment_expr} AS instruction_comment,
                CASE s."inst_type"
                    WHEN 0  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NONE'
                    WHEN 1  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU'
                    WHEN 2  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX'
                    WHEN 3  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR'
                    WHEN 4  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX'
                    WHEN 5  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS'
                    WHEN 6  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS_DIRECT'
                    WHEN 7  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT'
                    WHEN 8  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_EXPORT'
                    WHEN 9  THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE'
                    WHEN 10 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER'
                    WHEN 11 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN'
                    WHEN 12 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN'
                    WHEN 13 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP'
                    WHEN 14 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER'
                    WHEN 15 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST'
                    WHEN 16 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU'
                    ELSE NULL
                END AS inst_type_name,
                CASE s."stall_reason"
                    WHEN 0 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE'
                    WHEN 1 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE'
                    WHEN 2 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY'
                    WHEN 3 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT'
                    WHEN 4 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION'
                    WHEN 5 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT'
                    WHEN 6 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN'
                    WHEN 7 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL'
                    WHEN 8 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT'
                    WHEN 9 THEN 'ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT'
                    ELSE NULL
                END AS stall_reason_name
            FROM {_quote_identifier(blob_source)} s
            {disasm_join}
            """)


def setup_blob_views(conn):
    """Create generic, merge-safe decoded TEMP VIEWs from blob metadata.

    Metadata identity is ``(guid, schema_id)`` rather than the database-local
    schema id alone. For each source table, the canonical
    ``<source>_blob_decoded`` view supports every registered schema version;
    ``<source>_decoded`` is a compatibility alias that feature-specific layers
    may enrich without changing this generic decoder.
    """
    import struct as _struct

    try:
        schemas = conn.execute(
            "SELECT guid, id, source_table, byte_order, alignment, struct_size "
            "FROM rocpd_info_blob_schema ORDER BY source_table, guid, id"
        ).fetchall()
    except sqlite3.OperationalError as error:
        if _missing_object_error(error):
            return
        raise

    if not schemas:
        return

    field_cache = {}
    source_fields = {}
    seen_schemas = {}

    for guid, schema_id, source_table, byte_order, alignment, struct_size in schemas:
        if not isinstance(source_table, str):
            raise ValueError("Blob schema source_table must be text")
        _quote_identifier(source_table)
        if byte_order not in ("little", "big"):
            raise ValueError(f"Invalid byte order for blob schema {(guid, schema_id)!r}")
        if not isinstance(alignment, int) or alignment <= 0:
            raise ValueError(f"Invalid alignment for blob schema {(guid, schema_id)!r}")
        if not isinstance(struct_size, int) or struct_size <= 0:
            raise ValueError(f"Invalid struct size for blob schema {(guid, schema_id)!r}")

        schema_key = (str(guid), int(schema_id))
        schema_descriptor = (source_table, byte_order, alignment, struct_size)
        previous = seen_schemas.setdefault(schema_key, schema_descriptor)
        if previous != schema_descriptor:
            raise ValueError(f"Conflicting blob schema metadata for {schema_key!r}")

        fields = conn.execute(
            "SELECT guid, name, offset, size, data_type, is_signed "
            "FROM rocpd_info_blob_field "
            "WHERE guid = ? AND schema_id = ? ORDER BY offset, id",
            (guid, schema_id),
        ).fetchall()
        if not fields:
            raise ValueError(f"Blob schema {schema_key!r} has no fields")

        schema_field_names = set()
        for field_guid, name, offset, size, data_type, is_signed in fields:
            if str(field_guid) != schema_key[0]:
                raise ValueError(f"Blob field GUID does not match schema {schema_key!r}")
            _quote_identifier(name)
            normalized_name = name.casefold()
            if normalized_name in schema_field_names:
                raise ValueError(
                    f"Duplicate blob field {name!r} in schema {schema_key!r}"
                )
            schema_field_names.add(normalized_name)
            if not isinstance(offset, int) or offset < 0:
                raise ValueError(f"Invalid offset for blob field {name!r}")
            if not isinstance(size, int) or offset + size > struct_size:
                raise ValueError(f"Blob field {name!r} exceeds its struct bounds")

            fmt = _blob_struct_fmt(size, data_type, is_signed)
            field_cache[(schema_key[0], schema_key[1], name)] = (
                offset,
                "<" if byte_order == "little" else ">",
                fmt,
            )
            canonical_name = source_fields.setdefault(source_table, {}).setdefault(
                normalized_name, name
            )
            if canonical_name != name:
                raise ValueError(
                    f"Blob field casing differs across schemas: {canonical_name!r}, {name!r}"
                )

    def _rocpd_blob_field(blob, guid, schema_id, field_name):
        if blob is None or guid is None or schema_id is None or field_name is None:
            return None
        entry = field_cache.get((str(guid), int(schema_id), str(field_name)))
        if entry is None:
            return None
        offset, endian, fmt = entry
        try:
            value = _struct.unpack_from(endian + fmt, blob, offset)[0]
            if fmt == "Q" and value > 0x7FFFFFFFFFFFFFFF:
                return str(value)
            return value
        except (_struct.error, TypeError):
            return None

    try:
        conn.create_function("rocpd_blob_field", 4, _rocpd_blob_field, deterministic=True)
    except TypeError:
        conn.create_function("rocpd_blob_field", 4, _rocpd_blob_field)

    for source_table, fields_by_name in source_fields.items():
        domain_cols = _table_columns(conn, source_table)
        if not domain_cols:
            raise ValueError(f"Blob schema source table does not exist: {source_table!r}")

        domain_names = {column.casefold() for column in domain_cols}
        duplicate_names = domain_names.intersection(fields_by_name)
        if duplicate_names:
            raise ValueError(
                f"Blob fields conflict with columns in {source_table!r}: "
                f"{sorted(duplicate_names)!r}"
            )

        has_guid = "guid" in domain_cols
        if not has_guid:
            raise ValueError(f"Blob source table {source_table!r} needs a guid column")
        if "blob_event_id" in domain_cols:
            join_on = 'e."id" = s."blob_event_id"'
        elif "event_id" in domain_cols:
            join_on = 'e."event_id" = s."event_id"'
        else:
            raise ValueError(
                f"Blob source table {source_table!r} needs event_id or blob_event_id"
            )
        join_on += ' AND e."guid" = s."guid"'

        domain_select = ",\n    ".join(
            f"s.{_quote_identifier(column)}" for column in domain_cols
        )
        blob_select = ",\n    ".join(
            'rocpd_blob_field(e."blob", bs."guid", bs."id", '
            f"{_quote_sql_literal(name)}) AS {_quote_identifier(name)}"
            for name in fields_by_name.values()
        )
        schema_join = (
            'bs."guid" = e."guid" AND bs."id" = e."schema_id" '
            f'AND bs."source_table" = {_quote_sql_literal(source_table)}'
        )

        separator = ",\n    " if domain_select and blob_select else ""
        for view_name in (f"{source_table}_blob_decoded", f"{source_table}_decoded"):
            quoted_view = _quote_identifier(view_name)
            conn.execute(f"DROP VIEW IF EXISTS temp.{quoted_view}")
            view_sql = (
                f"CREATE TEMP VIEW {quoted_view} AS\n"
                f"SELECT\n"
                f"    {domain_select}{separator}\n"
                f"    {blob_select}\n"
                f"FROM {_quote_identifier(source_table)} s\n"
                f'LEFT JOIN "rocpd_blob_event" e ON {join_on}\n'
                'LEFT JOIN (SELECT DISTINCT "guid", "id", "source_table" '
                f'FROM "rocpd_info_blob_schema") bs ON {schema_join}'
            )
            conn.execute(view_sql)


def _create_temp_views(connection, input):
    """Create temporary unified views from multiple database files."""

    assert isinstance(connection, sqlite3.Connection)
    assert isinstance(input, list)

    # Attach each database and extract the uuid from each database
    dbinfo = []
    uuids = []
    for i, inp in enumerate(input):
        execute_statement(connection, f"ATTACH DATABASE '{inp}' AS db{i}")
        _uuids = [
            itr[0]
            for itr in execute_statement(
                connection,
                f"SELECT value FROM db{i}.rocpd_metadata WHERE tag='uuid'",
            ).fetchall()
        ]
        dbinfo += [f"db{i}"]
        uuids += [itr for itr in _uuids if itr not in uuids]

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

    return all_tables


def _create_meta_views(connection):
    schema = RocpdSchema()
    sql_script = schema.views.replace("CREATE VIEW", "CREATE TEMPORARY VIEW")
    execute_statement(connection, sql_script, is_script=True)
    # Decoded blob/ISA views are optional conveniences layered on top of the base
    # rocpd_ views. A malformed or unrecognized blob schema (e.g. a newer database or
    # third-party input) must not make the whole database unusable, so degrade
    # gracefully: skip the decoded views while leaving the base tables and standard
    # rocpd_ views fully queryable.
    try:
        setup_blob_views(connection)
        setup_isa_decode_views(connection)
    except (
        Exception
    ) as error:  # noqa: BLE001 - keep base views usable on decode-setup failure
        print(f"rocpd: skipping decoded blob/ISA views: {error}", file=sys.stderr)
