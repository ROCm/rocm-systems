"""PC-sampling blob views, ISA decoding, and SQL query rewriting."""

import atexit
import os
import sqlite3
import stat
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import libpyrocpd

__all__ = [
    "add_decode_args",
    "get_decode_args",
    "setup_blob_views",
    "setup_isa_decode_views",
    "update_query_for_blob_views",
]


def add_decode_args(parser, *, query_profile=False):
    """Register the ISA decode options for a command that opens databases."""
    decode = parser.add_argument_group("PC Sampling - ISA decode options")
    decode.add_argument(
        "--no-cache-disassembly",
        dest="cache_disassembly",
        action="store_false",
        default=True,
        help=(
            "Disable disassembly write-back to keep input databases read-only "
            "(default: cache verified instructions at exit)."
        ),
    )
    decode.add_argument(
        "--codeobj-path-prefix-map",
        metavar="SRC=DST",
        nargs="+",
        default=None,
        help=(
            "Remap code-object paths for offline decoding using SRC=DST pairs "
            "(e.g. /container/app=/host/app)."
        ),
    )
    if query_profile:
        decode.add_argument(
            "--blob-view-profile",
            action="store_true",
            help="Show PC-sampling query rewrite timing.",
        )
    return decode


def get_decode_args(args):
    """Return the RocpdImportData keyword arguments parsed by add_decode_args."""
    return {
        "codeobj_path_prefix_map": getattr(args, "codeobj_path_prefix_map", None),
        "cache_disassembly": getattr(args, "cache_disassembly", True),
    }


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


def _normalize_codeobj_path_prefix_map(mapping):
    """Normalize a code-object path prefix map into a list of
    ``(source_prefix, replacement_prefix)`` pairs.

    The map relocates code-object paths recorded under a different filesystem
    layout than post-processing runs in -- for example a code object recorded
    inside a container is decoded on the host under a different mount point (or
    vice versa).  It is consulted only as a fallback when the recorded path does
    not exist, and it only ever resolves to an *existing* regular file.  As with
    the recorded path, bytes that do not match the code object's recorded content
    hash are rejected (decoding reports "unavailable"), and decodes that could not
    be checked against a recorded hash are never written back.

    Supply it through the API
    (``RocpdImportData(..., codeobj_path_prefix_map=...)``) or the ``rocpd``
    command line (``--codeobj-path-prefix-map SRC=DST ...``); there is no
    environment-variable or built-in default.  Accepts ``None`` (no
    relocation), a ``dict`` of ``{source: replacement}``, or an iterable of
    ``"source=replacement"`` strings / ``(source, replacement)`` pairs.  When
    empty, recorded code-object paths are used exactly as stored.
    """
    if not mapping:
        return []
    if isinstance(mapping, str):
        mapping = [mapping]
    items = mapping.items() if isinstance(mapping, dict) else mapping
    pairs = []
    for entry in items:
        if isinstance(entry, str):
            source_prefix, sep, replacement_prefix = entry.partition("=")
            if not sep or not source_prefix:
                continue
        else:
            source_prefix, replacement_prefix = entry
        source_prefix = str(source_prefix)
        replacement_prefix = str(replacement_prefix)
        if source_prefix:
            pairs.append((source_prefix, replacement_prefix))
    return pairs


# Lazy-decode cache limits.  Each decoder holds its own copy of the code object.
_MAX_DECODER_CACHE_BYTES = 256 * 1024 * 1024
_MAX_DECODED_INSTRUCTIONS = 1 << 16


def setup_isa_decode_views(conn, codeobj_path_prefix_map=None, cache_disassembly=True):
    """Register lazy ISA decode functions and enrich PC-sample decoded TEMP VIEWs.

    ``setup_blob_views`` remains the generic self-describing blob decoder. This
    helper layers PC-sampling-specific ISA disassembly and enum-name expansion on
    top of its generic ``rocpd_gpu_pc_sample_decoded`` output when available.

    ``codeobj_path_prefix_map`` relocates recorded code-object paths when
    decoding in a different filesystem layout than profiling ran in (see
    _normalize_codeobj_path_prefix_map).  ``cache_disassembly`` (default True)
    writes lazily disassembled instructions back into the input database(s) at
    exit so later opens reuse them; only instructions decoded from bytes that
    match the code object's recorded content hash are written.  Pass False
    (rocpd ``--no-cache-disassembly``) to keep the input read-only.  Both are
    supplied through the API / ``rocpd`` command line, never an environment
    variable.

    If setup fails, ``rocpd_gpu_pc_sample_decoded`` is dropped rather than left
    without its ISA columns; ``rocpd_gpu_pc_sample_blob_decoded`` still exposes
    the generic blob fields.
    """
    try:
        _setup_isa_decode_views(conn, codeobj_path_prefix_map, cache_disassembly)
    except BaseException:
        conn.execute("DROP VIEW IF EXISTS temp.rocpd_gpu_pc_sample_decoded")
        raise


def _setup_isa_decode_views(conn, codeobj_path_prefix_map, cache_disassembly):
    blob_source = "rocpd_gpu_pc_sample_blob_decoded"
    if not _table_columns(conn, blob_source):
        blob_source = "rocpd_gpu_pc_sample"

    source_columns = _table_columns(conn, blob_source)
    if not source_columns:
        return
    if "guid" not in source_columns:
        raise ValueError("PC-sampling ISA decoding requires a guid column")

    # Decoder metadata and write-back targets only serve sample decoding.
    has_samples = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM rocpd_gpu_pc_sample)"
    ).fetchone()[0]

    code_objects = []
    if has_samples:
        try:
            code_objects = conn.execute("""
                SELECT
                    guid,
                    id,
                    nid,
                    pid,
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
                    JSON_EXTRACT(extdata, '$.memory_size') AS memory_size,
                    JSON_EXTRACT(extdata, '$.content_hash') AS content_hash
                FROM rocpd_info_code_object
                """).fetchall()
        except sqlite3.OperationalError as error:
            if not (
                _missing_object_error(error) or "no such column" in str(error).lower()
            ):
                raise

    code_object_metadata = {}
    for row in code_objects:
        (
            guid,
            code_object_id,
            nid,
            pid,
            uri,
            load_delta,
            load_size,
            storage_type,
            memory_size,
            content_hash,
        ) = row
        key = (str(guid), int(code_object_id))
        metadata = {
            "nid": nid,
            "pid": pid,
            "uri": uri,
            "load_delta": int(load_delta or 0),
            "load_size": int(load_size or 0),
            "storage_type": storage_type,
            "memory_size": int(memory_size or 0),
            "content_hash": content_hash if isinstance(content_hash, str) else None,
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

    path_prefix_map = _normalize_codeobj_path_prefix_map(codeobj_path_prefix_map)

    def _existing_path(path):
        if os.path.exists(path):
            return path
        # Fallback only: relocate the recorded path across known filesystem
        # layout differences (see _normalize_codeobj_path_prefix_map).  The bytes
        # found there are still size- and content-hash-checked before decoding.
        for source_prefix, replacement_prefix in path_prefix_map:
            if path.startswith(source_prefix):
                candidate = replacement_prefix + path[len(source_prefix) :]
                if os.path.exists(candidate):
                    return candidate
        return None

    max_code_object_bytes = 512 * 1024 * 1024

    def _read_code_object_file(uri, expected_size=0):
        parsed = urlparse(uri)
        if parsed.scheme.lower() != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("Only local file:// code-object URIs are supported")
        if parsed.query:
            raise ValueError("Code-object URI query parameters are unsupported")

        # strict_parsing rejects an empty fragment, which whole-file URIs omit.
        params = (
            parse_qs(parsed.fragment, keep_blank_values=True, strict_parsing=True)
            if parsed.fragment
            else {}
        )
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

        with open(path, "rb") as file:
            file.seek(offset)
            data = file.read(size)
        if len(data) != size:
            raise ValueError("Code-object file was truncated while being read")
        return data

    decoder_cache = OrderedDict()  # key -> (decoder, verified, code-object bytes)
    decoder_cache_bytes = 0
    decode_cache = OrderedDict()

    # decoder_cache value for a (guid, code_object_id) whose decoder cannot be
    # built, so the I/O-heavy attempt (URI validation, file read, hashing) runs
    # at most once instead of once per sampled program counter.
    decoder_unavailable = (None, False, 0)

    # ------------------------------------------------------------------
    # Cross-session disassembly cache.  On by default; disable per call with the
    # API's ``cache_disassembly=False`` argument or the rocpd
    # ``--no-cache-disassembly`` flag to keep the input database read-only.  When
    # enabled, program counters disassembled lazily during this session from a
    # code object whose bytes match its recorded content hash are written back
    # into rocpd_disassembly_data at interpreter exit, so a later open serves the
    # stored text through the decoded view's COALESCE instead of decoding it
    # again.  Strictly best-effort: any problem (read-only or deleted database,
    # lock contention, a pre-3.0.6 input without the table, ...) skips the
    # write-back and leaves the database untouched.
    # ------------------------------------------------------------------
    writeback_enabled = bool(cache_disassembly) and bool(has_samples)
    writeback_pending = {}
    writeback_targets = {}  # guid -> (db_path, uuid)
    max_writeback_pending = 1 << 20

    def _discover_writeback_targets():
        # A merged database holds one rocpd_metadata<uuid> table per recording,
        # so map every recording's guid to its own rocpd_disassembly_data<uuid>.
        try:
            attached = conn.execute("PRAGMA database_list").fetchall()
        except sqlite3.OperationalError:
            return
        for _seq, alias, path in attached:
            if not path or alias == "temp":
                continue
            schema = _quote_identifier(alias)
            try:
                tables = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT name FROM {schema}.sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
            except sqlite3.OperationalError:
                continue
            for metadata_table in sorted(tables):
                if not metadata_table.startswith("rocpd_metadata"):
                    continue
                try:
                    tags = dict(
                        conn.execute(
                            f"SELECT tag, value FROM {schema}."
                            f"{_quote_identifier(metadata_table)} "
                            "WHERE tag IN ('uuid', 'guid')"
                        ).fetchall()
                    )
                except sqlite3.OperationalError:
                    continue
                uuid, guid_val = tags.get("uuid"), tags.get("guid")
                if uuid is None or guid_val is None:
                    continue
                if metadata_table != f"rocpd_metadata{uuid}":
                    continue
                if f"rocpd_disassembly_data{uuid}" in tables:
                    writeback_targets[str(guid_val)] = (path, str(uuid))

    if writeback_enabled:
        _discover_writeback_targets()
        # With nothing to route rows to, keep the pure in-memory decode path.
        writeback_enabled = bool(writeback_targets)

    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _build_decoder(key):
        metadata = code_object_metadata.get(key)
        if not metadata or not metadata["uri"]:
            return decoder_unavailable
        if metadata["storage_type"] == "FILE":
            expected_size = 0
        elif metadata["storage_type"] == "MEMORY":
            # Limitation: lazy post-processing decode can only read a
            # memory-backed code object when its URI resolves to a real
            # on-disk file. A code object loaded purely from process
            # memory has a non-file URI (e.g. ``memory://...``) and is
            # not snapshotted to disk during PC sampling (unlike ATT),
            # so its bytes are gone once the profiled process exits and
            # _read_code_object_file raises below -> the decoded view then
            # reports "Decode unavailable" for those program counters.
            # Pass --complete-isa-decode at collection time to bake in
            # the disassembly for such code objects (it disassembles
            # them in-process from live memory).
            expected_size = metadata["memory_size"]
        else:
            return decoder_unavailable
        try:
            data = _read_code_object_file(metadata["uri"], expected_size=expected_size)
            # A rebuilt or remapped file can decode to plausible but wrong
            # instructions: reject bytes that differ from the profiled object.
            recorded_hash = metadata["content_hash"]
            if (
                recorded_hash
                and libpyrocpd.code_object_content_hash(data) != recorded_hash
            ):
                return decoder_unavailable
            decoder = libpyrocpd.isa_decoder()
            decoder.add_code_object_memory(
                data,
                key[1],
                metadata["load_delta"],
                metadata["load_size"],
            )
        except (OSError, ValueError, TypeError, RuntimeError):
            return decoder_unavailable
        return (decoder, bool(recorded_hash), len(data))

    def _get_decoder(key):
        """Return ``(decoder, verified)``; decoder is None when unavailable."""
        nonlocal decoder_cache_bytes
        cached = decoder_cache.get(key)
        if cached is None:
            cached = decoder_cache[key] = _build_decoder(key)
            decoder_cache_bytes += cached[2]
            # Keep the newest decoder even when it alone exceeds the budget.
            while (
                decoder_cache_bytes > _MAX_DECODER_CACHE_BYTES and len(decoder_cache) > 1
            ):
                decoder_cache_bytes -= decoder_cache.popitem(last=False)[1][2]
        decoder_cache.move_to_end(key)
        return cached[:2]

    def _decode(guid, code_object_id, code_object_offset):
        if guid is None or code_object_id is None or code_object_offset is None:
            return None

        key = (str(guid), int(code_object_id), int(code_object_offset))
        if key in decode_cache:
            decode_cache.move_to_end(key)
            return decode_cache[key]
        if key in writeback_pending:
            # Evicted from decode_cache but still held for write-back.
            instruction, comment = writeback_pending[key]
            return {"instruction": instruction, "comment": comment}
        decoder, verified = _get_decoder(key[:2])
        result = None
        if decoder is not None:
            try:
                result = decoder.decode(key[1], key[2])
            except (ValueError, TypeError, RuntimeError):
                result = None
        # Persist only decodes whose bytes matched the recorded content hash.
        if (
            writeback_enabled
            and verified
            and result is not None
            and result.get("instruction") is not None
            and key[0] in writeback_targets
            and len(writeback_pending) < max_writeback_pending
        ):
            writeback_pending[key] = (
                result.get("instruction"),
                result.get("comment") or "",
            )
        return _remember(decode_cache, key, result, _MAX_DECODED_INSTRUCTIONS)

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

    # Resolve PC-sampling enum codes to their ROCPROFILER_* names via the SDK's own
    # lookup (exposed through libpyrocpd) instead of a hardcoded mapping, so new
    # hardware inst_type / stall_reason values are named automatically.  The bindings
    # are optional: an older libpyrocpd without them yields NULL names rather than a
    # failed view.
    _inst_type_name_fn = getattr(libpyrocpd, "pc_sampling_instruction_type_name", None)
    _stall_reason_name_fn = getattr(
        libpyrocpd, "pc_sampling_instruction_not_issued_reason_name", None
    )

    def _enum_name(lookup, value):
        if value is None or lookup is None:
            return None
        try:
            return lookup(int(value))
        except (ValueError, TypeError):
            return None

    conn.create_function(
        "rocpd_pc_sample_inst_type_name",
        1,
        lambda value: _enum_name(_inst_type_name_fn, value),
    )
    conn.create_function(
        "rocpd_pc_sample_stall_reason_name",
        1,
        lambda value: _enum_name(_stall_reason_name_fn, value),
    )

    def _write_disassembly(path, recordings):
        writer = None
        try:
            # mode=rw: an input deleted since it was opened must stay deleted.
            writer = sqlite3.connect(
                f"{Path(path).resolve().as_uri()}?mode=rw", uri=True, timeout=5.0
            )
            writer.execute("PRAGMA busy_timeout = 5000")
            writer.execute("BEGIN IMMEDIATE")
            for (uuid, guid_val), rows in recordings.items():
                table = _quote_identifier(f"rocpd_disassembly_data{uuid}")
                try:
                    # The file may have been replaced since it was opened.
                    stored = writer.execute(
                        f"SELECT value FROM {_quote_identifier(f'rocpd_metadata{uuid}')} "
                        "WHERE tag = 'guid'"
                    ).fetchone()
                    if not stored or str(stored[0]) != guid_val:
                        continue
                    # Skip PCs already stored, e.g. by another process.
                    pending = {(row[3], row[4]): row for row in rows}
                    for key in writer.execute(
                        f'SELECT "code_object_id", "code_object_offset" FROM {table} '
                        'WHERE "guid" = ?',
                        (guid_val,),
                    ):
                        pending.pop(key, None)
                except sqlite3.DatabaseError:
                    continue
                writer.executemany(
                    f"INSERT INTO {table} "
                    '("guid", "nid", "pid", "code_object_id", '
                    '"code_object_offset", "instruction", "comment") '
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    pending.values(),
                )
            writer.commit()
        except Exception:  # noqa: BLE001 - best-effort cache write
            if writer is not None:
                try:
                    writer.rollback()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass

    def _flush_writeback():
        # Persist lazily decoded, hash-verified instructions into
        # rocpd_disassembly_data so a future open reads the stored text instead of
        # decoding again.  Runs at interpreter exit through a private writer
        # connection (the import connection may already be closed) and routes each
        # row to its recording's table.  Entirely best-effort; a failure just
        # leaves the cache unwritten.
        if not writeback_pending:
            return
        try:
            by_path = {}
            for key, value in writeback_pending.items():
                guid_val, co_id, offset = key
                instr, comment = value
                target = writeback_targets.get(guid_val)
                code_object = code_object_metadata.get((guid_val, co_id))
                if not target or not code_object:
                    continue
                if code_object["nid"] is None or code_object["pid"] is None:
                    continue
                path, uuid = target
                by_path.setdefault(path, {}).setdefault((uuid, guid_val), []).append(
                    (
                        guid_val,
                        code_object["nid"],
                        code_object["pid"],
                        co_id,
                        offset,
                        instr,
                        comment,
                    )
                )
            for path, recordings in by_path.items():
                _write_disassembly(path, recordings)
        except Exception:  # noqa: BLE001 - never raise from an atexit handler
            pass
        finally:
            writeback_pending.clear()

    if writeback_enabled:
        atexit.register(_flush_writeback)

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
        missing = sorted(
            {"guid", "code_object_id", "code_object_offset", "instruction", "comment"}
            - set(disasm_columns)
        )
        if missing:
            raise ValueError(f"Persisted disassembly data lacks columns {missing}")
        instruction_expr = (
            "COALESCE(d.instruction, "
            "rocpd_isa_instruction(s.guid, s.code_object_id, s.code_object_offset))"
        )
        comment_expr = (
            "CASE WHEN d.instruction IS NOT NULL THEN NULLIF(d.comment, '') "
            "ELSE rocpd_isa_comment(s.guid, s.code_object_id, s.code_object_offset) END"
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

    # CREATE VIEW does not resolve table or column names, so run a zero-row query
    # over the body before replacing the generic alias.
    view_body = f"""
            SELECT
                {source_select},
                {instruction_expr} AS instruction,
                {comment_expr} AS instruction_comment,
                rocpd_pc_sample_inst_type_name(s."inst_type") AS inst_type_name,
                rocpd_pc_sample_stall_reason_name(s."stall_reason")
                    AS stall_reason_name
            FROM {_quote_identifier(blob_source)} s
            {disasm_join}
            """

    conn.execute(f"SELECT * FROM ({view_body}) LIMIT 0").fetchall()
    conn.execute("DROP VIEW IF EXISTS temp.rocpd_gpu_pc_sample_decoded")
    conn.execute(f"CREATE TEMP VIEW rocpd_gpu_pc_sample_decoded AS {view_body}")


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

    layouts = {}
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
        endian = "<" if byte_order == "little" else ">"
        layout_format = [endian]
        layout_fields = []
        layout_end = 0
        has_overlaps = False
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
            if offset < layout_end:
                has_overlaps = True
            elif offset > layout_end:
                layout_format.append(f"{offset - layout_end}x")
            layout_format.append(fmt)
            layout_end = offset + size
            layout_fields.append((name, offset, _struct.Struct(endian + fmt), fmt == "Q"))
            canonical_name = source_fields.setdefault(source_table, {}).setdefault(
                normalized_name, name
            )
            if canonical_name != name:
                raise ValueError(
                    f"Blob field casing differs across schemas: {canonical_name!r}, {name!r}"
                )

        layouts[schema_key] = (
            None if has_overlaps else _struct.Struct("".join(layout_format)),
            layout_fields,
            {
                name: (index, is_unsigned_64)
                for index, (name, offset, field_unpacker, is_unsigned_64) in enumerate(
                    layout_fields
                )
            },
        )

    last_key = None
    last_values = ()
    last_field_indices = {}

    def _rocpd_blob_field(blob, guid, schema_id, field_name):
        nonlocal last_key, last_values, last_field_indices
        if blob is None or guid is None or schema_id is None or field_name is None:
            return None
        key = (guid, schema_id, blob)
        if key != last_key:
            layout = layouts.get((str(guid), int(schema_id)))
            if layout is None:
                return None
            unpacker, layout_fields, field_indices = layout
            try:
                unpacked = unpacker.unpack_from(blob) if unpacker is not None else None
            except (_struct.error, TypeError):
                unpacked = None

            if unpacked is None:
                unpacked = []
                for name, offset, field_unpacker, is_unsigned_64 in layout_fields:
                    try:
                        value = field_unpacker.unpack_from(blob, offset)[0]
                    except (_struct.error, TypeError):
                        value = None
                    unpacked.append(value)
            last_key, last_values, last_field_indices = key, unpacked, field_indices
        field = last_field_indices.get(str(field_name))
        if field is None:
            return None
        index, is_unsigned_64 = field
        value = last_values[index]
        if is_unsigned_64 and value is not None and value > 0x7FFFFFFFFFFFFFFF:
            return str(value)
        return value

    try:
        conn.create_function("rocpd_blob_field", 4, _rocpd_blob_field, deterministic=True)
    except (TypeError, sqlite3.NotSupportedError):
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
        # Filter blobs to this source's schemas inside the join: a blob another
        # source attached to the same event must not add a row here.
        join_on += (
            ' AND (e."guid", e."schema_id") IN (SELECT "guid", "id" '
            'FROM "rocpd_info_blob_schema" WHERE "source_table" = '
            f"{_quote_sql_literal(source_table)})"
        )

        domain_select = ",\n    ".join(
            f"s.{_quote_identifier(column)}" for column in domain_cols
        )
        blob_select = ",\n    ".join(
            'rocpd_blob_field(e."blob", e."guid", e."schema_id", '
            f"{_quote_sql_literal(name)}) AS {_quote_identifier(name)}"
            for name in fields_by_name.values()
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
                f'LEFT JOIN "rocpd_blob_event" e ON {join_on}'
            )
            conn.execute(view_sql)


_BLOB_DECODED_SUFFIX = "_decoded"


def _sql_tokens(query: str):
    """Yield lossless ``(kind, text, start, end)`` spans for SQLite query rewriting.

    Strings, comments, and double-quoted/backtick/bracket identifiers remain
    single tokens; doubled string and identifier quotes are retained verbatim.
    Parentheses are individual symbols so callers can track nested expressions
    and CTE bodies without treating their contents as top-level table factors.
    Unterminated quotes/comments consume the remaining text; SQL validation is
    left to SQLite rather than silently repairing the query.
    """
    index = 0
    while index < len(query):
        start = index
        char = query[index]
        if char.isspace():
            index += 1
            while index < len(query) and query[index].isspace():
                index += 1
            yield "space", query[start:index], start, index
        elif query.startswith("--", index):
            end = query.find("\n", index + 2)
            index = len(query) if end < 0 else end
            yield "comment", query[start:index], start, index
        elif query.startswith("/*", index):
            end = query.find("*/", index + 2)
            index = len(query) if end < 0 else end + 2
            yield "comment", query[start:index], start, index
        elif char == "'":
            index += 1
            while index < len(query):
                if query[index] == "'":
                    index += 1
                    if index < len(query) and query[index] == "'":
                        index += 1
                        continue
                    break
                index += 1
            yield "string", query[start:index], start, index
        elif char in ('"', "`", "["):
            closing = "]" if char == "[" else char
            index += 1
            while index < len(query):
                if query[index] == closing:
                    index += 1
                    if closing != "]" and index < len(query) and query[index] == closing:
                        index += 1
                        continue
                    break
                index += 1
            yield "identifier", query[start:index], start, index
        elif char.isalpha() or char == "_":
            index += 1
            while index < len(query) and (query[index].isalnum() or query[index] in "_$"):
                index += 1
            yield "identifier", query[start:index], start, index
        else:
            index += 1
            yield "symbol", query[start:index], start, index


def _identifier_value(text: str) -> str:
    if text.startswith("[") and text.endswith("]"):
        return text[1:-1]
    if text[:1] in ('"', "`") and text.endswith(text[0]):
        return text[1:-1].replace(text[0] * 2, text[0])
    return text


def _quote_like_identifier(original: str, value: str) -> str:
    if original.startswith("["):
        return "[" + value.replace("]", "]]") + "]"
    if original[:1] in ('"', "`"):
        quote = original[0]
        return quote + value.replace(quote, quote * 2) + quote
    return value


def _significant_tokens(tokens):
    return [token for token in tokens if token[0] not in ("space", "comment")]


def _skip_parenthesized(significant, index):
    """Return the index just past the parenthesized group opening at *index*."""
    depth = 1
    index += 1
    while index < len(significant) and depth:
        depth += significant[index][1] == "("
        depth -= significant[index][1] == ")"
        index += 1
    return index


def _cte_list_names(significant, index):
    """Return the CTE names of the WITH list whose first token is at *index*."""
    names = set()
    if index < len(significant) and significant[index][1].casefold() == "recursive":
        index += 1
    while index < len(significant):
        token = significant[index]
        if token[0] != "identifier":
            break
        names.add(_identifier_value(token[1]).casefold())
        index += 1
        if index < len(significant) and significant[index][1] == "(":
            index = _skip_parenthesized(significant, index)
        if index >= len(significant) or significant[index][1].casefold() != "as":
            break
        index += 1
        if index < len(significant) and significant[index][1].casefold() == "not":
            index += 1
        if (
            index < len(significant)
            and significant[index][1].casefold() == "materialized"
        ):
            index += 1
        if index >= len(significant) or significant[index][1] != "(":
            break
        index = _skip_parenthesized(significant, index)
        if index >= len(significant) or significant[index][1] != ",":
            break
        index += 1
    return names


def _cte_names(tokens):
    """Return the names of every CTE in the query, at any nesting depth.

    Identifier quoting is removed and names are case-folded. Comments and
    literals cannot introduce CTEs. WITH RECURSIVE, optional column lists, and
    materialization hints are accepted; balanced nested bodies are skipped when
    advancing through each WITH list, then inspected for their own WITH lists.

    Names are not scoped: a CTE that shadows a blob source table anywhere keeps
    that name unrewritten everywhere, so the query runs exactly as written.
    """
    significant = _significant_tokens(tokens)
    names = set()
    for index, token in enumerate(significant):
        if token[0] == "identifier" and token[1].casefold() == "with":
            names.update(_cte_list_names(significant, index + 1))
    return names


# Keywords that can directly follow a table name in a FROM/JOIN without being an alias.
_TABLE_FACTOR_FOLLOWERS = frozenset(
    (
        "cross",
        "except",
        "full",
        "group",
        "having",
        "indexed",
        "inner",
        "intersect",
        "join",
        "left",
        "limit",
        "natural",
        "not",
        "on",
        "order",
        "outer",
        "returning",
        "right",
        "union",
        "using",
        "where",
        "window",
    )
)


def _starts_table_alias(token):
    """Return True when *token*, which follows a table name, begins its alias."""
    if token is None or token[0] != "identifier":
        return False
    keyword = token[1].casefold()
    return keyword == "as" or keyword not in _TABLE_FACTOR_FOLLOWERS


def update_query_for_blob_views(conn, query: str = None, profile: bool = False):
    """Rewrite *query* replacing blob-source table references with their
    ``_decoded`` TEMP VIEW counterparts.

    The decoded TEMP VIEWs and the ``rocpd_blob_field`` scalar function are
    created during ``RocpdImportData`` initialisation by
    :func:`setup_blob_views`.  This function only rewrites the
    query string so callers do not need to manually use the ``_decoded`` suffix.

    A source table is rewritten only when its ``<source>_decoded`` view actually
    exists -- blob-view setup is skipped for malformed/unknown blob metadata, and
    rewriting to a non-existent view would turn a still-queryable base table into
    an error.  Only unqualified references, or references explicitly qualified
    with ``temp``, are rewritten (the decoded views are TEMP views, so a
    ``main.``/attached-database qualifier is left untouched), and every table
    factor in a comma-separated ``FROM`` list is rewritten, not just the first.
    A rewritten table factor without an alias is aliased to its original name,
    so ``<source>.<column>`` references keep resolving.  A name defined as a CTE
    anywhere in *query* is never rewritten; select the ``_decoded`` view
    explicitly where a query needs both.

    When no rewritable blob source tables are registered, or *query* is ``None``,
    returns *query* unchanged.
    """
    import sqlite3

    _t0 = time.perf_counter() if profile else None

    if not query:
        return query

    try:
        rows = conn.execute(
            "SELECT DISTINCT source_table FROM rocpd_info_blob_schema "
            "WHERE source_table IS NOT NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        return query

    if not rows:
        return query

    try:
        existing_temp_views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_temp_master WHERE type = 'view'"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        return query

    replacements = {}
    for (source_table,) in rows:
        if not source_table:
            continue
        decoded = f"{source_table}{_BLOB_DECODED_SUFFIX}"
        if decoded in existing_temp_views:
            replacements[str(source_table).casefold()] = decoded
    if not replacements:
        return query

    tokens = list(_sql_tokens(query))
    cte_names = _cte_names(tokens)
    significant_indexes = [
        index
        for index, token in enumerate(tokens)
        if token[0] not in ("space", "comment")
    ]

    # Clause keywords that close an open FROM list at its own parenthesis depth.
    _from_list_enders = frozenset(
        (
            "where",
            "group",
            "having",
            "window",
            "order",
            "limit",
            "union",
            "except",
            "intersect",
            "returning",
        )
    )

    edits = []
    expect_table = False
    paren_depth = 0
    from_list_depths = []  # stack of parenthesis depths with an active FROM list
    for position, token_index in enumerate(significant_indexes):
        kind, text, start, end = tokens[token_index]
        keyword = text.casefold() if kind == "identifier" else text

        if text == "(":
            paren_depth += 1
            expect_table = False
            continue
        if text == ")":
            paren_depth -= 1
            while from_list_depths and from_list_depths[-1] > paren_depth:
                from_list_depths.pop()
            expect_table = False
            continue

        if keyword in ("from", "join"):
            expect_table = True
            if keyword == "from":
                from_list_depths.append(paren_depth)
            continue

        # A top-level comma inside an active FROM list starts another table factor.
        if text == "," and from_list_depths and from_list_depths[-1] == paren_depth:
            expect_table = True
            continue

        # A clause keyword at the FROM list's depth closes that FROM list.
        if (
            kind == "identifier"
            and keyword in _from_list_enders
            and from_list_depths
            and from_list_depths[-1] == paren_depth
        ):
            from_list_depths.pop()
            expect_table = False
            continue

        if not expect_table:
            continue
        expect_table = False
        if kind != "identifier":
            continue

        # A qualified name such as main.table / temp.table.  Only rewrite when
        # unqualified or explicitly qualified with temp (where decoded views live).
        name_token = tokens[token_index]
        qualifier = None
        if position + 2 < len(significant_indexes):
            dot = tokens[significant_indexes[position + 1]]
            qualified = tokens[significant_indexes[position + 2]]
            if dot[1] == "." and qualified[0] == "identifier":
                qualifier = _identifier_value(name_token[1]).casefold()
                name_token = qualified

        if qualifier is not None and qualifier != "temp":
            continue

        name = _identifier_value(name_token[1])
        replacement = replacements.get(name.casefold())
        if not replacement:
            continue
        # An unqualified name matching a CTE is the CTE, not a table.
        if qualifier is None and name.casefold() in cte_names:
            continue
        replacement_text = _quote_like_identifier(name_token[1], replacement)
        alias_position = position + (3 if qualifier is not None else 1)
        alias_token = (
            tokens[significant_indexes[alias_position]]
            if alias_position < len(significant_indexes)
            else None
        )
        if not _starts_table_alias(alias_token):
            replacement_text += f" AS {name_token[1]}"
        edits.append((name_token[2], name_token[3], replacement_text))

    for start, end, replacement in reversed(edits):
        query = query[:start] + replacement + query[end:]

    if profile and _t0 is not None:
        elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        print(f"Blob view query rewrite: elapsed={elapsed_ms:.2f} ms")

    return query
