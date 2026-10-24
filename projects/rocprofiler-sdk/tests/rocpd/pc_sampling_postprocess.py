#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GPU-independent tests for rocpd PC-sampling post-processing.

Each test writes schema-3.0.6 databases holding synthetic PC samples and replaces
the COMGR-backed ``libpyrocpd.isa_decoder`` with :class:`FakeDecoder`, whose
instruction text is derived from the code-object bytes.  Write-back happens at
interpreter exit, so tests that inspect it run the importer in a subprocess.
"""

import contextlib
import inspect
import json
import sqlite3
import struct
import subprocess
import sys
import uuid as uuidlib

import pytest

from rocpd import libpyrocpd
from rocpd.importer import RocpdImportData

SCHEMA_VERSION = "3.0.6"
NID = 1
PID = 100
TID = 200
AGENT_ID = 1
PC_SCHEMA_ID = 1
REGION_SCHEMA_ID = 2
FULL_EXEC_MASK = "18446744073709551615"


class FakeDecoder:
    """Stand-in for libpyrocpd.isa_decoder that names the byte at each offset."""

    def __init__(self):
        self.objects = {}

    def add_code_object_memory(self, data, code_object_id, load_delta, load_size):
        self.objects[code_object_id] = bytes(data)

    def add_code_object_file(self, uri, code_object_id, load_delta, load_size):
        from urllib.parse import parse_qs, unquote, urlparse

        parsed = urlparse(uri)
        params = parse_qs(parsed.fragment)
        with open(unquote(parsed.path), "rb") as ifs:
            ifs.seek(int(params.get("offset", ["0"])[0], 0))
            size = int(params.get("size", ["-1"])[0], 0)
            self.add_code_object_memory(
                ifs.read(size), code_object_id, load_delta, load_size
            )

    def decode(self, code_object_id, code_object_offset):
        data = self.objects.get(code_object_id)
        if data is None or code_object_offset >= len(data):
            return None
        return {
            "instruction": f"insn_{data[code_object_offset]:02x}",
            "comment": f"offset {code_object_offset}",
        }


def insn(data, offset):
    return f"insn_{data[offset]:02x}"


class CountingDecoder(FakeDecoder):
    """FakeDecoder that counts decoder builds and instruction decodes."""

    builds = 0
    decodes = 0

    def add_code_object_memory(self, *args):
        type(self).builds += 1
        super().add_code_object_memory(*args)

    def decode(self, *args):
        type(self).decodes += 1
        return super().decode(*args)


DRIVER = """
import json
import os
import sys

from rocpd import libpyrocpd

{fake_decoder}

libpyrocpd.isa_decoder = FakeDecoder

spec = json.loads(sys.argv[1])
if "csv_main" in spec:
    from rocpd import csv

    csv.main(spec["csv_main"])
elif "query_main" in spec:
    from rocpd import query

    query.main(spec["query_main"])
else:
    import sqlite3
    from rocpd.importer import RocpdImportData

    data = RocpdImportData(
        spec["inputs"],
        skip_auto_merge=True,
        cache_disassembly=spec.get("cache", True),
        codeobj_path_prefix_map=spec.get("prefix_map"),
    )
    rows = data.execute(
        "SELECT guid, code_object_id, code_object_offset, instruction, "
        "instruction_comment FROM rocpd_gpu_pc_sample_decoded ORDER BY guid, id"
    ).fetchall()
    print("ROWS=" + json.dumps(rows))
    for table, offset, text in spec.get("concurrent_rows", []):
        samples = table.replace("rocpd_disassembly_data", "rocpd_gpu_pc_sample")
        writer = sqlite3.connect(spec["inputs"][0])
        writer.execute(
            'INSERT INTO "' + table + '" (guid, nid, pid, code_object_id, '
            "code_object_offset, instruction, comment) "
            "SELECT guid, nid, pid, code_object_id, code_object_offset, ?, '' "
            'FROM "' + samples + '" WHERE code_object_offset = ? LIMIT 1',
            (text, offset),
        )
        writer.commit()
        writer.close()
    if spec.get("close_and_delete"):
        data.connection.close()
        for path in spec["inputs"]:
            os.remove(path)
"""


def run_driver(tmp_path, **spec):
    """Run the importer (or a module CLI) in a subprocess so atexit write-back runs."""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(fake_decoder=inspect.getsource(FakeDecoder)))
    result = subprocess.run(
        [sys.executable, str(driver), json.dumps(spec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("ROWS="):
            return [tuple(row) for row in json.loads(line[len("ROWS=") :])]
    return None


def query_db(path, sql, params=()):
    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        return conn.execute(sql, params).fetchall()


def disassembly_table(path):
    return query_db(
        path,
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name LIKE 'rocpd_disassembly_data%' ORDER BY name",
    )[0][0]


def stored_disassembly(path):
    return query_db(
        path,
        "SELECT guid, code_object_id, code_object_offset, instruction "
        "FROM rocpd_disassembly_data ORDER BY guid, code_object_id, code_object_offset",
    )


def content_hash(data):
    return libpyrocpd.code_object_content_hash(data)


def file_code_object(path, data):
    """Code object 1 stored at *path*, recorded with the content hash of *data*."""
    return {"id": 1, "uri": path.as_uri(), "content_hash": content_hash(data)}


def build_db(path, *, code_objects=(), samples=(), regions=(), guid=None):
    """Write a schema-3.0.6 database and return its guid.

    ``code_objects``: dicts with id, uri and optional storage_type, memory_size, content_hash.
    ``samples``: dicts with timestamp and optional offset, code_object_id, exec_mask, event_id.
    ``regions``: dicts with event_id, start and end; each gets a region-schema blob.
    """
    from rocpd.schema import RocpdSchema

    guid = guid or str(uuidlib.uuid4())
    uuid = "_" + guid.replace("-", "_")

    def table(name):
        return f'"{name}{uuid}"'

    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        RocpdSchema(uuid=uuid, guid=guid, version=SCHEMA_VERSION).write_schema(conn)
        conn.execute(
            f"INSERT INTO {table('rocpd_info_node')} (id, guid, hash, machine_id, "
            "system_name, hostname, release, version, hardware_name, domain_name) "
            "VALUES (?, ?, 1, 'machine', 'Linux', 'host', 'r', 'v', 'x86_64', 'd')",
            (NID, guid),
        )
        conn.execute(
            f"INSERT INTO {table('rocpd_info_process')} (id, guid, nid, ppid, pid, "
            "init, fini, start, end, command) VALUES (?, ?, ?, 1, ?, 0, 0, 0, 0, 'app')",
            (PID, guid, NID, PID),
        )
        conn.execute(
            f"INSERT INTO {table('rocpd_info_thread')} (id, guid, nid, ppid, pid, tid) "
            "VALUES (?, ?, ?, 1, ?, ?)",
            (TID, guid, NID, PID, TID),
        )
        conn.execute(
            f"INSERT INTO {table('rocpd_info_agent')} (id, guid, nid, pid, type, "
            "absolute_index, logical_index, type_index) VALUES (?, ?, ?, ?, 'GPU', 1, 0, 0)",
            (AGENT_ID, guid, NID, PID),
        )
        conn.execute(
            f"INSERT INTO {table('rocpd_string')} (id, guid, string) VALUES (1, ?, 'region')",
            (guid,),
        )

        schemas = [(PC_SCHEMA_ID, "rocpd_gpu_pc_sample", "hw_id_wave_id")]
        if regions:
            schemas.append((REGION_SCHEMA_ID, "rocpd_region", "region_flag"))
        for schema_id, source_table, field in schemas:
            conn.execute(
                f"INSERT INTO {table('rocpd_info_blob_schema')} (id, guid, nid, pid, name, "
                "source_table, byte_order, alignment, struct_size, version) "
                "VALUES (?, ?, ?, ?, ?, ?, 'little', 1, 4, 1)",
                (schema_id, guid, NID, PID, f"{source_table}_test", source_table),
            )
            conn.execute(
                f"INSERT INTO {table('rocpd_info_blob_field')} (guid, schema_id, name, "
                "offset, size, data_type, is_signed) VALUES (?, ?, ?, 0, 4, 'uint32_t', 0)",
                (guid, schema_id, field),
            )

        for code_object in code_objects:
            extdata = {"memory_size": code_object.get("memory_size", 0)}
            if code_object.get("content_hash"):
                extdata["content_hash"] = code_object["content_hash"]
            conn.execute(
                f"INSERT INTO {table('rocpd_info_code_object')} (id, guid, nid, pid, "
                "agent_id, uri, load_base, load_size, load_delta, storage_type, extdata) "
                "VALUES (?, ?, ?, ?, ?, ?, 4096, 65536, 0, ?, ?)",
                (
                    code_object["id"],
                    guid,
                    NID,
                    PID,
                    AGENT_ID,
                    code_object["uri"],
                    code_object.get("storage_type", "FILE"),
                    json.dumps(extdata),
                ),
            )

        for index, sample in enumerate(samples, start=1):
            event_id = sample.get("event_id", index)
            correlation_id = sample.get("correlation_id", index)
            conn.execute(
                f"INSERT OR IGNORE INTO {table('rocpd_event')} "
                "(id, guid, correlation_id, stack_id) VALUES (?, ?, ?, ?)",
                (event_id, guid, correlation_id, sample.get("stack_id", index)),
            )
            conn.execute(
                f"INSERT INTO {table('rocpd_blob_event')} (guid, nid, pid, event_id, "
                "schema_id, blob) VALUES (?, ?, ?, ?, ?, ?)",
                (guid, NID, PID, event_id, PC_SCHEMA_ID, struct.pack("<I", index)),
            )
            conn.execute(
                f"INSERT INTO {table('rocpd_gpu_pc_sample')} (id, guid, timestamp, nid, "
                "pid, tid, agent_id, event_id, dispatch_id, correlation_id, exec_mask, "
                "code_object_id, code_object_offset, wave_issued) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (
                    index,
                    guid,
                    sample["timestamp"],
                    NID,
                    PID,
                    TID,
                    AGENT_ID,
                    event_id,
                    correlation_id,
                    sample.get("exec_mask", "1"),
                    sample.get("code_object_id", 1),
                    sample.get("offset", 0),
                    sample.get("wave_issued"),
                ),
            )

        for index, region in enumerate(regions, start=1):
            conn.execute(
                f"INSERT OR IGNORE INTO {table('rocpd_event')} (id, guid, correlation_id) "
                "VALUES (?, ?, 0)",
                (region["event_id"], guid),
            )
            conn.execute(
                f"INSERT INTO {table('rocpd_blob_event')} (guid, nid, pid, event_id, "
                "schema_id, blob) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    guid,
                    NID,
                    PID,
                    region["event_id"],
                    REGION_SCHEMA_ID,
                    struct.pack("<I", 7),
                ),
            )
            conn.execute(
                f"INSERT INTO {table('rocpd_region')} (id, guid, nid, pid, tid, start, end, "
                "name_id, event_id) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    index,
                    guid,
                    NID,
                    PID,
                    TID,
                    region["start"],
                    region["end"],
                    region["event_id"],
                ),
            )
        conn.commit()
    return guid


def open_db(paths, **kwargs):
    kwargs.setdefault("cache_disassembly", False)
    return RocpdImportData([str(path) for path in paths], skip_auto_merge=True, **kwargs)


@pytest.fixture
def fake_decoder(monkeypatch):
    monkeypatch.setattr(libpyrocpd, "isa_decoder", FakeDecoder)


@pytest.fixture
def counting_decoder(monkeypatch):
    monkeypatch.setattr(libpyrocpd, "isa_decoder", CountingDecoder)
    monkeypatch.setattr(CountingDecoder, "builds", 0)
    monkeypatch.setattr(CountingDecoder, "decodes", 0)
    return CountingDecoder


@pytest.fixture
def code_object(tmp_path):
    data = bytes(range(16, 80))
    path = tmp_path / "kernel.co"
    path.write_bytes(data)
    return path, data


def test_content_hash_is_fnv1a64():
    assert content_hash(b"") == "fnv1a64:cbf29ce484222325"
    assert content_hash(b"foobar") == "fnv1a64:85944171f73967e8"


@pytest.mark.parametrize("storage_type", ["FILE", "MEMORY"])
def test_verified_decode_is_written_back(tmp_path, code_object, storage_type):
    path, data = code_object
    db = tmp_path / "pc.db"
    recorded_object = file_code_object(path, data)
    recorded_object.update(
        storage_type=storage_type,
        memory_size=len(data) if storage_type == "MEMORY" else 0,
    )
    guid = build_db(
        db,
        code_objects=[recorded_object],
        samples=[{"timestamp": 100, "offset": 4}],
    )
    rows = run_driver(tmp_path, inputs=[str(db)])
    assert rows[0][3] == insn(data, 4)
    assert stored_disassembly(db) == [(guid, 1, 4, insn(data, 4))]


@pytest.mark.parametrize("comment", [None, "", "source line"])
def test_cached_comment_matches_lazy_decode(
    tmp_path, monkeypatch, counting_decoder, code_object, comment
):
    import rocpd.importer as importer

    callbacks = []
    monkeypatch.setattr(importer.atexit, "register", callbacks.append)
    decode = counting_decoder.decode

    def decode_with_comment(self, *args):
        result = decode(self, *args)
        result["comment"] = comment
        return result

    monkeypatch.setattr(counting_decoder, "decode", decode_with_comment)
    path, data = code_object
    db = tmp_path / "pc.db"
    guid = build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100, "offset": 4}],
    )
    query = (
        "SELECT instruction, instruction_comment, instruction_comment IS NULL "
        "FROM rocpd_gpu_pc_sample_decoded"
    )
    with open_db([db], cache_disassembly=True) as connection:
        cold_rows = connection.execute(query).fetchall()
    assert cold_rows == [(insn(data, 4), comment or None, int(not comment))]
    assert callbacks
    for callback in callbacks:
        callback()
    assert stored_disassembly(db) == [(guid, 1, 4, insn(data, 4))]
    with open_db([db]) as connection:
        assert connection.execute(query).fetchall() == cold_rows
    assert counting_decoder.decodes == 1


def test_rebuilt_code_object_is_rejected(tmp_path, code_object):
    path, data = code_object
    db = tmp_path / "pc.db"
    build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100, "offset": 4}],
    )
    path.write_bytes(bytes(reversed(data)))  # same size, different bytes
    rows = run_driver(tmp_path, inputs=[str(db)])
    assert rows[0][3] is None
    assert rows[0][4].startswith("Decode unavailable")
    assert stored_disassembly(db) == []


@pytest.mark.parametrize("same_bytes", [True, False])
def test_remapped_code_object_must_match_recorded_hash(tmp_path, code_object, same_bytes):
    _, data = code_object
    recorded_dir = tmp_path / "recorded"
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    (relocated_dir / "kernel.co").write_bytes(
        data if same_bytes else bytes(reversed(data))
    )
    db = tmp_path / "pc.db"
    guid = build_db(
        db,
        code_objects=[
            {
                "id": 1,
                "uri": (recorded_dir / "kernel.co").as_uri(),
                "content_hash": content_hash(data),
            }
        ],
        samples=[{"timestamp": 100, "offset": 9}],
    )
    rows = run_driver(
        tmp_path, inputs=[str(db)], prefix_map=[f"{recorded_dir}={relocated_dir}"]
    )
    if same_bytes:
        assert rows[0][3] == insn(data, 9)
        assert stored_disassembly(db) == [(guid, 1, 9, insn(data, 9))]
    else:
        assert rows[0][3] is None
        assert stored_disassembly(db) == []


def test_unverified_decode_is_not_written_back(tmp_path, code_object):
    path, data = code_object
    db = tmp_path / "pc.db"
    build_db(
        db,
        code_objects=[{"id": 1, "uri": path.as_uri()}],
        samples=[{"timestamp": 100, "offset": 2}],
    )
    rows = run_driver(tmp_path, inputs=[str(db)])
    assert rows[0][3] == insn(data, 2)
    assert stored_disassembly(db) == []


def test_memory_code_object_size_mismatch_is_rejected(tmp_path, code_object):
    path, data = code_object
    db = tmp_path / "pc.db"
    build_db(
        db,
        code_objects=[
            {
                "id": 1,
                "uri": f"{path.as_uri()}#offset=0&size={len(data)}",
                "storage_type": "MEMORY",
                "memory_size": len(data) + 1,
                "content_hash": content_hash(data),
            }
        ],
        samples=[{"timestamp": 100, "offset": 2}],
    )
    rows = run_driver(tmp_path, inputs=[str(db)])
    assert rows[0][3] is None
    assert stored_disassembly(db) == []


def test_writeback_skips_stored_rows_without_an_index(tmp_path):
    data = bytes(range(256)) * 32
    path = tmp_path / "kernel.co"
    path.write_bytes(data)
    db = tmp_path / "pc.db"
    offsets = list(range(4000))
    build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100 + offset, "offset": offset} for offset in offsets],
    )
    table = disassembly_table(db)

    # Another writer stores some of the same PCs while this process still holds them
    # in memory, so its write-back must skip them instead of failing or duplicating.
    run_driver(
        tmp_path,
        inputs=[str(db)],
        concurrent_rows=[[table, offset, "concurrent"] for offset in range(100)],
    )

    assert query_db(db, f'PRAGMA index_list("{table}")') == []
    stored = query_db(db, f'SELECT code_object_offset, instruction FROM "{table}"')
    assert len(stored) == len(offsets)
    by_offset = dict(stored)
    assert len(by_offset) == len(offsets)
    assert all(by_offset[offset] == "concurrent" for offset in range(100))
    assert all(by_offset[offset] == insn(data, offset) for offset in offsets[100:])


def test_deleted_input_is_not_recreated(tmp_path, code_object):
    path, data = code_object
    db = tmp_path / "pc.db"
    build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100, "offset": 1}],
    )
    rows = run_driver(tmp_path, inputs=[str(db)], close_and_delete=True)
    assert rows[0][3] == insn(data, 1)
    assert not db.exists()


def test_merged_database_caches_every_recording(tmp_path):
    from rocpd.merge import merge_sqlite_dbs

    expected = {}
    inputs = []
    for name, data in (("a", bytes(range(64))), ("b", bytes(range(64, 128)))):
        path = tmp_path / f"{name}.co"
        path.write_bytes(data)
        db = tmp_path / f"{name}.db"
        # Identical local ids in both recordings: only guid tells them apart.
        guid = build_db(
            db,
            code_objects=[file_code_object(path, data)],
            samples=[{"timestamp": 100, "offset": 3}],
        )
        expected[guid] = insn(data, 3)
        inputs.append(str(db))
    merged = tmp_path / "merged.db"
    merge_sqlite_dbs(inputs, str(merged))

    rows = run_driver(tmp_path, inputs=[str(merged)])
    assert {row[0]: row[3] for row in rows} == expected
    stored = {(guid, text) for guid, _, _, text in stored_disassembly(merged)}
    assert stored == set(expected.items())


def test_merged_recordings_keep_their_blob_layouts_and_parents(tmp_path):
    from rocpd.merge import merge_sqlite_dbs

    inputs, guids = [], []
    # (schema id, field offset): one id shared with a different layout, one distinct id.
    layouts = ((1, 0), (1, 4), (7, 0))
    for index, (schema_id, offset) in enumerate(layouts):
        db = tmp_path / f"recording{index}.db"
        guid = build_db(db, samples=[{"timestamp": 100}])
        uuid = "_" + guid.replace("-", "_")
        with contextlib.closing(sqlite3.connect(str(db))) as conn:
            conn.execute(
                f'UPDATE "rocpd_info_blob_schema{uuid}" SET id = ?, struct_size = ?',
                (schema_id, offset + 4),
            )
            conn.execute(
                f'UPDATE "rocpd_info_blob_field{uuid}" SET schema_id = ?, "offset" = ?',
                (schema_id, offset),
            )
            conn.execute(
                f'UPDATE "rocpd_blob_event{uuid}" SET schema_id = ?, blob = ?',
                (schema_id, b"\xff" * offset + struct.pack("<I", 1)),
            )
            conn.execute(
                f'INSERT INTO "rocpd_event{uuid}" (id, guid, correlation_id) '
                "VALUES (50, ?, 0)",
                (guid,),
            )
            conn.execute(f'UPDATE "rocpd_event{uuid}" SET parent_id = 50 WHERE id = 1')
            conn.commit()
        inputs.append(str(db))
        guids.append(guid)
    merged = tmp_path / "merged.db"
    merge_sqlite_dbs(inputs, str(merged))

    with open_db([merged]) as connection:
        assert sorted(
            connection.execute(
                "SELECT guid, hw_id_wave_id FROM rocpd_gpu_pc_sample_decoded"
            ).fetchall()
        ) == sorted((guid, 1) for guid in guids)
        # Local event ids repeat across recordings, so parent joins need the guid.
        assert sorted(
            connection.execute(
                "SELECT s.guid, p.guid FROM rocpd_gpu_pc_sample s "
                "JOIN rocpd_event e ON e.guid = s.guid AND e.id = s.event_id "
                "JOIN rocpd_event p ON p.guid = e.guid AND p.id = e.parent_id"
            ).fetchall()
        ) == sorted((guid, guid) for guid in guids)
        assert sorted(
            connection.execute(
                "SELECT guid, event_id, stack_id, parent_id, schema_id, blob, "
                "code_object_uri FROM pc_samples"
            ).fetchall()
        ) == sorted(
            (guid, 1, 1, 50, schema_id, b"\xff" * offset + struct.pack("<I", 1), None)
            for guid, (schema_id, offset) in zip(guids, layouts)
        )


@pytest.mark.parametrize("code_object_id", [0, 77])
def test_pc_samples_view_keeps_unknown_code_objects(tmp_path, code_object_id):
    db = tmp_path / "pc.db"
    guid = build_db(db, samples=[{"timestamp": 100, "code_object_id": code_object_id}])
    uuid = "_" + guid.replace("-", "_")
    with contextlib.closing(sqlite3.connect(str(db))) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f'UPDATE "rocpd_gpu_pc_sample{uuid}" SET tid = NULL, agent_id = NULL'
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        for name in ("rocpd_gpu_pc_sample", "rocpd_disassembly_data"):
            columns = {
                column[1]: column[2]
                for column in connection.execute(f'PRAGMA table_info("{name}{uuid}")')
            }
            assert columns["code_object_id"] == "INTEGER"
        connection.commit()
    with open_db([db]) as connection:
        assert connection.execute(
            "SELECT guid, code_object_id, tid, agent_abs_index, parent_id, "
            "code_object_uri, schema_id, blob FROM pc_samples"
        ).fetchall() == [
            (
                guid,
                code_object_id,
                None,
                None,
                None,
                None,
                PC_SCHEMA_ID,
                struct.pack("<I", 1),
            )
        ]


def test_merge_skips_views_bound_to_one_recording(tmp_path):
    from rocpd.merge import merge_sqlite_dbs

    inputs = []
    for name in ("a", "b"):
        db = tmp_path / f"{name}.db"
        uuid = "_" + build_db(db).replace("-", "_")
        with contextlib.closing(sqlite3.connect(str(db))) as conn:
            conn.execute(
                f'CREATE VIEW rocpd_pinned_strings AS SELECT * FROM "rocpd_string{uuid}"'
            )
            conn.execute("CREATE VIEW rocpd_all_strings AS SELECT * FROM rocpd_string")
            conn.commit()
        inputs.append(str(db))
    merged = tmp_path / "merged.db"
    merge_sqlite_dbs(inputs, str(merged))

    views = {
        name
        for (name,) in query_db(
            merged, "SELECT name FROM sqlite_master WHERE type = 'view'"
        )
    }
    assert "rocpd_pinned_strings" not in views
    assert query_db(merged, "SELECT COUNT(*) FROM rocpd_all_strings") == [(2,)]


@pytest.mark.parametrize("method", ["host_trap", "stochastic"])
def test_pc_csv_identifies_recordings(tmp_path, fake_decoder, method):
    import csv
    from rocpd import csv as rocpd_csv

    data = bytes(range(64))
    path = tmp_path / "kernel.co"
    path.write_bytes(data)
    dbs = [tmp_path / "a.db", tmp_path / "b.db"]
    expected_correlations = {
        build_db(
            db,
            code_objects=[file_code_object(path, data)],
            samples=[
                {
                    "timestamp": 100,
                    "offset": 5,
                    "exec_mask": FULL_EXEC_MASK,
                    "correlation_id": 1000 + index,
                    "stack_id": index,
                    "wave_issued": 1 if method == "stochastic" else None,
                }
            ],
        ): index
        for index, db in enumerate(dbs, start=1)
    }
    out_dir = tmp_path / "out"
    with open_db(dbs) as connection:
        rocpd_csv.execute(connection, output_path=str(out_dir), output_file="out")

    output = out_dir / f"out_pc_sampling_{method}_trace.csv"
    with open(output, newline="") as ifs:
        rows = list(csv.DictReader(ifs))
    assert list(rows[0].keys())[0] == "Guid"
    assert len(rows) == len(expected_correlations)
    assert {
        row["Guid"]: int(row["Correlation_Id"]) for row in rows
    } == expected_correlations
    assert all(row["Instruction"] == insn(data, 5) for row in rows)
    # The full 64-bit mask is exported exactly and unquoted.
    assert f",{FULL_EXEC_MASK}," in output.read_text()


def test_generic_csv_export_keeps_fractional_exec_mask(tmp_path):
    from rocpd.query import export_sqlite_query

    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        output = export_sqlite_query(
            conn,
            "SELECT 1.5 AS Exec_Mask",
            export_format="csv",
            export_path=str(tmp_path / "generic"),
        )
    assert output is not None
    assert open(output).read().splitlines()[1] == "1.5"


@pytest.mark.parametrize("start, end", [("125", "175"), ("25%", "75%")])
def test_time_window_on_pc_only_database(tmp_path, start, end):
    from rocpd.time_window import apply_time_window

    db = tmp_path / "pc.db"
    build_db(db, samples=[{"timestamp": timestamp} for timestamp in (100, 150, 200)])
    with open_db([db]) as connection:
        apply_time_window(connection, start=start, end=end)
        assert connection.execute(
            "SELECT timestamp FROM rocpd_gpu_pc_sample"
        ).fetchall() == [(150,)]
        assert connection.execute(
            "SELECT timestamp FROM rocpd_gpu_pc_sample_decoded"
        ).fetchall() == [(150,)]


def test_time_window_without_timed_records_reports_error(tmp_path):
    from rocpd.time_window import apply_time_window

    db = tmp_path / "empty.db"
    build_db(db)
    with open_db([db]) as connection:
        with pytest.raises(ValueError, match="no timing data"):
            apply_time_window(connection, start="10%", end="90%")


@pytest.mark.parametrize(
    "text, kind",
    [
        ('"quoted""identifier"', "identifier"),
        ("`quoted``identifier`", "identifier"),
        ("[quoted identifier]", "identifier"),
        ("'WITH fake AS (SELECT ''FROM rocpd_gpu_pc_sample'')'", "string"),
        ("/* WITH fake AS (SELECT * FROM rocpd_gpu_pc_sample) */", "comment"),
        ("-- WITH fake AS (SELECT * FROM rocpd_gpu_pc_sample)", "comment"),
    ],
)
def test_sql_tokenizer_preserves_quoted_spans(text, kind):
    from rocpd.query import _sql_tokens

    query = f"SELECT {text}\nFROM rocpd_gpu_pc_sample"
    tokens = list(_sql_tokens(query))
    assert "".join(token[1] for token in tokens) == query
    assert (kind, text, 7, 7 + len(text)) in tokens
    assert all(query[start:end] == value for _, value, start, end in tokens)


def test_cte_names_include_quoted_nested_definitions():
    from rocpd.query import _cte_names, _sql_tokens

    query = (
        'WITH RECURSIVE "Outer"(value) AS NOT MATERIALIZED ('
        "WITH [Inner](value) AS (SELECT 1) SELECT value FROM [Inner]), "
        '`Escaped``Name` AS (SELECT "WITH fake AS (SELECT 1)" FROM "Outer") '
        'SELECT * FROM "Outer" /* WITH ignored AS (SELECT 2) */'
    )
    assert _cte_names(list(_sql_tokens(query))) == {"outer", "inner", "escaped`name"}


def test_query_rewrite_keeps_qualified_columns_and_ctes(tmp_path):
    from rocpd.query import update_query_for_blob_views

    db = tmp_path / "pc.db"
    build_db(db, samples=[{"timestamp": 100}, {"timestamp": 200}])
    with open_db([db]) as connection:

        def run(sql):
            rewritten = update_query_for_blob_views(connection, sql)
            return connection.execute(rewritten).fetchall()

        assert run(
            "SELECT rocpd_gpu_pc_sample.hw_id_wave_id FROM rocpd_gpu_pc_sample ORDER BY 1"
        ) == [(1,), (2,)]
        assert run(
            'SELECT "rocpd_gpu_pc_sample".timestamp FROM "rocpd_gpu_pc_sample" '
            "WHERE rocpd_gpu_pc_sample.timestamp > 150"
        ) == [(200,)]
        assert run("SELECT s.hw_id_wave_id FROM rocpd_gpu_pc_sample AS s ORDER BY 1") == [
            (1,),
            (2,),
        ]
        assert run("SELECT s.timestamp FROM rocpd_gpu_pc_sample s ORDER BY 1") == [
            (100,),
            (200,),
        ]
        # A CTE that shadows the source table keeps its own meaning at any depth.
        assert run(
            "SELECT * FROM (WITH rocpd_gpu_pc_sample AS (SELECT 42 AS timestamp) "
            "SELECT timestamp FROM rocpd_gpu_pc_sample)"
        ) == [(42,)]
        materialized_query = (
            "WITH a AS MATERIALIZED (SELECT 1), rocpd_gpu_pc_sample AS "
            "(SELECT 9 AS timestamp) SELECT timestamp FROM rocpd_gpu_pc_sample"
        )
        assert (
            update_query_for_blob_views(connection, materialized_query)
            == materialized_query
        )
        if sqlite3.sqlite_version_info >= (3, 35, 0):
            assert run(materialized_query) == [(9,)]
        assert run(
            'WITH "outer" AS (WITH [inner] AS '
            '(SELECT hw_id_wave_id FROM "rocpd_gpu_pc_sample") '
            'SELECT * FROM [inner]) SELECT * FROM "outer" ORDER BY 1'
        ) == [(1,), (2,)]
        assert run(
            'WITH "ROCPD_GPU_PC_SAMPLE"(timestamp) AS (SELECT 42) '
            'SELECT timestamp FROM "ROCPD_GPU_PC_SAMPLE"'
        ) == [(42,)]
        assert run(
            'SELECT "escaped""alias".hw_id_wave_id FROM "rocpd_gpu_pc_sample" '
            'AS "escaped""alias" ORDER BY 1'
        ) == [(1,), (2,)]


def test_query_rewrite_handles_quoting_qualifiers_comments_and_literals(tmp_path):
    from rocpd.query import update_query_for_blob_views

    db = tmp_path / "pc.db"
    build_db(db, samples=[{"timestamp": 100}, {"timestamp": 200}])
    with open_db([db]) as connection:

        def rewrite(sql):
            return update_query_for_blob_views(connection, sql)

        def run(sql):
            return connection.execute(rewrite(sql)).fetchall()

        for quoted in ("[rocpd_gpu_pc_sample]", "`rocpd_gpu_pc_sample`"):
            assert run(f"SELECT {quoted}.hw_id_wave_id FROM {quoted} ORDER BY 1") == [
                (1,),
                (2,),
            ]
        # Every table factor of a comma-separated FROM list is rewritten.
        assert run(
            "SELECT a.hw_id_wave_id, b.hw_id_wave_id "
            "FROM rocpd_gpu_pc_sample a, rocpd_gpu_pc_sample AS b "
            "WHERE a.id = b.id ORDER BY 1"
        ) == [(1, 1), (2, 2)]
        # temp-qualified names are rewritten; other schema qualifiers are not.
        assert run("SELECT hw_id_wave_id FROM temp.rocpd_gpu_pc_sample ORDER BY 1") == [
            (1,),
            (2,),
        ]
        main_query = "SELECT timestamp FROM main.rocpd_gpu_pc_sample"
        assert rewrite(main_query) == main_query
        # Comments and string literals that mention the table are left alone.
        commented = (
            "SELECT hw_id_wave_id /* FROM rocpd_gpu_pc_sample */ "
            "FROM rocpd_gpu_pc_sample -- JOIN rocpd_gpu_pc_sample\nORDER BY 1"
        )
        rewritten = rewrite(commented)
        assert "/* FROM rocpd_gpu_pc_sample */" in rewritten
        assert "-- JOIN rocpd_gpu_pc_sample\n" in rewritten
        assert connection.execute(rewritten).fetchall() == [(1,), (2,)]
        literal = "FROM rocpd_gpu_pc_sample"
        assert run(
            f"SELECT '{literal}', hw_id_wave_id FROM rocpd_gpu_pc_sample ORDER BY 2"
        ) == [(literal, 1), (literal, 2)]


def test_per_architecture_layouts_share_one_decoded_view(tmp_path):
    db = tmp_path / "pc.db"
    guid = build_db(db, samples=[{"timestamp": 100}])
    uuid = "_" + guid.replace("-", "_")
    # A second stochastic layout, as the writer registers for a GFX1250 sample with
    # memory counters: the shared field plus architecture-specific ones.
    gfx1250_schema_id = 3
    fields = [
        ("hw_id_wave_id", 0, 4, "uint32_t"),
        ("sampling_lock_error", 4, 1, "uint8_t"),
        ("memory_counter_xnack", 5, 1, "uint8_t"),
    ]
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            f'INSERT INTO "rocpd_info_blob_schema{uuid}" (id, guid, nid, pid, name, '
            "source_table, byte_order, alignment, struct_size, version) VALUES "
            "(?, ?, ?, ?, 'pc_sample_extdata_stochastic_gfx1250', "
            "'rocpd_gpu_pc_sample', 'little', 1, 6, 1)",
            (gfx1250_schema_id, guid, NID, PID),
        )
        conn.executemany(
            f'INSERT INTO "rocpd_info_blob_field{uuid}" (guid, schema_id, name, '
            "offset, size, data_type, is_signed) VALUES (?, ?, ?, ?, ?, ?, 0)",
            [(guid, gfx1250_schema_id, *field) for field in fields],
        )
        conn.execute(
            f'INSERT INTO "rocpd_event{uuid}" (id, guid, correlation_id) VALUES (2, ?, 2)',
            (guid,),
        )
        conn.execute(
            f'INSERT INTO "rocpd_blob_event{uuid}" (guid, nid, pid, event_id, '
            "schema_id, blob) VALUES (?, ?, ?, 2, ?, ?)",
            (guid, NID, PID, gfx1250_schema_id, struct.pack("<IBB", 2, 1, 5)),
        )
        conn.execute(
            f'INSERT INTO "rocpd_gpu_pc_sample{uuid}" (id, guid, timestamp, nid, pid, '
            "event_id, dispatch_id, correlation_id, exec_mask, code_object_id, "
            "code_object_offset) VALUES (2, ?, 200, ?, ?, 2, 1, 2, '1', 1, 0)",
            (guid, NID, PID),
        )
        conn.commit()

    with open_db([db]) as connection:
        assert connection.execute(
            "SELECT hw_id_wave_id, sampling_lock_error, memory_counter_xnack "
            "FROM rocpd_gpu_pc_sample_decoded ORDER BY id"
        ).fetchall() == [(1, None, None), (2, 1, 5)]


def test_shared_event_blobs_do_not_multiply_rows(tmp_path):
    db = tmp_path / "pc.db"
    build_db(
        db,
        samples=[{"timestamp": 100, "event_id": 1}, {"timestamp": 200, "event_id": 2}],
        regions=[{"event_id": 1, "start": 90, "end": 110}],
    )
    with open_db([db]) as connection:
        assert connection.execute(
            "SELECT hw_id_wave_id FROM rocpd_gpu_pc_sample_decoded ORDER BY id"
        ).fetchall() == [(1,), (2,)]
        assert connection.execute(
            "SELECT region_flag FROM rocpd_region_decoded"
        ).fetchall() == [(7,)]


@pytest.mark.parametrize("bypass_column_check", [False, True])
def test_failed_isa_setup_publishes_no_partial_view(
    tmp_path, monkeypatch, capsys, bypass_column_check
):
    import rocpd.importer as importer
    from rocpd import csv as rocpd_csv

    db = tmp_path / "pc.db"
    build_db(db, samples=[{"timestamp": 100}])
    table = disassembly_table(db)
    with contextlib.closing(sqlite3.connect(str(db))) as conn:
        conn.execute(f'DROP TABLE "{table}"')
        conn.execute(
            f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, guid TEXT, nid INTEGER, '
            "pid INTEGER, code_object_id BIGINT, code_object_offset BIGINT, comment TEXT)"
        )
        conn.commit()

    if bypass_column_check:
        # Let the view body reference the missing column so only the zero-row
        # validation can reject it.
        table_columns = importer._table_columns

        def report_instruction_column(conn, name):
            columns = table_columns(conn, name)
            if name == "rocpd_disassembly_data" and columns:
                return columns + ["instruction"]
            return columns

        monkeypatch.setattr(importer, "_table_columns", report_instruction_column)

    out_dir = tmp_path / "out"
    with open_db([db]) as connection:
        views = {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_temp_master WHERE type = 'view'"
            ).fetchall()
        }
        assert "rocpd_gpu_pc_sample_decoded" not in views
        assert connection.execute(
            "SELECT hw_id_wave_id FROM rocpd_gpu_pc_sample_blob_decoded"
        ).fetchall() == [(1,)]
        # Only the PC-sampling CSV files are skipped.
        rocpd_csv.execute(connection, output_path=str(out_dir), output_file="out")
    assert not list(out_dir.glob("out_pc_sampling_*"))
    assert (out_dir / "out_agent_info.csv").exists()
    assert "skipping decoded blob/ISA views" in capsys.readouterr().err


def test_pc_csv_skips_decoded_view_without_isa_columns(tmp_path):
    from rocpd import csv as rocpd_csv

    db = tmp_path / "pc.db"
    build_db(db, samples=[{"timestamp": 100}])
    out_dir = tmp_path / "out"
    with open_db([db]) as connection:
        connection.execute("DROP VIEW temp.rocpd_gpu_pc_sample_decoded")
        connection.execute(
            "CREATE TEMP VIEW rocpd_gpu_pc_sample_decoded AS "
            "SELECT * FROM rocpd_gpu_pc_sample_blob_decoded"
        )
        rocpd_csv.execute(connection, output_path=str(out_dir), output_file="out")
    assert not list(out_dir.glob("out_pc_sampling_*"))
    assert (out_dir / "out_agent_info.csv").exists()


def test_csv_and_query_launchers_accept_decode_options(tmp_path, code_object):
    path, data = code_object
    db = tmp_path / "pc.db"
    guid = build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100, "offset": 6}],
    )
    original = db.read_bytes()

    read_only_dir = tmp_path / "ro"
    run_driver(
        tmp_path,
        csv_main=[
            "-i",
            str(db),
            "-d",
            str(read_only_dir),
            "-o",
            "out",
            "--no-cache-disassembly",
        ],
    )
    csv_text = (read_only_dir / "out_pc_sampling_host_trap_trace.csv").read_text()
    assert insn(data, 6) in csv_text
    run_driver(
        tmp_path,
        query_main=[
            "-i",
            str(db),
            "--query",
            "SELECT instruction FROM rocpd_gpu_pc_sample",
            "--no-cache-disassembly",
            "--codeobj-path-prefix-map",
            "/unused=/unused",
        ],
    )
    assert db.read_bytes() == original

    read_write_dir = tmp_path / "rw"
    run_driver(tmp_path, csv_main=["-i", str(db), "-d", str(read_write_dir), "-o", "out"])
    assert stored_disassembly(db) == [(guid, 1, 6, insn(data, 6))]


def test_open_without_decoding_does_not_scan_samples(tmp_path, monkeypatch, code_object):
    import rocpd.importer as importer

    statements = []
    connect = importer.libpyrocpd.connect

    def traced_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(lambda statement: statements.append(statement))
        return connection

    monkeypatch.setattr(importer.libpyrocpd, "connect", traced_connect)

    path, data = code_object
    pc_db = tmp_path / "pc.db"
    build_db(
        pc_db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100}],
    )
    with open_db([pc_db], cache_disassembly=True) as connection:
        connection.execute("SELECT COUNT(*) FROM rocpd_gpu_pc_sample").fetchone()
    assert not [
        sql for sql in statements if "GROUP BY" in sql and "rocpd_gpu_pc_sample" in sql
    ]

    statements.clear()
    empty_db = tmp_path / "empty.db"
    build_db(empty_db, code_objects=[{"id": 1, "uri": path.as_uri()}])
    with open_db([empty_db], cache_disassembly=True):
        pass
    assert not [sql for sql in statements if "$.content_hash" in sql]


@pytest.mark.parametrize("cache_disassembly", [False, True])
def test_each_pc_is_decoded_once(
    tmp_path, monkeypatch, counting_decoder, cache_disassembly
):
    import rocpd.importer as importer

    if cache_disassembly:
        # Results evicted from a small LRU must come from the pending write-back.
        monkeypatch.setattr(importer, "_MAX_DECODED_INSTRUCTIONS", 16)
    data = bytes(range(256)) * 20
    path = tmp_path / "kernel.co"
    path.write_bytes(data)
    db = tmp_path / "pc.db"
    offsets = range(5000)
    build_db(
        db,
        code_objects=[file_code_object(path, data)],
        samples=[{"timestamp": 100 + offset, "offset": offset} for offset in offsets],
    )
    expected = [(insn(data, offset),) for offset in offsets]
    with open_db([db], cache_disassembly=cache_disassembly) as connection:
        for _ in range(2):
            assert (
                connection.execute(
                    "SELECT instruction FROM rocpd_gpu_pc_sample_decoded ORDER BY id"
                ).fetchall()
                == expected
            )
    assert counting_decoder.decodes == len(offsets)


def test_decoder_cache_is_bounded_by_bytes(tmp_path, monkeypatch, counting_decoder):
    import rocpd.importer as importer

    # Two 64-byte code objects under a 100-byte budget: one decoder at a time.
    monkeypatch.setattr(importer, "_MAX_DECODER_CACHE_BYTES", 100)
    code_objects = []
    for code_object_id in (1, 2):
        data = bytes([code_object_id]) * 64
        path = tmp_path / f"kernel{code_object_id}.co"
        path.write_bytes(data)
        code_objects.append(
            {
                "id": code_object_id,
                "uri": path.as_uri(),
                "content_hash": content_hash(data),
            }
        )
    db = tmp_path / "pc.db"
    build_db(
        db,
        code_objects=code_objects,
        samples=[
            {
                "timestamp": 100 + index,
                "code_object_id": 1 + index % 2,
                "offset": index // 2,
            }
            for index in range(4)
        ],
    )
    with open_db([db]) as connection:
        rows = connection.execute(
            "SELECT instruction FROM rocpd_gpu_pc_sample_decoded ORDER BY id"
        ).fetchall()
    assert rows == [("insn_01",), ("insn_02",), ("insn_01",), ("insn_02",)]
    assert counting_decoder.builds == 4
