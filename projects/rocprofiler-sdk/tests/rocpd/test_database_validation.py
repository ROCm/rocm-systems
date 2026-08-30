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

import sqlite3
import stat
import tempfile
from contextlib import closing
from pathlib import Path
import unittest

from rocpd.database import (
    attach_readonly,
    configure_untrusted_schema,
    inspect_attached_rocpd,
)
from rocpd.features import get_supported_features_from_version
from rocpd.importer import RocpdImportData, _create_meta_views, _create_temp_views
from rocpd import merge as merge_module
from rocpd.merge import merge_sqlite_dbs
from rocpd.schema import RocpdSchema, query_supported_schema_versions


class RocpdDatabaseValidationTest(unittest.TestCase):
    UUIDS = (
        "_12345678_1234_7123_8123_123456789abc",
        "_22345678_1234_7123_8123_123456789abc",
    )
    GUIDS = (
        "12345678-1234-7123-8123-123456789abc",
        "22345678-1234-7123-8123-123456789abc",
    )

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)
        self.version = str(query_supported_schema_versions()[-1])

    def tearDown(self):
        self._temporary_directory.cleanup()

    def create_database(self, path, uuid=None, guid=None, version=None):
        uuid = uuid or self.UUIDS[0]
        guid = guid or self.GUIDS[0]
        version = version or self.version
        schema = RocpdSchema(uuid=uuid, guid=guid, version=version)
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.executescript(schema.tables)
            connection.executescript(schema.metadata)
            connection.executescript(schema.indexes)
            connection.executescript(schema.views)
            connection.execute(
                f'INSERT INTO "rocpd_string{uuid}" (guid, string) VALUES (?, ?)',
                (guid, f"value-{uuid}"),
            )
        return schema

    def add_region_data(self, path, uuid, guid, index):
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                f'INSERT INTO "rocpd_string{uuid}" (id, guid, string) '
                "VALUES (2, ?, ?)",
                (guid, "HIP_API"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_string{uuid}" (id, guid, string) '
                "VALUES (3, ?, ?)",
                (guid, f"region-{index}"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_node{uuid}" '
                "(id, guid, hash, machine_id, hostname) VALUES (1, ?, ?, ?, ?)",
                (guid, 100 + index, f"machine-{index}", f"host-{index}"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_process{uuid}" '
                "(id, guid, nid, ppid, pid, init, fini, start, end, command) "
                "VALUES (1, ?, 1, 1, ?, 100, 900, 100, 900, ?)",
                (guid, 1000 + index, f"process-{index}"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_thread{uuid}" '
                "(id, guid, nid, ppid, pid, tid, name, start, end) "
                "VALUES (1, ?, 1, 1, 1, ?, ?, 100, 900)",
                (guid, 2000 + index, f"thread-{index}"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_event{uuid}" '
                "(id, guid, category_id, correlation_id) VALUES (1, ?, 2, 1)",
                (guid,),
            )
            connection.execute(
                f'INSERT INTO "rocpd_region{uuid}" '
                "(id, guid, nid, pid, tid, start, end, name_id, event_id) "
                "VALUES (1, ?, 1, 1, 1, 200, 300, 3, 1)",
                (guid,),
            )

    def test_merge_uses_trusted_schema_and_preserves_constraints(self):
        source = self.directory / "source.db"
        second_source = self.directory / "second-source.db"
        destination = self.directory / "merged.db"
        self.create_database(source)
        self.create_database(second_source, uuid=self.UUIDS[1], guid=self.GUIDS[1])
        self.add_region_data(source, self.UUIDS[0], self.GUIDS[0], 0)
        self.add_region_data(second_source, self.UUIDS[1], self.GUIDS[1], 1)

        with closing(sqlite3.connect(source)) as connection, connection:
            source_table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (f"rocpd_string{self.UUIDS[0]}",),
            ).fetchone()[0]
            connection.execute("DROP VIEW processes")
            connection.execute(
                "CREATE VIEW processes AS "
                "SELECT randomblob(1000000) AS attacker_expression"
            )

        merge_sqlite_dbs([str(source), str(second_source)], str(destination))

        with closing(sqlite3.connect(destination)) as connection, connection:
            merged_table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (f"rocpd_string{self.UUIDS[0]}",),
            ).fetchone()[0]
            merged_view_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='view' AND name='processes'"
            ).fetchone()[0]
            self.assertEqual(merged_table_sql, source_table_sql)
            self.assertNotIn("randomblob", merged_view_sql.lower())
            self.assertEqual(
                connection.execute(
                    "SELECT string FROM rocpd_string "
                    "WHERE string LIKE 'value-%' ORDER BY string"
                ).fetchall(),
                sorted(
                    [
                        (f"value-{self.UUIDS[0]}",),
                        (f"value-{self.UUIDS[1]}",),
                    ]
                ),
            )
            self.assertEqual(
                connection.execute("PRAGMA foreign_key_check").fetchall(), []
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM regions").fetchone(), (2,)
            )

        imported = RocpdImportData(str(destination), skip_auto_merge=True)
        try:
            self.assertEqual(
                imported.connection.execute(
                    "SELECT string FROM rocpd_string "
                    "WHERE string LIKE 'value-%' ORDER BY string"
                ).fetchall(),
                sorted(
                    [
                        (f"value-{self.UUIDS[0]}",),
                        (f"value-{self.UUIDS[1]}",),
                    ]
                ),
            )
        finally:
            imported.connection.close()

        remerged = self.directory / "remerged.db"
        merge_sqlite_dbs([str(destination)], str(remerged))
        with closing(sqlite3.connect(remerged)) as connection, connection:
            self.assertEqual(
                connection.execute(
                    "SELECT string FROM rocpd_string "
                    "WHERE string LIKE 'value-%' ORDER BY string"
                ).fetchall(),
                sorted(
                    [
                        (f"value-{self.UUIDS[0]}",),
                        (f"value-{self.UUIDS[1]}",),
                    ]
                ),
            )

    def test_trigger_is_rejected_without_replacing_destination(self):
        source = self.directory / "trigger.db"
        destination = self.directory / "existing.db"
        self.create_database(source)
        destination.write_bytes(b"existing destination")

        with closing(sqlite3.connect(source)) as connection, connection:
            connection.execute(f"""
                CREATE TRIGGER attacker_trigger
                AFTER INSERT ON "rocpd_metadata{self.UUIDS[0]}"
                BEGIN
                    DELETE FROM "rocpd_metadata{self.UUIDS[0]}";
                END
                """)

        with self.assertRaisesRegex(ValueError, "trigger"):
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertEqual(destination.read_bytes(), b"existing destination")

    def test_unexpected_tables_views_and_schema_changes_are_rejected(self):
        mutations = {
            "table": "CREATE TABLE rocpd_attacker_controlled (payload TEXT)",
            "view": "CREATE VIEW attacker_view AS SELECT 1",
            "index": (
                f'CREATE INDEX attacker_index ON "rocpd_string{self.UUIDS[0]}" '
                "(string)"
            ),
            "missing_view": "DROP VIEW processes",
            "column": (
                f'ALTER TABLE "rocpd_string{self.UUIDS[0]}" '
                "ADD COLUMN attacker_value TEXT"
            ),
        }
        for name, statement in mutations.items():
            with self.subTest(name=name):
                source = self.directory / f"{name}.db"
                destination = self.directory / f"{name}-merged.db"
                self.create_database(source)
                with closing(sqlite3.connect(source)) as connection, connection:
                    connection.execute(statement)
                with self.assertRaises(ValueError):
                    merge_sqlite_dbs([str(source)], str(destination))
                self.assertFalse(destination.exists())

    def test_identifier_breakout_is_rejected(self):
        source = self.directory / "identifier.db"
        destination = self.directory / "identifier-merged.db"
        payload = self.directory / "payload.db"
        self.create_database(source)
        injected_name = (
            'rocpd_x"; ATTACH DATABASE ' + repr(str(payload)) + " AS attacker; --"
        )
        with closing(sqlite3.connect(source)) as connection, connection:
            quoted_name = '"' + injected_name.replace('"', '""') + '"'
            connection.execute(f"CREATE TABLE {quoted_name} (value INTEGER)")

        with self.assertRaisesRegex(ValueError, "identifier"):
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertFalse(payload.exists())

    def test_import_uses_read_only_tables_and_trusted_views(self):
        source = self.directory / "source'with-quote.db"
        self.create_database(source)
        with closing(sqlite3.connect(source)) as connection, connection:
            connection.execute("DROP VIEW processes")
            connection.execute(
                "CREATE VIEW processes AS SELECT randomblob(1000000) AS payload"
            )

        with closing(sqlite3.connect(":memory:", uri=True)) as connection, connection:
            configure_untrusted_schema(connection)
            table_info = _create_temp_views(connection, [str(source)])
            _create_meta_views(connection, table_info.schema_version)
            view_sql = connection.execute(
                "SELECT sql FROM sqlite_temp_master "
                "WHERE type='view' AND name='processes'"
            ).fetchone()[0]
            self.assertNotIn("randomblob", view_sql.lower())
            self.assertEqual(
                connection.execute("SELECT string FROM rocpd_string").fetchall(),
                [(f"value-{self.UUIDS[0]}",)],
            )
            with self.assertRaisesRegex(sqlite3.OperationalError, "readonly"):
                connection.execute(
                    f'INSERT INTO db0."rocpd_string{self.UUIDS[0]}" '
                    "(guid, string) VALUES (?, ?)",
                    (self.GUIDS[0], "write-attempt"),
                )

        imported = RocpdImportData(str(source), skip_auto_merge=True)
        try:
            self.assertEqual(str(imported.schema_version), self.version)
            self.assertEqual(
                set(imported.supported_features),
                set(get_supported_features_from_version(imported.schema_version)),
            )
            self.assertEqual(
                imported.connection.execute("SELECT string FROM rocpd_string").fetchall(),
                [(f"value-{self.UUIDS[0]}",)],
            )
        finally:
            imported.connection.close()

    def test_duplicate_uuids_and_mixed_versions_are_rejected(self):
        first = self.directory / "first.db"
        second = self.directory / "second.db"
        destination = self.directory / "merged.db"
        self.create_database(first)
        self.create_database(second)
        with self.assertRaisesRegex(ValueError, "Duplicate rocPD UUID"):
            merge_sqlite_dbs([str(first), str(second)], str(destination))

        versions = [str(item) for item in query_supported_schema_versions()]
        if len(versions) > 1:
            second.unlink()
            other_version = next(
                version for version in versions if version != self.version
            )
            self.create_database(
                second,
                uuid=self.UUIDS[1],
                guid=self.GUIDS[1],
                version=other_version,
            )
            with self.assertRaisesRegex(RuntimeError, "Multiple schema versions"):
                merge_sqlite_dbs([str(first), str(second)], str(destination))

    def test_additional_metadata_is_preserved(self):
        source = self.directory / "extended-metadata.db"
        destination = self.directory / "extended-metadata-merged.db"
        self.create_database(source)
        with closing(sqlite3.connect(source)) as connection, connection:
            connection.execute(
                f'INSERT INTO "rocpd_metadata{self.UUIDS[0]}" (tag, value) '
                "VALUES (?, ?)",
                ("custom_test_metadata", "preserved-value"),
            )

        merge_sqlite_dbs([str(source)], str(destination))
        imported = RocpdImportData(str(destination), skip_auto_merge=True)
        try:
            self.assertEqual(
                imported.connection.execute(
                    "SELECT value FROM rocpd_metadata WHERE tag = ?",
                    ("custom_test_metadata",),
                ).fetchall(),
                [("preserved-value",)],
            )
        finally:
            imported.connection.close()

    def test_all_supported_schemas_pass_validation(self):
        for version_index, version in enumerate(query_supported_schema_versions()):
            version = str(version)
            source = self.directory / f"schema-{version}.db"
            uuid = f"_{version_index + 1:032x}"
            guid = f"guid-{version_index + 1}"
            self.create_database(source, uuid=uuid, guid=guid, version=version)
            with closing(sqlite3.connect(":memory:", uri=True)) as connection, connection:
                configure_untrusted_schema(connection)
                attach_readonly(connection, str(source), "source")
                inspected = inspect_attached_rocpd(connection, "source", str(source))
                self.assertEqual(inspected.version, version)

    def test_standalone_merge_entry_point(self):
        source = self.directory / "cli input.db"
        self.create_database(source)
        merge_module.main(
            [
                "-i",
                str(source),
                "-d",
                str(self.directory),
                "-o",
                "cli-output",
            ]
        )
        output = self.directory / "cli-output.db"
        with closing(sqlite3.connect(output)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT string FROM rocpd_string").fetchall(),
                [(f"value-{self.UUIDS[0]}",)],
            )

    def test_uncheckpointed_wal_is_rejected(self):
        source = self.directory / "live-wal.db"
        destination = self.directory / "existing-wal-output.db"
        self.create_database(source)
        destination.write_bytes(b"existing destination")

        with closing(sqlite3.connect(source)) as writer:
            writer.execute("PRAGMA journal_mode = WAL")
            writer.execute("PRAGMA wal_autocheckpoint = 0")
            writer.execute(
                f'INSERT INTO "rocpd_string{self.UUIDS[0]}" (guid, string) '
                "VALUES (?, ?)",
                (self.GUIDS[0], "uncheckpointed"),
            )
            writer.commit()
            self.assertGreater(Path(f"{source}-wal").stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "uncheckpointed"):
                merge_sqlite_dbs([str(source)], str(destination))

        self.assertEqual(destination.read_bytes(), b"existing destination")

    def test_output_permissions_follow_existing_mode_or_process_umask(self):
        source = self.directory / "permissions-source.db"
        self.create_database(source)

        existing = self.directory / "existing-mode.db"
        existing.write_bytes(b"replace me")
        existing.chmod(0o640)
        merge_sqlite_dbs([str(source)], str(existing))
        self.assertEqual(stat.S_IMODE(existing.stat().st_mode), 0o640)

        mode_reference = self.directory / "mode-reference.db"
        with closing(sqlite3.connect(mode_reference)) as connection, connection:
            connection.execute("CREATE TABLE mode_reference (value INTEGER)")
        expected_new_mode = stat.S_IMODE(mode_reference.stat().st_mode)
        new_output = self.directory / "new-mode.db"
        merge_sqlite_dbs([str(source)], str(new_output))
        self.assertEqual(stat.S_IMODE(new_output.stat().st_mode), expected_new_mode)


if __name__ == "__main__":
    unittest.main()
