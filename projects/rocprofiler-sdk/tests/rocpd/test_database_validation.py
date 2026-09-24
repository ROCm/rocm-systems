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

import io
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
from contextlib import closing, redirect_stderr
from pathlib import Path
import unittest
from unittest import mock

from rocpd.database import (
    attach_readonly,
    configure_untrusted_schema,
    inspect_attached_rocpd,
)
from rocpd.features import get_supported_features_from_version
from rocpd.importer import RocpdImportData, _create_meta_views, _create_temp_views
from rocpd import database as database_module
from rocpd import merge as merge_module
from rocpd import package as package_module
from rocpd.merge import merge_sqlite_dbs
from rocpd.schema import RocpdSchema, query_supported_schema_versions


class RocpdDatabaseValidationTest(unittest.TestCase):
    UUIDS = (
        "_12345678_1234_7123_8123_123456789abc",
        "_22345678_1234_7123_8123_123456789abc",
        "_32345678_1234_7123_8123_123456789abc",
    )
    GUIDS = (
        "12345678-1234-7123-8123-123456789abc",
        "22345678-1234-7123-8123-123456789abc",
        "32345678-1234-7123-8123-123456789abc",
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
        with closing(sqlite3.connect(str(path))) as connection, connection:
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
        with closing(sqlite3.connect(str(path))) as connection, connection:
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

        with closing(sqlite3.connect(str(source))) as connection, connection:
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

        with closing(sqlite3.connect(str(destination))) as connection, connection:
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
        with closing(sqlite3.connect(str(remerged))) as connection, connection:
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

    def test_trigger_is_ignored_and_not_replayed(self):
        source = self.directory / "trigger.db"
        destination = self.directory / "existing.db"
        self.create_database(source)
        destination.write_bytes(b"existing destination")

        with closing(sqlite3.connect(str(source))) as connection, connection:
            connection.execute(f"""
                CREATE TRIGGER attacker_trigger
                AFTER INSERT ON "rocpd_metadata{self.UUIDS[0]}"
                BEGIN
                    DELETE FROM "rocpd_metadata{self.UUIDS[0]}";
                END
                """)

        with redirect_stderr(io.StringIO()) as stderr:
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertIn("attacker_trigger", stderr.getvalue())
        with closing(sqlite3.connect(str(destination))) as connection, connection:
            self.assertEqual(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                ).fetchall(),
                [],
            )
            # Writes to the merged output are unaffected by the source trigger.
            connection.execute(
                f'INSERT INTO "rocpd_metadata{self.UUIDS[0]}" (tag, value) '
                "VALUES ('after-merge', 'kept')"
            )
            self.assertGreater(
                connection.execute("SELECT COUNT(*) FROM rocpd_metadata").fetchone()[0],
                1,
            )

    def test_unfamiliar_objects_are_ignored_and_not_copied(self):
        mutations = {
            "table": ("CREATE TABLE rocpd_attacker_controlled (payload TEXT)",),
            "sqlite_prefix": ("CREATE TABLE sqliteXattacker (payload TEXT)",),
            "view": ("CREATE VIEW attacker_view AS SELECT randomblob(1000000)",),
            "index": (
                f'CREATE INDEX attacker_index ON "rocpd_string{self.UUIDS[0]}" '
                "(string)",
            ),
        }
        for name, statements in mutations.items():
            with self.subTest(name=name):
                source = self.directory / f"{name}.db"
                destination = self.directory / f"{name}-merged.db"
                self.create_database(source)
                with closing(sqlite3.connect(str(source))) as connection, connection:
                    for statement in statements:
                        connection.execute(statement)
                with closing(sqlite3.connect(":memory:", uri=True)) as connection:
                    attach_readonly(connection, str(source), "source")
                    with redirect_stderr(io.StringIO()):
                        inspected = inspect_attached_rocpd(
                            connection, "source", str(source)
                        )
                self.assertEqual(len(inspected.ignored), 1)
                with redirect_stderr(io.StringIO()):
                    merge_sqlite_dbs([str(source)], str(destination))
                with closing(sqlite3.connect(str(destination))) as connection:
                    names = {
                        row[0]
                        for row in connection.execute("SELECT name FROM sqlite_master")
                    }
                self.assertFalse(
                    names
                    & {
                        "rocpd_attacker_controlled",
                        "sqliteXattacker",
                        "attacker_view",
                        "attacker_index",
                    }
                )

    def test_required_table_changes_are_rejected(self):
        table = f"rocpd_string{self.UUIDS[0]}"
        mutations = {
            "missing_table": (f'DROP TABLE "rocpd_arg{self.UUIDS[0]}"',),
            "added_column": (f'ALTER TABLE "{table}" ADD COLUMN attacker_value TEXT',),
            "renamed_column": (
                # Skip rewriting the canonical views, which are optional.
                "PRAGMA legacy_alter_table = ON",
                f'ALTER TABLE "{table}" RENAME COLUMN string TO text',
            ),
            "generated_column": (
                f'ALTER TABLE "{table}" ADD COLUMN generated '
                "TEXT GENERATED ALWAYS AS (upper(string)) VIRTUAL",
            ),
        }
        for name, statements in mutations.items():
            with self.subTest(name=name):
                source = self.directory / f"{name}.db"
                destination = self.directory / f"{name}-merged.db"
                self.create_database(source)
                with closing(sqlite3.connect(str(source))) as connection, connection:
                    for statement in statements:
                        connection.execute(statement)
                with self.assertRaisesRegex(ValueError, "rocPD table"):
                    merge_sqlite_dbs([str(source)], str(destination))
                self.assertFalse(destination.exists())
                with self.assertRaisesRegex(ValueError, "rocPD table"):
                    imported = RocpdImportData(str(source), skip_auto_merge=True)
                    imported.connection.close()

    def test_view_in_place_of_required_table_is_rejected(self):
        source = self.directory / "view-as-table.db"
        destination = self.directory / "view-as-table-merged.db"
        self.create_database(source)
        table = f"rocpd_arg{self.UUIDS[0]}"
        with closing(sqlite3.connect(str(source))) as connection, connection:
            columns = [
                row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            connection.execute(f'DROP TABLE "{table}"')
            connection.execute(
                f'CREATE VIEW "{table}" AS SELECT '
                + ", ".join(f'NULL AS "{column}"' for column in columns)
            )
        with self.assertRaisesRegex(ValueError, "not a regular table"):
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertFalse(destination.exists())

    def test_missing_source_views_are_rebuilt_for_merge_and_import(self):
        for version in query_supported_schema_versions():
            with self.subTest(version=str(version)):
                source = self.directory / f"no-views-{version}.db"
                destination = self.directory / f"rebuilt-views-{version}.db"
                self.create_database(source, version=str(version))
                with closing(sqlite3.connect(str(source))) as connection:
                    views = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='view'"
                        )
                    }
                    self.assertTrue(views)
                    for view in views:
                        connection.execute(f'DROP VIEW "{view}"')
                before = source.read_bytes()

                merge_sqlite_dbs([str(source)], str(destination))
                with closing(sqlite3.connect(str(destination))) as connection:
                    self.assertEqual(
                        {
                            row[0]
                            for row in connection.execute(
                                "SELECT name FROM sqlite_master WHERE type='view'"
                            )
                        },
                        views,
                    )
                for path in (source, destination):
                    imported = RocpdImportData(str(path), skip_auto_merge=True)
                    try:
                        self.assertEqual(
                            imported.connection.execute(
                                "SELECT string FROM rocpd_string"
                            ).fetchall(),
                            [(f"value-{self.UUIDS[0]}",)],
                        )
                        self.assertEqual(
                            {
                                row[0]
                                for row in imported.connection.execute(
                                    "SELECT name FROM sqlite_temp_master WHERE type='view'"
                                )
                            },
                            views,
                        )
                    finally:
                        imported.connection.close()
                self.assertEqual(source.read_bytes(), before)

    def test_indexes_are_rebuilt_from_trusted_schema(self):
        source = self.directory / "no-index.db"
        destination = self.directory / "rebuilt-index.db"
        self.create_database(source)
        index = f"rocpd_string{self.UUIDS[0]}_test_idx"

        # Current bundled schemas have no standalone indexes. Add one to the
        # trusted reference so this test exercises real index reconstruction.
        def indexed_schema(**kwargs):
            schema = RocpdSchema(**kwargs)
            uuid = kwargs.get("uuid", "")
            if uuid:
                schema.indexes += (
                    f'\nCREATE INDEX "rocpd_string{uuid}_test_idx" '
                    f'ON "rocpd_string{uuid}" (string);'
                )
            return schema

        # A familiar name must not carry an altered definition into the output.
        with closing(sqlite3.connect(str(source))) as connection:
            connection.execute(
                f'CREATE INDEX "{index}" ON "rocpd_string{self.UUIDS[0]}" (guid)'
            )
        with mock.patch.object(
            database_module, "RocpdSchema", side_effect=indexed_schema
        ), mock.patch.object(merge_module, "RocpdSchema", side_effect=indexed_schema):
            merge_sqlite_dbs([str(source)], str(destination))
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute(f'PRAGMA index_info("{index}")').fetchall(),
                [(0, 2, "string")],
            )

    def test_merge_enforces_trusted_constraints_on_untrusted_rows(self):
        # Schema text cannot vouch for row values: a constraint can be hidden
        # in a trailing unterminated comment, or bypassed while writing.
        table = f"rocpd_info_code_object{self.UUIDS[0]}"
        constraint = "CHECK (\"storage_type\" IN ('FILE', 'MEMORY'))"
        rows = (
            f'INSERT INTO "{table}" (guid, nid, pid, uri, storage_type) '
            "VALUES (?, 1, 1, 'uri', 'EVIL')"
        )

        def hidden_check(connection):
            table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (table,)
            ).fetchone()[0]
            self.assertIn(constraint, table_sql)
            connection.execute("PRAGMA writable_schema = ON")
            connection.execute(
                "UPDATE sqlite_master SET sql=? WHERE name=?",
                (table_sql.replace(constraint, "") + f" /* {constraint}", table),
            )
            connection.commit()

        def ignored_check(connection):
            connection.execute("PRAGMA ignore_check_constraints = ON")

        for name, mutate in (("hidden", hidden_check), ("ignored", ignored_check)):
            with self.subTest(mutation=name):
                source = self.directory / f"check-{name}.db"
                destination = self.directory / f"check-{name}-output.db"
                self.create_database(source)
                with closing(sqlite3.connect(str(source))) as connection:
                    mutate(connection)
                with closing(sqlite3.connect(str(source))) as connection:
                    connection.execute("PRAGMA ignore_check_constraints = ON")
                    connection.execute("PRAGMA foreign_keys = OFF")
                    connection.execute(rows, (self.GUIDS[0],))
                    connection.commit()
                with self.assertRaisesRegex(sqlite3.IntegrityError, "CHECK"):
                    merge_sqlite_dbs([str(source)], str(destination))
                self.assertFalse(destination.exists())
                self.assertEqual(list(self.directory.glob(f".{destination.name}.*")), [])

    def test_merge_does_not_depend_on_source_column_order(self):
        source = self.directory / "reordered.db"
        destination = self.directory / "reordered-output.db"
        self.create_database(source)
        table = f"rocpd_string{self.UUIDS[0]}"
        with closing(sqlite3.connect(str(source))) as connection, connection:
            connection.execute(f'ALTER TABLE "{table}" RENAME TO old_string')
            connection.execute(
                f'CREATE TABLE "{table}" (string TEXT, guid TEXT, id INTEGER)'
            )
            connection.execute(
                f'INSERT INTO "{table}" (id, guid, string) '
                "SELECT id, guid, string FROM old_string"
            )
            connection.execute("DROP TABLE old_string")
        merge_sqlite_dbs([str(source)], str(destination))
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT id, guid, string FROM rocpd_string"
                ).fetchall(),
                [(1, self.GUIDS[0], f"value-{self.UUIDS[0]}")],
            )

    def test_duplicate_required_metadata_is_rejected(self):
        source = self.directory / "duplicate-metadata.db"
        self.create_database(source)
        with closing(sqlite3.connect(str(source))) as connection, connection:
            connection.execute(
                f'INSERT INTO "rocpd_metadata{self.UUIDS[0]}" (tag, value) '
                "VALUES ('uuid', ?)",
                (self.UUIDS[0],),
            )
        with self.assertRaisesRegex(ValueError, "duplicate required"):
            merge_sqlite_dbs([str(source)], str(self.directory / "output.db"))

    def test_merge_accepts_generator_sources(self):
        sources = self.create_batch_sources()
        destination = self.directory / "generator-output.db"
        merge_sqlite_dbs((path for path in sources), str(destination))
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT string FROM rocpd_string ORDER BY string"
                ).fetchall(),
                sorted((f"value-{uuid}",) for uuid in self.UUIDS),
            )

    def test_identifier_breakout_is_never_interpolated(self):
        source = self.directory / "identifier.db"
        destination = self.directory / "identifier-merged.db"
        payload = self.directory / "payload.db"
        self.create_database(source)
        injected_name = (
            'rocpd_x"; ATTACH DATABASE ' + repr(str(payload)) + " AS attacker; --"
        )
        with closing(sqlite3.connect(str(source))) as connection, connection:
            quoted_name = '"' + injected_name.replace('"', '""') + '"'
            connection.execute(f"CREATE TABLE {quoted_name} (value INTEGER)")

        with redirect_stderr(io.StringIO()) as stderr:
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertIn("ignoring unsupported objects", stderr.getvalue())
        self.assertFalse(payload.exists())
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name=?", (injected_name,)
                ).fetchall(),
                [],
            )

    def test_import_uses_read_only_tables_and_trusted_views(self):
        source = self.directory / "source'with-quote.db"
        self.create_database(source)
        with closing(sqlite3.connect(str(source))) as connection, connection:
            connection.execute("DROP VIEW processes")
            connection.execute(
                "CREATE VIEW processes AS SELECT randomblob(1000000) AS payload"
            )

        with closing(sqlite3.connect(":memory:", uri=True)) as connection, connection:
            configure_untrusted_schema(connection)
            _, schema_version = _create_temp_views(connection, [str(source)])
            _create_meta_views(connection, schema_version)
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
            with self.assertRaisesRegex(ValueError, "Multiple schema versions"):
                merge_sqlite_dbs([str(first), str(second)], str(destination))

    def test_additional_metadata_is_preserved(self):
        source = self.directory / "extended-metadata.db"
        destination = self.directory / "extended-metadata-merged.db"
        self.create_database(source)
        with closing(sqlite3.connect(str(source))) as connection, connection:
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

    def test_foreign_key_violations_are_reported_not_fatal(self):
        source = self.directory / "invalid-relationships.db"
        destination = self.directory / "invalid-relationships-output.db"
        self.create_database(source)
        table = f"rocpd_event{self.UUIDS[0]}"
        with closing(sqlite3.connect(str(source))) as connection, connection:
            connection.executemany(
                f'INSERT INTO "{table}" (category_id) VALUES (?)',
                ((1000 + index,) for index in range(100)),
            )
        with redirect_stderr(io.StringIO()) as stderr:
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertIn("foreign-key violations", stderr.getvalue())
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone(),
                (100,),
            )

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
        with closing(sqlite3.connect(str(output))) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT string FROM rocpd_string").fetchall(),
                [(f"value-{self.UUIDS[0]}",)],
            )

    def test_uncheckpointed_wal_is_rejected(self):
        source = self.directory / "live-wal.db"
        destination = self.directory / "existing-wal-output.db"
        self.create_database(source)
        destination.write_bytes(b"existing destination")

        with closing(sqlite3.connect(str(source))) as writer:
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

    def test_destination_cannot_alias_input_through_home_expansion(self):
        source = self.directory / "source.db"
        self.create_database(source)
        before = source.read_bytes()
        with mock.patch.dict(
            os.environ,
            {"HOME": str(self.directory), "USERPROFILE": str(self.directory)},
        ):
            for input_path in (str(source), "~/source.db"):
                with self.subTest(input_path=input_path):
                    with self.assertRaisesRegex(ValueError, "must not also be an input"):
                        merge_sqlite_dbs([input_path], str(source))
                    self.assertEqual(source.read_bytes(), before)

    @unittest.skipUnless(hasattr(os, "link"), "Requires hard links")
    def test_destination_cannot_alias_input_through_hard_link(self):
        source = self.directory / "source.db"
        destination = self.directory / "linked.db"
        self.create_database(source)
        os.link(source, destination)
        with self.assertRaisesRegex(ValueError, "must not also be an input"):
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertTrue(os.path.samefile(source, destination))

    def test_destination_wal_is_preserved_until_recovered(self):
        source = self.directory / "source.db"
        destination = self.directory / "destination.db"
        self.create_database(source)
        self.create_database(destination, uuid=self.UUIDS[1], guid=self.GUIDS[1])

        # Exit without closing SQLite to leave a real committed WAL behind.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sqlite3, sys\n"
                "connection = sqlite3.connect(sys.argv[1])\n"
                "connection.execute('PRAGMA journal_mode = WAL')\n"
                "connection.execute('PRAGMA wal_autocheckpoint = 0')\n"
                "connection.execute('CREATE TABLE old_destination (value TEXT)')\n"
                "connection.execute('INSERT INTO old_destination VALUES (?)', "
                "('committed-in-wal',))\n"
                "connection.commit()\n"
                "os._exit(0)\n",
                str(destination),
            ],
            check=True,
            timeout=30,
        )
        self.assertGreater(Path(f"{destination}-wal").stat().st_size, 0)
        paths = [destination] + [
            Path(f"{destination}{suffix}") for suffix in ("-wal", "-shm", "-journal")
        ]
        before = {path: path.read_bytes() for path in paths if path.exists()}
        with self.assertRaisesRegex(
            ValueError, "Destination database has SQLite sidecar"
        ):
            merge_sqlite_dbs([str(source)], str(destination))
        self.assertEqual(
            {path: path.read_bytes() for path in paths if path.exists()}, before
        )
        self.assertEqual(list(self.directory.glob(f".{destination.name}.*")), [])

        # Explicit recovery preserves the old committed data. Once the old
        # database is closed, publishing and reopening the new result is safe.
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute("SELECT value FROM old_destination").fetchall(),
                [("committed-in-wal",)],
            )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("PRAGMA journal_mode = DELETE")
        merge_sqlite_dbs([str(source)], str(destination))
        with closing(sqlite3.connect(str(destination))) as connection:
            self.assertEqual(
                connection.execute("SELECT string FROM rocpd_string").fetchall(),
                [(f"value-{self.UUIDS[0]}",)],
            )
            self.assertEqual(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='old_destination'"
                ).fetchall(),
                [],
            )
            inspected = inspect_attached_rocpd(connection, "main", str(destination))
            self.assertEqual(inspected.uuids, (self.UUIDS[0],))

    def test_destination_sidecars_are_rejected_even_without_main_file(self):
        source = self.directory / "source.db"
        self.create_database(source)
        for suffix, contents in (
            ("-wal", b""),
            ("-shm", b""),
            ("-journal", b"existing journal"),
        ):
            with self.subTest(suffix=suffix):
                destination = self.directory / "new-output.db"
                sidecar = Path(f"{destination}{suffix}")
                sidecar.write_bytes(contents)
                try:
                    with self.assertRaisesRegex(ValueError, "SQLite sidecar"):
                        merge_sqlite_dbs([str(source)], str(destination))
                    self.assertFalse(destination.exists())
                    self.assertEqual(sidecar.read_bytes(), contents)
                    self.assertEqual(
                        list(self.directory.glob(f".{destination.name}.*")), []
                    )
                finally:
                    sidecar.unlink()

    def test_destination_sidecar_appearing_during_merge_prevents_publication(self):
        source = self.directory / "source.db"
        destination = self.directory / "existing-output.db"
        self.create_database(source)
        destination.write_bytes(b"existing destination")
        sidecar = Path(f"{destination}-journal")

        def create_sidecar(message):
            if "Detached" in message:
                sidecar.write_bytes(b"journal appeared during merge")

        with self.assertRaisesRegex(ValueError, "SQLite sidecar"):
            merge_sqlite_dbs([str(source)], str(destination), on_log=create_sidecar)
        self.assertEqual(destination.read_bytes(), b"existing destination")
        self.assertEqual(sidecar.read_bytes(), b"journal appeared during merge")
        self.assertEqual(list(self.directory.glob(f".{destination.name}.*")), [])

    def create_batch_sources(self):
        sources = []
        for index, (uuid, guid) in enumerate(zip(self.UUIDS, self.GUIDS)):
            source = self.directory / f"batch-input-{index}.db"
            self.create_database(source, uuid=uuid, guid=guid)
            sources.append(str(source))
        return sources

    def import_batched_sources(self, sources):
        # Keep generated packages in the test directory while exercising the
        # public importer and its real flatten/merge/repackage path.
        with mock.patch.object(
            package_module,
            "prepare_output_folder",
            return_value=str(self.directory / "batched.rpdb"),
        ):
            return RocpdImportData(sources, automerge_limit=2)

    def test_auto_merge_singleton_wal_is_rejected_then_preserved_after_close(self):
        sources = self.create_batch_sources()
        singleton = sources[-1]
        with closing(sqlite3.connect(singleton)) as writer:
            writer.execute("PRAGMA journal_mode = WAL")
            writer.execute("PRAGMA wal_autocheckpoint = 0")
            writer.execute(
                f'INSERT INTO "rocpd_string{self.UUIDS[-1]}" (guid, string) '
                "VALUES (?, ?)",
                (self.GUIDS[-1], "committed-in-wal"),
            )
            writer.commit()
            wal = Path(f"{singleton}-wal")
            self.assertGreater(wal.stat().st_size, 0)
            before = (Path(singleton).read_bytes(), wal.read_bytes())
            with self.assertRaisesRegex(ValueError, "uncheckpointed"):
                imported = self.import_batched_sources(sources)
                imported.connection.close()
            self.assertEqual((Path(singleton).read_bytes(), wal.read_bytes()), before)
            self.assertEqual(
                list((self.directory / "batched.rpdb").glob("merged_db_1_*.db")), []
            )

        imported = self.import_batched_sources(sources)
        try:
            self.assertEqual(len(imported.databases), 2)
            self.assertEqual(
                imported.connection.execute(
                    "SELECT string FROM rocpd_string ORDER BY string"
                ).fetchall(),
                sorted(
                    [(f"value-{uuid}",) for uuid in self.UUIDS] + [("committed-in-wal",)]
                ),
            )
        finally:
            imported.connection.close()

    def test_auto_merge_singleton_validates_schema_and_rebuilds_views(self):
        sources = self.create_batch_sources()
        singleton = sources[-1]
        table = f"rocpd_arg{self.UUIDS[-1]}"
        with closing(sqlite3.connect(singleton)) as connection:
            table_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name=?", (table,)
            ).fetchone()[0]
            connection.execute(f'DROP TABLE "{table}"')
        before = Path(singleton).read_bytes()
        with self.assertRaisesRegex(ValueError, "Missing rocPD table"):
            imported = self.import_batched_sources(sources)
            imported.connection.close()
        self.assertEqual(Path(singleton).read_bytes(), before)
        self.assertEqual(
            list((self.directory / "batched.rpdb").glob("merged_db_1_*.db")), []
        )

        # Once the table is restored, the singleton can be published, but its
        # source-controlled view definition and custom objects are not copied.
        with closing(sqlite3.connect(singleton)) as connection:
            connection.execute(table_sql)
            connection.execute("CREATE TABLE unexpected_table (value TEXT)")
            connection.execute("DROP VIEW processes")
            connection.execute("CREATE VIEW processes AS SELECT randomblob(1000000)")
        with redirect_stderr(io.StringIO()):
            imported = self.import_batched_sources(sources)
        try:
            self.assertEqual(len(imported.databases), 2)
            self.assertEqual(
                imported.connection.execute(
                    "SELECT COUNT(*) FROM rocpd_string"
                ).fetchone(),
                (3,),
            )
            singleton_output = imported.databases[-1]
        finally:
            imported.connection.close()
        with closing(sqlite3.connect(singleton_output)) as connection:
            view_sql = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='processes'"
            ).fetchone()[0]
            self.assertNotIn("randomblob", view_sql.lower())
            self.assertEqual(
                connection.execute(
                    "SELECT name FROM sqlite_master WHERE name='unexpected_table'"
                ).fetchall(),
                [],
            )
            inspected = inspect_attached_rocpd(connection, "main", singleton_output)
            self.assertEqual(inspected.uuids, (self.UUIDS[-1],))

    @unittest.skipUnless(os.name == "posix", "Requires POSIX file modes")
    def test_output_permissions_follow_existing_mode_or_process_umask(self):
        source = self.directory / "permissions-source.db"
        self.create_database(source)

        existing = self.directory / "existing-mode.db"
        existing.write_bytes(b"replace me")
        existing.chmod(0o640)
        merge_sqlite_dbs([str(source)], str(existing))
        self.assertEqual(stat.S_IMODE(existing.stat().st_mode), 0o640)

        mode_reference = self.directory / "mode-reference.db"
        with closing(sqlite3.connect(str(mode_reference))) as connection, connection:
            connection.execute("CREATE TABLE mode_reference (value INTEGER)")
        expected_new_mode = stat.S_IMODE(mode_reference.stat().st_mode)
        new_output = self.directory / "new-mode.db"
        merge_sqlite_dbs([str(source)], str(new_output))
        self.assertEqual(stat.S_IMODE(new_output.stat().st_mode), expected_new_mode)


if __name__ == "__main__":
    unittest.main()
