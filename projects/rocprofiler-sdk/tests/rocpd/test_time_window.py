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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

from rocpd import time_window
from rocpd.importer import RocpdImportData
from rocpd.schema import RocpdSchema, query_supported_schema_versions


class RocpdTimeWindowTest(unittest.TestCase):
    PROCESSES = (
        ("_12345678_1234_7123_8123_123456789abc", "12345678-1234-7123-8123-123456789abc"),
        ("_22345678_1234_7123_8123_123456789abc", "22345678-1234-7123-8123-123456789abc"),
    )

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary_directory.name)
        self.version = str(query_supported_schema_versions()[-1])

    def tearDown(self):
        self._temporary_directory.cleanup()

    def create_database(self, index, regions):
        """Create a profile whose regions span the given (start, end) pairs."""
        uuid, guid = self.PROCESSES[index]
        path = self.directory / f"process-{index}.db"
        schema = RocpdSchema(uuid=uuid, guid=guid, version=self.version)
        with closing(sqlite3.connect(str(path))) as connection, connection:
            for script in (schema.tables, schema.metadata, schema.indexes, schema.views):
                connection.executescript(script)
            connection.execute(
                f'INSERT INTO "rocpd_string{uuid}" (id, guid, string) VALUES (1, ?, ?)',
                (guid, "HIP_API"),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_node{uuid}" '
                "(id, guid, hash, machine_id, hostname) VALUES (1, ?, 1, 'm', 'h')",
                (guid,),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_process{uuid}" '
                "(id, guid, nid, ppid, pid, init, fini, start, end, command) "
                "VALUES (1, ?, 1, 1, ?, 0, 0, 0, 0, 'app')",
                (guid, 1000 + index),
            )
            connection.execute(
                f'INSERT INTO "rocpd_info_thread{uuid}" '
                "(id, guid, nid, ppid, pid, tid, name, start, end) "
                "VALUES (1, ?, 1, 1, 1, ?, 'main', 0, 0)",
                (guid, 2000 + index),
            )
            connection.execute(
                f'INSERT INTO "rocpd_event{uuid}" (id, guid, category_id) VALUES (1, ?, 1)',
                (guid,),
            )
            connection.executemany(
                f'INSERT INTO "rocpd_region{uuid}" '
                "(guid, nid, pid, tid, start, end, name_id, event_id) "
                "VALUES (?, 1, 1, 1, ?, ?, 1, 1)",
                ((guid, start, end) for start, end in regions),
            )
        return str(path)

    def apply(self, databases, start, end):
        data = RocpdImportData(databases, skip_auto_merge=True)
        try:
            with redirect_stdout(io.StringIO()) as stdout:
                time_window.apply_time_window(data, start=start, end=end)
            regions = data.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
            return regions, stdout.getvalue()
        finally:
            data.connection.close()

    def test_window_filters_records(self):
        database = self.create_database(0, [(100, 200), (300, 400), (500, 600)])
        regions, output = self.apply([database], "250", "450")
        self.assertEqual(regions, 1)
        self.assertIn("reduced the duration", output)

    def test_full_window_keeps_all_records(self):
        database = self.create_database(0, [(100, 200), (300, 400)])
        regions, _ = self.apply([database], "0%", "100%")
        self.assertEqual(regions, 2)

    def test_window_between_processes_reports_no_records(self):
        # Two processes that ran at different times leave a gap in the combined trace.
        databases = [
            self.create_database(0, [(100, 200)]),
            self.create_database(1, [(1000, 1100)]),
        ]
        with self.assertRaisesRegex(ValueError, "contains no trace records"):
            self.apply(databases, "300", "900")

    def test_trace_without_timing_data_is_reported(self):
        database = self.create_database(0, [])
        with self.assertRaisesRegex(ValueError, "contains no timing data"):
            self.apply([database], "0%", "100%")

    def test_single_instant_trace(self):
        # A zero-length trace must not divide by zero when reporting the reduction.
        database = self.create_database(0, [(100, 100)])
        regions, output = self.apply([database], "50", "150")
        self.assertEqual(regions, 1)
        self.assertIn("reduced the duration by   0.00%", output)


if __name__ == "__main__":
    unittest.main()
