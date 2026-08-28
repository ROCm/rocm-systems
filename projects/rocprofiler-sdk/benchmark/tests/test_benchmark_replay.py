#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

"""
Tests for the kernel replay bookkeeping in the rocprofv3 benchmark harness.

These cover the parts that can be checked without a GPU: reading the collection
mode off a rocprofv3 command line, keeping replay and non-replay runs in
separate benchmark_config rows, migrating a database created before the replay
columns existed, and the schema and views parsing under sqlite3.
"""

import os
import sqlite3
import argparse
import importlib.util
import unittest

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BENCHMARK_DIR = os.path.dirname(THIS_DIR)
RUNNER = os.path.join(BENCHMARK_DIR, "source", "bin", "rocprofv3-benchmark.py")
SHARE_DIR = os.path.join(BENCHMARK_DIR, "source", "share", "rocprofiler-sdk")
TABLES_SQL = os.path.join(SHARE_DIR, "benchmark_tables.sql")
VIEWS_SQL = os.path.join(SHARE_DIR, "benchmark_views.sql")

# the same substitutions connect_to_database applies for sqlite3
SQLITE_REPLACEMENTS = {
    " INT ": " INTEGER ",
    "AUTO_INCREMENT": "AUTOINCREMENT",
    '("{}")': '"{}"',
    '("[]")': '"[]"',
}


def load_runner():
    spec = importlib.util.spec_from_file_location("rocprofv3_benchmark", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_runner()


def read_sql(path, repl):
    with open(path, "r") as ifs:
        data = ifs.read()

    for key, value in repl.items():
        data = data.replace(key, value)

    return data


def create_database(views=False):
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.executescript(read_sql(TABLES_SQL, SQLITE_REPLACEMENTS))

    if views:
        for metric in benchmark.CONST_METRIC_LIST:
            cursor.executescript(read_sql(VIEWS_SQL, {"{{metric}}": metric}))

    return connection, cursor


def sqlite_args():
    return argparse.Namespace(db_backend="sqlite3", db_placeholder="?")


class TestDeriveReplayConfig(unittest.TestCase):
    """Reading the collection mode off a rocprofv3 command line"""

    def test_no_arguments_is_not_a_counter_run(self):
        self.assertIsNone(benchmark.derive_replay_config(None))
        self.assertIsNone(benchmark.derive_replay_config([]))

    def test_trace_only_run_is_not_a_counter_run(self):
        self.assertIsNone(benchmark.derive_replay_config(["--kernel-trace"]))

    def test_single_pmc_group(self):
        data = benchmark.derive_replay_config(["--pmc", "SQ_WAVES", "SQ_INSTS_VALU"])
        self.assertEqual(data["counter_group_count"], 1)
        self.assertEqual(data["counter_collection_mode"], "single-pass")
        self.assertNotIn("kernel_replay", data)

    def test_several_pmc_groups_without_replay_are_multiplexed(self):
        data = benchmark.derive_replay_config(
            ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--pmc", "FETCH_SIZE"]
        )
        self.assertEqual(data["counter_group_count"], 3)
        self.assertEqual(data["counter_collection_mode"], "multiplexed")

    def test_replay_flag_without_value(self):
        data = benchmark.derive_replay_config(
            ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--kernel-replay-beta-enabled"]
        )
        self.assertEqual(data["kernel_replay"], 1)
        self.assertEqual(data["counter_group_count"], 2)
        self.assertEqual(data["counter_collection_mode"], "kernel-replay")

    def test_replay_flag_with_separate_value(self):
        for value in ("true", "TRUE", "yes", "on", "1", "t", "y"):
            data = benchmark.derive_replay_config(
                ["--pmc", "SQ_WAVES", "--kernel-replay-beta-enabled", value]
            )
            self.assertEqual(data["kernel_replay"], 1, msg=value)
            self.assertEqual(data["counter_collection_mode"], "kernel-replay", msg=value)

    def test_replay_flag_disabled_by_value(self):
        for value in ("false", "FALSE", "no", "off", "0", "f", "n"):
            data = benchmark.derive_replay_config(
                ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT"]
                + ["--kernel-replay-beta-enabled", value]
            )
            self.assertEqual(data["kernel_replay"], 0, msg=value)
            self.assertEqual(data["counter_collection_mode"], "multiplexed", msg=value)

    def test_replay_flag_with_inline_value(self):
        enabled = benchmark.derive_replay_config(
            ["--pmc", "SQ_WAVES", "--kernel-replay-beta-enabled=true"]
        )
        disabled = benchmark.derive_replay_config(
            ["--pmc", "SQ_WAVES", "--kernel-replay-beta-enabled=false"]
        )
        self.assertEqual(enabled["kernel_replay"], 1)
        self.assertEqual(disabled["kernel_replay"], 0)

    def test_counter_name_that_looks_like_a_truth_value_is_not_consumed(self):
        # the token after the replay flag is only read as a truth value, never
        # as the start of the next option
        data = benchmark.derive_replay_config(
            ["--kernel-replay-beta-enabled", "--pmc", "SQ_WAVES"]
        )
        self.assertEqual(data["kernel_replay"], 1)
        self.assertEqual(data["counter_group_count"], 1)

    def test_input_file_leaves_the_group_count_unknown(self):
        data = benchmark.derive_replay_config(["-i", "counters.txt"])
        self.assertEqual(data["counter_collection_mode"], "unknown")
        self.assertNotIn("counter_group_count", data)

    def test_input_file_with_replay_is_still_a_replay_run(self):
        data = benchmark.derive_replay_config(
            ["--input", "config.yaml", "--kernel-replay-beta-enabled"]
        )
        self.assertEqual(data["kernel_replay"], 1)
        self.assertEqual(data["counter_collection_mode"], "kernel-replay")

    def test_replay_flag_with_an_invalid_value_is_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.derive_replay_config(["--kernel-replay-beta-enabled=maybe"])


class TestConfigRows(unittest.TestCase):
    """Replay and non-replay runs have to land in separate rows"""

    def setUp(self):
        self.connection, self.cursor = create_database()
        self.args = sqlite_args()
        self.config_record = {
            "config": {
                "benchmark_mode": "tool-runtime-overhead",
                "counter_collection": True,
            }
        }

    def tearDown(self):
        self.connection.close()

    def insert(self, rocprofv3_args):
        return benchmark.insert_benchmark_config(
            self.cursor, None, dict(self.config_record), self.args, rocprofv3_args
        )

    def test_replay_and_multiplexed_runs_are_distinct_configs(self):
        groups = ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT"]
        multiplexed = self.insert(groups)
        replay = self.insert(groups + ["--kernel-replay-beta-enabled"])

        self.assertNotEqual(multiplexed, replay)

        self.cursor.execute(
            "SELECT id, kernel_replay, counter_group_count, counter_collection_mode "
            "FROM benchmark_config ORDER BY id"
        )
        rows = self.cursor.fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1:], (None, 2, "multiplexed"))
        self.assertEqual(rows[1][1:], (1, 2, "kernel-replay"))

    def test_group_count_separates_otherwise_identical_runs(self):
        two = self.insert(["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT"])
        four = self.insert(
            ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT"]
            + ["--pmc", "FETCH_SIZE", "--pmc", "WRITE_SIZE"]
        )
        self.assertNotEqual(two, four)

    def test_identical_runs_reuse_one_row(self):
        args = ["--pmc", "SQ_WAVES", "--kernel-replay-beta-enabled"]
        self.assertEqual(self.insert(args), self.insert(args))

    def test_label_names_the_collection_mode(self):
        cfg_id = self.insert(
            ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--kernel-replay-beta-enabled"]
        )
        self.cursor.execute("SELECT label FROM benchmark_config WHERE id = ?", (cfg_id,))
        label = self.cursor.fetchone()[0]
        self.assertIn("Kernel Replay", label)
        self.assertIn("Counter Groups=2", label)

    def test_trace_only_run_records_no_replay_columns(self):
        cfg_id = benchmark.insert_benchmark_config(
            self.cursor,
            None,
            {"config": {"benchmark_mode": "tool-runtime-overhead", "kernel_trace": True}},
            self.args,
            ["--kernel-trace"],
        )
        self.cursor.execute(
            "SELECT kernel_replay, counter_group_count, counter_collection_mode "
            "FROM benchmark_config WHERE id = ?",
            (cfg_id,),
        )
        self.assertEqual(self.cursor.fetchone(), (None, None, None))

    def test_baseline_run_is_unaffected(self):
        cfg_id = benchmark.insert_benchmark_config(self.cursor, None, None, self.args)
        self.cursor.execute(
            "SELECT benchmark_mode, counter_collection_mode FROM benchmark_config "
            "WHERE id = ?",
            (cfg_id,),
        )
        self.assertEqual(self.cursor.fetchone(), ("baseline", None))

    def test_collection_mode_is_constrained(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.cursor.execute(
                "INSERT INTO benchmark_config "
                "(hash_id, benchmark_mode, counter_collection_mode) VALUES (?, ?, ?)",
                ("deadbeef", "baseline", "replay-ish"),
            )


class TestMigration(unittest.TestCase):
    """A database created before the replay columns existed keeps working"""

    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
            CREATE TABLE benchmark_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                hash_id TEXT NOT NULL,
                sdk_id INT,
                label TEXT,
                benchmark_mode TEXT NOT NULL
            )
            """)

    def tearDown(self):
        self.connection.close()

    def test_missing_columns_are_added(self):
        added = benchmark.add_missing_columns(
            self.cursor,
            "sqlite3",
            "benchmark_config",
            benchmark.CONST_ADDED_COLUMNS["benchmark_config"],
        )
        self.assertEqual(
            sorted(added),
            ["counter_collection_mode", "counter_group_count", "kernel_replay"],
        )

        columns = benchmark.get_table_columns(self.cursor, "sqlite3", "benchmark_config")
        self.assertTrue(
            set(benchmark.CONST_ADDED_COLUMNS["benchmark_config"]).issubset(columns)
        )

    def test_migration_preserves_existing_rows(self):
        self.cursor.execute(
            "INSERT INTO benchmark_config (hash_id, benchmark_mode) VALUES (?, ?)",
            ("cafe", "baseline"),
        )
        benchmark.add_missing_columns(
            self.cursor,
            "sqlite3",
            "benchmark_config",
            benchmark.CONST_ADDED_COLUMNS["benchmark_config"],
        )
        self.cursor.execute("SELECT hash_id, kernel_replay FROM benchmark_config")
        self.assertEqual(self.cursor.fetchall(), [("cafe", None)])

    def test_migration_is_idempotent(self):
        columns = benchmark.CONST_ADDED_COLUMNS["benchmark_config"]
        benchmark.add_missing_columns(self.cursor, "sqlite3", "benchmark_config", columns)
        self.assertEqual(
            benchmark.add_missing_columns(
                self.cursor, "sqlite3", "benchmark_config", columns
            ),
            [],
        )

    def test_current_schema_needs_no_migration(self):
        connection, cursor = create_database()
        try:
            self.assertEqual(
                benchmark.add_missing_columns(
                    cursor,
                    "sqlite3",
                    "benchmark_config",
                    benchmark.CONST_ADDED_COLUMNS["benchmark_config"],
                ),
                [],
            )
        finally:
            connection.close()


class TestReplayView(unittest.TestCase):
    """The comparison view puts replay and application replay on one axis"""

    def setUp(self):
        self.connection, self.cursor = create_database(views=True)
        self.cursor.execute(
            "INSERT INTO benchmarked_app (id, hash_id, md5sum, command) "
            'VALUES (1, "app", "md5", "[]")'
        )
        self.cursor.execute(
            "INSERT INTO benchmarked_sdk "
            "(id, hash_id, version_major, version_minor, version_patch, soversion, "
            " compiler_id, compiler_version, git_revision, library_arch, system_name, "
            " system_processor, system_version) "
            'VALUES (1, "sdk", 1, 0, 0, 1, "GNU", "13", "abc123", "x86_64", "Linux", '
            '"x86_64", "6.8")'
        )

    def tearDown(self):
        self.connection.close()

    def add_config(self, cfg_id, label, mode, groups, kernel_replay):
        self.cursor.execute(
            "INSERT INTO benchmark_config "
            "(id, hash_id, sdk_id, label, benchmark_mode, counter_collection_mode, "
            " counter_group_count, kernel_replay) "
            "VALUES (?, ?, 1, ?, 'tool-runtime-overhead', ?, ?, ?)",
            (cfg_id, f"hash{cfg_id}", label, mode, groups, kernel_replay),
        )

    def add_statistic(self, cfg_id, mean):
        self.cursor.execute(
            "INSERT INTO benchmark_statistics "
            "(app_id, cfg_id, sdk_id, metric_name, metric_unit, count, sum, mean, "
            " min, max, std_dev) "
            "VALUES (1, ?, 1, 'wall_time', 'sec', 3, ?, ?, ?, ?, 0.0)",
            (cfg_id, mean * 3, mean, mean, mean),
        )

    def test_projection_and_speedup(self):
        self.add_config(1, "one group", "single-pass", 1, None)
        self.add_config(2, "four groups, replay", "kernel-replay", 4, 1)
        self.add_statistic(1, 10.0)
        self.add_statistic(2, 20.0)

        self.cursor.execute(
            "SELECT counter_collection_mode, measured, application_replay_projected, "
            '"speedup vs application replay" FROM benchmark_replay_wall_time '
            "ORDER BY cfg_id"
        )
        rows = self.cursor.fetchall()

        # the single-group run is its own reference: one run collects one group
        self.assertEqual(rows[0], ("single-pass", 10.0, 10.0, 1.0))
        # four groups by application replay would be four runs of ten seconds,
        # against twenty seconds for one run that replays each dispatch
        self.assertEqual(rows[1], ("kernel-replay", 20.0, 40.0, 2.0))

    def test_run_without_a_single_group_reference_still_appears(self):
        self.add_config(1, "four groups, replay", "kernel-replay", 4, 1)
        self.add_statistic(1, 20.0)

        self.cursor.execute(
            "SELECT measured, application_replay_projected "
            "FROM benchmark_replay_wall_time"
        )
        self.assertEqual(self.cursor.fetchall(), [(20.0, None)])

    def test_reference_row_is_not_duplicated_by_repeated_runs(self):
        self.add_config(1, "one group", "single-pass", 1, None)
        self.add_config(2, "one group, again", "single-pass", 1, None)
        self.add_config(3, "four groups, replay", "kernel-replay", 4, 1)
        self.add_statistic(1, 10.0)
        self.add_statistic(2, 11.0)
        self.add_statistic(3, 20.0)

        self.cursor.execute(
            "SELECT COUNT(*) FROM benchmark_replay_wall_time WHERE cfg_id = 3"
        )
        self.assertEqual(self.cursor.fetchone()[0], 1)

    def test_runs_without_counters_are_excluded(self):
        self.add_config(1, "trace only", None, None, None)
        self.add_statistic(1, 5.0)

        self.cursor.execute("SELECT COUNT(*) FROM benchmark_replay_wall_time")
        self.assertEqual(self.cursor.fetchone()[0], 0)


class TestSchemaParses(unittest.TestCase):
    """The schema and the views have to load on every supported backend"""

    def test_tables_and_views_execute(self):
        connection, cursor = create_database(views=True)
        try:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'view' "
                "AND name LIKE 'benchmark_replay_%'"
            )
            views = {itr[0] for itr in cursor.fetchall()}
        finally:
            connection.close()

        self.assertEqual(
            views,
            {f"benchmark_replay_{itr}" for itr in benchmark.CONST_METRIC_LIST},
        )

    def test_views_are_idempotent(self):
        connection, cursor = create_database(views=True)
        try:
            for metric in benchmark.CONST_METRIC_LIST:
                cursor.executescript(read_sql(VIEWS_SQL, {"{{metric}}": metric}))
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
