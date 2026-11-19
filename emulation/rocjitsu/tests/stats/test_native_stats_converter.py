#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check native statistics export through the SQLite and CSV converter CLIs."""

import argparse
from contextlib import closing
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TABLES = ('runs', 'metrics', 'scopes', 'snapshots', 'samples', 'histogram_buckets')
FIXTURE = None
CONVERTER = ROOT / 'scripts' / 'rocjitsu-stats.py'


def run_checked(command):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode:
        raise AssertionError(
            f'{command!r} exited with {result.returncode}\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )
    return result


class NativeStatisticsConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FIXTURE is None:
            raise RuntimeError('pass --fixture with the native export executable')
        temporary = tempfile.TemporaryDirectory(prefix='rocjitsu-native-stats-')
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        native_output = cls.root / 'native'
        run_checked([str(FIXTURE), str(native_output)])
        outputs = list(native_output.glob('*.jsonl'))
        if len(outputs) != 1:
            raise AssertionError(f'expected one native output file, found {outputs}')
        cls.source = outputs[0]

    def setUp(self):
        self.directory = self.root / self._testMethodName
        self.directory.mkdir()

    def convert(self, option, output):
        run_checked(
            [
                sys.executable,
                '-B',
                str(CONVERTER),
                str(self.source),
                option,
                str(output),
            ]
        )

    def database(self):
        path = self.directory / 'statistics.sqlite'
        self.convert('--sqlite', path)
        return path

    def assert_metadata(self, metadata):
        footer = metadata['session_end']
        metadata = metadata['run']
        self.assertEqual(
            metadata['selection'],
            {
                'include': ['cache.*', 'precision.*', 'memory.*', 'host.*'],
                'scopes': ['dispatch', 'component'],
                'max_dispatch_scopes': 1,
            },
        )
        self.assertEqual(metadata['dropped_dispatch_scopes'], '1')
        self.assertEqual(footer['dropped_dispatch_scopes'], '1')
        self.assertFalse(footer['complete'])
        self.assertEqual(metadata['schema'], 'rocjitsu.statistics.v1')
        self.assertEqual(metadata['node'], 'fixture "node"\n')
        self.assertEqual(metadata['execution_model'], 'functional')
        self.assertEqual(metadata['accumulation_scope'], 'session')
        self.assertTrue(metadata['build'])
        devices = {device['gpu_id']: device for device in metadata['devices']}
        self.assertEqual(set(devices), {0, 1})
        for gpu, kind, attempts, completed, failed in (
            (0, 'function_level', 1, 1, 0),
            (0, 'bus', 0, 0, 0),
            (0, 'lost_connection', 0, 0, 0),
            (1, 'function_level', 0, 0, 0),
            (1, 'bus', 1, 0, 1),
            (1, 'lost_connection', 1, 1, 0),
        ):
            self.assertEqual(
                devices[gpu]['resets'][kind],
                {
                    'attempts': str(attempts),
                    'completed': str(completed),
                    'failed': str(failed),
                    'overflow': False,
                },
            )
            self.assertEqual(devices[gpu]['resets_in_progress'], '0')

    def test_sqlite_preserves_scalar_types_and_known_totals(self):
        with closing(sqlite3.connect(self.database())) as connection:
            counts = dict(
                connection.execute(
                    'SELECT scopes.gpu_id, samples.exact_value FROM samples '
                    'JOIN metrics USING (run_id, metric_id) '
                    'JOIN scopes USING (run_id, scope_id) '
                    "WHERE metrics.name = 'cache.requests'"
                )
            )
            self.assertEqual(counts, {'0': '8', '1': '13'})
            self.assertEqual(sum(int(value) for value in counts.values()), 21)
            values = {
                row[0]: row[1:]
                for row in connection.execute(
                    'SELECT metrics.name, samples.exact_value, samples.value_integer, '
                    'samples.value_real, samples.integer_out_of_range, samples.overflow '
                    'FROM samples JOIN metrics USING (run_id, metric_id) '
                    "WHERE metrics.name LIKE 'precision.%' OR metrics.name = 'memory.pending'"
                )
            }
            self.assertEqual(
                values,
                {
                    'precision.above_double': (
                        '9007199254740993',
                        9007199254740993,
                        None,
                        0,
                        0,
                    ),
                    'precision.max_u64': ('18446744073709551615', None, None, 1, 0),
                    'precision.overflow': ('18446744073709551615', None, None, 1, 1),
                    'memory.pending': ('-3', -3, None, 0, 0),
                },
            )
            descriptors = dict(
                connection.execute(
                    'SELECT name, value_type FROM metrics '
                    "WHERE name IN ('precision.max_u64', 'memory.pending')"
                )
            )
            self.assertEqual(
                descriptors, {'precision.max_u64': 'u64', 'memory.pending': 'i64'}
            )
            self.assertEqual(
                connection.execute(
                    'SELECT epoch, snapshot_id, complete FROM snapshots'
                ).fetchall(),
                [('0', '1', 0)],
            )
            complete, metadata = connection.execute(
                'SELECT complete, metadata_json FROM runs'
            ).fetchone()
            self.assertEqual(complete, 0)
            self.assert_metadata(json.loads(metadata))
            self.assertEqual(
                connection.execute('PRAGMA foreign_key_check').fetchall(), []
            )

    def test_histogram_and_summary_match_recorded_observations(self):
        with closing(sqlite3.connect(self.database())) as connection:
            buckets = connection.execute(
                'SELECT bucket_index, upper_bound, exact_count, count_integer, overflow '
                'FROM histogram_buckets JOIN metrics USING (run_id, metric_id) '
                "WHERE metrics.name = 'memory.request_bytes' ORDER BY bucket_index"
            ).fetchall()
            self.assertEqual(
                buckets,
                [(0, 4.0, '2', 2, 0), (1, 16.0, '2', 2, 0), (2, None, '1', 1, 0)],
            )
            self.assertEqual(sum(int(row[2]) for row in buckets), 5)
            samples = {
                name: (exact, real, availability)
                for name, exact, real, availability in connection.execute(
                    'SELECT metrics.name, exact_value, value_real, availability '
                    'FROM samples JOIN metrics USING (run_id, metric_id)'
                )
            }
            self.assertEqual(
                samples['memory.request_bytes.count'], ('5', None, 'available')
            )
            for name, expected in (
                ('memory.request_bytes.sum', 43.0),
                ('memory.request_bytes.minimum', 1.0),
                ('memory.request_bytes.maximum', 17.0),
                ('memory.request_bytes.mean', 8.6),
                ('memory.request_bytes.variance', 43.44),
                ('host.dispatch.duration.sum', 0.75),
                ('host.dispatch.duration.minimum', 0.125),
                ('host.dispatch.duration.maximum', 0.375),
                ('host.dispatch.duration.mean', 0.25),
                ('host.dispatch.duration.variance', 1 / 96),
            ):
                with self.subTest(metric=name):
                    exact, real, availability = samples[name]
                    self.assertIsNone(exact)
                    self.assertEqual(availability, 'available')
                    self.assertAlmostEqual(real, expected)
            self.assertEqual(
                samples['host.dispatch.duration.count'], ('3', None, 'available')
            )
            self.assertEqual(
                samples['host.empty_duration.count'], ('0', None, 'available')
            )
            for suffix in ('sum', 'minimum', 'maximum', 'mean', 'variance'):
                self.assertEqual(
                    samples[f'host.empty_duration.{suffix}'],
                    (None, None, 'unavailable'),
                )
            domains = connection.execute(
                'SELECT kind, value_type, unit, domain FROM metrics '
                "WHERE name = 'host.dispatch.duration.mean'"
            ).fetchall()
            self.assertEqual(domains, [('summary', 'f64', 'seconds', 'host_wall_time')])

    def test_csv_preserves_values_scopes_and_reset_metadata(self):
        directory = self.directory / 'csv'
        self.convert('--csv', directory)
        self.assertEqual({path.stem for path in directory.glob('*.csv')}, set(TABLES))
        tables = {}
        for table in TABLES:
            with (directory / f'{table}.csv').open(
                encoding='utf-8', newline=''
            ) as source:
                tables[table] = list(csv.DictReader(source))
        metrics = {row['metric_id']: row for row in tables['metrics']}
        samples = {metrics[row['metric_id']]['name']: row for row in tables['samples']}
        self.assertEqual(
            samples['precision.max_u64']['exact_value'], '18446744073709551615'
        )
        self.assertEqual(samples['precision.max_u64']['value_integer'], '')
        self.assertEqual(samples['precision.max_u64']['integer_out_of_range'], '1')
        self.assertEqual(samples['precision.overflow']['overflow'], '1')
        self.assertEqual(samples['memory.pending']['exact_value'], '-3')
        self.assertEqual(
            samples['host.empty_duration.mean']['availability'], 'unavailable'
        )
        self.assertEqual(samples['host.empty_duration.mean']['value_real'], '')
        scopes = {row['name']: row for row in tables['scopes']}
        first_scope = scopes['gpu0.l2 "cache"\n']
        self.assertEqual(first_scope['gpu_id'], '0')
        self.assertEqual(
            json.loads(first_scope['metadata_json'])['attributes'],
            {'target': 'gfx1250', 'note': 'comma, quote" and newline\n'},
        )
        buckets = sorted(
            tables['histogram_buckets'], key=lambda row: int(row['bucket_index'])
        )
        self.assertEqual([row['exact_count'] for row in buckets], ['2', '2', '1'])
        self.assertEqual([row['upper_bound'] for row in buckets], ['4.0', '16.0', ''])
        self.assertEqual(len(tables['runs']), 1)
        self.assertEqual(tables['runs'][0]['complete'], '0')
        self.assert_metadata(json.loads(tables['runs'][0]['metadata_json']))

    def test_identical_reimport_preserves_every_row(self):
        path = self.database()
        with closing(sqlite3.connect(path)) as connection:
            before = {
                table: connection.execute(
                    f'SELECT * FROM {table} ORDER BY rowid'
                ).fetchall()
                for table in TABLES
            }
        self.convert('--sqlite', path)
        with closing(sqlite3.connect(path)) as connection:
            after = {
                table: connection.execute(
                    f'SELECT * FROM {table} ORDER BY rowid'
                ).fetchall()
                for table in TABLES
            }
            self.assertEqual(
                connection.execute('PRAGMA foreign_key_check').fetchall(), []
            )
        self.assertEqual(before, after)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--converter', type=Path, default=CONVERTER)
    arguments, unittest_arguments = parser.parse_known_args()
    FIXTURE = arguments.fixture.resolve()
    CONVERTER = arguments.converter.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_arguments])
