# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the statistics consumer against independently specified records."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts' / 'rocjitsu-stats.py'
SPEC = importlib.util.spec_from_file_location('rocjitsu_stats', SCRIPT)
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


def fixture():
    return [
        {
            'schema': STATS.SCHEMA,
            'record': 'run',
            'run_id': 'run-1',
            'target': 'gfx942',
        },
        {'record': 'scope', 'scope_id': 'gpu0.l2', 'name': 'L2 "cache"\n', 'gpu_id': 0},
        {
            'record': 'metric',
            'metric_id': 'hits',
            'name': 'cache.hits',
            'kind': 'counter',
            'value_type': 'u64',
            'unit': 'requests',
            'domain': 'model',
        },
        {
            'record': 'metric',
            'metric_id': 'misses',
            'name': 'cache.misses',
            'kind': 'counter',
            'value_type': 'u64',
            'unit': 'requests',
            'domain': 'model',
        },
        {
            'record': 'metric',
            'metric_id': 'sizes',
            'name': 'memory.request_bytes',
            'kind': 'histogram',
            'value_type': 'f64',
            'unit': 'bytes',
            'domain': 'model',
        },
        {'record': 'snapshot_begin', 'snapshot_id': 1, 'epoch': 0, 'complete': True},
        {'record': 'sample', 'scope_id': 'gpu0.l2', 'metric_id': 'hits', 'value': '6'},
        {
            'record': 'sample',
            'scope_id': 'gpu0.l2',
            'metric_id': 'misses',
            'value': '2',
        },
        {
            'record': 'histogram_bucket',
            'scope_id': 'gpu0.l2',
            'metric_id': 'sizes',
            'bucket_index': 0,
            'upper_bound': 4.0,
            'count': '3',
        },
        {
            'record': 'histogram_bucket',
            'scope_id': 'gpu0.l2',
            'metric_id': 'sizes',
            'bucket_index': 1,
            'upper_bound': 16.0,
            'count': '4',
        },
        {
            'record': 'histogram_bucket',
            'scope_id': 'gpu0.l2',
            'metric_id': 'sizes',
            'bucket_index': 2,
            'upper_bound': None,
            'count': '1',
        },
        {'record': 'snapshot_end', 'snapshot_id': 1, 'epoch': 0},
        {'record': 'session_end', 'complete': True},
    ]


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def source(self, records=None, tail=b'', name='stats.jsonl'):
        path = self.directory / name
        records = fixture() if records is None else records
        content = ''.join(
            json.dumps(record, allow_nan=False) + '\n' for record in records
        )
        path.write_bytes(content.encode() + tail)
        return path

    def database(self, records=None):
        dataset = STATS.load_statistics(self.source(records))
        path = self.directory / 'stats.sqlite'
        STATS.write_sqlite(dataset, path)
        return sqlite3.connect(path)

    def test_sql_queries_recover_counts_and_histogram(self):
        with self.database() as connection:
            counts = dict(
                connection.execute('SELECT metric_id, value_integer FROM samples')
            )
            self.assertEqual(counts, {'hits': 6, 'misses': 2})
            self.assertEqual(counts['misses'] / sum(counts.values()), 0.25)
            buckets = connection.execute(
                'SELECT upper_bound, count_integer FROM histogram_buckets ORDER BY bucket_index'
            ).fetchall()
            self.assertEqual(buckets, [(4.0, 3), (16.0, 4), (None, 1)])
            self.assertEqual(
                connection.execute('SELECT complete FROM runs').fetchone(), (1,)
            )
            self.assertEqual(
                connection.execute('PRAGMA foreign_key_check').fetchall(), []
            )

    def test_generic_schema_preserves_application_metadata_without_gpu_identity(self):
        records = fixture()
        records[0] = {
            'schema': 'util.statistics.v1',
            'record': 'run',
            'run_id': 'run-1',
            'application': {'name': 'worker "service"\n', 'version': '1.0'},
            'session_end': 'application field retained without collision',
        }
        records[1] = {
            'schema': 'util.statistics.v1',
            'record': 'scope',
            'scope_id': 'worker-0',
            'name': 'worker',
            'attributes': {'thread': 'primary'},
        }
        for record in records[2:]:
            if 'scope_id' in record:
                record['scope_id'] = 'worker-0'
        records[-1].update(
            schema='util.statistics.v1',
            dropped_dispatch_scopes='3',
            application={'status': 'done \"now\"\n'},
        )
        dataset = STATS.load_statistics(self.source(records))
        path = self.directory / 'generic.sqlite'
        STATS.write_sqlite(dataset, path)
        with sqlite3.connect(path) as connection:
            schema, complete, metadata = connection.execute(
                'SELECT schema, complete, metadata_json FROM runs'
            ).fetchone()
            self.assertEqual(schema, 'util.statistics.v1')
            self.assertEqual(complete, 1)
            self.assertEqual(
                json.loads(metadata), {'run': records[0], 'session_end': records[-1]}
            )
            gpu_id, metadata = connection.execute(
                'SELECT gpu_id, metadata_json FROM scopes'
            ).fetchone()
            self.assertIsNone(gpu_id)
            self.assertEqual(json.loads(metadata), records[1])
            self.assertEqual(
                connection.execute('PRAGMA foreign_key_check').fetchall(), []
            )
        output = self.directory / 'generic-csv'
        STATS.write_csv(dataset, output)
        with (output / 'runs.csv').open(newline='') as source:
            runs = list(csv.DictReader(source))
        self.assertEqual(runs[0]['schema'], 'util.statistics.v1')
        self.assertEqual(runs[0]['complete'], '1')
        self.assertEqual(
            json.loads(runs[0]['metadata_json']),
            {'run': records[0], 'session_end': records[-1]},
        )
        with (output / 'scopes.csv').open(newline='') as source:
            scopes = list(csv.DictReader(source))
        self.assertEqual(scopes[0]['gpu_id'], '')
        self.assertEqual(json.loads(scopes[0]['metadata_json']), records[1])

    def test_missing_unknown_and_mismatched_schemas_are_rejected(self):
        records = fixture()
        del records[0]['schema']
        with self.assertRaisesRegex(ValueError, 'versioned run'):
            STATS.load_statistics(self.source(records))
        for run_schema in ('rocjitsu.statistics.v1', 'util.statistics.v1'):
            other_schema = (
                'util.statistics.v1'
                if run_schema == 'rocjitsu.statistics.v1'
                else 'rocjitsu.statistics.v1'
            )
            for index, schema, diagnostic in (
                (0, 'statistics.v2', 'unsupported schema'),
                (1, 'statistics.v2', 'unsupported schema'),
                (1, None, 'unsupported schema'),
                (1, other_schema, 'different schema'),
                (-1, other_schema, 'different schema'),
            ):
                with self.subTest(run_schema=run_schema, index=index, schema=schema):
                    records = fixture()
                    records[0]['schema'] = run_schema
                    records[index]['schema'] = schema
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        STATS.load_statistics(self.source(records))

    def test_large_integers_remain_exact(self):
        records = fixture()
        records[6]['value'] = str((1 << 53) + 1)
        records[7]['value'] = str((1 << 64) - 1)
        with self.database(records) as connection:
            values = connection.execute(
                'SELECT exact_value, value_integer, value_real FROM samples ORDER BY metric_id'
            ).fetchall()
            self.assertEqual(
                values,
                [
                    (str((1 << 53) + 1), (1 << 53) + 1, None),
                    (str((1 << 64) - 1), None, None),
                ],
            )
            self.assertEqual(sum(int(row[0]) for row in values), (1 << 64) + (1 << 53))

    def test_overflow_status_survives_scalar_and_bucket_import(self):
        records = fixture()
        records[6].update(value=str((1 << 64) - 1), overflow=True)
        records[8].update(count=str((1 << 64) - 1), overflow=True)
        with self.database(records) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT integer_out_of_range, overflow FROM samples WHERE metric_id = 'hits'"
                ).fetchone(),
                (1, 1),
            )
            self.assertEqual(
                connection.execute(
                    'SELECT integer_out_of_range, overflow FROM histogram_buckets WHERE bucket_index = 0'
                ).fetchone(),
                (1, 1),
            )

    def test_unavailable_is_different_from_zero(self):
        records = fixture()
        records[6]['value'] = '0'
        records[7].update(
            value=None, availability='unsupported', reason='functional model'
        )
        with self.database(records) as connection:
            values = connection.execute(
                'SELECT availability, exact_value, value_integer FROM samples ORDER BY metric_id'
            ).fetchall()
            self.assertEqual(
                values, [('available', '0', 0), ('unsupported', None, None)]
            )

    def test_signed_gauges_and_real_values_retain_their_types(self):
        records = fixture()
        records[2].update(kind='gauge', value_type='i64')
        records[3].update(kind='gauge', value_type='f64')
        records[6]['value'] = str(-(1 << 63))
        records[7]['value'] = 0.125
        with self.database(records) as connection:
            values = connection.execute(
                'SELECT exact_value, value_integer, value_real FROM samples ORDER BY metric_id'
            ).fetchall()
            self.assertEqual(
                values, [(str(-(1 << 63)), -(1 << 63), None), (None, None, 0.125)]
            )

    def test_invalid_histogram_layouts_are_rejected(self):
        for index, field, value in (
            (8, 'bucket_index', 4),
            (9, 'upper_bound', 4.0),
            (8, 'upper_bound', None),
            (10, 'upper_bound', 128.0),
        ):
            records = fixture()
            records[index][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(
                ValueError, 'histogram'
            ):
                STATS.load_statistics(self.source(records))

    def test_csv_preserves_escaping_and_types(self):
        dataset = STATS.load_statistics(self.source())
        output = self.directory / 'csv'
        STATS.write_csv(dataset, output)
        with (output / 'scopes.csv').open(newline='') as source:
            scopes = list(csv.DictReader(source))
        self.assertEqual(scopes[0]['name'], 'L2 "cache"\n')
        with (output / 'samples.csv').open(newline='') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]['exact_value'], '6')
        self.assertEqual(rows[0]['value_real'], '')
        self.assertEqual(len(list(output.glob('*.csv'))), 6)
        with self.assertRaisesRegex(ValueError, 'already exist'):
            STATS.write_csv(dataset, output)

    def test_identical_database_import_is_idempotent(self):
        dataset = STATS.load_statistics(self.source())
        path = self.directory / 'stats.sqlite'
        STATS.write_sqlite(dataset, path)
        STATS.write_sqlite(dataset, path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute('SELECT COUNT(*) FROM samples').fetchone(), (2,)
            )

    def test_conflicting_import_rolls_back_entire_import(self):
        path = self.directory / 'stats.sqlite'
        STATS.write_sqlite(STATS.load_statistics(self.source()), path)
        records = fixture()
        records[7]['value'] = '20'
        records.insert(
            1,
            {
                'record': 'metric',
                'metric_id': 'new',
                'name': 'new',
                'kind': 'counter',
                'value_type': 'u64',
                'unit': 'events',
                'domain': 'model',
            },
        )
        changed = STATS.load_statistics(self.source(records, name='conflict.jsonl'))
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate samples'):
            STATS.write_sqlite(changed, path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(
                connection.execute('SELECT COUNT(*) FROM metrics').fetchone(), (3,)
            )
            self.assertEqual(
                connection.execute(
                    "SELECT value_integer FROM samples WHERE metric_id = 'misses'"
                ).fetchone(),
                (2,),
            )

    def test_partial_snapshot_is_discarded_and_prior_snapshot_survives(self):
        records = fixture()[:-1]
        records.extend(
            [
                {'record': 'snapshot_begin', 'snapshot_id': 2, 'epoch': 0},
                {
                    'record': 'sample',
                    'scope_id': 'gpu0.l2',
                    'metric_id': 'hits',
                    'value': '100',
                },
            ]
        )
        dataset = STATS.load_statistics(
            self.source(records, tail=b'{"record":"sample"')
        )
        self.assertEqual(len(dataset.tables['snapshots']), 1)
        self.assertEqual(len(dataset.tables['samples']), 2)
        self.assertEqual(dataset.tables['runs'][('run-1',)][2], 0)

    def test_truncated_utf8_tail_retains_completed_snapshots(self):
        dataset = STATS.load_statistics(
            self.source(fixture()[:-1], tail=b'{"name":"\xe2\x82')
        )
        self.assertEqual(len(dataset.tables['snapshots']), 1)
        self.assertEqual(dataset.tables['runs'][('run-1',)][2], 0)

    def test_complete_import_upgrades_partial_run(self):
        records = fixture()
        records[-1]['dropped_dispatch_scopes'] = '3'
        partial = STATS.load_statistics(self.source(records[:-1], name='partial.jsonl'))
        complete = STATS.load_statistics(self.source(records))
        records[-1]['complete'] = False
        incomplete = STATS.load_statistics(
            self.source(records, name='incomplete.jsonl')
        )
        records[-1]['complete'] = True
        for index, datasets in enumerate(
            (
                (partial, complete, partial),
                (complete, partial),
                (incomplete, complete, incomplete),
            )
        ):
            with self.subTest(order=index):
                merged = STATS.Statistics()
                path = self.directory / f'stats-{index}.sqlite'
                for dataset in datasets:
                    merged.merge(dataset)
                    STATS.write_sqlite(dataset, path)
                expected = (
                    'run-1',
                    STATS.SCHEMA,
                    1,
                    STATS.canonical_json(
                        {'run': records[0], 'session_end': records[-1]}
                    ),
                )
                self.assertEqual(merged.tables['runs'][('run-1',)], expected)
                with sqlite3.connect(path) as connection:
                    self.assertEqual(
                        connection.execute('SELECT * FROM runs').fetchone(), expected
                    )

    def test_conflicting_run_or_footer_metadata_is_rejected(self):
        original = fixture()
        original[-1]['dropped_dispatch_scopes'] = '3'
        original[0]['application'] = {'nested': [1]}
        original[-1]['application'] = {'nested': [1]}
        initial = STATS.load_statistics(self.source(original))
        for label, field, index, value in (
            ('target', 'target', 0, 'different'),
            ('dropped', 'dropped_dispatch_scopes', -1, '4'),
            ('run-bool', 'application', 0, {'nested': [True]}),
            ('footer-bool', 'application', -1, {'nested': [True]}),
            ('run-float', 'application', 0, {'nested': [1.0]}),
            ('footer-float', 'application', -1, {'nested': [1.0]}),
        ):
            with self.subTest(case=label):
                records = copy.deepcopy(original)
                records[index][field] = value
                changed = STATS.load_statistics(
                    self.source(records, name='conflict.jsonl')
                )
                merged = STATS.Statistics()
                merged.merge(initial)
                with self.assertRaisesRegex(ValueError, 'conflicting duplicate runs'):
                    merged.merge(changed)
                path = self.directory / f'{label}.sqlite'
                STATS.write_sqlite(initial, path)
                with self.assertRaisesRegex(ValueError, 'conflicting duplicate runs'):
                    STATS.write_sqlite(changed, path)
                with sqlite3.connect(path) as connection:
                    self.assertEqual(
                        connection.execute('SELECT * FROM runs').fetchone(),
                        initial.tables['runs'][('run-1',)],
                    )

    def test_cumulative_snapshots_remain_separate(self):
        records = fixture()[:-1]
        records.extend(
            [
                {'record': 'snapshot_begin', 'snapshot_id': 2, 'epoch': 0},
                {
                    'record': 'sample',
                    'scope_id': 'gpu0.l2',
                    'metric_id': 'hits',
                    'value': '10',
                },
                {'record': 'snapshot_end', 'snapshot_id': 2, 'epoch': 0},
                {'record': 'session_end', 'complete': True},
            ]
        )
        with self.database(records) as connection:
            values = connection.execute(
                "SELECT value_integer FROM samples WHERE metric_id = 'hits' ORDER BY snapshot_id"
            ).fetchall()
            self.assertEqual(values, [(6,), (10,)])
            self.assertEqual(values[1][0] - values[0][0], 4)

    def test_nonfinite_out_of_range_and_wrong_types_are_rejected(self):
        for invalid in (True, 1.5, '-1', str(1 << 64), '01', '1e3'):
            with self.subTest(value=invalid):
                records = fixture()
                records[6]['value'] = invalid
                with self.assertRaises(ValueError):
                    STATS.load_statistics(self.source(records))
        records = fixture()
        records[2]['value_type'] = 'f64'
        path = self.source(records)
        path.write_bytes(path.read_bytes().replace(b'"value": "6"', b'"value": NaN'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            STATS.load_statistics(path)

    def test_unknown_references_and_invalid_order_are_rejected(self):
        for field in ('scope_id', 'metric_id'):
            records = fixture()
            records[6][field] = 'missing'
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, 'unknown'
            ):
                STATS.load_statistics(self.source(records))
        for records in (
            fixture()[1:],
            fixture() + [fixture()[1]],
            fixture()[:6] + [fixture()[5]],
            fixture()[:6] + [fixture()[-1]],
        ):
            with self.assertRaises(ValueError):
                STATS.load_statistics(self.source(records))

    def test_malformed_complete_line_and_duplicate_keys_fail(self):
        path = self.source(fixture()[:-1], tail=b'not json\n')
        with self.assertRaisesRegex(ValueError, 'invalid JSON'):
            STATS.load_statistics(path)
        path = self.source(
            fixture()[:-1],
            tail=b'{"record":"session_end","complete":true,"complete":false}\n',
        )
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            STATS.load_statistics(path)
        path = self.source(tail=b'{"record":')
        with self.assertRaisesRegex(ValueError, 'follows session_end'):
            STATS.load_statistics(path)

    def test_cli_success_and_error(self):
        source = self.source()
        output = self.directory / 'cli.sqlite'
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), str(source), '--sqlite', str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source.write_text('{}\n')
        failed = subprocess.run(
            [sys.executable, str(SCRIPT), str(source), '--sqlite', str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(failed.returncode, 1)
        self.assertIn('rocjitsu-stats:', failed.stderr)
        self.assertNotIn('Traceback', failed.stderr)


if __name__ == '__main__':
    unittest.main()
