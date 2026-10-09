#!/usr/bin/env python3
"""Producer integration tests; all measurements are fictional fixtures."""

import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / 'prepare-dashboard-data.py'
PUBLISH = HERE.parent / 'publish-dashboard-run.py'
SHA = 'a' * 40
CORPUS = 'b' * 40


def raw(mode='single'):
    return {
        'execution_summary': {
            'warmups': 1,
            'samples': 3,
            'timeout_seconds': 60,
            'benchmark_suite': 'smoke',
            'threading_mode': mode,
            'timestamp': '2026-01-01T01:00:00Z',
            'finished_at': '2026-01-01T01:01:00Z',
            'wall_time_s': 60,
        },
        'benchmark_results': [
            {
                'id': 'case-1',
                'name': 'Example',
                'target': 'gfx950',
                'suite': 'smoke',
                'problem': {'size': 8},
                'status': 'completed',
                'exit_code': 0,
                'error': None,
                'timing_results_s': [1, 9, 2],
            }
        ],
        'provenance': {
            'machine': {
                'hostname': 'worker',
                'platform': 'Linux',
                'kernel': '6.8',
                'cpu': 'Example CPU',
            },
            'rocjitsu': {
                'rocjitsu_commit_sha': SHA,
                'rocjitsu_commit_timestamp': '2026-01-01T00:00:00Z',
                'target_config_sha256': {'gfx950': 'c' * 64},
                'rocm_sdk_version': '7.0',
            },
            'corpus': {
                'corpus_commit_sha': CORPUS,
                'corpus_commit_timestamp': '2025-12-01T00:00:00Z',
            },
            'auxiliary': {},
        },
    }


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'

    def invoke(self, inputs=None, run_id='attempt-1', extra=(), env=None):
        extra = list(extra)
        publication = self.data
        filtered = []
        index = 0
        while index < len(extra):
            if extra[index] == '--data-dir':
                publication = Path(extra[index + 1])
                index += 2
                continue
            filtered.append(extra[index])
            index += 1
        staging = Path(tempfile.mkdtemp(dir=self.root))
        args = [
            sys.executable,
            str(SCRIPT),
            '--data-dir',
            str(staging),
            '--run-id',
            run_id,
            '--repository',
            'https://github.com/ROCm/rocm-systems',
            '--expected-sha',
            SHA,
            '--expected-corpus-sha',
            CORPUS,
            '--machine-id',
            'worker-1',
            '--trigger',
            'auto',
            '--branch',
            'develop',
            '--commit-message',
            'Example commit',
        ]
        for i, value in enumerate(inputs if inputs is not None else [raw()]):
            path = self.root / f'raw-{i}.json'
            path.write_text(json.dumps(value))
            args.extend(['--raw-run', str(path)])
        built = subprocess.run(args + filtered, text=True, capture_output=True, env=env)
        if built.returncode != 0:
            return built
        self.assertFalse((staging / 'index.json').exists())
        self.assertFalse((staging / 'metadata.json').exists())
        return subprocess.run(
            [
                sys.executable,
                str(PUBLISH),
                '--from',
                str(staging),
                '--data-dir',
                str(publication),
                '--repository',
                'https://github.com/ROCm/rocm-systems',
                '--is-beta',
            ],
            text=True,
            capture_output=True,
            env=env,
        )

    def read(self, name):
        return json.loads((self.data / name).read_text())

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def snapshot(self):
        return {
            str(p.relative_to(self.data)): p.read_bytes()
            for p in self.data.rglob('*')
            if p.is_file()
        }

    def test_cli_help_explains_ignored_metadata_compatibility_flags(self):
        for script in (SCRIPT, PUBLISH):
            result = subprocess.run(
                [sys.executable, str(script), '--help'], text=True, capture_output=True
            )
            self.assert_ok(result)
            self.assertIn('compatibility', result.stdout.lower())
            self.assertIn('ignored', result.stdout.lower())
            self.assertIn('bundled', result.stdout.lower())

    def test_cli_requires_raw_runs_not_a_prepared_dataset(self):
        prepared = self.root / 'prepared'
        self.assert_ok(self.invoke(extra=('--data-dir', str(prepared))))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                '--prepared-data',
                str(prepared),
                '--data-dir',
                str(self.data),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('--raw-run', result.stderr)
        self.assertFalse(self.data.exists())

    def test_builder_leaves_existing_index_and_metadata_untouched(self):
        staging = self.root / 'staging'
        staging.mkdir()
        metadata = staging / 'metadata.json'
        index = staging / 'index.json'
        metadata.write_bytes(b'{"keep": true}\n')
        index.write_bytes(b'{"runFiles": []}\n')
        raw_path = self.root / 'one-raw.json'
        raw_path.write_text(json.dumps(raw()))
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                '--raw-run',
                str(raw_path),
                '--data-dir',
                str(staging),
                '--run-id',
                'attempt-1',
                '--repository',
                'https://github.com/ROCm/rocm-systems',
                '--expected-sha',
                SHA,
                '--expected-corpus-sha',
                CORPUS,
                '--machine-id',
                'worker-1',
                '--trigger',
                'auto',
                '--branch',
                'develop',
                '--commit-message',
                'Example commit',
            ],
            text=True,
            capture_output=True,
        )
        self.assert_ok(result)
        self.assertEqual(metadata.read_bytes(), b'{"keep": true}\n')
        self.assertEqual(index.read_bytes(), b'{"runFiles": []}\n')
        self.assertTrue(
            (staging / 'runs' / 'default-branch' / 'attempt-1.json').is_file()
        )

    def test_preparer_runs_without_node_or_path(self):
        environment = dict(os.environ)
        environment.pop('PATH', None)
        # An empty PATH also prevents platform-default executable lookup.
        environment['PATH'] = ''
        self.assert_ok(self.invoke(env=environment))
        self.assertEqual(
            self.read('index.json')['runFiles'], ['runs/default-branch/attempt-1.json']
        )

    def test_produces_schema_two_consumable_median_results(self):
        self.assert_ok(self.invoke())
        self.assertFalse((self.data / 'metadata.json').exists())
        run = self.read('runs/default-branch/attempt-1.json')
        self.assertIs(type(run['schemaVersion']), int)
        self.assertEqual(run['schemaVersion'], 2)
        self.assertNotIn('schemaVersion', self.read('index.json'))
        self.assertNotIn('schemaVersion', self.read(run['testCatalog']))
        self.assertEqual(run['source']['commit'], SHA)
        self.assertEqual(run['execution']['completedAt'], '2026-01-01T01:01:00Z')
        config = run['configurations'][0]
        self.assertEqual(config['threadingMode'], 'ST')
        self.assertEqual(config['results'][0]['durationSeconds'], 2)
        self.assertEqual(config['results'][0]['error'], None)
        self.assertEqual(
            self.read('index.json')['runFiles'], ['runs/default-branch/attempt-1.json']
        )

    def test_uppercase_git_shas_are_published_in_lowercase(self):
        value = raw()
        value['provenance']['rocjitsu']['rocjitsu_commit_sha'] = SHA.upper()
        value['provenance']['corpus']['corpus_commit_sha'] = CORPUS.upper()
        self.assert_ok(
            self.invoke(
                [value],
                extra=(
                    '--expected-sha',
                    SHA.upper(),
                    '--expected-corpus-sha',
                    CORPUS.upper(),
                ),
            )
        )
        run = self.read('runs/default-branch/attempt-1.json')
        self.assertEqual(run['source']['commit'], SHA)
        details = {entry['key']: entry['value'] for entry in run['environment']}
        self.assertEqual(details['corpus.corpus_commit_sha'], CORPUS)
        before = self.snapshot()
        self.assert_ok(self.invoke())
        self.assertEqual(self.snapshot(), before)

    def test_git_sha_provenance_matching_ignores_only_hex_case(self):
        single, multi = raw(), raw('default')
        single['provenance']['rocjitsu']['rocjitsu_commit_sha'] = 'Aa' * 20
        single['provenance']['corpus']['corpus_commit_sha'] = CORPUS.upper()
        self.assert_ok(
            self.invoke(
                [single, multi],
                extra=(
                    '--expected-sha',
                    SHA.upper(),
                    '--expected-corpus-sha',
                    'bB' * 20,
                ),
            )
        )
        run = self.read('runs/default-branch/attempt-1.json')
        self.assertEqual(run['source']['commit'], SHA)
        self.assertEqual(len(run['configurations']), 2)
        details = {entry['key']: entry['value'] for entry in run['environment']}
        self.assertEqual(details['corpus.corpus_commit_sha'], CORPUS)
        before = self.snapshot()
        for flag, expected in (
            ('--expected-sha', 'D' * 40),
            ('--expected-corpus-sha', 'D' * 40),
            ('--expected-sha', 'A' * 39),
            ('--expected-corpus-sha', 'B' * 40 + ' '),
        ):
            with self.subTest(flag=flag, expected=expected):
                result = self.invoke([single, multi], extra=(flag, expected))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('SHA', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
                self.assertEqual(self.snapshot(), before)

    def test_completed_result_accepts_null_exit_code(self):
        value = raw()
        value['benchmark_results'][0]['exit_code'] = None
        self.assert_ok(self.invoke([value]))
        result = self.read('runs/default-branch/attempt-1.json')['configurations'][0][
            'results'
        ][0]
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['durationSeconds'], 2)
        self.assertIsNone(result['error'])

    def test_explicit_modes_share_definitions_with_asymmetric_membership(self):
        single, multi = raw(), raw('default')
        single['execution_summary']['benchmark_suite'] = 'nightly-single'
        multi['execution_summary']['benchmark_suite'] = 'nightly'
        single['benchmark_results'][0]['id'] = 'example.single'
        multi['benchmark_results'][0]['id'] = 'example.default'
        problem = {
            'nested': {'z': None, 'a': [1, False]},
            'nil': None,
            'bool': False,
            'zero': 0,
            'text': 'value',
        }
        for value in (single, multi):
            value['benchmark_results'][0]['problem'] = copy.deepcopy(problem)
            value['provenance']['auxiliary'] = {'nested': {'b': 1, 'a': None}}
        multi['provenance']['rocjitsu']['target_config_sha256']['gfx950'] = 'd' * 64
        failed = copy.deepcopy(single['benchmark_results'][0])
        failed.update(
            id='failure',
            name='Failed workload',
            status='failed',
            error='Exit failure',
            exit_code=1,
            timing_results_s=[5],
        )
        timeout = copy.deepcopy(failed)
        timeout.update(
            id='timeout', name='Timeout workload', status='timeout', error=None
        )
        single['benchmark_results'].extend([failed, timeout])
        self.assert_ok(self.invoke([single, multi]))
        run = self.read('runs/default-branch/attempt-1.json')
        configs = {c['threadingMode']: c for c in run['configurations']}
        st_success = next(
            r for r in configs['ST']['results'] if r['status'] == 'completed'
        )
        self.assertEqual(st_success['testId'], configs['MT']['results'][0]['testId'])
        for result in configs['ST']['results']:
            if result['status'] != 'completed':
                self.assertIsNone(result['durationSeconds'])
        catalog = self.read(run['testCatalog'])
        definition = next(
            t for t in catalog['tests'] if t['id'] == st_success['testId']
        )
        self.assertEqual(
            definition['problem'],
            {
                'nested': '{"a":[1,false],"z":null}',
                'nil': 'null',
                'bool': False,
                'zero': 0,
                'text': 'value',
            },
        )
        details = {entry['key']: entry['value'] for entry in run['environment']}
        self.assertEqual(details['machine.hostname'], 'worker')
        self.assertEqual(
            details['single.rocjitsu.target_config_sha256'],
            '{"gfx950":"' + 'c' * 64 + '"}',
        )
        self.assertEqual(
            details['default.rocjitsu.target_config_sha256'],
            '{"gfx950":"' + 'd' * 64 + '"}',
        )
        self.assertEqual(details['auxiliary.nested'], '{"a":null,"b":1}')
        self.assertFalse(
            any(
                'timestamp' in key or 'wall_time' in key
                for key in details
                if key.startswith(('single.execution.', 'default.execution.'))
            )
        )

    def test_history_retry_and_definition_changes_are_immutable(self):
        self.assert_ok(self.invoke())
        original = self.snapshot()
        self.assert_ok(self.invoke())
        self.assertEqual(
            self.snapshot(), original, 'Exact retry must be byte-preserving'
        )
        revised = raw()
        revised['execution_summary'].update(
            timestamp='2026-01-02T01:00:00Z',
            finished_at='2026-01-02T01:01:00Z',
            wall_time_s=50,
        )
        self.assert_ok(self.invoke([revised], run_id='attempt-2'))
        old = self.read('runs/default-branch/attempt-1.json')
        new = self.read('runs/default-branch/attempt-2.json')
        self.assertEqual(old['testCatalog'], new['testCatalog'])
        self.assertEqual(old['environment'], new['environment'])
        revised['benchmark_results'][0]['problem']['size'] = 16
        self.assert_ok(self.invoke([revised], run_id='attempt-3'))
        changed = self.read('runs/default-branch/attempt-3.json')
        self.assertNotEqual(old['testCatalog'], changed['testCatalog'])
        self.assertNotEqual(
            old['configurations'][0]['results'][0]['testId'],
            changed['configurations'][0]['results'][0]['testId'],
        )
        self.assertEqual(
            self.read('index.json')['runFiles'],
            [
                'runs/default-branch/attempt-1.json',
                'runs/default-branch/attempt-2.json',
                'runs/default-branch/attempt-3.json',
            ],
        )
        self.assertEqual(
            (self.data / 'runs/default-branch/attempt-1.json').read_bytes(),
            original['runs/default-branch/attempt-1.json'],
        )
        before_conflict = self.snapshot()
        conflict = self.invoke([revised])
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn('immutable', conflict.stderr.lower())
        self.assertEqual(self.snapshot(), before_conflict)

    def test_invalid_raw_facts_reject_before_any_writes(self):
        cases = [
            (('execution_summary', 'finished_at'), None),
            (('execution_summary', 'finished_at'), '2026-02-30T01:00:00Z'),
            (('execution_summary', 'timestamp'), '2026-01-01'),
            (('execution_summary', 'timestamp'), '2026-01-02T01:00:00Z'),
            (('execution_summary', 'warmups'), -1),
            (('execution_summary', 'warmups'), True),
            (('execution_summary', 'samples'), 0),
            (('execution_summary', 'samples'), 1.5),
            (('execution_summary', 'timeout_seconds'), 0),
            (('execution_summary', 'wall_time_s'), float('inf')),
            (('execution_summary', 'threading_mode'), 'ST'),
            (('provenance', 'rocjitsu', 'rocjitsu_commit_sha'), None),
            (('provenance', 'rocjitsu', 'rocjitsu_commit_sha'), 'd' * 40),
            (('provenance', 'rocjitsu', 'rocjitsu_commit_timestamp'), None),
            (('provenance', 'rocjitsu', 'rocm_sdk_version'), ''),
            (('provenance', 'rocjitsu', 'target_config_sha256'), {}),
            (('provenance', 'rocjitsu', 'target_config_sha256'), {'gfx950': 'oops'}),
            (('provenance', 'corpus', 'corpus_commit_sha'), 'd' * 40),
            (('provenance', 'corpus', 'corpus_commit_timestamp'), None),
            (('provenance', 'machine', 'hostname'), ''),
            (('provenance', 'auxiliary'), []),
            (('benchmark_results', 0, 'timing_results_s'), []),
            (('benchmark_results', 0, 'timing_results_s'), [0]),
            (('benchmark_results', 0, 'timing_results_s'), [True]),
            (('benchmark_results', 0, 'timing_results_s'), [float('nan')]),
            (('benchmark_results', 0, 'exit_code'), '0'),
            (('benchmark_results', 0, 'exit_code'), 1),
            (('benchmark_results', 0, 'error'), 'completed with error'),
            (('benchmark_results', 0, 'problem'), []),
            (('benchmark_results', 0, 'name'), ''),
            (('benchmark_results', 0, 'target'), '../escape'),
        ]
        for keys, value in cases:
            with self.subTest(keys=keys, value=value):
                record = raw()
                obj = record
                for key in keys[:-1]:
                    obj = obj[key]
                obj[keys[-1]] = value
                result = self.invoke([record])
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('error:', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
                self.assertFalse(self.data.exists())

    def test_cross_input_provenance_and_protocol_must_agree(self):
        changes = [
            (('provenance', 'machine', 'cpu'), 'Different CPU'),
            (('provenance', 'rocjitsu', 'rocm_sdk_version'), '8.0'),
            (
                ('provenance', 'rocjitsu', 'rocjitsu_commit_timestamp'),
                '2025-12-31T00:00:00Z',
            ),
            (
                ('provenance', 'corpus', 'corpus_commit_timestamp'),
                '2025-11-30T00:00:00Z',
            ),
            (('provenance', 'auxiliary'), {'different': True}),
            (('execution_summary', 'samples'), 4),
            (('execution_summary', 'warmups'), 2),
            (('execution_summary', 'timeout_seconds'), 120),
        ]
        for keys, value in changes:
            with self.subTest(keys=keys):
                multi = raw('default')
                obj = multi
                for key in keys[:-1]:
                    obj = obj[key]
                obj[keys[-1]] = value
                result = self.invoke([raw(), multi])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('inconsistent', result.stderr.lower())
                self.assertFalse(self.data.exists())

    def test_reordered_inputs_are_idempotent_and_completion_uses_instants(self):
        single, multi = raw(), raw('default')
        single['execution_summary']['finished_at'] = '2026-01-01T04:00:00+02:00'
        multi['execution_summary']['finished_at'] = '2026-01-01T03:00:00Z'
        for value in (single, multi):
            extra = copy.deepcopy(value['benchmark_results'][0])
            extra.update(id='second', name='Another definition')
            value['benchmark_results'].append(extra)
        self.assert_ok(self.invoke([single, multi]))
        self.assertEqual(
            self.read('runs/default-branch/attempt-1.json')['execution']['completedAt'],
            '2026-01-01T03:00:00Z',
        )
        original = self.snapshot()
        single['benchmark_results'].reverse()
        multi['benchmark_results'].reverse()
        self.assert_ok(self.invoke([multi, single]))
        self.assertEqual(self.snapshot(), original)

    def test_path_traversal_rejected_without_staging_escape(self):
        for run_id in ('../../escaped', '/absolute', 'a/b', '%2e%2e', '.', '..', 'a?b'):
            with self.subTest(run_id=run_id):
                result = self.invoke(
                    run_id=run_id, env={**os.environ, 'TMPDIR': str(self.root)}
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('run-id', result.stderr)
                self.assertFalse((self.root / 'escaped.json').exists())
                self.assertFalse(self.data.exists())

    def test_symlink_output_rejected_without_touching_external_files(self):
        outside = self.root / 'outside'
        outside.mkdir()
        marker = outside / 'marker'
        marker.write_text('keep')
        self.data.mkdir()
        (self.data / 'runs').symlink_to(outside, target_is_directory=True)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('symlink', result.stderr.lower())
        self.assertEqual(list(outside.iterdir()), [marker])
        self.assertFalse((self.data / 'metadata.json').exists())
        (self.data / 'runs').unlink()
        self.data.rmdir()
        self.data.symlink_to(outside, target_is_directory=True)
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('symlink', result.stderr.lower())
        self.assertEqual(list(outside.iterdir()), [marker])

    def test_cli_errors_are_actionable(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT)], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn('--raw-run', result.stderr)
        for extra in (
            ('--trigger', 'invalid'),
            ('--expected-sha', 'short'),
            ('--repository', 'file:///not-a-repository'),
            ('--branch', ''),
        ):
            with self.subTest(extra=extra):
                result = self.invoke(extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('Traceback', result.stderr)
                self.assertFalse(self.data.exists())

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / 'duplicate.json'
        value = json.dumps(raw()).replace('"warmups": 1', '"warmups": -1, "warmups": 1')
        path.write_text(value)
        result = self.invoke(inputs=[], extra=('--raw-run', str(path)))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Duplicate JSON key', result.stderr)
        self.assertFalse(self.data.exists())

    def test_stale_metadata_is_ignored_and_history_preserved_for_external_validation(
        self,
    ):
        self.data.mkdir()
        metadata = self.data / 'metadata.json'
        metadata.write_bytes(b'not even JSON: stale website settings\n')
        self.assert_ok(self.invoke())
        self.assertEqual(
            metadata.read_bytes(), b'not even JSON: stale website settings\n'
        )
        before = self.snapshot()
        self.assert_ok(self.invoke())
        self.assertEqual(self.snapshot(), before)
        old_run = self.read('runs/default-branch/attempt-1.json')
        old_run['configurations'][0]['results'] = []
        (self.data / 'runs/default-branch/attempt-1.json').write_text(
            json.dumps(old_run)
        )
        before = self.snapshot()
        # The separate workflow validator owns full wire-schema checks. Preparation
        # must preserve existing run bytes, including invalid historical results.
        self.assert_ok(self.invoke(run_id='attempt-2'))
        self.assertEqual(
            (self.data / 'runs/default-branch/attempt-1.json').read_bytes(),
            before['runs/default-branch/attempt-1.json'],
        )
        self.assertEqual(
            self.read('index.json')['runFiles'],
            [
                'runs/default-branch/attempt-1.json',
                'runs/default-branch/attempt-2.json',
            ],
        )

    def test_staged_entire_history_checks_source_timestamp_conflicts(self):
        self.assert_ok(self.invoke())
        before = self.snapshot()
        record = raw()
        record['provenance']['rocjitsu'][
            'rocjitsu_commit_timestamp'
        ] = '2025-12-31T00:00:00Z'
        result = self.invoke([record], run_id='attempt-2')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('conflicting committedAt', result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_orphan_immutable_resources_are_reused_but_conflicts_rejected(self):
        self.assert_ok(self.invoke())
        index = self.read('index.json')
        index['runFiles'] = []
        (self.data / 'index.json').write_text(json.dumps(index))
        existing_run = (self.data / 'runs/default-branch/attempt-1.json').read_bytes()
        self.assert_ok(self.invoke())
        self.assertEqual(
            self.read('index.json')['runFiles'], ['runs/default-branch/attempt-1.json']
        )
        self.assertEqual(
            (self.data / 'runs/default-branch/attempt-1.json').read_bytes(),
            existing_run,
        )
        (self.data / 'index.json').write_text(json.dumps(index))
        run = self.read('runs/default-branch/attempt-1.json')
        catalog = self.read(run['testCatalog'])
        catalog['tests'][0]['name'] = 'Corrupted orphan'
        (self.data / run['testCatalog']).write_text(json.dumps(catalog))
        before = self.snapshot()
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Immutable resource conflict', result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_duplicate_logical_results_and_future_completion_are_rejected(self):
        value = raw()
        value['benchmark_results'].append(copy.deepcopy(value['benchmark_results'][0]))
        result = self.invoke([value])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.data.exists())
        value = raw()
        value['execution_summary']['finished_at'] = '9999-01-01T00:00:00Z'
        result = self.invoke([value])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('completed by publication', result.stderr)
        self.assertFalse(self.data.exists())

    def test_content_addressed_catalog_and_target_independent_workloads(self):
        import hashlib

        value = raw()
        other = copy.deepcopy(value['benchmark_results'][0])
        other.update(id='case-for-other-target', target='gfx1250')
        value['benchmark_results'].append(other)
        value['provenance']['rocjitsu']['target_config_sha256']['gfx1250'] = 'e' * 64
        self.assert_ok(
            self.invoke(
                [value], extra=('--branch', 'feature/example', '--trigger', 'manual')
            )
        )
        self.assertEqual(
            self.read('index.json')['runFiles'], ['runs/side-branches/attempt-1.json']
        )
        run = self.read('runs/side-branches/attempt-1.json')
        self.assertEqual(run['execution']['trigger'], 'manual')
        self.assertEqual(run['source']['branch'], 'feature/example')
        self.assertEqual(
            run['configurations'][0]['results'][0]['testId'],
            run['configurations'][1]['results'][0]['testId'],
        )
        catalog = self.read(run['testCatalog'])
        catalog_id = catalog.pop('id')
        encoded = json.dumps(
            catalog, sort_keys=True, separators=(',', ':'), ensure_ascii=True
        ).encode()
        self.assertEqual(catalog_id, 'catalog-' + hashlib.sha256(encoded).hexdigest())
        self.assertEqual(run['execution']['machine'], 'worker-1')

    def test_retry_preserves_existing_configuration_order_for_prefix_targets(self):
        value = raw()
        variant = copy.deepcopy(value['benchmark_results'][0])
        variant.update(id='variant-case', target='gfx950-variant')
        value['benchmark_results'].append(variant)
        value['provenance']['rocjitsu']['target_config_sha256']['gfx950-variant'] = (
            'd' * 64
        )
        self.assert_ok(self.invoke([value]))
        run = self.read('runs/default-branch/attempt-1.json')
        # Existing runs order configurations by the combined target:mode key.
        run['configurations'].sort(
            key=lambda item: item['target'] + ':' + item['threadingMode']
        )
        (self.data / 'runs/default-branch/attempt-1.json').write_text(json.dumps(run))
        before = self.snapshot()
        self.assert_ok(self.invoke([value]))
        self.assertEqual(self.snapshot(), before)

    def test_source_revision_does_not_change_environment_identity(self):
        self.assert_ok(self.invoke())
        first = self.read('runs/default-branch/attempt-1.json')
        value = raw()
        value['provenance']['rocjitsu'].update(
            rocjitsu_commit_sha='f' * 40,
            rocjitsu_commit_timestamp='2026-01-01T00:30:00Z',
        )
        self.assert_ok(
            self.invoke([value], run_id='attempt-2', extra=('--expected-sha', 'f' * 40))
        )
        second = self.read('runs/default-branch/attempt-2.json')
        self.assertEqual(first['environment'], second['environment'])
        self.assertNotEqual(first['source']['commit'], second['source']['commit'])

    def test_same_mode_inputs_cannot_silently_overwrite_provenance(self):
        first, second = raw(), raw()
        second['benchmark_results'][0]['target'] = 'gfx1250'
        second['provenance']['rocjitsu']['target_config_sha256'] = {'gfx1250': 'd' * 64}
        result = self.invoke([first, second])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Duplicate threading_mode', result.stderr)
        self.assertFalse(self.data.exists())

    def test_immutable_comparison_distinguishes_booleans_from_numbers(self):
        value = raw()
        value['benchmark_results'][0]['problem'] = {'flag': False}
        self.assert_ok(self.invoke([value]))
        run = self.read('runs/default-branch/attempt-1.json')
        catalog = self.read(run['testCatalog'])
        catalog['tests'][0]['problem']['flag'] = 0
        (self.data / run['testCatalog']).write_text(json.dumps(catalog))
        before = self.snapshot()
        result = self.invoke([value])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Immutable resource conflict', result.stderr)
        self.assertEqual(before, self.snapshot())


# The optional export is fictional test data for the workflow's standalone validator.
# It is never used by the benchmark publication job.
def test_export_prepared_data_for_workflow(tmp_path):
    single, multi = raw(), raw('default')
    for value in (single, multi):
        value['benchmark_results'][0]['problem']['nested'] = {'values': [1, None, True]}
        for status in ('failed', 'timeout'):
            result = copy.deepcopy(value['benchmark_results'][0])
            result.update(
                id=status,
                name=status,
                status=status,
                exit_code=None,
                error='Fictional diagnostic',
                timing_results_s=[],
            )
            value['benchmark_results'].append(result)
    inputs = []
    for name, value in (('single', single), ('multi', multi)):
        file = tmp_path / f'{name}.json'
        file.write_text(json.dumps(value))
        inputs.extend(['--raw-run', str(file)])
    destination = tmp_path / 'prepared'
    for branch, run_id in (
        ('develop', 'develop-fixture'),
        ('feature/example', 'branch-fixture'),
    ):
        build = tmp_path / f'build-{run_id}'
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                *inputs,
                '--data-dir',
                str(build),
                '--run-id',
                run_id,
                '--repository',
                'https://github.com/ROCm/rocm-systems',
                '--expected-sha',
                SHA,
                '--expected-corpus-sha',
                CORPUS,
                '--machine-id',
                'fictional-worker',
                '--trigger',
                'manual',
                '--branch',
                branch,
                '--commit-message',
                'Fictional fixture',
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (build / 'index.json').exists()
        assert not (build / 'metadata.json').exists()
        result = subprocess.run(
            [
                sys.executable,
                str(PUBLISH),
                '--from',
                str(build),
                '--data-dir',
                str(destination),
                '--repository',
                'https://github.com/ROCm/rocm-systems',
                '--is-beta',
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    index = json.loads((destination / 'index.json').read_text())
    assert len(index['runFiles']) == 2
    assert index['runFiles'] == [
        'runs/default-branch/develop-fixture.json',
        'runs/side-branches/branch-fixture.json',
    ]
    assert 'schemaVersion' not in index
    assert not (destination / 'metadata.json').exists()
    result = subprocess.run(
        [
            'node',
            str(HERE.parent.parent / 'validate-dashboard-data.mjs'),
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    if output := os.environ.get('DASHBOARD_VALIDATION_DIR'):
        shutil.copytree(destination, output)


if __name__ == '__main__':
    unittest.main()
