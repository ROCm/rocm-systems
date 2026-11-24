# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check wrapper evidence and replay cleanup without IREE or a GPU."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import consan_iree_prefill_e2e as prefill


def complete_analysis():
    keys = {prefill.SCATTER: 2048, prefill.ELEMENTWISE: 2112}
    entries = [
        {'entry': name, 'entry_key': key, 'disposition': 'selected'}
        for name, key in keys.items()
    ]
    entries.extend(
        {
            'entry': f'native_{index}',
            'entry_key': 2176 + index * 64,
            'disposition': 'native_uncovered',
        }
        for index in range(53)
    )
    executions = []
    for serial, name in enumerate(
        (prefill.SCATTER, prefill.ELEMENTWISE, prefill.SCATTER) * 3, 1
    ):
        groups = 16 if name == prefill.SCATTER else 1
        executions.append(
            {
                'serial': serial,
                'entry': name,
                'entry_key': keys[name],
                'grid': [groups * 32, 1, 1],
                'workgroup': [32, 1, 1],
                'expected_rows': groups,
                'completed_rows': groups,
                'entry_events': groups,
                'exit_events': groups,
                'integrity_events': 0,
                'issues': [],
            }
        )
    return {
        'scope': 'execution_trace',
        'lds_analysis': 'not_performed',
        'transport_complete': True,
        'entries': entries,
        'executions': executions,
        'expected_rows': 99,
        'completed_rows': 99,
        'entry_events': 99,
        'exit_events': 99,
    }


def lds_analysis():
    executions = [
        {
            'serial': serial,
            'entry': 'prefill_bs1$async_dispatch_3_matmul_Dx128x256_f16xf16xf32',
            'entry_key': 2048,
            'workgroup_count': 2,
            'waves_per_workgroup': 4,
            'valid_observations': 4608,
            'rejected_records': 0,
            'integrity_events': 0,
            'lost_occurrences': 576,
            'synchronization_loss': 0,
            'address_group_exhaustion': 576,
            'counter_exhausted_waves': 0,
            'coverage_complete': False,
            'issues': [{'reason': 'range_reservation_exhausted'}],
        }
        for serial in range(1, 7)
    ]
    return {
        'verdict': 'no_observed_findings',
        'scope': 'native_lds_multi_range_epoch_workgroup_barriers',
        'lds_analysis': 'performed',
        'execution_conditions': 8,
        'coverage_complete': False,
        'entries': [
            {
                'entry': executions[0]['entry'],
                'entry_key': 2048,
                'disposition': 'selected',
            }
        ]
        + [
            {
                'entry': f'native_{index}',
                'entry_key': 2112 + index * 64,
                'disposition': 'native_uncovered',
            }
            for index in range(54)
        ],
        'executions': executions,
    }


class PrefillEvidence(unittest.TestCase):
    def test_static_summary_uses_owning_index_and_skips_native_only_images(self):
        commands = mock.Mock()
        commands.run.return_value = '{"entries": []}'
        root = Path('/private/run')
        identity = '11' * 32
        index = {
            'images': [
                {
                    'preparation': {'original_sha256': identity},
                    'capture': 'captures/model.bin',
                },
                {'preparation': {'original_sha256': '22' * 32}, 'capture': None},
            ]
        }
        summaries = prefill.collect_capture_summaries(
            commands,
            root / 'inspect',
            root / 'trace.bin',
            root / 'lds.bin',
            index,
            root,
        )
        self.assertEqual(
            summaries,
            {
                'trace': {'entries': []},
                'pinned_lds': {'entries': []},
                'all_images': [{'entries': []}],
            },
        )
        self.assertEqual(
            commands.run.call_args_list,
            [
                mock.call(
                    'inspect-trace', [root / 'inspect', '--summary', root / 'trace.bin']
                ),
                mock.call(
                    'inspect-lds', [root / 'inspect', '--summary', root / 'lds.bin']
                ),
                mock.call(
                    'inspect-all-' + identity,
                    [root / 'inspect', '--summary', root / 'captures/model.bin'],
                ),
            ],
        )

    def test_independent_case_evidence_merges_without_replacing_sibling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'commands.json').write_text('[]')
            args = SimpleNamespace(fixture=root, cpu_thread_budget=4)
            fixture = {'baseline': {'passed': True}}
            cases = {
                'trace': {
                    'passed': True,
                    'capture_summaries': {'trace': {'entries': ['trace witness']}},
                    'preparation_reports': {
                        'trace': {'requested_scope': 'execution_trace'}
                    },
                },
                'lds': {
                    'passed': True,
                    'capture_summaries': {
                        'pinned_lds': {'entries': ['LDS witness']},
                        'all_images': [{'entries': ['automatic selection witness']}],
                    },
                    'preparation_reports': {'pinned_lds': {'requested_scope': 'lds'}},
                },
            }
            report_json, report_text = (
                root / 'coverage-report.json',
                root / 'coverage-report.txt',
            )
            report_json.write_text('{}')
            report_text.write_text('report')
            with mock.patch.object(
                prefill, 'write_report', return_value=(report_json, report_text)
            ) as report, mock.patch('builtins.print'):
                combined = prefill.combine_case_results(args, fixture, cases)
                report.assert_called_once_with(root / 'result.json')
                self.assertEqual(
                    set(combined['capture_summaries']),
                    {'trace', 'pinned_lds', 'all_images'},
                )
                self.assertEqual(
                    set(combined['preparation_reports']), {'trace', 'pinned_lds'}
                )
                self.assertEqual(
                    combined['capture_summaries']['trace'],
                    cases['trace']['capture_summaries']['trace'],
                )
                self.assertEqual(
                    combined['capture_summaries']['pinned_lds'],
                    cases['lds']['capture_summaries']['pinned_lds'],
                )
                report.reset_mock()
                cases['trace']['passed'] = False
                combined = prefill.combine_case_results(args, fixture, cases)
                self.assertFalse(combined['passed'])
                report.assert_not_called()
                cases['lds']['capture_summaries']['trace'] = {
                    'entries': ['wrong owner']
                }
                with self.assertRaisesRegex(
                    ValueError, 'duplicate case evidence scope'
                ):
                    prefill.combine_case_results(args, fixture, cases)

    def test_all_image_evidence_requires_every_attempt_and_actual_lds_observations(
        self,
    ):
        native = lds_analysis()
        native['entries'][0]['entry'] = prefill.LDS_FIRST_MATMUL
        native['entries'][1]['entry'] = prefill.LDS_MATMUL
        native['executions'] = native['executions'][:4]
        for execution in native['executions']:
            execution.update(
                entry=prefill.LDS_FIRST_MATMUL,
                workgroup_count=4,
                valid_observations=9216,
                lost_occurrences=1152,
                address_group_exhaustion=1152,
            )
        model_entries = [
            dict(
                entry,
                disposition=(
                    'instrumented'
                    if entry['disposition'] == 'selected'
                    else 'native_uncovered'
                ),
                reason_code=(
                    'none'
                    if entry['disposition'] == 'selected'
                    else 'unsupported_instruction'
                ),
                reason='' if entry['disposition'] == 'selected' else 'outside subset',
            )
            for entry in native['entries']
        ]
        model_entries[1]['reason_code'] = 'unsupported_control_transfer_range'
        index = {
            'schema_version': 2,
            'scope': 'provided_application_images',
            'images': [
                {
                    'preparation': {
                        'original_sha256': '11' * 32,
                        'requested': 55,
                        'entries': model_entries,
                    }
                },
                {
                    'preparation': {
                        'original_sha256': '22' * 32,
                        'requested': 13,
                        'entries': [
                            {
                                'entry': f'helper_{index}',
                                'disposition': 'native_uncovered',
                                'reason_code': 'unsupported_entry',
                                'reason': 'no LDS sites',
                            }
                            for index in range(13)
                        ],
                    }
                },
            ],
        }
        for image_index, image in enumerate(index['images']):
            prepared = image['preparation']
            for entry_index, entry in enumerate(prepared['entries']):
                entry.setdefault('entry_key', 8192 + entry_index * 64)
                no_probes = (
                    (2 <= entry_index < 50) if image_index == 0 else entry_index < 12
                )
                entry.update(
                    stage='qualification', no_probes_needed=None, text_offset=None
                )
                if no_probes:
                    entry.update(
                        disposition='no_probes_needed',
                        stage='applicability',
                        reason_code='none',
                        reason='closed original code has no LDS accesses',
                        no_probes_needed={
                            'version': 1,
                            'scope': 'lds',
                            'instruction_count': 1,
                            'code_bytes': 4,
                            'code_sha256': '33' * 32,
                        },
                    )
                elif entry['disposition'] != 'instrumented':
                    entry['disposition'] = 'blocked'
            counts = {
                name: sum(entry['disposition'] == name for entry in prepared['entries'])
                for name in ('instrumented', 'no_probes_needed', 'blocked')
            }
            prepared.update(
                schema_version=2,
                requested_scope='lds',
                target=1250,
                inventory_entries=len(prepared['entries']),
                not_requested=0,
                original_bytes=1 << 20,
                skipped=counts['blocked'],
                outcome='instrumented' if counts['instrumented'] else 'native_only',
                selection_profile='44' * 32 if counts['instrumented'] else None,
                requested_coverage='partial',
                **counts,
            )
        analysis = {
            'scope': 'provided_application_images',
            'whole_application_coverage': 'not_established',
            'lds_analysis': 'performed',
            'lds_observations_present': True,
            'coverage_complete': False,
            'images': [
                {'original_sha256': '11' * 32, 'analysis': native},
                {'original_sha256': '22' * 32, 'analysis': None},
            ],
        }
        prefill.validate_all_images(index, analysis)
        for field, value in (
            ('scope', 'application'),
            ('lds_analysis', 'not_performed'),
            ('lds_observations_present', False),
            ('coverage_complete', True),
        ):
            changed = deepcopy(analysis)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_all_images(index, changed)
        changed = deepcopy(index)
        changed['images'][0]['preparation']['requested'] = 1
        with self.assertRaisesRegex(ValueError, 'every model and helper'):
            prefill.validate_all_images(changed, analysis)
        changed = deepcopy(analysis)
        changed['images'].pop()
        with self.assertRaisesRegex(ValueError, 'dropped a provided image'):
            prefill.validate_all_images(index, changed)
        changed = deepcopy(analysis)
        changed['images'][0]['analysis']['executions'][0]['valid_observations'] = 0
        with self.assertRaisesRegex(ValueError, 'observation/loss'):
            prefill.validate_all_images(index, changed)

    def test_default_all_attempts_lds_after_trace_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'commands.json').write_text('[]')
            args = SimpleNamespace(case='all', fixture=None, cpu_thread_budget=4)
            fixture = {'baseline': {'passed': True}}

            def execute(_args, _tools, _fixture, case):
                if case == 'trace':
                    raise ValueError('trace probe did not execute')
                return {
                    'case': case,
                    'passed': True,
                    'stage_seconds': {},
                    'lds_analysis': {'lds_analysis': 'performed'},
                    'all_images_analysis': {'lds_analysis': 'performed'},
                    'all_images_instrumented': {'passed': True},
                }

            with mock.patch.object(
                prefill, 'validate_environment', return_value={}
            ), mock.patch.object(
                prefill, 'setup_fixture', return_value=root
            ) as setup, mock.patch.object(
                prefill, 'load_fixture', return_value=fixture
            ), mock.patch.object(
                prefill, 'run_case', side_effect=execute
            ) as cases, mock.patch.object(
                prefill.traceback, 'print_exc'
            ):
                with self.assertRaisesRegex(ValueError, 'prefill case failed'):
                    prefill.run(args)
            setup.assert_called_once()
            self.assertEqual(
                [call.args[-1] for call in cases.call_args_list], ['trace', 'lds']
            )
            result = json.loads((root / 'result.json').read_text())
            self.assertFalse(result['passed'])
            self.assertFalse(result['cases']['trace']['passed'])
            self.assertTrue(result['cases']['lds']['passed'])
            self.assertEqual(result['lds_analysis']['lds_analysis'], 'performed')
            self.assertEqual(result['all_images_analysis']['lds_analysis'], 'performed')
            self.assertTrue(result['all_images_instrumented']['passed'])

    def test_case_budget_reserves_sibling_and_reclaims_failed_case_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = {'timeout': 60, 'setup_seconds': 8}
            self.assertEqual(prefill.case_timeout(root, fixture, 'trace'), 26)
            self.assertEqual(prefill.case_timeout(root, fixture, 'lds'), 26)
            (root / 'trace').mkdir()
            with self.assertRaisesRegex(ValueError, 'sequentially'):
                prefill.case_timeout(root, fixture, 'lds')
            (root / 'trace/result.json').write_text(
                json.dumps(
                    {
                        'case': 'trace',
                        'passed': False,
                        'stage_elapsed_seconds': 3,
                    }
                )
            )
            self.assertEqual(prefill.case_timeout(root, fixture, 'lds'), 49)

    def test_failed_trace_receipt_and_lds_replay_keep_shared_inputs_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'original'
            inputs.mkdir()
            (inputs / 'model.hsaco').write_bytes(b'model')
            (inputs / 'helper.hsaco').write_bytes(b'helper')
            (root / 'baseline').mkdir()
            (root / 'baseline/logits.npy').write_bytes(b'unchanged logits')
            args = SimpleNamespace(
                fixture=root, source=root, cpu_thread_budget=4, timeout=60
            )
            fixture = {
                'timeout': 60,
                'setup_seconds': 4,
                'model': 'model.hsaco',
                'helper': 'helper.hsaco',
                'baseline': {'passed': True},
            }
            tools = {
                'prepare': root / 'prepare',
                'analyze': root / 'analyze',
                'plugin': root / 'plugin',
                'inspect': root / 'inspect',
            }
            identity = 'a' * 64
            analysis = lds_analysis() | {
                'selection_profile': identity,
                'original_sha256': prefill.digest(inputs / 'model.hsaco'),
            }

            def failed_trace(commands, stage, _argv, **_kwargs):
                commands.records.append({'stage': stage, 'exit': 2, 'seconds': 2})
                raise ValueError('trace preparation failed')

            with mock.patch.object(prefill.Commands, 'run', failed_trace), mock.patch(
                'builtins.print'
            ):
                with self.assertRaisesRegex(ValueError, 'trace preparation failed'):
                    prefill.run_case(args, tools, fixture, 'trace')
            trace = json.loads((root / 'trace/result.json').read_text())
            self.assertFalse(trace['passed'])
            self.assertEqual(trace['stage_elapsed_seconds'], 2)
            self.assertIn('trace preparation failed', trace['error'])

            all_analysis = {
                'scope': 'provided_application_images',
                'lds_analysis': 'performed',
            }

            def lds_command(commands, stage, argv, **_kwargs):
                code = (
                    1
                    if stage
                    in ('analyze-all-images', 'fresh-all-image-analysis-without-inputs')
                    else 0
                )
                commands.records.append({'stage': stage, 'exit': code, 'seconds': 0.1})
                if stage == 'prepare-lds':
                    (commands.directory / 'artifact').mkdir()
                    (commands.directory / 'artifact/preparation.json').write_text('{}')
                    self.assertIn('--require-all', argv)
                    return 'selection_profile=' + identity
                if stage == 'prepare-all-images':
                    self.assertNotIn('--entry', argv)
                    artifacts = commands.directory / 'all-images'
                    (artifacts / 'images').mkdir(parents=True)
                    (artifacts / 'application.bin').write_bytes(b'catalog')
                    (artifacts / 'application.json').write_text('{"images": []}')
                    (artifacts / 'launch-environment.json').write_text('{}')
                    return ''
                if stage in (
                    'analyze-all-images',
                    'fresh-all-image-analysis-without-inputs',
                ):
                    self.assertIn('--fail-on-findings', argv)
                    self.assertIn('--require-complete', argv)
                    if stage.startswith('fresh-'):
                        self.assertFalse((commands.directory / 'original').exists())
                        self.assertFalse((commands.directory / 'artifact').exists())
                        self.assertFalse(
                            (commands.directory / 'all-images/images').exists()
                        )
                        self.assertFalse(
                            (commands.directory / 'all-images/application.bin').exists()
                        )
                        self.assertTrue(inputs.exists())
                        self.assertTrue((root / 'trace/original').exists())
                    return json.dumps(all_analysis)
                if stage in ('analyze-lds', 'fresh-lds-analysis-without-inputs'):
                    self.assertIn('--fail-on-findings', argv)
                    self.assertNotIn('--require-complete', argv)
                if stage == 'inspect-lds':
                    return json.dumps(
                        {
                            'entries': [
                                {
                                    'selected': True,
                                    'lds_access_sites': 72,
                                    'barrier_completion_sites': 5,
                                }
                            ]
                        }
                    )
                if stage == 'fresh-lds-analysis-without-inputs':
                    self.assertFalse((commands.directory / 'original').exists())
                    self.assertFalse((commands.directory / 'artifact').exists())
                    self.assertTrue(inputs.exists())
                    self.assertTrue((root / 'trace/original').exists())
                return json.dumps(analysis)

            def application(
                _args, commands, _tools, _inputs, name, _bindings, **_kwargs
            ):
                output = commands.directory / name
                output.mkdir()
                (output / 'logits.npy').write_bytes(b'unchanged logits')
                return output, {'passed': True}

            with mock.patch.object(
                prefill.Commands, 'run', lds_command
            ), mock.patch.object(
                prefill, 'application', side_effect=application
            ), mock.patch.object(
                prefill, 'validate_all_images'
            ) as validate_all, mock.patch(
                'builtins.print'
            ):
                result = prefill.run_case(args, tools, fixture, 'lds')
            self.assertTrue(result['passed'])
            self.assertEqual(result['all_images_analysis'], all_analysis)
            validate_all.assert_called_once_with({'images': []}, all_analysis)
            self.assertTrue((root / 'lds/all-images-instrumented/logits.npy').is_file())
            self.assertTrue((root / 'lds/all-images/application.bin').is_file())
            self.assertEqual(result['timeout'], 54)
            self.assertTrue((root / 'lds/original/model.hsaco').is_file())
            self.assertEqual((inputs / 'model.hsaco').read_bytes(), b'model')
            self.assertFalse((root / 'trace/artifact').exists())

    def test_standalone_lds_success_is_not_overridden_by_failed_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'commands.json').write_text('[]')
            (root / 'trace').mkdir()
            (root / 'trace/result.json').write_text(
                json.dumps({'case': 'trace', 'passed': False})
            )
            args = SimpleNamespace(case='lds', fixture=root, cpu_thread_budget=4)

            def execute(*_args):
                (root / 'lds').mkdir()
                (root / 'lds/result.json').write_text(
                    json.dumps({'case': 'lds', 'passed': True})
                )

            with mock.patch.object(
                prefill, 'validate_environment', return_value={}
            ), mock.patch.object(
                prefill, 'load_fixture', return_value={'baseline': {'passed': True}}
            ), mock.patch.object(
                prefill, 'run_case', side_effect=execute
            ):
                prefill.run(args)
            result = json.loads((root / 'result.json').read_text())
            self.assertFalse(result['passed'])
            self.assertTrue(result['cases']['lds']['passed'])

    def test_fixture_rejects_changed_build_tools_inputs_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'original/images').mkdir(parents=True)
            original = root / 'original/images/model'
            original.write_bytes(b'original image')
            args = SimpleNamespace(
                fixture=root,
                source=root / 'source',
                build=root / 'build',
                cpu_thread_budget=4,
            )
            identity = {
                'prepare': {'path': str(root / 'build/prepare'), 'sha256': 'a' * 64}
            }
            fixture = {
                'schema_version': 1,
                'source': str(args.source),
                'build': str(args.build),
                'suite_revision': prefill.SUITE_REVISION,
                'tools': identity,
                'cpu_thread_budget': 4,
                'model': 'images/model',
                'helper': 'images/model',
                'files': {'original/images/model': prefill.digest(original)},
            }

            def publish():
                (root / 'fixture.json').write_text(json.dumps(fixture))

            publish()
            with mock.patch.object(prefill, 'tool_identities', return_value=identity):
                self.assertEqual(prefill.load_fixture(args, {}), fixture)
                original.write_bytes(b'changed image')
                with self.assertRaisesRegex(ValueError, 'shared fixture input differs'):
                    prefill.load_fixture(args, {})
                original.write_bytes(b'original image')
                fixture['files'] = {'../outside': 'a' * 64}
                publish()
                with self.assertRaisesRegex(ValueError, 'fixture path escapes root'):
                    prefill.load_fixture(args, {})
            with mock.patch.object(prefill, 'tool_identities', return_value={}):
                with self.assertRaisesRegex(ValueError, 'fixture identity differs'):
                    prefill.load_fixture(args, {})

    def test_lds_analysis_requires_observations_for_every_selected_execution(self):
        prefill.validate_lds(lds_analysis())
        for field, value in (
            ('lds_analysis', 'not_performed'),
            ('execution_conditions', 15),
            ('executions', []),
            ('scope', 'execution_trace'),
        ):
            analysis = lds_analysis()
            analysis[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_lds(analysis)
        for field, value in (
            ('serial', 9),
            ('entry', prefill.SCATTER),
            ('entry_key', 2112),
            ('workgroup_count', 1),
            ('waves_per_workgroup', 1),
            ('valid_observations', 0),
            ('valid_observations', 4607),
            ('lost_occurrences', 575),
            ('address_group_exhaustion', 575),
            ('synchronization_loss', 1),
            ('integrity_events', 1),
            ('rejected_records', 1),
            ('counter_exhausted_waves', 1),
            ('issues', [{'reason': 'invalid_record_commit_identity_mask_or_sequence'}]),
        ):
            analysis = lds_analysis()
            analysis['executions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_lds(analysis)

    def test_findings_are_a_completed_analysis_and_input_errors_are_rejected(self):
        analysis = lds_analysis()
        analysis['verdict'] = 'findings'
        prefill.validate_lds(analysis, 1)
        with self.assertRaisesRegex(ValueError, 'status contradicts'):
            prefill.validate_lds(analysis, 0)
        for code in (0, 1, 2, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                commands = prefill.Commands(root, root, {}, 10)
                with mock.patch.object(
                    prefill.subprocess, 'run', return_value=mock.Mock(returncode=code)
                ):
                    if code in (0, 1):
                        commands.run('analysis', ['analyzer'], allowed_exits=(0, 1))
                    else:
                        with self.assertRaisesRegex(ValueError, f'exit {code}'):
                            commands.run('analysis', ['analyzer'], allowed_exits=(0, 1))
                self.assertEqual(commands.records[-1]['exit'], code)

    def test_findings_replay_requires_the_same_analyzer_exit_status(self):
        for code in (0, 1):
            commands = mock.Mock()
            commands.run.return_value = 'analysis'
            commands.records = [{'exit': code}]
            if code == 1:
                prefill.offline_replay(
                    commands,
                    'analyzer',
                    'capture',
                    (),
                    'analysis',
                    allowed_exits=(0, 1),
                    expected_exit=1,
                )
            else:
                with self.assertRaisesRegex(ValueError, 'status differs'):
                    prefill.offline_replay(
                        commands,
                        'analyzer',
                        'capture',
                        (),
                        'analysis',
                        allowed_exits=(0, 1),
                        expected_exit=1,
                    )
            self.assertEqual(commands.run.call_args.kwargs['allowed_exits'], (0, 1))

    def test_bounded_lds_loss_is_reported_without_claiming_complete_coverage(self):
        analysis = lds_analysis()
        execution = analysis['executions'][0]
        prefill.validate_lds(analysis)
        execution['coverage_complete'] = True
        with self.assertRaises(ValueError):
            prefill.validate_lds(analysis)
        execution['coverage_complete'] = False
        analysis['coverage_complete'] = True
        with self.assertRaises(ValueError):
            prefill.validate_lds(analysis)

    def test_complete_trace_and_rejection_of_scope_or_inventory_gaps(self):
        self.assertEqual(
            prefill.validate_trace(complete_analysis()),
            {prefill.SCATTER: 2048, prefill.ELEMENTWISE: 2112},
        )
        for field, value in (
            ('scope', 'native_lds'),
            ('lds_analysis', 'performed'),
            ('transport_complete', False),
            ('executions', []),
            ('completed_rows', 98),
        ):
            analysis = complete_analysis()
            analysis[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_trace(analysis)
        analysis = complete_analysis()
        analysis['entries'] = analysis['entries'][1:]
        with self.assertRaisesRegex(ValueError, 'inventory'):
            prefill.validate_trace(analysis)

    def test_each_execution_identity_geometry_and_boundary_is_checked(self):
        for field, value in (
            ('serial', 2),
            ('entry', prefill.ELEMENTWISE),
            ('entry_key', 2112),
            ('grid', [32, 1, 1]),
            ('workgroup', [64, 1, 1]),
            ('completed_rows', 15),
            ('entry_events', 0),
            ('exit_events', 2),
            ('integrity_events', 1),
            ('issues', ['invalid_entry_exit_row_0']),
        ):
            analysis = complete_analysis()
            analysis['executions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prefill.validate_trace(analysis)

    def test_clean_environment_preserves_sdk_paths_and_removes_old_tool_settings(self):
        removed = {
            'RJ_CONSAN_ARTIFACT_DIR': '/stale/artifact',
            'RJ_CONSAN_CAPTURE_PATH': '/shared/capture',
            'HSA_TOOLS_LIB': '/stale/plugin.so',
            'HSA_TOOLS_DISABLE_REGISTER': '1',
            'LD_PRELOAD': '/other/interposer.so',
            'LOADER_OPTIONS_APPEND': '-dump-code',
            'GPU_DUMP_CODE_OBJECT': '1',
        }
        sdk = {'PATH': '/own/venv/bin:/own/sdk/bin', 'LD_LIBRARY_PATH': '/own/sdk/lib'}
        inherited = (
            removed | sdk | {'HSA_HOTSWAP_DISABLE': '0', 'HSA_HOTSWAP_ENABLE': '1'}
        )
        cleaned = prefill.clean_environment(inherited)
        for key in removed:
            self.assertNotIn(key, cleaned)
            self.assertIn(key, inherited)
        for key, value in sdk.items():
            self.assertEqual(cleaned[key], value)
        self.assertEqual(cleaned['HSA_HOTSWAP_DISABLE'], '1')
        self.assertEqual(cleaned['HSA_HOTSWAP_ENABLE'], '0')

    def test_replay_hides_inputs_and_restores_them_on_success_or_failure(self):
        for outcome in ('same', 'changed', 'process-failure'):
            with self.subTest(
                outcome=outcome
            ), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = (root / 'original', root / 'artifact')
                for path in paths:
                    path.mkdir()
                    (path / 'witness').write_text(path.name)
                analyzer, capture = root / 'analyzer', root / 'capture.bin'
                commands = mock.Mock()

                def analyze(_stage, argv, *, allowed_exits):
                    self.assertEqual(allowed_exits, (0,))
                    self.assertEqual(argv, [analyzer, capture])
                    self.assertTrue(all(not path.exists() for path in paths))
                    if outcome == 'process-failure':
                        raise RuntimeError('analyzer failed')
                    return 'analysis' if outcome == 'same' else 'different analysis'

                commands.run.side_effect = analyze
                if outcome == 'same':
                    prefill.offline_replay(
                        commands, analyzer, capture, paths, 'analysis'
                    )
                else:
                    error = RuntimeError if outcome == 'process-failure' else ValueError
                    with self.assertRaises(error):
                        prefill.offline_replay(
                            commands, analyzer, capture, paths, 'analysis'
                        )
                commands.run.assert_called_once()
                for path in paths:
                    self.assertEqual((path / 'witness').read_text(), path.name)
                    self.assertFalse(path.with_name(path.name + '.hidden').exists())

    def test_replay_never_clobbers_an_existing_hidden_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'original'
            original.mkdir()
            hidden = root / 'original.hidden'
            hidden.mkdir()
            (hidden / 'unrelated').write_text('preserve')
            commands = mock.Mock()
            with self.assertRaisesRegex(ValueError, 'scratch paths already exist'):
                prefill.offline_replay(
                    commands,
                    root / 'analyzer',
                    root / 'capture.bin',
                    (original,),
                    'analysis',
                )
            commands.run.assert_not_called()
            self.assertTrue(original.is_dir())
            self.assertEqual((hidden / 'unrelated').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
