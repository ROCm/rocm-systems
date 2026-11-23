# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check wrapper evidence and replay cleanup without IREE or a GPU."""

from pathlib import Path
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


class PrefillEvidence(unittest.TestCase):
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

                def analyze(_stage, argv):
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
