# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check owning no-probes-needed evidence and application coverage semantics."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
import consan_analyze_application as application
import consan_prepare_application as preparation_tool
from consan_application_contract import validate_preparation_report


def proof():
    return {
        'version': 1,
        'scope': 'lds',
        'instruction_count': 3,
        'code_bytes': 16,
        'code_sha256': 'cd' * 32,
    }


def preparation(dispositions):
    entries = []
    for index, disposition in enumerate(dispositions):
        no_probes = disposition == 'no_probes_needed'
        entries.append(
            {
                'entry': 'kernel_' + str(index),
                'entry_key': 2048 + index * 64,
                'disposition': disposition,
                'reason_code': (
                    'unsupported_entry' if disposition == 'blocked' else 'none'
                ),
                'reason': (
                    'entry ABI is unsupported' if disposition == 'blocked' else ''
                ),
                'text_offset': None,
                'stage': 'applicability' if no_probes else 'qualification',
                'no_probes_needed': proof() if no_probes else None,
            }
        )
    counts = {
        name: dispositions.count(name)
        for name in ('instrumented', 'no_probes_needed', 'blocked')
    }
    return {
        'schema_version': 2,
        'outcome': 'instrumented' if counts['instrumented'] else 'native_only',
        'original_sha256': 'ab' * 32,
        'original_bytes': 4096,
        'selection_profile': 'ef' * 32 if counts['instrumented'] else None,
        'target': 1250,
        'requested_scope': 'lds',
        'inventory_entries': len(entries),
        'not_requested': 0,
        'requested': len(entries),
        **counts,
        'skipped': counts['blocked'],
        'requested_coverage': (
            'complete'
            if counts['blocked'] == 0
            else 'not_performed' if counts['blocked'] == len(entries) else 'partial'
        ),
        'entries': entries,
    }


def captured_analysis(report):
    selected = [
        entry for entry in report['entries'] if entry['disposition'] == 'instrumented'
    ]
    return {
        'original_sha256': report['original_sha256'],
        'selection_profile': report['selection_profile'],
        'target': report['target'],
        'run_identity': '01' * 16,
        'scope': 'native_lds_single_range_mixed_scopes',
        'coverage_complete': True,
        'provided_entry_coverage_complete': report['blocked'] == 0,
        'cross_wave_pairs': 0,
        'collision_groups': 0,
        'entries': [
            {
                'entry': entry['entry'],
                'entry_key': entry['entry_key'],
                'disposition': {
                    'instrumented': 'selected',
                    'no_probes_needed': 'no_probes_needed',
                    'blocked': 'native_uncovered',
                }[entry['disposition']],
                'no_probes_needed': entry['no_probes_needed'],
            }
            for entry in report['entries']
        ],
        'executions': [
            {'entry_key': entry['entry_key'], 'valid_observations': 2}
            for entry in selected
        ],
    }


class ApplicationContract(unittest.TestCase):
    def analyze(self, report, analysis=None, *, capture=True):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            relative = 'capture.bin' if report['instrumented'] else None
            if capture and relative:
                (directory / relative).touch()
            index = {
                'schema_version': 2,
                'scope': 'provided_application_images',
                'images': [{'capture': relative, 'preparation': report}],
            }
            path = directory / 'application.json'
            path.write_text(json.dumps(index))
            response = subprocess.CompletedProcess(
                [], 0, stdout=json.dumps(analysis), stderr=''
            )
            with mock.patch.object(
                application.subprocess, 'run', return_value=response
            ) as run:
                result = application.analyze_application(path, Path('/unused-analyzer'))
                return result, run.call_count

    def test_all_no_probes_needs_no_capture_or_analyzer_and_does_not_claim_analysis(
        self,
    ):
        report = preparation(['no_probes_needed', 'no_probes_needed'])
        result, calls = self.analyze(report)
        self.assertEqual(calls, 0)
        self.assertTrue(result['coverage_complete'])
        self.assertTrue(result['provided_entry_coverage_complete'])
        self.assertEqual(result['lds_analysis'], 'not_performed')
        self.assertFalse(result['lds_observations_present'])
        self.assertIsNone(result['run_identity'])
        self.assertIsNone(result['images'][0]['analysis'])
        self.assertFalse(result['images'][0]['issues'])
        self.assertEqual(result['no_probes_needed_entries'], 2)
        self.assertEqual(result['blocked_entries'], 0)

    def test_native_only_with_blocked_entries_keeps_gap_without_fake_analysis(self):
        report = preparation(['no_probes_needed', 'blocked'])
        result, calls = self.analyze(report)
        self.assertEqual(calls, 0)
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['lds_analysis'], 'not_performed')
        self.assertEqual(result['images'][0]['issues'], ['blocked_entries'])
        self.assertEqual(result['no_probes_needed_entries'], 1)
        self.assertEqual(result['blocked_entries'], 1)

    def test_partial_request_does_not_attest_full_provided_image_inventory(self):
        for disposition in ('instrumented', 'no_probes_needed', 'blocked'):
            with self.subTest(disposition=disposition):
                report = preparation([disposition])
                report['inventory_entries'] = 3
                report['not_requested'] = 2
                self.assertIs(validate_preparation_report(report), report)
                with self.assertRaisesRegex(
                    ValueError, 'complete original entry inventory'
                ):
                    self.analyze(report, captured_analysis(report))

    def test_original_inventory_counts_are_explicit_and_consistent(self):
        for field, value in (
            ('inventory_entries', None),
            ('inventory_entries', True),
            ('inventory_entries', 0),
            ('inventory_entries', 2),
            ('not_requested', None),
            ('not_requested', True),
            ('not_requested', -1),
            ('not_requested', 1),
        ):
            with self.subTest(field=field, value=value):
                report = preparation(['no_probes_needed'])
                report[field] = value
                with self.assertRaises(ValueError):
                    validate_preparation_report(report)

    def test_coordinator_rejects_partial_native_inventory_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / 'original.hsaco'
            original.write_bytes(bytes(4096))
            report = preparation(['no_probes_needed'])
            report['original_sha256'] = preparation_tool.file_digest(original)
            report['inventory_entries'] = 3
            report['not_requested'] = 2

            def prepare(command, **unused):
                if '--producer-contract' in command:
                    return subprocess.CompletedProcess(command, 0, '01' * 16, '')
                artifact = Path(command[3])
                artifact.mkdir(parents=True)
                (artifact / 'preparation.json').write_text(json.dumps(report))
                return subprocess.CompletedProcess(command, 0, '', '')

            output = directory / 'application'
            options = SimpleNamespace(
                images=[original],
                images_dir=None,
                output=output,
                prepare=Path('/unused-prepare'),
                target='gfx1250',
                profile='lds',
                require_all=False,
                timeout=60,
            )
            with mock.patch.object(
                preparation_tool.subprocess, 'run', side_effect=prepare
            ):
                with self.assertRaisesRegex(
                    ValueError, 'complete original entry inventory'
                ):
                    preparation_tool.prepare_application(options)
            self.assertFalse((output / 'application.bin').exists())
            self.assertFalse((output / 'application.json').exists())

    def test_mixed_capture_excludes_only_proven_no_probe_entries_from_gap(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        result, calls = self.analyze(report, captured_analysis(report))
        self.assertEqual(calls, 1)
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['lds_analysis'], 'performed')
        self.assertTrue(result['lds_observations_present'])
        self.assertEqual(result['instrumented_entries'], 1)
        self.assertEqual(result['no_probes_needed_entries'], 1)
        blocked = preparation(['instrumented', 'no_probes_needed', 'blocked'])
        result, _ = self.analyze(blocked, captured_analysis(blocked))
        self.assertFalse(result['coverage_complete'])
        self.assertTrue(result['images'][0]['analysis']['coverage_complete'])
        self.assertEqual(result['images'][0]['issues'], ['blocked_entries'])

    def test_selected_entry_without_execution_still_has_a_coverage_gap(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        analysis = captured_analysis(report)
        analysis['executions'] = []
        analysis['provided_entry_coverage_complete'] = False
        result, _ = self.analyze(report, analysis)
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['lds_analysis'], 'not_performed')
        self.assertEqual(
            result['images'][0]['selected_entries_not_observed'], ['kernel_0']
        )

    def test_zero_observations_do_not_erase_real_execution_or_claim_memory(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        analysis = captured_analysis(report)
        analysis['executions'][0]['valid_observations'] = 0
        result, _ = self.analyze(report, analysis)
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['lds_analysis'], 'performed')
        self.assertFalse(result['lds_observations_present'])

    def test_missing_selected_capture_cannot_be_covered_by_native_proof(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        result, calls = self.analyze(report, capture=False)
        self.assertEqual(calls, 0)
        self.assertFalse(result['coverage_complete'])
        self.assertIn(
            'selected_image_has_no_closed_capture', result['images'][0]['issues']
        )

    def test_captured_proof_and_disposition_must_match_owning_preparation(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        for field, value in (('code_sha256', '99' * 32), ('code_bytes', 20)):
            with self.subTest(field=field):
                analysis = deepcopy(captured_analysis(report))
                analysis['entries'][1]['no_probes_needed'][field] = value
                with self.assertRaisesRegex(ValueError, 'selection differs'):
                    self.analyze(report, analysis)
        analysis = deepcopy(captured_analysis(report))
        analysis['entries'][1]['disposition'] = 'native_uncovered'
        with self.assertRaisesRegex(ValueError, 'selection differs'):
            self.analyze(report, analysis)

    def test_capture_provided_coverage_cannot_override_a_blocked_entry(self):
        report = preparation(['instrumented', 'blocked'])
        analysis = captured_analysis(report)
        analysis['provided_entry_coverage_complete'] = True
        with self.assertRaisesRegex(ValueError, 'provided-entry coverage contradicts'):
            self.analyze(report, analysis)

    def test_no_probe_entry_cannot_own_a_captured_execution(self):
        report = preparation(['instrumented', 'no_probes_needed'])
        analysis = captured_analysis(report)
        analysis['executions'][0]['entry_key'] = report['entries'][1]['entry_key']
        with self.assertRaisesRegex(ValueError, 'unselected entry'):
            self.analyze(report, analysis)

    def test_proof_validation_rejects_missing_scope_bounds_and_invalid_digest(self):
        for field, value in (
            ('version', 2),
            ('version', True),
            ('scope', 'execution_trace'),
            ('instruction_count', 0),
            ('instruction_count', True),
            ('instruction_count', 5),
            ('code_bytes', 0),
            ('code_bytes', 15),
            ('code_bytes', 4100),
            ('code_sha256', '0' * 64),
            ('code_sha256', 'G' * 64),
        ):
            with self.subTest(field=field, value=value):
                report = preparation(['no_probes_needed'])
                report['entries'][0]['no_probes_needed'][field] = value
                with self.assertRaises(ValueError):
                    validate_preparation_report(report)
        report = preparation(['no_probes_needed'])
        report['entries'][0]['no_probes_needed'] = None
        with self.assertRaises(ValueError):
            validate_preparation_report(report)

    def test_proof_is_only_allowed_for_lds_no_probes_needed_outcome(self):
        report = preparation(['no_probes_needed'])
        report['requested_scope'] = 'execution_trace'
        with self.assertRaisesRegex(ValueError, 'invalid scope'):
            validate_preparation_report(report)
        for disposition in ('instrumented', 'blocked'):
            with self.subTest(disposition=disposition):
                report = preparation([disposition])
                report['entries'][0]['no_probes_needed'] = proof()
                with self.assertRaisesRegex(ValueError, 'unexpected no-probes'):
                    validate_preparation_report(report)

    def test_outcome_counts_and_coverage_cannot_silently_relabel_a_skip(self):
        for key, value in (
            ('schema_version', 1),
            ('requested', True),
            ('instrumented', 1),
            ('no_probes_needed', 0),
            ('blocked', 1),
            ('skipped', 1),
            ('requested_coverage', 'not_performed'),
            ('outcome', 'instrumented'),
        ):
            with self.subTest(key=key):
                report = preparation(['no_probes_needed'])
                report[key] = value
                with self.assertRaises(ValueError):
                    validate_preparation_report(report)
        report = preparation(['blocked'])
        report['entries'][0]['reason_code'] = 'invalid_image'
        with self.assertRaisesRegex(ValueError, 'blocking reason'):
            validate_preparation_report(report)
        report = preparation(['no_probes_needed'])
        del report['selection_profile']
        with self.assertRaisesRegex(
            ValueError, 'missing preparation selection profile'
        ):
            validate_preparation_report(report)


if __name__ == '__main__':
    unittest.main()
