# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise saved-result reporting without invoking compilation or execution."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import consan_prefill_report as report

MODEL = '11' * 32
HELPER = '22' * 32
LDS_COUNTERS = (
    'valid_observations',
    'lost_occurrences',
    'address_group_exhaustion',
    'synchronization_loss',
    'rejected_records',
    'integrity_events',
    'counter_exhausted_waves',
    'cross_wave_pairs',
    'collision_groups',
    'ordering_unresolved_pairs',
)


def analysis(selected, dispatches, *, trace=False):
    entries = [
        {
            'entry': f'model_{index}',
            'entry_key': 2048 + 64 * index,
            'disposition': 'selected' if index < selected else 'native_uncovered',
        }
        for index in range(55)
    ]
    executions = []
    for serial in range(1, dispatches + 1):
        entry = entries[(serial - 1) % selected]
        execution = {
            'serial': serial,
            'entry': entry['entry'],
            'entry_key': entry['entry_key'],
            'integrity_events': 0,
            'issues': [],
        }
        if trace:
            execution.update(
                expected_rows=1, completed_rows=1, entry_events=1, exit_events=1
            )
        else:
            execution.update({counter: 0 for counter in LDS_COUNTERS})
            execution.update(
                valid_observations=9216,
                lost_occurrences=1152,
                address_group_exhaustion=1152,
                coverage_complete=False,
            )
        executions.append(execution)
    result = {
        'original_sha256': MODEL,
        'run_identity': 'trace' if trace else 'lds',
        'selection_profile': 'profile',
        'target': 'gfx942',
        'scope': 'execution_trace' if trace else 'native_lds_test',
        'lds_analysis': 'not_performed' if trace else 'performed',
        'entries': entries,
        'executions': executions,
    }
    if trace:
        result.update(
            transport_complete=True,
            expected_rows=dispatches,
            completed_rows=dispatches,
            entry_events=dispatches,
            exit_events=dispatches,
        )
    else:
        result.update(
            verdict='no_observed_findings',
            coverage_complete=False,
            cross_wave_pairs=0,
            collision_groups=0,
            ordering_unresolved_pairs=0,
        )
    return result


def preparation(entries, identity):
    prepared = [
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
        for entry in entries
    ]
    count = sum(entry['disposition'] == 'instrumented' for entry in prepared)
    return {
        'original_sha256': identity,
        'requested': len(prepared),
        'instrumented': count,
        'skipped': len(prepared) - count,
        'outcome': 'instrumented' if count else 'native_only',
        'selection_profile': 'profile',
        'target': 'gfx942',
        'entries': prepared,
    }


def saved_result():
    automatic = analysis(1, 4)
    model = {
        'original_sha256': MODEL,
        'preparation': preparation(automatic['entries'], MODEL),
        'analysis': automatic,
        'issues': [],
        'selected_entries_not_observed': [],
    }
    helper_entries = [
        {
            'entry': f'helper_{index}',
            'entry_key': index,
            'disposition': 'native_uncovered',
        }
        for index in range(13)
    ]
    helper = {
        'original_sha256': HELPER,
        'preparation': preparation(helper_entries, HELPER),
        'analysis': None,
        'issues': ['native_uncovered'],
        'selected_entries_not_observed': [],
    }
    return {
        'passed': True,
        'analysis': analysis(2, 9, trace=True),
        'lds_analysis': analysis(1, 6),
        'all_images_analysis': {
            'scope': 'provided_application_images',
            'whole_application_coverage': 'not_established',
            'lds_analysis': 'performed',
            'lds_observations_present': True,
            'coverage_complete': False,
            'cross_wave_pairs': 0,
            'collision_groups': 0,
            'images': [model, helper],
        },
    }


def static_summary(analysis_result):
    return {
        'original_sha256': analysis_result['original_sha256'],
        'entries': [
            {
                'entry': entry['entry'],
                'entry_key': entry['entry_key'],
                'selected': entry['disposition'] == 'selected',
                'lds_access_sites': 72 if entry['disposition'] == 'selected' else 0,
                'barrier_completion_sites': (
                    5 if entry['disposition'] == 'selected' else 0
                ),
                'trace_boundary_sites': 0,
            }
            for entry in analysis_result['entries']
        ],
    }


def classified_result():
    result = saved_result()
    for image in result['all_images_analysis']['images']:
        prepared = image['preparation']
        captured = image['analysis']
        proof_count = 48 if captured is not None else 12
        remaining = iter(range(proof_count))
        for row in prepared['entries']:
            row.update(stage='qualification', no_probes_needed=None, text_offset=None)
            if row['disposition'] == 'instrumented':
                continue
            if next(remaining, None) is None:
                row['disposition'] = 'blocked'
                continue
            proof = {
                'version': 1,
                'scope': 'lds',
                'instruction_count': 1,
                'code_bytes': 4,
                'code_sha256': '33' * 32,
            }
            row.update(
                disposition='no_probes_needed',
                stage='applicability',
                reason_code='none',
                reason='closed original code has no LDS accesses',
                no_probes_needed=proof,
            )
            if captured is not None:
                entry = next(
                    e for e in captured['entries'] if e['entry_key'] == row['entry_key']
                )
                entry.update(
                    disposition='no_probes_needed', no_probes_needed=deepcopy(proof)
                )
        counts = {
            name: sum(row['disposition'] == name for row in prepared['entries'])
            for name in ('instrumented', 'no_probes_needed', 'blocked')
        }
        prepared.update(
            schema_version=2,
            inventory_entries=len(prepared['entries']),
            not_requested=0,
            original_bytes=1 << 20,
            target=1250,
            requested_scope='lds',
            selection_profile='44' * 32 if counts['instrumented'] else None,
            requested_coverage='partial',
            skipped=counts['blocked'],
            **counts,
        )
        if captured is not None:
            captured.update(
                target=1250,
                selection_profile='44' * 32,
                provided_entry_coverage_complete=False,
            )
    return result


class PrefillReport(unittest.TestCase):
    def setUp(self):
        self.result = saved_result()

    def test_classified_preparation_separates_no_probes_needed_from_blocked(self):
        generated = report.build_report(classified_result())
        run = generated['runs']['all_images']
        self.assertEqual(run['instrumented_entries'], 1)
        self.assertEqual(run['no_probes_needed_entries'], 60)
        self.assertEqual(run['blocked_entries'], 7)
        self.assertEqual(run['retained_native_entries'], 67)
        self.assertEqual(run['preparation_excluded_entries'], 7)
        self.assertEqual(
            sum(group['count'] for group in run['first_preparation_exclusions']), 7
        )
        entry = next(
            e
            for e in run['images'][0]['entries']
            if e['disposition'] == 'no_probes_needed'
        )
        self.assertEqual(entry['static_probe_sites']['lds_access_sites'], 0)
        self.assertIsNone(entry['captured_dispatches'])
        self.assertIsNone(entry['first_preparation_exclusion'])
        self.assertIsNotNone(entry['no_probes_needed'])
        text = report.render_report(generated)
        self.assertIn('no probes needed: 60; blocked: 7', text)
        self.assertNotIn('Applicability unknown', text)

    def test_classified_no_probes_needed_requires_matching_owning_capture_proof(self):
        result = classified_result()
        entry = result['all_images_analysis']['images'][0]['analysis']['entries'][1]
        entry['no_probes_needed']['code_sha256'] = '55' * 32
        with self.assertRaisesRegex(ValueError, 'proof'):
            report.build_report(result)

    def test_all_no_probes_needed_is_complete_without_performed_analysis(self):
        result = classified_result()
        aggregate = result['all_images_analysis']
        for image in aggregate['images']:
            prepared = image['preparation']
            for entry in prepared['entries']:
                entry.update(
                    disposition='no_probes_needed',
                    stage='applicability',
                    reason_code='none',
                    reason='no LDS memory accesses',
                    no_probes_needed={
                        'version': 1,
                        'scope': 'lds',
                        'instruction_count': 1,
                        'code_bytes': 4,
                        'code_sha256': '33' * 32,
                    },
                )
            prepared.update(
                outcome='native_only',
                selection_profile=None,
                instrumented=0,
                no_probes_needed=len(prepared['entries']),
                blocked=0,
                skipped=0,
                requested_coverage='complete',
            )
            image.update(analysis=None, issues=[], selected_entries_not_observed=[])
        aggregate.update(
            lds_analysis='not_performed',
            lds_observations_present=False,
            coverage_complete=True,
            cross_wave_pairs=0,
            collision_groups=0,
        )
        run = report.build_report(result)['runs']['all_images']
        self.assertEqual(run['no_probes_needed_entries'], 68)
        self.assertEqual(run['blocked_entries'], 0)
        self.assertEqual(run['captured_dispatches'], 0)
        self.assertEqual(run['analysis']['lds_analysis'], 'not_performed')
        self.assertEqual(run['analysis']['verdict'], 'not_performed')
        self.assertTrue(run['provided_entry_coverage_complete'])

    def test_native_only_partial_request_does_not_cover_full_inventory(self):
        prepared = classified_result()['all_images_analysis']['images'][1][
            'preparation'
        ]
        prepared['entries'] = [prepared['entries'][0]]
        prepared.update(
            inventory_entries=13,
            requested=1,
            not_requested=12,
            instrumented=0,
            no_probes_needed=1,
            blocked=0,
            skipped=0,
            requested_coverage='complete',
        )
        image = report.image_report(None, prepared, prepared['original_sha256'])
        self.assertEqual(image['entry_inventory_count'], 13)
        self.assertEqual(image['entries_not_requested'], 12)
        self.assertEqual(image['retained_native_entries'], 13)
        self.assertFalse(image['entry_inventory_details_complete'])
        self.assertFalse(image['provided_entry_coverage_complete'])
        self.assertEqual(image['analysis']['lds_analysis'], 'not_performed')

    def test_capture_must_match_preparation_full_inventory_count(self):
        result = classified_result()
        prepared = result['all_images_analysis']['images'][0]['preparation']
        prepared['inventory_entries'] += 1
        prepared['not_requested'] += 1
        with self.assertRaisesRegex(ValueError, 'full inventory counts'):
            report.build_report(result)

    def test_schema_two_cannot_claim_no_probes_without_a_proof(self):
        result = classified_result()
        result['all_images_analysis']['images'][0]['preparation']['entries'][1][
            'no_probes_needed'
        ] = None
        with self.assertRaisesRegex(ValueError, 'proof'):
            report.build_report(result)

    def test_provided_coverage_cannot_hide_blocked_entries(self):
        result = classified_result()
        result['all_images_analysis']['images'][0]['analysis'][
            'provided_entry_coverage_complete'
        ] = True
        with self.assertRaisesRegex(ValueError, 'provided-entry coverage'):
            report.build_report(result)

    def test_blocked_entry_retains_diagnostic_stage(self):
        result = classified_result()
        result['all_images_analysis']['images'][0]['preparation']['entries'][-1][
            'stage'
        ] = 'lowering'
        run = report.build_report(result)['runs']['all_images']
        blocked = run['images'][0]['entries'][-1]
        self.assertEqual(blocked['disposition'], 'blocked')
        self.assertEqual(blocked['preparation_stage'], 'lowering')
        self.assertEqual(blocked['first_preparation_exclusion']['stage'], 'lowering')

    def test_legacy_report_does_not_invent_classification_evidence(self):
        run = report.build_report(saved_result())['runs']['all_images']
        self.assertIsNone(run['no_probes_needed_entries'])
        self.assertIsNone(run['blocked_entries'])

    def test_inventory_and_execution_are_distinct_per_run(self):
        generated = report.build_report(self.result)
        automatic = generated['runs']['all_images']
        self.assertEqual(automatic['preparation_requested_entries'], 68)
        self.assertEqual(automatic['instrumented_entries'], 1)
        self.assertEqual(automatic['retained_native_entries'], 67)
        self.assertEqual(automatic['observed_instrumented_entries'], 1)
        self.assertEqual(automatic['captured_dispatches'], 4)
        self.assertIsNone(automatic['native_dispatches'])
        native = automatic['images'][1]['entries'][0]
        self.assertIsNone(native['captured_dispatches'])
        self.assertEqual(generated['runs']['trace']['captured_dispatches'], 9)
        self.assertEqual(generated['runs']['pinned_lds']['captured_dispatches'], 6)
        self.assertEqual(len(automatic['images'][0]['entries']), 55)
        reasons = automatic['first_preparation_exclusions']
        self.assertEqual(reasons[0]['count'], 67)
        self.assertEqual(len(reasons[0]['entries']), 67)

    def test_partial_zero_findings_does_not_claim_race_freedom(self):
        automatic = report.build_report(self.result)['runs']['all_images']
        self.assertEqual(automatic['analysis']['verdict'], 'no_observed_findings')
        self.assertFalse(automatic['analysis']['coverage_complete'])
        self.assertFalse(automatic['provided_entry_coverage_complete'])
        text = report.render_report(report.build_report(self.result))
        self.assertIn('Native dispatches: unknown', text)
        self.assertIn('First preparation exclusions', text)
        self.assertNotIn('race_free', text)
        self.assertIn('model_54', text)

    def test_requested_controls_are_distinct_from_unrequested_native_entries(self):
        self.result['preparation_reports'] = {
            name: preparation(
                [
                    entry
                    for entry in self.result[source]['entries']
                    if entry['disposition'] == 'selected'
                ],
                MODEL,
            )
            for name, source in (('trace', 'analysis'), ('pinned_lds', 'lds_analysis'))
        }
        runs = report.build_report(self.result)['runs']
        for name, requested, unrequested in (('trace', 2, 53), ('pinned_lds', 1, 54)):
            with self.subTest(run=name):
                self.assertEqual(runs[name]['preparation_requested_entries'], requested)
                self.assertEqual(runs[name]['preparation_excluded_entries'], 0)
                self.assertEqual(runs[name]['entries_not_requested'], unrequested)
                self.assertEqual(runs[name]['first_preparation_exclusions'], [])
                self.assertFalse(
                    runs[name]['images'][0]['entries'][-1]['preparation_requested']
                )
        self.assertEqual(runs['all_images']['preparation_excluded_entries'], 67)
        self.assertEqual(runs['all_images']['entries_not_requested'], 0)

    def test_absent_control_preparation_counts_are_unknown(self):
        runs = report.build_report(self.result)['runs']
        for name in ('trace', 'pinned_lds'):
            self.assertIsNone(runs[name]['preparation_requested_entries'])
            self.assertIsNone(runs[name]['preparation_excluded_entries'])
            self.assertIsNone(runs[name]['entries_not_requested'])
            self.assertEqual(runs[name]['first_preparation_exclusions'], [])

    def test_trace_has_no_lds_counters_or_verdict(self):
        trace = report.build_report(self.result)['runs']['trace']['analysis']
        self.assertEqual(trace['verdict'], 'not_performed')
        for counter in LDS_COUNTERS:
            self.assertIsNone(trace[counter])

    def test_overlapping_loss_counters_are_not_added_together(self):
        analysis_result = report.build_report(self.result)['runs']['all_images'][
            'analysis'
        ]
        self.assertEqual(analysis_result['valid_observations'], 36864)
        self.assertEqual(analysis_result['lost_occurrences'], 4608)
        self.assertEqual(analysis_result['address_group_exhaustion'], 4608)
        self.assertNotIn('total_loss', analysis_result)

    def test_missing_static_metadata_is_unknown(self):
        automatic = report.build_report(self.result)['runs']['all_images']
        entry = automatic['images'][0]['entries'][0]
        self.assertEqual(set(entry['static_probe_sites'].values()), {None})
        self.assertEqual(set(automatic['static_probe_sites'].values()), {None})

    def test_static_sites_are_per_entry_not_dispatch_totals(self):
        automatic = self.result['all_images_analysis']['images'][0]['analysis']
        self.result['capture_summaries'] = {'all_images': [static_summary(automatic)]}
        run = report.build_report(self.result)['runs']['all_images']
        entry = run['images'][0]['entries'][0]
        self.assertEqual(entry['static_probe_sites']['lds_access_sites'], 72)
        self.assertEqual(entry['static_probe_sites']['barrier_completion_sites'], 5)
        self.assertEqual(entry['captured_dispatches'], 4)
        self.assertEqual(
            run['static_probe_sites'],
            {
                'lds_access_sites': 72,
                'barrier_completion_sites': 5,
                'trace_boundary_sites': 0,
            },
        )

    def test_source_test_and_replay_evidence_are_visible(self):
        self.result['fresh_replay_identical'] = True
        text = report.render_report(report.build_report(self.result))
        self.assertIn('Source test passed: true; fresh replay identical: true', text)

    def test_confirmed_findings_remain_visible(self):
        automatic = self.result['all_images_analysis']
        nested = automatic['images'][0]['analysis']
        nested['executions'][0]['cross_wave_pairs'] = 2
        nested['cross_wave_pairs'] = automatic['cross_wave_pairs'] = 2
        nested['verdict'] = 'findings'
        generated = report.build_report(self.result)
        self.assertEqual(
            generated['runs']['all_images']['analysis']['verdict'], 'findings'
        )

    def test_masked_memory_execution_is_not_invalid(self):
        automatic = self.result['all_images_analysis']
        nested = automatic['images'][0]['analysis']
        for execution in nested['executions']:
            execution.update({counter: 0 for counter in LDS_COUNTERS})
            execution['coverage_complete'] = True
        nested['coverage_complete'] = True
        automatic['lds_observations_present'] = False
        generated = report.build_report(self.result)['runs']['all_images']['analysis']
        self.assertEqual(generated['lds_analysis'], 'performed')
        self.assertFalse(generated['lds_observations_present'])
        self.assertEqual(generated['verdict'], 'no_observed_findings')

    def test_missing_selected_capture_is_reported(self):
        automatic = self.result['all_images_analysis']
        model = automatic['images'][0]
        model['analysis'] = None
        model['issues'] = ['selected_image_has_no_closed_capture']
        model['selected_entries_not_observed'] = ['model_0']
        automatic['lds_analysis'] = 'not_performed'
        automatic['lds_observations_present'] = False
        generated = report.build_report(self.result)['runs']['all_images']
        self.assertEqual(generated['captured_dispatches'], 0)
        self.assertEqual(len(generated['selected_entries_not_observed']), 1)
        self.assertEqual(generated['analysis']['verdict'], 'not_performed')

    def test_inconsistent_preparation_counts_are_rejected(self):
        self.result['all_images_analysis']['images'][0]['preparation'][
            'instrumented'
        ] = 2
        with self.assertRaisesRegex(ValueError, 'counts'):
            report.build_report(self.result)

    def test_identity_and_disposition_mismatches_are_rejected(self):
        mutations = (
            ('original_sha256', HELPER),
            ('entry', 'different'),
            ('disposition', 'native_uncovered'),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                result = deepcopy(self.result)
                nested = result['all_images_analysis']['images'][0]['analysis']
                target = nested if field == 'original_sha256' else nested['entries'][0]
                target[field] = value
                with self.assertRaises(ValueError):
                    report.build_report(result)

    def test_static_capture_identity_and_selection_are_rejected(self):
        for field, value in (
            ('entry', 'wrong'),
            ('selected', False),
            ('entry_key', 999),
        ):
            with self.subTest(field=field):
                result = deepcopy(self.result)
                summary = static_summary(
                    result['all_images_analysis']['images'][0]['analysis']
                )
                summary['entries'][0][field] = value
                result['capture_summaries'] = {'all_images': [summary]}
                with self.assertRaises(ValueError):
                    report.build_report(result)

    def test_unobserved_selected_entries_are_not_assumed_executed(self):
        automatic = self.result['all_images_analysis']
        model = automatic['images'][0]
        model['analysis']['executions'] = []
        model['selected_entries_not_observed'] = ['model_0']
        model['issues'] = ['selected_entries_not_executed']
        automatic['lds_analysis'] = 'not_performed'
        automatic['lds_observations_present'] = False
        generated = report.build_report(self.result)['runs']['all_images']
        self.assertEqual(generated['instrumented_entries'], 1)
        self.assertEqual(generated['observed_instrumented_entries'], 0)
        self.assertEqual(generated['captured_dispatches'], 0)
        self.assertEqual(
            generated['selected_entries_not_observed'][0]['entry'], 'model_0'
        )

    def test_exclusion_groups_keep_distinct_first_reasons(self):
        entries = self.result['all_images_analysis']['images'][0]['preparation'][
            'entries'
        ]
        entries[1]['reason'] = 'second first reason'
        generated = report.build_report(self.result)['runs']['all_images']
        self.assertEqual(
            sorted(row['count'] for row in generated['first_preparation_exclusions']),
            [1, 66],
        )

    def test_negative_and_boolean_counters_are_rejected(self):
        for value in (-1, True):
            with self.subTest(value=value):
                result = deepcopy(self.result)
                result['all_images_analysis']['images'][0]['analysis']['executions'][0][
                    'lost_occurrences'
                ] = value
                with self.assertRaisesRegex(ValueError, 'count'):
                    report.build_report(result)

    def test_execution_for_native_entry_is_rejected(self):
        execution = self.result['all_images_analysis']['images'][0]['analysis'][
            'executions'
        ][0]
        execution.update(entry_key=2112, entry='model_1')
        with self.assertRaises(ValueError):
            report.build_report(self.result)

    def test_aggregate_counters_cannot_hide_findings(self):
        nested = self.result['all_images_analysis']['images'][0]['analysis']
        nested['executions'][0]['collision_groups'] = 1
        with self.assertRaisesRegex(ValueError, 'counter|counts'):
            report.build_report(self.result)

    def test_duplicate_execution_serials_are_rejected(self):
        nested = self.result['all_images_analysis']['images'][0]['analysis']
        nested['executions'][1]['serial'] = 1
        with self.assertRaises(ValueError):
            report.build_report(self.result)

    def test_cli_replays_only_saved_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'result.json'
            source.write_text(json.dumps(self.result))
            completed = subprocess.run(
                [sys.executable, str(Path(report.__file__)), '--result', str(source)],
                capture_output=True,
                text=True,
                check=False,
                env={'PATH': ''},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated = json.loads((directory / 'coverage-report.json').read_text())
            self.assertEqual(generated, report.build_report(self.result))
            self.assertEqual(
                (directory / 'coverage-report.txt').read_text(),
                report.render_report(generated),
            )
            self.assertIn('ConSan prefill report', completed.stdout)

    def test_output_cannot_overwrite_result_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'coverage-report.json'
            source.write_text(json.dumps(self.result))
            with self.assertRaises(ValueError):
                report.write_report(source)
            self.assertEqual(json.loads(source.read_text()), self.result)


if __name__ == '__main__':
    unittest.main()
