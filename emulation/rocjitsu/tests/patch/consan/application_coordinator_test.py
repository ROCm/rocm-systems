# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise automatic whole-application preparation and owning index checks."""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest.mock


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--analyze-coordinator', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix='all-images-', dir=args.artifacts)
    ).resolve()
    print(f'evidence: {directory}', flush=True)
    inputs = directory / 'inputs'
    inputs.mkdir()
    for name in ('mixed', 'unsupported', 'unsupported-only'):
        shutil.copyfile(
            args.fixtures / f'multi-entry-{name}.hsaco', inputs / f'{name}.hsaco'
        )
    # Deduplicate byte-identical inputs while preserving every distinct image.
    shutil.copyfile(inputs / 'mixed.hsaco', inputs / 'duplicate.hsaco')
    output = directory / 'application'
    command = [
        str(args.coordinator),
        '--prepare',
        str(args.prepare),
        '--target',
        'gfx1250',
        '--images-dir',
        str(inputs),
        '--output',
        str(output),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    (directory / 'prepare.json').write_text(
        json.dumps(
            {
                'command': command,
                'exit': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            },
            indent=2,
        )
    )
    assert result.returncode == 0, result.stderr
    index_path = output / 'application.json'
    index = json.loads(index_path.read_text())
    assert len(index['images']) == 3, index
    assert index['schema_version'] == 2, index
    outcomes = sorted(
        (
            row['preparation']['requested'],
            row['preparation']['instrumented'],
            row['preparation']['skipped'],
            row['preparation']['no_probes_needed'],
            row['preparation']['blocked'],
        )
        for row in index['images']
    )
    assert outcomes == [(1, 0, 1, 0, 1), (3, 2, 1, 0, 1), (3, 3, 0, 0, 0)], outcomes
    for row in index['images']:
        report = row['preparation']
        assert report['inventory_entries'] == report['requested'], report
        assert report['not_requested'] == 0, report
        names = [entry['entry'] for entry in report['entries']]
        assert names == (
            ['mk_c'] if report['requested'] == 1 else ['mk_a', 'mk_b', 'mk_c']
        ), report
        artifact = output / 'images' / report['original_sha256']
        assert (artifact / 'original.hsaco').is_file()
        assert (artifact / 'ready').exists() == (report['instrumented'] != 0)
    catalog = output / 'application.bin'
    check = subprocess.run(
        [str(args.prepare), '--validate-application-catalog', str(catalog)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert check.returncode == 0 and check.stdout.strip() == 'images=3', check
    strict = subprocess.run(
        [
            str(args.coordinator),
            '--prepare',
            str(args.prepare),
            '--target',
            'gfx1250',
            '--images-dir',
            str(inputs),
            '--output',
            str(directory / 'strict'),
            '--require-all',
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert (
        strict.returncode == 2 and not (directory / 'strict/application.bin').exists()
    ), strict.stderr
    assert not (directory / 'strict/application.json').exists()
    # A whole image with no LDS operations satisfies --require-all without
    # manufacturing prepared code, a capture, or a performed analysis.
    scalar = directory / 'native-scalar-only.hsaco'
    shutil.copyfile(args.fixtures / 'multi-entry-native-scalar-only.hsaco', scalar)
    no_probe_output = directory / 'no-probes-needed'
    no_probe_command = [
        str(args.coordinator),
        '--prepare',
        str(args.prepare),
        '--target',
        'gfx1250',
        '--output',
        str(no_probe_output),
        '--require-all',
        str(scalar),
    ]
    no_probe = subprocess.run(
        no_probe_command, capture_output=True, text=True, timeout=120, check=False
    )
    (directory / 'no-probes-needed-command.json').write_text(
        json.dumps(
            {
                'argv': no_probe_command,
                'exit': no_probe.returncode,
                'stdout': no_probe.stdout,
                'stderr': no_probe.stderr,
            },
            indent=2,
        )
    )
    assert no_probe.returncode == 0, no_probe.stderr
    no_probe_path = no_probe_output / 'application.json'
    no_probe_index = json.loads(no_probe_path.read_text())
    assert len(no_probe_index['images']) == 1, no_probe_index
    no_probe_image = no_probe_index['images'][0]
    no_probe_report = no_probe_image['preparation']
    assert no_probe_image['capture'] is None, no_probe_image
    assert no_probe_report['outcome'] == 'native_only', no_probe_report
    assert no_probe_report['requested_coverage'] == 'complete', no_probe_report
    assert (
        no_probe_report['instrumented'] == no_probe_report['blocked'] == 0
        and no_probe_report['requested'] == no_probe_report['no_probes_needed'] == 1
        and no_probe_report['inventory_entries'] == 1
        and no_probe_report['not_requested'] == 0
    ), no_probe_report
    assert no_probe_report['entries'][0]['entry'] == 'mk_c', no_probe_report
    assert no_probe_report['entries'][0]['no_probes_needed']['instruction_count'] > 0
    owned = no_probe_output / 'images' / no_probe_report['original_sha256']
    assert (owned / 'original.hsaco').read_bytes() == scalar.read_bytes()
    for filename in ('ready', 'facts.bin', 'prepared.hsaco'):
        assert not (owned / filename).exists(), filename
    scalar.rename(directory / 'native-scalar-only.hsaco.hidden')
    (no_probe_output / 'images').rename(no_probe_output / 'images.hidden')
    no_probe_analysis = subprocess.run(
        [
            str(args.analyze_coordinator),
            str(no_probe_path),
            '--analyzer',
            str(directory / 'does-not-exist'),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert no_probe_analysis.returncode == 0, no_probe_analysis.stderr
    native_summary = json.loads(no_probe_analysis.stdout)
    assert native_summary['coverage_complete'], native_summary
    assert native_summary['provided_entry_coverage_complete'], native_summary
    assert native_summary['lds_analysis'] == 'not_performed', native_summary
    assert not native_summary['lds_observations_present'], native_summary
    assert native_summary['images'][0]['analysis'] is None, native_summary
    assert not native_summary['images'][0]['issues'], native_summary
    mixed_proof_output = directory / 'mixed-proofs'
    mixed_proof_command = [
        str(args.coordinator),
        '--prepare',
        str(args.prepare),
        '--target',
        'gfx1250',
        '--output',
        str(mixed_proof_output),
        '--require-all',
        str(args.fixtures / 'multi-entry-native-scalar.hsaco'),
    ]
    mixed_proof = subprocess.run(
        mixed_proof_command, capture_output=True, text=True, timeout=120, check=False
    )
    assert mixed_proof.returncode == 0, mixed_proof.stderr
    mixed_proof_index = json.loads(
        (mixed_proof_output / 'application.json').read_text()
    )
    mixed_report = mixed_proof_index['images'][0]['preparation']
    assert (
        mixed_report['requested'] == 3
        and mixed_report['inventory_entries'] == 3
        and mixed_report['not_requested'] == 0
        and mixed_report['instrumented'] == 2
        and mixed_report['no_probes_needed'] == 1
        and mixed_report['blocked'] == mixed_report['skipped'] == 0
        and mixed_report['requested_coverage'] == 'complete'
    ), mixed_report
    assert mixed_proof_index['images'][0]['capture'] is not None
    # Index replay consumes owning reports, so absent captures remain explicit
    # coverage gaps even after all original/prepared input paths disappear.
    inputs.rename(directory / 'inputs.hidden')
    (output / 'images').rename(output / 'images.hidden')
    analyzer = load_module('consan_analyze_application', args.analyze_coordinator)
    missing = analyzer.analyze_application(index_path, Path('/unused-analyzer'))
    assert missing['scope'] == 'provided_application_images', missing
    assert missing['whole_application_coverage'] == 'not_established', missing
    assert not missing['lds_observations_present'], missing
    assert (
        missing['lds_analysis'] == 'not_performed' and not missing['coverage_complete']
    ), missing
    assert (
        sum('blocked_entries' in row['issues'] for row in missing['images']) == 1
    ), missing
    assert (
        sum(
            'selected_image_has_no_closed_capture' in row['issues']
            for row in missing['images']
        )
        == 2
    ), missing

    # The following tests isolate index aggregation from the native capture
    # decoder. The runtime/simulator tests exercise actual binary captures.
    complete = next(
        row for row in index['images'] if row['preparation']['skipped'] == 0
    )
    selected = complete['preparation']
    capture = output / complete['capture']
    capture.touch()
    fixture = {
        'original_sha256': selected['original_sha256'],
        'selection_profile': selected['selection_profile'],
        'target': selected['target'],
        'run_identity': '01' * 16,
        'scope': 'native_lds_single_range_mixed_scopes',
        'coverage_complete': True,
        'provided_entry_coverage_complete': True,
        'cross_wave_pairs': 0,
        'collision_groups': 0,
        'entries': [
            {
                'entry_key': entry['entry_key'],
                'entry': entry['entry'],
                'disposition': 'selected',
            }
            for entry in selected['entries']
        ],
        'executions': [
            {'entry_key': entry['entry_key'], 'valid_observations': 2}
            for entry in selected['entries']
        ],
    }
    single = {'schema_version': 2, 'scope': index['scope'], 'images': [complete]}
    single_path = output / 'single.json'
    single_path.write_text(json.dumps(single))

    def analyze(native):
        response = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(native), stderr=''
        )
        with unittest.mock.patch.object(
            analyzer.subprocess, 'run', return_value=response
        ):
            return analyzer.analyze_application(single_path, Path('/native-analyzer'))

    def analyze_main(native, flags):
        response = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps(native), stderr=''
        )
        arguments = [
            str(args.analyze_coordinator),
            str(single_path),
            '--analyzer',
            '/native-analyzer',
            *flags,
        ]
        emitted = io.StringIO()
        with (
            unittest.mock.patch.object(
                analyzer.subprocess, 'run', return_value=response
            ),
            unittest.mock.patch.object(sys, 'argv', arguments),
            redirect_stdout(emitted),
        ):
            status = analyzer.main()
        return status, json.loads(emitted.getvalue())

    combined = analyze(fixture)
    assert (
        combined['lds_analysis'] == 'performed' and combined['coverage_complete']
    ), combined
    assert combined['lds_observations_present'], combined
    empty_memory = deepcopy(fixture)
    for execution in empty_memory['executions']:
        execution['valid_observations'] = 0
    combined = analyze(empty_memory)
    assert combined['lds_analysis'] == 'performed', combined
    assert not combined['lds_observations_present'], combined
    assert combined['coverage_complete'], combined
    assert combined['images'][0]['analysis']['coverage_complete'], combined
    assert not combined['images'][0]['issues'], combined
    absent_entry = deepcopy(fixture)
    absent_entry['executions'].pop()
    absent_entry['provided_entry_coverage_complete'] = False
    combined = analyze(absent_entry)
    assert not combined['coverage_complete'], combined
    assert combined['images'][0]['selected_entries_not_observed'] == [
        selected['entries'][-1]['entry']
    ], combined
    policies = (
        (),
        ('--fail-on-findings',),
        ('--require-complete',),
        ('--fail-on-findings', '--require-complete'),
    )
    # A successful report is independent of the caller's optional CI policy.
    # Exercise both conflict counters and coverage, including combined failures.
    for cross_wave, collisions, coverage, expected_statuses in (
        (0, 0, False, (0, 0, 1, 1)),
        (0, 0, True, (0, 0, 0, 0)),
        (1, 0, True, (0, 1, 0, 1)),
        (0, 1, True, (0, 1, 0, 1)),
        (1, 0, False, (0, 1, 1, 1)),
        (0, 1, False, (0, 1, 1, 1)),
    ):
        native = deepcopy(fixture)
        native['cross_wave_pairs'] = cross_wave
        native['collision_groups'] = collisions
        native['coverage_complete'] = coverage
        native['provided_entry_coverage_complete'] = coverage
        expected_report = analyze(native)
        for flags, expected_status in zip(policies, expected_statuses):
            status, report = analyze_main(native, flags)
            assert status == expected_status, (native, flags, status)
            assert report == expected_report, (flags, report, expected_report)
    # Invalid input and a failed child tool remain errors under every policy.
    invalid_path = output / 'invalid-index.json'
    invalid_path.write_text('{}')
    for invalid_input, native_tool in (
        (output / 'missing-index.json', '/unused-analyzer'),
        (invalid_path, '/unused-analyzer'),
        (single_path, str(directory / 'missing-native-analyzer')),
    ):
        for flags in policies:
            failed = subprocess.run(
                [
                    sys.executable,
                    '-B',
                    str(args.analyze_coordinator),
                    str(invalid_input),
                    '--analyzer',
                    native_tool,
                    *flags,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert failed.returncode == 2 and not failed.stdout, failed
            assert failed.stderr.startswith('ConSan application analysis:'), failed
    invalid = deepcopy(fixture)
    invalid['original_sha256'] = 'ff' * 32
    try:
        analyze(invalid)
    except ValueError as error:
        assert 'identities differ' in str(error)
    else:
        raise AssertionError('capture with a different original identity was accepted')
    single['images'].append(deepcopy(complete))
    single_path.write_text(json.dumps(single))
    try:
        analyze(fixture)
    except ValueError as error:
        assert 'duplicate original' in str(error)
    else:
        raise AssertionError('duplicate original identity was accepted')


if __name__ == '__main__':
    main()
