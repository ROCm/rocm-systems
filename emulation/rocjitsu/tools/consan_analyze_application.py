#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Analyze an owning application index without original/prepared code images."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from consan_application_contract import (
    APPLICATION_SCHEMA_VERSION,
    unsigned,
    validate_preparation_report,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze_application(index_path, analyzer):
    index = json.loads(index_path.read_text())
    require(
        isinstance(index, dict)
        and type(index.get('schema_version')) is int
        and index['schema_version'] == APPLICATION_SCHEMA_VERSION
        and index.get('scope') == 'provided_application_images',
        'unsupported application index schema; regenerate schema-2 evidence',
    )
    require(
        isinstance(index['images'], list) and 0 < len(index['images']) <= 1024,
        'empty or oversized application image list',
    )
    output = {
        'scope': 'provided_application_images',
        'whole_application_coverage': 'not_established',
        'lds_analysis': 'not_performed',
        'lds_observations_present': False,
        'coverage_complete': True,
        'cross_wave_pairs': 0,
        'collision_groups': 0,
        'inventory_entries': 0,
        'requested_entries': 0,
        'instrumented_entries': 0,
        'no_probes_needed_entries': 0,
        'blocked_entries': 0,
        'not_requested_entries': 0,
        'images': [],
    }
    identities = set()
    captures = set()
    run_identity = None
    for image in index['images']:
        report = validate_preparation_report(image['preparation'])
        require(
            report['not_requested'] == 0,
            'application preparation did not request the complete original entry inventory',
        )
        identity = report['original_sha256']
        require(
            identity not in identities,
            'invalid or duplicate original image identity',
        )
        identities.add(identity)
        entries = report['entries']
        selected = {
            entry['entry_key']: entry['entry']
            for entry in entries
            if entry['disposition'] == 'instrumented'
        }
        for source, destination in (
            ('inventory_entries', 'inventory_entries'),
            ('requested', 'requested_entries'),
            ('instrumented', 'instrumented_entries'),
            ('no_probes_needed', 'no_probes_needed_entries'),
            ('blocked', 'blocked_entries'),
            ('not_requested', 'not_requested_entries'),
        ):
            output[destination] += report[source]
        row = {
            'original_sha256': identity,
            'preparation': report,
            'analysis': None,
            'issues': [],
            'coverage_complete': False,
            'selected_entries_not_observed': sorted(selected.values()),
        }
        output['images'].append(row)
        if not selected:
            require(
                report['outcome'] == 'native_only' and image['capture'] is None,
                'native-only image claims selected coverage',
            )
            if report['blocked']:
                row['issues'].append('blocked_entries')
            row['coverage_complete'] = report['blocked'] == 0
            output['coverage_complete'] &= row['coverage_complete']
            continue
        require(
            report['outcome'] == 'instrumented' and isinstance(image['capture'], str),
            'selected image has no capture declaration',
        )
        relative = Path(image['capture'])
        require(
            not relative.is_absolute()
            and '..' not in relative.parts
            and relative.as_posix() == image['capture']
            and relative not in captures,
            'invalid or duplicate relative capture path',
        )
        captures.add(relative)
        capture = index_path.parent / relative
        if not capture.exists():
            row['issues'].append('selected_image_has_no_closed_capture')
            output['coverage_complete'] = False
            continue
        result = subprocess.run(
            [str(analyzer), str(capture)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        require(
            result.returncode in (0, 1),
            f'invalid capture {relative}: {result.stderr.strip()}',
        )
        analysis = json.loads(result.stdout)
        require(
            analysis['original_sha256'] == identity
            and analysis['selection_profile'] == report['selection_profile']
            and analysis['target'] == report['target'],
            'capture and preparation identities differ',
        )
        if run_identity is None:
            run_identity = analysis['run_identity']
        require(
            analysis['run_identity'] == run_identity,
            'captures belong to different application runs',
        )
        captured_entries = {entry['entry_key']: entry for entry in analysis['entries']}
        require(
            len(captured_entries) == report['inventory_entries']
            and len(captured_entries) == len(analysis['entries']),
            'capture entry inventory differs',
        )
        dispositions = {
            'instrumented': 'selected',
            'no_probes_needed': 'no_probes_needed',
            'blocked': 'native_uncovered',
        }
        for entry in entries:
            captured = captured_entries.get(entry['entry_key'])
            require(
                captured is not None
                and captured['entry'] == entry['entry']
                and captured['disposition'] == dispositions[entry['disposition']]
                and captured.get('no_probes_needed') == entry['no_probes_needed'],
                'capture entry selection differs from preparation report',
            )
        executions = analysis['executions']
        observed = {execution['entry_key'] for execution in executions}
        require(observed.issubset(selected), 'execution references an unselected entry')
        row['selected_entries_not_observed'] = sorted(
            selected[key] for key in selected.keys() - observed
        )
        row['analysis'] = analysis
        trace = analysis['scope'] == 'execution_trace'
        complete = (
            analysis['transport_complete'] if trace else analysis['coverage_complete']
        )
        require(type(complete) is bool, 'invalid capture coverage result')
        if not trace:
            observations = sum(
                unsigned(execution['valid_observations'], 'valid observation count')
                for execution in executions
            )
            if executions:
                output['lds_analysis'] = 'performed'
            output['lds_observations_present'] |= observations > 0
            output['cross_wave_pairs'] += unsigned(
                analysis['cross_wave_pairs'], 'cross-wave finding count'
            )
            output['collision_groups'] += unsigned(
                analysis['collision_groups'], 'collision finding count'
            )
        if row['selected_entries_not_observed']:
            row['issues'].append('selected_entries_not_executed')
        if report['blocked']:
            row['issues'].append('blocked_entries')
        row['coverage_complete'] = complete and not row['issues']
        if not trace:
            require(
                type(analysis.get('provided_entry_coverage_complete')) is bool
                and analysis['provided_entry_coverage_complete']
                == row['coverage_complete'],
                'capture provided-entry coverage contradicts owning entries',
            )
        output['coverage_complete'] &= row['coverage_complete']
    output['provided_entry_coverage_complete'] = output['coverage_complete']
    output['run_identity'] = run_identity
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=lambda value: Path(value).resolve())
    parser.add_argument(
        '--analyzer', type=lambda value: Path(value).resolve(), required=True
    )
    parser.add_argument(
        '--fail-on-findings',
        action='store_true',
        help='exit 1 when the report contains race or collision findings',
    )
    parser.add_argument(
        '--require-complete',
        action='store_true',
        help='exit 1 when provided-entry coverage is incomplete',
    )
    args = parser.parse_args()
    result = analyze_application(args.index, args.analyzer)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(
        (args.require_complete and not result['coverage_complete'])
        or (
            args.fail_on_findings
            and bool(result['cross_wave_pairs'] or result['collision_groups'])
        )
    )


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f'ConSan application analysis: {error}', file=sys.stderr)
        sys.exit(2)
