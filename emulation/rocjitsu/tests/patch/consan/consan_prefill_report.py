# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report ConSan prefill preparation, execution, and analysis from saved JSON."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from consan_application_contract import (
    validate_no_probes_needed_proof,
    validate_preparation_report,
)

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
STATIC_COUNTERS = (
    'lds_access_sites',
    'barrier_completion_sites',
    'trace_boundary_sites',
)
RUN_LABELS = {
    'all_images': 'All-image automatic selection',
    'pinned_lds': 'Pinned LDS control',
    'trace': 'Execution trace controls',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, label):
    require(type(value) is int and value >= 0, f'invalid {label} count')
    return value


def identity(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value),
        'invalid original image identity',
    )
    return value


def entry_map(entries, dispositions):
    require(isinstance(entries, list), 'invalid entry inventory')
    output = {}
    names = set()
    for entry in entries:
        key = count(entry['entry_key'], 'entry key')
        name = entry['entry']
        require(isinstance(name, str) and name, 'invalid entry name')
        require(key not in output and name not in names, 'duplicate entry identity')
        require(entry['disposition'] in dispositions, 'invalid entry disposition')
        output[key] = entry
        names.add(name)
    return output


def metrics(analysis):
    if analysis is None:
        return {
            'scope': None,
            'lds_analysis': 'not_performed',
            'lds_observations_present': False,
            'verdict': 'not_performed',
            'coverage_complete': False,
            **{field: None for field in LDS_COUNTERS},
        }
    trace = analysis['scope'] == 'execution_trace'
    require(
        (
            analysis['lds_analysis'] == 'not_performed'
            if trace
            else (
                analysis['lds_analysis'] in ('performed', 'not_performed')
                and (
                    not analysis['executions']
                    or analysis['lds_analysis'] == 'performed'
                )
            )
        ),
        'analysis scope and LDS status differ',
    )
    complete = (
        analysis['transport_complete'] if trace else analysis['coverage_complete']
    )
    require(type(complete) is bool, 'invalid analysis coverage flag')
    output = {
        'scope': analysis['scope'],
        'lds_analysis': (
            'performed' if not trace and analysis['executions'] else 'not_performed'
        ),
        'lds_observations_present': False,
        'verdict': 'not_performed',
        'coverage_complete': complete,
    }
    if trace:
        output.update({field: None for field in LDS_COUNTERS})
        for field in ('expected_rows', 'completed_rows', 'entry_events', 'exit_events'):
            total = sum(count(row[field], field) for row in analysis['executions'])
            require(count(analysis[field], field) == total, 'trace row counts differ')
            output[field] = total
        return output
    for field in LDS_COUNTERS:
        total = sum(count(row[field], field) for row in analysis['executions'])
        if field in analysis:
            require(
                count(analysis[field], field) == total,
                f'aggregate {field} counter differs',
            )
        output[field] = total
    output['lds_observations_present'] = output['valid_observations'] > 0
    output['verdict'] = (
        'not_performed'
        if not analysis['executions']
        else (
            'findings'
            if output['cross_wave_pairs'] or output['collision_groups']
            else 'no_observed_findings'
        )
    )
    require(
        analysis['verdict'] == output['verdict']
        or (
            not analysis['executions'] and analysis['verdict'] == 'no_observed_findings'
        ),
        'analysis verdict and counters differ',
    )
    require(
        not complete or all(row['coverage_complete'] for row in analysis['executions']),
        'analysis coverage contradicts execution coverage',
    )
    return output


def image_report(analysis, preparation, original, summary=None, issues=None):
    original = identity(original)
    prepared = None
    classified = False
    if preparation is not None:
        require(
            preparation['original_sha256'] == original,
            'preparation image identity differs',
        )
        version = preparation.get('schema_version', 1)
        require(version in (1, 2), 'unsupported preparation report schema')
        classified = version == 2
        if classified:
            validate_preparation_report(preparation)
        prepared = entry_map(
            preparation['entries'],
            (
                {'instrumented', 'no_probes_needed', 'blocked'}
                if classified
                else {'instrumented', 'native_uncovered'}
            ),
        )
        selected_count = sum(
            row['disposition'] == 'instrumented' for row in prepared.values()
        )
        require(
            count(preparation['requested'], 'requested') == len(prepared)
            and count(preparation['instrumented'], 'instrumented') == selected_count
            and count(preparation['skipped'], 'skipped')
            == sum(
                row['disposition'] in ('native_uncovered', 'blocked')
                for row in prepared.values()
            ),
            'preparation counts differ from owning entries',
        )
        require(
            preparation['outcome']
            == ('instrumented' if selected_count else 'native_only'),
            'preparation outcome and entry dispositions differ',
        )
    if analysis is None:
        require(prepared is not None, 'missing both preparation and analysis')
        inventory = {
            key: dict(
                row,
                disposition=(
                    'selected'
                    if row['disposition'] == 'instrumented'
                    else (
                        'no_probes_needed'
                        if row['disposition'] == 'no_probes_needed'
                        else 'native_uncovered'
                    )
                ),
            )
            for key, row in prepared.items()
        }
    else:
        require(
            analysis['original_sha256'] == original, 'analysis image identity differs'
        )
        inventory = entry_map(
            analysis['entries'], {'selected', 'native_uncovered', 'no_probes_needed'}
        )
        if prepared is not None:
            for field in ('selection_profile', 'target'):
                require(
                    analysis[field] == preparation[field],
                    f'analysis {field} identity differs',
                )
    if prepared is not None:
        for key, row in prepared.items():
            owning = inventory.get(key)
            require(
                owning is not None
                and row['entry'] == owning['entry']
                and (row['disposition'] == 'instrumented')
                == (owning['disposition'] == 'selected'),
                'preparation and analysis entry identity or disposition differs',
            )
            if classified:
                no_probes = row['disposition'] == 'no_probes_needed'
                require(
                    no_probes == (owning['disposition'] == 'no_probes_needed'),
                    'preparation and capture no-probes-needed proof disposition differs',
                )
                require(
                    owning.get('no_probes_needed') == row.get('no_probes_needed'),
                    'preparation and capture no-probes-needed proof differs',
                )
        require(
            all(
                row['disposition'] != 'selected' or key in prepared
                for key, row in inventory.items()
            ),
            'selected entry missing from preparation',
        )
    require(inventory, 'empty owning image entry inventory')
    inventory_count = preparation['inventory_entries'] if classified else len(inventory)
    if analysis is not None and classified:
        require(
            len(inventory) == inventory_count,
            'preparation and capture full inventory counts differ',
        )
    not_requested = (
        preparation['not_requested']
        if classified
        else None if prepared is None else len(inventory) - len(prepared)
    )
    static = {}
    if summary is not None:
        require(
            summary['original_sha256'] == original,
            'static capture image identity differs',
        )
        require(
            isinstance(summary['entries'], list), 'invalid static capture inventory'
        )
        for row in summary['entries']:
            key = count(row['entry_key'], 'static entry key')
            require(
                key not in static and key in inventory,
                'static capture entry identity differs',
            )
            owning = inventory[key]
            require(
                row['entry'] == owning['entry']
                and type(row['selected']) is bool
                and row['selected'] == (owning['disposition'] == 'selected'),
                'static capture entry identity or selection differs',
            )
            static[key] = {
                field: None if row[field] is None else count(row[field], field)
                for field in STATIC_COUNTERS
            }
        require(
            static.keys() == inventory.keys(), 'static capture entry inventory differs'
        )
    executions = [] if analysis is None else analysis['executions']
    observed = Counter()
    serials = set()
    for execution in executions:
        serial = count(execution['serial'], 'execution serial')
        require(serial not in serials, 'duplicate execution serial')
        serials.add(serial)
        key = execution['entry_key']
        require(
            key in inventory
            and inventory[key]['entry'] == execution['entry']
            and inventory[key]['disposition'] == 'selected',
            'execution references a different or native entry',
        )
        observed[key] += 1
    entries = []
    for key, row in inventory.items():
        selected = row['disposition'] == 'selected'
        no_probes = row['disposition'] == 'no_probes_needed'
        proof = row.get('no_probes_needed')
        if no_probes:
            validate_no_probes_needed_proof(
                proof,
                preparation['original_bytes'] if classified else 512 * 1024 * 1024,
            )
            require(
                analysis is None or analysis['scope'] != 'execution_trace',
                'execution trace cannot use an LDS no-probes-needed proof',
            )
        else:
            require(proof is None, 'unexpected no-probes-needed proof')
        requested = None if prepared is None else key in prepared
        reason = None
        if requested and not selected and not no_probes:
            excluded = prepared[key]
            require(
                isinstance(excluded['reason_code'], str)
                and excluded['reason_code'] not in ('', 'none')
                and isinstance(excluded['reason'], str)
                and excluded['reason'],
                'preparation exclusion has no owning reason',
            )
            reason = {
                'code': excluded['reason_code'],
                'reason': excluded['reason'],
                'stage': excluded.get('stage'),
            }
        entries.append(
            {
                'entry': row['entry'],
                'entry_key': key,
                'disposition': (
                    'instrumented'
                    if selected
                    else (
                        'no_probes_needed'
                        if no_probes
                        else (
                            'blocked'
                            if classified and requested
                            else (
                                'not_requested'
                                if requested is False
                                else 'native_uncovered'
                            )
                        )
                    )
                ),
                'no_probes_needed': proof,
                'preparation_stage': prepared[key].get('stage') if requested else None,
                'preparation_requested': requested,
                'first_preparation_exclusion': reason,
                'captured_dispatches': observed[key] if selected else None,
                'static_probe_sites': (
                    {field: 0 for field in STATIC_COUNTERS}
                    if no_probes
                    else static.get(key, {field: None for field in STATIC_COUNTERS})
                ),
            }
        )
    scoped_metrics = metrics(analysis)
    provided_complete = len(inventory) == inventory_count and all(
        entry['disposition'] == 'no_probes_needed'
        or (
            entry['disposition'] == 'instrumented'
            and observed[entry['entry_key']]
            and scoped_metrics['coverage_complete']
        )
        for entry in entries
    )
    if analysis is not None and 'provided_entry_coverage_complete' in analysis:
        require(
            analysis['provided_entry_coverage_complete'] == provided_complete,
            'provided-entry coverage contradicts owning capture evidence',
        )
    return {
        'original_sha256': original,
        'entry_inventory_count': inventory_count,
        'entry_inventory_details_complete': len(entries) == inventory_count,
        'preparation_requested_entries': None if prepared is None else len(prepared),
        'preparation_excluded_entries': (
            None if prepared is None else preparation['skipped']
        ),
        'entries_not_requested': not_requested,
        'no_probes_needed_entries': (
            sum(row['disposition'] == 'no_probes_needed' for row in entries)
            if classified
            else None
        ),
        'blocked_entries': preparation['blocked'] if classified else None,
        'instrumented_entries': sum(
            row['disposition'] == 'selected' for row in inventory.values()
        ),
        'retained_native_entries': inventory_count
        - sum(row['disposition'] == 'selected' for row in inventory.values()),
        'observed_instrumented_entries': len(observed),
        'captured_dispatches': len(executions),
        'native_dispatches': None,
        'selected_entries_not_observed': [
            {'original_sha256': original, 'entry_key': key, 'entry': row['entry']}
            for key, row in inventory.items()
            if row['disposition'] == 'selected' and key not in observed
        ],
        'entries': entries,
        'issues': list(issues or []),
        'analysis': scoped_metrics,
        'provided_entry_coverage_complete': bool(provided_complete) and not issues,
    }


def summarize(images, aggregate=None):
    require(images, 'empty image inventory')
    output = {'images': images, 'native_dispatches': None}
    for field in (
        'entry_inventory_count',
        'preparation_requested_entries',
        'preparation_excluded_entries',
        'entries_not_requested',
        'no_probes_needed_entries',
        'blocked_entries',
        'instrumented_entries',
        'retained_native_entries',
        'observed_instrumented_entries',
        'captured_dispatches',
    ):
        values = [image[field] for image in images]
        output[field] = None if None in values else sum(values)
    output['selected_entries_not_observed'] = [
        row for image in images for row in image['selected_entries_not_observed']
    ]
    selected = [
        entry
        for image in images
        for entry in image['entries']
        if entry['disposition'] == 'instrumented'
    ]
    output['static_probe_sites'] = {}
    for field in STATIC_COUNTERS:
        values = [entry['static_probe_sites'][field] for entry in selected]
        output['static_probe_sites'][field] = None if None in values else sum(values)
    exclusions = {}
    for image in images:
        for entry in image['entries']:
            reason = entry['first_preparation_exclusion']
            if reason is None:
                continue
            key = (reason['code'], reason['reason'], reason['stage'])
            exclusions.setdefault(key, []).append(
                {field: entry[field] for field in ('entry', 'entry_key')}
                | {'original_sha256': image['original_sha256']}
            )
    output['first_preparation_exclusions'] = [
        {
            'code': code,
            'reason': reason,
            'stage': stage,
            'count': len(entries),
            'entries': entries,
        }
        for (code, reason, stage), entries in sorted(
            exclusions.items(), key=lambda item: str(item[0])
        )
    ]
    if aggregate is None:
        output['analysis'] = dict(images[0]['analysis'])
    else:
        performed = [
            image
            for image in images
            if image['analysis']['lds_analysis'] == 'performed'
            and image['captured_dispatches']
        ]
        analysis = {
            field: (
                sum(image['analysis'][field] for image in performed)
                if performed
                else None
            )
            for field in LDS_COUNTERS
        }
        analysis.update(
            scope=aggregate['scope'],
            lds_analysis='performed' if performed else 'not_performed',
            lds_observations_present=any(
                image['analysis']['lds_observations_present'] for image in images
            ),
            coverage_complete=all(
                image['provided_entry_coverage_complete'] for image in images
            ),
        )
        analysis['verdict'] = (
            'not_performed'
            if not performed
            else (
                'findings'
                if analysis['cross_wave_pairs'] or analysis['collision_groups']
                else 'no_observed_findings'
            )
        )
        for field in ('lds_analysis', 'lds_observations_present', 'coverage_complete'):
            require(aggregate[field] == analysis[field], f'aggregate {field} differs')
        for field in ('cross_wave_pairs', 'collision_groups'):
            require(
                count(aggregate[field], field) == (analysis[field] or 0),
                f'aggregate {field} counter differs',
            )
        analysis['whole_application_coverage'] = aggregate['whole_application_coverage']
        require(
            analysis['whole_application_coverage'] == 'not_established',
            'unexpected whole-application coverage claim',
        )
        output['analysis'] = analysis
    output['provided_entry_coverage_complete'] = all(
        image['provided_entry_coverage_complete'] for image in images
    )
    return output


def build_report(result):
    """Validate owning evidence and keep independently executed runs separate."""
    summaries = result.get('capture_summaries', {})
    preparations = result.get('preparation_reports', {})
    runs = {}
    for name, source in (('trace', 'analysis'), ('pinned_lds', 'lds_analysis')):
        analysis = result[source]
        runs[name] = summarize(
            [
                image_report(
                    analysis,
                    preparations.get(name),
                    analysis['original_sha256'],
                    summaries.get(name),
                )
            ]
        )
    aggregate = result['all_images_analysis']
    require(
        aggregate['scope'] == 'provided_application_images', 'invalid all-image scope'
    )
    static = {}
    for summary in summaries.get('all_images', []):
        original = identity(summary['original_sha256'])
        require(original not in static, 'duplicate static capture image identity')
        static[original] = summary
    images = []
    identities = set()
    run_identity = None
    for image in aggregate['images']:
        original = identity(image['original_sha256'])
        require(original not in identities, 'duplicate application image identity')
        identities.add(original)
        analysis = image['analysis']
        if analysis is not None:
            if run_identity is None:
                run_identity = analysis['run_identity']
            require(
                analysis['run_identity'] == run_identity,
                'captures belong to different runs',
            )
        require(
            original not in static or analysis is not None,
            'static capture has no owning analysis',
        )
        reported = image_report(
            analysis,
            image['preparation'],
            original,
            static.pop(original, None),
            image['issues'],
        )
        require(
            reported['preparation_requested_entries']
            == reported['entry_inventory_count'],
            'all-image preparation omitted an inventory entry',
        )
        require(
            sorted(image['selected_entries_not_observed'])
            == sorted(
                row['entry'] for row in reported['selected_entries_not_observed']
            ),
            'unobserved selected entry inventory differs',
        )
        images.append(reported)
    require(not static, 'static capture references an unknown application image')
    runs['all_images'] = summarize(images, aggregate)
    return {
        'schema_version': 2,
        'source_test_passed': result.get('passed'),
        'fresh_replay_identical': result.get('fresh_replay_identical'),
        'run_accounting': 'independent_application_runs',
        'runs': runs,
    }


def display(value):
    return (
        'unknown'
        if value is None
        else str(value).lower() if type(value) is bool else str(value)
    )


def render_report(report):
    """Render the same counters and per-entry ownership as the JSON report."""
    lines = [
        'ConSan prefill report',
        f"Source test passed: {display(report['source_test_passed'])}; fresh replay identical: {display(report['fresh_replay_identical'])}",
        'The three application runs below are independent; do not add their coverage.',
        'Preparation exclusions retain native execution and are not compiler/process failures.',
        'No probes needed means the producer proved that this LDS profile has no applicable memory sites.',
        'Native dispatch counts are unknown; inventory entries do not prove execution.',
        'Zero observed findings with incomplete coverage does not establish race freedom.',
        'Static sites count insertion locations, not dynamic callbacks or observations.',
        'LDS entry/exit callback counts are unavailable in the saved manifest.',
        'address_group_exhaustion overlaps lost_occurrences; do not add these counters.',
    ]
    for name, label in RUN_LABELS.items():
        run = report['runs'][name]
        analysis = run['analysis']
        lines.extend(
            [
                '',
                label,
                f"Entry inventory: {run['entry_inventory_count']}; preparation requested: {display(run['preparation_requested_entries'])}; preparation excluded: {display(run['preparation_excluded_entries'])}; not requested: {display(run['entries_not_requested'])}",
                f"Instrumented entries: {run['instrumented_entries']}; retained native: {run['retained_native_entries']}",
                f"Preparation classification: no probes needed: {display(run['no_probes_needed_entries'])}; blocked: {display(run['blocked_entries'])}",
                f"Observed instrumented entries: {run['observed_instrumented_entries']}; captured dispatches: {run['captured_dispatches']}",
                'Selected static insertion sites: '
                + '; '.join(
                    f"{field}={display(run['static_probe_sites'][field])}"
                    for field in STATIC_COUNTERS
                ),
                f"Native dispatches: {display(run['native_dispatches'])}",
                f"LDS analysis: {analysis['lds_analysis']}; observations present: {display(analysis['lds_observations_present'])}; verdict: {analysis['verdict']}",
                f"Analysis scope: {display(analysis['scope'])}",
                f"Reported scope coverage complete: {display(analysis['coverage_complete'])}; provided-entry coverage complete: {display(run['provided_entry_coverage_complete'])}",
            ]
        )
        if 'whole_application_coverage' in analysis:
            lines.append(
                f"Whole-application coverage: {analysis['whole_application_coverage']}"
            )
        lines.extend(f'{field}: {display(analysis[field])}' for field in LDS_COUNTERS)
        lines.append(
            'Selected entries not observed: '
            + display(len(run['selected_entries_not_observed']))
        )
        lines.extend(
            f"  {row['original_sha256']}:{row['entry_key']} {row['entry']}"
            for row in run['selected_entries_not_observed']
        )
        lines.append('First preparation exclusions (first reason only):')
        if not run['first_preparation_exclusions']:
            lines.append(
                '  none recorded'
                if run['preparation_requested_entries'] is not None
                else '  unknown (preparation reports absent)'
            )
        lines.extend(
            f"  {row['count']} x {row['code']}: {row['reason']}"
            for row in run['first_preparation_exclusions']
        )
        for image in run['images']:
            lines.extend(
                [
                    f"Image {image['original_sha256']}",
                    'key | disposition | requested | stage | dispatches | LDS sites | barrier sites | trace sites | entry | first preparation exclusion',
                ]
            )
            for entry in image['entries']:
                static = entry['static_probe_sites']
                reason = entry['first_preparation_exclusion']
                cells = [
                    entry['entry_key'],
                    entry['disposition'],
                    entry['preparation_requested'],
                    entry['preparation_stage'],
                    entry['captured_dispatches'],
                    *(static[field] for field in STATIC_COUNTERS),
                    entry['entry'],
                    '' if reason is None else f"{reason['code']}: {reason['reason']}",
                ]
                lines.append(' | '.join(display(cell) for cell in cells))
    return '\n'.join(lines) + '\n'


def write_report(result_path, output_dir=None):
    """Regenerate report files using saved JSON, without invoking other tools."""
    result_path = Path(result_path)
    output_dir = result_path.parent if output_dir is None else Path(output_dir)
    json_path = output_dir / 'coverage-report.json'
    text_path = output_dir / 'coverage-report.txt'
    require(
        result_path.resolve() not in (json_path.resolve(), text_path.resolve()),
        'report output would overwrite result source',
    )
    report = build_report(json.loads(result_path.read_text()))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    text_path.write_text(render_report(report))
    return json_path, text_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    _, text_path = write_report(args.result, args.output)
    print(text_path.read_text(), end='')


if __name__ == '__main__':
    main()
