#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Convert raw benchmark JSON to one default-policy dashboard run JSON.

Existing dashboard catalogs are referenced, not created or published here.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode(
        'utf-8'
    )


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def build_published_run(raw, workflow):
    github = {
        'run_id': workflow['run_id'],
        'run_attempt': workflow['run_attempt'],
        'trigger': workflow['trigger'],
    }
    source = {
        'commit': workflow['source_sha'],
        'commit_subject': workflow['source_subject'],
        'repository_url': 'https://github.com/' + workflow['repository'],
    }
    summary = raw['execution_summary']
    require(summary['finished_at'] is not None, 'Cannot convert unfinished checkpoint')
    require(
        parse_timestamp(summary['finished_at'])
        >= parse_timestamp(summary['timestamp']),
        'Finish precedes start',
    )
    require(
        type(summary['wall_time_s']) in (int, float)
        and math.isfinite(summary['wall_time_s'])
        and summary['wall_time_s'] >= 0,
        'Invalid wall time',
    )
    committed = raw['provenance']['rocjitsu']['rocjitsu_commit_timestamp']
    require(
        parse_timestamp(committed) <= parse_timestamp(summary['timestamp']),
        'Source commit is later than benchmark start',
    )
    samples = summary['samples']
    require(
        type(samples) is int and samples > 0 and samples % 2 == 1,
        'Invalid sample count',
    )
    selected = [
        result
        for result in raw['benchmark_results']
        if result.get('thread_policy') == 'default'
    ]
    identities = [(result['id'], result['target']) for result in selected]
    require(
        identities and len(identities) == len(set(identities)),
        'Missing or duplicate default-policy results',
    )
    definitions, targets = {}, {}
    for result in selected:
        problem = dict(result['problem'])
        require(
            'operation' not in problem or problem['operation'] == result['operation'],
            'Operation metadata collision',
        )
        problem['operation'] = result['operation']
        # The dashboard data contract requires scalar problem values. Serialize
        # containers and null as JSON strings to preserve their contents.
        problem = {
            key: (
                value
                if type(value) in (str, int, float, bool)
                else json.dumps(value, sort_keys=True, allow_nan=False)
            )
            for key, value in problem.items()
        }
        definition = {
            'id': result['id'],
            'suite': result['suite'],
            'name': result['name'],
            'problem': problem,
        }
        require(
            result['id'] not in definitions or definitions[result['id']] == definition,
            'Conflicting workload identity',
        )
        definitions[result['id']] = definition
        timings = result['timing_results_s']
        require(
            isinstance(timings, list)
            and all(
                type(t) in (int, float) and math.isfinite(t) and t > 0 for t in timings
            ),
            'Invalid timing samples',
        )
        status = result['status']
        require(status in ('completed', 'failed', 'timeout'), 'Invalid result status')
        require(
            result['error'] is None
            or isinstance(result['error'], str)
            and result['error'].strip(),
            'Invalid result error',
        )
        if status == 'completed':
            require(
                len(timings) == samples
                and result['error'] is None
                and result['exit_code'] in (0, None),
                'Invalid completed measurement',
            )
        targets.setdefault(result['target'], []).append(
            {
                'testId': result['id'],
                'status': status,
                'durationSeconds': (
                    statistics.median(timings) if status == 'completed' else None
                ),
                'error': result['error'],
                'timing_results_s': timings,
            }
        )
    catalog = {
        'tests': [definitions[key] for key in sorted(definitions)],
        'targets': {
            target: sorted(result['testId'] for result in results)
            for target, results in sorted(targets.items())
        },
    }
    catalog['id'] = 'default-' + hashlib.sha256(encoded(catalog)).hexdigest()
    run_id = f"github-{github['run_id']}-attempt-{github['run_attempt']}-default"
    details = {'thread_policy': 'default'}
    for section, values in raw['provenance'].items():
        for key, value in values.items():
            if section == 'rocjitsu' and key in (
                'rocjitsu_commit_sha',
                'rocjitsu_commit_timestamp',
            ):
                continue
            details[f'{section}.{key}'] = (
                value
                if type(value) in (str, int, float, bool) and value != ''
                else json.dumps(value, sort_keys=True, allow_nan=False)
            )
    details['workflow.commit_sha'] = workflow['workflow_sha']
    details['tensile.commit_sha'] = workflow['tensile_sha']
    for key in ('warmups', 'samples', 'timeout_seconds', 'benchmark_suite'):
        details['execution.' + key] = summary[key]
    labels = {
        'thread_policy': 'Thread policy',
        'workflow.commit_sha': 'Workflow commit',
        'tensile.commit_sha': 'Tensile commit',
        'machine.hostname': 'Hostname',
        'machine.platform': 'Platform',
        'machine.kernel': 'Kernel',
        'machine.cpu': 'CPU',
        'rocjitsu.rocm_sdk_version': 'ROCm SDK',
        'corpus.corpus_commit_sha': 'Corpus commit',
        'corpus.corpus_commit_timestamp': 'Corpus commit timestamp',
        'auxiliary.python': 'Python',
        'execution.warmups': 'Warmups',
        'execution.samples': 'Samples',
        'execution.timeout_seconds': 'Timeout (seconds)',
        'execution.benchmark_suite': 'Benchmark suite',
    }
    run = {
        'id': run_id,
        'comparisonId': f"github-{github['run_id']}-attempt-{github['run_attempt']}",
        'testCatalog': f"test-catalogs/{catalog['id']}.json",
        'plugin': {'id': 'vanilla', 'name': 'Vanilla'},
        'source': {
            # The workflow requires the measured commit to be in develop history.
            # This is its source branch, not the workflow dispatch branch.
            'branch': 'develop',
            'commit': source['commit'],
            'committedAt': committed,
            'message': source['commit_subject'],
        },
        'execution': {
            'completedAt': summary['finished_at'],
            'trigger': github['trigger'],
            'machine': workflow['runner'],
        },
        'environment': [
            {
                'key': key,
                'label': labels.get(
                    key,
                    key.replace('.', ' ').replace('_', ' ').replace('-', ' ').title(),
                ),
                'value': value,
            }
            for key, value in sorted(details.items())
        ],
        'targets': [
            {'id': target, 'results': sorted(results, key=lambda r: r['testId'])}
            for target, results in sorted(targets.items())
        ],
    }
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--run-attempt', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--commit-message', required=True)
    parser.add_argument('--machine', required=True)
    parser.add_argument('--workflow-sha', required=True)
    parser.add_argument('--tensile-sha', required=True)
    parser.add_argument('--event', choices=('push', 'workflow_dispatch'), required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw_run.read_text())
    workflow = {
        'run_id': args.run_id,
        'run_attempt': args.run_attempt,
        'repository': args.repository,
        'source_subject': args.commit_message,
        'source_sha': raw['provenance']['rocjitsu']['rocjitsu_commit_sha'],
        'runner': args.machine,
        'workflow_sha': args.workflow_sha,
        'tensile_sha': args.tensile_sha,
        'trigger': 'manual' if args.event == 'workflow_dispatch' else 'auto',
    }
    result = build_published_run(raw, workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded(result))


if __name__ == '__main__':
    main()
