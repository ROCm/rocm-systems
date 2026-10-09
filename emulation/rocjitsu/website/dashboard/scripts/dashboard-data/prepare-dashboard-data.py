#!/usr/bin/env python3
"""Build one schema-2 run file and its catalog from recorded benchmark executions.

This does not create or modify index.json or metadata.json.
publish-dashboard-run.py copies the built files onto a publication snapshot
and updates that snapshot's index. Metadata is bundled website configuration.

Completed results use the median of accepted timing_results_s samples. Failed
and timed-out results always have null duration, even with partial samples:
partial work is not a completed-workload measurement. No completion is invented.
Logical workload IDs hash suite/name/raw problem, never target, mode or case ID.
Full wire-schema validation is a separate workflow step before publication.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def scalar(value):
    """Preserve native scalars; encode null/containers without lossy flattening."""
    return (
        canonical(value) if value is None or isinstance(value, (list, dict)) else value
    )


def environment(inputs):
    details = {}
    for raw in inputs:
        mode = raw['execution_summary']['threading_mode']
        for section, values in raw['provenance'].items():
            for key, value in values.items():
                # Source revision/time already live in source, not environment
                # identity: comparing different rocjitsu commits is the purpose.
                if section == 'rocjitsu' and key in (
                    'rocjitsu_commit_sha',
                    'rocjitsu_commit_timestamp',
                ):
                    continue
                prefix = (
                    mode + '.'
                    if section == 'rocjitsu' and key == 'target_config_sha256'
                    else ''
                )
                details[f'{prefix}{section}.{key}'] = scalar(value)
        for key in (
            'warmups',
            'samples',
            'timeout_seconds',
            'benchmark_suite',
            'threading_mode',
        ):
            details[f'{mode}.execution.{key}'] = raw['execution_summary'][key]
    return [
        {'key': key, 'label': key, 'value': value}
        for key, value in sorted(details.items())
    ]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def instant(value):
    require(
        isinstance(value, str)
        and re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})', value
        ),
        f'Expected strict ISO timestamp, got {value!r}',
    )
    if value[-1] != 'Z':
        require(
            int(value[-5:-3]) <= 23 and int(value[-2:]) <= 59,
            f'Invalid timestamp offset: {value}',
        )
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def validate_raw(raw, *, expected_sha, expected_corpus_sha):
    require(isinstance(raw, dict), 'Raw run must be an object')
    summary = raw['execution_summary']
    require(isinstance(summary, dict), 'execution_summary must be an object')
    for key, minimum in (('warmups', 0), ('samples', 1)):
        require(
            type(summary[key]) is int and summary[key] >= minimum,
            f'{key} must be an integer >= {minimum}',
        )
    require(
        number(summary['timeout_seconds']) and summary['timeout_seconds'] > 0,
        'timeout_seconds must be positive and finite',
    )
    require(
        number(summary['wall_time_s']) and summary['wall_time_s'] >= 0,
        'wall_time_s must be nonnegative and finite',
    )
    require(text(summary['benchmark_suite']), 'benchmark_suite must be nonempty text')
    require(
        summary['threading_mode'] in ('single', 'default'), 'Invalid threading_mode'
    )
    require(
        summary['finished_at'] is not None, 'Missing finished_at: incomplete raw run'
    )
    start, finish = instant(summary['timestamp']), instant(summary['finished_at'])
    require(start <= finish, 'finished_at precedes execution timestamp')
    require(
        finish <= datetime.now(timezone.utc), 'Run must be completed by publication'
    )
    provenance = raw['provenance']
    require(isinstance(provenance, dict), 'provenance must be an object')
    for section in ('machine', 'rocjitsu', 'corpus', 'auxiliary'):
        require(
            isinstance(provenance[section], dict),
            f'provenance.{section} must be an object',
        )
    for key in ('hostname', 'platform', 'kernel', 'cpu'):
        require(
            text(provenance['machine'][key]), f'machine.{key} must be nonempty text'
        )
    for section, prefix, expected in (
        ('rocjitsu', 'rocjitsu', expected_sha),
        ('corpus', 'corpus', expected_corpus_sha),
    ):
        actual = provenance[section][prefix + '_commit_sha']
        require(
            isinstance(actual, str) and re.fullmatch(r'[a-fA-F0-9]{40}', actual),
            f'{prefix}_commit_sha must be a full 40-hex SHA',
        )
        require(
            isinstance(expected, str) and re.fullmatch(r'[a-fA-F0-9]{40}', expected),
            f'Expected {prefix} SHA must be a full 40-hex SHA',
        )
        require(
            actual.lower() == expected.lower(),
            f'{prefix}_commit_sha does not match expected SHA',
        )
        committed = instant(provenance[section][prefix + '_commit_timestamp'])
        require(committed <= start, f'{prefix} commit is later than execution start')
    rocjitsu = provenance['rocjitsu']
    require(
        text(rocjitsu['rocm_sdk_version']), 'rocm_sdk_version must be nonempty text'
    )
    hashes = rocjitsu['target_config_sha256']
    require(
        isinstance(hashes, dict) and hashes,
        'target_config_sha256 must be a nonempty object',
    )
    for target, sha in hashes.items():
        require(
            re.fullmatch(r'[A-Za-z0-9._-]+', target)
            and isinstance(sha, str)
            and re.fullmatch(r'[a-fA-F0-9]{64}', sha),
            f'Invalid target_config_sha256 for {target}',
        )
    results = raw['benchmark_results']
    require(
        isinstance(results, list) and results,
        'benchmark_results must be a nonempty array',
    )
    for result in results:
        require(isinstance(result, dict), 'Benchmark result must be an object')
        for key in ('id', 'name', 'suite', 'target'):
            require(text(result[key]), f'Result {key} must be nonempty text')
        require(
            result['target'] in hashes,
            f'Missing target config hash for {result["target"]}',
        )
        require(isinstance(result['problem'], dict), 'problem must be an object')
        require(
            all(text(key) for key in result['problem']), 'problem keys must be nonempty'
        )
        require(
            result['status'] in ('completed', 'failed', 'timeout'),
            'Invalid result status',
        )
        require(
            result['exit_code'] is None or type(result['exit_code']) is int,
            'exit_code must be an integer or null',
        )
        require(
            result['error'] is None or text(result['error']),
            'error must be nonempty text or null',
        )
        timings = result['timing_results_s']
        require(
            isinstance(timings, list) and all(number(t) and t > 0 for t in timings),
            'timing_results_s must contain positive finite numbers',
        )
        if result['status'] == 'completed':
            require(
                timings
                and result['error'] is None
                and result['exit_code'] in (0, None),
                'Completed result requires timings, zero or null exit_code and null error',
            )
    # Enforce finite JSON throughout, including arbitrary nested provenance/problem.
    canonical(raw)


def validate_consistency(inputs):
    modes = [raw['execution_summary']['threading_mode'] for raw in inputs]
    require(
        len(set(modes)) == len(modes),
        'Duplicate threading_mode: supply one raw run per mode',
    )
    reference = inputs[0]
    for raw in inputs[1:]:
        for section in ('machine', 'corpus', 'auxiliary', 'rocjitsu'):
            left, right = reference['provenance'][section], raw['provenance'][section]
            if section == 'rocjitsu':
                left = {k: v for k, v in left.items() if k != 'target_config_sha256'}
                right = {k: v for k, v in right.items() if k != 'target_config_sha256'}
            require(
                canonical(left) == canonical(right),
                f'Inconsistent {section} provenance across raw runs',
            )
        for key in ('warmups', 'samples', 'timeout_seconds'):
            require(
                reference['execution_summary'][key] == raw['execution_summary'][key],
                f'Inconsistent {key} across raw runs',
            )


def normalize_runs(raw_runs, *, run_id, branch, commit_message, machine_id, trigger):
    """Build one run and its catalog, independent of file I/O or CLI parsing."""
    tests = {}
    groups = {}
    for raw in raw_runs:
        mode = {'single': 'ST', 'default': 'MT'}[
            raw['execution_summary']['threading_mode']
        ]
        for result in raw['benchmark_results']:
            # Identity uses raw values, before converting nested values to strings.
            definition = {key: result[key] for key in ('suite', 'name', 'problem')}
            test_id = 'test-' + digest(definition)
            tests[test_id] = {
                'id': test_id,
                **definition,
                'problem': {
                    key: scalar(value) for key, value in definition['problem'].items()
                },
            }
            results = groups.setdefault((result['target'], mode), {})
            require(
                test_id not in results,
                f'Duplicate logical result in {result["target"]}:{mode}: {test_id}',
            )
            results[test_id] = {
                'testId': test_id,
                'status': result['status'],
                'durationSeconds': (
                    statistics.median(result['timing_results_s'])
                    if result['status'] == 'completed'
                    else None
                ),
                'error': result['error'],
            }

    # Preserve the published target:mode ordering so old runs remain retryable.
    ordered_groups = sorted(groups.items(), key=lambda item: ':'.join(item[0]))
    catalog = {
        'tests': [tests[key] for key in sorted(tests)],
        'configurations': {
            f'{target}:{mode}': sorted(results)
            for (target, mode), results in ordered_groups
        },
    }
    catalog['id'] = 'catalog-' + digest(catalog)
    source = raw_runs[0]['provenance']['rocjitsu']
    completed_at = max(
        (raw['execution_summary']['finished_at'] for raw in raw_runs),
        key=lambda value: (instant(value), value),
    )
    run = {
        'schemaVersion': 2,
        'id': run_id,
        'testCatalog': f"test-catalogs/{catalog['id']}.json",
        'source': {
            'commit': source['rocjitsu_commit_sha'],
            'committedAt': source['rocjitsu_commit_timestamp'],
            'branch': branch,
            'message': commit_message,
        },
        'execution': {
            'completedAt': completed_at,
            'machine': machine_id,
            'trigger': trigger,
        },
        'environment': environment(raw_runs),
        'configurations': [
            {
                'target': target,
                'threadingMode': mode,
                'results': [results[key] for key in sorted(results)],
            }
            for (target, mode), results in ordered_groups
        ],
    }
    return run, catalog


def write_json(path, value):
    """Replace a file atomically; callers check immutable conflicts first."""
    if path.exists() and canonical(read_json(path)) == canonical(value):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.dashboard-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_output_path(root):
    """Refuse symlinks (including ancestors/orphans), not just lexical escapes.

    The caller must serialize publishers and keep this staging tree private;
    these checks are not a sandbox against concurrent hostile path replacement.
    """
    for path in (root, *root.parents):
        require(not path.is_symlink(), f'Output path contains a symlink: {path}')
    if root.exists():
        require(root.is_dir(), 'data-dir must be a directory')
        for directory, folders, files in os.walk(root, followlinks=False):
            for name in folders + files:
                path = Path(directory) / name
                require(not path.is_symlink(), f'Output contains a symlink: {path}')
                require(
                    path.is_dir() or path.is_file(),
                    f'Output contains a special file: {path}',
                )


def run_filename(run):
    directory = (
        'default-branch' if run['source']['branch'] == 'develop' else 'side-branches'
    )
    return f"runs/{directory}/{run['id']}.json"


def write_built_run(root, run, catalog):
    """Write one run and its catalog. Leave index.json and metadata.json unchanged."""
    validate_output_path(root)
    canonical([run, catalog])  # Reject non-finite output before any writes.
    resources = {
        run['testCatalog']: catalog,
        run_filename(run): run,
    }
    for name, value in resources.items():
        if (root / name).exists():
            require(
                canonical(read_json(root / name)) == canonical(value),
                f'Immutable resource conflict: {name}',
            )
    for name, value in resources.items():
        write_json(root / name, value)


def prepare(
    raw_run,
    *,
    data_dir,
    run_id,
    expected_sha,
    expected_corpus_sha,
    machine_id,
    branch,
    commit_message,
    trigger,
):
    """Load raw ST/MT runs and write one built run. Does not write index or metadata."""
    require(
        re.fullmatch(r'[A-Za-z0-9._-]+', run_id) and run_id not in ('.', '..'),
        '--run-id must be a safe filename token',
    )
    for name, value in (
        ('branch', branch),
        ('commit_message', commit_message),
        ('machine_id', machine_id),
    ):
        require(text(value), f'{name} must be nonempty text')
    raw_runs = [read_json(Path(path)) for path in raw_run]
    require(raw_runs, 'Supply at least one raw run')
    for raw in raw_runs:
        validate_raw(
            raw, expected_sha=expected_sha, expected_corpus_sha=expected_corpus_sha
        )
        for section in ('rocjitsu', 'corpus'):
            key = section + '_commit_sha'
            raw['provenance'][section][key] = raw['provenance'][section][key].lower()
    validate_consistency(raw_runs)
    run, catalog = normalize_runs(
        raw_runs,
        run_id=run_id,
        branch=branch,
        commit_message=commit_message,
        machine_id=machine_id,
        trigger=trigger,
    )
    write_built_run(Path(os.path.abspath(data_dir)), run, catalog)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-run', action='append', required=True)
    parser.add_argument('--data-dir', required=True)
    for name in (
        'run-id',
        'expected-sha',
        'expected-corpus-sha',
        'machine-id',
        'branch',
        'commit-message',
    ):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--trigger', choices=('auto', 'manual'), required=True)
    args = parser.parse_args()
    try:
        run = prepare(**vars(args))
        print(f"Prepared schema-2 run {run['id']}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
