# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execute all entries across two images and replay their owning application index."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        'prepare',
        'runner',
        'inspect',
        'analyze',
        'plugin',
        'launcher',
        'coordinator',
        'analyze-coordinator',
        'first',
        'second',
        'config',
        'root',
        'source',
    ):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='all-images-sim-', dir=args.root))
    print(f'evidence: {directory}', flush=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('RJ_CONSAN_') and key != 'HSA_TOOLS_LIB'
    }
    commands = []

    def execute(label, command, env=environment, expected=0):
        argv = [str(part) for part in command]
        result = subprocess.run(
            argv, env=env, capture_output=True, text=True, timeout=90, check=False
        )
        commands.append({'label': label, 'argv': argv, 'exit': result.returncode})
        (directory / (label + '.stdout')).write_text(result.stdout)
        (directory / (label + '.stderr')).write_text(result.stderr)
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == expected, (label, result.returncode, result.stderr)
        return result.stdout

    binaries = (
        args.prepare,
        args.runner,
        args.inspect,
        args.analyze,
        args.plugin,
        args.launcher,
        args.first,
        args.second,
    )
    roots = set()
    for binary in binaries:
        roots.add(
            next(
                parent
                for parent in binary.parents
                if (parent / 'CMakeCache.txt').is_file()
            )
        )
    assert len(roots) == 1, 'mixed checkout artifacts'
    assert (
        f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}'
        in (roots.pop() / 'CMakeCache.txt').read_text()
    )
    (directory / 'inputs.json').write_text(
        json.dumps(
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in binaries
            },
            indent=2,
        )
    )
    inputs = directory / 'inputs'
    inputs.mkdir()
    first, second = inputs / 'first.hsaco', inputs / 'second.hsaco'
    shutil.copyfile(args.first, first)
    shutil.copyfile(args.second, second)
    app = [args.launcher, '--config', args.config, '--', args.runner, first]
    options = ['--application-workgroups=1', '--second-image=' + str(second)]
    baseline = [
        json.loads(line)
        for line in execute('baseline', [*app, 'baseline', *options]).splitlines()
        if line.startswith('{')
    ]
    assert baseline[-1]['loads'] == 2 and baseline[-1]['completed'] == 8, baseline
    output = directory / 'application'
    execute(
        'prepare',
        [
            args.coordinator,
            '--prepare',
            args.prepare,
            '--target',
            'gfx1250',
            '--images-dir',
            inputs,
            '--output',
            output,
            '--require-all',
        ],
    )
    index_path = output / 'application.json'
    index = json.loads(index_path.read_text())
    assert len(index['images']) == 2, index
    for image in index['images']:
        report = image['preparation']
        assert (
            report['requested'] == report['instrumented'] == 3
            and report['skipped'] == 0
        ), report
        assert [entry['entry'] for entry in report['entries']] == [
            'mk_a',
            'mk_b',
            'mk_c',
        ], report
    active = (
        environment
        | json.loads((output / 'launch-environment.json').read_text())
        | {'HSA_TOOLS_LIB': str(args.plugin), 'RJ_CONSAN_PLUGIN_PATH': str(args.plugin)}
    )
    ledger = [
        json.loads(line)
        for line in execute(
            'instrumented', [*app, 'instrumented', *options], env=active
        ).splitlines()
        if line.startswith('{')
    ]
    assert ledger[:-1] == baseline[:-1], 'original ABI/output/completion changed'
    summary = ledger[-1]
    assert summary['app_status'] == summary['finish_status'] == 0, summary
    assert summary['loads'] == 2 and summary['completed'] == 8, summary
    assert summary['submitted'] == summary['copied'] == summary['written'] == 8, summary
    assert (
        summary['forwarded']
        == summary['native_submitted']
        == summary['native_completed']
        == 0
    ), summary
    for ordinal, image in enumerate(index['images']):
        capture = output / image['capture']
        inspected = json.loads(execute(f'inspect-{ordinal}', [args.inspect, capture]))
        assert len(inspected['entries']) == 3 and all(
            entry['selected'] for entry in inspected['entries']
        ), inspected
        assert [execution['serial'] for execution in inspected['executions']] == [
            1,
            2,
            3,
            4,
        ], inspected
        assert {execution['completion'] for execution in inspected['executions']} == {
            3
        }, inspected
    analyzed = execute(
        'analyze', [args.analyze_coordinator, index_path, '--analyzer', args.analyze]
    )
    analysis = json.loads(analyzed)
    assert (
        analysis['lds_analysis'] == 'performed' and analysis['coverage_complete']
    ), analysis
    assert (
        len(analysis['images']) == 2 and len(analysis['run_identity']) == 32
    ), analysis
    assert any(
        image['analysis']['scope'] == 'native_lds_single_range_mixed_scopes'
        for image in analysis['images']
    ), analysis
    for image in analysis['images']:
        assert not image['selected_entries_not_observed'] and not image['issues'], image
        assert len(image['analysis']['executions']) == 4, image
        assert all(
            execution['valid_observations'] > 0
            for execution in image['analysis']['executions']
        ), image
    # A late image is still original code. Admit it natively and preserve the
    # explicit provided-image scope, rather than rejecting useful execution.
    bounded = directory / 'provided-first-image'
    execute(
        'prepare-first-only',
        [
            args.coordinator,
            first,
            '--prepare',
            args.prepare,
            '--target',
            'gfx1250',
            '--output',
            bounded,
            '--require-all',
        ],
    )
    native_active = (
        environment
        | json.loads((bounded / 'launch-environment.json').read_text())
        | {'HSA_TOOLS_LIB': str(args.plugin), 'RJ_CONSAN_PLUGIN_PATH': str(args.plugin)}
    )
    unlisted = [
        json.loads(line)
        for line in execute(
            'unlisted-native', [*app, 'instrumented', *options], env=native_active
        ).splitlines()
        if line.startswith('{')
    ]
    assert (
        unlisted[:-1] == baseline[:-1]
    ), 'unlisted native image changed application output'
    native_summary = unlisted[-1]
    assert (
        native_summary['loads'] == 2 and native_summary['completed'] == 8
    ), native_summary
    assert (
        native_summary['submitted']
        == native_summary['copied']
        == native_summary['written']
        == 4
    ), native_summary
    assert (
        native_summary['native_submitted'] == native_summary['native_completed'] == 4
    ), native_summary
    warning = (directory / 'unlisted-native.stderr').read_text()
    assert warning.count('outside provided_application_images') == 1, warning
    provided = json.loads(
        execute(
            'analyze-first-only',
            [
                args.analyze_coordinator,
                bounded / 'application.json',
                '--analyzer',
                args.analyze,
            ],
        )
    )
    assert provided['scope'] == 'provided_application_images', provided
    assert provided['whole_application_coverage'] == 'not_established', provided
    assert provided['coverage_complete'] and len(provided['images']) == 1, provided
    # A fresh analyzer invocation needs only the index and its capture files.
    inputs.rename(directory / 'inputs.hidden')
    (output / 'images').rename(output / 'images.hidden')
    (output / 'application.bin').rename(output / 'application.bin.hidden')
    replay = execute(
        'replay', [args.analyze_coordinator, index_path, '--analyzer', args.analyze]
    )
    assert (
        replay == analyzed
    ), 'offline replay changed after hiding preparation artifacts'


if __name__ == '__main__':
    main()
