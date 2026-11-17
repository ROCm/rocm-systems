# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-image execution trace with native mixing and artifact-free replay."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='execution-trace-', dir=args.root))
    commands = []
    environment = dict(os.environ)
    for name in (
        'HSA_TOOLS_LIB',
        'RJ_CONSAN_PLUGIN_PATH',
        'RJ_CONSAN_ARTIFACT_DIR',
        'RJ_CONSAN_CAPTURE_PATH',
        'RJ_CONSAN_SELECTION_PROFILE',
        'RJ_CONSAN_APPLICATION_PROFILE',
        'RJ_CONSAN_NATIVE_IMAGES',
    ):
        environment.pop(name, None)

    def execute(label, command, expected=0, env=None):
        argv = [str(part) for part in command]
        result = subprocess.run(
            argv,
            env=env or environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        commands.append({'label': label, 'argv': argv, 'exit': result.returncode})
        (directory / (label + '.stdout')).write_text(result.stdout)
        (directory / (label + '.stderr')).write_text(result.stderr)
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2))
        assert (
            result.returncode == expected
        ), f'{label}: expected {expected}, got {result.returncode}\n{result.stderr}'
        return result.stdout

    # Every executable and fixture must belong to this source checkout's build.
    def owning_build(path):
        for parent in path.parents:
            if (parent / 'CMakeCache.txt').is_file():
                return parent
        raise AssertionError(f'no owning build: {path}')

    tools = (
        args.prepare,
        args.runner,
        args.inspect,
        args.analyze,
        args.plugin,
        args.launcher,
        args.fixture,
    )
    roots = {owning_build(path) for path in tools}
    assert len(roots) == 1, 'mixed-checkout binaries'
    cache = (roots.pop() / 'CMakeCache.txt').read_text()
    assert f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}' in cache
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tools
    }
    (directory / 'inputs.json').write_text(json.dumps(hashes, indent=2))
    original = directory / 'original.hsaco'
    shutil.copyfile(args.fixture, original)
    app = [args.launcher, '--config', args.config, '--', args.runner, original]
    baseline_text = execute(
        'original-baseline',
        [*app, 'baseline', '--trace-workgroups=' + str(args.workgroups)],
    )
    baseline = [
        json.loads(line) for line in baseline_text.splitlines() if line.startswith('{')
    ]
    assert baseline[-1]['loads'] == 1 and baseline[-1]['completed'] == 4
    assert [row['entry'] for row in baseline[:-1]] == [
        'mk_a.kd',
        'mk_b.kd',
        'mk_c.kd',
        'mk_a.kd',
    ]
    artifact = directory / 'artifact'
    preparation = execute(
        'prepare',
        [
            args.prepare,
            original,
            'gfx1250',
            artifact,
            '--profile',
            'execution-trace',
            '--entry',
            'mk_a',
            '--entry',
            'mk_b',
        ],
    )
    profile = next(
        token.split('=', 1)[1]
        for token in preparation.split()
        if token.startswith('selection_profile=')
    )
    assert len(profile) == 64 and all(char in '0123456789abcdef' for char in profile)
    capture = directory / 'capture.bin'
    active = environment | {
        'RJ_CONSAN_APPLICATION_PROFILE': '1',
        'HSA_TOOLS_LIB': str(args.plugin),
        'RJ_CONSAN_PLUGIN_PATH': str(args.plugin),
        'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
        'RJ_CONSAN_CAPTURE_PATH': str(capture),
        'RJ_CONSAN_SELECTION_PROFILE': '0' * 64 if args.wrong_profile else profile,
    }
    application = execute(
        'instrumented',
        [*app, 'instrumented', '--trace-workgroups=' + str(args.workgroups)],
        expected=20 if args.wrong_profile else 0,
        env=active,
    )
    if args.wrong_profile:
        assert not capture.exists(), 'wrong selection profile published success capture'
        print(directory)
        return
    ledger = [
        json.loads(line) for line in application.splitlines() if line.startswith('{')
    ]
    assert ledger[:-1] == baseline[:-1], 'original ABI/output/completion changed'
    summary = ledger[-1]
    assert summary['app_status'] == summary['finish_status'] == 0
    assert summary['loads'] == 1 and summary['completed'] == 4
    for counter in ('submitted', 'copied', 'written'):
        assert summary[counter] == 3, (counter, summary)
    assert summary['forwarded'] == 0, summary
    assert summary['reused'] == 2, summary
    assert summary['native_submitted'] == summary['native_completed'] == 1, summary
    if args.workgroups == 16:
        # Compare the same real probe path in format 3 with unit clusters.
        unit_capture = directory / 'unit-cluster.bin'
        unit_text = execute(
            'unit-cluster',
            [*app, 'instrumented', '--trace-workgroups=16', '--cluster-size=1'],
            env=active | {'RJ_CONSAN_CAPTURE_PATH': str(unit_capture)},
        )
        unit = [
            json.loads(line) for line in unit_text.splitlines() if line.startswith('{')
        ]
        assert unit[:-1] == baseline[:-1]
        assert unit[-1]['app_status'] == unit[-1]['finish_status'] == 0
        unit_analysis = json.loads(
            execute('unit-cluster-analysis', [args.analyze, unit_capture])
        )
        assert unit_analysis['transport_complete']
        assert unit_analysis['expected_rows'] == unit_analysis['completed_rows'] == 96

        clustered = [*app, 'baseline', '--trace-workgroups=16', '--cluster-size=2']
        clustered_baseline = execute('native-cluster-baseline', clustered)
        native_capture = directory / 'native-cluster.bin'
        native_environment = active | {
            'RJ_CONSAN_NATIVE_IMAGES': str(original),
            'RJ_CONSAN_CAPTURE_PATH': str(native_capture),
        }
        native_environment.pop('RJ_CONSAN_ARTIFACT_DIR')
        native_environment.pop('RJ_CONSAN_SELECTION_PROFILE')
        native_text = execute(
            'native-cluster',
            [*app, 'instrumented', '--trace-workgroups=16', '--cluster-size=2'],
            env=native_environment,
        )
        native = [
            json.loads(line)
            for line in native_text.splitlines()
            if line.startswith('{')
        ]
        original_cluster = [
            json.loads(line)
            for line in clustered_baseline.splitlines()
            if line.startswith('{')
        ]
        assert native[:-1] == original_cluster[:-1] == baseline[:-1]
        assert native[-1]['app_status'] == native[-1]['finish_status'] == 0
        assert native[-1]['completed'] == native[-1]['native_completed'] == 4
        assert native[-1]['native_submitted'] == 4 and native[-1]['submitted'] == 0

        rejected_capture = directory / 'rejected-cluster.bin'
        rejected_text = execute(
            'selected-nonunit-cluster',
            [*app, 'instrumented', '--trace-workgroups=16', '--cluster-size=2'],
            expected=21,
            env=active | {'RJ_CONSAN_CAPTURE_PATH': str(rejected_capture)},
        )
        rejected = [
            json.loads(line)
            for line in rejected_text.splitlines()
            if line.startswith('{')
        ]
        assert len(rejected) == 1
        assert rejected[0]['app_status'] == 21 and rejected[0]['finish_status'] != 0
        assert rejected[0]['submitted'] == rejected[0]['completed'] == 0
        assert not rejected_capture.exists()

    inspected = json.loads(execute('inspect', [args.inspect, capture]))
    assert len(inspected['entries']) == 3
    selected_keys = [
        row['entry_key'] for row in inspected['entries'] if row['selected']
    ]
    native_keys = [
        row['entry_key'] for row in inspected['entries'] if not row['selected']
    ]
    assert len(selected_keys) == 2 and len(native_keys) == 1
    executions = inspected['executions']
    assert [row['entry_key'] for row in executions] == [
        selected_keys[0],
        selected_keys[1],
        selected_keys[0],
    ]
    assert [row['serial'] for row in executions] == [1, 2, 3]
    assert [row['generation'] for row in executions] == [1, 2, 3]
    assert len({row['slot'] for row in executions}) == 1
    for execution in executions:
        assert len(execution['rows']) == args.workgroups * 2
        for index, row in enumerate(execution['rows']):
            assert row['wave'] == index
            assert row['workgroup_x'] == index // 2
            assert row['local_wave'] == index % 2
            assert row['serial'] == execution['serial']
            assert row['status'] == 3 and row['entries'] == row['exits'] == 1
    native_ledger = {
        'scope': 'application native accounting; native dispatch is absent from selected capture',
        'original_sha256': hashes[str(args.fixture)],
        'entry_key': native_keys[0],
        'dispatch': ledger[2],
    }
    (directory / 'native-ledger.json').write_text(json.dumps(native_ledger, indent=2))
    before = json.loads(execute('analysis-before-removal', [args.analyze, capture]))
    assert [row['entry'] for row in before['executions']] == ['mk_a', 'mk_b', 'mk_a']
    expected_rows = args.workgroups * 2
    assert (
        before['scope'] == 'execution_trace'
        and before['lds_analysis'] == 'not_performed'
    )
    assert before['transport_complete'], before
    assert before['expected_rows'] == before['completed_rows'] == 3 * expected_rows
    assert before['entry_events'] == before['exit_events'] == 3 * expected_rows
    for execution in before['executions']:
        assert (
            execution['expected_rows'] == execution['completed_rows'] == expected_rows
        )
        assert execution['grid'] == [args.workgroups * 64, 1, 1]
        assert execution['workgroup'] == [64, 1, 1]
        assert execution['integrity_events'] == 0 and not execution['issues']
    for forbidden in (
        'verdict',
        'coverage_complete',
        'cross_wave_pairs',
        'collision_groups',
        'findings',
    ):
        assert forbidden not in before
    original.unlink()
    shutil.rmtree(artifact)
    assert not original.exists() and not artifact.exists()
    after = json.loads(
        execute('fresh-analysis-without-artifacts', [args.analyze, capture])
    )
    assert after == before
    assert (
        json.loads(execute('fresh-inspect-without-artifacts', [args.inspect, capture]))
        == inspected
    )
    print(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        'prepare',
        'runner',
        'inspect',
        'analyze',
        'plugin',
        'launcher',
        'fixture',
        'config',
        'root',
        'source',
    ):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--wrong-profile', action='store_true')
    parser.add_argument('--workgroups', type=int, choices=(1, 16, 512), required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
