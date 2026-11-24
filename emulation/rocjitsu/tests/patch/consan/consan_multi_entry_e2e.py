# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whole-image A/B/C/A outputs, native application dispatch, and capture replay."""

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
    directory = Path(tempfile.mkdtemp(prefix='multi-entry-', dir=args.root))
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
    launch_options = ['--application-workgroups=1'] if args.application else []
    baseline_text = execute('original-baseline', [*app, 'baseline', *launch_options])
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
    if args.native_application:
        active = environment | {
            'HSA_TOOLS_LIB': str(args.plugin),
            'RJ_CONSAN_PLUGIN_PATH': str(args.plugin),
            'RJ_CONSAN_APPLICATION_PROFILE': '1',
            'RJ_CONSAN_NATIVE_IMAGES': str(original),
        }
        application = execute(
            'native-application', [*app, 'native-application'], env=active
        )
        ledger = [
            json.loads(line)
            for line in application.splitlines()
            if line.startswith('{')
        ]
        expected = [row | {'native': True} for row in baseline[:-1]]
        assert ledger[:-1] == expected, 'native original ABI/output changed'
        summary = ledger[-1]
        assert summary['app_status'] == 0 and summary['finish_status'] is None
        assert summary['initializations'] == summary['shutdowns'] == 2
        assert summary['queues_created'] == summary['queues_destroyed'] == 17
        assert summary['loads'] == 1 and summary['completed'] == 4
        assert summary['native_submitted'] == summary['native_completed'] == 4
        for counter in ('submitted', 'forwarded', 'copied', 'written', 'reused'):
            assert summary[counter] == 0, (counter, summary)
        assert not (directory / 'capture.bin').exists()
        assert not (directory / 'artifact').exists()
        stderr = (directory / 'native-application.stderr').read_text()
        assert 'ConSan unloaded without finish' not in stderr, stderr
        (directory / 'native-ledger.json').write_text(json.dumps(ledger, indent=2))
        print(directory)
        return
    artifact = directory / 'artifact'
    preparation = execute(
        'prepare',
        [
            args.prepare,
            original,
            'gfx1250',
            artifact,
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
        'HSA_TOOLS_LIB': str(args.plugin),
        'RJ_CONSAN_PLUGIN_PATH': str(args.plugin),
        'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
        'RJ_CONSAN_CAPTURE_PATH': str(capture),
        'RJ_CONSAN_SELECTION_PROFILE': '0' * 64 if args.wrong_profile else profile,
    }
    if args.application:
        active['RJ_CONSAN_APPLICATION_PROFILE'] = '1'
    application = execute(
        'instrumented',
        [*app, 'instrumented', *launch_options],
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
    assert summary['forwarded'] == (0 if args.application else 3), summary
    assert summary['reused'] == 2, summary
    assert summary['native_submitted'] == summary['native_completed'] == 1, summary
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
    records = 0
    for index, execution in enumerate(executions):
        assert len(execution['rows']) == 2
        for wave, row in enumerate(execution['rows']):
            assert (
                row['status'] == (7 if args.variant == 'loop' else 3)
                and row['loss'] == (2 if args.variant == 'loop' else 0)
                and row['final_sequence'] == (4 if args.variant == 'loop' else 2)
            )
            assert row['serial'] == index + 1 and row['wave'] == wave
            phases = {'no-sync': 0, 'barrier': 1, 'loop': 2}[args.variant]
            assert row['final_phase'] == phases
            assert row['sync_loss'] == row['counter_exhausted'] == 0
            assert len(row['completions']) == phases
            for occurrence, completion in enumerate(row['completions'], 1):
                assert completion['group'] == 0 and completion['participants'] == 2
                assert completion['occurrence'] == completion['phase'] == occurrence
                assert completion['sequence'] == 2 * occurrence - 1
            assert len(row['records']) == 2
            expected_offset = (128 if index == 1 else 0) + 4 * wave
            assert {record['offset'] for record in row['records']} == {expected_offset}
            records += len(row['records'])
    assert records == 12
    native_ledger = {
        'scope': 'application native accounting; native dispatch is absent from selected capture',
        'original_sha256': hashes[str(args.fixture)],
        'entry_key': native_keys[0],
        'dispatch': ledger[2],
    }
    (directory / 'native-ledger.json').write_text(json.dumps(native_ledger, indent=2))
    before = json.loads(execute('analysis-before-removal', [args.analyze, capture]))
    assert [row['entry'] for row in before['executions']] == ['mk_a', 'mk_b', 'mk_a']
    assert [row['valid_observations'] for row in before['executions']] == [4, 4, 4]
    assert before['coverage_complete'] == (args.variant != 'loop')
    assert before['cross_wave_pairs'] == before['collision_groups'] == 0
    assert [row['lost_occurrences'] for row in before['executions']] == (
        [4, 4, 4] if args.variant == 'loop' else [0, 0, 0]
    )
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
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--wrong-profile', action='store_true')
    profile.add_argument('--native-application', action='store_true')
    profile.add_argument('--application', action='store_true')
    parser.add_argument(
        '--variant', choices=('no-sync', 'barrier', 'loop'), default='no-sync'
    )
    run(parser.parse_args())


if __name__ == '__main__':
    main()
