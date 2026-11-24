# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execute original and ConSan-instrumented workgroup oracles for one target/wave."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

CONFIGS = {
    'gfx942': 'gfx942_cdna3.json',
    'gfx950': 'gfx950_mi350x.json',
    'gfx1100': 'gfx1100_w7900.json',
    'gfx1201': 'gfx1201_r9700.json',
    'gfx1250': 'gfx1250_mi455x.json',
}


def run(args):
    cache = (args.build / 'CMakeCache.txt').read_text()
    assert f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}\n' in cache
    tools = {
        'prepare': args.build / 'tools/rj_consan_prepare',
        'runtime': args.build / 'librocjitsu.so',
        'runner': args.build / 'tests/patch/consan/consan_hsa_runner',
        'analyze': args.build / 'tools/rj_consan_analyze',
        'inspect': args.build / 'tests/consan_capture_inspect',
        'plugin': args.build
        / 'lib/rocjitsu/src/rocjitsu/hooks/consan/libhsa_consan.so',
        'launcher': args.build / 'tools/rocjitsu/rocjitsu',
        'fixture': args.build
        / f'tests/patch/consan/fixtures/targets/{args.target}/wave{args.wave_size}/group-1.hsaco',
    }
    assert all(path.resolve().is_relative_to(args.build) for path in tools.values())
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(
            prefix=f'application-{args.target}-w{args.wave_size}-', dir=args.root
        )
    )
    print(directory, flush=True)
    (directory / 'inputs.json').write_text(
        json.dumps(
            {
                name: {
                    'path': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in tools.items()
            },
            indent=2,
        )
        + '\n'
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith('RJ_CONSAN_')
        and name not in ('HSA_TOOLS_LIB', 'HSA_TOOLS_DISABLE_REGISTER', 'LD_PRELOAD')
    }
    environment.update(HSA_HOTSWAP_DISABLE='1', HSA_HOTSWAP_ENABLE='0')
    commands = []

    def execute(label, command, bindings=None):
        argv = [str(part) for part in command]
        supervised = [
            'setpriv',
            '--pdeathsig',
            'TERM',
            str(args.source / 'tests/corpus/corpus-process-supervisor.sh'),
            'timeout',
            '--foreground',
            '--kill-after=5s',
            '60s',
            *argv,
        ]
        result = subprocess.run(
            supervised,
            env=environment | (bindings or {}),
            capture_output=True,
            text=True,
            check=False,
        )
        (directory / f'{label}.stdout').write_text(result.stdout)
        (directory / f'{label}.stderr').write_text(result.stderr)
        commands.append({'stage': label, 'argv': argv, 'exit': result.returncode})
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == 0, (label, result.returncode, result.stderr)
        return result.stdout

    original = directory / 'original.hsaco'
    shutil.copyfile(tools['fixture'], original)
    threads = 2 * args.wave_size + 1
    groups = 3
    launches = 3
    app = [
        tools['launcher'],
        '--config',
        args.source / 'configs' / CONFIGS[args.target],
        '--cpu-thread-budget',
        '4',
        '--',
        tools['runner'],
        original,
        'lds_producer_consumer',
        str(launches),
        'grouped',
        '--application-dispatch',
        '--target',
        args.target,
        '--wave-size',
        str(args.wave_size),
        '--threads',
        str(threads),
        '--workgroups',
        str(groups),
    ]
    baseline = execute('baseline', [*app, '--baseline'])
    profiles = ('execution-trace', 'lds') if args.profile == 'both' else (args.profile,)
    results = {}
    for profile_name in profiles:
        artifact = directory / profile_name
        preparation = execute(
            f'{profile_name}-prepare',
            [
                tools['prepare'],
                original,
                args.target,
                artifact,
                '--profile',
                profile_name,
                '--entry',
                'lds_producer_consumer',
            ],
        )
        selection = next(
            token.split('=', 1)[1]
            for token in preparation.split()
            if token.startswith('selection_profile=')
        )
        capture = directory / f'{profile_name}.bin'
        instrumented = execute(
            f'{profile_name}-instrumented',
            app,
            {
                'HSA_TOOLS_LIB': str(tools['plugin']),
                'RJ_CONSAN_PLUGIN_PATH': str(tools['plugin']),
                'RJ_CONSAN_APPLICATION_PROFILE': '1',
                'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
                'RJ_CONSAN_SELECTION_PROFILE': selection,
                'RJ_CONSAN_CAPTURE_PATH': str(capture),
            },
        )
        oracle = lambda text: [
            line for line in text.splitlines() if line.startswith('launch=')
        ]
        assert (
            oracle(instrumented) == oracle(baseline)
            and len(oracle(baseline)) == launches
        )
        for output in (baseline, instrumented):
            summary = json.loads(
                next(line for line in output.splitlines() if line.startswith('{'))
            )
            assert summary['app_status'] == summary['finish_status'] == 0
            assert summary['completed'] == launches
        before = execute(f'{profile_name}-analyze', [tools['analyze'], capture])
        analysis = json.loads(before)
        assert analysis['selection_profile'] == selection
        assert (
            analysis['original_sha256']
            == hashlib.sha256(original.read_bytes()).hexdigest()
        )
        assert len(analysis['executions']) == launches
        expected_rows = groups * 3
        if profile_name == 'execution-trace':
            assert analysis['scope'] == 'execution_trace'
            assert (
                analysis['lds_analysis'] == 'not_performed'
                and analysis['transport_complete']
            )
            for key in (
                'expected_rows',
                'completed_rows',
                'entry_events',
                'exit_events',
            ):
                assert analysis[key] == launches * expected_rows, (key, analysis)
        else:
            assert (
                analysis['target'] == int(args.target[3:])
                and analysis['wave_size'] == args.wave_size
            )
            assert (
                analysis['scope'].startswith('native_lds_')
                and analysis['coverage_complete']
            )
            assert analysis['cross_wave_pairs'] == analysis['collision_groups'] == 0
            for execution in analysis['executions']:
                assert (
                    execution['workgroup_count'] == groups
                    and execution['waves_per_workgroup'] == 3
                )
                assert execution['valid_observations'] == 2 * expected_rows
                assert (
                    execution['coverage_complete']
                    and execution['integrity_events'] == 0
                )
                assert not execution['issues']
        inspected = json.loads(
            execute(f'{profile_name}-inspect', [tools['inspect'], capture])
        )
        assert inspected['execution_conditions'] == 8
        for serial, execution in enumerate(inspected['executions'], 1):
            assert (
                execution['serial'] == serial
                and len(execution['rows']) == expected_rows
            )
            for index, row in enumerate(execution['rows']):
                assert row['serial'] == serial
                assert row['workgroup_x'] == index // 3 and row['status'] == 3
                if profile_name == 'execution-trace':
                    assert row['wave'] == index
                    assert row['local_wave'] == index % 3
                    assert row['entries'] == row['exits'] == 1
                else:
                    assert row['wave'] == index % 3
                    assert row['loss'] == row['sync_loss'] == 0
                    assert row['counter_exhausted'] == 0
                    assert row['final_sequence'] == 2 and row['final_phase'] == 0
                    assert len(row['records']) == 2 and not row['completions']
                    for site, record in enumerate(row['records']):
                        assert record['site'] == site
                        assert record['sequence'] == site + 1
                        assert record['offset'] == (index % 3) * 4
                        assert record['width'] == 4 and record['mask'] == 1
        hidden = original.with_suffix('.hidden')
        original.rename(hidden)
        shutil.rmtree(artifact)
        try:
            assert not original.exists() and not artifact.exists()
            assert (
                execute(f'{profile_name}-fresh-replay', [tools['analyze'], capture])
                == before
            )
        finally:
            hidden.rename(original)
        results[profile_name] = analysis
    (directory / 'result.json').write_text(
        json.dumps(
            {
                'target': args.target,
                'wave_size': args.wave_size,
                'passed': True,
                'analysis': results,
                'fresh_replay_identical': True,
            },
            indent=2,
        )
        + '\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'root'):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--target', choices=tuple(CONFIGS), required=True)
    parser.add_argument('--wave-size', type=int, choices=(32, 64), required=True)
    parser.add_argument(
        '--profile', choices=('both', 'lds', 'execution-trace'), default='both'
    )
    run(parser.parse_args())


if __name__ == '__main__':
    main()
