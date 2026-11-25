# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run pinned upstream toy Llama through native and ConSan application wrappers."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

SUITE_REVISION = '08a7187a92d6496e50c7f2f7fb0251711990ed09'
IREE_VERSION = '3.13.0rc20261006'
INPUTS = {
    'test_llama.py': 'c8104ec09bce75f6d41c5af0bf31048fe1950ee68cfced0ec84c813b69faae06',
    'README.md': '050751056c3bf3fef3d7e948e00659b970765f3554180144e8a6ead7031995a2',
    'assets/toy_llama.mlir': '6eb3170f11fcd40d561e66214a1ce2d49277e8953950d96afc8ea0c9efd3a680',
    'assets/toy_llama.irpa': '37f90b4754197b1e732c43010e3979223136d91f8c843998aa376afc8c997d09',
}
SCATTER = 'prefill_bs1$async_dispatch_8_scatter_Dx32x4x32xf16_dispatch_tensor_store'
ELEMENTWISE = 'prefill_bs1$async_dispatch_9_elementwise_D_i64'
LDS_MATMUL = 'prefill_bs1$async_dispatch_3_matmul_Dx128x256_f16xf16xf32'
SELECTED = (SCATTER, ELEMENTWISE)
SEQUENCE = (SCATTER, ELEMENTWISE, SCATTER) * 3
CASES = ('trace', 'lds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_suite(suite):
    for name, expected in INPUTS.items():
        require(digest(suite / name) == expected, f'upstream input differs: {name}')


def clean_environment(environment):
    result = {
        key: value
        for key, value in environment.items()
        if not key.startswith('RJ_CONSAN_')
        and key
        not in (
            'HSA_TOOLS_LIB',
            'HSA_TOOLS_DISABLE_REGISTER',
            'LD_PRELOAD',
            'LOADER_OPTIONS_APPEND',
            'GPU_DUMP_CODE_OBJECT',
        )
    }
    result.update(HSA_HOTSWAP_DISABLE='1', HSA_HOTSWAP_ENABLE='0')
    return result


def validate_trace(analysis):
    require(
        analysis['scope'] == 'execution_trace'
        and analysis['lds_analysis'] == 'not_performed'
        and analysis['transport_complete'] is True,
        'incomplete or incorrect analysis scope',
    )
    require(
        Counter(row['disposition'] for row in analysis['entries'])
        == Counter(selected=2, native_uncovered=53),
        'whole-image inventory differs',
    )
    selected = {
        row['entry']: row['entry_key']
        for row in analysis['entries']
        if row['disposition'] == 'selected'
    }
    require(set(selected) == set(SELECTED), 'selected entries differ')
    require(len(set(selected.values())) == 2, 'selected entry identities alias')
    executions = analysis['executions']
    require(
        tuple(row['entry'] for row in executions) == SEQUENCE,
        'selected execution order differs',
    )
    for serial, row in enumerate(executions, 1):
        groups = 16 if row['entry'] == SCATTER else 1
        require(
            row['serial'] == serial
            and row['entry_key'] == selected[row['entry']]
            and row['grid'] == [groups * 32, 1, 1]
            and row['workgroup'] == [32, 1, 1],
            'selected execution identity or geometry differs',
        )
        require(
            all(
                row[key] == groups
                for key in (
                    'expected_rows',
                    'completed_rows',
                    'entry_events',
                    'exit_events',
                )
            )
            and row['integrity_events'] == 0
            and not row['issues'],
            'selected entry/exit observations differ',
        )
    require(
        all(
            analysis[key] == 99
            for key in (
                'expected_rows',
                'completed_rows',
                'entry_events',
                'exit_events',
            )
        ),
        'total entry/exit observations differ',
    )
    return selected


def validate_lds(analysis, exit_code=None):
    require(
        analysis['lds_analysis'] == 'performed'
        and analysis['scope'] == 'native_lds_multi_range_epoch_workgroup_barriers'
        and analysis['execution_conditions'] == 8,
        'LDS analysis was not performed with application completion evidence',
    )
    require(
        Counter(row['disposition'] for row in analysis['entries'])
        == Counter(selected=1, native_uncovered=54),
        'LDS image inventory differs',
    )
    require(
        analysis['verdict'] in ('findings', 'no_observed_findings'),
        'invalid LDS verdict',
    )
    if exit_code is not None:
        require(
            exit_code == int(analysis['verdict'] == 'findings'),
            'LDS analyzer status contradicts its verdict',
        )
    selected = {
        row['entry']: row['entry_key']
        for row in analysis['entries']
        if row['disposition'] == 'selected'
    }
    require(set(selected) == {LDS_MATMUL}, 'LDS selected entry differs')
    executions = analysis['executions']
    require(len(executions) == 6, 'LDS selected execution count differs')
    for serial, row in enumerate(executions, 1):
        require(
            row['serial'] == serial
            and row['entry'] == LDS_MATMUL
            and row['entry_key'] == selected[LDS_MATMUL]
            and row['workgroup_count'] == 2
            and row['waves_per_workgroup'] == 4,
            'LDS selected execution identity or geometry differs',
        )
        require(
            row['valid_observations'] == 4608
            and row['rejected_records'] == 0
            and row['integrity_events'] == 0
            and row['counter_exhausted_waves'] == 0,
            'LDS observation count or integrity differs',
        )
        require(
            all(
                issue['reason']
                in ('range_reservation_exhausted', 'synchronization_journal_exhausted')
                for issue in row['issues']
            ),
            'unexpected LDS integrity issue',
        )
        # The pinned kernel has 72 physical dual-range sites, four retained
        # bases per site and eight waves. Exceeding the base-group capacity
        # increments loss once per instruction occurrence in each wave.
        require(
            row['lost_occurrences'] == 576
            and row['address_group_exhaustion'] == 576
            and row['synchronization_loss'] == 0,
            'pinned LDS loss accounting differs',
        )
        loss = (
            row['lost_occurrences']
            + row['synchronization_loss']
            + row['address_group_exhaustion']
        )
        require(
            row['coverage_complete'] == (loss == 0),
            'LDS coverage contradicts loss counters',
        )
    require(
        analysis['coverage_complete']
        == all(row['coverage_complete'] for row in executions),
        'aggregate LDS coverage contradicts execution evidence',
    )
    return selected


class Commands:
    def __init__(self, directory, source, environment, timeout):
        self.directory = directory
        self.source = source
        self.environment = environment
        self.deadline = time.monotonic() + timeout
        self.records = []

    def run(self, name, command, environment=None, cwd=None, *, allowed_exits=(0,)):
        remaining = math.ceil(self.deadline - time.monotonic())
        require(remaining > 0, 'prefill test deadline exceeded')
        argv = [str(part) for part in command]
        supervised = [
            'setpriv',
            '--pdeathsig',
            'TERM',
            str(self.source / 'tests/corpus/corpus-process-supervisor.sh'),
            'timeout',
            '--foreground',
            '--signal=TERM',
            '--kill-after=5s',
            f'{remaining}s',
            *argv,
        ]
        start = time.monotonic()
        with (self.directory / f'{name}.stdout').open('w') as stdout, (
            self.directory / f'{name}.stderr'
        ).open('w') as stderr:
            result = subprocess.run(
                supervised,
                cwd=cwd,
                env=environment or self.environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        self.records.append(
            {
                'stage': name,
                'argv': argv,
                'supervised_argv': supervised,
                'cwd': str(cwd) if cwd else os.getcwd(),
                'exit': result.returncode,
                'seconds': time.monotonic() - start,
            }
        )
        (self.directory / 'commands.json').write_text(
            json.dumps(self.records, indent=2) + '\n'
        )
        require(
            result.returncode in allowed_exits,
            f'{name}: exit {result.returncode}; see {self.directory}',
        )
        return (self.directory / f'{name}.stdout').read_text()


def offline_replay(
    commands,
    analyzer,
    capture,
    paths,
    before,
    stage='fresh-analysis-without-inputs',
    *,
    allowed_exits=(0,),
    expected_exit=None,
):
    # Test-owned copies only: no upstream checkout or shared inputs are renamed.
    hidden = [(path, path.with_name(path.name + '.hidden')) for path in paths]
    require(
        not any(unavailable.exists() for _, unavailable in hidden),
        'replay scratch paths already exist',
    )
    try:
        for original, unavailable in hidden:
            original.rename(unavailable)
        require(
            not any(path.exists() for path in paths),
            'replay input paths are still available',
        )
        after = commands.run(stage, [analyzer, capture], allowed_exits=allowed_exits)
        if expected_exit is not None:
            require(
                commands.records[-1]['exit'] == expected_exit,
                'offline analyzer status differs',
            )
        require(after == before, 'independent offline replay differs')
    finally:
        for original, unavailable in reversed(hidden):
            if unavailable.exists():
                unavailable.rename(original)


def validate_environment(args):
    for package in ('iree-base-runtime', 'iree-base-compiler'):
        require(
            importlib.metadata.version(package) == IREE_VERSION,
            f'incorrect {package} version',
        )
    cache = (args.build / 'CMakeCache.txt').read_text()
    require(
        f'CMAKE_HOME_DIRECTORY:INTERNAL={args.source}\n' in cache,
        'mixed-checkout build',
    )
    tools = {
        'launcher': args.build / 'tools/rocjitsu/rocjitsu',
        'prepare': args.build / 'tools/rj_consan_prepare',
        'analyze': args.build / 'tools/rj_consan_analyze',
        'plugin': args.build
        / 'lib/rocjitsu/src/rocjitsu/hooks/consan/libhsa_consan.so',
    }
    require(
        all(path.resolve().is_relative_to(args.build) for path in tools.values()),
        'tools resolve outside the owning build',
    )
    return tools


def application(args, commands, tools, inputs, name, bindings, *, with_plugin=False):
    output = commands.directory / name
    output.mkdir()
    commands.run(
        name,
        [
            tools['launcher'],
            '--config',
            args.source / 'configs/gfx1250_mi455x.json',
            '--cpu-thread-budget',
            args.cpu_thread_budget,
            '--',
            sys.executable,
            Path(__file__).with_name('consan_iree_prefill_driver.py'),
            '--suite',
            inputs / 'llama3.1',
            '--vmfb',
            inputs / 'model.vmfb',
            '--output',
            output,
            *(['--plugin', tools['plugin']] if with_plugin else []),
        ],
        commands.environment | bindings,
        cwd=inputs / 'images' if name == 'baseline' else None,
    )
    oracle = json.loads((output / 'oracle.json').read_text())
    require(
        oracle['passed']
        and oracle['shape'] == [1, 16, 256]
        and oracle['finite']
        and oracle['dtype'] == 'float32',
        'original numerical oracle failed',
    )
    if with_plugin:
        require(oracle['finish'] == 0, 'capture finish failed')
    return output, oracle


def tool_identities(tools):
    return {
        name: {'path': str(path), 'sha256': digest(path)}
        for name, path in tools.items()
    }


def setup_fixture(args, tools):
    validate_suite(args.suite)
    args.root.mkdir(parents=True, exist_ok=True)
    if args.fixture is None:
        directory = Path(tempfile.mkdtemp(prefix='toy-llama-', dir=args.root))
    else:
        directory = args.fixture
        directory.mkdir(parents=True)
    print(directory, flush=True)
    identities = tool_identities(tools)
    (directory / 'inputs.json').write_text(
        json.dumps(
            {'suite_revision': SUITE_REVISION, 'inputs': INPUTS, 'tools': identities},
            indent=2,
        )
        + '\n'
    )
    inputs = directory / 'original'
    suite = inputs / 'llama3.1'
    for name in INPUTS:
        destination = suite / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.suite / name, destination)
    environment = clean_environment(os.environ) | {'HIP_TARGET': 'gfx1250'}
    commands = Commands(directory, args.source, environment, args.timeout)
    commands.run(
        'compile',
        [
            Path(sys.executable).parent / 'iree-compile',
            suite / 'assets/toy_llama.mlir',
            '--iree-hal-target-device=hip[0]',
            '--iree-rocm-target=gfx1250',
            '-o',
            inputs / 'model.vmfb',
        ],
    )
    images = inputs / 'images'
    images.mkdir()
    baseline, baseline_oracle = application(
        args,
        commands,
        tools,
        inputs,
        'baseline',
        {'LOADER_OPTIONS_APPEND': '-dump-code'},
    )
    unique = {digest(path): path for path in images.glob('amdcode*.hsaco')}
    require(len(unique) == 2, 'expected the pinned model and native helper images')
    models = [
        path
        for path in unique.values()
        if all((name + '.kd').encode() in path.read_bytes() for name in SELECTED)
    ]
    require(len(models) == 1, 'model image is absent or ambiguous')
    model = models[0]
    helper = next(path for path in unique.values() if path != model)
    fixture = {
        'schema_version': 1,
        'source': str(args.source),
        'build': str(args.build),
        'suite_revision': SUITE_REVISION,
        'tools': identities,
        'cpu_thread_budget': args.cpu_thread_budget,
        'timeout': args.timeout,
        'setup_seconds': sum(row['seconds'] for row in commands.records),
        'model': str(model.relative_to(inputs)),
        'helper': str(helper.relative_to(inputs)),
        'baseline': baseline_oracle,
        'files': {
            str(path.relative_to(directory)): digest(path)
            for path in [
                *(inputs / 'llama3.1' / name for name in INPUTS),
                inputs / 'model.vmfb',
                model,
                helper,
                baseline / 'logits.npy',
            ]
        },
    }
    require(fixture['setup_seconds'] < args.timeout, 'prefill setup exhausted budget')
    # Publication certifies that compilation and the numerical baseline succeeded.
    (directory / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    return directory


def load_fixture(args, tools):
    require(args.fixture is not None, 'trace and LDS cases require --fixture')
    directory = args.fixture
    fixture = json.loads((directory / 'fixture.json').read_text())
    require(
        fixture['schema_version'] == 1
        and fixture['source'] == str(args.source)
        and fixture['build'] == str(args.build)
        and fixture['suite_revision'] == SUITE_REVISION
        and fixture['tools'] == tool_identities(tools)
        and fixture['cpu_thread_budget'] == args.cpu_thread_budget,
        'shared prefill fixture identity differs',
    )
    for name, expected in fixture['files'].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory), 'fixture path escapes root')
        require(digest(path) == expected, f'shared fixture input differs: {name}')
    inputs = directory / 'original'
    for name in ('model', 'helper'):
        require(
            (inputs / fixture[name]).resolve().is_relative_to(inputs),
            'fixture image escapes original inputs',
        )
    return fixture


def case_timeout(directory, fixture, case):
    remaining = fixture['timeout'] - fixture['setup_seconds']
    pending = 1
    for other in CASES:
        if other == case:
            continue
        other_directory = directory / other
        receipt = other_directory / 'result.json'
        require(
            not other_directory.exists() or receipt.is_file(),
            'prefill cases must consume the shared fixture sequentially',
        )
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            seconds = result['stage_elapsed_seconds']
            require(
                result['case'] == other
                and isinstance(seconds, (int, float))
                and seconds >= 0,
                'invalid prefill case budget receipt',
            )
            remaining -= seconds
        else:
            pending += 1
    # Reserve the other case's opportunity even when this case times out. The
    # second case may reclaim unused time from a completed or failed first case.
    require(remaining > 0, 'prefill shared execution budget exhausted')
    return remaining / pending


def selection_profile(preparation):
    profiles = [
        token.split('=', 1)[1]
        for token in preparation.split()
        if token.startswith('selection_profile=')
    ]
    require(len(profiles) == 1 and len(profiles[0]) == 64, 'missing selection identity')
    return profiles[0]


def run_case(args, tools, fixture, case):
    directory = args.fixture / case
    timeout = min(args.timeout, case_timeout(args.fixture, fixture, case))
    directory.mkdir()
    environment = clean_environment(os.environ) | {'HIP_TARGET': 'gfx1250'}
    commands = Commands(directory, args.source, environment, timeout)
    result = {'case': case, 'passed': False, 'timeout': timeout}
    try:
        # Replay hides only this case's copies, never the shared setup or a sibling.
        inputs = directory / 'original'
        shutil.copytree(args.fixture / 'original', inputs)
        model, helper = (inputs / fixture[name] for name in ('model', 'helper'))
        artifact = directory / 'artifact'
        trace = case == 'trace'
        profile = 'execution-trace' if trace else 'lds'
        selected = SELECTED if trace else (LDS_MATMUL,)
        preparation = commands.run(
            'prepare' if trace else 'prepare-lds',
            [
                tools['prepare'],
                model,
                'gfx1250',
                artifact,
                '--profile',
                profile,
                *sum((['--entry', name] for name in selected), []),
                *([] if trace else ['--require-all']),
            ],
        )
        identity = selection_profile(preparation)
        capture = directory / 'capture.bin'
        output, oracle = application(
            args,
            commands,
            tools,
            inputs,
            'instrumented' if trace else 'lds-instrumented',
            {
                'HSA_TOOLS_LIB': str(tools['plugin']),
                'RJ_CONSAN_APPLICATION_PROFILE': '1',
                'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
                'RJ_CONSAN_SELECTION_PROFILE': identity,
                'RJ_CONSAN_NATIVE_IMAGES': str(helper),
                'RJ_CONSAN_CAPTURE_PATH': str(capture),
            },
            with_plugin=True,
        )
        require(
            (args.fixture / 'baseline/logits.npy').read_bytes()
            == (output / 'logits.npy').read_bytes(),
            f'{case} complete logits differ from original baseline',
        )
        allowed_exits = (0,) if trace else (0, 1)
        before = commands.run(
            'analyze' if trace else 'analyze-lds',
            [tools['analyze'], capture],
            allowed_exits=allowed_exits,
        )
        analyzer_exit = commands.records[-1]['exit']
        analysis = json.loads(before)
        require(
            analysis['selection_profile'] == identity
            and analysis['original_sha256'] == digest(model),
            f'{case} capture image/selection identity differs',
        )
        if trace:
            validate_trace(analysis)
        else:
            validate_lds(analysis, analyzer_exit)
        offline_replay(
            commands,
            tools['analyze'],
            capture,
            (inputs, artifact),
            before,
            stage=(
                'fresh-analysis-without-inputs'
                if trace
                else 'fresh-lds-analysis-without-inputs'
            ),
            allowed_exits=allowed_exits,
            expected_exit=analyzer_exit,
        )
        result.update(
            passed=True,
            baseline=fixture['baseline'],
            fresh_replay_identical=True,
            **(
                {'instrumented': oracle, 'analysis': analysis}
                if trace
                else {'lds_instrumented': oracle, 'lds_analysis': analysis}
            ),
        )
        return result
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        result['stage_elapsed_seconds'] = sum(
            row['seconds'] for row in commands.records
        )
        result['stage_seconds'] = {
            row['stage']: row['seconds'] for row in commands.records
        }
        (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(
            json.dumps(
                {
                    'case': case,
                    'passed': result['passed'],
                    'artifact_directory': str(directory),
                }
            ),
            flush=True,
        )


def combine_case_results(args, fixture, results=None):
    if results is None:
        receipts = [args.fixture / case / 'result.json' for case in CASES]
        if not all(receipt.is_file() for receipt in receipts):
            return None
        results = {
            case: json.loads(receipt.read_text())
            for case, receipt in zip(CASES, receipts)
        }
    passed = all(result['passed'] for result in results.values())
    result = {
        'passed': passed,
        'baseline': fixture['baseline'],
        'cases': results,
        'cpu_thread_budget': args.cpu_thread_budget,
        'fresh_replay_identical': passed,
        'stage_seconds': {
            row['stage']: row['seconds']
            for row in json.loads((args.fixture / 'commands.json').read_text())
        },
    }
    for case in CASES:
        result.update(
            {
                key: value
                for key, value in results[case].items()
                if key
                in ('instrumented', 'analysis', 'lds_instrumented', 'lds_analysis')
            }
        )
        result['stage_seconds'].update(results[case].get('stage_seconds', {}))
    (args.fixture / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def run_cases(args, tools, fixture):
    results = {}
    for case in CASES:
        try:
            results[case] = run_case(args, tools, fixture, case)
        except Exception as error:
            traceback.print_exc()
            receipt = args.fixture / case / 'result.json'
            results[case] = (
                json.loads(receipt.read_text())
                if receipt.is_file()
                else {'passed': False, 'error': f'{type(error).__name__}: {error}'}
            )
    result = combine_case_results(args, fixture, results)
    require(result['passed'], f'prefill case failed; see {args.fixture}')
    print(
        json.dumps({'passed': True, 'artifact_directory': str(args.fixture)}),
        flush=True,
    )
    return result


def run(args):
    tools = validate_environment(args)
    if args.case in ('setup', 'all'):
        args.fixture = setup_fixture(args, tools)
        if args.case == 'setup':
            print(
                json.dumps({'passed': True, 'fixture_directory': str(args.fixture)}),
                flush=True,
            )
            return
    fixture = load_fixture(args, tools)
    if args.case == 'all':
        run_cases(args, tools, fixture)
    else:
        try:
            run_case(args, tools, fixture, args.case)
        finally:
            combine_case_results(args, fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'suite', 'root'):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--case', choices=('all', 'setup', *CASES), default='all')
    parser.add_argument('--fixture', type=lambda value: Path(value).resolve())
    parser.add_argument('--cpu-thread-budget', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    require(args.cpu_thread_budget > 0 and args.timeout > 0, 'budgets must be positive')
    run(args)


if __name__ == '__main__':
    main()
