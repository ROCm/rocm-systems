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
SELECTED = (SCATTER, ELEMENTWISE)
SEQUENCE = (SCATTER, ELEMENTWISE, SCATTER) * 3


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


class Commands:
    def __init__(self, directory, source, environment, timeout):
        self.directory = directory
        self.source = source
        self.environment = environment
        self.deadline = time.monotonic() + timeout
        self.records = []

    def run(self, name, command, environment=None, cwd=None):
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
            result.returncode == 0,
            f'{name}: exit {result.returncode}; see {self.directory}',
        )
        return (self.directory / f'{name}.stdout').read_text()


def offline_replay(commands, analyzer, capture, paths, before):
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
        after = commands.run('fresh-analysis-without-inputs', [analyzer, capture])
        require(after == before, 'independent offline replay differs')
    finally:
        for original, unavailable in reversed(hidden):
            if unavailable.exists():
                unavailable.rename(original)


def run(args):
    for package in ('iree-base-runtime', 'iree-base-compiler'):
        require(
            importlib.metadata.version(package) == IREE_VERSION,
            f'incorrect {package} version',
        )
    validate_suite(args.suite)
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
    args.root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='toy-llama-', dir=args.root))
    print(directory, flush=True)
    (directory / 'inputs.json').write_text(
        json.dumps(
            {
                'suite_revision': SUITE_REVISION,
                'inputs': INPUTS,
                'tools': {
                    name: {'path': str(path), 'sha256': digest(path)}
                    for name, path in tools.items()
                },
            },
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
    environment = clean_environment(os.environ)
    environment['HIP_TARGET'] = 'gfx1250'
    commands = Commands(directory, args.source, environment, args.timeout)
    compiler = Path(sys.executable).parent / 'iree-compile'
    vmfb = inputs / 'model.vmfb'
    commands.run(
        'compile',
        [
            compiler,
            suite / 'assets/toy_llama.mlir',
            '--iree-hal-target-device=hip[0]',
            '--iree-rocm-target=gfx1250',
            '-o',
            vmfb,
        ],
    )
    images = inputs / 'images'
    images.mkdir()
    driver = Path(__file__).with_name('consan_iree_prefill_driver.py')

    def application(name, bindings):
        output = directory / name
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
                driver,
                '--suite',
                suite,
                '--vmfb',
                vmfb,
                '--output',
                output,
                *(['--plugin', tools['plugin']] if name == 'instrumented' else []),
            ],
            environment | bindings,
            cwd=images if name == 'baseline' else None,
        )
        oracle = json.loads((output / 'oracle.json').read_text())
        require(
            oracle['passed']
            and oracle['shape'] == [1, 16, 256]
            and oracle['finite']
            and oracle['dtype'] == 'float32',
            'original numerical oracle failed',
        )
        if name == 'instrumented':
            require(oracle['finish'] == 0, 'capture finish failed')
        return output, oracle

    # Reuse ROCr's existing dump option; no custom HSA hook is needed.
    baseline, baseline_oracle = application(
        'baseline', {'LOADER_OPTIONS_APPEND': '-dump-code'}
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
    artifact = directory / 'artifact'
    preparation = commands.run(
        'prepare',
        [
            tools['prepare'],
            model,
            'gfx1250',
            artifact,
            '--profile',
            'execution-trace',
            *sum((['--entry', name] for name in SELECTED), []),
        ],
    )
    profiles = [
        token.split('=', 1)[1]
        for token in preparation.split()
        if token.startswith('selection_profile=')
    ]
    require(len(profiles) == 1 and len(profiles[0]) == 64, 'missing selection identity')
    capture = directory / 'capture.bin'
    instrumented, oracle = application(
        'instrumented',
        {
            'HSA_TOOLS_LIB': str(tools['plugin']),
            'RJ_CONSAN_APPLICATION_PROFILE': '1',
            'RJ_CONSAN_ARTIFACT_DIR': str(artifact),
            'RJ_CONSAN_SELECTION_PROFILE': profiles[0],
            'RJ_CONSAN_NATIVE_IMAGES': str(helper),
            'RJ_CONSAN_CAPTURE_PATH': str(capture),
        },
    )
    require(
        (baseline / 'logits.npy').read_bytes()
        == (instrumented / 'logits.npy').read_bytes(),
        'complete logits differ from original baseline',
    )
    before = commands.run('analyze', [tools['analyze'], capture])
    analysis = json.loads(before)
    require(
        analysis['selection_profile'] == profiles[0]
        and analysis['original_sha256'] == digest(model),
        'capture image/selection identity differs',
    )
    validate_trace(analysis)
    offline_replay(commands, tools['analyze'], capture, (inputs, artifact), before)
    (directory / 'result.json').write_text(
        json.dumps(
            {
                'passed': True,
                'baseline': baseline_oracle,
                'instrumented': oracle,
                'analysis': analysis,
                'fresh_replay_identical': True,
                'cpu_thread_budget': args.cpu_thread_budget,
                'stage_seconds': {
                    row['stage']: row['seconds'] for row in commands.records
                },
            },
            indent=2,
        )
        + '\n'
    )
    print(
        json.dumps({'passed': True, 'artifact_directory': str(directory)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'build', 'suite', 'root'):
        parser.add_argument(
            '--' + name, type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument('--cpu-thread-budget', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    require(args.cpu_thread_budget > 0 and args.timeout > 0, 'budgets must be positive')
    run(args)


if __name__ == '__main__':
    main()
