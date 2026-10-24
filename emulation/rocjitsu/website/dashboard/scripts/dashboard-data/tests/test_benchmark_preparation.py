# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Workflow preparation tests. All build/install/native commands are intercepted."""

import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

HELPER = (
    Path(__file__).resolve().parents[7]
    / 'emulation/rocjitsu/scripts/prepare-benchmarks.py'
)


def load_helper():
    spec = importlib.util.spec_from_file_location('benchmark_preparation', HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(tmp_path):
    root = tmp_path / 'rocjitsu-benchmark-123-1'
    root.mkdir()
    (root / '.owned-attempt').write_text('123:1\n')
    return dict(
        BENCHMARK_DIR=str(root),
        RUNNER_TEMP=str(tmp_path),
        GITHUB_RUN_ID='123',
        GITHUB_RUN_ATTEMPT='1',
        ROCJITSU_SOURCE_DIR=str(tmp_path / 'source/emulation/rocjitsu'),
        CORPUS_SOURCE_DIR=str(tmp_path / 'corpus'),
        TENSILE_SOURCE=str(tmp_path / 'tensile'),
        NANOBIND_SOURCE=str(tmp_path / 'nanobind'),
        CORPUS_SHA='a' * 40,
        GITHUB_SHA='b' * 40,
        GITHUB_ENV=str(tmp_path / 'github-env'),
        SETUP_STARTED_AT='0',
    )


def test_early_failure_handlers_run_without_python_setup_or_venv(tmp_path):
    import subprocess
    import tarfile
    import yaml

    env = environment(tmp_path)
    env['ROCJITSU_SOURCE_DIR'] = str(HELPER.parents[1])
    empty_path = tmp_path / 'empty-path'
    empty_path.mkdir()
    env['PATH'] = str(empty_path)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    root = Path(env['BENCHMARK_DIR'])
    outside = tmp_path / 'keep.txt'
    outside.write_text('preserve unrelated files')
    assert not (root / 'venv').exists()
    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    steps = {step['name']: step for step in workflow['jobs']['benchmark']['steps']}
    for name in ('Collect preparation diagnostics', 'Clean owned benchmark scratch'):
        result = subprocess.run(
            ['/bin/bash', '-e', '-o', 'pipefail', '-c', steps[name]['run']],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if name == 'Collect preparation diagnostics':
            with tarfile.open(root / 'preparation-diagnostics.tar.gz') as archive:
                assert 'diagnostic-inventory.json' in archive.getnames()
                assert 'reports/source-preservation-error.txt' in archive.getnames()
            assert not (root / 'venv').exists()
    assert not root.exists()
    assert outside.read_text() == 'preserve unrelated files'


def test_early_cleanup_still_rejects_unowned_scratch(tmp_path):
    import subprocess
    import yaml

    env = environment(tmp_path)
    env['ROCJITSU_SOURCE_DIR'] = str(HELPER.parents[1])
    env['PATH'] = ''
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    root = Path(env['BENCHMARK_DIR'])
    (root / '.owned-attempt').write_text('different attempt')
    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    step = next(
        step
        for step in workflow['jobs']['benchmark']['steps']
        if step['name'] == 'Clean owned benchmark scratch'
    )
    result = subprocess.run(
        ['/bin/bash', '-e', '-o', 'pipefail', '-c', step['run']],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert 'Missing or mismatched ownership marker' in result.stderr
    assert (root / '.owned-attempt').read_text() == 'different attempt'


def test_owned_paths_are_outside_runner_output_and_native_backend(tmp_path):
    module = load_helper()
    ctx = module.Preparation(environment(tmp_path))
    for key in (
        'HOME',
        'TMPDIR',
        'XDG_CACHE_HOME',
        'TENSILE_HELPER_CACHE_DIR',
        'PIP_CACHE_DIR',
    ):
        assert Path(ctx.env[key]).is_relative_to(ctx.root)
        assert not Path(ctx.env[key]).is_relative_to(ctx.root / 'results')
    assert ctx.env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert ctx.env['CMAKE_BUILD_PARALLEL_LEVEL'] == '4'
    assert ctx.env['ROCISA_BACKEND'] == 'native'
    assert ctx.cwd.is_dir() and not list(ctx.cwd.iterdir())


@pytest.mark.parametrize('change', ['empty', 'foreign', 'marker', 'symlink'])
def test_cleanup_refuses_unowned_roots(tmp_path, change):
    module = load_helper()
    env = environment(tmp_path)
    root = Path(env['BENCHMARK_DIR'])
    if change == 'empty':
        env['BENCHMARK_DIR'] = ''
    if change == 'foreign':
        env['BENCHMARK_DIR'] = str(tmp_path)
    if change == 'marker':
        (root / '.owned-attempt').write_text('different')
    if change == 'symlink':
        other = tmp_path / 'outside'
        other.mkdir()
        (other / 'keep').write_text('keep')
        root.rename(tmp_path / 'saved')
        root.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        module.Preparation(env)


def test_generation_rejects_wrong_artifact_before_ready_export(tmp_path):
    module = load_helper()
    ctx = module.Preparation(environment(tmp_path))
    ctx.artifacts.mkdir()
    output = ctx.artifacts / 'bf16_streamk'
    output.mkdir()
    (output / 'artifacts.json').write_text(
        json.dumps(
            {
                'variant': 'bf16_streamk',
                'target': 'gfx1250',
                'shape': [4096] * 3,
                'source_revision': module.TENSILE_SHA,
                'files': {},
            }
        )
    )
    with pytest.raises(ValueError, match='provenance'):
        ctx.validate_artifacts(
            {'bf16_streamk': {'target': 'gfx950', 'shape': [4096] * 3}}
        )
    assert not (ctx.root / 'ready.json').exists()


def test_workflow_preparation_order_pins_and_phase_ceiling(tmp_path):
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    steps = workflow['jobs']['benchmark']['steps']
    names = [step['name'] for step in steps]
    required = [
        'Install benchmark dependencies',
        'Build rocjitsu',
        'Prepare Tensile wheel and native adapter',
        'Generate pinned Tensile artifacts',
        'Run performance suite',
    ]
    assert [names.index(name) for name in required] == sorted(
        names.index(name) for name in required
    )
    by_name = {step['name']: step for step in steps}
    setup = [
        'Set benchmark paths',
        'Install base dependencies',
        'Checkout rocm-systems',
        'Record benchmark source',
        'Checkout benchmark corpus',
        'Checkout pinned Tensile source',
        'Checkout pinned native nanobind',
        'Record benchmark machine',
        'Set up Python',
        'Build rocjitsu',
    ]
    assert by_name['Checkout pinned Tensile source']['timeout-minutes'] == 10
    assert sum(by_name[name]['timeout-minutes'] for name in setup) == 33
    assert [by_name[name]['timeout-minutes'] for name in required] == [
        20,
        8,
        45,
        45,
        60,
    ]
    assert workflow['jobs']['benchmark']['timeout-minutes'] == 210
    for name, revision in [
        ('Checkout pinned Tensile source', 'dd77374194a3ea1a7258cd54f87af6747b6362a5'),
        ('Checkout pinned native nanobind', '9b3afa9dbdc23641daf26fadef7743e7127ff92f'),
    ]:
        assert by_name[name]['with']['ref'] == revision
        assert by_name[name]['with']['persist-credentials'] is False
    assert (
        by_name['Checkout pinned native nanobind']['with']['submodules'] == 'recursive'
    )
    assert 'steps.prepare.outcome' in by_name['Run performance suite']['if']
    assert by_name['Collect preparation diagnostics']['if'] == 'always()'
    assert by_name['Clean owned benchmark scratch']['if'] == 'always()'
    assert (
        sum(
            by_name[name]['timeout-minutes']
            for name in [
                'Collect preparation diagnostics',
                'Upload benchmark diagnostics',
                'Upload preparation diagnostics',
                'Clean owned benchmark scratch',
            ]
        )
        == 15
    )


def fixture_preparation(tmp_path, inherited_target=None):
    import time

    module = load_helper()
    env = environment(tmp_path)
    if inherited_target is not None:
        env['HCC_AMDGPU_TARGET'] = inherited_target
    env['SETUP_STARTED_AT'] = str(time.time())
    tensile = Path(env['TENSILE_SOURCE']) / 'projects/hipblaslt/tensilelite'
    (tensile / 'rocisa').mkdir(parents=True)
    (tensile / 'requirements.txt').write_text('pyyaml\nnanobind>=2.0,!=3.0.0\n')
    (tensile / 'rocisa/pyproject.toml').write_text(
        '[build-system]\nrequires=["nanobind>=2.0,!=3.0.0","scikit-build-core>=0.10"]\n'
    )
    corpus = Path(env['CORPUS_SOURCE_DIR'])
    (corpus / 'benchmarks/suites').mkdir(parents=True)
    (corpus / 'benchmarks/requirements.txt').write_text(
        '--index-url https://rocm.nightlies.amd.com/whl-multi-arch/\nfictional-sdk==1.0\n'
    )
    candidates = corpus / 'corpus/benchmarks/tensile_candidates'
    (candidates / 'upstream').mkdir(parents=True)
    supported = {
        name: [target, name + '.yaml', 0, [4096] * 3]
        for name, target in [
            ('bf16_streamk', 'gfx950'),
            ('bf16_subtile', 'gfx1250'),
            ('mxfp8_subtile', 'gfx950'),
            ('mxfp4_streamk', 'gfx1250'),
        ]
    }
    (candidates / 'prepare.py').write_text('CANDIDATES = ' + repr(supported))
    text = 'name="fictional-performance"\n'
    for name, (target, filename, _, shape) in supported.items():
        (candidates / 'upstream' / filename).write_text('FICTIONAL source template')
        text += f'[[cases]]\nworkload="tensile_candidate"\ntargets=["{target}"]\nparams={{variant="{name}",m=4096,n=4096,k=4096}}\n'
    (corpus / 'benchmarks/suites/performance-v1.toml').write_text(text)
    header = Path(env['NANOBIND_SOURCE']) / 'ext/robin_map/include/tsl/robin_map.h'
    header.parent.mkdir(parents=True)
    header.write_text('FICTIONAL header')
    return module, module.Preparation(env)


class FictionalCommands:
    """Synthetic command results for orchestration tests, never native evidence."""

    def __init__(self, module, ctx, failure=None, corrupt=False):
        self.module, self.ctx, self.failure, self.corrupt = (
            module,
            ctx,
            failure,
            corrupt,
        )
        self.calls = []

    def __call__(self, label, argv, capture=False, env_overrides=None):
        import zipfile

        self.calls.append((label, [str(arg) for arg in argv]))
        if label == self.failure:
            raise ValueError('Fictional command failure: ' + label)
        if label.endswith('-revision'):
            return {
                'tensile': self.module.TENSILE_SHA,
                'nanobind': self.module.NANOBIND_SHA,
                'corpus': self.ctx.env['CORPUS_SHA'],
                'rocjitsu': self.ctx.env['GITHUB_SHA'],
            }[label.removesuffix('-revision')]
        if label.endswith('-clean'):
            return ''
        if label == 'venv':
            self.ctx.python.parent.mkdir(parents=True)
            sdk = self.ctx.root / 'venv/sdk'
            for name in [
                'lib/cmake/hip/hip-config.cmake',
                'share/rocmcmakebuildtools/cmake/ROCmCMakeBuildToolsConfig.cmake',
                'lib/llvm/lib/cmake/llvm/LLVMConfig.cmake',
                'lib/llvm/lib/libLLVMObjectYAML.a',
                'lib/llvm/lib/libomp.so',
            ]:
                file = sdk / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('FICTIONAL SDK prerequisite')
        if label == 'sdk-path':
            return str(self.ctx.root / 'venv/sdk')
        if label == 'cmake-version':
            return 'cmake version 3.28.3'
        if label == 'nanobind-dir':
            directory = self.ctx.root / 'venv/nanobind/cmake'
            directory.mkdir(parents=True)
            (directory / 'nanobind-config.cmake').write_text('FICTIONAL config')
            return str(directory)
        if label == 'wheel':
            with zipfile.ZipFile(
                self.ctx.root / 'wheelhouse/rocisa-0.1.0-cp312-linux_x86_64.whl', 'w'
            ) as archive:
                archive.writestr(
                    'rocisa/_rocisa.fictional.so', b'FICTIONAL extension - never loaded'
                )
                archive.writestr(
                    'rocisa/libstinkytofu.so.0', b'FICTIONAL library - never loaded'
                )
        if label in ('build', 'adapter-build'):
            names = (
                ['build/tools/rocjitsu/rocjitsu', 'build/librocjitsu.so']
                if label == 'build'
                else [
                    'tensile-native/tensile_candidate',
                    'tensile-native/tensile_candidate_host_tests',
                ]
            )
            for relative in names:
                file = self.ctx.root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'FICTIONAL executable - never executed')
        if label == 'runtime-version':
            return 'git revision: ' + self.ctx.env['GITHUB_SHA']
        if label.startswith('generate-'):
            variant = label.removeprefix('generate-')
            directory = self.ctx.artifacts / variant
            directory.mkdir(parents=True)
            for name in ['TensileLibrary.yaml', 'fictional.hsaco']:
                (directory / name).write_bytes(
                    b'FICTIONAL generated bytes - never loaded'
                )
            (directory / 'candidate.yaml').write_text(
                '{"GlobalParameters":{"CpuThreads":4}}'
            )
            metadata = {
                'variant': variant,
                'source_revision': self.module.TENSILE_SHA,
                **self.ctx.inventory()[variant],
                'files': {
                    name: self.module.sha256(directory / name)
                    for name in ['TensileLibrary.yaml', 'fictional.hsaco']
                },
                'config_sha256': self.module.sha256(directory / 'candidate.yaml'),
            }
            (directory / 'artifacts.json').write_text(json.dumps(metadata))
            if self.corrupt:
                (directory / 'fictional.hsaco').write_text('Corrupted fictional bytes')
        return ''


def phases(module, ctx, commands):
    with mock.patch.object(ctx, 'run', side_effect=commands):
        for name in ('dependencies', 'rocjitsu', 'native', 'generate'):
            ctx.start(name)
            try:
                getattr(ctx, name)()
            except Exception:
                ctx.finish(False)
                raise
            ctx.finish(True)


def test_intercepted_preparation_orders_exact_owned_wheel_and_four_generators(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx)
    phases(module, ctx, commands)
    labels = [label for label, _ in commands.calls]
    assert (
        labels.index('dependency-preflight')
        < labels.index('build')
        < labels.index('wheel')
        < labels.index('install-exact-wheel')
        < labels.index('exact-wheel-import')
        < labels.index('adapter-build')
        < labels.index('host-tests')
    )
    generation = [
        (label, args) for label, args in commands.calls if label.startswith('generate-')
    ]
    assert [name for name, _ in generation] == [
        'generate-' + variant for variant in ctx.inventory()
    ]
    assert all(args[-3:] == ['4096', '4096', '4096'] for _, args in generation)
    wheel = next(args for label, args in commands.calls if label == 'wheel')
    assert (
        '--no-build-isolation' in wheel
        and '--no-index' in wheel
        and '--no-deps' in wheel
    )
    assert any(
        arg.startswith('--config-settings=cmake.define.nanobind_DIR=') for arg in wheel
    )
    assert any(
        arg == '--config-settings=build-dir=' + str(ctx.root / 'rocisa-build')
        for arg in wheel
    )
    assert not any('GPU_TARGETS' in arg for _, args in commands.calls for arg in args)
    native = next(
        args for label, args in commands.calls if label == 'adapter-configure'
    )
    assert '-DFETCHCONTENT_SOURCE_DIR_NANOBIND=' + ctx.env['NANOBIND_SOURCE'] in native
    assert '-DFETCHCONTENT_FULLY_DISCONNECTED=ON' in native
    assert all(
        args[args.index('--parallel') + 1] == '4'
        for _, args in commands.calls
        if '--parallel' in args
    )
    assert ctx.env['LD_LIBRARY_PATH'].split(':') == [
        str(ctx.root / 'venv/sdk' / p)
        for p in ('lib', 'lib/rocm_sysdeps/lib', 'lib/llvm/lib', 'lib/host-math/lib')
    ]
    assert json.loads((ctx.root / 'ready.json').read_text()) == ctx.inventory()
    exported = Path(ctx.env['GITHUB_ENV']).read_text()
    assert (
        'ROCISA_BACKEND=native\n' in exported
        and 'TENSILE_CANDIDATE_ARTIFACTS=' + str(ctx.artifacts) in exported
    )
    query = next(
        args for label, args in commands.calls if label == 'exact-wheel-import'
    )
    compile(query[query.index('-c') + 1], '<wheel-import-check>', 'exec')
    assert (
        'archive_info' in query[query.index('-c') + 1]
        and 'libstinkytofu.so' in query[query.index('-c') + 1]
    )


@pytest.mark.parametrize(
    'failure',
    ['tensile-clean', 'dependency-preflight', 'wheel', 'generate-bf16_subtile'],
)
def test_intercepted_failure_never_exports_ready_or_continues(tmp_path, failure):
    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx, failure=failure)
    with pytest.raises(ValueError, match='Fictional command failure'):
        phases(module, ctx, commands)
    labels = [label for label, _ in commands.calls]
    assert labels[-1] == failure
    assert not (ctx.root / 'ready.json').exists()
    if Path(ctx.env['GITHUB_ENV']).exists():
        assert (
            'TENSILE_CANDIDATE_ARTIFACTS='
            not in Path(ctx.env['GITHUB_ENV']).read_text()
        )


def test_intercepted_tampered_generation_rejects_before_performance_export(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    with pytest.raises(ValueError, match='hash/path mismatch'):
        phases(module, ctx, FictionalCommands(module, ctx, corrupt=True))
    assert not (ctx.root / 'ready.json').exists()


def test_timeout_terminates_only_owned_process_group_and_records_failure(tmp_path):
    import signal
    import subprocess

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('native')
    process = mock.Mock(pid=987654)
    process.wait.side_effect = [
        subprocess.TimeoutExpired('fictional', 1),
        subprocess.TimeoutExpired('fictional', 3),
        0,
    ]
    with mock.patch.object(
        module.subprocess, 'Popen', return_value=process
    ) as launch, mock.patch.object(module.os, 'killpg') as kill:
        with pytest.raises(ValueError, match='exit 124'):
            ctx.run('fictional-timeout', ['fictional-compiler'])
    assert launch.call_args.kwargs['start_new_session'] is True
    assert launch.call_args.kwargs['cwd'] == ctx.cwd
    assert kill.call_args_list == [
        mock.call(987654, signal.SIGTERM),
        mock.call(987654, signal.SIGKILL),
    ]
    assert json.loads((ctx.root / 'commands.jsonl').read_text())['timed_out'] is True


@pytest.mark.parametrize('absent', [False, True])
def test_timeout_escalates_even_when_leader_exits_before_child(tmp_path, absent):
    import signal
    import subprocess

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('native')
    process = mock.Mock(pid=987654)
    process.wait.side_effect = [subprocess.TimeoutExpired('fictional', 1), 0]

    def signal_group(pid, sig):
        if absent:
            raise ProcessLookupError()

    with mock.patch.object(
        module.subprocess, 'Popen', return_value=process
    ), mock.patch.object(
        module.os, 'killpg', side_effect=signal_group
    ) as kill, mock.patch.object(
        module.time, 'sleep'
    ):
        with pytest.raises(ValueError, match='exit 124'):
            ctx.run('leader-exits', ['fictional-command'])
    assert kill.call_args_list == [
        mock.call(987654, signal.SIGTERM),
        mock.call(987654, signal.SIGKILL),
    ]
    assert json.loads((ctx.root / 'commands.jsonl').read_text())['timed_out'] is True


def test_foreign_loader_file_rejected_but_owned_assembler_output_allowed(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    ctx.start('native')
    (ctx.cwd / 'libstinkytofu.so.0').write_text('foreign')
    with mock.patch.object(module.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError, match='foreign files'):
            ctx.run('import', ['fictional-python'])
        launch.assert_not_called()
    (ctx.cwd / 'libstinkytofu.so.0').unlink()
    (ctx.cwd / 'a.out').write_text('FICTIONAL owned compiler output')
    with mock.patch.object(module.subprocess, 'Popen') as launch:
        launch.return_value.wait.return_value = 0
        ctx.run('import', ['fictional-python'])
        launch.assert_called_once()


def test_readonly_preparation_cwd_rejected_before_process_launch(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    ctx.start('native')
    ctx.cwd.chmod(0o500)
    try:
        with mock.patch.object(module.subprocess, 'Popen') as launch:
            with pytest.raises(ValueError, match='must remain writable'):
                ctx.run('import', ['fictional-python'])
            launch.assert_not_called()
    finally:
        ctx.cwd.chmod(0o700)


def test_owned_cleanup_does_not_follow_runner_output_symlink(tmp_path):
    import os
    import sys

    module = load_helper()
    env = environment(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('keep')
    (Path(env['BENCHMARK_DIR']) / 'results').symlink_to(
        outside, target_is_directory=True
    )
    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        sys, 'argv', ['prepare-benchmarks.py', 'cleanup']
    ):
        module.main()
    assert not Path(env['BENCHMARK_DIR']).exists()
    assert (outside / 'keep').read_text() == 'keep'


def test_diagnostics_are_bounded_and_do_not_follow_symlinks(tmp_path):
    import tarfile

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('diagnostics')
    (ctx.logs / 'large.log').write_bytes(b'x' * 300000)
    outside = tmp_path / 'secret.log'
    outside.write_text('must not archive')
    (ctx.logs / 'outside.log').symlink_to(outside)
    with mock.patch.object(ctx, 'sources'):
        ctx.diagnostics()
    with tarfile.open(ctx.root / 'preparation-diagnostics.tar.gz') as archive:
        assert not any('outside.log' in name for name in archive.getnames())
        stream = archive.extractfile('diagnostic-inventory.json')
        assert stream is not None
        inventory = json.load(stream)
        assert inventory['per_file_limit_bytes'] == 256 * 1024
        assert inventory['entries'][0]['truncated'] is True
        assert archive.getmember('logs/large.log').size == 256 * 1024


@pytest.mark.parametrize(
    'relative',
    ['build', 'rocisa-build', 'tensile-native', 'dependency-preflight/build'],
)
def test_diagnostics_reject_symbolic_cache_ancestors(tmp_path, relative):
    import tarfile

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('diagnostics')
    outside = tmp_path / 'outside-cache'
    outside.mkdir()
    (outside / 'CMakeCache.txt').write_text('PRIVATE OUTSIDE SENTINEL')
    link = ctx.root / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with mock.patch.object(ctx, 'sources'):
        ctx.diagnostics()
    with tarfile.open(ctx.root / 'preparation-diagnostics.tar.gz') as archive:
        assert relative + '/CMakeCache.txt' not in archive.getnames()
    assert (outside / 'CMakeCache.txt').read_text() == 'PRIVATE OUTSIDE SENTINEL'


@pytest.mark.parametrize('symbolic', [False, True])
def test_diagnostics_output_is_exclusive_and_never_follows_link(tmp_path, symbolic):
    module, ctx = fixture_preparation(tmp_path)
    ctx.start('diagnostics')
    archive = ctx.root / 'preparation-diagnostics.tar.gz'
    sentinel = tmp_path / 'outside-sentinel'
    sentinel.write_bytes(b'KEEP EXACT BYTES')
    if symbolic:
        archive.symlink_to(sentinel)
    else:
        archive.write_bytes(b'KEEP EXISTING ARCHIVE')
    with mock.patch.object(ctx, 'sources'):
        with pytest.raises(FileExistsError):
            ctx.diagnostics()
    assert sentinel.read_bytes() == b'KEEP EXACT BYTES'
    if not symbolic:
        assert archive.read_bytes() == b'KEEP EXISTING ARCHIVE'


def test_actual_workflow_bootstrap_is_attempt_local_and_refuses_reuse(tmp_path):
    import os
    import subprocess
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    script = next(
        step['run']
        for step in workflow['jobs']['benchmark']['steps']
        if step['name'] == 'Set benchmark paths'
    )
    env = {
        **os.environ,
        'RUNNER_TEMP': str(tmp_path),
        'GITHUB_RUN_ID': '123',
        'GITHUB_RUN_ATTEMPT': '1',
        'GITHUB_ENV': str(tmp_path / 'exported'),
        'SOURCE_REF_TYPE': 'branch',
    }
    first = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', script],
        env=env,
        text=True,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr
    root = tmp_path / 'rocjitsu-benchmark-123-1'
    assert (root / '.owned-attempt').read_text() == '123:1\n'
    assert 'BENCHMARK_DIR=' + str(root) in (tmp_path / 'exported').read_text()
    before = (root / '.owned-attempt').read_bytes()
    second = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', script],
        env=env,
        text=True,
        capture_output=True,
    )
    assert second.returncode != 0 and (root / '.owned-attempt').read_bytes() == before
    tagged = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', script],
        env={**env, 'SOURCE_REF_TYPE': 'tag', 'GITHUB_RUN_ATTEMPT': '2'},
        text=True,
        capture_output=True,
    )
    assert (
        tagged.returncode == 1 and not (tmp_path / 'rocjitsu-benchmark-123-2').exists()
    )


def test_actual_performance_condition_blocks_preparation_failure():
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    condition = next(
        step['if']
        for step in workflow['jobs']['benchmark']['steps']
        if step['name'] == 'Run performance suite'
    )
    expression = condition.removeprefix('${{ ').removesuffix(' }}')
    for success in (False, True):
        for outcome in ('success', 'failure', 'skipped'):
            actual = (
                expression.replace('success()', str(success))
                .replace('steps.prepare.outcome', repr(outcome))
                .replace('&&', 'and')
            )
            assert eval(actual, {'__builtins__': {}}) == (
                success and outcome == 'success'
            )


def test_exhausted_aggregate_setup_budget_prevents_build(tmp_path):
    import time

    module, ctx = fixture_preparation(tmp_path)
    ctx.env['SETUP_STARTED_AT'] = str(time.time() - 1600)
    (ctx.root / 'phase-dependencies.json').write_text(
        '{"success":true,"elapsed_seconds":0}'
    )
    with mock.patch.object(module.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError, match='budget exhausted'):
            ctx.start('rocjitsu')
        launch.assert_not_called()


def test_early_deadline_prevents_process_launch(tmp_path):
    import time

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('native')
    ctx.deadline = time.monotonic() - 1
    with mock.patch.object(module.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError, match='budget exhausted'):
            ctx.run('compile', ['fictional-compiler'])
        launch.assert_not_called()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_generator_target_overlay_reaches_child_without_leakage(tmp_path, outcome):
    import subprocess

    module, ctx = fixture_preparation(tmp_path, inherited_target='gfx9999')
    assert 'HCC_AMDGPU_TARGET' not in ctx.env
    commands = FictionalCommands(module, ctx)
    base_environment = dict(ctx.env)
    launched = []

    def launch(argv, **kwargs):
        launched.append((argv, kwargs['env']))
        assert kwargs['env'] is not ctx.env
        process = mock.Mock(pid=987654)
        if len(launched) == 2 and outcome == 'timeout':
            process.wait.side_effect = [subprocess.TimeoutExpired('fictional', 1), 0]
        else:
            process.wait.return_value = (
                1 if len(launched) == 2 and outcome == 'failure' else 0
            )
        return process

    def intercept(label, argv, capture=False, env_overrides=None):
        result = commands(label, argv, capture, env_overrides)
        if label.startswith('generate-'):
            return module.Preparation.run(
                ctx, label, argv, capture, env_overrides=env_overrides
            )
        return result

    with mock.patch.object(
        module.subprocess, 'Popen', side_effect=launch
    ), mock.patch.object(module.os, 'killpg'), mock.patch.object(module.time, 'sleep'):
        with mock.patch.object(ctx, 'run', side_effect=intercept):
            if outcome == 'success':
                phases(module, ctx, intercept)
            else:
                with pytest.raises(ValueError, match='failed'):
                    phases(module, ctx, intercept)
        expected = (
            ['gfx950', 'gfx1250', 'gfx950', 'gfx1250']
            if outcome == 'success'
            else ['gfx950', 'gfx1250']
        )
        assert [env['HCC_AMDGPU_TARGET'] for _, env in launched] == expected
        assert 'HCC_AMDGPU_TARGET' not in ctx.env
        # SDK setup may legitimately change other base fields; the target never does.
        assert 'HCC_AMDGPU_TARGET' not in base_environment
        assert 'HCC_AMDGPU_TARGET' not in Path(ctx.env['GITHUB_ENV']).read_text()
        ctx.start('diagnostics')
        module.Preparation.run(ctx, 'unrelated', ['fictional-host-command'])
        assert 'HCC_AMDGPU_TARGET' not in launched[-1][1]
    records = [
        json.loads(line)
        for line in (ctx.root / 'commands.jsonl').read_text().splitlines()
    ]
    assert [item['HCC_AMDGPU_TARGET'] for item in records[:-1]] == expected
    assert records[-1]['HCC_AMDGPU_TARGET'] is None


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_command_overlay_does_not_mutate_base_environment(tmp_path, outcome):
    import subprocess

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('generate')
    before = dict(ctx.env)
    process = mock.Mock(pid=987654)
    if outcome == 'timeout':
        process.wait.side_effect = [subprocess.TimeoutExpired('fictional', 1), 0]
    else:
        process.wait.return_value = 1 if outcome == 'failure' else 0
    with mock.patch.object(
        module.subprocess, 'Popen', return_value=process
    ) as launch, mock.patch.object(module.os, 'killpg'), mock.patch.object(
        module.time, 'sleep'
    ):
        if outcome == 'success':
            ctx.run(
                'explicit-target',
                ['fictional-generator'],
                env_overrides={'HCC_AMDGPU_TARGET': 'gfx1250'},
            )
        else:
            with pytest.raises(ValueError, match='failed'):
                ctx.run(
                    'explicit-target',
                    ['fictional-generator'],
                    env_overrides={'HCC_AMDGPU_TARGET': 'gfx1250'},
                )
    assert ctx.env == before
    assert launch.call_args.kwargs['env']['HCC_AMDGPU_TARGET'] == 'gfx1250'
    assert launch.call_args.kwargs['env'] is not ctx.env


@pytest.mark.parametrize('mode', ['good', 'extension', 'origin', 'mapping'])
def test_exact_wheel_import_predicates_with_explicitly_fake_modules(tmp_path, mode):
    import importlib.metadata
    import sys
    from types import SimpleNamespace

    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx)
    phases(module, ctx, commands)
    query = next(
        args for label, args in commands.calls if label == 'exact-wheel-import'
    )
    code = query[query.index('-c') + 1]
    argv = ['-c', *query[query.index('-c') + 2 :]]
    site = ctx.root / 'venv/site packages/rocisa'
    site.mkdir(parents=True)
    package = site / '__init__.py'
    package.write_text('# FICTIONAL module, not imported from disk')
    extension = site / '_rocisa.fictional.so'
    extension.write_bytes(b'FICTIONAL extension - never loaded')
    library = site / 'libstinkytofu.so.0'
    library.write_bytes(b'FICTIONAL library - never loaded')
    fake = SimpleNamespace(
        __file__=str(package), _rocisa=SimpleNamespace(__file__=str(extension))
    )
    direct = {
        'url': Path(argv[3]).as_uri(),
        'archive_info': {'hashes': {'sha256': argv[4]}},
    }
    if mode == 'extension':
        extension.write_bytes(b'Wrong fictional extension')
    if mode == 'origin':
        direct['url'] = 'file:///other-wheel.whl'
    mapped = library if mode != 'mapping' else tmp_path / 'foreign-libstinkytofu.so.0'
    maps = (
        '0000-ffff r-xp 0 00:00 0 ' + str(mapped).replace(' ', chr(92) + '040') + '\n'
    )
    read_text = Path.read_text

    def controlled_read(path, *args, **kwargs):
        return (
            maps if str(path) == '/proc/self/maps' else read_text(path, *args, **kwargs)
        )

    with mock.patch.dict(sys.modules, {'rocisa': fake}), mock.patch.object(
        sys, 'argv', argv
    ), mock.patch.object(
        importlib.metadata,
        'distribution',
        return_value=SimpleNamespace(read_text=lambda _: json.dumps(direct)),
    ), mock.patch.object(
        Path, 'read_text', controlled_read
    ):
        if mode == 'good':
            exec(compile(code, '<fictional-wheel-predicates>', 'exec'), {})
        else:
            with pytest.raises(AssertionError):
                exec(compile(code, '<fictional-wheel-predicates>', 'exec'), {})
