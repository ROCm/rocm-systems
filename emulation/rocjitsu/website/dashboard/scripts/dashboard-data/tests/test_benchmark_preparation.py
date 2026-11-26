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
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    root = tmp_path / 'rocjitsu-benchmark-123-1'
    root.mkdir()
    (root / '.owned-attempt').write_text('123:1\n')
    return dict(
        BENCHMARK_DIR=str(root),
        UV_EXECUTABLE='/tools/uv',
        RUNNER_TEMP=str(tmp_path),
        GITHUB_RUN_ID='123',
        GITHUB_RUN_ATTEMPT='1',
        ROCJITSU_SOURCE_DIR=str(tmp_path / 'source/emulation/rocjitsu'),
        WORKFLOW_SOURCE_DIR=str(HELPER.parents[1]),
        CORPUS_SOURCE_DIR=str(tmp_path / 'corpus'),
        TENSILE_SOURCE=str(tmp_path / 'tensile'),
        NANOBIND_SOURCE=str(tmp_path / 'nanobind'),
        TENSILE_SHA=workflow['env']['TENSILE_SHA'],
        NANOBIND_SHA=workflow['env']['NANOBIND_SHA'],
        CORPUS_SHA='a' * 40,
        GITHUB_SHA='c' * 40,
        BENCHMARK_SHA='b' * 40,
        GITHUB_ENV=str(tmp_path / 'github-env'),
    )


def rdna_context(tmp_path, cases=None):
    env = environment(tmp_path)
    env.update(
        FULL_TENSILE_SOURCE=str(tmp_path / 'full-tensile'),
        FULL_TENSILE_SHA='1c63e26254a357007ce2d963266085a30b39515b',
        LLAMA_SOURCE=str(tmp_path / 'llama'),
        LLAMA_SHA='26d52b0f9543f441fc2bfa383557c992a4fbf4b9',
    )
    corpus = Path(env['CORPUS_SOURCE_DIR'])
    provider = corpus / 'corpus/benchmarks/full_tensile'
    provider.mkdir(parents=True)
    (provider / 'prepare.py').write_text(
        "CASES = {'valu': dict(variant='valu', dtype='fp32', m=3072, n=2048, k=1024, beta=0)}\n"
        "TARGETS = {'gfx1100': (11,0,0), 'gfx1201': (12,0,1)}\n"
        "def validate_parameters(params, target):\n"
        " if target not in TARGETS or params != CASES.get(params.get('variant')): raise ValueError('fixed parameters')\n"
        " return params['variant']\n"
    )
    manifest = corpus / 'benchmarks/suites/rdna.toml'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(cases or '''[[cases]]
id = "full"
workload = "full_tensile"
targets = ["gfx1100", "gfx1201"]
[cases.params]
variant = "valu"
dtype = "fp32"
m = 3072
n = 2048
k = 1024
beta = 0
''')
    return load_helper().Preparation(env, 'benchmarks/suites/rdna.toml')


def test_rdna_inventory_uses_provider_parameters_and_only_selected_cells(tmp_path):
    ctx = rdna_context(tmp_path)
    selected = ctx.rdna_inventory()
    assert [(item['target'], item['variant']) for item in selected] == [
        ('gfx1100', 'valu'),
        ('gfx1201', 'valu'),
    ]
    assert selected[0]['parameters']['m'] == 3072
    manifest = Path(ctx.env['CORPUS_SOURCE_DIR']) / ctx.manifest
    manifest.write_text(manifest.read_text().replace('m = 3072', 'm = 3073'))
    with pytest.raises(ValueError, match='fixed parameters'):
        ctx.rdna_inventory()


@pytest.mark.parametrize(
    'fault', ['none', 'revision', 'hash', 'symlink', 'extra', 'threads']
)
def test_rdna_artifacts_validate_exact_inventory_and_provenance(tmp_path, fault):
    import hashlib

    ctx = rdna_context(tmp_path)
    selected = ctx.rdna_inventory()[:1]
    output = ctx.root / 'full-tensile-artifacts/gfx1100/valu'
    output.mkdir(parents=True)
    files = {}
    for name in ('TensileLibrary.dat', 'selection.yaml', 'kernel.co'):
        (output / name).write_text(name)
        files[name] = hashlib.sha256(name.encode()).hexdigest()
    metadata = dict(
        source_revision=ctx.env['FULL_TENSILE_SHA'],
        target='gfx1100',
        parameters=selected[0]['parameters'],
        files=files,
        generator_cpu_threads=1,
        generator_sha256=load_helper().sha256(
            Path(ctx.env['CORPUS_SOURCE_DIR'])
            / 'corpus/benchmarks/full_tensile/prepare.py'
        ),
    )
    if fault == 'revision':
        metadata['source_revision'] = ctx.env['TENSILE_SHA']
    if fault == 'threads':
        metadata['generator_cpu_threads'] = 4
    if fault == 'hash':
        (output / 'kernel.co').write_text('wrong')
    if fault == 'extra':
        (output / 'extra.hsaco').write_text('extra')
    if fault == 'symlink':
        external = tmp_path / 'external'
        external.write_text('kernel.co')
        (output / 'kernel.co').unlink()
        (output / 'kernel.co').symlink_to(external)
    (output / 'artifacts.json').write_text(json.dumps(metadata))
    if fault == 'none':
        ctx.validate_rdna_artifacts(selected)
    else:
        with pytest.raises(ValueError):
            ctx.validate_rdna_artifacts(selected)


@pytest.mark.parametrize(
    'fault',
    [
        'none',
        'schema',
        'source',
        'pin',
        'backend',
        'target',
        'adapter-source',
        'missing-adapter-hash',
        'symlink',
        'receipt-symlink',
        'build-symlink',
    ],
)
def test_llama_build_receipt_verified_before_export(tmp_path, fault):
    ctx = rdna_context(tmp_path)
    build = ctx.root / 'llama-hip'
    build.mkdir()
    adapter = Path(ctx.env['CORPUS_SOURCE_DIR']) / 'corpus/benchmarks/llama/runner.cpp'
    adapter.parent.mkdir(parents=True)
    adapter.write_text('// intercepted adapter source\n')
    binary = build / 'llama_bounded_ops'
    binary.write_text('intercepted build; not native evidence')
    receipt = dict(
        schema='rocjitsu.benchmark.llama-build.v1',
        source_pin=ctx.env['LLAMA_SHA'],
        source=ctx.env['LLAMA_SOURCE'],
        backend='hip',
        targets=['gfx1100', 'gfx1201'],
        adapter_sha256=load_helper().sha256(adapter),
    )
    assert 'binary_sha256' not in receipt
    if fault == 'schema':
        receipt['schema'] = 'unknown'
    if fault == 'source':
        receipt['source'] = str(tmp_path / 'other-source')
    if fault == 'pin':
        receipt['source_pin'] = 'a' * 40
    if fault == 'backend':
        receipt['backend'] = 'cpu'
    if fault == 'target':
        receipt['targets'] = ['gfx1100']
    if fault == 'adapter-source':
        adapter.write_text('// modified adapter source\n')
    if fault == 'missing-adapter-hash':
        del receipt['adapter_sha256']
    if fault == 'symlink':
        outside = tmp_path / 'outside-binary'
        binary.rename(outside)
        binary.symlink_to(outside)
    (build / 'benchmark-build.json').write_text(json.dumps(receipt))
    if fault == 'receipt-symlink':
        receipt_path = build / 'benchmark-build.json'
        outside = tmp_path / 'outside-receipt.json'
        receipt_path.rename(outside)
        receipt_path.symlink_to(outside)
    if fault == 'build-symlink':
        outside = tmp_path / 'outside-build'
        build.rename(outside)
        build.symlink_to(outside, target_is_directory=True)
    if fault == 'none':
        ctx.validate_llama_build()
    else:
        with pytest.raises(ValueError):
            ctx.validate_llama_build()
    assert not Path(ctx.env['GITHUB_ENV']).exists()


@pytest.mark.parametrize(
    'phase,prerequisite,stale_output',
    [
        ('tensile-build', 'tensilelite-generate', 'full-tensile-native'),
        ('tensile-generate', 'tensile-build', 'full-tensile-artifacts'),
        ('llama-build', 'tensile-generate', 'llama-hip'),
        ('gluon-verify', 'llama-build', 'reports/gluon-source.json'),
    ],
)
@pytest.mark.parametrize('fault', ['manifest-path', 'parameters', 'stale'])
def test_rdna_phase_failure_never_exports_ready_state(
    tmp_path, phase, prerequisite, stale_output, fault
):
    ctx = rdna_context(tmp_path)
    (ctx.root / 'phase-tensilelite-generate.json').write_text('{"success":true}')
    ctx.env['ROCM_PATH'] = str(ctx.root / 'venv/sdk')
    (ctx.root / f'phase-{prerequisite}.json').write_text('{"success":true}')
    if fault == 'manifest-path':
        ctx.manifest = '../outside.toml'
    if fault == 'parameters':
        manifest = Path(ctx.env['CORPUS_SOURCE_DIR']) / ctx.manifest
        manifest.write_text(manifest.read_text().replace('m = 3072', 'm = 1'))
    if fault == 'stale':
        (ctx.root / stale_output).symlink_to(tmp_path)
    with mock.patch.object(ctx, 'sources'), mock.patch.object(
        ctx, 'sdk_environment'
    ), mock.patch.object(ctx, 'pins'), mock.patch.object(
        ctx, 'run', return_value='{"installed": []}'
    ) as run:
        with pytest.raises(ValueError):
            getattr(ctx, phase.replace('-', '_'))()
    run.assert_not_called()
    assert not (ctx.root / 'reports/providers-ready.json').exists()
    assert not Path(ctx.env['GITHUB_ENV']).exists()


@pytest.mark.parametrize(
    'phase,prerequisite',
    [
        ('rocisa', 'rocjitsu'),
        ('tensilelite-build', 'rocisa'),
        ('tensilelite-generate', 'tensilelite-build'),
        ('tensile-build', 'tensilelite-generate'),
        ('tensile-generate', 'tensile-build'),
        ('llama-build', 'tensile-generate'),
        ('gluon-verify', 'llama-build'),
    ],
)
@pytest.mark.parametrize('receipt', [None, False])
def test_provider_phase_requires_successful_predecessor(
    tmp_path, phase, prerequisite, receipt
):
    ctx = rdna_context(tmp_path)
    if receipt is not None:
        (ctx.root / f'phase-{prerequisite}.json').write_text(
            json.dumps({'success': receipt})
        )
    with mock.patch.object(ctx, 'run') as run:
        with pytest.raises((ValueError, FileNotFoundError)):
            getattr(ctx, phase.replace('-', '_'))()
    run.assert_not_called()
    assert not Path(ctx.env['GITHUB_ENV']).exists()


def test_unselected_optional_providers_skip_work_but_validate_each_phase(tmp_path):
    ctx = rdna_context(tmp_path, '[[cases]]\nworkload="tensile_candidate"\n')
    (ctx.root / 'phase-tensilelite-generate.json').write_text('{"success":true}')
    with mock.patch.object(ctx, 'sources') as sources, mock.patch.object(
        ctx, 'pins'
    ) as pins, mock.patch.object(ctx, 'sdk_environment'), mock.patch.object(
        ctx, 'run'
    ) as run, mock.patch.object(
        ctx, 'install'
    ) as install:
        for phase in (
            'tensile-build',
            'tensile-generate',
            'llama-build',
            'gluon-verify',
        ):
            ctx.start(phase)
            getattr(ctx, phase.replace('-', '_'))()
            ctx.finish(True)
    assert sources.call_count == pins.call_count == 8
    assert [call.args[0] for call in run.call_args_list] == [
        'provider-packages-check'
    ] * 4
    install.assert_not_called()
    assert not Path(ctx.env['GITHUB_ENV']).exists()
    assert (
        json.loads((ctx.root / 'reports/providers-ready.json').read_text())[
            'full_tensile'
        ]
        == []
    )


@pytest.mark.parametrize(
    'phase,prerequisite',
    [
        ('tensile-build', 'tensilelite-generate'),
        ('tensile-generate', 'tensile-build'),
        ('llama-build', 'tensile-generate'),
        ('gluon-verify', 'llama-build'),
    ],
)
def test_provider_validation_failure_does_not_export_paths(
    tmp_path, phase, prerequisite
):
    ctx = rdna_context(tmp_path)
    manifest = Path(ctx.env['CORPUS_SOURCE_DIR']) / ctx.manifest
    manifest.write_text(
        manifest.read_text()
        + '\n[[cases]]\nworkload="llama_rope"\n[[cases]]\nworkload="gluon_copy"\n'
    )
    (ctx.root / f'phase-{prerequisite}.json').write_text('{"success":true}')
    ctx.env['ROCM_PATH'] = str(ctx.root / 'venv/sdk')

    def run(label, argv, capture=False, env_overrides=None):
        if label == 'gluon-prepare':
            raise ValueError('source/license verification failed')
        return ''

    # Successful child-command status alone is insufficient: real binary,
    # artifact and llama receipt validators must still reject absent outputs.
    with mock.patch.object(ctx, 'sources'), mock.patch.object(
        ctx, 'pins'
    ), mock.patch.object(ctx, 'sdk_environment'), mock.patch.object(
        ctx, 'install'
    ), mock.patch.object(
        ctx, 'run', side_effect=run
    ):
        with pytest.raises((ValueError, FileNotFoundError)):
            getattr(ctx, phase.replace('-', '_'))()
    assert not Path(ctx.env['GITHUB_ENV']).exists()
    assert not (ctx.root / 'reports/providers-ready.json').exists()


def test_provider_final_source_check_precedes_environment_export(tmp_path):
    ctx = rdna_context(tmp_path, '[[cases]]\nworkload="llama_rope"\n')
    (ctx.root / 'phase-tensile-generate.json').write_text('{"success":true}')
    ctx.env['ROCM_PATH'] = str(ctx.root / 'venv/sdk')
    with mock.patch.object(
        ctx, 'sources', side_effect=[None, ValueError('source is dirty')]
    ), mock.patch.object(ctx, 'pins'), mock.patch.object(
        ctx, 'sdk_environment'
    ), mock.patch.object(
        ctx, 'run'
    ), mock.patch.object(
        ctx, 'validate_llama_build'
    ) as validated:
        with pytest.raises(ValueError, match='source is dirty'):
            ctx.llama_build()
    validated.assert_called_once()
    assert not Path(ctx.env['GITHUB_ENV']).exists()


def test_rdna_diagnostics_include_owned_build_logs_and_cell_receipts(tmp_path):
    import tarfile

    ctx = rdna_context(tmp_path)
    paths = [
        'full-tensile-native/CMakeCache.txt',
        'llama-hip/configure.log',
        'llama-hip/build.log',
        'llama-hip/benchmark-build.json',
        'full-tensile-artifacts/gfx1100/valu/artifacts.json',
        'full-tensile-artifacts/gfx1100/valu/selection.yaml',
    ]
    for name in paths:
        path = ctx.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('intercepted diagnostic')
    with mock.patch.object(ctx, 'sources'):
        ctx.diagnostics()
    with tarfile.open(ctx.root / 'preparation-diagnostics.tar.gz') as archive:
        assert set(paths) <= set(archive.getnames())


def test_rdna_orchestration_builds_owned_providers_and_exports_after_validation(
    tmp_path,
):
    ctx = rdna_context(tmp_path)
    manifest = Path(ctx.env['CORPUS_SOURCE_DIR']) / ctx.manifest
    manifest.write_text(manifest.read_text() + '''
[[cases]]
workload = "llama_rope"
[[cases]]
workload = "gluon_copy"
''')
    (ctx.root / 'phase-tensilelite-generate.json').write_text('{"success":true}')
    (ctx.root / 'benchmark-constraints.txt').write_text('PyYAML==6.0.2\n')
    ctx.env['ROCM_PATH'] = str(ctx.root / 'venv/sdk')
    commands = {}

    def run(label, argv, capture=False, env_overrides=None):
        commands[label] = [str(a) for a in argv]
        if label.endswith('-inspect'):
            return '{"installed": []}'
        if label == 'full-tensile-build':
            output = ctx.root / 'full-tensile-native/full_tensile_runner'
            output.parent.mkdir()
            output.write_text('intercepted native build')
        if label == 'llama-prepare':
            output = ctx.root / 'llama-hip/llama_bounded_ops'
            output.parent.mkdir()
            output.write_text('intercepted HIP build')

    with mock.patch.object(ctx, 'run', side_effect=run), mock.patch.object(
        ctx, 'sources'
    ) as sources, mock.patch.object(ctx, 'sdk_environment'), mock.patch.object(
        ctx, 'pins'
    ), mock.patch.object(
        ctx, 'validate_rdna_artifacts'
    ) as artifacts, mock.patch.object(
        ctx, 'validate_llama_build', create=True
    ) as llama:
        for phase in (
            'tensile-build',
            'tensile-generate',
            'llama-build',
            'gluon-verify',
        ):
            assert hasattr(ctx, phase.replace('-', '_')), phase
            # The first prerequisite is the preceding legacy generation phase.
            if phase == 'tensile-build':
                (ctx.root / 'phase-tensilelite-generate.json').write_text(
                    '{"success":true}'
                )
            ctx.start(phase)
            getattr(ctx, phase.replace('-', '_'))()
            ctx.finish(True)
            assert (
                json.loads((ctx.root / f'phase-{phase}.json').read_text())['phase']
                == phase
            )
            if phase == 'tensile-build':
                assert not any(
                    label.startswith('full-tensile-generate-') for label in commands
                )
                assert not Path(ctx.env['GITHUB_ENV']).exists()
            if phase == 'tensile-generate':
                assert 'FULL_TENSILE_RUNNER=' in Path(ctx.env['GITHUB_ENV']).read_text()
                assert 'llama-prepare' not in commands
            if phase == 'llama-build':
                assert (
                    'ROCJITSU_LLAMA_BINARY=' in Path(ctx.env['GITHUB_ENV']).read_text()
                )
                assert 'gluon-prepare' not in commands
    assert sources.call_count == 8
    artifacts.assert_called_once_with(ctx.rdna_inventory())
    llama.assert_called_once()
    assert commands['full-tensile-build'][-4:] == [
        '--target',
        'full_tensile_runner',
        '--parallel',
        '16',
    ]
    configure = commands['full-tensile-configure']
    assert '-DTENSILE_SOURCE=' + ctx.env['FULL_TENSILE_SOURCE'] in configure
    assert '-DCMAKE_PREFIX_PATH=' + ctx.env['ROCM_PATH'] in configure
    assert (
        '-DCMAKE_CXX_COMPILER=' + ctx.env['ROCM_PATH'] + '/bin/amdclang++' in configure
    )
    install = commands['full-tensile-requirements']
    assert install[:4] == ['/tools/uv', '--no-config', 'pip', 'install']
    assert install[install.index('--python') + 1] == str(ctx.python)
    assert install[install.index('--constraint') + 1] == str(
        ctx.root / 'benchmark-constraints.txt'
    )
    assert (
        install[-1]
        == ctx.env['FULL_TENSILE_SOURCE'] + '/shared/tensile/requirements.txt'
    )
    generations = [
        argv
        for label, argv in commands.items()
        if label.startswith('full-tensile-generate-')
    ]
    assert len(generations) == 2
    for argv, target in zip(generations, ('gfx1100', 'gfx1201')):
        assert argv[-4:] == ['--target', target, '--variant', 'valu']
    assert commands['llama-prepare'][-4:] == [
        '--backend',
        'hip',
        '--rocm-path',
        ctx.env['ROCM_PATH'],
    ]
    assert commands['gluon-prepare'][-3:] == [
        '--upstream',
        '--output',
        str(ctx.root / 'reports/gluon-source.json'),
    ]
    exported = Path(ctx.env['GITHUB_ENV']).read_text()
    assert f'ROCJITSU_LLAMA_BINARY={ctx.root}/llama-hip/llama_bounded_ops\n' in exported
    for key in (
        'FULL_TENSILE_ARTIFACTS',
        'FULL_TENSILE_RUNNER',
        'ROCJITSU_LLAMA_BINARY',
    ):
        assert f'{key}={ctx.root}/' in exported
    assert (ctx.root / 'reports/providers-ready.json').is_file()


@pytest.mark.parametrize('name', ['FULL_TENSILE', 'LLAMA'])
def test_rdna_sources_checked_and_never_owned_as_scratch(tmp_path, name):
    ctx = rdna_context(tmp_path)
    calls = []

    def run(label, argv, capture=False):
        calls.append(label)
        if label.endswith('-revision'):
            return ctx.env[
                {
                    'tensile': 'TENSILE_SHA',
                    'nanobind': 'NANOBIND_SHA',
                    'corpus': 'CORPUS_SHA',
                    'rocjitsu': 'BENCHMARK_SHA',
                    'full-tensile': 'FULL_TENSILE_SHA',
                    'llama': 'LLAMA_SHA',
                }[label[:-9]]
            ]
        if label == name.lower().replace('_', '-') + '-clean':
            assert (
                '--ignored' in argv
            )  # Ignored cache/build files are not pristine sources.
            return '!! ignored-cache'
        return ''

    with mock.patch.object(ctx, 'run', side_effect=run):
        with pytest.raises(ValueError, match='source is dirty'):
            ctx.sources()
    assert name.lower().replace('_', '-') + '-clean' in calls
    for path in (ctx.root / 'nested-source', ctx.root.parent):
        env = {**ctx.env, name + '_SOURCE': str(path)}
        with pytest.raises(ValueError, match='source.*scratch'):
            load_helper().Preparation(env)


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


@pytest.mark.parametrize(
    'incoming',
    [None, '', ':', '/toolcache/lib::/custom/lib:/toolcache/lib:', 'sdk-duplicates'],
)
def test_sdk_loader_environment_survives_export_and_child_handoff(tmp_path, incoming):
    import sys

    module = load_helper()
    env = environment(tmp_path)
    if incoming == 'sdk-duplicates':
        sdk = Path(env['BENCHMARK_DIR']) / 'venv/sdk'
        incoming = (
            f':/toolcache/lib:{sdk}/lib:/custom/lib:{sdk}/lib/llvm/lib:/toolcache/lib:'
        )
    if incoming is not None:
        env['LD_LIBRARY_PATH'] = incoming
    for key in ('LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME'):
        env[key] = '/untrusted/injection'
    original = dict(env)
    ctx = module.Preparation(env)
    sdk = ctx.root / 'venv/sdk'
    inherited = (
        ['/toolcache/lib', '/custom/lib']
        if incoming and 'toolcache' in incoming
        else []
    )
    expected = [
        str(sdk / name)
        for name in ('lib', 'lib/rocm_sysdeps/lib', 'lib/llvm/lib', 'lib/host-math/lib')
    ] + inherited
    with mock.patch.object(ctx, 'run', return_value=str(sdk)):
        ctx.sdk_environment()
        ctx.sdk_environment()
    assert ctx.env['LD_LIBRARY_PATH'].split(':') == expected
    assert env == original
    ctx.export_environment()
    exported = dict(
        line.split('=', 1) for line in Path(env['GITHUB_ENV']).read_text().splitlines()
    )
    next_ctx = module.Preparation({**env, **exported})
    with mock.patch.object(next_ctx, 'run', return_value=str(sdk)):
        next_ctx.sdk_environment()
    assert next_ctx.env['LD_LIBRARY_PATH'].split(':') == expected
    assert all(
        key not in next_ctx.env for key in ('LD_PRELOAD', 'PYTHONPATH', 'PYTHONHOME')
    )
    next_ctx.start('dependencies')
    # Real fresh child proves transport only, not exact CI Python/native imports.
    payload = next_ctx.run(
        'environment-handoff',
        [
            sys.executable,
            '-B',
            '-c',
            'import json,os; print(json.dumps({k:os.environ.get(k) for k in ("LD_LIBRARY_PATH","LD_PRELOAD","PYTHONPATH","PYTHONHOME")}))',
        ],
        capture=True,
    )
    assert json.loads(payload) == {
        'LD_LIBRARY_PATH': ':'.join(expected),
        'LD_PRELOAD': None,
        'PYTHONPATH': None,
        'PYTHONHOME': None,
    }


@pytest.mark.parametrize(
    'key',
    [
        'ROCJITSU_SOURCE_DIR',
        'WORKFLOW_SOURCE_DIR',
        'CORPUS_SOURCE_DIR',
        'TENSILE_SOURCE',
        'NANOBIND_SOURCE',
    ],
)
def test_all_sources_must_stay_external_to_scratch(tmp_path, key):
    env = environment(tmp_path)
    env[key] = str(Path(env['BENCHMARK_DIR']) / 'nested-source')
    with pytest.raises(ValueError, match='source.*scratch'):
        load_helper().Preparation(env)


@pytest.mark.parametrize(
    'fault', ['none', 'inside', 'ancestor', 'relative', 'symlink', 'source']
)
def test_package_cache_survives_owned_cleanup_but_rejects_unsafe_roots(tmp_path, fault):
    import shutil

    env = environment(tmp_path)
    cache = tmp_path / 'persistent-cache'
    cache.mkdir()
    (cache / 'keep').write_text('cached package')
    env['ROCJITSU_PACKAGE_CACHE'] = str(cache)
    if fault == 'inside':
        env['ROCJITSU_PACKAGE_CACHE'] = env['BENCHMARK_DIR'] + '/cache'
    if fault == 'ancestor':
        env['ROCJITSU_PACKAGE_CACHE'] = str(tmp_path)
    if fault == 'relative':
        env['ROCJITSU_PACKAGE_CACHE'] = 'cache'
    if fault == 'symlink':
        link = tmp_path / 'cache-link'
        link.symlink_to(cache, target_is_directory=True)
        env['ROCJITSU_PACKAGE_CACHE'] = str(link)
    if fault == 'source':
        env['ROCJITSU_PACKAGE_CACHE'] = env['CORPUS_SOURCE_DIR'] + '/cache'
    if fault != 'none':
        with pytest.raises(ValueError, match='[Cc]ache'):
            load_helper().Preparation(env)
        return
    ctx = load_helper().Preparation(env)
    assert ctx.env['UV_CACHE_DIR'] == str(cache / 'uv')
    assert 'PIP_CACHE_DIR' not in ctx.env
    assert Path(ctx.env['TRITON_CACHE_DIR']).is_relative_to(ctx.root)
    assert Path(ctx.env['TENSILE_HELPER_CACHE_DIR']).is_relative_to(ctx.root)
    shutil.rmtree(ctx.root)
    assert (cache / 'keep').read_text() == 'cached package'


def test_owned_paths_are_outside_runner_output_and_native_backend(tmp_path):
    module = load_helper()
    ctx = module.Preparation(environment(tmp_path))
    for key in (
        'HOME',
        'TMPDIR',
        'XDG_CACHE_HOME',
        'TENSILE_HELPER_CACHE_DIR',
    ):
        assert Path(ctx.env[key]).is_relative_to(ctx.root)
        assert not Path(ctx.env[key]).is_relative_to(ctx.root / 'results')
    assert ctx.env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert 'CMAKE_BUILD_PARALLEL_LEVEL' not in ctx.env
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
                'source_revision': ctx.env['TENSILE_SHA'],
                'files': {},
            }
        )
    )
    with pytest.raises(ValueError, match='provenance'):
        ctx.validate_artifacts(
            {'bf16_streamk': {'target': 'gfx950', 'shape': [4096] * 3}}
        )
    assert not (ctx.root / 'ready.json').exists()


def test_workflow_preparation_order_pins_and_timeout_authority(tmp_path):
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    steps = workflow['jobs']['benchmark']['steps']
    names = [step['name'] for step in steps]
    required = [
        'Install benchmark dependencies',
        'Build rocjitsu',
        'Prepare benchmark providers',
        'Run performance suite',
    ]
    assert [names.index(name) for name in required] == sorted(
        names.index(name) for name in required
    )
    by_name = {step['name']: step for step in steps}
    assert all(
        not ('uses' in step and 'run' in step) for step in steps
    ), 'An Actions step must not combine uses and run'
    python_setup = by_name['Set up Python']
    assert python_setup['uses'] == (
        'actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405'
    )
    assert python_setup['with']['python-version'] == '3.12'
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
    setup += ['Record checked Tensile revision', 'Install benchmark dependencies']
    setup += required[:-1]
    assert all('timeout-minutes' not in by_name[name] for name in setup)
    # Workflow limits are authoritative, not additive or reserved headroom.
    assert {
        step['name']: step['timeout-minutes']
        for step in steps
        if 'timeout-minutes' in step
    } == {
        'Run performance suite': 180,
    }
    assert workflow['jobs']['benchmark']['timeout-minutes'] == 240
    assert workflow['jobs']['publish']['timeout-minutes'] == 30
    for name, key, revision in [
        (
            'Checkout pinned Tensile source',
            'TENSILE_SHA',
            'dd77374194a3ea1a7258cd54f87af6747b6362a5',
        ),
        (
            'Checkout pinned native nanobind',
            'NANOBIND_SHA',
            '9b3afa9dbdc23641daf26fadef7743e7127ff92f',
        ),
    ]:
        assert workflow['env'][key] == revision
        assert by_name[name]['with']['ref'] == '${{ env.' + key + ' }}'
        assert by_name[name]['with']['persist-credentials'] is False
    assert 'fetch-depth' not in by_name['Checkout pinned Tensile source']['with']
    assert (
        by_name['Checkout pinned native nanobind']['with']['submodules'] == 'recursive'
    )
    assert 'steps.providers.outcome' in by_name['Run performance suite']['if']
    assert by_name['Collect preparation diagnostics']['if'] == 'always()'
    assert by_name['Clean owned benchmark scratch']['if'] == 'always()'


def fixture_preparation(tmp_path, inherited_target=None):
    module = load_helper()
    env = environment(tmp_path)
    if inherited_target is not None:
        env['HCC_AMDGPU_TARGET'] = inherited_target

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
    return module, module.Preparation(env, 'benchmarks/suites/performance-v1.toml')


@pytest.mark.parametrize('alternative', [False, True])
def test_cli_manifest_reaches_preparation_inventory(tmp_path, alternative):
    import os
    import sys

    module, ctx = fixture_preparation(tmp_path)
    corpus = Path(ctx.env['CORPUS_SOURCE_DIR'])
    manifest = 'benchmarks/suites/performance-v1.toml'
    if alternative:
        manifest = 'benchmarks/suites/alternative recipe.toml'
        (corpus / manifest).write_text(
            '[[cases]]\nworkload="tensile_candidate"\ntargets=["gfx1250"]\n'
            'params={variant="bf16_subtile",m=128,n=256,k=512}\n'
        )
    selected = {}

    def inventory_only(self):
        selected.update(self.inventory())

    with mock.patch.dict(os.environ, ctx.env, clear=True), mock.patch.object(
        sys,
        'argv',
        ['prepare-benchmarks.py', 'tensilelite-generate', '--manifest', manifest],
    ), mock.patch.object(module.Preparation, 'tensilelite_generate', inventory_only):
        module.main()
    assert list(selected) == (
        ['bf16_subtile']
        if alternative
        else ['bf16_streamk', 'bf16_subtile', 'mxfp8_subtile', 'mxfp4_streamk']
    )
    assert selected['bf16_subtile']['shape'] == (
        [128, 256, 512] if alternative else [4096] * 3
    )


@pytest.mark.parametrize(
    'manifest',
    [
        None,
        '',
        '/outside.toml',
        '../outside.toml',
        'missing.toml',
        'directory.toml',
        'recipe.json',
        'linked.toml',
        'linked-dir/performance-v1.toml',
        'benchmarks/../outside.toml',
    ],
)
def test_inventory_rejects_invalid_manifest_path(tmp_path, manifest):
    module, ctx = fixture_preparation(tmp_path)
    corpus = Path(ctx.env['CORPUS_SOURCE_DIR'])
    (corpus / 'directory.toml').mkdir()
    text = (corpus / ctx.manifest).read_text()
    (corpus / 'recipe.json').write_text(text)
    (corpus.parent / 'outside.toml').write_text(text)
    (corpus / 'outside.toml').write_text(text)
    (corpus / 'linked.toml').symlink_to(corpus / ctx.manifest)
    (corpus / 'linked-dir').symlink_to(
        corpus / 'benchmarks/suites', target_is_directory=True
    )
    ctx.manifest = manifest
    with pytest.raises(ValueError, match='[Mm]anifest'):
        ctx.inventory()


@pytest.mark.parametrize(
    'phase',
    [
        'tensilelite-generate',
        'tensile-build',
        'tensile-generate',
        'llama-build',
        'gluon-verify',
    ],
)
def test_generate_cli_requires_explicit_manifest_before_preparation(capsys, phase):
    import sys

    module = load_helper()
    with mock.patch.object(
        sys, 'argv', ['prepare-benchmarks.py', phase]
    ), mock.patch.object(module, 'Preparation') as preparation:
        with pytest.raises(SystemExit) as error:
            module.main()
    assert error.value.code == 2
    assert '--manifest is required for ' + phase in capsys.readouterr().err
    preparation.assert_not_called()


@pytest.mark.parametrize(
    'phase', ['dependencies', 'rocjitsu', 'rocisa', 'tensilelite-build']
)
def test_noninventory_cli_phases_do_not_require_manifest(tmp_path, phase):
    import os
    import sys

    module = load_helper()
    env = environment(tmp_path)
    root = Path(env['BENCHMARK_DIR'])
    (root / 'phase-dependencies.json').write_text('{"success":true}')
    if phase == 'dependencies':
        (root / 'phase-dependencies.json').unlink()

    def no_inventory(self):
        assert self.manifest is None

    with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
        sys, 'argv', ['prepare-benchmarks.py', phase]
    ), mock.patch.object(module.Preparation, phase.replace('-', '_'), no_inventory):
        module.main()
    assert json.loads((root / f'phase-{phase}.json').read_text())['success']


def test_inventory_prepares_only_selected_variants(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    corpus = Path(ctx.env['CORPUS_SOURCE_DIR'])
    recipe = corpus / ctx.manifest
    first_case = recipe.read_text().split('[[cases]]')[1]
    recipe.write_text('name="fictional-subset"\n[[cases]]' + first_case)
    candidates = corpus / 'corpus/benchmarks/tensile_candidates/prepare.py'
    # Unselected supported entries need not even have a local template.
    import ast

    supported = ast.literal_eval(candidates.read_text().removeprefix('CANDIDATES = '))
    supported['unused'] = ['gfx950', 'unused.yaml', 0, [4096] * 3]
    candidates.write_text('CANDIDATES = ' + repr(supported))
    selected = ctx.inventory()
    assert list(selected) == ['bf16_streamk']
    assert selected['bf16_streamk'] == {
        'target': 'gfx950',
        'shape': [4096, 4096, 4096],
        'upstream_file': 'bf16_streamk.yaml',
        'upstream_sha256': module.sha256(
            candidates.parent / 'upstream/bf16_streamk.yaml'
        ),
    }
    commands = FictionalCommands(module, ctx)
    phases(module, ctx, commands)
    assert [label for label, _ in commands.calls if label.startswith('generate-')] == [
        'generate-bf16_streamk'
    ]
    assert list(json.loads((ctx.root / 'ready.json').read_text())) == ['bf16_streamk']


@pytest.mark.parametrize(
    'declaration',
    [
        'OTHER = {}',
        'CANDIDATES: dict = {}',
        'CANDIDATES = dict()',
        'CANDIDATES = []',
        'CANDIDATES = {}',
    ],
)
def test_inventory_rejects_unsupported_candidate_declaration(tmp_path, declaration):
    _, ctx = fixture_preparation(tmp_path)
    candidates = (
        Path(ctx.env['CORPUS_SOURCE_DIR'])
        / 'corpus/benchmarks/tensile_candidates/prepare.py'
    )
    candidates.write_text(declaration)
    with pytest.raises(ValueError, match='Unsupported CANDIDATES declaration'):
        ctx.inventory()


@pytest.mark.parametrize(
    'change', ['empty', 'variant', 'target', 'shape', 'conflicting-shape']
)
def test_inventory_rejects_invalid_recipe_selection(tmp_path, change):
    module, ctx = fixture_preparation(tmp_path)
    recipe = Path(ctx.env['CORPUS_SOURCE_DIR']) / ctx.manifest
    text = recipe.read_text()
    if change == 'empty':
        text = 'cases=[]\n'
    elif change == 'variant':
        text = text.replace('variant="bf16_streamk"', 'variant="unknown"')
    elif change == 'target':
        text = text.replace('targets=["gfx950"]', 'targets=["gfx9999"]', 1)
    elif change == 'shape':
        text = text.replace('m=4096', 'm=0', 1)
    else:
        text += '[[cases]]' + text.split('[[cases]]')[1].replace('m=4096', 'm=2048')
    recipe.write_text(text)
    with pytest.raises(ValueError, match='Manifest Tensile|Multiple shapes'):
        ctx.inventory()


@pytest.mark.parametrize('key', ['TENSILE_SHA', 'NANOBIND_SHA', 'BENCHMARK_SHA'])
def test_source_checks_consume_workflow_environment_pins(tmp_path, key):
    module, ctx = fixture_preparation(tmp_path)
    ctx.env[key] = 'c' * 40
    with mock.patch.object(ctx, 'run', side_effect=FictionalCommands(module, ctx)):
        ctx.sources()
    ctx.env[key] = 'not-an-immutable-pin'
    with mock.patch.object(ctx, 'run', side_effect=FictionalCommands(module, ctx)):
        with pytest.raises(ValueError, match='exact immutable source pin'):
            ctx.sources()


@pytest.mark.parametrize('boundary', ['rocjitsu-revision', 'runtime-version'])
def test_workflow_revision_cannot_pass_source_or_binary_attribution(tmp_path, boundary):
    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx)

    def wrong_revision(label, argv, capture=False, env_overrides=None):
        if label == boundary:
            return (
                'git revision: ' if boundary == 'runtime-version' else ''
            ) + ctx.env['GITHUB_SHA']
        return commands(label, argv, capture, env_overrides)

    with pytest.raises(
        ValueError, match='source revision mismatch|source attribution mismatch'
    ):
        phases(module, ctx, wrong_revision)


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
                'tensile': self.ctx.env['TENSILE_SHA'],
                'nanobind': self.ctx.env['NANOBIND_SHA'],
                'corpus': self.ctx.env['CORPUS_SHA'],
                'rocjitsu': self.ctx.env['BENCHMARK_SHA'],
            }[label.removesuffix('-revision')]
        if label.endswith('-inspect'):
            return '{"installed": []}'
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
            return 'git revision: ' + self.ctx.env['BENCHMARK_SHA']
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
                'source_revision': self.ctx.env['TENSILE_SHA'],
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


def test_legacy_provider_phases_have_separate_receipts_and_build_boundaries(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx)
    expected = (
        ('dependencies', 'python-loader-smoke'),
        ('rocjitsu', 'runtime-version'),
        ('rocisa', 'exact-wheel-import'),
        ('tensilelite-build', 'host-tests'),
        ('tensilelite-generate', 'generate-bf16_streamk'),
    )
    with mock.patch.object(ctx, 'run', side_effect=commands):
        for phase, required_command in expected:
            assert hasattr(ctx, phase.replace('-', '_')), phase
            ctx.start(phase)
            getattr(ctx, phase.replace('-', '_'))()
            ctx.finish(True)
            assert required_command in dict(commands.calls)
            receipt = json.loads((ctx.root / f'phase-{phase}.json').read_text())
            assert receipt['phase'] == phase and receipt['success']
            if phase == 'rocisa':
                assert 'adapter-build' not in dict(commands.calls)
            if phase == 'tensilelite-build':
                assert (ctx.root / 'reports/tensilelite-build.json').is_file()
                assert not ctx.artifacts.exists()
                assert (
                    'TENSILE_CANDIDATE_RUNNER='
                    not in Path(ctx.env['GITHUB_ENV']).read_text()
                )
    assert 'TENSILE_CANDIDATE_RUNNER=' in Path(ctx.env['GITHUB_ENV']).read_text()


def phases(module, ctx, commands):
    with mock.patch.object(ctx, 'run', side_effect=commands):
        for name in (
            'dependencies',
            'rocjitsu',
            'rocisa',
            'tensilelite-build',
            'tensilelite-generate',
        ):
            ctx.start(name)
            try:
                getattr(ctx, name.replace('-', '_'))()
            except Exception:
                ctx.finish(False)
                raise
            ctx.finish(True)


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_prepared_python_loader_smoke_is_early_and_propagates_failure(
    tmp_path, outcome
):
    module, ctx = fixture_preparation(tmp_path)
    ctx.env['LD_LIBRARY_PATH'] = '/toolcache/lib'
    commands = FictionalCommands(module, ctx)
    process = mock.Mock(pid=987654)
    process.wait.return_value = 0 if outcome == 'success' else 1

    def intercept(label, argv, capture=False, env_overrides=None):
        if label == 'python-loader-smoke':
            return module.Preparation.run(
                ctx,
                label,
                argv,
                capture,
                env_overrides,
            )
        return commands(label, argv, capture, env_overrides)

    with mock.patch.object(ctx, 'run', side_effect=intercept), mock.patch.object(
        module.subprocess, 'Popen', return_value=process
    ) as launch:
        ctx.start('dependencies')
        if outcome == 'success':
            ctx.dependencies()
        else:
            with pytest.raises(ValueError, match='python-loader-smoke failed'):
                ctx.dependencies()
    launch.assert_called_once()
    assert launch.call_args.args[0] == [
        str(ctx.python),
        '-B',
        '-c',
        'import ctypes, joblib',
    ]
    assert launch.call_args.kwargs['env']['LD_LIBRARY_PATH'].split(':') == [
        str(ctx.root / 'venv/sdk' / name)
        for name in ('lib', 'lib/rocm_sysdeps/lib', 'lib/llvm/lib', 'lib/host-math/lib')
    ] + ['/toolcache/lib']
    process.wait.assert_called_once_with()
    labels = [label for label, _ in commands.calls]
    if outcome == 'success':
        assert 'dependency-preflight' in labels
    else:
        assert 'dependency-preflight' not in labels
        assert not Path(ctx.env['GITHUB_ENV']).exists()
        assert not (ctx.root / 'ready.json').exists()


def test_setup_uv_cache_path_is_preserved_for_child_commands(tmp_path):
    import sys

    env = environment(tmp_path)
    cache = tmp_path / 'packages'
    cache.mkdir()
    supplied = cache / 'setup-uv-cache'
    env.update(
        ROCJITSU_PACKAGE_CACHE=str(cache),
        UV_CACHE_DIR=str(supplied),
        UV_INDEX_URL='https://untrusted.invalid',
    )
    ctx = load_helper().Preparation(env)
    assert ctx.env['UV_CACHE_DIR'] == str(supplied)
    assert 'UV_INDEX_URL' not in ctx.env
    ctx.start('dependencies')
    observed = ctx.run(
        'cache-path',
        [sys.executable, '-c', 'import os; print(os.environ["UV_CACHE_DIR"])'],
        capture=True,
    )
    assert observed == str(supplied)
    assert 'PIP_CACHE_DIR' not in ctx.env


def test_uv_cache_path_cannot_escape_package_cache(tmp_path):
    env = environment(tmp_path)
    env['UV_CACHE_DIR'] = str(tmp_path / 'outside-cache')
    with pytest.raises(ValueError, match='UV_CACHE_DIR'):
        load_helper().Preparation(env)
    assert not Path(env['UV_CACHE_DIR']).exists()


def test_uv_install_uses_explicit_interpreter_and_compact_origin_inventory(tmp_path):
    module = load_helper()
    env = environment(tmp_path)
    env.update(
        UV_EXECUTABLE='/tools/uv',
        UV_INDEX_URL='https://untrusted.invalid',
        UV_NO_CACHE='1',
        UV_PYTHON='/wrong/python',
    )
    ctx = module.Preparation(env)
    ctx.env['ROCM_PATH'] = str(ctx.root / 'venv/sdk')
    with mock.patch.object(ctx, 'run', return_value=str(ctx.root / 'venv/sdk')):
        ctx.sdk_environment()
    inventory = {
        'installed': [
            {
                'metadata': {
                    'name': 'example',
                    'version': '1.2',
                    'requires_dist': ['dep==1'],
                    'description': 'x' * 300000,
                },
                'installer': 'uv',
                'direct_url': {'url': 'https://example.org/example.whl'},
            }
        ]
    }
    import contextlib
    import importlib.metadata
    import io
    from types import SimpleNamespace

    item = inventory['installed'][0]
    distribution = SimpleNamespace(
        metadata={
            'Name': item['metadata']['name'],
            'Description': item['metadata']['description'],
        },
        version=item['metadata']['version'],
        requires=item['metadata']['requires_dist'],
        read_text=lambda name: {
            'INSTALLER': 'uv\n',
            'direct_url.json': json.dumps(item['direct_url']),
        }.get(name),
    )

    def execute_inventory(label, argv, capture=False):
        if not capture:
            return ''  # The package installation itself is intercepted.
        output = io.StringIO()
        with mock.patch.object(
            importlib.metadata, 'distributions', return_value=[distribution]
        ), contextlib.redirect_stdout(output):
            exec(compile(argv[3], '<installed-metadata>', 'exec'), {})
        return output.getvalue()

    with mock.patch.object(ctx, 'run', side_effect=execute_inventory) as run:
        ctx.install('example', ['example==1.2'], 'https://pypi.org/simple')
    inventory_command = run.call_args_list[1].args[1]
    assert inventory_command[:3] == [ctx.python, '-B', '-c']
    assert 'importlib.metadata' in inventory_command[3]
    argv = list(map(str, run.call_args_list[0].args[1]))
    assert argv[:4] == ['/tools/uv', '--no-config', 'pip', 'install']
    assert argv[argv.index('--python') + 1] == str(ctx.python)
    assert argv[argv.index('--constraint') + 1] == str(
        ctx.root / 'benchmark-constraints.txt'
    )
    assert argv[argv.index('--index-url') + 1] == 'https://pypi.org/simple'
    assert '--compile-bytecode' in argv and '--link-mode=copy' in argv
    assert '--report' not in argv
    assert (
        '--verbose' in argv
    )  # Keep registry/wheel origin diagnostics in the command log.
    assert ctx.env['UV_PYTHON_DOWNLOADS'] == 'never'
    assert all(
        key not in ctx.env for key in ('UV_INDEX_URL', 'UV_NO_CACHE', 'UV_PYTHON')
    )
    report = json.loads((ctx.root / 'reports/example-inventory.json').read_text())
    assert report == [
        {
            'name': 'example',
            'version': '1.2',
            'requires_dist': ['dep==1'],
            'installer': 'uv',
            'direct_url': {'url': 'https://example.org/example.whl'},
        }
    ]


def test_dependencies_use_unseeded_python312_uv_environment(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    commands = FictionalCommands(module, ctx)
    with mock.patch.object(ctx, 'run', side_effect=commands):
        ctx.dependencies()
    calls = dict(commands.calls)
    assert calls['venv'] == [
        '/tools/uv',
        '--no-config',
        'venv',
        '--python',
        '3.12',
        str(ctx.root / 'venv'),
    ]
    for label in ('packages-before', 'packages-resolved', 'packages-check'):
        argv = calls[label]
        assert argv[:3] == ['/tools/uv', '--no-config', 'pip']
        assert argv[argv.index('--python') + 1] == str(ctx.python)
    assert not any(key.startswith('PIP_') for key in ctx.env)
    assert not (ctx.root / 'cache/pip').exists()


def test_intercepted_preparation_orders_exact_owned_wheel_and_four_generators(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    assert list(ctx.inventory()) == [
        'bf16_streamk',
        'bf16_subtile',
        'mxfp8_subtile',
        'mxfp4_streamk',
    ]
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
    for label in (
        'benchmark-requirements',
        'tensile-requirements',
        'wheel-requirements',
    ):
        install = next(args for name, args in commands.calls if name == label)
        assert install[:4] == ['/tools/uv', '--no-config', 'pip', 'install']
        assert install[install.index('--python') + 1] == str(ctx.python)
        assert '--report' not in install
        if label == 'benchmark-requirements':
            assert (
                install[install.index('--index-url') + 1]
                == 'https://rocm.nightlies.amd.com/whl-multi-arch/'
            )
        else:
            assert '--constraint' in install
    wheel = next(args for label, args in commands.calls if label == 'wheel')
    assert wheel[:3] == ['/tools/uv', '--no-config', 'build']
    assert '--wheel' in wheel and '--no-create-gitignore' in wheel
    assert '--config-settings=build.tool-args=-j16' in wheel
    assert '--no-build-isolation' in wheel and '--no-index' in wheel
    assert wheel[wheel.index('--python') + 1] == str(ctx.python)
    install_wheel = next(
        args for label, args in commands.calls if label == 'install-exact-wheel'
    )
    assert install_wheel[:4] == ['/tools/uv', '--no-config', 'pip', 'install']
    assert '--require-hashes' in install_wheel and '--no-deps' in install_wheel
    assert '--no-index' in install_wheel and '--compile-bytecode' not in install_wheel
    receipt = Path(install_wheel[install_wheel.index('-r') + 1]).read_text()
    built_wheel = next((ctx.root / 'wheelhouse').glob('*.whl'))
    assert (
        receipt
        == built_wheel.as_uri() + ' --hash=sha256:' + module.sha256(built_wheel) + '\n'
    )
    import_command = next(
        args for label, args in commands.calls if label == 'exact-wheel-import'
    )
    assert 'archive_info' not in import_command[3]
    assert 'hashlib.sha256(e.read_bytes())' in import_command[3]
    assert 'direct["url"]' in import_command[3]
    assert 'editable' in import_command[3]
    assert 'No wheel-local stinkytofu mapping' in import_command[3]
    assert '"/opt/rocm" not in maps' in import_command[3]
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
    assert all('--parallel' not in args for _, args in commands.calls)
    assert ctx.env['LD_LIBRARY_PATH'].split(':') == [
        str(ctx.root / 'venv/sdk' / p)
        for p in ('lib', 'lib/rocm_sysdeps/lib', 'lib/llvm/lib', 'lib/host-math/lib')
    ]
    assert json.loads((ctx.root / 'ready.json').read_text()) == ctx.inventory()
    exported = Path(ctx.env['GITHUB_ENV']).read_text()
    assert 'CMAKE_BUILD_PARALLEL_LEVEL=' not in exported
    assert (
        'ROCISA_BACKEND=native\n' in exported
        and 'TENSILE_CANDIDATE_ARTIFACTS=' + str(ctx.artifacts) in exported
    )
    query = next(
        args for label, args in commands.calls if label == 'exact-wheel-import'
    )
    compile(query[query.index('-c') + 1], '<wheel-import-check>', 'exec')
    assert (
        'direct["url"]' in query[query.index('-c') + 1]
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


def test_helper_delegates_time_limits_to_workflow():
    import ast
    import inspect

    module = load_helper()
    tree = ast.parse(HELPER.read_text())
    # This simplicity contract covers every helper phase, including diagnostics
    # and the early loader import: no second timeout policy inside Python.
    forbidden = {'PHASE_SECONDS', 'deadline', 'remaining', 'timeout_seconds'}
    assert (
        not {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        & forbidden
    )
    assert (
        not {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        & forbidden
    )
    assert 'timeout_seconds' not in inspect.signature(module.Preparation.run).parameters
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not {'timeout'} & {keyword.arg for keyword in node.keywords}
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'sleep', 'alarm', 'setitimer', 'Timer'}


@pytest.mark.parametrize('signalled', [False, True])
def test_actual_child_failure_preserves_status_output_and_elapsed_receipts(
    tmp_path, signalled
):
    import signal
    import sys

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('rocisa')
    code = -signal.SIGTERM if signalled else 7
    child = 'import os,signal; print("child failure evidence", flush=True); ' + (
        'os.kill(os.getpid(), signal.SIGTERM)' if signalled else 'raise SystemExit(7)'
    )
    with pytest.raises(ValueError, match=f'exit {code}'):
        ctx.run('actual-failure', [sys.executable, '-B', '-c', child])
    ctx.finish(False)
    receipt = json.loads((ctx.root / 'commands.jsonl').read_text())
    assert receipt['exit_code'] == code
    assert receipt['elapsed_seconds'] >= 0
    assert Path(receipt['log']).read_text() == 'child failure evidence\n'
    phase = json.loads((ctx.root / 'phase-rocisa.json').read_text())
    assert phase['success'] is False and phase['elapsed_seconds'] >= 0
    assert not (ctx.root / 'ready.json').exists()


@pytest.mark.parametrize('interrupt', ['SIGINT', 'SIGTERM'])
def test_actual_interrupt_cleans_owned_group_and_records_failed_phase(
    tmp_path, interrupt
):
    import os
    import signal
    import subprocess
    import sys
    import time

    env = environment(tmp_path)
    ready = tmp_path / 'child-pids.json'
    # Both leader and descendant ignore graceful signals. Cleanup must address
    # the owned group, not wait for the leader or use a timed grace period.
    child = '\n'.join(
        [
            'import json,os,pathlib,signal,subprocess,sys',
            'signal.signal(signal.SIGINT, signal.SIG_IGN)',
            'signal.signal(signal.SIGTERM, signal.SIG_IGN)',
            'descendant=subprocess.Popen([sys.executable,"-c","import signal; signal.pause()"] )',
            'pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),descendant.pid]))',
            'signal.pause()',
        ]
    )
    wrapper = '\n'.join(
        [
            'import importlib.util,sys',
            'spec=importlib.util.spec_from_file_location("preparation",sys.argv[1])',
            'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)',
            'child,ready=sys.argv[2:4]',
            'module.Preparation.dependencies=lambda self: self.run("interrupted-child",[sys.executable,"-B","-c",child,ready])',
            'sys.argv=[sys.argv[1],"dependencies"]',
            'module.main()',
        ]
    )
    process = subprocess.Popen(
        [sys.executable, '-B', '-c', wrapper, str(HELPER), child, str(ready)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pids = []
    try:
        until = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < until, 'actual child did not become ready'
            time.sleep(0.01)
        pids = json.loads(ready.read_text())
        process.send_signal(getattr(signal, interrupt))
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode != 0, stdout + stderr
        for pid in pids:
            status = Path(f'/proc/{pid}/stat')
            assert not status.exists() or status.read_text().split()[2] == 'Z'
        root = Path(env['BENCHMARK_DIR'])
        phase = json.loads((root / 'phase-dependencies.json').read_text())
        assert phase['success'] is False and phase['elapsed_seconds'] >= 0
        receipt = json.loads((root / 'commands.jsonl').read_text())
        assert receipt['exit_code'] == -signal.SIGKILL
        assert receipt['interrupted'] is True
        assert receipt['elapsed_seconds'] >= 0
        assert not (root / 'ready.json').exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if not pids and ready.exists():
            pids = json.loads(ready.read_text())
        if pids:
            try:
                os.killpg(pids[0], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_foreign_loader_file_rejected_but_owned_assembler_output_allowed(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    ctx.start('rocisa')
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
    ctx.start('rocisa')
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


@pytest.mark.parametrize('truncated_commands', [False, True])
def test_diagnostics_prioritizes_failure_evidence_under_file_cap(
    tmp_path, truncated_commands
):
    import sys
    import tarfile

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('gluon-verify')
    bulk_logs = [ctx.logs / f'aaa-earlier-{index:03}.log' for index in range(128)]
    for path in bulk_logs:
        path.write_text('unrelated successful command output\n')
    with pytest.raises(ValueError, match='verification failed'):
        ctx.run(
            'verification',
            [
                sys.executable,
                '-c',
                'print("specific verification failure"); raise SystemExit(7)',
            ],
        )
    ctx.finish(False)
    commands = ctx.root / 'commands.jsonl'
    failed_log = Path(json.loads(commands.read_text().splitlines()[-1])['log'])
    if truncated_commands:
        with commands.open('a') as stream:
            stream.write('{"interrupted_record":')
    phase = ctx.root / 'phase-gluon-verify.json'
    report = ctx.root / 'reports/gluon-verification.json'
    report.write_text(json.dumps({'verified': False}))
    critical = {commands, failed_log, phase, report, ctx.root / 'reports/ccache.conf'}
    expected = {path.relative_to(ctx.root).as_posix() for path in critical}
    with mock.patch.object(ctx, 'sources'):
        ctx.diagnostics()
    with tarfile.open(ctx.root / 'preparation-diagnostics.tar.gz') as archive:

        def read_member(name):
            stream = archive.extractfile(name)
            assert stream is not None
            return stream.read()

        assert expected <= set(archive.getnames())
        assert b'specific verification failure' in read_member(
            failed_log.relative_to(ctx.root).as_posix()
        )
        assert json.loads(read_member('phase-gluon-verify.json'))['success'] is False
        assert json.loads(read_member('reports/gluon-verification.json')) == {
            'verified': False
        }
        inventory = json.loads(read_member('diagnostic-inventory.json'))
        assert inventory['file_limit'] == len(inventory['entries']) == 128
        assert inventory['omitted_files'] == len(bulk_logs) + len(critical) - 128
        assert {entry['path'] for entry in inventory['entries']} == set(
            archive.getnames()
        ) - {'diagnostic-inventory.json'}


def test_diagnostics_size_limits_and_symlink_exclusion(tmp_path):
    import tarfile

    module, ctx = fixture_preparation(tmp_path)
    ctx.start('diagnostics')
    (ctx.logs / 'large.log').write_bytes(b'x' * 300000)
    for index in range(130):
        (ctx.logs / f'z-{index:03}.log').write_text('diagnostic output\n')
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
        assert inventory['file_limit'] == len(inventory['entries']) == 128
        assert inventory['omitted_files'] == 4  # Includes the ccache config report.
        assert len(archive.getmembers()) == 129  # Files plus inventory.
        large = next(e for e in inventory['entries'] if e['path'] == 'logs/large.log')
        assert large['truncated'] is True
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


@pytest.mark.parametrize(
    'failed_phase',
    [
        None,
        'rocisa',
        'tensilelite-build',
        'tensilelite-generate',
        'tensile-build',
        'tensile-generate',
        'llama-build',
        'gluon-verify',
    ],
)
def test_grouped_provider_step_runs_serially_and_stops_on_failure(
    tmp_path, failed_phase
):
    import os
    import subprocess
    import yaml

    workflow = yaml.safe_load(
        (HELPER.parents[3] / '.github/workflows/rocjitsu-benchmarks.yml').read_text()
    )
    steps = workflow['jobs']['benchmark']['steps']
    grouped = [step for step in steps if step['name'] == 'Prepare benchmark providers']
    assert len(grouped) == 1
    step = grouped[0]
    assert step['id'] == 'providers'
    phases = [
        'rocisa',
        'tensilelite-build',
        'tensilelite-generate',
        'tensile-build',
        'tensile-generate',
        'llama-build',
        'gluon-verify',
    ]
    python = tmp_path / 'venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n[[ "$2" != "$FAILED_PHASE" ]]\n'
    )
    python.chmod(0o755)
    log = tmp_path / 'calls'
    result = subprocess.run(
        ['bash', '-e', '-c', step['run']],
        env={
            **os.environ,
            'ROCJITSU_VENV': str(python.parents[1]),
            'WORKFLOW_SOURCE_DIR': '/workflow',
            'BENCHMARK_MANIFEST': 'selected recipe.toml',
            'CALLS': str(log),
            'FAILED_PHASE': failed_phase or '',
        },
        capture_output=True,
        text=True,
    )
    expected = (
        phases if failed_phase is None else phases[: phases.index(failed_phase) + 1]
    )
    assert (result.returncode == 0) == (failed_phase is None), result.stderr
    assert log.read_text().splitlines() == [
        '/workflow/scripts/prepare-benchmarks.py '
        + phase
        + (
            ' --manifest selected recipe.toml'
            if phase not in ('rocisa', 'tensilelite-build')
            else ''
        )
        for phase in expected
    ]


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
                .replace('steps.providers.outcome', repr(outcome))
                .replace('&&', 'and')
            )
            assert eval(actual, {'__builtins__': {}}) == (
                success and outcome == 'success'
            )


def test_rocjitsu_still_requires_successful_dependencies(tmp_path):
    _, ctx = fixture_preparation(tmp_path)
    (ctx.root / 'phase-dependencies.json').write_text('{"success":false}')
    with pytest.raises(ValueError, match='Dependencies did not complete'):
        ctx.start('rocjitsu')


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_generator_target_overlay_reaches_child_without_leakage(tmp_path, outcome):
    module, ctx = fixture_preparation(tmp_path, inherited_target='gfx9999')
    assert 'HCC_AMDGPU_TARGET' not in ctx.env
    commands = FictionalCommands(module, ctx)
    base_environment = dict(ctx.env)
    launched = []

    def launch(argv, **kwargs):
        launched.append((argv, kwargs['env']))
        assert kwargs['env'] is not ctx.env
        process = mock.Mock(pid=987654)
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

    with mock.patch.object(module.subprocess, 'Popen', side_effect=launch):
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


@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_command_overlay_does_not_mutate_base_environment(tmp_path, outcome):
    module, ctx = fixture_preparation(tmp_path)
    ctx.start('tensilelite-generate')
    before = dict(ctx.env)
    process = mock.Mock(pid=987654)
    process.wait.return_value = 1 if outcome == 'failure' else 0
    with mock.patch.object(module.subprocess, 'Popen', return_value=process) as launch:
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


@pytest.mark.parametrize(
    'mode',
    ['good', 'extension', 'origin', 'editable', 'mapping', 'library-hash', 'loader'],
)
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
        'archive_info': {},  # uv records origin, not the install-time hash receipt.
    }
    if mode == 'editable':
        direct['dir_info'] = {'editable': True}
    if mode == 'library-hash':
        library.write_bytes(b'Wrong fictional library')
    if mode == 'extension':
        extension.write_bytes(b'Wrong fictional extension')
    if mode == 'origin':
        direct['url'] = 'file:///other-wheel.whl'
    mapped = library if mode != 'mapping' else tmp_path / 'foreign-libstinkytofu.so.0'
    maps = (
        '0000-ffff r-xp 0 00:00 0 ' + str(mapped).replace(' ', chr(92) + '040') + '\n'
    )
    if mode == 'loader':
        maps += '0000-ffff r-xp 0 00:00 0 /opt/rocm/lib/libamdhip64.so\n'
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
