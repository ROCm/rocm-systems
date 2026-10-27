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
        RUNNER_TEMP=str(tmp_path),
        GITHUB_RUN_ID='123',
        GITHUB_RUN_ATTEMPT='1',
        ROCJITSU_SOURCE_DIR=str(tmp_path / 'source/emulation/rocjitsu'),
        CORPUS_SOURCE_DIR=str(tmp_path / 'corpus'),
        TENSILE_SOURCE=str(tmp_path / 'tensile'),
        NANOBIND_SOURCE=str(tmp_path / 'nanobind'),
        TENSILE_SHA=workflow['env']['TENSILE_SHA'],
        NANOBIND_SHA=workflow['env']['NANOBIND_SHA'],
        CORPUS_SHA='a' * 40,
        GITHUB_SHA='b' * 40,
        GITHUB_ENV=str(tmp_path / 'github-env'),
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
    setup += ['Record checked Tensile revision', 'Install benchmark dependencies']
    setup += required[:-1]
    assert all('timeout-minutes' not in by_name[name] for name in setup)
    # Workflow limits are authoritative, not additive or reserved headroom.
    assert {
        step['name']: step['timeout-minutes']
        for step in steps
        if 'timeout-minutes' in step
    } == {
        'Run performance suite': 60,
    }
    assert workflow['jobs']['benchmark']['timeout-minutes'] == 120
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
    assert 'steps.prepare.outcome' in by_name['Run performance suite']['if']
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
        sys, 'argv', ['prepare-benchmarks.py', 'generate', '--manifest', manifest]
    ), mock.patch.object(module.Preparation, 'generate', inventory_only):
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


def test_generate_cli_requires_explicit_manifest_before_preparation(capsys):
    import sys

    module = load_helper()
    with mock.patch.object(
        sys, 'argv', ['prepare-benchmarks.py', 'generate']
    ), mock.patch.object(module, 'Preparation') as preparation:
        with pytest.raises(SystemExit) as error:
            module.main()
    assert error.value.code == 2
    assert '--manifest is required for generate' in capsys.readouterr().err
    preparation.assert_not_called()


@pytest.mark.parametrize('phase', ['dependencies', 'rocjitsu', 'native'])
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
    ), mock.patch.object(module.Preparation, phase, no_inventory):
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


@pytest.mark.parametrize('key', ['TENSILE_SHA', 'NANOBIND_SHA'])
def test_source_checks_consume_workflow_environment_pins(tmp_path, key):
    module, ctx = fixture_preparation(tmp_path)
    ctx.env[key] = 'c' * 40
    with mock.patch.object(ctx, 'run', side_effect=FictionalCommands(module, ctx)):
        ctx.sources()
    ctx.env[key] = 'not-an-immutable-pin'
    with mock.patch.object(ctx, 'run', side_effect=FictionalCommands(module, ctx)):
        with pytest.raises(ValueError, match='exact immutable source pin'):
            ctx.sources()


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


def test_build_worker_policy_reaches_both_builds_wheel_and_export(tmp_path):
    module, ctx = fixture_preparation(tmp_path)
    assert ctx.env['CMAKE_BUILD_PARALLEL_LEVEL'] == '4'
    # Model changing only preparation's owned build policy, not an incoming override.
    ctx.env['CMAKE_BUILD_PARALLEL_LEVEL'] = '2'
    commands = FictionalCommands(module, ctx)
    worker_env = {}

    def intercept(label, argv, capture=False, env_overrides=None):
        if label in ('build', 'adapter-build', 'wheel'):
            worker_env[label] = ctx.env['CMAKE_BUILD_PARALLEL_LEVEL']
        return commands(label, argv, capture, env_overrides)

    phases(module, ctx, intercept)
    builds = {
        label: args[args.index('--parallel') + 1]
        for label, args in commands.calls
        if label in ('build', 'adapter-build')
    }
    assert builds == {'build': '2', 'adapter-build': '2'}
    assert worker_env == {'build': '2', 'wheel': '2', 'adapter-build': '2'}
    assert 'CMAKE_BUILD_PARALLEL_LEVEL=2\n' in Path(ctx.env['GITHUB_ENV']).read_text()
    # Generator workers are a different policy and must remain four.
    for variant in ctx.inventory():
        import yaml

        config = yaml.safe_load(
            (ctx.artifacts / variant / 'candidate.yaml').read_text()
        )
        assert config['GlobalParameters']['CpuThreads'] == 4


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
    ctx.start('native')
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
    phase = json.loads((ctx.root / 'phase-native.json').read_text())
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


def test_diagnostics_size_limits_and_symlink_exclusion(tmp_path):
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
    ctx.start('generate')
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
