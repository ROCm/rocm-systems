#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Attempt-local benchmark preparation. Never publish or change source checkouts."""

import argparse
import ast
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Preparation:
    def __init__(self, environment, manifest=None):
        self.env = dict(environment)
        self.manifest = manifest
        attempt = f"{self.env['GITHUB_RUN_ID']}:{self.env['GITHUB_RUN_ATTEMPT']}"
        require(re.fullmatch(r'\d+:\d+', attempt), 'Invalid attempt identity')
        value = self.env.get('BENCHMARK_DIR', '')
        require(
            value and Path(value).is_absolute(),
            'BENCHMARK_DIR must be nonempty and absolute',
        )
        self.root = Path(value)
        expected = Path(self.env['RUNNER_TEMP']) / (
            'rocjitsu-benchmark-' + attempt.replace(':', '-')
        )
        require(self.root == expected, 'Refusing unowned benchmark directory')
        protected = [self.root]
        for key in (
            'ROCJITSU_SOURCE_DIR',
            'WORKFLOW_SOURCE_DIR',
            'CORPUS_SOURCE_DIR',
            'TENSILE_SOURCE',
            'NANOBIND_SOURCE',
            'FULL_TENSILE_SOURCE',
            'LLAMA_SOURCE',
        ):
            if key not in self.env:
                continue  # Legacy preparation does not require the RDNA sources.
            source = Path(self.env[key])
            require(
                source.is_absolute()
                and not any(p.is_symlink() for p in (source, *source.parents))
                and not source.resolve().is_relative_to(self.root.resolve())
                and not self.root.resolve().is_relative_to(source.resolve()),
                'Benchmark source must remain external to owned scratch without symlinks',
            )
            protected.append(source)
        require(
            not any(path.is_symlink() for path in (self.root, *self.root.parents)),
            'Owned path contains a symlink',
        )
        marker = self.root / '.owned-attempt'
        require(
            marker.is_file()
            and not marker.is_symlink()
            and marker.read_text().strip() == attempt,
            'Missing or mismatched ownership marker',
        )
        require(
            self.root.stat().st_uid == os.getuid(),
            'Benchmark directory has a different owner',
        )
        self.logs = self.root / 'logs'
        self.cwd = self.root / 'cwd'
        self.python = self.root / 'venv/bin/python'
        self.artifacts = self.root / 'tensile-artifacts'
        package_cache = self.root / 'cache'
        if 'ROCJITSU_PACKAGE_CACHE' in self.env:
            package_cache = Path(self.env['ROCJITSU_PACKAGE_CACHE'])
            require(
                package_cache.is_absolute()
                and package_cache.is_dir()
                and not any(
                    p.is_symlink() for p in (package_cache, *package_cache.parents)
                )
                and package_cache.stat().st_uid == os.getuid()
                and all(
                    not package_cache.resolve().is_relative_to(p.resolve())
                    and not p.resolve().is_relative_to(package_cache.resolve())
                    for p in protected
                ),
                'Package cache must be an owned external directory without symlinks, separate from scratch and sources',
            )
        self.uv = self.env.get('UV_EXECUTABLE')
        for key in list(self.env):
            if key.startswith('PIP_') or (
                key.startswith('UV_') and key != 'UV_CACHE_DIR'
            ):
                self.env.pop(key)
        # setup-uv owns the workflow cache path; standalone callers get a default.
        self.env.setdefault('UV_CACHE_DIR', str(package_cache / 'uv'))
        uv_cache = Path(self.env['UV_CACHE_DIR'])
        require(
            uv_cache.is_absolute()
            and uv_cache.resolve().is_relative_to(package_cache.resolve()),
            'UV_CACHE_DIR must remain within the package cache',
        )
        self.env['UV_PYTHON_DOWNLOADS'] = 'never'
        # CMake initializes launchers from these variables for every fresh build,
        # including scikit-build's ROCISA wheel. The pinned llama builder opts out.
        for key in list(self.env):
            if key.startswith('CCACHE_') or key in (
                'CMAKE_C_COMPILER_LAUNCHER',
                'CMAKE_CXX_COMPILER_LAUNCHER',
                'CMAKE_HIP_COMPILER_LAUNCHER',
            ):
                self.env.pop(key)
        base_dir = os.path.commonpath([str(p.resolve()) for p in protected])
        if base_dir == '/':
            base_dir = str(self.root.parent)
        self.env.update(
            CCACHE_DIR=str(package_cache / 'ccache'),
            CCACHE_BASEDIR=base_dir,
            CCACHE_CONFIGPATH=str(self.root / 'reports/ccache.conf'),
            CCACHE_STATSLOG=str(self.logs / 'ccache-stats.log'),
            CMAKE_C_COMPILER_LAUNCHER='ccache',
            CMAKE_CXX_COMPILER_LAUNCHER='ccache',
        )
        for key, directory in {
            'HOME': 'home',
            'TMPDIR': 'tmp',
            'XDG_CACHE_HOME': 'cache',
            'TENSILE_HELPER_CACHE_DIR': 'cache/tensile',
            'TRITON_CACHE_DIR': 'cache/triton',
            'JOBLIB_TEMP_FOLDER': 'tmp/joblib',
        }.items():
            self.env[key] = str(self.root / directory)
        for directory in [
            Path(self.env['UV_CACHE_DIR']),
            Path(self.env['CCACHE_DIR']),
            self.logs,
            self.cwd,
            self.root / 'reports',
            *[
                Path(self.env[k])
                for k in (
                    'HOME',
                    'TMPDIR',
                    'XDG_CACHE_HOME',
                    'TENSILE_HELPER_CACHE_DIR',
                    'TRITON_CACHE_DIR',
                    'JOBLIB_TEMP_FOLDER',
                )
            ],
        ]:
            require(
                not any(path.is_symlink() for path in (directory, *directory.parents)),
                'Attempt directory contains a symlink',
            )
            directory.mkdir(parents=True, exist_ok=True)
            require(not directory.is_symlink(), 'Attempt directory contains a symlink')
        Path(self.env['CCACHE_CONFIGPATH']).write_text(
            'compiler_check = content\nmax_size = 5G\n'
        )
        for key in (
            'LD_PRELOAD',
            'PYTHONPATH',
            'PYTHONHOME',
            'RJ_CONFIG',
            'HSA_MODEL_LIB',
            'HSA_MODEL_TOPOLOGY',
            'HSA_OVERRIDE_GFX_VERSION',
            'HCC_AMDGPU_TARGET',
        ):
            self.env.pop(key, None)
        self.env.update(
            PYTHONDONTWRITEBYTECODE='1',
            PYTHONNOUSERSITE='1',
            ROCISA_BACKEND='native',
            HSA_ENABLE_SDMA_COPY_SIZE_OVERRIDE='0',
        )
        self.phase = None
        self.started = None
        self.sequence = 0

    def start(self, phase):
        require(
            not (self.root / f'phase-{phase}.json').exists(),
            'Refusing to replay a preparation phase',
        )
        self.phase, self.started = phase, time.monotonic()
        self.env['CCACHE_STATSLOG'] = str(self.logs / f'{phase}-ccache-stats.log')
        if phase == 'rocjitsu':
            deps = json.loads((self.root / 'phase-dependencies.json').read_text())
            require(deps['success'], 'Dependencies did not complete')

    def finish(self, success):
        (self.root / f'phase-{self.phase}.json').write_text(
            json.dumps(
                {
                    'phase': self.phase,
                    'success': success,
                    'elapsed_seconds': time.monotonic() - self.started,
                },
                indent=2,
            )
            + '\n'
        )

    def run(self, label, argv, capture=False, env_overrides=None):
        require(not self.cwd.is_symlink(), 'Controlled working directory is symbolic')
        require(
            self.cwd.stat().st_mode & 0o200,
            'Controlled preparation working directory must remain writable',
        )
        require(
            all(
                path.name == 'a.out' and path.is_file() and not path.is_symlink()
                for path in self.cwd.iterdir()
            ),
            'Controlled working directory contains foreign files',
        )
        self.sequence += 1
        output = self.logs / f'{self.phase}-{self.sequence:03}-{label}.log'
        started = time.monotonic()
        code = None
        interrupted = False
        command_env = {**self.env, **(env_overrides or {})}
        with output.open('wb') as stream:
            process = subprocess.Popen(
                [str(arg) for arg in argv],
                cwd=self.cwd,
                env=command_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait()
            except BaseException:
                interrupted = True
                # No helper timer or grace period: on ordinary interruption,
                # stop the whole owned group before reaping its leader. This is
                # best effort; hard termination can prevent Python cleanup.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait()
                raise
            finally:
                with (self.root / 'commands.jsonl').open('a') as log:
                    log.write(
                        json.dumps(
                            {
                                'phase': self.phase,
                                'label': label,
                                'argv': [str(a) for a in argv],
                                'cwd': str(self.cwd),
                                'exit_code': code,
                                'interrupted': interrupted,
                                'HCC_AMDGPU_TARGET': command_env.get(
                                    'HCC_AMDGPU_TARGET'
                                ),
                                'log': str(output),
                                'elapsed_seconds': time.monotonic() - started,
                            }
                        )
                        + '\n'
                    )

        require(code == 0, f'{label} failed (exit {code}); see {output}')
        return output.read_text(errors='replace').strip() if capture else ''

    def sources(self):
        sources = [
            ('tensile', self.env['TENSILE_SOURCE'], self.env['TENSILE_SHA']),
            ('nanobind', self.env['NANOBIND_SOURCE'], self.env['NANOBIND_SHA']),
            ('corpus', self.env['CORPUS_SOURCE_DIR'], self.env['CORPUS_SHA']),
            ('rocjitsu', self.env['ROCJITSU_SOURCE_DIR'], self.env['BENCHMARK_SHA']),
        ]
        for prefix in ('FULL_TENSILE', 'LLAMA'):
            if prefix + '_SOURCE' in self.env or prefix + '_SHA' in self.env:
                sources.append(
                    (
                        prefix.lower().replace('_', '-'),
                        self.env[prefix + '_SOURCE'],
                        self.env[prefix + '_SHA'],
                    )
                )
        for name, path, revision in sources:
            require(
                re.fullmatch(r'[a-f0-9]{40}', revision),
                'Expected exact immutable source pin',
            )
            actual = self.run(
                name + '-revision', ['git', '-C', path, 'rev-parse', 'HEAD'], True
            )
            require(actual == revision, f'{name} source revision mismatch')
            require(
                not self.run(
                    name + '-clean',
                    [
                        'git',
                        '-C',
                        path,
                        'status',
                        '--porcelain',
                        '--untracked-files=all',
                        '--ignore-submodules=none',
                        *(['--ignored'] if name in ('full-tensile', 'llama') else []),
                    ],
                    True,
                ),
                f'{name} source is dirty; no source cleanup attempted',
            )

    def sdk_environment(self):
        sdk = self.run(
            'sdk-path', [self.root / 'venv/bin/rocm-sdk', 'path', '--root'], True
        )
        require(
            sdk
            and Path(sdk).is_absolute()
            and Path(sdk).resolve().is_relative_to((self.root / 'venv').resolve()),
            'SDK must be in the owned venv',
        )
        self.env.update(ROCM_PATH=sdk, HIP_PATH=sdk, PYTHON=str(self.python))
        library_paths = [
            *[
                str(Path(sdk) / p)
                for p in (
                    'lib',
                    'lib/rocm_sysdeps/lib',
                    'lib/llvm/lib',
                    'lib/host-math/lib',
                )
            ],
            *self.env.get('LD_LIBRARY_PATH', '').split(':'),
        ]
        self.env['LD_LIBRARY_PATH'] = ':'.join(
            dict.fromkeys(path for path in library_paths if path)
        )
        self.env['PATH'] = (
            f'{sdk}/bin:{sdk}/lib/llvm/bin:{self.python.parent}:/usr/bin:/bin'
        )

    def pins(self):
        expected = {}
        requirements = (
            Path(self.env['CORPUS_SOURCE_DIR']) / 'benchmarks/requirements.txt'
        )
        for line in requirements.read_text().splitlines():
            if line.strip() and not line.startswith(('#', '--')):
                match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([^\s]+)', line)
                require(match, 'Benchmark requirements must remain exact pins')
                expected[match[1]] = match[2]
        require(expected, 'Benchmark prerequisite pins missing')
        (self.root / 'benchmark-pins.json').write_text(
            json.dumps(expected, indent=2) + '\n'
        )
        code = 'import importlib.metadata as m,json,sys; expected=json.load(open(sys.argv[1])); actual={name:m.version(name) for name in expected}; print(json.dumps(actual)); assert actual==expected, (actual,expected)'
        self.run(
            'benchmark-pins',
            [self.python, '-B', '-c', code, self.root / 'benchmark-pins.json'],
        )
        return expected

    def install(self, label, requirements, index, constrained=True):
        require(
            self.uv and Path(self.uv).is_absolute(),
            'UV_EXECUTABLE must be the absolute setup-uv executable path',
        )
        self.run(
            label,
            [
                self.uv,
                '--no-config',
                'pip',
                'install',
                '--verbose',
                '--python',
                self.python,
                '--index-url',
                index,
                '--compile-bytecode',
                '--link-mode=copy',
                *(
                    ['--constraint', self.root / 'benchmark-constraints.txt']
                    if constrained
                    else []
                ),
                *requirements,
            ],
        )
        installed = json.loads(
            self.run(
                label + '-inspect',
                [
                    self.python,
                    '-B',
                    '-c',
                    '\n'.join(
                        [
                            'import importlib.metadata as m,json',
                            'installed=[]',
                            'for d in m.distributions():',
                            ' direct=d.read_text("direct_url.json")',
                            ' installed.append({"metadata":{"name":d.metadata["Name"],"version":d.version,"requires_dist":d.requires or []},"installer":(d.read_text("INSTALLER") or "").strip() or None,"direct_url":json.loads(direct) if direct else None})',
                            'print(json.dumps({"installed":installed}))',
                        ]
                    ),
                ],
                True,
            )
        )['installed']
        compact = [
            {
                'name': item['metadata']['name'],
                'version': item['metadata']['version'],
                'requires_dist': item['metadata'].get('requires_dist', []),
                'installer': item.get('installer'),
                'direct_url': item.get('direct_url'),
            }
            for item in installed
        ]
        (self.root / 'reports' / (label + '-inventory.json')).write_text(
            json.dumps(compact, indent=2) + '\n'
        )

    def dependencies(self):
        self.sources()
        require(not (self.root / 'venv').exists(), 'Refusing to reuse an old venv')
        self.run(
            'venv',
            [self.uv, '--no-config', 'venv', '--python', '3.12', self.root / 'venv'],
        )
        self.run(
            'packages-before',
            [
                self.uv,
                '--no-config',
                'pip',
                'list',
                '--python',
                self.python,
                '--format=json',
            ],
        )
        self.install(
            'benchmark-requirements',
            ['-r', Path(self.env['CORPUS_SOURCE_DIR']) / 'benchmarks/requirements.txt'],
            'https://rocm.nightlies.amd.com/whl-multi-arch/',
            constrained=False,
        )
        expected = self.pins()
        constraint = self.root / 'benchmark-constraints.txt'
        constraint.write_text(
            ''.join(f'{name}=={version}\n' for name, version in expected.items())
        )
        tensile = Path(self.env['TENSILE_SOURCE']) / 'projects/hipblaslt/tensilelite'
        self.install(
            'tensile-requirements',
            ['-r', tensile / 'requirements.txt'],
            'https://pypi.org/simple',
        )
        build_requires = tomllib.loads((tensile / 'rocisa/pyproject.toml').read_text())[
            'build-system'
        ]['requires']
        self.install('wheel-requirements', build_requires, 'https://pypi.org/simple')
        self.pins()
        self.run(
            'packages-check',
            [self.uv, '--no-config', 'pip', 'check', '--python', self.python],
        )
        self.run(
            'packages-resolved',
            [
                self.uv,
                '--no-config',
                'pip',
                'list',
                '--python',
                self.python,
                '--format=json',
            ],
        )
        (self.root / 'reports/indexes.json').write_text(
            json.dumps(
                {
                    'benchmark': 'https://rocm.nightlies.amd.com/whl-multi-arch/',
                    'tensile_and_build': 'https://pypi.org/simple',
                }
            )
            + '\n'
        )
        self.sdk_environment()
        self.run(
            'python-loader-smoke',
            [self.python, '-B', '-c', 'import ctypes, joblib'],
        )
        self.preflight()
        self.export_environment()
        self.sources()

    def export_environment(self):
        keys = [
            'ROCM_PATH',
            'HIP_PATH',
            'PYTHON',
            'TENSILE_SOURCE',
            'PATH',
            'LD_LIBRARY_PATH',
            'ROCISA_BACKEND',
            'PYTHONDONTWRITEBYTECODE',
            'PYTHONNOUSERSITE',
            'HOME',
            'TMPDIR',
            'XDG_CACHE_HOME',
            'TRITON_CACHE_DIR',
            'TENSILE_HELPER_CACHE_DIR',
            'JOBLIB_TEMP_FOLDER',
            'HSA_ENABLE_SDMA_COPY_SIZE_OVERRIDE',
        ]
        with Path(self.env['GITHUB_ENV']).open('a') as output:
            for key in keys:
                value = self.env[key]
                require(
                    '\n' not in value and '\r' not in value, 'Invalid environment value'
                )
                output.write(f'{key}={value}\n')

    def rocjitsu(self):
        self.sources()
        self.sdk_environment()
        self.run(
            'configure',
            [
                'cmake',
                '-S',
                self.env['ROCJITSU_SOURCE_DIR'],
                '-B',
                self.root / 'build',
                '-G',
                'Ninja',
                '-DCMAKE_BUILD_TYPE=Release',
                '-DBUILD_TESTING=OFF',
                '-DLTO=OFF',
                '-DRJ_ENABLE_ASAN=OFF',
                '-DRJ_ENABLE_MSAN=OFF',
                '-DRJ_ENABLE_TSAN=OFF',
                '-DRJ_ENABLE_UBSAN=OFF',
                f'-DROCM_PATH={self.env["ROCM_PATH"]}',
                f'-DCMAKE_C_COMPILER={self.env["ROCM_PATH"]}/lib/llvm/bin/amdclang',
                f'-DCMAKE_CXX_COMPILER={self.env["ROCM_PATH"]}/lib/llvm/bin/amdclang++',
            ],
        )
        self.run(
            'build',
            [
                'cmake',
                '--build',
                self.root / 'build',
                '--target',
                'rocjitsu_bin',
                'rocjitsu_shared',
            ],
        )
        version = self.run(
            'runtime-version',
            [self.root / 'build/tools/rocjitsu/rocjitsu', '--version'],
            True,
        )
        require(
            f'git revision: {self.env["BENCHMARK_SHA"]}' in version,
            'Built runtime source attribution mismatch',
        )
        self.binary_inventory(
            'rocjitsu-build', ['build/tools/rocjitsu/rocjitsu', 'build/librocjitsu.so']
        )
        self.sources()

    def binary_inventory(self, name, paths):
        inventory = {}
        for relative in paths:
            path = self.root / relative
            require(
                path.is_file() and not path.is_symlink(),
                f'Built binary missing or symbolic: {relative}',
            )
            inventory[relative] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        (self.root / 'reports' / (name + '.json')).write_text(
            json.dumps(inventory, indent=2) + '\n'
        )

    def cmake_flags(self):
        sdk = self.env['ROCM_PATH']
        return [
            f'-DROCM_PATH={sdk}',
            f'-DCMAKE_PREFIX_PATH={sdk}',
            f'-DCMAKE_CXX_COMPILER={sdk}/bin/amdclang++',
            f'-DROCmCMakeBuildTools_DIR={sdk}/share/rocmcmakebuildtools/cmake',
            f'-DLLVM_DIR={sdk}/lib/llvm/lib/cmake/llvm',
        ]

    def preflight(self):
        self.run('ccache-version', ['ccache', '--version'])
        for relative in (
            'lib/cmake/hip/hip-config.cmake',
            'share/rocmcmakebuildtools/cmake/ROCmCMakeBuildToolsConfig.cmake',
            'lib/llvm/lib/cmake/llvm/LLVMConfig.cmake',
            'lib/llvm/lib/libLLVMObjectYAML.a',
            'lib/llvm/lib/libomp.so',
        ):
            require(
                (Path(self.env['ROCM_PATH']) / relative).is_file(),
                f'Missing pinned SDK prerequisite: {relative}',
            )
        version = self.run('cmake-version', ['cmake', '--version'], True)
        match = re.search(r'cmake version (\d+)\.(\d+)\.(\d+)', version)
        require(
            match and tuple(map(int, match.groups())) >= (3, 25, 2),
            'CMake >=3.25.2 required',
        )
        self.run(
            'compiler-version',
            [Path(self.env['ROCM_PATH']) / 'bin/amdclang++', '--version'],
        )
        resources = {
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'disk_free_bytes': shutil.disk_usage(self.root).free,
            'memory': Path('/proc/meminfo').read_text(),
        }
        (self.root / 'reports/resources.json').write_text(
            json.dumps(resources, indent=2) + '\n'
        )
        preflight = self.root / 'dependency-preflight'
        preflight.mkdir()
        (preflight / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.25.2)\nproject(preflight LANGUAGES CXX)\nfind_package(hip REQUIRED CONFIG)\nfind_package(ROCmCMakeBuildTools REQUIRED CONFIG)\nfind_package(LLVM REQUIRED CONFIG)\nif(NOT TARGET LLVMObjectYAML)\n message(FATAL_ERROR "LLVMObjectYAML required")\nendif()\nfind_package(ZLIB REQUIRED)\nfind_package(OpenMP REQUIRED COMPONENTS CXX)\n'
        )
        flags = self.cmake_flags()
        self.run(
            'dependency-preflight',
            [
                'cmake',
                '-S',
                preflight,
                '-B',
                preflight / 'build',
                '-G',
                'Ninja',
                *flags,
            ],
        )

    def require_phase(self, phase):
        require(
            json.loads((self.root / f'phase-{phase}.json').read_text())['success'],
            f'{phase} preparation did not complete',
        )

    def fresh_outputs(self, *paths):
        for path in paths:
            require(
                not path.exists() and not path.is_symlink(),
                'Refusing stale preparation output',
            )

    def rocisa(self):
        require(
            json.loads((self.root / 'phase-rocjitsu.json').read_text())['success'],
            'Rocjitsu setup did not complete',
        )
        self.sources()
        self.sdk_environment()
        self.pins()
        sdk = self.env['ROCM_PATH']
        nanobind_dir = self.run(
            'nanobind-dir', [self.python, '-m', 'nanobind', '--cmake_dir'], True
        )
        require(
            Path(nanobind_dir).is_relative_to(self.root / 'venv')
            and (Path(nanobind_dir) / 'nanobind-config.cmake').is_file(),
            'Installed nanobind CMake config missing',
        )
        wheelhouse = self.root / 'wheelhouse'
        wheelhouse.mkdir()
        self.run(
            'wheel',
            [
                self.uv,
                '--no-config',
                'build',
                '--wheel',
                '--python',
                self.python,
                '--no-index',
                '--no-build-isolation',
                '--no-create-gitignore',
                '--out-dir',
                wheelhouse,
                '--config-settings=build.tool-args=-j16',
                f'--config-settings=build-dir={self.root / "rocisa-build"}',
                f'--config-settings=cmake.define.ROCM_PATH={sdk}',
                f'--config-settings=cmake.define.CMAKE_PREFIX_PATH={sdk}',
                f'--config-settings=cmake.define.CMAKE_CXX_COMPILER={sdk}/bin/amdclang++',
                f'--config-settings=cmake.define.nanobind_DIR={nanobind_dir}',
                Path(self.env['TENSILE_SOURCE'])
                / 'projects/hipblaslt/tensilelite/rocisa',
            ],
        )
        wheels = list(wheelhouse.glob('rocisa-*.whl'))
        require(len(wheels) == 1, 'Expected exactly one built rocisa wheel')
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            require(archive.testzip() is None, 'Wheel CRC failure')
            extensions = [
                name
                for name in archive.namelist()
                if name.startswith('rocisa/_rocisa.') and name.endswith('.so')
            ]
            require(len(extensions) == 1, 'Expected one native rocisa extension')
            extension_sha = hashlib.sha256(archive.read(extensions[0])).hexdigest()
            library_hashes = [
                hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if Path(name).name.startswith('libstinkytofu.so')
            ]
            require(library_hashes, 'Wheel-local stinkytofu library missing')
        (self.root / 'reports/wheel.json').write_text(
            json.dumps(
                {
                    'path': str(wheel),
                    'sha256': sha256(wheel),
                    'extension_sha256': extension_sha,
                },
                indent=2,
            )
            + '\n'
        )
        self.sources()
        receipt = self.root / 'reports/rocisa-wheel-requirements.txt'
        receipt.write_text(wheel.as_uri() + ' --hash=sha256:' + sha256(wheel) + '\n')
        self.run(
            'install-exact-wheel',
            [
                self.uv,
                '--no-config',
                'pip',
                'install',
                '--python',
                self.python,
                '--no-index',
                '--no-deps',
                '--require-hashes',
                '--link-mode=copy',
                '-r',
                receipt,
            ],
        )
        code = '\n'.join(
            [
                'import hashlib,importlib.metadata,json,pathlib,sys,rocisa',
                'root=pathlib.Path(sys.argv[1]).resolve(); p=pathlib.Path(rocisa.__file__).resolve(); e=pathlib.Path(rocisa._rocisa.__file__).resolve()',
                'assert p.is_relative_to(root) and e.is_relative_to(root)',
                'assert hashlib.sha256(e.read_bytes()).hexdigest()==sys.argv[2]',
                'direct=json.loads(importlib.metadata.distribution("rocisa").read_text("direct_url.json"))',
                'assert not direct.get("dir_info",{}).get("editable",False)',
                'assert direct["url"]==pathlib.Path(sys.argv[3]).as_uri()',
                'maps=pathlib.Path("/proc/self/maps").read_text(); assert "/opt/rocm" not in maps',
                'libraries=[]',
                'for line in maps.splitlines():',
                ' if "libstinkytofu.so" in line:',
                '  path=pathlib.Path(line.split(maxsplit=5)[-1].replace(chr(92)+"040"," ")); assert path.resolve().is_relative_to(root)',
                '  assert hashlib.sha256(path.read_bytes()).hexdigest() in json.loads(sys.argv[4]); libraries.append(str(path))',
                'assert libraries, "No wheel-local stinkytofu mapping"',
                'print(json.dumps({"rocisa":str(p),"extension":str(e),"libraries":libraries,"maps":maps}))',
            ]
        )
        self.run(
            'exact-wheel-import',
            [
                self.python,
                '-B',
                '-c',
                code,
                self.root / 'venv',
                extension_sha,
                wheel,
                json.dumps(library_hashes),
            ],
        )
        self.run(
            'packages-check',
            [self.uv, '--no-config', 'pip', 'check', '--python', self.python],
        )
        self.pins()
        self.sources()

    def tensilelite_build(self):
        self.require_phase('rocisa')
        self.sources()
        self.sdk_environment()
        self.pins()
        flags = self.cmake_flags()
        self.fresh_outputs(self.root / 'tensile-native')
        require(
            (
                Path(self.env['NANOBIND_SOURCE'])
                / 'ext/robin_map/include/tsl/robin_map.h'
            ).is_file(),
            'Pinned native nanobind submodule missing',
        )
        self.run(
            'adapter-configure',
            [
                'cmake',
                '-S',
                Path(self.env['CORPUS_SOURCE_DIR'])
                / 'corpus/benchmarks/tensile_candidates',
                '-B',
                self.root / 'tensile-native',
                '-G',
                'Ninja',
                '-DCMAKE_BUILD_TYPE=Release',
                *flags,
                f'-DTENSILE_SOURCE={self.env["TENSILE_SOURCE"]}',
                f'-DPython_EXECUTABLE={self.python}',
                '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
                f'-DFETCHCONTENT_SOURCE_DIR_NANOBIND={self.env["NANOBIND_SOURCE"]}',
                '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
                '-DFETCHCONTENT_UPDATES_DISCONNECTED=ON',
            ],
        )
        self.run(
            'adapter-build',
            [
                'cmake',
                '--build',
                self.root / 'tensile-native',
                '--target',
                'tensile_candidate',
                'tensile_candidate_host_tests',
            ],
        )
        self.run(
            'host-tests', [self.root / 'tensile-native/tensile_candidate_host_tests']
        )
        self.binary_inventory(
            'tensilelite-build',
            [
                'tensile-native/tensile_candidate',
                'tensile-native/tensile_candidate_host_tests',
            ],
        )
        self.pins()
        self.run(
            'packages-after',
            [
                self.uv,
                '--no-config',
                'pip',
                'list',
                '--python',
                self.python,
                '--format=json',
            ],
        )
        self.sources()

    def recipe(self):
        require(self.manifest, 'Manifest is required for inventory')
        relative = Path(self.manifest)
        require(
            not relative.is_absolute()
            and '..' not in relative.parts
            and relative.suffix == '.toml',
            'Manifest must be a corpus-relative TOML path without traversal',
        )
        corpus = Path(self.env['CORPUS_SOURCE_DIR'])
        manifest = corpus / relative
        require(
            not any(path.is_symlink() for path in (manifest, *manifest.parents))
            and manifest.resolve().is_relative_to(corpus.resolve())
            and manifest.is_file(),
            'Manifest must be a contained regular file without symlinks',
        )
        return tomllib.loads(manifest.read_text())

    def full_tensile_provider(self):
        corpus = Path(self.env['CORPUS_SOURCE_DIR'])
        path = corpus / 'corpus/benchmarks/full_tensile/prepare.py'
        require(
            path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents))
            and path.resolve().is_relative_to(corpus.resolve()),
            'Full Tensile provider must be a contained regular file',
        )
        spec = importlib.util.spec_from_file_location('full_tensile_preparation', path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # Do not let importlib write __pycache__ into the preserved corpus.
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        return module

    def rdna_inventory(self):
        recipe = self.recipe()
        provider = self.full_tensile_provider()
        selected = {}
        for case in recipe['cases']:
            if case['workload'] != 'full_tensile':
                continue
            require(case['targets'], 'Full Tensile selection needs targets')
            for target in case['targets']:
                variant = provider.validate_parameters(case['params'], target)
                selected[target, variant] = dict(
                    target=target, variant=variant, parameters=case['params']
                )
        return list(selected.values())

    def validate_rdna_artifacts(self, selected):
        artifacts = self.root / 'full-tensile-artifacts'
        require(
            artifacts.is_dir() and not artifacts.is_symlink(),
            'Full Tensile artifact root missing or symbolic',
        )
        expected_cells = {(item['target'], item['variant']) for item in selected}
        actual_cells = set()
        for target in artifacts.iterdir():
            require(
                target.is_dir() and not target.is_symlink(), 'Invalid artifact target'
            )
            for variant in target.iterdir():
                require(
                    variant.is_dir() and not variant.is_symlink(),
                    'Invalid artifact variant',
                )
                actual_cells.add((target.name, variant.name))
        require(actual_cells == expected_cells, 'Full Tensile cell inventory mismatch')
        generator = (
            Path(self.env['CORPUS_SOURCE_DIR'])
            / 'corpus/benchmarks/full_tensile/prepare.py'
        )
        for item in selected:
            output = artifacts / item['target'] / item['variant']
            metadata_path = output / 'artifacts.json'
            require(
                self.diagnostic_file(metadata_path),
                'Invalid full Tensile metadata path',
            )
            metadata = json.loads(metadata_path.read_text())
            expected = dict(
                source_revision=self.env['FULL_TENSILE_SHA'],
                target=item['target'],
                parameters=item['parameters'],
                generator_cpu_threads=1,
                generator_sha256=sha256(generator),
            )
            require(
                all(metadata.get(k) == v for k, v in expected.items()),
                'Full Tensile artifact provenance mismatch',
            )
            files = metadata.get('files')
            objects = {
                p.name for p in output.iterdir() if p.suffix in ('.co', '.hsaco')
            }
            require(
                isinstance(files, dict)
                and objects
                and set(files) == objects | {'TensileLibrary.dat', 'selection.yaml'},
                'Full Tensile artifact file inventory mismatch',
            )
            for name, digest in files.items():
                path = output / name
                require(
                    Path(name).name == name
                    and self.diagnostic_file(path)
                    and sha256(path) == digest,
                    'Full Tensile artifact hash/path mismatch',
                )

    def inventory(self):
        recipe = self.recipe()
        candidates_file = (
            Path(self.env['CORPUS_SOURCE_DIR'])
            / 'corpus/benchmarks/tensile_candidates/prepare.py'
        )
        try:
            declaration = next(
                (
                    node
                    for node in ast.parse(candidates_file.read_text()).body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Name) and t.id == 'CANDIDATES'
                        for t in node.targets
                    )
                ),
                None,
            )
            supported = ast.literal_eval(declaration.value) if declaration else None
        except (ValueError, SyntaxError) as error:
            raise ValueError(
                'Unsupported CANDIDATES declaration: expected a nonempty literal dict'
            ) from error
        if not isinstance(supported, dict) or not supported:
            raise ValueError(
                'Unsupported CANDIDATES declaration: expected a nonempty literal dict'
            )
        selected = {}
        for case in recipe['cases']:
            if case['workload'] != 'tensile_candidate':
                continue
            params = case['params']
            variant = params['variant']
            shape = [params[key] for key in ('m', 'n', 'k')]
            require(
                variant in supported and case['targets'] == [supported[variant][0]],
                'Manifest Tensile target/variant mismatch',
            )
            require(
                all(type(size) is int and size > 0 for size in shape),
                'Manifest Tensile shape invalid',
            )
            upstream_file = supported[variant][1]
            template = candidates_file.parent / 'upstream' / upstream_file
            item = {
                'target': case['targets'][0],
                'shape': shape,
                'upstream_file': upstream_file,
                'upstream_sha256': sha256(template),
            }
            require(
                variant not in selected or selected[variant] == item,
                'Multiple shapes for one variant require a reviewed preparation change',
            )
            selected[variant] = item
        require(selected, 'Manifest Tensile selection is empty')
        return {
            variant: selected[variant] for variant in supported if variant in selected
        }

    def validate_artifacts(self, selected):
        require(
            self.artifacts.is_dir() and not self.artifacts.is_symlink(),
            'Artifact directory missing or symbolic',
        )
        require(
            {p.name for p in self.artifacts.iterdir()} == set(selected),
            'Generated variant inventory mismatch',
        )
        for variant, expected in selected.items():
            output = self.artifacts / variant
            require(
                output.is_dir() and not output.is_symlink(),
                'Artifact variant path mismatch',
            )
            require(
                (output / 'artifacts.json').is_file()
                and not (output / 'artifacts.json').is_symlink(),
                'Artifact metadata path mismatch',
            )
            metadata = json.loads((output / 'artifacts.json').read_text())
            require(
                all(
                    metadata.get(k) == v
                    for k, v in {
                        'variant': variant,
                        'source_revision': self.env['TENSILE_SHA'],
                        **expected,
                    }.items()
                ),
                'Artifact provenance mismatch',
            )
            files = metadata.get('files')
            actual = {
                p.name for p in output.iterdir() if p.suffix in ('.co', '.hsaco')
            } | {'TensileLibrary.yaml'}
            require(
                isinstance(files, dict) and set(files) == actual and len(actual) > 1,
                'Artifact file inventory mismatch',
            )
            for filename, expected_hash in files.items():
                path = output / filename
                require(
                    path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(output.resolve())
                    and sha256(path) == expected_hash,
                    'Artifact hash/path mismatch',
                )
            require(
                not (output / 'candidate.yaml').is_symlink()
                and sha256(output / 'candidate.yaml') == metadata['config_sha256'],
                'Generated config hash mismatch',
            )
            import yaml

            config = yaml.safe_load((output / 'candidate.yaml').read_text())
            require(
                config['GlobalParameters']['CpuThreads'] == 4,
                'Generator CpuThreads must remain four',
            )

    def tensilelite_generate(self):
        self.require_phase('tensilelite-build')
        self.sources()
        self.sdk_environment()
        selected = self.inventory()
        require(not self.artifacts.exists(), 'Refusing stale generated artifacts')
        for variant, expected in selected.items():
            self.run(
                'generate-' + variant,
                [
                    self.python,
                    '-B',
                    Path(self.env['CORPUS_SOURCE_DIR'])
                    / 'corpus/benchmarks/tensile_candidates/generate.py',
                    '--source-root',
                    self.env['TENSILE_SOURCE'],
                    '--output',
                    self.artifacts,
                    '--variant',
                    variant,
                    '--shape',
                    *map(str, expected['shape']),
                ],
                env_overrides={'HCC_AMDGPU_TARGET': expected['target']},
            )
        self.validate_artifacts(selected)
        self.sources()
        self.pins()
        (self.root / 'ready.json').write_text(json.dumps(selected, indent=2) + '\n')
        self.env['TENSILE_CANDIDATE_ARTIFACTS'] = str(self.artifacts)
        self.env['TENSILE_CANDIDATE_RUNNER'] = str(
            self.root / 'tensile-native/tensile_candidate'
        )
        self.export_environment()
        with Path(self.env['GITHUB_ENV']).open('a') as stream:
            stream.write(
                f'TENSILE_CANDIDATE_ARTIFACTS={self.artifacts}\nTENSILE_CANDIDATE_RUNNER={self.env["TENSILE_CANDIDATE_RUNNER"]}\n'
            )

    def validate_llama_build(self):
        build = self.root / 'llama-hip'
        binary = build / 'llama_bounded_ops'
        receipt_path = build / 'benchmark-build.json'
        require(
            self.diagnostic_file(binary) and self.diagnostic_file(receipt_path),
            'Invalid llama binary/receipt path',
        )
        receipt = json.loads(receipt_path.read_text())
        expected = dict(
            schema='rocjitsu.benchmark.llama-build.v1',
            source_pin=self.env['LLAMA_SHA'],
            source=self.env['LLAMA_SOURCE'],
            backend='hip',
            targets=['gfx1100', 'gfx1201'],
            adapter_sha256=sha256(
                Path(self.env['CORPUS_SOURCE_DIR'])
                / 'corpus/benchmarks/llama/runner.cpp'
            ),
        )
        require(
            all(receipt.get(k) == v for k, v in expected.items()),
            'Llama build receipt provenance mismatch',
        )

    def provider_setup(self, prerequisite, *outputs):
        self.require_phase(prerequisite)
        for prefix in ('FULL_TENSILE', 'LLAMA'):
            require(
                self.env.get(prefix + '_SOURCE') and self.env.get(prefix + '_SHA'),
                'RDNA preparation requires separate pinned source roots',
            )
        self.sources()
        selected = self.rdna_inventory()  # Validate all fixed shapes before any build.
        workloads = {case['workload'] for case in self.recipe()['cases']}
        self.sdk_environment()
        self.pins()
        provider = Path(self.env['CORPUS_SOURCE_DIR']) / 'corpus/benchmarks'
        self.fresh_outputs(*outputs)
        return selected, workloads, provider

    def provider_complete(self, exports=None):
        self.pins()
        self.run(
            'provider-packages-check',
            [self.uv, '--no-config', 'pip', 'check', '--python', self.python],
        )
        self.sources()
        if exports:
            with Path(self.env['GITHUB_ENV']).open('a') as output:
                for key, value in exports.items():
                    require(
                        '\n' not in value and '\r' not in value,
                        'Invalid environment value',
                    )
                    output.write(f'{key}={value}\n')

    def tensile_build(self):
        full_build = self.root / 'full-tensile-native'
        selected, _, provider = self.provider_setup('tensilelite-generate', full_build)
        if selected:
            self.install(
                'full-tensile-requirements',
                [
                    '-r',
                    Path(self.env['FULL_TENSILE_SOURCE'])
                    / 'shared/tensile/requirements.txt',
                ],
                'https://pypi.org/simple',
            )
            self.run(
                'full-tensile-configure',
                [
                    'cmake',
                    '-S',
                    provider / 'full_tensile',
                    '-B',
                    full_build,
                    '-G',
                    'Ninja',
                    '-DCMAKE_BUILD_TYPE=Release',
                    f'-DTENSILE_SOURCE={self.env["FULL_TENSILE_SOURCE"]}',
                    f'-DCMAKE_PREFIX_PATH={self.env["ROCM_PATH"]}',
                    f'-DCMAKE_CXX_COMPILER={self.env["ROCM_PATH"]}/bin/amdclang++',
                ],
            )
            self.run(
                'full-tensile-build',
                [
                    'cmake',
                    '--build',
                    full_build,
                    '--target',
                    'full_tensile_runner',
                    '--parallel',
                    '16',
                ],
            )
            binary = full_build / 'full_tensile_runner'
            require(self.diagnostic_file(binary), 'Invalid full Tensile runner path')
            self.binary_inventory(
                'full-tensile-build', ['full-tensile-native/full_tensile_runner']
            )
        self.provider_complete()

    def tensile_generate(self):
        full_artifacts = self.root / 'full-tensile-artifacts'
        selected, _, provider = self.provider_setup('tensile-build', full_artifacts)
        exports = {}
        if selected:
            for item in selected:
                self.run(
                    'full-tensile-generate-' + item['target'] + '-' + item['variant'],
                    [
                        self.python,
                        '-B',
                        provider / 'full_tensile/prepare.py',
                        '--source-root',
                        self.env['FULL_TENSILE_SOURCE'],
                        '--sdk-root',
                        self.env['ROCM_PATH'],
                        '--output',
                        full_artifacts,
                        '--target',
                        item['target'],
                        '--variant',
                        item['variant'],
                    ],
                    env_overrides={'HCC_AMDGPU_TARGET': item['target']},
                )
            self.validate_rdna_artifacts(selected)
            binary = self.root / 'full-tensile-native/full_tensile_runner'
            require(self.diagnostic_file(binary), 'Invalid full Tensile runner path')
            exports.update(
                FULL_TENSILE_ARTIFACTS=str(full_artifacts),
                FULL_TENSILE_RUNNER=str(binary),
            )
        self.provider_complete(exports)

    def llama_build(self):
        llama_build = self.root / 'llama-hip'
        _, workloads, provider = self.provider_setup('tensile-generate', llama_build)
        exports = {}
        if any(name.startswith('llama_') for name in workloads):
            self.run(
                'llama-prepare',
                [
                    self.python,
                    '-B',
                    provider / 'llama/prepare.py',
                    '--source',
                    self.env['LLAMA_SOURCE'],
                    '--build',
                    llama_build,
                    '--backend',
                    'hip',
                    '--rocm-path',
                    self.env['ROCM_PATH'],
                ],
            )
            self.validate_llama_build()
            exports['ROCJITSU_LLAMA_BINARY'] = str(llama_build / 'llama_bounded_ops')
        self.provider_complete(exports)

    def gluon_verify(self):
        selected, workloads, provider = self.provider_setup(
            'llama-build',
            self.root / 'reports/gluon-source.json',
            self.root / 'reports/providers-ready.json',
        )
        if any(name.startswith('gluon_') for name in workloads):
            self.run(
                'gluon-prepare',
                [
                    self.python,
                    '-B',
                    provider / 'gluon/prepare.py',
                    '--upstream',
                    '--output',
                    self.root / 'reports/gluon-source.json',
                ],
            )
        self.provider_complete()
        (self.root / 'reports/providers-ready.json').write_text(
            json.dumps(
                dict(
                    manifest=self.manifest,
                    full_tensile=selected,
                    workloads=sorted(workloads),
                ),
                indent=2,
            )
            + '\n'
        )

    def diagnostic_file(self, path):
        return (
            self.root in path.parents
            and not any(parent.is_symlink() for parent in (path, *path.parents))
            and path.resolve().is_relative_to(self.root.resolve())
            and path.is_file()
        )

    def diagnostics(self):
        for stats in sorted(self.logs.glob('*ccache-stats.log')):
            if not self.diagnostic_file(stats):
                continue
            try:
                self.run(
                    'ccache-summary-' + stats.stem,
                    ['ccache', '--show-log-stats', '--verbose'],
                    env_overrides={'CCACHE_STATSLOG': str(stats)},
                )
            except (ValueError, OSError) as error:
                (self.root / 'reports/ccache-summary-error.txt').write_text(str(error))
        try:
            self.sources()
        except (ValueError, OSError) as error:
            (self.root / 'reports/source-preservation-error.txt').write_text(str(error))
        files = [
            *self.logs.glob('*.log'),
            *list((self.root / 'reports').glob('*')),
            *self.root.glob('phase-*.json'),
            self.root / 'commands.jsonl',
            self.root / 'ready.json',
        ]
        files.extend(
            self.root / p
            for p in (
                'build/CMakeCache.txt',
                'rocisa-build/CMakeCache.txt',
                'tensile-native/CMakeCache.txt',
                'full-tensile-native/CMakeCache.txt',
                'llama-hip/CMakeCache.txt',
                'llama-hip/configure.log',
                'llama-hip/build.log',
                'llama-hip/benchmark-build.json',
                'dependency-preflight/build/CMakeCache.txt',
            )
        )
        if self.artifacts.is_dir() and not self.artifacts.is_symlink():
            for output in self.artifacts.iterdir():
                if output.is_dir() and not output.is_symlink():
                    files.extend(
                        output / name
                        for name in (
                            'generate.log',
                            'artifacts.json',
                            'source.json',
                            'candidate.yaml',
                            'generation-command.json',
                        )
                    )
        full_artifacts = self.root / 'full-tensile-artifacts'
        if full_artifacts.is_dir() and not full_artifacts.is_symlink():
            for target in full_artifacts.iterdir():
                if target.is_dir() and not target.is_symlink():
                    for variant in target.iterdir():
                        if variant.is_dir() and not variant.is_symlink():
                            files.extend(
                                variant / name
                                for name in ('artifacts.json', 'selection.yaml')
                            )
        files = {path for path in files if self.diagnostic_file(path)}
        commands = self.root / 'commands.jsonl'
        failed_logs = set()
        if commands in files:
            with commands.open(errors='replace') as stream:
                for line in stream:
                    try:
                        command = json.loads(line)
                    except json.JSONDecodeError:
                        # An interrupted write must not prevent diagnostic collection.
                        continue
                    if (
                        isinstance(command, dict)
                        and isinstance(command.get('log'), str)
                        and (
                            command.get('exit_code') != 0 or command.get('interrupted')
                        )
                    ):
                        failed_logs.add(Path(command['log']))

        def priority(path):
            if path == commands or (
                path.parent == self.root and path.name.startswith('phase-')
            ):
                return 0
            if path in failed_logs:
                return 1
            if path.parent == self.root / 'reports':
                return 2
            return 3

        # Reserve space for failure evidence before bulk logs/build artifacts.
        # The same cap still applies if prioritized evidence itself exceeds it.
        files = sorted(files, key=lambda path: (priority(path), path))
        summary = []
        # Exclusive creation refuses existing files, dangling links and links to
        # external files without ever truncating their targets.
        with (self.root / 'preparation-diagnostics.tar.gz').open(
            'xb'
        ) as output, tarfile.open(fileobj=output, mode='w:gz') as archive:
            for path in files[:128]:
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 256 * 1024))
                    data = stream.read(256 * 1024)
                name = path.relative_to(self.root).as_posix()
                item = tarfile.TarInfo(name)
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
                summary.append(
                    {
                        'path': name,
                        'original_bytes': path.stat().st_size,
                        'archived_bytes': len(data),
                        'truncated': len(data) != path.stat().st_size,
                    }
                )
            data = json.dumps(
                {
                    'entries': summary,
                    'omitted_files': max(0, len(files) - 128),
                    'file_limit': 128,
                    'per_file_limit_bytes': 256 * 1024,
                },
                indent=2,
            ).encode()
            item = tarfile.TarInfo('diagnostic-inventory.json')
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'phase',
        choices=(
            'dependencies',
            'rocjitsu',
            'rocisa',
            'tensilelite-build',
            'tensilelite-generate',
            'tensile-build',
            'tensile-generate',
            'llama-build',
            'gluon-verify',
            'diagnostics',
            'cleanup',
        ),
    )
    parser.add_argument(
        '--manifest', help='Corpus-relative TOML recipe for provider inventory'
    )
    args = parser.parse_args()
    if (
        args.phase
        in (
            'tensilelite-generate',
            'tensile-build',
            'tensile-generate',
            'llama-build',
            'gluon-verify',
        )
        and not args.manifest
    ):
        parser.error('--manifest is required for ' + args.phase)
    ctx = None

    # Actions may deliver SIGTERM to this helper rather than its separate child
    # session. Unwind normally so run() can stop that owned process group.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        ctx = Preparation(os.environ, args.manifest)
        if args.phase == 'cleanup':
            shutil.rmtree(ctx.root)
            print('Removed only verified owned benchmark scratch')
            return
        ctx.start(args.phase)
        try:
            getattr(ctx, args.phase.replace('-', '_'))()
        except BaseException:
            ctx.finish(False)
            raise
        ctx.finish(True)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f'error: {error}\n')
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
