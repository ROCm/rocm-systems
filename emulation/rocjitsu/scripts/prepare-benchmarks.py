#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Attempt-local benchmark preparation. Never publish or change source checkouts."""

import argparse
import ast
import hashlib
import io
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
        for key, directory in {
            'HOME': 'home',
            'TMPDIR': 'tmp',
            'XDG_CACHE_HOME': 'cache',
            'PIP_CACHE_DIR': 'cache/pip',
            'TENSILE_HELPER_CACHE_DIR': 'cache/tensile',
            'TRITON_CACHE_DIR': 'cache/triton',
            'JOBLIB_TEMP_FOLDER': 'tmp/joblib',
        }.items():
            self.env[key] = str(self.root / directory)
        for directory in [
            self.logs,
            self.cwd,
            self.root / 'reports',
            *[
                Path(self.env[k])
                for k in (
                    'HOME',
                    'TMPDIR',
                    'XDG_CACHE_HOME',
                    'PIP_CACHE_DIR',
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
        for key in (
            'LD_PRELOAD',
            'PYTHONPATH',
            'PYTHONHOME',
            'PIP_INDEX_URL',
            'PIP_EXTRA_INDEX_URL',
            'PIP_FIND_LINKS',
            'PIP_TRUSTED_HOST',
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
            CMAKE_BUILD_PARALLEL_LEVEL='4',
            PIP_CONFIG_FILE=os.devnull,
            PIP_NO_INPUT='1',
            PIP_DISABLE_PIP_VERSION_CHECK='1',
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
        for name, path, revision in [
            ('tensile', self.env['TENSILE_SOURCE'], self.env['TENSILE_SHA']),
            ('nanobind', self.env['NANOBIND_SOURCE'], self.env['NANOBIND_SHA']),
            ('corpus', self.env['CORPUS_SOURCE_DIR'], self.env['CORPUS_SHA']),
            ('rocjitsu', self.env['ROCJITSU_SOURCE_DIR'], self.env['BENCHMARK_SHA']),
        ]:
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

    def dependencies(self):
        self.sources()
        require(not (self.root / 'venv').exists(), 'Refusing to reuse an old venv')
        self.run('venv', [sys.executable, '-m', 'venv', self.root / 'venv'])
        self.run('packages-before', [self.python, '-m', 'pip', 'list', '--format=json'])
        self.run(
            'benchmark-requirements',
            [
                self.python,
                '-m',
                'pip',
                'install',
                '--report',
                self.root / 'reports/benchmark-resolution.json',
                '-r',
                Path(self.env['CORPUS_SOURCE_DIR']) / 'benchmarks/requirements.txt',
            ],
        )
        expected = self.pins()
        constraint = self.root / 'benchmark-constraints.txt'
        constraint.write_text(
            ''.join(f'{name}=={version}\n' for name, version in expected.items())
        )
        tensile = Path(self.env['TENSILE_SOURCE']) / 'projects/hipblaslt/tensilelite'
        self.run(
            'tensile-requirements',
            [
                self.python,
                '-m',
                'pip',
                'install',
                '--index-url',
                'https://pypi.org/simple',
                '--constraint',
                constraint,
                '--report',
                self.root / 'reports/tensile-resolution.json',
                '-r',
                tensile / 'requirements.txt',
            ],
        )
        build_requires = tomllib.loads((tensile / 'rocisa/pyproject.toml').read_text())[
            'build-system'
        ]['requires']
        self.run(
            'wheel-requirements',
            [
                self.python,
                '-m',
                'pip',
                'install',
                '--index-url',
                'https://pypi.org/simple',
                '--constraint',
                constraint,
                '--report',
                self.root / 'reports/build-resolution.json',
                *build_requires,
            ],
        )
        self.pins()
        self.run('pip-check', [self.python, '-m', 'pip', 'check'])
        self.run(
            'packages-resolved', [self.python, '-m', 'pip', 'list', '--format=json']
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
            'CMAKE_BUILD_PARALLEL_LEVEL',
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
                '--parallel',
                self.env['CMAKE_BUILD_PARALLEL_LEVEL'],
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

    def native(self):
        require(
            json.loads((self.root / 'phase-rocjitsu.json').read_text())['success'],
            'Rocjitsu setup did not complete',
        )
        self.sources()
        self.sdk_environment()
        self.pins()
        sdk = self.env['ROCM_PATH']
        flags = self.cmake_flags()
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
                self.python,
                '-m',
                'pip',
                'wheel',
                '--no-index',
                '--no-deps',
                '--no-build-isolation',
                '--wheel-dir',
                wheelhouse,
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
        self.run(
            'install-exact-wheel',
            [
                self.python,
                '-m',
                'pip',
                'install',
                '--no-index',
                '--no-deps',
                '--no-compile',
                wheel,
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
                'archive=direct["archive_info"]; assert archive.get("hashes",{}).get("sha256")==sys.argv[4] or archive.get("hash")=="sha256="+sys.argv[4]',
                'maps=pathlib.Path("/proc/self/maps").read_text(); assert "/opt/rocm" not in maps',
                'libraries=[]',
                'for line in maps.splitlines():',
                ' if "libstinkytofu.so" in line:',
                '  path=pathlib.Path(line.split(maxsplit=5)[-1].replace(chr(92)+"040"," ")); assert path.resolve().is_relative_to(root)',
                '  assert hashlib.sha256(path.read_bytes()).hexdigest() in json.loads(sys.argv[5]); libraries.append(str(path))',
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
                sha256(wheel),
                json.dumps(library_hashes),
            ],
        )
        self.run('pip-check', [self.python, '-m', 'pip', 'check'])
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
                '--parallel',
                self.env['CMAKE_BUILD_PARALLEL_LEVEL'],
                '--target',
                'tensile_candidate',
                'tensile_candidate_host_tests',
            ],
        )
        self.run(
            'host-tests', [self.root / 'tensile-native/tensile_candidate_host_tests']
        )
        self.binary_inventory(
            'tensile-build',
            [
                'tensile-native/tensile_candidate',
                'tensile-native/tensile_candidate_host_tests',
            ],
        )
        self.pins()
        self.run('packages-after', [self.python, '-m', 'pip', 'list', '--format=json'])
        self.sources()

    def inventory(self):
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
        recipe = tomllib.loads(manifest.read_text())
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

    def generate(self):
        require(
            json.loads((self.root / 'phase-native.json').read_text())['success'],
            'Native preparation did not complete',
        )
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

    def diagnostic_file(self, path):
        return (
            self.root in path.parents
            and not any(parent.is_symlink() for parent in (path, *path.parents))
            and path.resolve().is_relative_to(self.root.resolve())
            and path.is_file()
        )

    def diagnostics(self):
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
        files = sorted({path for path in files if self.diagnostic_file(path)})
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
            'native',
            'generate',
            'diagnostics',
            'cleanup',
        ),
    )
    parser.add_argument('--manifest', help='Corpus-relative TOML recipe for generate')
    args = parser.parse_args()
    if args.phase == 'generate' and not args.manifest:
        parser.error('--manifest is required for generate')
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
            getattr(ctx, args.phase)()
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
