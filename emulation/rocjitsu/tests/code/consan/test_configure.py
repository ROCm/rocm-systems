# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure ConSan from an environment-only SDK without test initialization."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def read_cache(path):
    return {
        line.split(':', 1)[0]: line.split('=', 1)[1]
        for line in path.read_text().splitlines()
        if line and not line.startswith(('#', '//')) and ':' in line and '=' in line
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--rocm', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    build = args.build.resolve()
    cache = read_cache(build / 'CMakeCache.txt')
    assert Path(cache['CMAKE_HOME_DIRECTORY']).resolve() == source
    directory = build / 'Testing' / 'ConSanConfigure'
    directory.mkdir(parents=True, exist_ok=True)
    fresh = Path(tempfile.mkdtemp(prefix='environment-', dir=directory))
    command = [
        shutil.which('cmake'),
        '-S',
        str(source),
        '-B',
        str(fresh),
        '-G',
        'Ninja',
        '-DBUILD_TESTING=OFF',
        '-DRJ_BUILD_CONSAN_COMPILER=ON',
        '-DRJ_BUILD_CONSAN_RUNTIME=OFF',
        '-DRJ_BUILD_CONSAN_ANALYZER=OFF',
        f'-DPython3_EXECUTABLE={sys.executable}',
    ]
    for variable in (
        'CMAKE_BUILD_TYPE',
        'CMAKE_C_COMPILER',
        'CMAKE_CXX_COMPILER',
        'CMAKE_MAKE_PROGRAM',
        'CMAKE_C_COMPILER_LAUNCHER',
        'CMAKE_CXX_COMPILER_LAUNCHER',
        'CMAKE_HIP_COMPILER_LAUNCHER',
        'CMAKE_C_FLAGS',
        'CMAKE_CXX_FLAGS',
        'CMAKE_C_FLAGS_RELWITHDEBINFO',
        'CMAKE_CXX_FLAGS_RELWITHDEBINFO',
        'ROCJITSU_HOST_CPU_BASELINE',
        'RJ_ENABLE_ASAN',
        'RJ_ENABLE_UBSAN',
        'RJ_ENABLE_TSAN',
        'RJ_ENABLE_EXPENSIVE_CHECKS',
        'RJ_SANITIZER_RUNTIME',
    ):
        if variable in cache:
            command.append(f'-D{variable}={cache[variable]}')
    # Reuse downloaded sources, never another build's generated files or cache.
    for dependency, project in (
        ('googletest', 'googletest-distribution'),
        ('flatbuffers', 'FlatBuffers'),
    ):
        path = cache[f'{project}_SOURCE_DIR']
        command.append(f'-DFETCHCONTENT_SOURCE_DIR_{dependency.upper()}={path}')
    environment = os.environ.copy()
    environment['ROCM_PATH'] = str(args.rocm.resolve())
    # The SDK compiler must be found through ROCM_PATH, not an activated PATH.
    environment['PATH'] = os.defpath
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=120
    )
    (fresh / 'configure.log').write_text(result.stdout + result.stderr)
    print(f'evidence: {fresh}', flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    configured = read_cache(fresh / 'CMakeCache.txt')
    assert Path(configured['ROCM_PATH']).resolve() == args.rocm.resolve()
    assert configured['BUILD_TESTING'] == 'OFF'
    targets = (fresh / 'CMakeFiles' / 'TargetDirectories.txt').read_text()
    assert '/rocjitsu_consan_compiler.dir' in targets
    stale = fresh / 'stale-sdk' / 'bin'
    reconfigure = [
        *command,
        f'-DAMDCXX={stale}/amdclang++',
        f'-DCLANG_OFFLOAD_BUNDLER={stale}/clang-offload-bundler',
    ]
    result = subprocess.run(
        reconfigure, env=environment, capture_output=True, text=True, timeout=120
    )
    (fresh / 'reconfigure.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    configured = read_cache(fresh / 'CMakeCache.txt')
    for tool in ('AMDCXX', 'CLANG_OFFLOAD_BUNDLER'):
        assert Path(configured[tool]).is_relative_to(args.rocm.resolve()), configured[
            tool
        ]
    assert str(stale) not in (fresh / 'build.ninja').read_text()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
