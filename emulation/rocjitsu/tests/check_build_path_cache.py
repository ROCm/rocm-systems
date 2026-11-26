# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    cmake, compiler, compiler_arg1, ninja = sys.argv[1:]
    ccache = shutil.which('ccache')
    if not ccache or '--print-stats' not in subprocess.check_output(
        [ccache, '--help'], text=True
    ):
        print('This test needs ccache with machine-readable statistics.')
        return 77

    launcher = '' if os.path.samefile(compiler, ccache) else ccache
    helper = Path(__file__).resolve().parents[1] / 'cmake' / 'rj_add_test_path.cmake'
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        root = Path(tmp)
        env = {k: v for k, v in os.environ.items() if not k.startswith('CCACHE_')}
        env.update(
            CCACHE_DIR=str(root / 'cache'),
            CCACHE_CONFIGPATH=os.devnull,
            CCACHE_NOHASHDIR='true',
        )
        for name in ('first', 'second'):
            workspace = root / name
            workspace.mkdir()
            build = workspace / 'build'
            env['CCACHE_BASEDIR'] = str(workspace)
            (workspace / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.22)\n'
                'project(path_cache LANGUAGES CXX)\n'
                'include("${HELPER}")\n'
                'add_executable(probe main.cpp)\n'
                'set_property(TARGET probe PROPERTY CXX_COMPILER_LAUNCHER "${LAUNCHER}")\n'
                'rj_add_build_path(probe FIXTURE_PATH "${CMAKE_CURRENT_BINARY_DIR}")\n'
            )
            (workspace / 'main.cpp').write_text(
                '#include <cstdio>\nint main() { std::puts(FIXTURE_PATH); }\n'
            )
            subprocess.run(
                [
                    cmake,
                    '-G',
                    'Ninja',
                    '-S',
                    workspace,
                    '-B',
                    build,
                    f'-DHELPER={helper}',
                    f'-DLAUNCHER={launcher}',
                    f'-DCMAKE_CXX_COMPILER={compiler}',
                    f'-DCMAKE_CXX_COMPILER_ARG1:STRING={compiler_arg1}',
                    f'-DCMAKE_MAKE_PROGRAM={ninja}',
                    '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
                ],
                check=True,
                # Compiler detection can itself use ccache through a wrapper.
                env={**env, 'CCACHE_DISABLE': '1'},
            )
            subprocess.run([cmake, '--build', build], check=True, env=env)
            actual = subprocess.check_output([build / 'probe'], text=True).rstrip('\n')
            if actual != str(build):
                raise RuntimeError(
                    f'{name} build resolved {actual!r}, expected {str(build)!r}'
                )

        # Of the four compiles, only the second main.cpp can hit. Each path
        # object must contain its own absolute path for the probes to pass.
        stats = subprocess.check_output([ccache, '--print-stats'], text=True, env=env)
        counters = dict(line.split() for line in stats.splitlines())
        if counters['direct_cache_hit'] != '1':
            raise RuntimeError(
                f"Expected the consumer to hit directly, got {counters['direct_cache_hit']} hits"
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
