#!/usr/bin/env python3
"""Run installed systems quick suites with identical HW/rocJITsu selections.

Absent suites and empty CTest selections fail explicitly. rocdbgapi is exercised
by the debug-agent suite, which uses the installed librocm-dbgapi library.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--rocjitsu', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--logs', type=Path, default=Path('logs'))
    parser.add_argument('--parallel', type=int, default=5)
    args = parser.parse_args()
    root = args.root.resolve()
    args.logs = args.logs.resolve()
    args.logs.mkdir(parents=True, exist_ok=True)
    mode = 'emu' if args.rocjitsu else 'hw'
    env = os.environ.copy()
    env.update(ROCM_PATH=str(root), HIP_PATH=str(root), HIP_PLATFORM='amd',
               AMDGPU_FAMILIES='gfx950-dcgpu', TEST_TYPE='quick',
               ROCPROFSYS_INSTALL_DIR=str(root), ROCPROFSYS_MAX_THREADS='64',
               ROCPROFILER_METRICS_PATH=str(root / 'share/rocprofiler-sdk'),
               DEBUG_HIP_GRAPH_MIN_OVERLAP='0')
    env.pop('GTEST_FILTER', None)
    env['PATH'] = str(root / 'bin') + os.pathsep + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = os.pathsep.join([
        str(root / 'lib'), str(root / 'lib/rocm_sysdeps/lib'),
        str(root / 'share/rocprofiler-systems/examples/lib'),
        env.get('LD_LIBRARY_PATH', '')])
    launcher = []
    if args.rocjitsu:
        if not args.config:
            parser.error('--config is required with --rocjitsu')
        launcher = [str(args.rocjitsu.resolve()), '--config', str(args.config.resolve()),
                    '--wait-checking=off', '--']

    def execute(command, cwd, output, emulate=True, timeout=3600):
        command = (launcher if emulate else []) + list(map(str, command))
        print(shlex.join(command), flush=True)
        with output.open('a', encoding='utf-8') as log:
            log.write(shlex.join(command) + '\n')
            log.flush()
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        return result.returncode

    def ctest(directory, selection, name):
        command = ['ctest', '--test-dir', str(directory), *selection,
                   '--no-tests=error', '--output-on-failure', '--timeout', '900',
                   '--parallel', str(1 if name.startswith('rocprofiler') else args.parallel)]
        # Discovery must occur under emulation too: some installed CTest files
        # invoke a device probe while being loaded.
        listing = launcher + command + ['--show-only=json-v1']
        found = subprocess.run(listing, env=env, capture_output=True, text=True,
                               timeout=180, check=True)
        tests = json.loads(found.stdout)['tests']
        if not tests:
            raise RuntimeError(f'{name}: no tests selected in {directory}')
        (args.logs / f'{mode}-{name}-inventory.json').write_text(
            found.stdout, encoding='utf-8')
        return execute(command + ['--output-junit', str(args.logs / f'{mode}-{name}.xml')],
                       directory, args.logs / f'{name}.log')

    results = {}
    suites = ['hip-tests', 'rocrtst', 'rocr-debug-agent', 'rocprofiler-sdk',
              'rocprofiler-compute', 'rocprofiler-systems']
    for name in suites:
        started = time.monotonic()
        status = 1
        try:
            if name == 'hip-tests':
                status = ctest(root / 'share/hip/catch_tests', ['-L', '^smoke$',
                    '-E', 'Unit_NonHost_Printf_(loop|multiple_Threads|BufferAvailability)'], name)
            elif name == 'rocrtst':
                # Use the canonical quick selection from the pinned TheRock harness.
                harness = Path('TheRock/build_tools/github_actions/test_executable_scripts/test_rocrtst.py')
                tree = ast.parse(harness.read_text(encoding='utf-8'))
                quick = ':'.join(next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                        and target.id == 'QUICK_TESTS' for target in node.targets)))
                status = execute([root / 'bin/rocrtst64', '--gtest_filter=' + quick,
                    '--gtest_output=xml:' + str(args.logs / f'{mode}-{name}.xml')],
                    root / 'bin', args.logs / f'{name}.log')
            elif name == 'rocr-debug-agent':
                # This installed custom harness has no tier/filter support; its
                # bounded suite is the same one used by TheRock's 10-minute job.
                if not (root / 'lib/librocm-dbgapi.so').exists():
                    raise RuntimeError('missing rocdbgapi library')
                directory = root / 'tests/rocm-debug-agent'
                status = execute([sys.executable, directory / 'test_rocr-debug-agent.py',
                    '--try-rocm-path', '--max-retries', '1'], directory,
                    args.logs / f'{name}.log', timeout=900)
            elif name == 'rocprofiler-sdk':
                source = root / 'share/rocprofiler-sdk/tests'
                directory = source / 'build'
                log = args.logs / f'{name}.log'
                requirements = source / 'requirements.txt'
                if requirements.exists():
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
                compiler = root / 'llvm/bin'
                command = ['cmake', '-S', source, '-B', directory, '-G', 'Ninja',
                    f'-DCMAKE_PREFIX_PATH={root};{root}/lib/rocm_sysdeps',
                    f'-DCMAKE_C_COMPILER={compiler}/amdclang',
                    f'-DCMAKE_CXX_COMPILER={compiler}/amdclang++',
                    f'-DCMAKE_HIP_COMPILER={compiler}/amdclang++',
                    '-DCMAKE_HIP_ARCHITECTURES=gfx950', f'-DPython3_EXECUTABLE={sys.executable}']
                if execute(command, root, log, emulate=False) or execute(
                        ['cmake', '--build', directory, '--parallel', '8'], root, log, emulate=False):
                    raise RuntimeError('rocprofiler-sdk installed-test build failed')
                # SDK's upstream harness has no quick tier. Run its unit subset
                # explicitly and fail if it is absent, rather than claim a full run.
                status = ctest(directory, ['-R', 'rocprofiler_sdk[.]unit[.]'], name)
            else:
                directory = root / ('libexec/rocprofiler-compute' if name.endswith('compute')
                                    else 'share/rocprofiler-systems/tests')
                for requirements in [directory / 'requirements.txt', directory / 'requirements-test.txt']:
                    if requirements.exists():
                        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
                status = ctest(directory, ['-L', '^quick$', '-LE',
                    'quick_exclude|quick_therock_ci_exclude|ex_gpu_gfx950'], name)
        except Exception as exc:
            with (args.logs / f'{name}.log').open('a', encoding='utf-8') as log:
                log.write(str(exc) + '\n')
            print(f'{name}: {exc}', flush=True)
        results[name] = status
        # Always emit a suite result, including missing artifacts/configure errors.
        suite = ET.Element('testsuite', name=name, tests='1', failures=str(int(status != 0)))
        case = ET.SubElement(suite, 'testcase', name='suite-execution', classname=name,
                             time=str(time.monotonic() - started))
        if status:
            ET.SubElement(case, 'failure', message=f'exit {status}; see {name}.log')
        ET.ElementTree(suite).write(args.logs / f'{mode}-{name}-execution.xml')
    (args.logs / f'{mode}-suite-status.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    return int(any(results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
