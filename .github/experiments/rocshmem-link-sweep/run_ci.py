"""Prepare only rocSHMEM and run a fixed-input link sweep on 96 logical CPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ['GITHUB_WORKSPACE']).resolve()
SOURCE = WORKSPACE / 'rocshmem-source'
BUILD = WORKSPACE / 'rocshmem-build'
RESULTS = WORKSPACE / 'rocshmem-results'
SDK_VENV = WORKSPACE / 'rocshmem-sdk'
PIN = 'a10dca087065a82f15ea89dc7d6fdd3c6bb2c246'


def run(argv, log_name):
    with (RESULTS / log_name).open('w') as log:
        subprocess.run([str(a) for a in argv], stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    RESULTS.mkdir(exist_ok=False)
    affinity = sorted(os.sched_getaffinity(0))
    context = {'source_pin': PIN, 'available_affinity': affinity,
               'runner_name': os.environ.get('RUNNER_NAME'),
               'run_id': os.environ.get('GITHUB_RUN_ID'),
               'sdk_version': '10.3.0a20261102', 'logical_cpu_budget': 96}
    for name, path in [('cpu_max', '/sys/fs/cgroup/cpu.max'),
                       ('memory_max', '/sys/fs/cgroup/memory.max'),
                       ('memory_current', '/sys/fs/cgroup/memory.current'),
                       ('meminfo', '/proc/meminfo'), ('cpuinfo', '/proc/cpuinfo')]:
        if Path(path).exists():
            context[name] = Path(path).read_text()
    (RESULTS / 'runner-context.json').write_text(json.dumps(context, indent=2))
    if len(affinity) < 96:
        raise RuntimeError('Runner exposes fewer than 96 logical CPUs; refusing a mislabeled result')
    cpu_max = context.get('cpu_max', 'max 100000').split()
    if cpu_max[0] != 'max' and int(cpu_max[0]) / int(cpu_max[1]) < 96:
        raise RuntimeError('Visible CPU quota is below 96 CPUs')
    os.sched_setaffinity(0, affinity[:96])
    context['selected_affinity'] = sorted(os.sched_getaffinity(0))
    (RESULTS / 'runner-context.json').write_text(json.dumps(context, indent=2))
    for key in ('CC_PRINT_PROC_STAT', 'CC_PRINT_PROC_STAT_FILE'):
        os.environ.pop(key, None)
    os.environ['CCACHE_DISABLE'] = '1'
    actual_pin = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_pin != PIN:
        raise RuntimeError(f'Source pin mismatch: {actual_pin}')
    patch = HERE / 'rocshmem-link-jobs.patch'
    run(['git', '-C', SOURCE, 'apply', '--check', patch], 'patch-check.log')
    run(['git', '-C', SOURCE, 'apply', patch], 'patch-apply.log')
    run(['git', '-C', SOURCE, 'diff', '--check'], 'diff-check.log')
    run(['python3', '-m', 'venv', SDK_VENV], 'venv.log')
    run([SDK_VENV / 'bin/pip', 'install', 'rocm[devel]==10.3.0a20261102',
         '--index-url=https://nightly.repo.amd.com/rocm/whl-next/'], 'sdk-install.log')
    run([SDK_VENV / 'bin/rocm-sdk', 'init'], 'sdk-init.log')
    sdk = Path(subprocess.check_output([str(SDK_VENV / 'bin/rocm-sdk'), 'path', '--root'], text=True).strip())
    run([sdk / 'llvm/bin/clang++', '--version'], 'compiler-version.txt')
    run([SDK_VENV / 'bin/pip', 'freeze'], 'sdk-packages.txt')
    archs = json.loads((HERE / 'architectures.json').read_text(encoding='utf-8-sig'))
    if len(archs) != 31 or len(set(archs)) != 31:
        raise RuntimeError('Expected all 31 unique architectures')
    command = ['cmake', '-S', SOURCE / 'projects/rocshmem', '-B', BUILD, '-GNinja',
               f'-DCMAKE_PREFIX_PATH={sdk}', f'-DCMAKE_CXX_COMPILER={sdk}/llvm/bin/clang++',
               f'-DROCM_PATH={sdk}', '-DGPU_TARGETS=' + ';'.join(archs),
               '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_FUNCTIONAL_TESTS=ON',
               '-DBUILD_UNIT_TESTS=OFF', '-DBUILD_CTESTS=OFF', '-DBUILD_TOOLS=ON',
               '-DROCSHMEM_NATIVE_LINK_JOBS=1', '-DHIP_CLANG_NUM_PARALLEL_JOBS=1',
               '-DUSE_RO=ON', '-DUSE_IPC=ON', '-DUSE_GDA=ON', '-DUSE_SDMA=ON',
               '-DUSE_ROCPROFILER_REGISTER=ON', '-DGDA_MLX5=ON', '-DGDA_BNXT=ON',
               '-DGDA_IONIC=ON', '-DGDA_MUX=ON', '-DUSE_EXTERNAL_MPI=OFF',
               '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
    (RESULTS / 'configure-command.json').write_text(json.dumps([str(a) for a in command], indent=2))
    run(command, 'configure.log')
    cache = {}
    for line in (BUILD / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key_type, value = line.split('=', 1)
            cache[key_type.split(':', 1)[0]] = value
    for flag in ('USE_RO', 'USE_IPC', 'USE_GDA', 'USE_SDMA', 'USE_ROCPROFILER_REGISTER',
                 'GDA_MLX5', 'GDA_BNXT', 'GDA_IONIC', 'GDA_MUX', 'BUILD_FUNCTIONAL_TESTS', 'BUILD_TOOLS'):
        if cache.get(flag) != 'ON':
            raise RuntimeError(f'Required feature was disabled: {flag}={cache.get(flag)}')
    if sorted(cache.get('GPU_TARGETS', '').split(';')) != sorted(archs):
        raise RuntimeError('Configured GPU architecture coverage changed')
    # Build each final link's direct inputs (including order-only runtime assets),
    # avoiding an extra untimed pair of expensive executable links.
    dependencies = set()
    targets = ('tools/rocshmem_info', 'tests/functional_tests/rocshmem_functional_tests')
    for target in targets:
        query = subprocess.check_output(['ninja', '-C', str(BUILD), '-t', 'query', target], text=True)
        (RESULTS / (Path(target).name + '-dependency-query.txt')).write_text(query)
        incoming = False
        for line in query.splitlines():
            if line.startswith('  input:'):
                if 'EXECUTABLE_LINKER' not in line:
                    raise RuntimeError('Unexpected target rule: ' + line)
                incoming = True
            elif line.startswith('  outputs:'):
                incoming = False
            elif incoming and line.startswith('    '):
                dependency = line.strip().lstrip('|').strip()
                if dependency:
                    dependencies.add(dependency)
    if not dependencies or any(target in dependencies for target in targets):
        raise RuntimeError('Invalid prerequisite-only preparation graph')
    (RESULTS / 'preparation-targets.json').write_text(json.dumps(sorted(dependencies), indent=2))
    run(['ninja', '-C', BUILD, '-j32', '--', *sorted(dependencies)], 'prepare.log')
    (RESULTS / 'CMakeCache.txt').write_bytes((BUILD / 'CMakeCache.txt').read_bytes())
    (RESULTS / 'compile_commands.json').write_bytes((BUILD / 'compile_commands.json').read_bytes())
    run(['python3', HERE / 'benchmark/sweep.py', '--source', SOURCE / 'projects/rocshmem',
         '--build', BUILD, '--output', RESULTS / 'trials', '--targets', 'both',
         '--order', '1,2,4,8,16,31', '--repeat-best', '--expected-affinity-cpus', '96',
         '--slot-lock', WORKSPACE / 'rocshmem-benchmark.lock'], 'sweep.log')
    script_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in HERE.iterdir() if p.is_file()}
    (RESULTS / 'experiment-files.json').write_text(json.dumps(script_hashes, indent=2))


if __name__ == '__main__':
    main()
