"""Compare rocSHMEM GPU behavior after linking identical inputs with 1 or 31 jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['GITHUB_WORKSPACE'])
SOURCE = ROOT / 'rocshmem-source'
BUILD = ROOT / 'rocshmem-gpu-build'
RESULTS = ROOT / 'rocshmem-gpu-results'
VENV = ROOT / 'rocshmem-gpu-sdk'
PIN = 'bccdf92e49f4a2cc25825090bc61d19a75193c51'
VERSION = '10.3.0a20261102'
TARGETS = ('tools/rocshmem_info', 'tests/functional_tests/rocshmem_functional_tests')


def run(args, log, *, cwd=None, env=None):
    with (RESULTS / log).open('w') as output:
        subprocess.run([str(x) for x in args], cwd=cwd, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    RESULTS.mkdir(exist_ok=False)
    pin = subprocess.check_output(['git', '-C', SOURCE, 'rev-parse', 'HEAD'], text=True).strip()
    if pin != PIN:
        raise RuntimeError(f'Source pin changed: {pin}')
    archs = json.loads((HERE / 'architectures.json').read_text(encoding='utf-8-sig'))
    if len(archs) != 31 or len(set(archs)) != 31 or 'gfx942' not in archs:
        raise RuntimeError('Expected 31 architectures including gfx942')
    if not Path('/dev/kfd').exists():
        raise RuntimeError('GPU device /dev/kfd is unavailable')
    run(['python3', '-m', 'venv', VENV], 'venv.log')
    run([VENV / 'bin/pip', 'install', f'rocm[devel]=={VERSION}',
         '--index-url=https://nightly.repo.amd.com/rocm/whl-next/'], 'sdk-install.log')
    run([VENV / 'bin/rocm-sdk', 'init'], 'sdk-init.log')
    run([VENV / 'bin/pip', 'install', 'matplotlib', 'numpy', 'pandas', 'seaborn'],
        'analysis-install.log')
    sdk = Path(subprocess.check_output([VENV / 'bin/rocm-sdk', 'path', '--root'], text=True).strip())
    env = os.environ.copy()
    env['PATH'] = f"{sdk / 'bin'}:{sdk / 'llvm/bin'}:{env['PATH']}"
    env['LD_LIBRARY_PATH'] = f"{sdk / 'lib'}:{sdk / 'lib64'}:{sdk / 'llvm/lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env['ROCSHMEM_TEST_SLR'] = '1'
    env['CCACHE_DISABLE'] = '1'
    common = ['cmake', '-S', SOURCE / 'projects/rocshmem', '-B', BUILD, '-GNinja',
              f'-DCMAKE_PREFIX_PATH={sdk}', f'-DCMAKE_CXX_COMPILER={sdk}/llvm/bin/clang++',
              f'-DROCM_PATH={sdk}', '-DGPU_TARGETS=' + ';'.join(archs),
              '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_FUNCTIONAL_TESTS=ON',
              '-DBUILD_UNIT_TESTS=OFF', '-DBUILD_CTESTS=OFF', '-DBUILD_TOOLS=ON',
              '-DUSE_RO=OFF', '-DUSE_IPC=ON', '-DUSE_GDA=OFF',
              '-DUSE_SINGLE_NODE=ON', '-DUSE_EXTERNAL_MPI=OFF',
              '-DHIP_CLANG_NUM_PARALLEL_JOBS=1']
    context = {'source_pin': pin, 'sdk_version': VERSION, 'architectures': archs,
               'runner': env.get('RUNNER_NAME'), 'run_id': env.get('GITHUB_RUN_ID'),
               'order': [1, 31, 31, 1, 31, 1], 'suite': 'heatmap',
               'comparison': 'same source and build directory; only link worker count changes'}
    (RESULTS / 'context.json').write_text(json.dumps(context, indent=2))
    saved = {}
    for width in (1, 31):
        run(common + [f'-DROCSHMEM_NATIVE_LINK_JOBS={width}'], f'configure-{width}.log', env=env)
        run(['ninja', '-C', BUILD, '-j16', *TARGETS], f'build-{width}.log', env=env)
        saved[width] = {}
        for target in TARGETS:
            path = BUILD / target
            copy = RESULTS / f'{path.name}-width{width}'
            shutil.copy2(path, copy)
            saved[width][target] = {'sha256': digest(copy), 'size': copy.stat().st_size}
    context['binaries'] = saved
    context['binary_parity'] = all(saved[1][t]['sha256'] == saved[31][t]['sha256'] for t in TARGETS)
    (RESULTS / 'context.json').write_text(json.dumps(context, indent=2))
    if not context['binary_parity']:
        raise RuntimeError('Linked executable hashes differ; manual review required before performance comparison')
    app = BUILD / TARGETS[1]
    info = BUILD / TARGETS[0]
    driver = SOURCE / 'projects/rocshmem/scripts/functional_tests/driver.sh'
    for index, width in enumerate(context['order']):
        shutil.copy2(RESULTS / f'{app.name}-width{width}', app)
        shutil.copy2(RESULTS / f'{info.name}-width{width}', info)
        log_dir = RESULTS / f'logs-heatmap-width{width}-iteration{index}'
        start = time.monotonic()
        run(['bash', driver, TARGETS[1], 'heatmap', log_dir],
            f'run-{index}-width{width}.log', cwd=BUILD, env=env)
        context.setdefault('gpu_runs', []).append({'width': width, 'index': index,
            'wall_seconds': time.monotonic() - start, 'logs': str(log_dir.relative_to(RESULTS))})
        (RESULTS / 'context.json').write_text(json.dumps(context, indent=2))
    run([VENV / 'bin/python3', SOURCE / 'projects/rocshmem/scripts/functional_tests/perf_compare.py',
         '--baseline', str(RESULTS / 'logs-heatmap-width1-*'),
         '--variants', 'width31:' + str(RESULTS / 'logs-heatmap-width31-*'),
         '--outdir', RESULTS / 'comparison'], 'compare.log', env=env)


if __name__ == '__main__':
    main()
