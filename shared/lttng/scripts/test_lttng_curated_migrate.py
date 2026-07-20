"""Tests for the HIP/HSA curated-wrapper migrators (schema v1).

Focus areas (from code review):
  1. Idempotency: running a migrator twice must produce byte-identical output
     and leave EXACTLY one `__ROCM_CURATED__: <api>` sentinel per API (the HSA
     migrator previously left duplicate/multiline sentinels and corrupted the
     one-line v5 snapshots on rerun).
  2. Duplicate-sentinel cleanup: a wrapper carrying a pre-existing duplicate
     (or multi-line) sentinel must be normalized to a single sentinel.
  3. prop->properties aliasing (HIP): a wrapper whose implementation parameter
     name differs from the public/YAML name must keep the IMPL-local name on
     the snapshot RHS while keeping the YAML name for the local + helper ABI.

These run against the REAL checked-in sources (copied into a tmp dir) + a live
libclang parse of the real headers, so they exercise exactly what the build
does. They skip cleanly when libclang or the curated artifacts aren't present.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

HIP_MIGRATE = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/lttng_curated_migrate.py')
HSA_MIGRATE = os.path.join(HERE, 'lttng_curated_migrate.py')

HIP_YAML = os.path.join(REPO_ROOT, 'projects/clr/hipamd/scripts/curated_apis.yaml')
HSA_YAML = os.path.join(
    REPO_ROOT, 'projects/rocr-runtime/runtime/hsa-runtime/scripts/curated_apis.yaml')

HIP_SOURCES = [
    'projects/clr/hipamd/src/hip_table_interface.cpp',
    'projects/clr/hipamd/src/hip_table_interface_c.cpp',
    'projects/clr/hipamd/src/hip_error.cpp',
    'projects/clr/hipamd/src/profiler/hip_clr_profiler.cpp',
    'projects/clr/hipamd/src/hip_device_runtime.cpp',
]
HIP_HEADERS = [
    'projects/hip/include/hip/hip_runtime_api.h',
    'projects/hip/include/hip/hip_ext.h',
]
HIP_EXTRA = [
    '-D__HIP_PLATFORM_AMD__=1',
    '-Iprojects/hip/include',
    '-Iprojects/clr/hipamd/src',
    '-Iprojects/clr/hipamd/include',
    '-Iprojects/clr/rocclr',
    '-Iprojects/rocr-runtime/runtime/hsa-runtime/inc',
]

HSA_SOURCE = 'projects/rocr-runtime/runtime/hsa-runtime/core/common/hsa_table_interface.cpp'
HSA_HEADERS = [
    'projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h',
    'projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h',
    'projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_api_trace.h',
    'projects/rocr-runtime/runtime/hsa-runtime/core/inc/hsa_table_interface.h',
]
HSA_EXTRA = [
    '-Iprojects/rocr-runtime/runtime/hsa-runtime/inc',
    '-Iprojects/rocr-runtime/runtime/hsa-runtime',
]

try:
    import clang.cindex  # noqa: F401
    _HAVE_LIBCLANG = True
except Exception:
    _HAVE_LIBCLANG = False

_HIP_ARTIFACTS = all(os.path.exists(os.path.join(REPO_ROOT, p))
                     for p in HIP_SOURCES + [HIP_YAML])
_HSA_ARTIFACTS = (os.path.exists(os.path.join(REPO_ROOT, HSA_SOURCE))
                  and os.path.exists(HSA_YAML))
_SKIP = 'requires libclang + the real curated sources/headers'


def _sentinel_counts(text):
    """Return Counter{api_name: occurrences} over ALL sentinels, tolerating
    whitespace/newlines inside the comment (a formatter may wrap it)."""
    names = re.findall(
        r'/\*\s*__ROCM_CURATED__:\s*([A-Za-z0-9_]+)\s*\*/', text, re.DOTALL)
    return Counter(names)


def _run(argv):
    return subprocess.run(argv, capture_output=True, text=True, cwd=REPO_ROOT)


def _run_hip_migrate(worktree_sources):
    argv = ['python3', HIP_MIGRATE, '--curated-yaml', HIP_YAML]
    for s in worktree_sources:
        argv += ['--source', s]
    for h in HIP_HEADERS:
        argv += ['--header', os.path.join(REPO_ROOT, h)]
    for e in HIP_EXTRA:
        argv.append(f'--extra-arg={e}')
    return _run(argv)


def _run_hsa_migrate(source):
    argv = ['python3', HSA_MIGRATE, '--source', source,
            '--curated-yaml', HSA_YAML]
    for h in HSA_HEADERS:
        argv += ['--header', os.path.join(REPO_ROOT, h)]
    for e in HSA_EXTRA:
        argv.append(f'--extra-arg={e}')
    return _run(argv)


@unittest.skipUnless(_HAVE_LIBCLANG and _HSA_ARTIFACTS, _SKIP)
def test_hsa_migrator_idempotent_and_single_sentinel():
    """HSA migrator: two runs are byte-identical, and every API has exactly
    one sentinel (cleans up any pre-existing duplicate/multiline sentinels)."""
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, 'hsa_table_interface.cpp')
        shutil.copy(os.path.join(REPO_ROOT, HSA_SOURCE), dst)

        r1 = _run_hsa_migrate(dst)
        assert r1.returncode == 0, f"HSA migrate run1 failed:\n{r1.stderr}"
        after1 = open(dst).read()
        r2 = _run_hsa_migrate(dst)
        assert r2.returncode == 0, f"HSA migrate run2 failed:\n{r2.stderr}"
        after2 = open(dst).read()

        assert after1 == after2, "HSA migrator is not idempotent (run1 != run2)"
        counts = _sentinel_counts(after1)
        dups = {k: v for k, v in counts.items() if v > 1}
        assert not dups, f"HSA has duplicate sentinels after migrate: {dups}"
        # No corrupted (truncated) snapshot lines like `uto const __rocm_in_`.
        assert not re.search(r'^[a-z]{1,3} const __rocm_in_', after1, re.MULTILINE), \
            "HSA migrate produced truncated snapshot fragments"
        # No dangling multi-line sentinel head (`__ROCM_CURATED__:` at EOL).
        assert not re.search(r'__ROCM_CURATED__:\s*$', after1, re.MULTILINE), \
            "HSA migrate left a dangling multi-line sentinel"


@unittest.skipUnless(_HAVE_LIBCLANG and _HSA_ARTIFACTS, _SKIP)
def test_hsa_migrator_cleans_injected_duplicate_sentinel():
    """A wrapper carrying a pre-existing duplicate (and multi-line) sentinel is
    normalized to exactly one sentinel by the migrator."""
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, 'hsa_table_interface.cpp')
        shutil.copy(os.path.join(REPO_ROOT, HSA_SOURCE), dst)
        # Normalize first so we know the shape, then inject a duplicate +
        # a multi-line sentinel right after an existing one.
        assert _run_hsa_migrate(dst).returncode == 0
        text = open(dst).read()
        api = 'hsa_agent_get_info'
        one = f'/* __ROCM_CURATED__: {api} */'
        assert one in text
        injected = (one + f' /* __ROCM_CURATED__: {api} */\n'
                    f'  /* __ROCM_CURATED__:\n     {api} */')
        text = text.replace(one, injected, 1)
        open(dst, 'w').write(text)
        assert _sentinel_counts(open(dst).read())[api] == 3

        r = _run_hsa_migrate(dst)
        assert r.returncode == 0, f"HSA migrate failed on injected dup:\n{r.stderr}"
        counts = _sentinel_counts(open(dst).read())
        assert counts[api] == 1, \
            f"expected 1 sentinel for {api} after cleanup, got {counts[api]}"
        dups = {k: v for k, v in counts.items() if v > 1}
        assert not dups, f"still have duplicate sentinels: {dups}"


@unittest.skipUnless(_HAVE_LIBCLANG and _HIP_ARTIFACTS, _SKIP)
def test_hip_migrator_idempotent_and_single_sentinel():
    """HIP migrator (incl. hip_device_runtime.cpp): two runs are byte-identical
    and each API has exactly one sentinel per defining TU."""
    with tempfile.TemporaryDirectory() as d:
        worktree = []
        for rel in HIP_SOURCES:
            dst = os.path.join(d, os.path.basename(rel))
            shutil.copy(os.path.join(REPO_ROOT, rel), dst)
            worktree.append(dst)

        r1 = _run_hip_migrate(worktree)
        assert r1.returncode == 0, f"HIP migrate run1 failed:\n{r1.stderr}"
        after1 = {p: open(p).read() for p in worktree}
        r2 = _run_hip_migrate(worktree)
        assert r2.returncode == 0, f"HIP migrate run2 failed:\n{r2.stderr}"
        after2 = {p: open(p).read() for p in worktree}

        for p in worktree:
            assert after1[p] == after2[p], \
                f"HIP migrator not idempotent for {os.path.basename(p)}"
        # Per-file: each API sentinel appears at most once in a given file.
        for p in worktree:
            counts = _sentinel_counts(after1[p])
            dups = {k: v for k, v in counts.items() if v > 1}
            assert not dups, \
                f"{os.path.basename(p)} has intra-file duplicate sentinels: {dups}"


@unittest.skipUnless(_HAVE_LIBCLANG and _HIP_ARTIFACTS, _SKIP)
def test_hip_migrator_preserves_prop_to_properties_alias():
    """hipChooseDeviceR0600's YAML name is `prop` but the impl parameter is
    `properties`; the migrator must emit `__rocm_in_prop = properties;` (impl
    name on RHS, YAML name for the local/ABI) and stay stable on rerun."""
    with tempfile.TemporaryDirectory() as d:
        worktree = []
        for rel in HIP_SOURCES:
            dst = os.path.join(d, os.path.basename(rel))
            shutil.copy(os.path.join(REPO_ROOT, rel), dst)
            worktree.append(dst)
        assert _run_hip_migrate(worktree).returncode == 0
        drt = os.path.join(d, 'hip_device_runtime.cpp')
        body = open(drt).read()
        m = re.search(r'hipError_t hipChooseDeviceR0600\([^)]*\)\s*\{(.*?)\n\}',
                      body, re.DOTALL)
        assert m, "hipChooseDeviceR0600 wrapper not found"
        wrapper = m.group(1)
        assert 'auto const __rocm_in_prop = properties;' in wrapper, \
            f"expected `__rocm_in_prop = properties;`, got:\n{wrapper}"
        assert 'auto const __rocm_in_prop = prop;' not in wrapper, \
            "migrator regressed to the non-compiling `= prop`"
        # rerun stable
        before = open(drt).read()
        assert _run_hip_migrate(worktree).returncode == 0
        assert open(drt).read() == before, \
            "hipChooseDeviceR0600 not stable on rerun"


if __name__ == '__main__':
    import inspect
    failures = skipped = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn) and not inspect.isclass(fn):
            try:
                fn()
                print(f'  ok  {name}')
            except unittest.SkipTest as e:
                print(f'  skip {name}: {e}')
                skipped += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f'  FAIL {name}: {e}')
                failures += 1
    print(f'\n{"PASS" if failures == 0 else "FAIL"}: {failures} failures, {skipped} skipped')
    sys.exit(1 if failures else 0)
