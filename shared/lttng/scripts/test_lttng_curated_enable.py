"""Tests for rocm_lttng_enable.sh --dry-run channel handling (schema v1).

Code-review MAJOR 3: the helper must NOT force `-c default` (LTTng's real
userspace default channel is channel0). By default it should omit `-c` and let
LTTng pick its own default; `--channel NAME` forces one.
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
HELPER = os.path.join(HERE, 'rocm_lttng_enable.sh')
HIP_MANIFEST = os.path.join(
    REPO_ROOT, 'projects/clr/hipamd/scripts/lttng_categories_manifest.json')

_HAVE = os.path.exists(HELPER) and os.path.exists(HIP_MANIFEST)
_SKIP = 'requires rocm_lttng_enable.sh + the HIP category manifest'


def _dry_run(extra):
    argv = ['bash', HELPER, '--session', 's', '--provider', 'hip',
            '--api', 'hipMalloc', '--manifest', HIP_MANIFEST, '--dry-run'] + extra
    return subprocess.run(argv, capture_output=True, text=True)


@unittest.skipUnless(_HAVE, _SKIP)
def test_dry_run_omits_channel_flag_by_default():
    """No --channel -> the emitted lttng command must NOT contain `-c`."""
    r = _dry_run([])
    assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
    line = [ln for ln in r.stdout.splitlines() if 'lttng enable-event' in ln]
    assert line, f"no enable-event line in output:\n{r.stdout}"
    for ln in line:
        assert ' -c ' not in f' {ln} ', f"unexpected -c in default-channel cmd: {ln}"
    # sanity: the event is still there
    assert 'rocm_hip:hipMalloc' in r.stdout


@unittest.skipUnless(_HAVE, _SKIP)
def test_dry_run_includes_channel_flag_when_given():
    """--channel mychan -> the emitted lttng command must contain `-c mychan`."""
    r = _dry_run(['--channel', 'mychan'])
    assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
    line = [ln for ln in r.stdout.splitlines() if 'lttng enable-event' in ln]
    assert line, f"no enable-event line in output:\n{r.stdout}"
    assert any('-c mychan' in ln for ln in line), \
        f"expected `-c mychan` in cmd, got:\n{r.stdout}"


@unittest.skipUnless(_HAVE, _SKIP)
def test_default_channel_not_hardcoded_to_the_word_default():
    """Regression guard: the emitted command must not force `-c default`."""
    r = _dry_run([])
    assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
    assert '-c default' not in r.stdout, \
        f"helper still hardcodes `-c default`:\n{r.stdout}"


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
