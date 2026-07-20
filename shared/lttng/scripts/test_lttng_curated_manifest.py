"""Tests for lttng_gen_category_manifest.py coverage/quality gating (v5).

Code-review MAJOR 5: the generator must FAIL (non-zero) on:
  - any uncategorized curated API (unless --allow-uncategorized),
  - an empty category (matches zero APIs),
  - a pattern that matches zero APIs (dead pattern),
  - a malformed taxonomy,
and must still support --check drift detection.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, 'lttng_gen_category_manifest.py')

# Minimal synthetic tp.h with two rocm_hip events (no libclang needed -- the
# generator parses tp.h textually).
_TP_H = """\
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip, hipAlpha,
    LTTNG_UST_TP_ARGS(int32_t, phase),
    LTTNG_UST_TP_FIELDS( lttng_ust_field_integer(int32_t, phase, phase) ) )
LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip, hipBeta,
    LTTNG_UST_TP_ARGS(int32_t, phase),
    LTTNG_UST_TP_FIELDS( lttng_ust_field_integer(int32_t, phase, phase) ) )
"""


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)


def _run(taxonomy_text, out_path, check=False, allow_uncat=False, tp_text=_TP_H):
    with tempfile.TemporaryDirectory() as d:
        tax = os.path.join(d, 'tax.yaml')
        tph = os.path.join(d, 'tp.h')
        _write(tax, taxonomy_text)
        _write(tph, tp_text)
        argv = ['python3', GEN, '--provider', 'hip',
                '--taxonomy', tax, '--tp-header', tph, '--out', out_path]
        if check:
            argv.append('--check')
        if allow_uncat:
            argv.append('--allow-uncategorized')
        return subprocess.run(argv, capture_output=True, text=True)


def test_full_coverage_passes():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n', out)
        assert r.returncode == 0, f"expected pass:\n{r.stderr}"
        assert os.path.exists(out)


def test_uncategorized_fails():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('hip:\n  a: ["hipAlpha"]\n', out)  # hipBeta uncategorized
        assert r.returncode != 0, "uncategorized API must fail"
        assert 'uncategorized' in (r.stdout + r.stderr).lower()
        assert 'hipBeta' in (r.stdout + r.stderr)


def test_uncategorized_passes_with_escape_hatch():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('hip:\n  a: ["hipAlpha"]\n', out, allow_uncat=True)
        assert r.returncode == 0, f"--allow-uncategorized must pass:\n{r.stderr}"


def test_empty_category_and_dead_pattern_fail():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n  dead: ["hipNoSuch*"]\n', out)
        assert r.returncode != 0, "dead pattern / empty category must fail"
        combined = r.stdout + r.stderr
        assert 'zero APIs' in combined
        assert 'dead' in combined


def test_no_match_pattern_fails_even_if_category_nonempty():
    """A category with one live and one dead pattern still fails on the dead
    pattern (a dead pattern is almost always a typo/renamed API)."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('hip:\n  a: ["hipAlpha", "hipBeta", "hipGone*"]\n', out)
        assert r.returncode != 0, "dead pattern must fail even in a live category"
        assert "hipGone*" in (r.stdout + r.stderr)


def test_malformed_taxonomy_fails():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        # patterns must be a list, not a bare string
        r = _run('hip:\n  a: "hipAlpha"\n', out)
        assert r.returncode != 0, "malformed taxonomy must fail"
        assert 'non-empty list' in (r.stdout + r.stderr)


def test_top_level_not_mapping_fails():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        r = _run('- just\n- a\n- list\n', out)
        assert r.returncode != 0, "non-mapping top-level taxonomy must fail"


def test_check_drift_detected():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        # First generate a good manifest.
        assert _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n', out).returncode == 0
        # Now corrupt it and --check must fail with DRIFT.
        _write(out, '{"provider": "rocm_hip"}\n')
        r = _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n', out, check=True)
        assert r.returncode != 0, "drift must be detected"
        assert 'DRIFT' in (r.stdout + r.stderr)


def test_check_matches_when_clean():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'm.json')
        assert _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n', out).returncode == 0
        r = _run('hip:\n  a: ["hipAlpha", "hipBeta"]\n', out, check=True)
        assert r.returncode == 0, f"clean --check must pass:\n{r.stderr}"


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
