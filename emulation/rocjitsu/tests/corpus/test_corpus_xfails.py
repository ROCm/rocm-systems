# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check pytest outcomes without compiling or launching GPU workloads."""

from pathlib import Path

import pytest

import rocjitsu_corpus_xfails as xfails

pytest_plugins = ['pytester']


def ctest_error(returncode, stdout, stderr='<empty>'):
    return (
        'CTS ctest failed.\nlog: /tmp/test.log\ncommand: ctest --output-on-failure\n'
        f'returncode: {returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}'
    )


BLOCK_ERROR = ctest_error(
    8,
    'Expected equality of these values:\n  load[1]\n  kWordCanary\n'
    f'[  FAILED  ] {xfails.BLOCK_GTEST} (149 ms)\n',
    'Errors while running CTest\n',
)
FLAT_ERROR = ctest_error(124, '    Start 73: memory_isa_gfx1250_flat_aperture_test\n')


@pytest.fixture
def corpus_pytester(pytester, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parent))
    # Child sessions need only the explicitly loaded plugin, not GPU tooling
    # or whatever unrelated plugins happen to be installed on the host.
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    return pytester


def write_cases(pytester, cases):
    pytester.makepyfile(test_corpus=f'''
from types import SimpleNamespace
import pytest

@pytest.fixture
def build_result(corpus_case):
    if corpus_case.build_error:
        raise RuntimeError(corpus_case.build_error)

@pytest.mark.parametrize('corpus_case', [SimpleNamespace(**c) for c in {cases!r}])
def test_corpus_case(corpus_case, build_result):
    if corpus_case.error:
        raise RuntimeError(corpus_case.error)
''')


def case(case_id, error=None, build_error=None):
    return {'id': case_id, 'error': error, 'build_error': build_error}


def test_known_failures_remain_executed_and_can_be_run_without_xfail(corpus_pytester):
    write_cases(
        corpus_pytester,
        [case(xfails.BLOCK_CASE, BLOCK_ERROR), case(xfails.FLAT_CASE, FLAT_ERROR)],
    )
    corpus_pytester.runpytest_subprocess(
        '-p', 'rocjitsu_corpus_xfails'
    ).assert_outcomes(xfailed=2)
    corpus_pytester.runpytest_subprocess(
        '-p', 'rocjitsu_corpus_xfails', '--runxfail'
    ).assert_outcomes(failed=2)
    corpus_pytester.runpytest_subprocess().assert_outcomes(failed=2)


def test_passes_require_removing_the_expectation(corpus_pytester):
    write_cases(corpus_pytester, [case(xfails.BLOCK_CASE), case(xfails.FLAT_CASE)])
    result = corpus_pytester.runpytest_subprocess('-p', 'rocjitsu_corpus_xfails')
    result.assert_outcomes(failed=2)
    result.stdout.fnmatch_lines(['*[XPASS(strict)]*'])


def test_build_errors_are_not_expected_simulator_failures(corpus_pytester):
    write_cases(corpus_pytester, [case(xfails.BLOCK_CASE, build_error=BLOCK_ERROR)])
    corpus_pytester.runpytest_subprocess(
        '-p', 'rocjitsu_corpus_xfails'
    ).assert_outcomes(errors=1)


def test_other_tests_and_diagnostic_changes_still_fail(corpus_pytester):
    write_cases(
        corpus_pytester,
        [
            case(xfails.BLOCK_CASE.replace('gfx1250', 'gfx950'), BLOCK_ERROR),
            case(xfails.BLOCK_CASE + '_other', BLOCK_ERROR),
            case(xfails.BLOCK_CASE, 'Missing required tool ctest'),
            case(
                xfails.BLOCK_CASE, BLOCK_ERROR.replace('returncode: 8', 'returncode: 1')
            ),
            case(xfails.BLOCK_CASE, BLOCK_ERROR.replace('kWordCanary', 'other_value')),
            case(
                xfails.BLOCK_CASE,
                BLOCK_ERROR + 'AddressSanitizer: heap-buffer-overflow',
            ),
            case(
                xfails.BLOCK_CASE,
                BLOCK_ERROR.replace(
                    'kWordCanary', 'kWordCanary\nruntime error: overflow'
                ),
            ),
            case(
                xfails.BLOCK_CASE,
                BLOCK_ERROR.replace('stdout:\n', 'stdout:\n[  FAILED  ] Other.Test\n'),
            ),
            case(
                xfails.FLAT_CASE, FLAT_ERROR.replace('returncode: 124', 'returncode: 8')
            ),
            case(xfails.FLAT_CASE, FLAT_ERROR.replace('<empty>', 'unexpected stderr')),
            case(xfails.FLAT_CASE, FLAT_ERROR.replace('Start 73:', 'not started:')),
        ],
    )
    corpus_pytester.runpytest_subprocess(
        '-p', 'rocjitsu_corpus_xfails'
    ).assert_outcomes(failed=11)
