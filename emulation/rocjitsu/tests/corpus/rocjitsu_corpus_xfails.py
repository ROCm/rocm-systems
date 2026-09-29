# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Temporary simulator expectations; loaded only by run-corpus-tests.sh.

Keep running these cases. A pass is a strict XPASS, and build errors or changed
runtime diagnostics remain failures. Remove the expectations as the fixes in
https://github.com/ROCm/rocm-systems/issues/12482 land.
"""

import re

import pytest

ISSUE = 'https://github.com/ROCm/rocm-systems/issues/12482'
BLOCK_CASE = 'cts.gfx1250.memory_isa.memory_isa_gfx1250_block_test'
FLAT_CASE = 'cts.gfx1250.memory_isa.memory_isa_gfx1250_flat_aperture_test'
BLOCK_GTEST = 'Gfx1250MemoryIsaBlock.SparseMasksPreserveRegisterAndMemoryHoles'
REASONS = {
    BLOCK_CASE: f'gfx1250 block transfers ignore sparse M0 masks; {ISSUE}',
    FLAT_CASE: f'gfx1250 flat-aperture test times out; {ISSUE}',
}


class ExpectedSimulatorFailure(RuntimeError):
    """A runtime failure matching one of the documented simulator failures."""


def _case_id(item):
    callspec = getattr(item, 'callspec', None)
    case = callspec.params.get('corpus_case') if callspec else None
    return getattr(case, 'id', None)


def pytest_collection_modifyitems(items):
    for item in items:
        reason = REASONS.get(_case_id(item))
        if reason:
            item.add_marker(
                pytest.mark.xfail(
                    strict=True, raises=ExpectedSimulatorFailure, reason=reason
                )
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    case_id = _case_id(item)
    if case_id not in REASONS or outcome.excinfo is None:
        return
    error = outcome.excinfo[1]
    if isinstance(error, RuntimeError) and _matches_failure(case_id, str(error)):
        # Only the call phase is eligible: configure/build fixture failures
        # must never be turned into expected simulator failures.
        outcome.force_exception(ExpectedSimulatorFailure(str(error)))


def _matches_failure(case_id, message):
    # The CTS adapter includes the subprocess status and captured streams in
    # its RuntimeError. Match that envelope before inspecting the known result.
    match = re.fullmatch(
        r'CTS ctest failed\.\nlog: [^\n]+\ncommand: [^\n]+\n'
        r'returncode: (\d+)\nstdout:\n(.*)\nstderr:\n(.*)',
        message,
        re.DOTALL,
    )
    if match is None or re.search(r'Sanitizer|runtime error:', message):
        return False
    returncode, stdout, stderr = match.groups()
    if case_id == BLOCK_CASE:
        failed_tests = set(re.findall(r'\[  FAILED  \] ([\w.]+\.[\w.]+)', stdout))
        return (
            returncode == '8'
            and failed_tests == {BLOCK_GTEST}
            and 'Expected equality of these values:' in stdout
            and 'kWordCanary' in stdout
            and stderr.strip() in ('', '<empty>', 'Errors while running CTest')
        )
    if case_id == FLAT_CASE:
        return (
            returncode == '124'
            and re.search(
                r'Start\s+\d+: memory_isa_gfx1250_flat_aperture_test\b', stdout
            )
            is not None
            and stderr.strip() in ('', '<empty>')
        )
    return False
