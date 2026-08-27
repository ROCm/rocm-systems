#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the expect_status/status_sweep assertion harness.

The two are tested together because expect_status changes behaviour depending on
whether a sweep is open: on its own it raises at the failing call, inside a sweep
it hands the failure over to be reported at the end.
"""

from __future__ import annotations

import unittest

from common.common import ERROR_MAP, PASS, VERBOSITY_QUIET, Common, amdsmi

SUCCESS = amdsmi.AmdSmiStatus.SUCCESS
INVAL = amdsmi.AmdSmiStatus.INVAL
NOT_SUPPORTED = amdsmi.AmdSmiStatus.NOT_SUPPORTED


def _harness():
    """A Common carrying only what the status helpers read, so no device is touched."""
    common = object.__new__(Common)
    common.error_map = ERROR_MAP
    common.verbose = VERBOSITY_QUIET
    common._status_failures = None
    return common


def _raise(status):
    """Fail a call the way the amdsmi wrapper does, with the status attached."""
    raise amdsmi.AmdSmiLibraryException(status.value)


class TestExpectStatusAlone(unittest.TestCase):
    """Without a sweep, the verdict is delivered at the call that produced it."""

    def setUp(self):
        self.common = _harness()

    def test_accepted_status_is_not_a_failure(self):
        with self.common.expect_status("msg", NOT_SUPPORTED):
            _raise(NOT_SUPPORTED)
        return

    def test_unaccepted_status_raises_naming_both_sides(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", NOT_SUPPORTED):
                _raise(INVAL)
        message = str(ctx.exception)
        self.assertIn("AMDSMI_STATUS_INVAL", message)
        self.assertIn("AMDSMI_STATUS_NOT_SUPPORTED", message)
        # Chained so the traceback still reaches the call that produced the status.
        self.assertIsInstance(ctx.exception.__cause__, amdsmi.AmdSmiLibraryException)
        return

    def test_call_fails_by_succeeding_when_pass_is_left_out(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", NOT_SUPPORTED):
                pass
        self.assertIn(PASS, str(ctx.exception))
        return

    def test_accept_is_taken_as_enum_or_name_and_scalar_or_list(self):
        for accept in (SUCCESS, [SUCCESS], PASS, [PASS]):
            with self.subTest(accept=accept):
                with self.common.expect_status("msg", accept):
                    pass
        return

    def test_unknown_status_name_is_rejected_before_the_call(self):
        with self.assertRaises(ValueError):
            with self.common.expect_status("msg", "AMDSMI_STATUS_NOPE"):
                self.fail("the block must not run")
        return

    def test_exception_without_a_status_propagates(self):
        # Raised here to stand in for a bug in a test body: it carries no status code,
        # so it has to escape rather than be recorded as something the driver returned.
        with self.assertRaises(KeyError):
            with self.common.expect_status("msg", SUCCESS):
                raise KeyError("not an amdsmi failure")
        return


class TestExpectStatusInSweep(unittest.TestCase):
    """Inside a sweep, failures accumulate and are reported once at the end."""

    def setUp(self):
        self.common = _harness()

    def test_sweep_reports_every_failed_call(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.status_sweep():
                for status in (INVAL, NOT_SUPPORTED):
                    with self.common.expect_status(f"call {status.name}", SUCCESS):
                        _raise(status)
        message = str(ctx.exception)
        self.assertIn("2 calls", message)
        self.assertIn("call INVAL", message)
        self.assertIn("call NOT_SUPPORTED", message)
        return

    def test_sweep_visits_every_call_despite_an_early_failure(self):
        visited = []
        with self.assertRaises(AssertionError):
            with self.common.status_sweep():
                for status in (INVAL, NOT_SUPPORTED):
                    with self.common.expect_status("msg", SUCCESS):
                        _raise(status)
                    visited.append(status)
        self.assertEqual(visited, [INVAL, NOT_SUPPORTED])
        return

    def test_sweep_without_failures_is_silent(self):
        with self.common.status_sweep():
            with self.common.expect_status("msg", SUCCESS):
                pass
        self.assertIsNone(self.common._status_failures)
        return

    def test_inner_sweep_failure_does_not_reach_the_outer_one(self):
        with self.common.status_sweep():
            with self.assertRaises(AssertionError):
                with self.common.status_sweep():
                    with self.common.expect_status("inner call", SUCCESS):
                        _raise(INVAL)
            # Leaving the outer sweep must stay silent, so it collected nothing.
        self.assertIsNone(self.common._status_failures)
        return

    def test_body_exception_discards_what_the_sweep_collected(self):
        with self.assertRaises(KeyError):
            with self.common.status_sweep():
                with self.common.expect_status("msg", SUCCESS):
                    _raise(INVAL)
                # Stands in for a bug in a test body, which wins over what was collected.
                raise KeyError("not an amdsmi failure")
        self.assertIsNone(self.common._status_failures)
        return
