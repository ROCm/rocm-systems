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

from common.common import PASS, VERBOSITY_QUIET, Common, amdsmi

SUCCESS = amdsmi.AmdSmiStatus.SUCCESS
INVAL = amdsmi.AmdSmiStatus.INVAL
NOT_SUPPORTED = amdsmi.AmdSmiStatus.NOT_SUPPORTED
ParameterError = amdsmi.AmdSmiParameterException


def _harness():
    """A Common carrying only what the status helpers read, so no device is touched."""
    return Common.for_unit_tests(VERBOSITY_QUIET)


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


class TestExpectStatusNeedsAJudgement(unittest.TestCase):
    """A call must say what it is judged against: a status, or an exception type."""

    def setUp(self):
        self.common = _harness()

    def test_neither_accept_nor_raises_is_rejected_before_the_call(self):
        with self.assertRaises(ValueError):
            with self.common.expect_status("msg"):
                self.fail("the block must not run")
        return

    def test_success_has_to_be_asked_for(self):
        with self.common.expect_status("msg", SUCCESS):
            pass
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", SUCCESS):
                _raise(INVAL)
        self.assertIn(PASS, str(ctx.exception))
        return


class TestExpectStatusRaises(unittest.TestCase):
    """A call rejected before it has a status is judged by the exception type it raises."""

    def setUp(self):
        self.common = _harness()

    def test_expected_exception_is_not_a_failure(self):
        with self.common.expect_status("msg", raises=ParameterError):
            raise ParameterError("bad", "arg")
        return

    def test_any_type_in_a_tuple_is_accepted(self):
        for exc in (ParameterError("bad", "arg"), amdsmi.AmdSmiLibraryException(INVAL.value)):
            with self.subTest(exc=type(exc).__name__):
                with self.common.expect_status(
                    "msg", raises=(ParameterError, amdsmi.AmdSmiLibraryException)
                ):
                    raise exc
        return

    def test_call_that_returns_fails(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", raises=ParameterError):
                pass
        message = str(ctx.exception)
        self.assertIn("returned without raising", message)
        self.assertIn("AmdSmiParameterException", message)
        return

    def test_wrong_exception_type_fails_and_is_chained(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", raises=ParameterError):
                _raise(INVAL)
        self.assertIn("raised AmdSmiLibraryException", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, amdsmi.AmdSmiLibraryException)
        return

    def test_a_library_status_is_not_enough_when_a_type_is_declared(self):
        # raises= judges the type alone, so a status that accept= would take is still a failure.
        with self.assertRaises(AssertionError):
            with self.common.expect_status("msg", raises=ParameterError):
                _raise(NOT_SUPPORTED)
        return

    def test_accept_and_raises_together_are_rejected(self):
        with self.assertRaises(ValueError):
            with self.common.expect_status("msg", SUCCESS, raises=ParameterError):
                self.fail("the block must not run")
        return

    def test_sweep_collects_failures_by_type(self):
        with self.assertRaises(AssertionError) as ctx:
            with self.common.status_sweep():
                for name in ("first", "second"):
                    with self.common.expect_status(name, raises=ParameterError):
                        pass
        message = str(ctx.exception)
        self.assertIn("2 calls", message)
        self.assertIn("first -> returned without raising", message)
        self.assertIn("second -> returned without raising", message)
        return


class TestForUnitTests(unittest.TestCase):
    def test_carries_only_what_the_status_helpers_read(self):
        common = Common.for_unit_tests(VERBOSITY_QUIET)
        self.assertEqual(common.verbose, VERBOSITY_QUIET)
        self.assertIsNone(common._status_failures)
        self.assertFalse(hasattr(common, "processors"))
        return

    def test_instances_do_not_share_sweep_state(self):
        first, second = Common.for_unit_tests(), Common.for_unit_tests()
        with first.status_sweep():
            self.assertIsNone(second._status_failures)
        return
