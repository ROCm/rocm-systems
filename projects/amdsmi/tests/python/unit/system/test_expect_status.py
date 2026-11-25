#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the expect_status/status_sweep assertion harness.

The two are tested together because expect_status changes behaviour depending on
whether a sweep is open: on its own it raises at the failing call, inside a sweep
it hands the failure over to be reported at the end.
"""

import unittest

from common.common import PASS, Common, amdsmi

SUCCESS = amdsmi.AmdSmiStatus.SUCCESS
INVAL = amdsmi.AmdSmiStatus.INVAL
NOT_SUPPORTED = amdsmi.AmdSmiStatus.NOT_SUPPORTED


def _raise(status):
    """Fail a call the way the amdsmi wrapper does, with the status attached."""
    raise amdsmi.AmdSmiLibraryException(status.value)


class TestExpectStatusAlone(unittest.TestCase):
    """Without a sweep, the verdict is delivered at the call that produced it."""

    def setUp(self):
        self.common = Common.for_unit_tests()

    def test_accepted_status_is_not_a_failure(self):
        self.common.print_func_name("")
        with self.common.expect_status("msg", NOT_SUPPORTED):
            _raise(NOT_SUPPORTED)
        return

    def test_unaccepted_status_raises_naming_both_sides(self):
        self.common.print_func_name("")
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
        self.common.print_func_name("")
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", NOT_SUPPORTED):
                pass
        self.assertIn(PASS, str(ctx.exception))
        return

    def test_accept_is_taken_as_enum_or_name_and_scalar_or_list(self):
        self.common.print_func_name("")
        for accept in (SUCCESS, [SUCCESS], PASS, [PASS]):
            with self.subTest(accept=accept):
                with self.common.expect_status("msg", accept):
                    pass
        return

    def test_unknown_status_name_is_rejected_before_the_call(self):
        self.common.print_func_name("")
        with self.assertRaises(ValueError):
            with self.common.expect_status("msg", "AMDSMI_STATUS_NOPE"):
                self.fail("the block must not run")
        return

    def test_exception_without_a_status_propagates(self):
        self.common.print_func_name("")
        # Raised here to stand in for a bug in a test body: it carries no status code,
        # so it has to escape rather than be recorded as something the driver returned.
        with self.assertRaises(KeyError):
            with self.common.expect_status("msg", SUCCESS):
                raise KeyError("not an amdsmi failure")
        return


class TestExpectStatusInSweep(unittest.TestCase):
    """Inside a sweep, failures accumulate and are reported once at the end."""

    def setUp(self):
        self.common = Common.for_unit_tests()

    def test_sweep_reports_every_failed_call(self):
        self.common.print_func_name("")
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
        self.common.print_func_name("")
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
        self.common.print_func_name("")
        with self.common.status_sweep():
            with self.common.expect_status("msg", SUCCESS):
                pass
        self.assertIsNone(self.common._status_failures)
        return

    def test_inner_sweep_failure_does_not_reach_the_outer_one(self):
        self.common.print_func_name("")
        with self.common.status_sweep():
            with self.assertRaises(AssertionError):
                with self.common.status_sweep():
                    with self.common.expect_status("inner call", SUCCESS):
                        _raise(INVAL)
            # Leaving the outer sweep must stay silent, so it collected nothing.
        self.assertIsNone(self.common._status_failures)
        return

    def test_body_exception_discards_what_the_sweep_collected(self):
        self.common.print_func_name("")
        with self.assertRaises(KeyError):
            with self.common.status_sweep():
                with self.common.expect_status("msg", SUCCESS):
                    _raise(INVAL)
                # Stands in for a bug in a test body, which wins over what was collected.
                raise KeyError("not an amdsmi failure")
        self.assertIsNone(self.common._status_failures)
        return


class TestExpectStatusRaises(unittest.TestCase):
    """A call rejected before it has a status is judged by the exception it raises."""

    def setUp(self):
        self.common = Common.for_unit_tests()

    def test_expected_exception_is_not_a_failure(self):
        self.common.print_func_name("")
        with self.common.expect_status("msg", raises=amdsmi.AmdSmiParameterException):
            raise amdsmi.AmdSmiParameterException("x", int)
        return

    def test_tuple_of_exception_types_is_accepted(self):
        self.common.print_func_name("")
        with self.common.expect_status("msg", raises=(KeyError, ValueError)):
            raise ValueError("bad")
        return

    def test_returning_without_raising_fails(self):
        self.common.print_func_name("")
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", raises=ValueError):
                pass
        self.assertIn("returned without raising", str(ctx.exception))
        self.assertIn("ValueError", str(ctx.exception))
        return

    def test_a_different_exception_fails_and_is_chained(self):
        self.common.print_func_name("")
        with self.assertRaises(AssertionError) as ctx:
            with self.common.expect_status("msg", raises=ValueError):
                raise KeyError("wrong type")
        self.assertIn("raised KeyError", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, KeyError)
        return

    def test_failure_is_collected_by_an_open_sweep(self):
        self.common.print_func_name("")
        with self.assertRaises(AssertionError) as ctx:
            with self.common.status_sweep():
                with self.common.expect_status("first", raises=ValueError):
                    pass
                with self.common.expect_status("second", raises=ValueError):
                    raise KeyError("wrong type")
        message = str(ctx.exception)
        self.assertIn("2 calls", message)
        self.assertIn("first", message)
        self.assertIn("second", message)
        return

    def test_exactly_one_of_accept_and_raises_is_required(self):
        self.common.print_func_name("")
        for kwargs in ({}, {"accept": SUCCESS, "raises": ValueError}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    with self.common.expect_status("msg", **kwargs):
                        self.fail("the block must not run")
        return
