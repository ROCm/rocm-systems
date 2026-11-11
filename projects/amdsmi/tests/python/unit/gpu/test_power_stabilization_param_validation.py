#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for
``amdsmi_set_power_stabilization_param()``'s int64 value validation.

``int(value)`` silently truncates a non-whole float (``int(3.9) == 3``) and
raises a raw ``ValueError``/``OverflowError`` -- not the documented
``AmdSmiParameterException`` -- for NaN/inf. These tests pin the fix: a float
value for an int64 param_id must be a whole, finite number, and non-whole
floats (including NaN/inf) raise ``AmdSmiParameterException`` instead of
truncating or leaking a raw Python exception.
"""

import ctypes
import importlib.util
import math
import os
import sys
import types
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)


def _load_source_amdsmi_interface():
    """Load amdsmi_interface.py from the in-tree py-interface source so this
    test exercises local changes, not a possibly-stale installed package. See
    test_event_timestamp.py for the same pattern."""
    candidate = os.path.join(_SOURCE_INTERFACE_DIR, "amdsmi_interface.py")
    if not os.path.isfile(candidate):
        from amdsmi import amdsmi_interface as installed

        return installed

    package_name = "amdsmi_source_under_test_power_stabilization"
    package = types.ModuleType(package_name)
    package.__path__ = [_SOURCE_INTERFACE_DIR]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(f"{package_name}.amdsmi_interface", candidate)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


ai = _load_source_amdsmi_interface()


@unittest.skipIf(
    not hasattr(ai, "amdsmi_set_power_stabilization_param"),
    "amdsmi_interface source does not expose amdsmi_set_power_stabilization_param",
)
class TestPowerStabilizationParamIntValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handle = ai.amdsmi_wrapper.amdsmi_processor_handle()
        cls.int_param = ai.AmdSmiPowerStabilizationParamId.POWER_FLOOR
        cls.double_param = ai.AmdSmiPowerStabilizationParamId.RAMP_MAX_RAMP_UP_RATE

    def setUp(self):
        # Overwrite directly rather than saving/restoring the previous
        # value: on a machine with a real (possibly stale) libamd_smi.so on
        # the loader path, merely *reading* the un-overridden attribute
        # resolves the symbol via ctypes and can raise AttributeError if that
        # library predates this function.
        self.calls = []

        def _fake_call(handle, param_id, value_ptr):
            # value_ptr is a ctypes.byref() CArgObject; cast() dereferences it
            # directly without needing a real ctypes pointer instance.
            value = ctypes.cast(
                value_ptr, ctypes.POINTER(ai.amdsmi_wrapper.amdsmi_value_t)
            ).contents
            self.calls.append((handle, param_id, value.data.int64_value))
            return ai.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        ai.amdsmi_wrapper.amdsmi_set_power_stabilization_param = _fake_call

    def tearDown(self):
        del ai.amdsmi_wrapper.amdsmi_set_power_stabilization_param

    def test_whole_number_float_is_accepted_and_truncates_safely(self):
        ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 70.0)
        self.assertEqual(self.calls[-1][2], 70)

    def test_int_value_is_accepted(self):
        ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 70)
        self.assertEqual(self.calls[-1][2], 70)

    def test_non_whole_float_raises_instead_of_silently_truncating(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 3.9)
        self.assertEqual(self.calls, [], "must not reach the wrapper call")

    def test_nan_raises_parameter_exception_not_raw_value_error(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, math.nan)
        self.assertEqual(self.calls, [])

    def test_positive_inf_raises_parameter_exception_not_raw_overflow_error(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, math.inf)
        self.assertEqual(self.calls, [])

    def test_negative_inf_raises_parameter_exception(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, -math.inf)
        self.assertEqual(self.calls, [])

    def test_double_param_still_accepts_non_whole_float(self):
        # RAMP_MAX_RAMP_UP_RATE takes AMDSMI_VALUE_TYPE_DOUBLE -- the new
        # whole-number check must only apply to the int64 branch.
        ai.amdsmi_set_power_stabilization_param(self.handle, self.double_param, 3.9)
