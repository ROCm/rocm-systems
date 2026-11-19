#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for
``amdsmi_set_power_stabilization_param()``'s int64 value validation.

The C API takes a signed 64-bit integer only, so the Python API accepts
``int`` and rejects everything else -- including ``bool`` (an ``int``
subclass) and ``float`` -- with ``AmdSmiParameterException``, and rejects
out-of-range ``int`` values the same way.
"""

import ctypes
import importlib.util
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

_INFO_STRUCT_SIZE = 128
_RAMP_MAX_RAMP_UP_RATE_OFFSET = 20
_RAMP_MAX_RAMP_DOWN_RATE_OFFSET = 24
_RAMP_MIN_TIME_AT_POWER_FLOOR_OFFSET = 28
_RAMP_STOP_DELAY_OFFSET = 32
_RESERVED_OFFSET = 40
_RESERVED_COUNT = 11


@unittest.skipIf(
    not hasattr(ai.amdsmi_wrapper, "struct_amdsmi_power_stabilization_info_t"),
    "amdsmi_wrapper does not expose struct_amdsmi_power_stabilization_info_t",
)
class TestPowerStabilizationInfoLayout(unittest.TestCase):
    """Pins the all-integer ramp fields' offsets and reserved[11] tail
    (see test_asic_info_layout.py for the same pattern on a sibling struct)."""

    def test_struct_size_unchanged(self):
        struct_type = ai.amdsmi_wrapper.struct_amdsmi_power_stabilization_info_t
        self.assertEqual(ctypes.sizeof(struct_type), _INFO_STRUCT_SIZE)

    def test_ramp_field_offsets(self):
        struct_type = ai.amdsmi_wrapper.struct_amdsmi_power_stabilization_info_t
        self.assertEqual(struct_type.ramp_max_ramp_up_rate.offset, _RAMP_MAX_RAMP_UP_RATE_OFFSET)
        self.assertEqual(
            struct_type.ramp_max_ramp_down_rate.offset, _RAMP_MAX_RAMP_DOWN_RATE_OFFSET
        )
        self.assertEqual(
            struct_type.ramp_min_time_at_power_floor.offset, _RAMP_MIN_TIME_AT_POWER_FLOOR_OFFSET
        )
        self.assertEqual(struct_type.ramp_stop_delay.offset, _RAMP_STOP_DELAY_OFFSET)

    def test_reserved_tail_size_and_offset(self):
        reserved = ai.amdsmi_wrapper.struct_amdsmi_power_stabilization_info_t.reserved
        self.assertEqual(reserved.offset, _RESERVED_OFFSET)
        self.assertEqual(reserved.size, _RESERVED_COUNT * ctypes.sizeof(ctypes.c_uint64))
        self.assertEqual(reserved.offset + reserved.size, _INFO_STRUCT_SIZE)


@unittest.skipIf(
    not hasattr(ai, "amdsmi_set_power_stabilization_param"),
    "amdsmi_interface source does not expose amdsmi_set_power_stabilization_param",
)
class TestPowerStabilizationParamIntValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handle = ai.amdsmi_wrapper.amdsmi_processor_handle()
        cls.int_param = ai.AmdSmiPowerStabilizationParamId.POWER_FLOOR

    def setUp(self):
        # Overwrite directly rather than saving/restoring the previous
        # value: on a machine with a real (possibly stale) libamd_smi.so on
        # the loader path, merely *reading* the un-overridden attribute
        # resolves the symbol via ctypes and can raise AttributeError if that
        # library predates this function.
        self.calls = []

        def _fake_call(handle, param_id, value):
            self.calls.append((handle, param_id, value))
            return ai.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        ai.amdsmi_wrapper.amdsmi_set_power_stabilization_param = _fake_call

    def tearDown(self):
        del ai.amdsmi_wrapper.amdsmi_set_power_stabilization_param

    def test_int_value_is_accepted(self):
        ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 70)
        self.assertEqual(self.calls[-1][2], 70)

    def test_bool_raises_parameter_exception(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, True)
        self.assertEqual(self.calls, [], "must not reach the wrapper call")

    def test_whole_number_float_raises_parameter_exception(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 70.0)
        self.assertEqual(self.calls, [])

    def test_out_of_range_int_raises_parameter_exception(self):
        with self.assertRaises(ai.AmdSmiParameterException):
            ai.amdsmi_set_power_stabilization_param(self.handle, self.int_param, 2**63)
        self.assertEqual(self.calls, [])
