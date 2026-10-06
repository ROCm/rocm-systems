#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Power info "unavailable" handling (hardware-free)."""

import unittest
from unittest import mock

from common.common import amdsmi

# The library marks an unavailable field with the maximum of the field's own width.
_UNAVAILABLE = {
    "socket_power": 0xFFFFFFFFFFFFFFFF,
    "current_socket_power": 0xFFFFFFFF,
    "average_socket_power": 0xFFFFFFFF,
    "gfx_voltage": 0xFFFFFFFFFFFFFFFF,
    "soc_voltage": 0xFFFFFFFFFFFFFFFF,
    "mem_voltage": 0xFFFFFFFFFFFFFFFF,
    "power_limit": 0xFFFFFFFF,
    "ubb_power": 0xFFFFFFFF,
}


def _power_info_with(values: dict) -> dict:
    """Return the public dict with a stub library that writes *values*."""
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _stub(_handle, info_ptr):
        for field, value in values.items():
            setattr(info_ptr._obj, field, value)
        return 0

    with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_power_info", _stub):
        return amdsmi.amdsmi_interface.amdsmi_get_power_info(handle)


class TestPowerInfoUnavailable(unittest.TestCase):
    """Only a field's own maximum means "unavailable"."""

    def test_narrower_maxima_are_real_readings(self):
        for value in (0xFF, 0xFFFF):
            readings = dict.fromkeys(_UNAVAILABLE, value)
            self.assertEqual(_power_info_with(readings), readings)

    def test_own_maximum_reports_na(self):
        self.assertEqual(_power_info_with(_UNAVAILABLE), dict.fromkeys(_UNAVAILABLE, "N/A"))

    def test_each_key_reports_its_own_field(self):
        readings = {field: index + 1 for index, field in enumerate(_UNAVAILABLE)}
        self.assertEqual(_power_info_with(readings), readings)
