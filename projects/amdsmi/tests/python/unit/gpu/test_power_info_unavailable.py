#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Power info "unavailable" handling (hardware-free)."""

import unittest
from unittest import mock

from common.common import amdsmi

_FIELDS = (
    "socket_power",
    "current_socket_power",
    "average_socket_power",
    "gfx_voltage",
    "soc_voltage",
    "mem_voltage",
    "power_limit",
    "ubb_power",
)
_UINT64_FIELDS = ("socket_power", "gfx_voltage", "soc_voltage", "mem_voltage")


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
    """UINT32_MAX and UINT64_MAX mean "unavailable"; smaller maxima are readings."""

    def test_narrower_maxima_are_real_readings(self):
        for value in (0xFF, 0xFFFF):
            readings = dict.fromkeys(_FIELDS, value)
            self.assertEqual(_power_info_with(readings), readings)

    def test_uint64_max_reports_na(self):
        # Native Linux fills the uint64_t members with UINT64_MAX.
        values = dict.fromkeys(_FIELDS, 0xFFFFFFFF)
        values.update(dict.fromkeys(_UINT64_FIELDS, 0xFFFFFFFFFFFFFFFF))
        self.assertEqual(_power_info_with(values), dict.fromkeys(_FIELDS, "N/A"))

    def test_uint32_max_reports_na_in_every_member(self):
        # The WSL backend copies a 32-bit "unavailable" sensor into the uint64_t members too.
        values = dict.fromkeys(_FIELDS, 0xFFFFFFFF)
        self.assertEqual(_power_info_with(values), dict.fromkeys(_FIELDS, "N/A"))

    def test_each_key_reports_its_own_field(self):
        readings = {field: index + 1 for index, field in enumerate(_FIELDS)}
        self.assertEqual(_power_info_with(readings), readings)
