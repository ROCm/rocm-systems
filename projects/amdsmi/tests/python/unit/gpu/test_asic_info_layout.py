#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ASIC info struct layout and revision id reporting (hardware-free)."""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

_ASIC_INFO_SIZE = 896
_CHIP_REV_ID_OFFSET = 828
_EXTERNAL_REV_ID_OFFSET = 832
_RESERVED_OFFSET = 836
_UINT32_MAX = 0xFFFFFFFF
_REVISION_KEYS = ("chip_rev_id", "external_rev_id")


def _asic_info_with(chip: int = None, external: int = None, rev_id: int = _UINT32_MAX) -> dict:
    """Return the public dict with a stub library that writes the given revisions.

    Leaving a revision None simulates a library predating the fields, which
    leaves the reserved slots exactly as the caller handed them over.
    """
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _stub(_handle, info_ptr):
        info_ptr._obj.rev_id = rev_id
        if chip is not None:
            info_ptr._obj.chip_rev_id = chip
        if external is not None:
            info_ptr._obj.external_rev_id = external
        return 0

    with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_asic_info", _stub):
        return amdsmi.amdsmi_interface.amdsmi_get_gpu_asic_info(handle)


class TestAsicInfoLayout(unittest.TestCase):
    """Pins the slots the revision ids took out of the reserved tail."""

    def test_struct_size_unchanged(self):
        self.assertEqual(
            ctypes.sizeof(amdsmi.amdsmi_wrapper.struct_amdsmi_asic_info_t), _ASIC_INFO_SIZE
        )

    def test_new_field_and_reserved_offsets(self):
        struct_type = amdsmi.amdsmi_wrapper.struct_amdsmi_asic_info_t
        self.assertEqual(struct_type.chip_rev_id.offset, _CHIP_REV_ID_OFFSET)
        self.assertEqual(struct_type.external_rev_id.offset, _EXTERNAL_REV_ID_OFFSET)
        self.assertEqual(struct_type.reserved.offset, _RESERVED_OFFSET)

    def test_reserved_tail_fills_struct(self):
        reserved = amdsmi.amdsmi_wrapper.struct_amdsmi_asic_info_t.reserved
        self.assertEqual(reserved.offset + reserved.size, _ASIC_INFO_SIZE)


class TestRevisionIdReporting(unittest.TestCase):
    """The not-supported value must render N/A, never a plausible revision."""

    def test_reports_the_value_the_library_wrote(self):
        asic_info = _asic_info_with(0x47, 0x47)
        for key in _REVISION_KEYS:
            self.assertEqual(asic_info[key], "0x47")

    def test_each_key_reports_its_own_field(self):
        # Distinct values catch a chip/external swap that equal values hide.
        asic_info = _asic_info_with(0x11, 0x22)
        self.assertEqual(asic_info["chip_rev_id"], "0x11")
        self.assertEqual(asic_info["external_rev_id"], "0x22")

    def test_pads_to_two_hex_digits(self):
        asic_info = _asic_info_with(0x8, 0x8)
        for key in _REVISION_KEYS:
            self.assertEqual(asic_info[key], "0x08")

    def test_not_supported_renders_na(self):
        asic_info = _asic_info_with(_UINT32_MAX, _UINT32_MAX)
        for key in _REVISION_KEYS:
            self.assertEqual(asic_info[key], "N/A")

    def test_zero_is_a_real_value_not_na(self):
        # Hardware really does report chip_rev 0, so zero must stay distinct from N/A.
        asic_info = _asic_info_with(0x0, 0x0)
        for key in _REVISION_KEYS:
            self.assertEqual(asic_info[key], "0x00")

    def test_library_without_the_fields_renders_na(self):
        asic_info = _asic_info_with()
        for key in _REVISION_KEYS:
            self.assertEqual(asic_info[key], "N/A")

    def test_rev_id_not_supported_renders_na(self):
        # rev_id shares the not-supported value, so it must not leak it raw.
        self.assertEqual(_asic_info_with(0x47, 0x47)["rev_id"], "N/A")


if __name__ == "__main__":
    unittest.main()
