#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU-independent checks for topology ABI, exports, arguments, and dict mapping."""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi


class TestLinkTopology(unittest.TestCase):
    def test_struct_size_matches_host_abi(self):
        self.assertEqual(ctypes.sizeof(amdsmi.amdsmi_wrapper.amdsmi_link_topology_t), 64)

    def test_struct_fields(self):
        struct_type = amdsmi.amdsmi_wrapper.amdsmi_link_topology_t
        field_names = [name for name, *_ in struct_type._fields_]
        for expected in (
            "weight",
            "link_status",
            "link_type",
            "num_hops",
            "fb_sharing",
            "reserved",
        ):
            self.assertIn(expected, field_names)

        # Catch field reordering even when the total size is unchanged.
        self.assertEqual(struct_type.weight.offset, 0)
        self.assertEqual(struct_type.link_status.offset, 8)
        self.assertEqual(struct_type.link_type.offset, 12)
        self.assertEqual(struct_type.num_hops.offset, 16)
        self.assertEqual(struct_type.fb_sharing.offset, 17)
        self.assertEqual(struct_type.reserved.offset, 20)
        instance = struct_type()
        self.assertEqual(len(instance.reserved), 10)

    def test_symbol_is_exported(self):
        self.assertTrue(hasattr(amdsmi, "amdsmi_get_link_topology"))

    def test_rejects_non_handle_arguments(self):
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology("not-a-handle", "also-bad")

    def test_rejects_bad_destination_handle(self):
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology(src, "also-bad")

    def test_success_path_returns_mapped_dict(self):
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        dst = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _fill(_src, _dst, topology_ref):
            # Access the struct passed through ctypes.byref().
            topology = topology_ref._obj
            topology.weight = 42
            # ENABLED (0), XGMI (2).
            topology.link_status = 0
            topology.link_type = 2
            topology.num_hops = 3
            topology.fb_sharing = 1
            return 0

        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_link_topology", side_effect=_fill
        ):
            result = amdsmi.amdsmi_interface.amdsmi_get_link_topology(src, dst)

        self.assertEqual(
            set(result), {"weight", "link_status", "link_type", "num_hops", "fb_sharing"}
        )
        self.assertEqual(result["weight"], 42)
        self.assertEqual(result["link_status"], 0)
        self.assertEqual(result["link_type"], 2)
        self.assertEqual(result["num_hops"], 3)
        self.assertEqual(result["fb_sharing"], 1)

    def test_success_path_self_pair(self) -> None:
        wrapper = amdsmi.amdsmi_wrapper
        handle = wrapper.amdsmi_processor_handle(1)

        def _fill(src: ctypes.c_void_p, dst: ctypes.c_void_p, topology_ref: object) -> int:
            self.assertIs(src, handle)
            self.assertIs(dst, handle)
            topology = ctypes.cast(
                topology_ref, ctypes.POINTER(wrapper.amdsmi_link_topology_t)
            ).contents
            topology.weight = 0
            topology.link_status = wrapper.AMDSMI_LINK_STATUS_ENABLED
            topology.link_type = wrapper.AMDSMI_LINK_TYPE_INTERNAL
            topology.num_hops = 0
            topology.fb_sharing = 1
            return wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(wrapper, "amdsmi_get_link_topology", side_effect=_fill) as query:
            result = amdsmi.amdsmi_get_link_topology(handle, handle)

        query.assert_called_once()
        self.assertEqual(
            result,
            {
                "weight": 0,
                "link_status": wrapper.AMDSMI_LINK_STATUS_ENABLED,
                "link_type": wrapper.AMDSMI_LINK_TYPE_INTERNAL,
                "num_hops": 0,
                "fb_sharing": 1,
            },
        )
