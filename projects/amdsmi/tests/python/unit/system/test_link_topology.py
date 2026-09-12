#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unified ``amdsmi_get_link_topology`` binding unit tests.

Hardware independent: validates that the ctypes ``amdsmi_link_topology_t``
structure matches the C ABI (64 bytes, matching the host struct), that the
high-level ``amdsmi_get_link_topology`` symbol is exported, and that argument
validation and the success-path dict mapping behave without a GPU present.
"""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi


class TestLinkTopology(unittest.TestCase):
    def test_struct_size_matches_host_abi(self):
        # 64 bytes keeps the baremetal and host interfaces binary compatible.
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

        # Offsets must match the C ABI; a same-size reorder would break it silently.
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
        # Validation happens before any library call, so no GPU is needed.
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology("not-a-handle", "also-bad")

    def test_rejects_bad_destination_handle(self):
        # A valid source with a bad destination exercises the second-argument guard.
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology(src, "also-bad")

    def test_success_path_returns_mapped_dict(self):
        # Mock the entry point so the success path runs without a GPU.
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        dst = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _fill(_src, _dst, topology_ref):
            # ._obj is the underlying struct the binding reads back.
            topology = topology_ref._obj
            topology.weight = 42
            # XGMI (2) is a concrete type, so pair it with link_status ENABLED (0).
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
