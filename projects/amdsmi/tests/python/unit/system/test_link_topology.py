#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
        # The unified struct must be 64 bytes so the baremetal and host
        # interfaces are binary compatible.
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

        # Field offsets must match the C ABI exactly; a reorder that preserves
        # the 64-byte size would otherwise pass silently and break host ABI.
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
        # Argument validation happens before any library call, so this raises
        # without needing a GPU present.
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology("not-a-handle", "also-bad")

    def test_rejects_bad_destination_handle(self):
        # The both-bad case above trips on the first argument, so a valid source
        # with an invalid destination is what exercises the second-argument guard.
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        with self.assertRaises(amdsmi.amdsmi_interface.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_link_topology(src, "also-bad")

    def test_success_path_returns_mapped_dict(self):
        # Mock the ctypes entry point so the success path runs without a GPU,
        # locking the struct-to-dict mapping.
        src = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        dst = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

        def _fill(_src, _dst, topology_ref):
            # topology_ref is the ctypes byref() argument; ._obj is the
            # underlying amdsmi_link_topology_t the binding reads back.
            topology = topology_ref._obj
            topology.weight = 42
            # link_type XGMI (2) is a concrete type, so a coherent result pairs
            # it with link_status ENABLED (0).
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
