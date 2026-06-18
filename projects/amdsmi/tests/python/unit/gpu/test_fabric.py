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
"""Hardware-free unit tests for the fabric telemetry Python binding."""

from __future__ import annotations

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

ALL_CATEGORIES = amdsmi.amdsmi_wrapper.AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_ALL_KNOWN


class TestGpuFabricTelemetry(unittest.TestCase):
    def test_fabric_telemetry_unknown_name_fallback(self):
        """An unrecognized telemetry id resolves to a name of "UNKNOWN"."""
        wrapper = amdsmi.amdsmi_wrapper

        item = wrapper.amdsmi_fabric_telemetry_item_t(id=0xDEADBEEF, value=7)
        instance = wrapper.amdsmi_fabric_telemetry_instance_t()
        instance.item_count = 1
        instance.items = ctypes.pointer(item)
        dataset = wrapper.amdsmi_fabric_telemetry_dataset_t()
        dataset.instance_count = 1
        dataset.instances = ctypes.pointer(instance)
        telemetry = wrapper.amdsmi_fabric_telemetry_t()
        telemetry.datasets[0] = ctypes.pointer(dataset)
        # Keep the ctypes object graph alive for the duration of the call.
        keepalive = (item, instance, dataset, telemetry)  # noqa: F841

        def fake_alloc(handle, mask, tel_ref):
            tel_ref._obj.contents = telemetry
            return wrapper.AMDSMI_STATUS_SUCCESS

        with (
            mock.patch.object(wrapper, "amdsmi_alloc_fabric_telemetry", side_effect=fake_alloc),
            mock.patch.object(
                wrapper,
                "amdsmi_get_fabric_telemetry_data",
                return_value=wrapper.AMDSMI_STATUS_SUCCESS,
            ),
            mock.patch.object(
                wrapper,
                "amdsmi_fabric_telem_id_to_string",
                return_value=wrapper.AMDSMI_STATUS_NOT_FOUND,
            ),
            mock.patch.object(
                wrapper, "amdsmi_free_fabric_telemetry", return_value=wrapper.AMDSMI_STATUS_SUCCESS
            ),
        ):
            handle = wrapper.amdsmi_processor_handle()
            result = amdsmi.amdsmi_get_fabric_telemetry_data(handle, ALL_CATEGORIES)

        self.assertEqual(result[0]["instances"][0]["items"][0]["name"], "UNKNOWN")
        return

    def test_fabric_telemetry_resolved_name(self):
        """A recognized telemetry id decodes to the name written through the out-param."""
        wrapper = amdsmi.amdsmi_wrapper

        item = wrapper.amdsmi_fabric_telemetry_item_t(id=0x1, value=42)
        instance = wrapper.amdsmi_fabric_telemetry_instance_t()
        instance.item_count = 1
        instance.items = ctypes.pointer(item)
        dataset = wrapper.amdsmi_fabric_telemetry_dataset_t()
        dataset.instance_count = 1
        dataset.instances = ctypes.pointer(instance)
        telemetry = wrapper.amdsmi_fabric_telemetry_t()
        telemetry.datasets[0] = ctypes.pointer(dataset)
        name_buf = ctypes.create_string_buffer(b"NETPORT_FEC_CW")
        # Keep the ctypes object graph alive for the duration of the call.
        keepalive = (item, instance, dataset, telemetry, name_buf)  # noqa: F841

        def fake_alloc(handle, mask, tel_ref):
            tel_ref._obj.contents = telemetry
            return wrapper.AMDSMI_STATUS_SUCCESS

        def fake_id_to_string(telem_id, name_ref):
            # Write name_buf's address into the caller's POINTER(c_char), the
            # same contract the C function fulfills on success.
            addr_slot = ctypes.cast(
                ctypes.addressof(name_ref._obj), ctypes.POINTER(ctypes.c_void_p)
            )
            addr_slot[0] = ctypes.addressof(name_buf)
            return wrapper.AMDSMI_STATUS_SUCCESS

        with (
            mock.patch.object(wrapper, "amdsmi_alloc_fabric_telemetry", side_effect=fake_alloc),
            mock.patch.object(
                wrapper,
                "amdsmi_get_fabric_telemetry_data",
                return_value=wrapper.AMDSMI_STATUS_SUCCESS,
            ),
            mock.patch.object(
                wrapper, "amdsmi_fabric_telem_id_to_string", side_effect=fake_id_to_string
            ),
            mock.patch.object(
                wrapper, "amdsmi_free_fabric_telemetry", return_value=wrapper.AMDSMI_STATUS_SUCCESS
            ),
        ):
            handle = wrapper.amdsmi_processor_handle()
            result = amdsmi.amdsmi_get_fabric_telemetry_data(handle, ALL_CATEGORIES)

        self.assertEqual(result[0]["instances"][0]["items"][0]["name"], "NETPORT_FEC_CW")
        return


if __name__ == "__main__":
    unittest.main()
