#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_gpu_cpu_links() Python wrapper unit tests (hardware-free)."""

from __future__ import annotations

import unittest
from unittest import mock

from common.common import amdsmi


class TestAmdSmiGetGpuCpuLinks(unittest.TestCase):
    def test_rejects_non_handle_argument(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_get_gpu_cpu_links("not_a_handle")

    def test_two_call_idiom_fetches_exact_count(self):
        wrapper = amdsmi.amdsmi_wrapper
        handle = wrapper.amdsmi_processor_handle(1)

        def _fill(_handle, num_cpu_links_ref, cpu_links_arg):
            count = num_cpu_links_ref._obj
            if not cpu_links_arg:
                # First call: NULL buffer, count-only query.
                count.value = 2
                return wrapper.AMDSMI_STATUS_SUCCESS
            # Second call: caller-sized buffer matching the reported count.
            self.assertEqual(count.value, 2)
            cpu_links_arg[0].package_id = 0
            cpu_links_arg[0].link_type = wrapper.AMDSMI_LINK_TYPE_XGMI
            cpu_links_arg[1].package_id = 1
            cpu_links_arg[1].link_type = wrapper.AMDSMI_LINK_TYPE_PCIE
            return wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(wrapper, "amdsmi_get_gpu_cpu_links", side_effect=_fill) as query:
            result = amdsmi.amdsmi_get_gpu_cpu_links(handle)

        self.assertEqual(query.call_count, 2)
        self.assertEqual(
            result,
            [
                {"package_id": 0, "link_type": wrapper.AMDSMI_LINK_TYPE_XGMI},
                {"package_id": 1, "link_type": wrapper.AMDSMI_LINK_TYPE_PCIE},
            ],
        )

    def test_empty_list_is_a_successful_result(self):
        wrapper = amdsmi.amdsmi_wrapper
        handle = wrapper.amdsmi_processor_handle(1)

        def _fill(_handle, num_cpu_links_ref, _cpu_links_arg):
            num_cpu_links_ref._obj.value = 0
            return wrapper.AMDSMI_STATUS_SUCCESS

        with mock.patch.object(wrapper, "amdsmi_get_gpu_cpu_links", side_effect=_fill) as query:
            result = amdsmi.amdsmi_get_gpu_cpu_links(handle)

        # The two-call idiom still runs in full even when the count is zero.
        self.assertEqual(query.call_count, 2)
        self.assertEqual(result, [])
