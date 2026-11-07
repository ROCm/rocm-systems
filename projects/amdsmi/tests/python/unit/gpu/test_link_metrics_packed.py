#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_link_metrics() returns only the packed native entries."""

from __future__ import annotations

import unittest
from unittest import mock

from common.common import amdsmi


def _query(num_links: int, fill: int):
    def fake(_handle, metrics_ptr):
        metrics = metrics_ptr._obj
        metrics.num_links = num_links
        for index in range(fill):
            link = metrics.links[index]
            link.bdf.as_uint = 0x500 + index
            link.bit_rate = 36
            link.max_bandwidth = 576
            link.read = 10 + index
            link.write = 20 + index
        return amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
    with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_link_metrics", side_effect=fake):
        return amdsmi.amdsmi_get_link_metrics(handle)


class TestLinkMetricsPacked(unittest.TestCase):
    def test_returns_only_counted_links(self):
        result = _query(num_links=3, fill=8)
        self.assertEqual(result["num_links"], 3)
        self.assertEqual([link["read"] for link in result["links"]], [10, 11, 12])

    def test_no_links(self):
        result = _query(num_links=0, fill=8)
        self.assertEqual(result, {"num_links": 0, "links": []})

    def test_count_is_bounded_by_capacity(self):
        capacity = len(amdsmi.amdsmi_wrapper.amdsmi_link_metrics_t().links)
        result = _query(num_links=0xFFFFFFFF, fill=0)
        self.assertEqual(result["num_links"], capacity)
        self.assertEqual(len(result["links"]), capacity)
