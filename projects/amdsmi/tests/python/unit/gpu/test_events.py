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
"""GPU performance counter and event notification APIs."""

import unittest

import common.common as common


class TestGpuEvents(common.ApiTestCase):
    def test_gpu_counter_group_supported(self):
        self.both(
            "amdsmi_gpu_counter_group_supported",
            self.handle,
            common.enum("event_group", common.EVENT_GROUPS),
        )

    def test_get_gpu_available_counters(self):
        self.both(
            "amdsmi_get_gpu_available_counters",
            self.handle,
            common.enum("event_group", common.EVENT_GROUPS),
        )

    def test_gpu_create_counter(self):
        # Allocates a counter that must then be controlled and destroyed; that
        # lifecycle belongs to the functional tier.
        self.reject_only(
            "amdsmi_gpu_create_counter", self.handle, common.enum("event_type", common.EVENT_TYPES)
        )

    def test_gpu_destroy_counter(self):
        self.reject_only("amdsmi_gpu_destroy_counter", common.opaque("event_handle"))

    def test_gpu_control_counter(self):
        event = common.Param(
            "event_handle",
            ("null", common.amdsmi.amdsmi_wrapper.amdsmi_event_handle_t()),
            [("invalid", common.BAD_HANDLE)],
        )
        self.reject_only(
            "amdsmi_gpu_control_counter",
            event,
            common.enum("counter_command", common.COUNTER_COMMANDS),
        )

    def test_gpu_read_counter(self):
        # Needs a counter created and started first.
        self.reject_only("amdsmi_gpu_read_counter", common.opaque("event_handle"))

    def test_init_gpu_event_notification(self):
        self.reject_only("amdsmi_init_gpu_event_notification", self.handle)

    def test_set_gpu_event_notification_mask(self):
        self.reject_only(
            "amdsmi_set_gpu_event_notification_mask", self.handle, common.integer("mask", 0)
        )

    def test_stop_gpu_event_notification(self):
        self.reject_only("amdsmi_stop_gpu_event_notification", self.handle)

    def test_get_gpu_event_notification(self):
        # Blocks for timeout_ms and needs notification started first.
        self.reject_only("amdsmi_get_gpu_event_notification", common.integer("timeout_ms", 1))


if __name__ == "__main__":
    unittest.main()
