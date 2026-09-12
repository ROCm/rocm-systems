#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU performance counter and event notification APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuEvents(api.ApiTestCase):
    def test_gpu_counter_group_supported(self):
        self.both(
            "amdsmi_gpu_counter_group_supported",
            self.handle,
            api.enum("event_group", common.EVENT_GROUPS),
        )

    def test_get_gpu_available_counters(self):
        self.both(
            "amdsmi_get_gpu_available_counters",
            self.handle,
            api.enum("event_group", common.EVENT_GROUPS),
        )

    def test_gpu_create_counter(self):
        # Allocates a counter that must then be controlled and destroyed; that
        # lifecycle belongs to the functional tier.
        self.reject_only(
            "amdsmi_gpu_create_counter", self.handle, api.enum("event_type", common.EVENT_TYPES)
        )

    def test_gpu_destroy_counter(self):
        self.reject_only("amdsmi_gpu_destroy_counter", api.opaque("event_handle"))

    def test_gpu_control_counter(self):
        event = api.Param(
            "event_handle",
            ("null", common.amdsmi.amdsmi_wrapper.amdsmi_event_handle_t()),
            [("invalid", api.BAD_HANDLE)],
        )
        self.reject_only(
            "amdsmi_gpu_control_counter",
            event,
            api.enum("counter_command", common.COUNTER_COMMANDS),
        )

    def test_gpu_read_counter(self):
        # Needs a counter created and started first.
        self.reject_only("amdsmi_gpu_read_counter", api.opaque("event_handle"))

    def test_init_gpu_event_notification(self):
        self.reject_only("amdsmi_init_gpu_event_notification", self.handle)

    def test_set_gpu_event_notification_mask(self):
        self.reject_only(
            "amdsmi_set_gpu_event_notification_mask", self.handle, api.integer("mask", 0)
        )

    def test_stop_gpu_event_notification(self):
        self.reject_only("amdsmi_stop_gpu_event_notification", self.handle)

    def test_get_gpu_event_notification(self):
        # Blocks for timeout_ms and needs notification started first.
        self.reject_only("amdsmi_get_gpu_event_notification", api.integer("timeout_ms", 1))


if __name__ == "__main__":
    unittest.main()
