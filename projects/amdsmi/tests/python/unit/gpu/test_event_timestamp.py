#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for event receipt timestamps."""

from datetime import datetime
import unittest
from unittest import mock

from amdsmi import amdsmi_interface as ai


class TestEventTimestamp(unittest.TestCase):
    def _events_from_mock(self, event_values):
        def _notify(timeout, count_ref, event_info):
            for idx, item in enumerate(event_values):
                event_info[idx].event = int(item[0])
                event_info[idx].processor_handle = 0
                event_info[idx].message = item[1]
            count_ref._obj.value = len(event_values)
            return ai.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS

        return _notify

    def test_timestamp_present_and_rfc3339(self):
        evt = ai.AmdSmiEvtNotificationType.PROCESS_START
        reader = ai.AmdSmiEventReader.__new__(ai.AmdSmiEventReader)
        with mock.patch.object(
            ai.amdsmi_wrapper,
            "amdsmi_get_gpu_event_notification",
            side_effect=self._events_from_mock([(evt, b"pid: 1234  process started")]),
        ):
            rec = reader.read(2000)[0]

        self.assertIn("timestamp", rec)
        datetime.fromisoformat(rec["timestamp"])
        self.assertEqual(rec["event"], "PROCESS_START")

    def test_one_record_per_event(self):
        evt = ai.AmdSmiEvtNotificationType.PROCESS_START
        reader = ai.AmdSmiEventReader.__new__(ai.AmdSmiEventReader)
        with mock.patch.object(
            ai.amdsmi_wrapper,
            "amdsmi_get_gpu_event_notification",
            side_effect=self._events_from_mock([(evt, b"pid: 1  a"), (evt, b"pid: 2  b")]),
        ):
            recs = reader.read(2000)

        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertIn("timestamp", rec)


if __name__ == "__main__":
    unittest.main()
