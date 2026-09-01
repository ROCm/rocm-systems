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
            side_effect=self._events_from_mock(
                [(evt, b"pid: 1  a"), (evt, b"pid: 2  b")]
            ),
        ):
            recs = reader.read(2000)

        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertIn("timestamp", rec)


if __name__ == "__main__":
    unittest.main()
