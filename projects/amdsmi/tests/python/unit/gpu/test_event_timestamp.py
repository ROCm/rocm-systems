#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for event receipt timestamps."""

import importlib.util
import os
import sys
import time
import types
import unittest
from unittest import mock

from amdsmi import amdsmi_interface as ai

try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _load_amdsmi_logger_cls():
    """Load ``AMDSMILogger`` directly from the amdsmi_logger source/install file.

    ``amdsmi_cli`` is not an importable top-level package in the installed test
    environment (it ships under ``libexec/amdsmi_cli``), so ``import
    amdsmi_cli.amdsmi_logger`` raises ``ModuleNotFoundError`` on the CI distro
    legs. Mirror ``test_event_output_format.py`` and load the module by path with
    its only non-stdlib dependency (``amdsmi_helpers``) stubbed.
    """
    module = types.ModuleType("amdsmi_helpers")
    module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    sys.modules["amdsmi_helpers"] = module
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        candidate = os.path.join(cli_dir, "amdsmi_logger.py") if cli_dir else ""
        if candidate and os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", candidate)
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
            return loaded.AMDSMILogger
    return None


AMDSMILogger = _load_amdsmi_logger_cls()


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

    def test_timestamp_present_and_epoch_seconds(self):
        evt = ai.AmdSmiEvtNotificationType.PROCESS_START
        reader = ai.AmdSmiEventReader.__new__(ai.AmdSmiEventReader)
        before = int(time.time())
        with mock.patch.object(
            ai.amdsmi_wrapper,
            "amdsmi_get_gpu_event_notification",
            side_effect=self._events_from_mock([(evt, b"pid: 1234  process started")]),
        ):
            rec = reader.read(2000)[0]
        after = int(time.time())

        self.assertIn("timestamp", rec)
        self.assertIsInstance(rec["timestamp"], int)
        self.assertTrue(before <= rec["timestamp"] <= after)
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

    def test_human_readable_event_is_block_with_timestamp(self):
        if AMDSMILogger is None:
            self.skipTest("amdsmi_logger.py not found in source or install")
        logger = AMDSMILogger()
        event = {
            "gpu": 0,
            "timestamp": 1780000000,
            "event": "PROCESS_START",
            "message": {"pid": "1234", "process": "started"},
        }

        output = logger._format_event_human_readable(event)

        self.assertEqual(
            output,
            "GPU: 0\n"
            "    TIMESTAMP: 1780000000\n"
            "    EVENT: PROCESS_START\n"
            "    MESSAGE:\n"
            "        PID: 1234\n"
            "        PROCESS: started\n",
        )


if __name__ == "__main__":
    unittest.main()
