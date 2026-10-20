#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for streaming event CSV/JSON output.

The event command prints one record at a time, once per event. These tests lock
in that each streaming format emits a single well-formed record per event: CSV
writes the header exactly once
followed by a row per event (no repeated headers, no blank separator lines),
and JSON emits one object per line (newline-delimited JSON).

Loads ``amdsmi_logger`` directly with its only non-stdlib dependency
(``amdsmi_helpers``) stubbed, so the formatting logic is exercised without GPU
hardware or the compiled ``amdsmi`` package. Prefers the in-tree source
checkout, falling back to an installed CLI.
"""

import importlib.util
import io
import json
import os
import unittest
from contextlib import redirect_stdout

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else ""


def _fake_helpers_modules():
    """``amdsmi_logger``'s only non-stdlib import, stubbed so it loads hardware-free."""
    helpers = fake_module("amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {}))
    return {"amdsmi_helpers": helpers}


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records():
    """Event payloads in the shape the read loop hands to ``store_event_output``:
    a device handle plus a ``{timestamp, event, message}`` dict with a nested
    message. Routing through the real store path means these tests lock in the
    schema the product actually emits (``gpu,timestamp,event,message``)."""
    return [
        (
            1,
            {
                "timestamp": 1790357343,
                "event": "PROCESS_START",
                "message": {"PID": "1604545", "task": "rocminfo"},
            },
        ),
        (
            0,
            {
                "timestamp": 1790357343,
                "event": "PROCESS_START",
                "message": {"PID": "1604545", "task": "rocminfo"},
            },
        ),
    ]


def _capture_event_stream(logger, records):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        for device_handle, values in records:
            logger.store_event_output(device_handle, values)
            logger.print_event_output()
    return buffer.getvalue()


class _LoggerTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LOGGER_PATH:
            raise unittest.SkipTest("amdsmi_logger.py not found in source or install")
        stub_modules(cls, _fake_helpers_modules())
        cls.module = _load_logger_module()

    def _make_logger(self, output_format):
        return self.module.AMDSMILogger(format=output_format, helpers=_IdentityHelpers())


class _IdentityHelpers:
    """Minimal helpers stub: treats the device handle as the GPU id."""

    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle


class TestEventCsvOutput(_LoggerTestBase):
    def test_header_printed_once_followed_by_rows(self):
        logger = self._make_logger("csv")
        records = _records()

        output = _capture_event_stream(logger, records)
        lines = [line for line in output.splitlines() if line != ""]

        header = "gpu,timestamp,event,message"
        self.assertEqual(lines[0], header)
        self.assertEqual(sum(1 for line in lines if line == header), 1)
        self.assertEqual(len(lines), len(records) + 1)
        self.assertEqual(
            lines[1], '1,1790357343,PROCESS_START,"{""PID"":""1604545"",""task"":""rocminfo""}"'
        )
        self.assertEqual(
            lines[2], '0,1790357343,PROCESS_START,"{""PID"":""1604545"",""task"":""rocminfo""}"'
        )

    def test_no_blank_separator_lines(self):
        logger = self._make_logger("csv")

        output = _capture_event_stream(logger, _records())

        self.assertNotIn("\n\n", output)
        self.assertNotIn("\r", output)

    def test_mixed_schema_events_do_not_leak_stale_values(self):
        """A later event with a different message schema must not inherit a
        prior event's flattened message columns.

        The CSV header is fixed from the first event, so an event that lacks
        those columns must render them as ``N/A`` rather than repeating the
        previous event's values. Exercises the real store path
        (``store_event_output``) which resets and re-flattens per event.
        """
        logger = self.module.AMDSMILogger(format="csv", helpers=_IdentityHelpers())

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            logger.store_event_output(
                0,
                {
                    "timestamp": 100,
                    "event": "PROCESS_START",
                    "message": {"pid": "1", "task": "foo"},
                },
            )
            logger.print_event_output()
            logger.store_event_output(
                0, {"timestamp": 101, "event": "VMFAULT", "message": {"addr": "0xdead"}}
            )
            logger.print_event_output()

        lines = [line for line in buffer.getvalue().splitlines() if line != ""]

        self.assertEqual(lines[0], "gpu,timestamp,event,message")
        self.assertEqual(lines[1], '0,100,PROCESS_START,"{""pid"":""1"",""task"":""foo""}"')
        self.assertEqual(lines[2], '0,101,VMFAULT,"{""addr"":""0xdead""}"')
        self.assertNotIn('"pid":', lines[2])
        self.assertNotIn("foo", lines[2])


class TestEventJsonOutput(_LoggerTestBase):
    def test_one_json_object_per_line(self):
        logger = self._make_logger("json")
        records = _records()

        output = _capture_event_stream(logger, records)
        lines = [line for line in output.splitlines() if line != ""]

        self.assertEqual(len(lines), len(records))
        for line, (device_handle, values) in zip(lines, records):
            expected = {
                "gpu": device_handle,
                "timestamp": values["timestamp"],
                "event": values["event"],
                "message": values["message"],
            }
            self.assertEqual(json.loads(line), expected)
