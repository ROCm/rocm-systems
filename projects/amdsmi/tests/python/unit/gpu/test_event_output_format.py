#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for streaming event CSV/JSON output.

The event command prints one record at a time, once per event, from one
listener thread per GPU. These tests lock in that each streaming format emits
a single well-formed record per event: CSV writes the header exactly once
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
import sys
import types
import unittest
from contextlib import redirect_stdout

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


def _resolve_logger_path():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        candidate = os.path.join(cli_dir, "amdsmi_logger.py") if cli_dir else ""
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


LOGGER_PATH = _resolve_logger_path()


def _install_fake_helpers():
    """Register a stub ``amdsmi_helpers`` so ``amdsmi_logger`` imports cleanly."""
    module = types.ModuleType("amdsmi_helpers")
    module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    sys.modules["amdsmi_helpers"] = module


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records():
    return [
        {
            "gpu": 1,
            "timestamp": 1790357343,
            "event": "PROCESS_START",
            "PID": 1604545,
            "task": "rocminfo",
        },
        {
            "gpu": 0,
            "timestamp": 1790357343,
            "event": "PROCESS_START",
            "PID": 1604545,
            "task": "rocminfo",
        },
    ]


def _capture_event_stream(logger, records):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        for record in records:
            logger.output = dict(record)
            logger.print_event_output()
    return buffer.getvalue()


class _LoggerTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not LOGGER_PATH:
            raise unittest.SkipTest("amdsmi_logger.py not found in source or install")
        _install_fake_helpers()
        cls.module = _load_logger_module()

    def _make_logger(self, output_format):
        return self.module.AMDSMILogger(format=output_format)


class TestEventCsvOutput(_LoggerTestBase):
    def test_header_printed_once_followed_by_rows(self):
        logger = self._make_logger("csv")
        records = _records()

        output = _capture_event_stream(logger, records)
        lines = [line for line in output.splitlines() if line != ""]

        header = "gpu,timestamp,event,PID,task"
        self.assertEqual(lines[0], header)
        self.assertEqual(sum(1 for line in lines if line == header), 1)
        self.assertEqual(len(lines), len(records) + 1)
        self.assertEqual(lines[1], "1,1790357343,PROCESS_START,1604545,rocminfo")
        self.assertEqual(lines[2], "0,1790357343,PROCESS_START,1604545,rocminfo")

    def test_no_blank_separator_lines(self):
        logger = self._make_logger("csv")

        output = _capture_event_stream(logger, _records())

        self.assertNotIn("\n\n", output)
        self.assertNotIn("\r", output)


class TestEventJsonOutput(_LoggerTestBase):
    def test_one_json_object_per_line(self):
        logger = self._make_logger("json")
        records = _records()

        output = _capture_event_stream(logger, records)
        lines = [line for line in output.splitlines() if line != ""]

        self.assertEqual(len(lines), len(records))
        for line, record in zip(lines, records):
            self.assertEqual(json.loads(line), record)


if __name__ == "__main__":
    unittest.main()
