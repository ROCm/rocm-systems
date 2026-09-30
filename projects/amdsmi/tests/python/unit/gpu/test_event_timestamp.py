#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression tests for event receipt timestamps."""

import ctypes
import importlib.util
import os
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from common.common import amdsmi_path
except (ImportError, FileNotFoundError):  # pragma: no cover - harness/install unavailable
    amdsmi_path = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_CLI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "..", "amdsmi_cli"))
_SOURCE_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)
_INSTALLED_CLI_DIR = (
    os.path.join(os.path.dirname(os.path.dirname(amdsmi_path)), "libexec", "amdsmi_cli")
    if amdsmi_path
    else ""
)


def _load_source_amdsmi_interface():
    package_name = "amdsmi_source_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [_SOURCE_INTERFACE_DIR]
    sys.modules[package_name] = package

    candidate = os.path.join(_SOURCE_INTERFACE_DIR, "amdsmi_interface.py")
    spec = importlib.util.spec_from_file_location(f"{package_name}.amdsmi_interface", candidate)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


ai = _load_source_amdsmi_interface()


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


def _load_event_commands_cls():
    for cli_dir in (_SOURCE_CLI_DIR, _INSTALLED_CLI_DIR):
        candidate = os.path.join(cli_dir, "subcommands", "event.py") if cli_dir else ""
        if candidate and os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("event_under_test", candidate)
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
            return loaded.EventCommands
    return None


EventCommands = _load_event_commands_cls()


class TestEventTimestamp(unittest.TestCase):
    def _events_from_mock(self, event_values):
        def _notify(timeout, count_ref, event_info):
            for idx, item in enumerate(event_values):
                event_info[idx].event = int(item[0])
                event_info[idx].processor_handle = 0
                event_info[idx].message = item[1]
            ctypes.cast(count_ref, ctypes.POINTER(ctypes.c_uint32))[0] = len(event_values)
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

    def test_multiple_gpus_use_one_event_reader_thread(self):
        if EventCommands is None:
            self.skipTest("event.py not found in source or install")

        threads = []
        readers = []
        stored_events = []

        class FakeReader:
            def __init__(self, device, event_types):
                self.read_count = 0
                readers.append(self)

            def read(self, timeout):
                self.read_count += 1
                command.stop = True
                return [
                    {
                        "processor_handle": 1,
                        "timestamp": 100,
                        "event": "PROCESS_START",
                        "message": "pid: 1  task: foo",
                    },
                    {
                        "processor_handle": 1,
                        "timestamp": 101,
                        "event": "PROCESS_END",
                        "message": "pid: 1  task: foo",
                    },
                ]

            def stop(self):
                pass

        class FakeThread:
            def __init__(self, target, args):
                self.target = target
                self.args = args
                threads.append(self)

            def start(self):
                self.target(*self.args)

            def join(self):
                pass

        class FakeLogger:
            def store_event_output(self, processor_handle, values):
                stored_events.append(dict(values))

            def print_event_output(self):
                pass

        command = EventCommands()
        command.device_handles = [object(), object()]
        command.group_check_printed = True
        command.logger = FakeLogger()
        args = SimpleNamespace(gpu=None)

        with (
            mock.patch("event_under_test.threading.Thread", FakeThread),
            mock.patch("event_under_test.amdsmi_interface.AmdSmiEventReader", FakeReader),
            mock.patch("builtins.input", return_value="q"),
            mock.patch("builtins.print"),
        ):
            command.event(args)

        self.assertEqual(len(threads), 1)
        self.assertEqual(threads[0].args[1], command.device_handles)
        self.assertEqual([reader.read_count for reader in readers], [1, 0])
        self.assertEqual(
            [event["event"] for event in stored_events], ["PROCESS_START", "PROCESS_END"]
        )


if __name__ == "__main__":
    unittest.main()
