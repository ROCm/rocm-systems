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

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_INTERFACE_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "..", "py-interface")
)
_CLI_DIR = find_cli_dir(*cli_search_order(_THIS_DIR))


def _load_source_amdsmi_interface():
    """Load the amd-smi Python interface under test.

    Prefer the in-tree ``py-interface`` copy so the test exercises the wrapper
    carrying local changes rather than a possibly-stale installed package. The
    module is loaded as a submodule of a synthetic package so its
    ``from . import amdsmi_wrapper`` resolves to the source wrapper too. When the
    source tree is absent (the installed CI distro legs ship no ``py-interface``
    directory), fall back to the installed ``amdsmi`` package, which already
    reflects the build under test.
    """
    candidate = os.path.join(_SOURCE_INTERFACE_DIR, "amdsmi_interface.py")
    if not os.path.isfile(candidate):
        from amdsmi import amdsmi_interface as installed

        return installed

    package_name = "amdsmi_source_under_test"
    package = types.ModuleType(package_name)
    package.__path__ = [_SOURCE_INTERFACE_DIR]
    sys.modules[package_name] = package

    spec = importlib.util.spec_from_file_location(f"{package_name}.amdsmi_interface", candidate)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


ai = _load_source_amdsmi_interface()


def _load_amdsmi_logger_cls():
    """Load ``AMDSMILogger`` from the resolved CLI dir, or ``None`` when unavailable.

    The CLI is not an importable top-level package in the installed test
    environment (it ships under ``libexec``), so the module is loaded by path. Its
    only non-stdlib dependency (``amdsmi_helpers``) must be stubbed by the caller
    via ``stub_modules`` before this runs.
    """
    candidate = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else ""
    if not candidate or not os.path.isfile(candidate):
        return None
    spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", candidate)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.AMDSMILogger


def _load_event_commands_cls():
    candidate = os.path.join(_CLI_DIR, "subcommands", "event.py") if _CLI_DIR else ""
    if not candidate or not os.path.isfile(candidate):
        return None
    spec = importlib.util.spec_from_file_location("event_under_test", candidate)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded.EventCommands


class TestEventTimestamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = fake_module("amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {}))
        stub_modules(cls, {"amdsmi_helpers": helpers})
        cls.logger_cls = _load_amdsmi_logger_cls()
        cls.event_cls = _load_event_commands_cls()

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
        if self.logger_cls is None:
            self.skipTest("amdsmi_logger.py not found in source or install")
        logger = self.logger_cls()
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
        if self.event_cls is None:
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

        command = self.event_cls()
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
