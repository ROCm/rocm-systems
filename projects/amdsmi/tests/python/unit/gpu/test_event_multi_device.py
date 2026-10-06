#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free regression test for multi-device event fan-out.

The ``event`` subcommand builds one listener per GPU but reads through a single
listener, because ``AmdSmiEventReader.read()`` is a global poll
(``amdsmi_get_gpu_event_notification``) that returns queued events for every
registered device, each tagged with its own ``processor_handle``. This test
locks in that one read loop prints events from *all* devices, not just the first
listener -- the exact behavior a reviewer could not confirm from the code alone.

Loads the real ``subcommands/event.py`` and ``amdsmi_logger.py`` with only the
compiled ``amdsmi`` package faked, so the real read -> store -> print pipeline is
exercised without GPU hardware.
"""

import importlib.util
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
LOGGER_PATH = os.path.join(_CLI_DIR, "amdsmi_logger.py") if _CLI_DIR else ""
EVENT_PATH = os.path.join(_CLI_DIR, "subcommands", "event.py") if _CLI_DIR else ""

_NO_DATA = 10


class _FakeLibraryException(Exception):
    def __init__(self, err_code):
        self.err_code = err_code
        super().__init__(f"err {err_code}")


def _fake_amdsmi_modules(event_queue, on_drain):
    """Build the ``amdsmi`` package stub that ``event.py`` imports.

    ``read()`` serves the whole queue once (a single global poll) and then
    reports NO_DATA, mirroring the real driver draining all registered devices
    in one call. ``on_drain`` is invoked when the queue is exhausted so the
    caller can stop the read loop deterministically.
    """

    class FakeEventReader:
        registered = []
        _served = False

        def __init__(self, device, event_types):
            FakeEventReader.registered.append(device)

        def read(self, timeout_ms, num_elem=10):
            if FakeEventReader._served:
                on_drain()
                raise _FakeLibraryException(_NO_DATA)
            FakeEventReader._served = True
            return list(event_queue)

        def stop(self):
            pass

    amdsmi_wrapper = fake_module(
        "amdsmi.amdsmi_interface.amdsmi_wrapper", AMDSMI_STATUS_NO_DATA=_NO_DATA
    )
    amdsmi_interface = fake_module(
        "amdsmi.amdsmi_interface",
        AmdSmiEventReader=FakeEventReader,
        AmdSmiEvtNotificationType=object,
        amdsmi_wrapper=amdsmi_wrapper,
    )
    amdsmi_exception = fake_module(
        "amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException
    )
    amdsmi_pkg = fake_module(
        "amdsmi", amdsmi_interface=amdsmi_interface, amdsmi_exception=amdsmi_exception
    )
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": amdsmi_interface,
        "amdsmi.amdsmi_exception": amdsmi_exception,
    }, FakeEventReader


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _IdentityHelpers:
    """Minimal helpers stub: the device handle doubles as the GPU id."""

    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle

    def check_required_groups(self):
        pass


class TestEventMultiDeviceFanout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (LOGGER_PATH and EVENT_PATH):
            raise unittest.SkipTest("CLI event/logger sources not found in source or install")
        stub_modules(
            cls,
            {
                "amdsmi_helpers": fake_module(
                    "amdsmi_helpers", AMDSMIHelpers=type("AMDSMIHelpers", (), {})
                )
            },
        )
        cls.logger_mod = _load_module("amdsmi_logger_under_test", LOGGER_PATH)

    def _run_event_thread(self, event_queue, devices):
        commands = None

        def on_drain():
            commands.stop = True

        stubs, reader = _fake_amdsmi_modules(event_queue, on_drain)
        with mock.patch.dict(sys.modules, stubs):
            event_mod = _load_module("event_under_test", EVENT_PATH)
            helpers = _IdentityHelpers()
            commands = event_mod.EventCommands()
            commands.stop = False
            commands.group_check_printed = True
            commands.helpers = helpers
            commands.logger = self.logger_mod.AMDSMILogger(format="human_readable", helpers=helpers)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                commands._event_thread(commands, devices)
        return buffer.getvalue(), reader

    def test_events_from_all_devices_are_printed(self):
        queue = [
            {
                "timestamp": 100,
                "processor_handle": 0,
                "event": "PROCESS_START",
                "message": "PID: 1  ",
            },
            {
                "timestamp": 101,
                "processor_handle": 1,
                "event": "PROCESS_START",
                "message": "PID: 2  ",
            },
            {"timestamp": 102, "processor_handle": 2, "event": "VMFAULT", "message": "PID: 3  "},
            {
                "timestamp": 103,
                "processor_handle": 3,
                "event": "PROCESS_END",
                "message": "PID: 1  ",
            },
        ]

        output, reader = self._run_event_thread(queue, devices=[0, 1, 2, 3])

        # Every device was registered (one listener per GPU)...
        self.assertEqual(reader.registered, [0, 1, 2, 3])
        # ...and every device's event was printed from the single read loop.
        for gpu in (0, 1, 2, 3):
            self.assertIn(f"GPU: {gpu}", output)
        self.assertEqual(output.count("EVENT:"), len(queue))
        self.assertIn("EVENT: VMFAULT", output)


if __name__ == "__main__":
    unittest.main()
