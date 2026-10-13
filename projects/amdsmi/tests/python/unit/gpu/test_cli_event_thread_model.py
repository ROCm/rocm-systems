#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``amd-smi event`` threading model.

These tests drive ``EventCommands`` with ``amdsmi_interface``/``amdsmi_exception``
fully stubbed and ``threading.Thread.start``/``join`` short-circuited where needed
to run synchronously, so they run without GPU hardware or the compiled ``amdsmi``
package. They lock in the current contract:

* ``_open_listeners`` opens one ``AmdSmiEventReader`` per device on the main
  thread before any listener thread starts, so a device without event support
  is reported there instead of stranding a listener thread while ``input()``
  waits; it raises only when every device failed to open, and stops any
  already-opened listeners before propagating an unexpected error.
* ``_event_thread`` parses a batch's ``message`` string into a dict and logs
  it, swallows ``AMDSMI_STATUS_NO_DATA`` while printing any other
  ``AmdSmiLibraryException`` to stderr, isolates a single malformed event
  instead of dropping its batch siblings, and always stops its listener.
* ``event()`` reads stdin with a blocking ``input()`` loop (``EOFError`` /
  ``KeyboardInterrupt`` also stop it), and always joins the per-listener
  threads, stops every listener, and restores the previous SIGTERM handler
  before returning or re-raising.
* ``_event_sigterm_handler`` sets ``stop`` and converts SIGTERM into a
  ``SystemExit`` so the ``finally`` cleanup in ``event()`` still runs.
"""

import argparse
import io
import os
import signal
import threading
import time
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from common.common import cli_search_order, fake_module, find_cli_dir, load_cli_module, stub_modules

_CLI_DIR = find_cli_dir(*cli_search_order(os.path.dirname(os.path.abspath(__file__))))
EVENT_PATH = os.path.join(_CLI_DIR, "subcommands", "event.py") if _CLI_DIR else None

_NO_DATA = 996


class _FakeLibraryException(Exception):
    def __init__(self, err_code, message="mock boom"):
        super().__init__(message)
        self.err_code = err_code
        self._message = message

    def __str__(self):
        return self._message


def _build_fake_amdsmi():
    """Register a stub ``amdsmi`` package so ``event.py`` imports cleanly."""
    interface = fake_module(
        "amdsmi.amdsmi_interface",
        AmdSmiEventReader=None,  # set per-test
        AmdSmiEvtNotificationType=object(),
        amdsmi_wrapper=types.SimpleNamespace(AMDSMI_STATUS_NO_DATA=_NO_DATA),
    )
    exception = fake_module("amdsmi.amdsmi_exception", AmdSmiLibraryException=_FakeLibraryException)
    amdsmi_pkg = fake_module("amdsmi", amdsmi_interface=interface, amdsmi_exception=exception)
    return {
        "amdsmi": amdsmi_pkg,
        "amdsmi.amdsmi_interface": interface,
        "amdsmi.amdsmi_exception": exception,
    }


def _load_event_module():
    return load_cli_module("event_under_test", EVENT_PATH)


class _FakeLogger:
    def __init__(self):
        self.calls = []
        self.print_calls = 0

    def store_event_output(self, processor_handle, values_dict):
        self.calls.append((processor_handle, dict(values_dict)))

    def print_event_output(self):
        self.print_calls += 1


class _FakeHelpers:
    def __init__(self):
        self.check_required_groups_calls = 0

    def check_required_groups(self):
        self.check_required_groups_calls += 1

    # event.py calls get_gpu_id_from_device (no "_handle" suffix); the real
    # AMDSMIHelpers class only defines get_gpu_id_from_device_handle, so this
    # stub matches event.py's actual call site rather than the real API.
    def get_gpu_id_from_device_handle(self, device_handle):
        return device_handle

    def record_or_raise(self, exception, context=None):
        return f"Error occurred for {context} - {exception}"


def _build_event_args(gpu):
    return argparse.Namespace(gpu=gpu)


class TestOpenListeners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.exception_module = modules["amdsmi.amdsmi_exception"]
        cls.event_module = _load_event_module()

    def _make_commands(self):
        commands = object.__new__(self.event_module.EventCommands)
        commands.group_check_printed = True
        commands.helpers = _FakeHelpers()
        return commands

    def _make_reader_class(self, failing=()):
        exception_cls = self.exception_module.AmdSmiLibraryException

        class _Reader:
            def __init__(self, device, event_types):
                if device in failing:
                    raise exception_cls(1, f"boom-{device}")
                self.device = device
                self.stopped = False

            def stop(self):
                self.stopped = True

        return _Reader

    def test_no_devices_prints_message_and_returns_empty_list(self):
        commands = self._make_commands()

        out = io.StringIO()
        with redirect_stdout(out):
            result = commands._open_listeners([])

        self.assertEqual(result, [])
        self.assertIn("No GPUs on machine", out.getvalue())

    def test_group_permissions_checked_exactly_once(self):
        commands = self._make_commands()
        commands.group_check_printed = False
        self.interface.AmdSmiEventReader = self._make_reader_class()

        commands._open_listeners(["gpu0", "gpu1"])

        self.assertEqual(commands.helpers.check_required_groups_calls, 1)
        self.assertTrue(commands.group_check_printed)

    def test_all_devices_open_successfully(self):
        commands = self._make_commands()
        self.interface.AmdSmiEventReader = self._make_reader_class()

        listeners = commands._open_listeners(["gpu0", "gpu1"])

        self.assertEqual([listener.device for listener in listeners], ["gpu0", "gpu1"])

    def test_single_device_failure_raises_original_exception(self):
        commands = self._make_commands()
        self.interface.AmdSmiEventReader = self._make_reader_class(failing=("gpu0",))

        with self.assertRaises(self.exception_module.AmdSmiLibraryException) as ctx:
            commands._open_listeners(["gpu0"])

        self.assertEqual(str(ctx.exception), "boom-gpu0")

    def test_failure_does_not_abort_remaining_devices(self):
        # A device that fails to open must not prevent the remaining devices
        # from being tried.
        commands = self._make_commands()
        self.interface.AmdSmiEventReader = self._make_reader_class(failing=("bad",))

        out = io.StringIO()
        with redirect_stdout(out):
            listeners = commands._open_listeners(["bad", "good"])

        self.assertEqual([listener.device for listener in listeners], ["good"])
        self.assertEqual(out.getvalue().count("boom-bad"), 1)

    def test_each_failed_device_is_reported_exactly_once(self):
        commands = self._make_commands()
        self.interface.AmdSmiEventReader = self._make_reader_class(failing=("bad",))

        out = io.StringIO()
        with redirect_stdout(out):
            listeners = commands._open_listeners(["good1", "bad", "good2"])

        self.assertEqual([listener.device for listener in listeners], ["good1", "good2"])
        self.assertEqual(out.getvalue().count("boom-bad"), 1)

    def test_unexpected_exception_stops_already_opened_listeners(self):
        commands = self._make_commands()
        opened = []

        class _Reader:
            def __init__(self, device, event_types):
                if device == "boom":
                    raise RuntimeError("unexpected")
                self.device = device
                self.stopped = False
                opened.append(self)

            def stop(self):
                self.stopped = True

        self.interface.AmdSmiEventReader = _Reader

        with self.assertRaises(RuntimeError):
            commands._open_listeners(["good", "boom"])

        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0].stopped)


class TestEventThreadMessageParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.exception_module = modules["amdsmi.amdsmi_exception"]
        cls.event_module = _load_event_module()

    def _make_commands(self):
        commands = object.__new__(self.event_module.EventCommands)
        commands.stop = False
        commands.logger = _FakeLogger()
        return commands

    def test_message_parsed_into_dict_and_logged(self):
        commands = self._make_commands()
        batch = [
            {
                "timestamp": 1234,
                "processor_handle": "handle-0",
                "event": "VMFAULT",
                "message": "process_id: 1234  client_id: UTCL2  ",
            }
        ]

        class _Listener:
            def read(self, timeout):
                commands.stop = True  # one batch is enough; unblock the loop
                return batch

            def stop(self):
                pass

        commands._event_thread(commands, _Listener())

        self.assertEqual(len(commands.logger.calls), 1)
        processor_handle, values = commands.logger.calls[0]
        self.assertEqual(processor_handle, "handle-0")
        self.assertEqual(values["timestamp"], 1234)
        self.assertEqual(values["event"], "VMFAULT")
        self.assertEqual(values["message"], {"process_id": "1234", "client_id": "UTCL2"})
        self.assertEqual(commands.logger.print_calls, 1)

    def test_no_data_status_is_silently_ignored(self):
        commands = self._make_commands()
        exc = self.exception_module.AmdSmiLibraryException(_NO_DATA)

        class _Listener:
            def read(self, timeout):
                commands.stop = True
                raise exc

            def stop(self):
                pass

        out = io.StringIO()
        with redirect_stdout(out):
            commands._event_thread(commands, _Listener())

        self.assertEqual(out.getvalue(), "")
        self.assertEqual(commands.logger.calls, [])

    def test_other_library_exception_is_printed_to_stderr(self):
        commands = self._make_commands()
        exc = self.exception_module.AmdSmiLibraryException(1, message="mock boom")

        class _Listener:
            def read(self, timeout):
                commands.stop = True
                raise exc

            def stop(self):
                pass

        err = io.StringIO()
        with redirect_stderr(err):
            commands._event_thread(commands, _Listener())

        self.assertIn("mock boom", err.getvalue())

    def test_malformed_event_is_isolated_and_remaining_events_still_processed(self):
        commands = self._make_commands()
        batch = [
            {"timestamp": 1, "processor_handle": "h0", "event": "BAD", "message": None},
            {"timestamp": 2, "processor_handle": "h1", "event": "OK", "message": "pid: 1  "},
        ]

        class _Listener:
            def read(self, timeout):
                commands.stop = True
                return batch

            def stop(self):
                pass

        err = io.StringIO()
        with redirect_stderr(err):
            commands._event_thread(commands, _Listener())

        self.assertEqual(len(commands.logger.calls), 1)
        self.assertEqual(commands.logger.calls[0][0], "h1")
        self.assertNotEqual(err.getvalue(), "")

    def test_listener_stop_called_on_normal_completion(self):
        commands = self._make_commands()
        stopped = []

        class _Listener:
            def read(self, timeout):
                commands.stop = True
                return []

            def stop(self):
                stopped.append(True)

        commands._event_thread(commands, _Listener())

        self.assertEqual(stopped, [True])

    def test_listener_stop_called_and_exception_propagates_on_unexpected_error(self):
        commands = self._make_commands()
        stopped = []

        class _Listener:
            def read(self, timeout):
                raise RuntimeError("driver crashed")

            def stop(self):
                stopped.append(True)

        with self.assertRaises(RuntimeError):
            commands._event_thread(commands, _Listener())

        self.assertEqual(stopped, [True])


class TestEventSigtermHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.event_module = _load_event_module()

    def test_sets_stop_and_raises_system_exit(self):
        commands = object.__new__(self.event_module.EventCommands)
        commands.stop = False

        with self.assertRaises(SystemExit) as ctx:
            commands._event_sigterm_handler(signal.SIGTERM, None)

        self.assertTrue(commands.stop)
        self.assertEqual(ctx.exception.code, 128 + signal.SIGTERM)


class TestEventOrchestration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.exception_module = modules["amdsmi.amdsmi_exception"]
        cls.event_module = _load_event_module()

    def _make_commands(self, listeners=()):
        commands = object.__new__(self.event_module.EventCommands)
        commands.device_handles = []
        commands._open_listeners = mock.Mock(return_value=list(listeners))
        commands._event_thread = mock.Mock()
        return commands

    def _run_synchronously(self, commands, args):
        # Thread.start() never actually starts an OS thread here, so Thread.join()
        # would otherwise raise "cannot join thread before it is started".
        with mock.patch.object(
            threading.Thread, "start", lambda self: self.run()
        ), mock.patch.object(threading.Thread, "join", lambda self, timeout=None: None):
            commands.event(args)

    def test_quit_message_stops_cleanly(self):
        listener = mock.Mock()
        commands = self._make_commands(listeners=[listener])
        previous_handler = signal.getsignal(signal.SIGTERM)

        with mock.patch("builtins.input", return_value="q"):
            self._run_synchronously(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        listener.stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_eof_stops_cleanly(self):
        listener = mock.Mock()
        commands = self._make_commands(listeners=[listener])
        previous_handler = signal.getsignal(signal.SIGTERM)

        with mock.patch("builtins.input", side_effect=EOFError):
            self._run_synchronously(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        listener.stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_keyboard_interrupt_stops_cleanly(self):
        listener = mock.Mock()
        commands = self._make_commands(listeners=[listener])
        previous_handler = signal.getsignal(signal.SIGTERM)

        with mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            self._run_synchronously(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        listener.stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_sigterm_triggered_system_exit_is_reraised_after_cleanup(self):
        # Mirrors what input() sees once _event_sigterm_handler fires mid-wait.
        listener = mock.Mock()
        commands = self._make_commands(listeners=[listener])
        previous_handler = signal.getsignal(signal.SIGTERM)

        with mock.patch("builtins.input", side_effect=SystemExit(143)):
            with self.assertRaises(SystemExit) as ctx:
                self._run_synchronously(commands, _build_event_args([0]))

        self.assertEqual(ctx.exception.code, 143)
        self.assertTrue(commands.stop)
        listener.stop.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_one_thread_started_per_listener(self):
        listeners = [mock.Mock(), mock.Mock()]
        commands = self._make_commands(listeners=listeners)

        with mock.patch("builtins.input", return_value="q"):
            self._run_synchronously(commands, _build_event_args([0, 1]))

        self.assertEqual(commands._event_thread.call_count, 2)
        for listener in listeners:
            listener.stop.assert_called_once()

    def test_gpu_arg_defaults_to_device_handles_when_none(self):
        commands = self._make_commands()
        commands.device_handles = ["dev-a", "dev-b"]

        with mock.patch("builtins.input", return_value="q"):
            self._run_synchronously(commands, _build_event_args(None))

        commands._open_listeners.assert_called_once_with(["dev-a", "dev-b"])

    def test_single_gpu_arg_is_wrapped_in_list(self):
        commands = self._make_commands()

        with mock.patch("builtins.input", return_value="q"):
            self._run_synchronously(commands, _build_event_args("dev-a"))

        commands._open_listeners.assert_called_once_with(["dev-a"])


class _EventTimeoutError(Exception):
    """Raised by the SIGALRM handler when event() outlives the test's timeout."""


def _raise_event_timeout(signum, frame):
    raise _EventTimeoutError("event() did not return before the timeout; threads are hung")


class TestEventRealThreads(unittest.TestCase):
    """Runs event() with real, unpatched Thread.start/join.

    The tests above patch Thread.start/join to run inline, which proves the
    message-passing logic but can't catch an actual hang: a worker thread that
    never observes ``stop`` would just make those tests block forever too.
    These run real OS threads and bound the wait with a SIGALRM (event()
    itself must stay on the main thread -- it calls signal.signal(SIGTERM,
    ...), which only works there), so a reintroduced hang fails the assertion
    instead of hanging the suite.
    """

    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.event_module = _load_event_module()

    def _make_commands(self, listeners):
        commands = object.__new__(self.event_module.EventCommands)
        commands.device_handles = []
        commands._open_listeners = mock.Mock(return_value=listeners)
        return commands

    def _run_with_timeout(self, commands, args, timeout_seconds=5):
        """Run commands.event(args) on the main thread, bounded by an alarm."""
        previous_handler = signal.signal(signal.SIGALRM, _raise_event_timeout)
        signal.alarm(timeout_seconds)
        try:
            commands.event(args)
        except _EventTimeoutError:
            self.fail("event() did not return within the timeout; threads are hung")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    def test_real_threads_quit_without_hanging(self):
        listener = mock.Mock()
        commands = self._make_commands([listener])

        # Mirrors _event_thread's "while not self.stop" shape on a real thread,
        # without touching AmdSmiEventReader.
        def _worker(cmds, _listener):
            while not cmds.stop:
                time.sleep(0.01)

        commands._event_thread = _worker

        with mock.patch("builtins.input", return_value="q"):
            self._run_with_timeout(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        listener.stop.assert_called_once()
