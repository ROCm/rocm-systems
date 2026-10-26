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
import sys
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

    def get_error_code(self):
        return self.err_code

    def get_error_info(self, detailed=True):
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

        err = io.StringIO()
        with redirect_stderr(err):
            listeners = commands._open_listeners(["bad", "good"])

        self.assertEqual([listener.device for listener in listeners], ["good"])
        self.assertEqual(err.getvalue().count("boom-bad"), 1)

    def test_each_failed_device_is_reported_exactly_once(self):
        commands = self._make_commands()
        self.interface.AmdSmiEventReader = self._make_reader_class(failing=("bad",))

        err = io.StringIO()
        with redirect_stderr(err):
            listeners = commands._open_listeners(["good1", "bad", "good2"])

        self.assertEqual([listener.device for listener in listeners], ["good1", "good2"])
        self.assertEqual(err.getvalue().count("boom-bad"), 1)

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

    def test_no_listeners_returns_immediately(self):
        commands = self._make_commands()

        commands._event_thread(commands, [])

        self.assertEqual(commands.logger.calls, [])

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

        commands._event_thread(commands, [_Listener()])

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
            commands._event_thread(commands, [_Listener()])

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
            commands._event_thread(commands, [_Listener()])

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
            commands._event_thread(commands, [_Listener()])

        self.assertEqual(len(commands.logger.calls), 1)
        self.assertEqual(commands.logger.calls[0][0], "h1")
        self.assertNotEqual(err.getvalue(), "")

    def test_does_not_stop_listener_itself(self):
        # event() owns listener.stop() centrally (once, in its own finally
        # block) now that a single thread reads through listeners[0];
        # _event_thread must not also stop it.
        commands = self._make_commands()
        stopped = []

        class _Listener:
            def read(self, timeout):
                commands.stop = True
                return []

            def stop(self):
                stopped.append(True)

        commands._event_thread(commands, [_Listener()])

        self.assertEqual(stopped, [])

    def test_unexpected_exception_propagates_without_stopping_listener(self):
        commands = self._make_commands()
        stopped = []

        class _Listener:
            def read(self, timeout):
                raise RuntimeError("driver crashed")

            def stop(self):
                stopped.append(True)

        with self.assertRaises(RuntimeError):
            commands._event_thread(commands, [_Listener()])

        self.assertEqual(stopped, [])


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

    def test_every_gpu_fails_to_open_raises_before_banner_or_thread(self):
        # if event() swallows _open_listeners()'s failure and presses on
        # with an empty/partial listener set, the CLI would sit at
        # "Press q and hit ENTER" forever with no indication anything failed
        commands = self._make_commands()
        exc = self.exception_module.AmdSmiLibraryException(7, message="all gpus failed")
        commands._open_listeners = mock.Mock(side_effect=exc)

        out = io.StringIO()
        with mock.patch.object(self.event_module.threading, "Thread") as thread_cls:
            with redirect_stdout(out):
                with self.assertRaises(self.exception_module.AmdSmiLibraryException) as ctx:
                    commands.event(_build_event_args([0]))

        self.assertIs(ctx.exception, exc)
        self.assertEqual(ctx.exception.err_code, 7)
        self.assertNotIn("EVENT LISTENING", out.getvalue())
        commands._event_thread.assert_not_called()
        thread_cls.assert_not_called()

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

    def test_one_thread_started_regardless_of_listener_count(self):
        # event() reads through a single thread (listeners[0].read() is a
        # global poll that returns every device's events), so N listeners
        # still start exactly one thread, given the whole list.
        listeners = [mock.Mock(), mock.Mock()]
        commands = self._make_commands(listeners=listeners)

        with mock.patch("builtins.input", return_value="q"):
            self._run_synchronously(commands, _build_event_args([0, 1]))

        commands._event_thread.assert_called_once_with(commands, listeners)
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

    * ``test_real_threads_quit_without_hanging`` and
      ``test_real_stdin_pipe_returns_only_after_quit`` both mock ``_open_listeners`` to return a canned
      listener list, and replace ``_event_thread`` with their own worker (a
      ``while not self.stop`` spin, or a no-op) to prove the real thread
      lifecycle and stdin handling in ``event()`` itself, not ``_open_listeners``
      or ``_event_thread``'s own logic -- so neither would catch a regression
      where ``event()`` swallows an ``_open_listeners()`` failure and presses on
      with no listeners (covered instead by
      ``TestEventOrchestration.test_every_gpu_fails_to_open_raises_before_banner_or_thread``,
      which mocks ``_open_listeners`` to raise).
    * ``test_one_gpu_fails_other_keeps_running_until_quit`` wires up the genuine
      ``_open_listeners``/``_event_thread`` against a fake ``AmdSmiEventReader``
      -- to exercise those components directly.
    """

    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        if not hasattr(signal, "SIGALRM"):
            raise unittest.SkipTest("SIGALRM is POSIX-only; unavailable on this platform")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.exception_module = modules["amdsmi.amdsmi_exception"]
        cls.event_module = _load_event_module()

    def _make_commands(self, listeners):
        commands = object.__new__(self.event_module.EventCommands)
        commands.device_handles = []
        commands._open_listeners = mock.Mock(return_value=listeners)
        return commands

    def _make_real_commands(self):
        """Commands with the real _open_listeners/_event_thread wired up, not mocked."""
        commands = object.__new__(self.event_module.EventCommands)
        commands.device_handles = []
        commands.group_check_printed = True
        commands.helpers = _FakeHelpers()
        commands.logger = _FakeLogger()
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

    def test_one_gpu_fails_other_keeps_running_until_quit(self):
        # One GPU failing to open must not stop the other from reporting
        # events, and its listener is stopped exactly once after the user quits.
        exception_cls = self.exception_module.AmdSmiLibraryException
        stop_calls = []

        class _Reader:
            def __init__(self, device, event_types):
                self.device = device
                if device == "bad":
                    raise exception_cls(1, "boom-bad")

            def read(self, timeout):
                time.sleep(0.01)
                return [
                    {
                        "timestamp": 1,
                        "processor_handle": self.device,
                        "event": "PROCESS_START",
                        "message": "pid: 1  ",
                    }
                ]

            def stop(self):
                stop_calls.append(self.device)

        self.interface.AmdSmiEventReader = _Reader
        commands = self._make_real_commands()

        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch(
            "builtins.input", return_value="q"
        ):
            self._run_with_timeout(commands, _build_event_args(["bad", "good"]))

        self.assertIn("boom-bad", err.getvalue())
        self.assertNotIn("boom-bad", out.getvalue())
        self.assertTrue(commands.logger.calls)
        self.assertTrue(all(handle == "good" for handle, _ in commands.logger.calls))
        self.assertEqual(stop_calls, ["good"])

    def test_real_stdin_pipe_returns_only_after_quit(self):
        # A real OS pipe as stdin: a non-"q" line must not be mistaken for
        # quit -- event() has to keep looping until "q" actually arrives.
        commands = self._make_commands([])
        commands._event_thread = lambda *_a, **_kw: None

        read_fd, write_fd = os.pipe()
        real_stdin = os.fdopen(read_fd)

        def _feed():
            os.write(write_fd, b"x\n")
            time.sleep(0.05)
            os.write(write_fd, b"q\n")
            os.close(write_fd)

        feeder = threading.Thread(target=_feed)
        feeder.start()
        start = time.monotonic()
        try:
            with mock.patch.object(sys, "stdin", real_stdin):
                self._run_with_timeout(commands, _build_event_args([0]))
        finally:
            feeder.join()
            real_stdin.close()
        elapsed = time.monotonic() - start

        self.assertTrue(commands.stop)
        self.assertGreaterEqual(elapsed, 0.05)


class TestEventOpenListenersRealRecordOrRaise(unittest.TestCase):
    """Exercises _open_listeners()/event() against the real
    ``amdsmi_helpers.AMDSMIHelpers.record_or_raise()``, so NO_PERM ->
    ``PermissionError`` is proven to be the production translation, not a test
    double's approximation of it (``_FakeHelpers.record_or_raise`` elsewhere in
    this file never raises).
    """

    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        if not _CLI_DIR:
            raise unittest.SkipTest("amd-smi CLI source not found")

        modules = _build_fake_amdsmi()
        interface = modules["amdsmi.amdsmi_interface"]
        wrapper = interface.amdsmi_wrapper
        # record_or_raise()'s own NO_PERM check, plus the constants
        # amdsmi_helpers.py's CPER_DECODE_MESSAGES class-body dict reads at
        # import time (unrelated to this test, but evaluated unconditionally).
        wrapper.AMDSMI_STATUS_NO_PERM = 10
        wrapper.AMDSMI_STATUS_NOT_SUPPORTED = 2
        wrapper.AMDSMI_STATUS_INVAL = 5
        wrapper.AMDSMI_STATUS_UNEXPECTED_SIZE = 100
        wrapper.AMDSMI_STATUS_UNEXPECTED_DATA = 101
        interface.AmdSmiInitFlags = types.SimpleNamespace(
            INIT_ALL_PROCESSORS=0xFFFFFFFF, INIT_AMD_GPUS=1, INIT_AMD_CPUS=2, INIT_AMD_NICS=4
        )
        interface.amdsmi_init = lambda _flag: None
        interface.amdsmi_shut_down = lambda: None
        interface.AmdSmiLibraryException = modules["amdsmi.amdsmi_exception"].AmdSmiLibraryException
        interface.AmdSmiParameterException = type("AmdSmiParameterException", (Exception,), {})

        # amdsmi_helpers.py does `from amdsmi_init import *`; evict any stale
        # fake a sibling test file left behind so this loads the real sources.
        modules.update(
            dict.fromkeys(("amdsmi_init", "amdsmi_helpers", "amdsmi_cli_exceptions", "BDF"))
        )
        stub_modules(cls, modules)
        if _CLI_DIR not in sys.path:
            sys.path.insert(0, _CLI_DIR)

        cls.interface = interface
        cls.exception_module = modules["amdsmi.amdsmi_exception"]
        import amdsmi_helpers as real_amdsmi_helpers

        cls.event_module = _load_event_module()
        # Only record_or_raise() itself needs to be the genuine method; the
        # NO_PERM branch never touches AMDSMIHelpers.__init__'s other state
        # (error_collector etc.), so a minimal stand-in is enough to host it.
        cls.helpers_cls = type(
            "_RealRecordOrRaiseHelpers",
            (),
            {
                "record_or_raise": real_amdsmi_helpers.AMDSMIHelpers.record_or_raise,
                "check_required_groups": lambda self: None,
                "get_gpu_id_from_device_handle": lambda self, device_handle: device_handle,
            },
        )

    def test_second_gpu_no_perm_raises_permission_error_via_real_helpers(self):
        exception_cls = self.exception_module.AmdSmiLibraryException
        no_perm = self.interface.amdsmi_wrapper.AMDSMI_STATUS_NO_PERM
        stop_calls = []

        class _Reader:
            def __init__(self, device, event_types):
                self.device = device
                if device == "bad":
                    raise exception_cls(no_perm, "needs elevation")

            def stop(self):
                stop_calls.append(self.device)

        self.interface.AmdSmiEventReader = _Reader
        commands = object.__new__(self.event_module.EventCommands)
        commands.device_handles = []
        commands.group_check_printed = True
        commands.helpers = self.helpers_cls()

        with mock.patch.object(self.event_module.threading, "Thread") as thread_cls:
            with self.assertRaises(PermissionError):
                commands.event(_build_event_args(["good", "bad"]))

        self.assertEqual(stop_calls, ["good"])
        thread_cls.assert_not_called()
