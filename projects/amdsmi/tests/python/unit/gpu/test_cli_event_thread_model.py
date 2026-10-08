#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``amd-smi event`` threading model.

These tests drive ``EventCommands`` with ``amdsmi_interface``/``amdsmi_exception``
fully stubbed and ``threading.Thread.start``/``join`` short-circuited to run
synchronously, so they run without GPU hardware or the compiled ``amdsmi``
package and without real OS threads. They lock in the behaviors introduced by
the WSL event-thread-hang fix:

* ``_read_stdin`` never blocks in ``input()`` while ``stop`` is already set, and
  polls stdin with ``select`` rather than blocking immediately.
* ``EventListenerThread`` reports a worker exception via the result queue
  instead of raising it out of ``join()``.
* ``event()`` re-raises an ``AmdSmiLibraryException`` from a worker, lets any
  other worker exception propagate, and always joins threads and restores the
  SIGTERM handler before returning or raising.
* ``_event_thread`` parses an event's ``message`` string into a dict and logs
  it, and swallows ``AMDSMI_STATUS_NO_DATA`` while printing any other error.
"""

import argparse
import io
import os
import queue
import signal
import sys
import threading
import types
import unittest
from contextlib import redirect_stdout
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

    def store_output(self, processor_handle, key, value):
        self.calls.append((processor_handle, key, dict(value)))

    def print_output(self, *args, **kwargs):
        pass


class _FakeHelpers:
    def check_required_groups(self):
        pass


def _build_event_args(gpu):
    return argparse.Namespace(gpu=gpu)


class TestEventThreadMessageParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.interface = modules["amdsmi.amdsmi_interface"]
        cls.event_module = _load_event_module()

    def _make_commands(self):
        commands = object.__new__(self.event_module.EventCommands)
        commands.stop = False
        commands.device_handles = ["fake-device"]
        commands.group_check_printed = True
        commands.helpers = _FakeHelpers()
        commands.logger = _FakeLogger()
        return commands

    def test_message_parsed_into_dict_and_logged(self):
        commands = self._make_commands()
        batch = [
            {
                "processor_handle": "handle-0",
                "event": "VMFAULT",
                "message": "process_id: 1234  client_id: UTCL2  ",
            }
        ]

        class _Reader:
            def __init__(self, device_handle, event_types):
                pass

            def read(self, timeout, num_elem=10):
                commands.stop = True  # one batch is enough; unblock the loop
                return batch

            def stop(self):
                pass

        self.interface.AmdSmiEventReader = _Reader

        commands._event_thread(commands, 0)

        self.assertEqual(len(commands.logger.calls), 1)
        processor_handle, key, value = commands.logger.calls[0]
        self.assertEqual(processor_handle, "handle-0")
        self.assertEqual(key, "values")
        self.assertEqual(value["event"], "VMFAULT")
        self.assertEqual(value["message"], {"process_id": "1234", "client_id": "UTCL2"})

    def test_no_data_status_is_silently_ignored(self):
        commands = self._make_commands()

        class _Reader:
            def __init__(self, device_handle, event_types):
                pass

            def read(self, timeout, num_elem=10):
                commands.stop = True
                raise self.exception_module.AmdSmiLibraryException(_NO_DATA)

            def stop(self):
                pass

        _Reader.exception_module = self.event_module.amdsmi_exception
        self.interface.AmdSmiEventReader = _Reader

        out = io.StringIO()
        with redirect_stdout(out):
            commands._event_thread(commands, 0)

        self.assertEqual(out.getvalue(), "")

    def test_other_library_exception_is_printed(self):
        commands = self._make_commands()

        class _Reader:
            def __init__(self, device_handle, event_types):
                pass

            def read(self, timeout, num_elem=10):
                commands.stop = True
                raise self.exception_module.AmdSmiLibraryException(1, message="mock boom")

            def stop(self):
                pass

        _Reader.exception_module = self.event_module.amdsmi_exception
        self.interface.AmdSmiEventReader = _Reader

        out = io.StringIO()
        with redirect_stdout(out):
            commands._event_thread(commands, 0)

        self.assertIn("mock boom", out.getvalue())


class TestEventListenerThread(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.event_module = _load_event_module()

    def test_exception_from_target_is_queued_not_raised(self):
        result_queue = queue.Queue()

        def _boom(*_args, **_kwargs):
            raise RuntimeError("thread boom")

        thread = self.event_module.EventCommands.EventListenerThread(
            target=_boom, args=(), result_queue=result_queue
        )
        thread.start()
        thread.join()  # must not raise; the exception travels via the queue

        kind, payload = result_queue.get_nowait()
        self.assertEqual(kind, "exception")
        self.assertIsInstance(payload, RuntimeError)
        self.assertEqual(str(payload), "thread boom")

    def test_normal_completion_leaves_queue_empty(self):
        result_queue = queue.Queue()
        thread = self.event_module.EventCommands.EventListenerThread(
            target=lambda: None, args=(), result_queue=result_queue
        )
        thread.start()
        thread.join()

        self.assertTrue(result_queue.empty())


class TestReadStdin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EVENT_PATH or not os.path.isfile(EVENT_PATH):
            raise unittest.SkipTest(f"amd-smi CLI event.py not found (looked in {_CLI_DIR})")
        modules = _build_fake_amdsmi()
        stub_modules(cls, modules)
        cls.event_module = _load_event_module()

    def _make_commands(self, stop=False):
        commands = object.__new__(self.event_module.EventCommands)
        commands.stop = stop
        return commands

    def test_returns_immediately_when_stop_already_set(self):
        # Regression guard: the WSL hang was _read_stdin blocking in input()
        # forever, never observing that stop had already flipped True.
        commands = self._make_commands(stop=True)
        result_queue = queue.Queue()

        with mock.patch.object(self.event_module.select, "select") as mock_select, mock.patch(
            "builtins.input"
        ) as mock_input:
            commands._read_stdin(result_queue)

        mock_select.assert_not_called()
        mock_input.assert_not_called()
        self.assertTrue(result_queue.empty())

    def test_polls_without_blocking_then_reads_quit(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        select_results = [([], [], []), ([sys.stdin], [], [])]

        with mock.patch.object(
            self.event_module.select, "select", side_effect=select_results
        ) as mock_select, mock.patch("builtins.input", return_value="q") as mock_input:
            commands._read_stdin(result_queue)

        self.assertEqual(mock_select.call_count, 2)
        mock_input.assert_called_once()
        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "quit")

    def test_eof_error_is_queued(self):
        commands = self._make_commands()
        result_queue = queue.Queue()

        with mock.patch.object(
            self.event_module.select, "select", return_value=([sys.stdin], [], [])
        ), mock.patch("builtins.input", side_effect=EOFError):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "eof")

    def test_keyboard_interrupt_is_queued(self):
        commands = self._make_commands()
        result_queue = queue.Queue()

        with mock.patch.object(
            self.event_module.select, "select", return_value=([sys.stdin], [], [])
        ), mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "interrupt")


class TestEventOrchestration(unittest.TestCase):
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
        commands.device_handles = []
        return commands

    def _run_synchronously(self, commands, args):
        # Thread.start() never actually starts an OS thread here, so Thread.join()
        # would otherwise raise "cannot join thread before it is started".
        with mock.patch.object(
            threading.Thread, "start", lambda self: self.run()
        ), mock.patch.object(threading.Thread, "join", lambda self, timeout=None: None):
            commands.event(args)

    def test_quit_message_stops_cleanly(self):
        commands = self._make_commands()
        commands._event_thread = lambda *_a, **_kw: None
        commands._read_stdin = lambda result_queue: result_queue.put(("quit", None))
        previous_handler = signal.getsignal(signal.SIGTERM)

        self._run_synchronously(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_library_exception_from_worker_is_reraised(self):
        commands = self._make_commands()
        exc = self.exception_module.AmdSmiLibraryException(1, message="library boom")
        commands._event_thread = mock.Mock(side_effect=exc)
        commands._read_stdin = lambda result_queue: None  # event thread wins the race
        previous_handler = signal.getsignal(signal.SIGTERM)

        with self.assertRaises(self.exception_module.AmdSmiLibraryException) as ctx:
            self._run_synchronously(commands, _build_event_args([0]))

        self.assertIs(ctx.exception, exc)
        self.assertTrue(commands.stop)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)

    def test_generic_exception_from_worker_propagates_after_cleanup(self):
        commands = self._make_commands()
        commands._event_thread = mock.Mock(side_effect=RuntimeError("worker boom"))
        commands._read_stdin = lambda result_queue: None
        previous_handler = signal.getsignal(signal.SIGTERM)

        with self.assertRaises(RuntimeError):
            self._run_synchronously(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_handler)
