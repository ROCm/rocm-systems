#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Mock-based unit tests for the ``amd-smi event`` threading model.

These tests drive ``EventCommands`` with ``amdsmi_interface``/``amdsmi_exception``
fully stubbed and ``threading.Thread.start``/``join`` short-circuited to run
synchronously, so they run without GPU hardware or the compiled ``amdsmi``
package and without real OS threads. They lock in the behaviors introduced by
the WSL event-thread-hang fix:

* ``_read_stdin`` never blocks while ``stop`` is already set, and reads stdin
  non-blockingly (``os.set_blocking`` + ``readline()``) instead of gating a
  blocking ``input()`` behind ``select()`` -- which could strand an
  already-buffered line (multiple lines delivered in one pipe write) behind a
  ``select()`` call that only sees the OS-level fd, not the decoder's internal
  buffer, and which raised an uncaught ``TypeError`` when stdin was closed
  (``sys.stdin is None``).
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
import time
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


class _FakeNonBlockingStdin:
    """Stands in for ``sys.stdin`` under ``os.set_blocking(fd, False)``.

    ``readline()`` returns queued lines immediately (as a real non-blocking fd
    would once data is buffered) and raises ``BlockingIOError`` once the queue
    is empty, instead of blocking. Thread-safe so it can back a real reader
    thread in the concurrency tests.
    """

    def __init__(self, lines=()):
        self._lines = list(lines)
        self._lock = threading.Lock()

    def fileno(self):
        return 0

    def readline(self):
        with self._lock:
            if self._lines:
                return self._lines.pop(0)
        raise BlockingIOError()


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

    def _patch_set_blocking(self):
        # The real os.set_blocking(0, False) would flip the test runner's own
        # stdin to non-blocking; stub it out everywhere except the dedicated
        # os.set_blocking-failure test below.
        return mock.patch.object(self.event_module.os, "set_blocking")

    def test_returns_immediately_when_stop_already_set(self):
        # Regression guard: the WSL hang was _read_stdin blocking forever on
        # input(), never observing that stop had already flipped True.
        commands = self._make_commands(stop=True)
        result_queue = queue.Queue()
        fake_stdin = _FakeNonBlockingStdin()

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        self.assertEqual(fake_stdin._lines, [])
        self.assertTrue(result_queue.empty())

    def test_reads_quit_without_blocking(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = _FakeNonBlockingStdin(["q\n"])

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "quit")

    def test_blocking_io_error_retries_until_data_ready(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        calls = {"n": 0}

        def _readline_then_blocking():
            # First poll finds nothing buffered yet; the retry sleep must not
            # be a real 0.1s wait in a test (sleep is mocked below).
            calls["n"] += 1
            if calls["n"] == 1:
                raise BlockingIOError()
            return "q\n"

        with self._patch_set_blocking(), mock.patch.object(
            sys, "stdin", mock.Mock(fileno=lambda: 0, readline=_readline_then_blocking)
        ), mock.patch.object(self.event_module.time, "sleep") as mock_sleep:
            commands._read_stdin(result_queue)

        mock_sleep.assert_called_once()
        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "quit")

    def test_eof_when_readline_returns_empty_string(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = _FakeNonBlockingStdin([""])

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "eof")

    def test_keyboard_interrupt_is_queued(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = mock.Mock(fileno=lambda: 0, readline=mock.Mock(side_effect=KeyboardInterrupt))

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "interrupt")

    def test_closed_stdin_is_queued_as_eof(self):
        # Regression guard: sys.stdin is None when fd 0 is closed outright
        # (e.g. `0<&-`); the old select()-based version raised an uncaught
        # TypeError here and the reader thread died silently.
        commands = self._make_commands()
        result_queue = queue.Queue()

        with mock.patch.object(sys, "stdin", None):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "eof")

    def test_invalid_fd_from_set_blocking_is_queued_as_eof(self):
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = mock.Mock(fileno=lambda: 0)

        with mock.patch.object(
            self.event_module.os, "set_blocking", side_effect=OSError("bad fd")
        ), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "eof")
        fake_stdin.readline.assert_not_called()

    def test_multiple_buffered_lines_are_each_processed_without_stranding(self):
        # Regression guard: select() only confirms the OS-level fd is
        # readable, not that a full line is buffered. Piping "x\nq\n" in one
        # write let input() consume "x" while "q" sat stranded behind a
        # select() that (correctly) saw nothing left at the OS level. Plain
        # readline() calls, with no select() in between, can't strand a line
        # that's already sitting in the decoder's own buffer.
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = _FakeNonBlockingStdin(["x\n", "q\n"])

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, _payload = result_queue.get_nowait()
        self.assertEqual(kind, "quit")
        self.assertEqual(fake_stdin._lines, [])

    def test_unexpected_exception_is_queued_not_raised(self):
        # Safety net: any exception type we didn't anticipate still unblocks
        # event()'s result_queue.get() instead of killing the thread silently.
        commands = self._make_commands()
        result_queue = queue.Queue()
        fake_stdin = mock.Mock(
            fileno=lambda: 0, readline=mock.Mock(side_effect=RuntimeError("boom"))
        )

        with self._patch_set_blocking(), mock.patch.object(sys, "stdin", fake_stdin):
            commands._read_stdin(result_queue)

        kind, payload = result_queue.get_nowait()
        self.assertEqual(kind, "exception")
        self.assertIsInstance(payload, RuntimeError)


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


class _EventTimeoutError(Exception):
    """Raised by the SIGALRM handler when event() outlives the test's timeout."""


def _raise_event_timeout(signum, frame):
    raise _EventTimeoutError("event() did not return before the timeout; threads are hung")


class TestEventRealThreads(unittest.TestCase):
    """Runs event() with real, unpatched Thread.start/join.

    The tests above patch Thread.start/join to run inline, which proves the
    message-passing logic but can't catch an actual hang: a worker or
    stdin-reader thread that never observes ``stop`` would just make those
    tests block forever too. These run real OS threads and bound the wait with
    a SIGALRM (event() itself must stay on the main thread -- it calls
    signal.signal(SIGTERM, ...), which only works there), so a reintroduced
    hang fails the assertion instead of hanging the suite.
    """

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
        commands = self._make_commands()

        # Mirrors _event_thread's "while not self.stop" shape on a real thread,
        # without touching AmdSmiEventReader.
        def _worker(cmds, _i):
            while not cmds.stop:
                time.sleep(0.01)

        commands._event_thread = _worker

        # A couple of BlockingIOError polls before "q" lands, same as a real
        # non-blocking fd with no data ready yet.
        fake_stdin = _FakeNonBlockingStdin([None, None, "q\n"])

        def _readline():
            line = fake_stdin._lines.pop(0)
            if line is None:
                raise BlockingIOError()
            return line

        with mock.patch.object(self.event_module.os, "set_blocking"), mock.patch.object(
            sys, "stdin", mock.Mock(fileno=lambda: 0, readline=_readline)
        ):
            self._run_with_timeout(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)

    def test_real_threads_worker_exception_unblocks_without_waiting_on_stdin(self):
        commands = self._make_commands()

        def _worker(cmds, _i):
            time.sleep(0.02)
            raise RuntimeError("worker boom")

        commands._event_thread = _worker

        # The reader thread keeps polling and never sees "q"; only the worker's
        # exception should unblock event(), so it must not wait on stdin.
        fake_readline = mock.Mock(side_effect=BlockingIOError)
        with mock.patch.object(self.event_module.os, "set_blocking"), mock.patch.object(
            sys, "stdin", mock.Mock(fileno=lambda: 0, readline=fake_readline)
        ):
            with self.assertRaises(RuntimeError):
                self._run_with_timeout(commands, _build_event_args([0]))

        self.assertTrue(commands.stop)
