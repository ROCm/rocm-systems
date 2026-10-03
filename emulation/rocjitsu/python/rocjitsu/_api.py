# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Callable worker lifetimes and startup-only process activation."""

import atexit
import codecs
import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from ._launch import (
    Config,
    INVOCATION_DIR_ENV,
    PROGRAMMATIC_ENV,
    budget_option,
    environment,
    launcher,
    launch_options,
    positive_timeout,
)
from ._protocol import PROTOCOL_VERSION, dumps, receive, send


class WorkerError(RuntimeError):
    """A failed worker, with its launch command, exit status and diagnostic tails."""

    def __init__(self, message, *, command=(), returncode=None, stdout="", stderr=""):
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = f"\nWorker stderr (tail):\n{stderr}" if stderr else ""
        super().__init__(message + detail)


class RemoteError(RuntimeError):
    """A callable raised in the worker; includes its original traceback."""

    def __init__(self, kind, message, remote_traceback):
        self.kind = kind
        self.message = message
        self.remote_traceback = remote_traceback
        super().__init__(f"{kind}: {message}\n\nWorker traceback:\n{remote_traceback}")


class _Output:
    """Drain independently of the RPC channel, even when native code is verbose."""

    def __init__(self, pipe, target, limit):
        self.tail = b""
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._read, args=(pipe, target, limit), daemon=True
        )
        self._thread.start()

    def _read(self, pipe, target, limit):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            with pipe:
                while chunk := pipe.read1(8192):
                    with self._lock:
                        self.tail = (self.tail + chunk)[-limit:]
                    if target is not None:
                        try:
                            target.write(decoder.decode(chunk))
                            target.flush()
                        except Exception:
                            # Closed notebook streams must not block the worker.
                            target = None
                if target is not None:
                    try:
                        target.write(decoder.decode(b"", final=True))
                        target.flush()
                    except Exception:
                        pass
        except OSError:
            pass

    def text(self):
        with self._lock:
            return self.tail.decode("utf-8", errors="replace")

    def join(self):
        self._thread.join(timeout=1)


class Session:
    """A persistent Python worker under rocjitsu, with serialized callable execution.

    config is a JSON path or mapping. mode selects local, daemon, or attach.
    cpu_thread_budget forwards the corresponding native CLI option. env and cwd
    affect only the worker; config paths are resolved in the caller. Output is
    forwarded to the caller's Python streams (including notebook streams), unless
    capture_output=True. The last log_limit bytes per stream remain available.
    """

    def __init__(
        self,
        config,
        *,
        executable=None,
        python=None,
        env=None,
        cwd=None,
        mode="local",
        cpu_thread_budget=None,
        startup_timeout=30,
        capture_output=False,
        log_limit=65536,
    ):
        positive_timeout(startup_timeout, "startup_timeout")
        options = launch_options(mode, cpu_thread_budget)
        if (
            isinstance(log_limit, bool)
            or not isinstance(log_limit, int)
            or log_limit <= 0
        ):
            raise ValueError("log_limit must be a positive integer")
        self._owner = os.getpid()
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._exit_watch_stop = threading.Event()
        self._exit_watch = None
        self._status_lost = signal.getsignal(signal.SIGCHLD) == signal.SIG_IGN
        self._process = None
        self._channel = None
        self._closed = False
        self._stdout = self._stderr = None
        self.command = ()
        self.pid = None
        self._config = Config(config)
        child = None
        try:
            self._channel, child = socket.socketpair()
            self.command = tuple(
                [
                    launcher(executable),
                    "--config",
                    self._config.path,
                    *options,
                    "--",
                    os.fspath(python or sys.executable),
                    "-u",
                    "-m",
                    "rocjitsu._worker",
                    str(child.fileno()),
                ]
            )
            self._process = subprocess.Popen(
                self.command,
                env=environment(env),
                cwd=cwd,
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            child.close()
            self._stdout = _Output(
                self._process.stdout, None if capture_output else sys.stdout, log_limit
            )
            self._stderr = _Output(
                self._process.stderr, None if capture_output else sys.stderr, log_limit
            )
            self._exit_watch = threading.Thread(target=self._watch_exit, daemon=True)
            self._exit_watch.start()
            deadline = time.monotonic() + startup_timeout
            ready = receive(self._channel, deadline)
            if ready != ("ready", PROTOCOL_VERSION, sys.version_info[:3]):
                raise self._error(
                    f"Worker protocol or Python version mismatch: {ready!r}"
                )
            # The native launcher execs Python, retaining the process group leader.
            self.pid = self._process.pid
            paths = [
                os.path.abspath(path)
                for path in sys.path
                if not any(
                    part in ("site-packages", "dist-packages")
                    for part in Path(path).parts
                )
            ]
            send(self._channel, dumps(paths), deadline)
            if receive(self._channel, deadline) != ("initialized", self.pid):
                raise self._error("Worker initialization handshake failed")
        except BaseException as error:
            if child is not None:
                child.close()
            self._stop()
            if isinstance(error, TimeoutError):
                raise self._error("rocjitsu worker startup timed out") from error
            if isinstance(error, (EOFError, OSError)) and self._process is not None:
                raise self._error("rocjitsu worker failed to start") from error
            raise
        atexit.register(self.close)

    def _watch_exit(self):
        # Forked helpers can inherit the worker's socket. Process death must
        # wake blocked RPC transfers even when those helpers keep the FD open.
        while not self._exit_watch_stop.is_set():
            try:
                exited = os.waitid(
                    os.P_PID, self._process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
            except ChildProcessError:
                # SIGCHLD=SIG_IGN (or an external reaper) removes the waitable
                # child. Its status is unavailable, but the RPC must still wake.
                self._status_lost = True
                exited = True
            if exited is not None:
                try:
                    self._channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return
            self._exit_watch_stop.wait(0.05)

    @property
    def closed(self):
        return self._closed

    @property
    def returncode(self):
        return (
            self._process.returncode
            if self._process is not None and not self._status_lost
            else None
        )

    @property
    def stdout(self):
        self._check_owner()
        return self._stdout.text() if self._stdout else ""

    @property
    def stderr(self):
        self._check_owner()
        return self._stderr.text() if self._stderr else ""

    def _error(self, message):
        return WorkerError(
            f"{message} (status {self.returncode})",
            command=self.command,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )

    def _check_owner(self):
        if os.getpid() != self._owner:
            raise RuntimeError(
                "A Session cannot be used by a forked child; create a new Session"
            )

    def run(self, function, *args, timeout=None, **kwargs):
        """Call function(*args, **kwargs), copying CPU values across the boundary.

        timeout covers request transfer and execution, after serialization and
        lock acquisition. A timeout closes the session. close() from another
        thread can cancel a running call, which then raises WorkerError.
        """
        self._check_owner()
        if not callable(function):
            raise TypeError("function must be callable")
        positive_timeout(timeout, "timeout", optional=True)
        if self._closed:
            raise RuntimeError("Session is closed")
        payload = dumps((function, args, kwargs))
        with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            deadline = None if timeout is None else time.monotonic() + timeout
            try:
                send(self._channel, payload, deadline)
                response = receive(self._channel, deadline)
                if not isinstance(response, tuple) or not response:
                    raise self._error("Invalid worker response")
                if response[0] == "error" and len(response) == 4:
                    raise RemoteError(*response[1:])
                if response[0] != "result" or len(response) != 2:
                    raise self._error("Invalid worker response")
                return response[1]
            except RemoteError:
                raise
            except TimeoutError:
                self._stop()
                raise TimeoutError(
                    "rocjitsu call timed out; session was closed"
                ) from None
            except (EOFError, OSError) as error:
                self._stop()
                raise self._error("rocjitsu worker connection closed") from error
            except BaseException:
                self._stop()
                raise

    def _stop(self):
        with self._stop_lock:
            if self._closed:
                return
            self._closed = True
            self._exit_watch_stop.set()
            if self._exit_watch is not None:
                self._exit_watch.join(timeout=1)
            if self._channel is not None:
                try:
                    self._channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self._channel.close()
            if self._process is not None:
                # Kill the private group before reaping the leader, preventing PID
                # reuse from targeting an unrelated group. Descendants are included.
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # An uninterruptible kernel wait cannot be cancelled from userspace.
                    threading.Thread(target=self._process.wait, daemon=True).start()
            for output in (self._stdout, self._stderr):
                if output is not None:
                    output.join()
            self._config.close()
            atexit.unregister(self.close)

    def close(self):
        """Close the session, cancelling an active call and its subprocess group.

        Idle workers get up to one second to perform ordinary Python teardown.
        Repeated calls are harmless; a forked child never terminates its parent.
        """
        if os.getpid() != self._owner:
            return
        if self._lock.acquire(blocking=False):
            try:
                if not self._closed:
                    try:
                        send(self._channel, dumps(None), time.monotonic() + 1)
                        # Wait for exit without reaping so group ownership is retained.
                        deadline = time.monotonic() + 1
                        while time.monotonic() < deadline:
                            if os.waitid(
                                os.P_PID,
                                self._process.pid,
                                os.WEXITED | os.WNOHANG | os.WNOWAIT,
                            ):
                                break
                            time.sleep(0.01)
                    except (OSError, EOFError):
                        pass
                    finally:
                        self._stop()
            finally:
                self._lock.release()
        else:
            self._stop()

    def __enter__(self):
        self._check_owner()
        if self._closed:
            raise RuntimeError("Session is closed")
        return self

    def __exit__(self, *exc):
        self.close()


def run(
    function,
    *args,
    config,
    executable=None,
    python=None,
    env=None,
    cwd=None,
    mode="local",
    cpu_thread_budget=None,
    startup_timeout=30,
    capture_output=False,
    log_limit=65536,
    timeout=None,
    **kwargs,
):
    """Execute a callable in a fresh Session; launch keywords match Session."""
    with Session(
        config,
        executable=executable,
        python=python,
        env=env,
        cwd=cwd,
        mode=mode,
        cpu_thread_budget=cpu_thread_budget,
        startup_timeout=startup_timeout,
        capture_output=capture_output,
        log_limit=log_limit,
    ) as session:
        return session.run(function, *args, timeout=timeout, **kwargs)


_enabled_config = None
_enabled_bytes = None
_enabled_pid = None
_enable_lock = threading.Lock()


def _after_fork():
    global _enable_lock
    _enable_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def is_enabled():
    """Whether this process owns global activation, including exec/spawn inheritance.

    This does not report configured CLI/Session mode or GPU runtime health.
    A fork that inherited an initialized GPU context returns False.
    """
    try:
        query = ctypes.CDLL(None).rj_interposer_is_enabled_v1
    except AttributeError:
        return False
    query.argtypes = []
    query.restype = ctypes.c_int
    return bool(query())


def enable(config, *, cpu_thread_budget=None):
    """Enable process-wide simulation before GPU discovery or initialization.

    Start Python with ``python -m rocjitsu script.py`` to preload the interposer.
    This function never restarts the interpreter. Activation is one-way, and
    changing configuration requires a new process. Use Session for isolation.
    """
    global _enabled_config, _enabled_bytes, _enabled_pid
    budget_option(cpu_thread_budget)
    # Check before taking a lock that could have been held across fork.
    if _enabled_pid is not None and _enabled_pid != os.getpid():
        if not is_enabled():
            raise RuntimeError(
                "Cannot enable an inherited Python GPU context after fork"
            )
        # An unused native context can be reconstructed by the atfork handler,
        # notably in a forkserver. Config ownership remains with the parent.
        _enabled_pid = os.getpid()
    with _enable_lock:
        candidate = Config(config)
        try:
            contents = Path(candidate.path).read_bytes()
            try:
                contents = json.dumps(json.loads(contents), sort_keys=True).encode()
            except (ValueError, UnicodeError):
                # Native FlatBuffers parsing also accepts relaxed JSON syntax.
                pass
            contents = (contents, cpu_thread_budget)
            if _enabled_config is not None:
                if _enabled_pid != os.getpid():
                    raise RuntimeError(
                        "Cannot enable an inherited Python GPU context after fork"
                    )
                if contents == _enabled_bytes:
                    return
                raise RuntimeError(
                    "Simulation is already enabled with a different configuration"
                )
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                raise RuntimeError(
                    "enable() must precede GPU initialization; use a Session for simulation"
                )
            try:
                activate = ctypes.CDLL(None).rj_interposer_enable_v1
            except AttributeError:
                raise RuntimeError(
                    "Start with 'python -m rocjitsu script.py' before calling enable()"
                ) from None
            activate.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            activate.restype = ctypes.c_int
            handoff = candidate.prepare_directory()
            budget = (
                None
                if cpu_thread_budget is None
                else ctypes.c_uint32(cpu_thread_budget)
            )
            detail = ctypes.create_string_buffer(4096)
            status = activate(
                os.fsencode(candidate.path),
                os.fsencode(handoff),
                None if budget is None else ctypes.byref(budget),
                detail,
                len(detail),
            )
            if status not in (0, errno.EALREADY):
                reason = {
                    errno.EBUSY: "GPU discovery has already started",
                    errno.ENOTSUP: "the interposer must be preloaded with --preload-only",
                    errno.EINVAL: "invalid or unsupported simulation configuration",
                }.get(status, os.strerror(status))
                diagnostic = detail.value.decode(errors="replace") or reason
                raise RuntimeError(f"Cannot enable rocjitsu: {diagnostic}")
            # Fresh exec children inherit the config just as CLI-launched
            # children do. The current interposer retains its activation state.
            if status == 0:
                os.environ[INVOCATION_DIR_ENV] = handoff
                # Distinguish global-activation descendants from configured CLI
                # launches; spawn can repeat an identical top-level enable().
                os.environ[PROGRAMMATIC_ENV] = "2"
            _enabled_config = candidate
            _enabled_bytes = contents
            _enabled_pid = os.getpid()
            atexit.register(candidate.close)
            candidate = None
        finally:
            if candidate is not None:
                candidate.close()
