"""Lifecycle management for a rocm-sdxi server subprocess.

Starts the binary, waits for its socket, and tears it down on exit.
An optional wrapper (e.g. valgrind) may be injected via
SDXI_TEST_SERVER_WRAPPER.

Copyright (C) Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
"""

import os
import shlex
import signal
import subprocess
import tempfile
import time

from sdxi.vfu import VfioUserClient, wait_for_socket


class ServerHung(Exception):
    """The server did not exit within the allowed timeout."""


class ServerFailed(Exception):
    """The server exited with a non-zero status."""

    def __init__(self, returncode, stderr=""):
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"server exited {returncode}: {stderr[:200]}")


class Server:
    """A rocm-sdxi subprocess with a known socket path."""

    _STOP_TIMEOUT = 5.0
    _SOCKET_TIMEOUT = 10.0

    def __init__(self, binary, socket_path=None):
        self.binary = binary
        if socket_path is None:
            self._tmpdir = tempfile.mkdtemp(prefix="sdxi-test-")
            self.socket_path = os.path.join(self._tmpdir, "sdxi.sock")
        else:
            self._tmpdir = None
            self.socket_path = socket_path
        self._proc = None
        self._stderr_buf = None

    def start(self):
        """Launch the server and wait until its socket is ready."""
        wrapper = os.environ.get("SDXI_TEST_SERVER_WRAPPER", "")
        prefix = shlex.split(wrapper) if wrapper.strip() else []
        cmd = prefix + [self.binary, "--socket", self.socket_path]
        self._stderr_buf = subprocess.PIPE
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if not wait_for_socket(self.socket_path, self._SOCKET_TIMEOUT, self._proc):
            stderr = self._drain_stderr()
            raise ServerFailed(
                self._proc.returncode if self._proc.poll() is not None else -1,
                stderr,
            )
        return self

    def stop(self):
        """Signal the server to stop and wait for it."""
        if self._proc is None:
            return None
        proc = self._proc
        self._proc = None
        if proc.poll() is not None:
            return proc.returncode
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # SIGCONT in case it was paused
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self._STOP_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return proc.returncode
            time.sleep(0.05)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise ServerHung(f"server did not exit within {self._STOP_TIMEOUT}s")

    def _drain_stderr(self):
        if self._proc is None:
            return ""
        try:
            stderr_data, _ = self._proc.communicate(timeout=2.0)
            return stderr_data.decode(errors="replace")
        except Exception:  # pylint: disable=broad-except
            return ""

    def returncode(self):
        if self._proc is None:
            return None
        return self._proc.poll()

    def connect(self, timeout=10.0):
        """Return a connected VfioUserClient context manager."""
        return VfioUserClient(self.socket_path).connect(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()
        if self._tmpdir:
            try:
                os.rmdir(self._tmpdir)
            except OSError:
                pass


def stop_and_check(suite, server):
    """Stop the server and report its exit status to the suite."""
    try:
        rc = server.stop()
        suite.check(
            rc == 0,
            "server exited cleanly (rc=%r)",
            rc,
        )
    except ServerHung as exc:
        suite.check(False, "server hung: %s", exc)


def run_vfu_suite(suite, server):
    """Call finish() after stopping the server; used as main() body."""
    try:
        yield
    finally:
        stop_and_check(suite, server)
        suite.finish()
