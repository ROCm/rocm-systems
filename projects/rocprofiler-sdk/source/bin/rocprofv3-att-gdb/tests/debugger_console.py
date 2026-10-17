# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Drive the actual GDB prompt without a third-party PTY dependency."""

import os
import select
import signal
import subprocess
import time


class Console:
    def __init__(self, args, log, env=None):
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self.pending = b""
        self.output = b""
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def send(self, command):
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()

    def expect(self, value):
        expected = value.encode()
        deadline = time.monotonic() + 30
        while expected not in self.pending:
            assert time.monotonic() < deadline, self.output.decode(errors="replace")
            if select.select([self.process.stdout], [], [], 0.1)[0]:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                assert chunk, self.output.decode(errors="replace")
                self.output += chunk
                self.pending += chunk
        offset = self.pending.index(expected) + len(expected)
        self.pending = self.pending[offset:]

    def close(self):
        try:
            if self.process.poll() is None:
                self.send("set confirm off\nquit")
            rest, _ = self.process.communicate(timeout=10)
            self.output += rest
        finally:
            # Also reap debugger/target children if the launcher had to be killed.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
            self.log.write_bytes(self.output)
