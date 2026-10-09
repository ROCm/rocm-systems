#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Starts a worker process through multiprocessing's forkserver under the profiler.

The forkserver saves every signal disposition with signal.signal() and restores
it in each forked child. A disposition the interpreter did not install itself is
recorded as None, and None cannot be restored -- so a profiler holding a C
handler when the interpreter starts breaks the forkserver, and with it every
worker process launched through it. PyTorch's DataLoader with num_workers > 0 is
the common case, and forkserver is the default start method on Linux from Python
3.14.

A single Process start mirrors how DataLoader launches a worker: the socket to
the dead forkserver breaks on the first write, so the failure arrives promptly
rather than through a pool's respawn loop.

The dispositions are reported first. Under a profiler that installs too early
they read None, which names the mechanism before the start is attempted.

No device work -- the processes under test are the forkserver and its children,
which never touch a GPU.
"""

import multiprocessing
import signal
import sys

WATCHED = ("SIGINT", "SIGQUIT", "SIGABRT", "SIGTERM")
JOIN_TIMEOUT_SEC = 30
SENTINEL = "worker-ran"


def worker_body(queue):
    queue.put(SENTINEL)


def main():
    unrestorable = []
    for name in WATCHED:
        disposition = signal.getsignal(getattr(signal, name))
        print(f"{name} disposition: {disposition!r}")
        if disposition is None:
            unrestorable.append(name)

    # The forkserver round-trips SIGINT; the rest are reported for diagnosis only.
    if "SIGINT" in unrestorable:
        print("SIGINT disposition is None, which the forkserver cannot restore")

    context = multiprocessing.get_context("forkserver")
    queue = context.Queue()
    worker = context.Process(target=worker_body, args=(queue,))
    worker.start()
    worker.join(JOIN_TIMEOUT_SEC)

    if worker.is_alive():
        worker.terminate()
        print(f"forkserver worker still alive after {JOIN_TIMEOUT_SEC}s")
        return 1
    if worker.exitcode != 0:
        print(f"forkserver worker exited {worker.exitcode}")
        return 1
    if queue.get(timeout=JOIN_TIMEOUT_SEC) != SENTINEL:
        print("forkserver worker produced no result")
        return 1

    print("forkserver worker completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - any failure here is a test failure
        print(f"forkserver worker failed: {type(exc).__name__}: {exc}")
        sys.exit(1)
