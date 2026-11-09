# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Own a private process group until its caller closes the control socket.

Run as a stdlib-only script under the caller's Python, outside native activation.
Only the worker receives its RPC descriptor; helpers cannot retain our control FD.
"""

import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import traceback


def main():
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Worker supervisor requires its own process group")
    with socket.socket(fileno=int(sys.argv[1])) as control:
        control.set_inheritable(False)
        # The caller may ignore SIGCHLD or reap any of its children. Keep worker
        # ownership here, where it is independent of both caller behaviors.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        worker_fd = int(sys.argv[2])
        worker = None
        worker_done = threading.Event()
        watch_started = False
        try:
            worker = subprocess.Popen(sys.argv[3:], pass_fds=(worker_fd,))
            os.close(worker_fd)
            control.sendall(struct.pack("!i", worker.pid))

            def watch_worker():
                # Leave the child waitable: only main() may reap it after the
                # cleanup request. Its PID remains owned even after worker exit.
                exited = os.waitid(os.P_PID, worker.pid, os.WEXITED | os.WNOWAIT)
                status = (
                    exited.si_status
                    if exited.si_code == os.CLD_EXITED
                    else -exited.si_status
                )
                try:
                    control.sendall(struct.pack("!i", status))
                except OSError:
                    pass
                finally:
                    worker_done.set()

            threading.Thread(target=watch_worker, daemon=True).start()
            watch_started = True
            # Remain the live group leader even after worker exit. Closing the
            # caller endpoint is the only cleanup request; no numeric group ID
            # crosses that boundary, so delayed cleanup cannot hit a reused ID.
            control.recv(1)
        except BaseException:
            traceback.print_exc()
        finally:
            try:
                if worker is not None:
                    # No thread reaps this child and the caller cannot reap its
                    # grandchild, so this numeric PID remains ours to signal.
                    try:
                        os.kill(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    # Report before consuming its status. Then reap the direct
                    # worker before the final group signal includes this anchor.
                    if watch_started:
                        worker_done.wait(timeout=5)
                    try:
                        worker.wait(timeout=0 if watch_started else 5)
                    except subprocess.TimeoutExpired:
                        # An uninterruptible kernel wait cannot be cancelled.
                        pass
            finally:
                os.killpg(0, signal.SIGKILL)


if __name__ == "__main__":
    main()
