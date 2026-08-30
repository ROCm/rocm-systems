#################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import os
import re
import signal
import subprocess
import time

import pytest

# Intermittent gfx950 cuMem-VMM connectivity-gate abort (not a data error).
CONN_GATE_RE = re.compile(r"LSA signal connectivity gate failed|unhandled system error", re.I)
DATA_FAIL_RE = re.compile(
    r"#wrong\s*=\s*[1-9]|mismatch|check.*fail|Out of bounds values\s*:\s*[1-9]", re.I)


def pytest_addoption(parser):
    parser.addoption("--hostfile", action="store", default="", help="specify MPI hostfile")


def launch_mpi_shell(cmd, timeout_s, hang_msg):
    """Run a shell MPI command in a new session; kill the process group on timeout."""
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(hang_msg.format(out=(out or "")[-2000:]))
    print(out)
    return proc.returncode, out


def run_with_conn_gate_retry(launch_fn, conn_retries, settle_s=3):
    """Retry launch_fn on connectivity-gate aborts; never retry genuine data failures."""
    rc, out = 1, ""
    for attempt in range(1, max(1, conn_retries) + 1):
        rc, out = launch_fn()
        if rc == 0:
            return rc, out
        if DATA_FAIL_RE.search(out or ""):
            return rc, out
        if CONN_GATE_RE.search(out or "") and attempt < conn_retries:
            print("=== connectivity-gate abort (attempt {}/{}); re-launching after settle ===".format(
                attempt, conn_retries))
            time.sleep(settle_s)
            continue
        return rc, out
    return rc, out
