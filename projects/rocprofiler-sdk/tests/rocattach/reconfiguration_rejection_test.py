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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TIMEOUT_SECONDS = 30


def _wait_for_attach_thread(process):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    task_dir = Path(f"/proc/{process.pid}/task")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"target exited with status {process.returncode} before attachment"
            )
        for comm in task_dir.glob("*/comm"):
            try:
                if comm.read_text().strip() == "rocp-bg-attach":
                    return
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for attachment listener")


def _load_rocattach(path):
    library = ctypes.CDLL(path)
    library.rocattach_attach.argtypes = [ctypes.c_int]
    library.rocattach_attach.restype = ctypes.c_int
    library.rocattach_detach.argtypes = [ctypes.c_int]
    library.rocattach_detach.restype = ctypes.c_int
    return library


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocattach", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    child_env["ROCP_TOOL_ATTACH"] = "1"
    child_env.pop("ROCPROF_KERNEL_TRACE", None)
    child_env.pop("ROCPROF_HIP_RUNTIME_API_TRACE", None)
    target = subprocess.Popen([args.target, "1", "1"], env=child_env)

    try:
        _wait_for_attach_thread(target)

        os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"] = args.tool
        os.environ["ROCPROF_KERNEL_TRACE"] = "1"
        os.environ.pop("ROCPROF_HIP_RUNTIME_API_TRACE", None)
        os.environ["ROCPROF_OUTPUT_PATH"] = str(output_dir)
        os.environ["ROCPROF_OUTPUT_FILE_NAME"] = "initial"
        os.environ["ROCPROF_OUTPUT_FORMAT"] = "json"
        os.environ["ROCPROF_ATTACH_OUTPUT_GENERATION_SYNC"] = "1"

        rocattach = _load_rocattach(args.rocattach)
        if rocattach.rocattach_attach(target.pid) != 0:
            raise RuntimeError("initial attachment failed")
        time.sleep(0.2)
        if rocattach.rocattach_detach(target.pid) != 0:
            raise RuntimeError("initial detachment failed")

        os.environ["ROCPROF_HIP_RUNTIME_API_TRACE"] = "1"
        if rocattach.rocattach_attach(target.pid) == 0:
            raise RuntimeError("changed attachment configuration unexpectedly succeeded")

        time.sleep(0.1)
        if target.poll() is not None:
            raise RuntimeError(
                f"changed attachment configuration terminated target with "
                f"status {target.returncode}"
            )

        target.send_signal(signal.SIGINT)
        if target.wait(timeout=TIMEOUT_SECONDS) != 0:
            raise RuntimeError(
                f"target failed after rejected configuration with status "
                f"{target.returncode}"
            )
        print("Attachment reconfiguration rejection test PASSED")
        return 0
    finally:
        if target.poll() is None:
            target.send_signal(signal.SIGINT)
            try:
                target.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                target.kill()
                target.wait()


if __name__ == "__main__":
    sys.exit(main())
