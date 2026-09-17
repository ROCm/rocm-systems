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
import subprocess
import sys
import time

NUM_STREAMS = 2
ITERATIONS_PER_PHASE = 3
TIMEOUT_SECONDS = 30


def _wait_for(path, process=None):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"phased workload exited with status {process.returncode} "
                f"before creating {path}"
            )
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {path}")


def _wait_for_attach_thread(pid):
    deadline = time.monotonic() + TIMEOUT_SECONDS
    task_dir = Path(f"/proc/{pid}/task")
    while time.monotonic() < deadline:
        for comm in task_dir.glob("*/comm"):
            try:
                if comm.read_text().strip() == "rocp-bg-attach":
                    return
            except OSError:
                pass
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for attachment listener")


def _run_phase(kernels):
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    for _ in range(ITERATIONS_PER_PHASE):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize failed")
    kernels.hip_kernels_destroy_streams()


def _tool_count(tool_path):
    tool = ctypes.CDLL(tool_path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    count = tool.rocprofiler_test_record_count
    count.restype = ctypes.c_uint64
    return count()


def _child(args):
    kernels = ctypes.CDLL(args.kernels, mode=ctypes.RTLD_GLOBAL)
    for phase in range(1, 4):
        _run_phase(kernels)
        startup_count = _tool_count(args.startup_tool)
        Path(args.sync_dir, f"startup-{phase}.count").write_text(
            f"{startup_count}\n", encoding="utf-8"
        )
        if phase >= 2:
            attachment_count = _tool_count(args.attachment_tool)
            Path(args.sync_dir, f"attachment-{phase}.count").write_text(
                f"{attachment_count}\n", encoding="utf-8"
            )
        Path(args.sync_dir, f"phase-{phase}.done").touch()
        if phase < 3:
            _wait_for(Path(args.sync_dir, f"phase-{phase}.continue"))
    return 0


def _dispatch_count(path):
    return int(path.read_text(encoding="utf-8").strip())


def _parent(args):
    sync_dir = Path(args.sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)
    for marker in sync_dir.glob("phase-*"):
        marker.unlink()
    for marker in sync_dir.glob("*.count"):
        marker.unlink()

    startup_output = sync_dir / "startup.txt"
    attachment_output = sync_dir / "attachment.txt"
    if startup_output.exists():
        startup_output.unlink()
    if attachment_output.exists():
        attachment_output.unlink()

    child_env = os.environ.copy()
    child_env.update(
        {
            "ROCP_TOOL_ATTACH": "1",
            "ROCP_TOOL_LIBRARIES": args.startup_tool,
            "ROCPROFILER_RECORD_TOOL_OUTPUT": str(startup_output),
        }
    )
    child_args = [
        sys.executable,
        __file__,
        "--child",
        "--kernels",
        args.kernels,
        "--startup-tool",
        args.startup_tool,
        "--attachment-tool",
        args.attachment_tool,
        "--sync-dir",
        str(sync_dir),
    ]
    child = subprocess.Popen(child_args, env=child_env)

    try:
        _wait_for(sync_dir / "phase-1.done", child)
        _wait_for_attach_thread(child.pid)

        os.environ["ROCPROF_ATTACH_TOOL_LIBRARY"] = args.attachment_tool
        os.environ["ROCPROFILER_RECORD_TOOL_OUTPUT"] = str(attachment_output)

        rocattach = ctypes.CDLL(args.rocattach)
        rocattach.rocattach_attach.argtypes = [ctypes.c_int]
        rocattach.rocattach_attach.restype = ctypes.c_int
        rocattach.rocattach_detach.argtypes = [ctypes.c_int]
        rocattach.rocattach_detach.restype = ctypes.c_int

        if rocattach.rocattach_attach(child.pid) != 0:
            raise RuntimeError("rocattach_attach failed")
        (sync_dir / "phase-1.continue").touch()
        _wait_for(sync_dir / "phase-2.done", child)
        if rocattach.rocattach_detach(child.pid) != 0:
            raise RuntimeError("rocattach_detach failed")
        (sync_dir / "phase-2.continue").touch()

        if child.wait(timeout=TIMEOUT_SECONDS) != 0:
            raise RuntimeError("phased workload failed")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    startup_count = _dispatch_count(startup_output)
    attachment_count = _dispatch_count(attachment_output)
    startup_phases = [
        _dispatch_count(sync_dir / f"startup-{phase}.count") for phase in range(1, 4)
    ]
    attachment_phases = [
        _dispatch_count(sync_dir / f"attachment-{phase}.count") for phase in range(2, 4)
    ]
    if not (
        startup_phases[0] > 0
        and startup_phases[1] > startup_phases[0]
        and startup_phases[2] > startup_phases[1]
        and startup_count == startup_phases[2]
    ):
        raise RuntimeError(
            f"startup tool did not record every phase: "
            f"checkpoints={startup_phases}, final={startup_count}"
        )
    if not (
        attachment_phases[0] > 0
        and attachment_phases[1] == attachment_phases[0]
        and attachment_count == attachment_phases[1]
    ):
        raise RuntimeError(
            f"attachment tool recorded outside its active phase: "
            f"checkpoints={attachment_phases}, final={attachment_count}"
        )

    print(
        "Attachment record coexistence test PASSED: "
        f"startup={startup_count}, attachment={attachment_count}"
    )
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rocattach")
    parser.add_argument("--kernels", required=True)
    parser.add_argument("--startup-tool")
    parser.add_argument("--attachment-tool")
    parser.add_argument("--sync-dir", required=True)
    args = parser.parse_args()
    if args.child:
        return _child(args)
    for name in ("rocattach", "startup_tool", "attachment_tool"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required")
    return _parent(args)


if __name__ == "__main__":
    sys.exit(main())
