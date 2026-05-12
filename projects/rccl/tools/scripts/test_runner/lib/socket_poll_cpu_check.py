#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
socket_poll_cpu_check.py
========================

Performance assertion for NCCL_SOCKET_POLL_TIMEOUT_MSEC (NCCL v2.29.2-1).

The release notes claim that setting NCCL_SOCKET_POLL_TIMEOUT_MSEC > 0 lets
bootstrap socketWait() block on poll() instead of spinning, "in order to
reduce CPU usage." This script measures that claim end-to-end by running
the rccl-tests sendrecv_perf binary twice with otherwise-identical
configurations:

  * Run A: NCCL_SOCKET_POLL_TIMEOUT_MSEC unset (legacy spin path).
  * Run B: NCCL_SOCKET_POLL_TIMEOUT_MSEC=10  (new poll path).

Child-process user+system CPU time is collected from
resource.getrusage(RUSAGE_CHILDREN). The script asserts that run B uses
meaningfully less CPU than run A.

This is intentionally a coarse, end-to-end check. It is deliberately not in
the unit-test config because (a) it requires GPUs / MPI and (b) CPU-time
deltas are noisy. It belongs in a perf config and is expected to fail today
because current RCCL has no poll path -- both runs will spin equally.
"""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import subprocess
import sys
from typing import List, Optional


def _which(binary: str) -> Optional[str]:
    """Find a perf binary either by absolute path or via $PATH / RCCL_TEST_BIN_DIR."""
    if os.path.isabs(binary) and os.path.isfile(binary):
        return binary
    test_bin_dir = os.environ.get("RCCL_TEST_BIN_DIR")
    if test_bin_dir:
        candidate = os.path.join(test_bin_dir, binary)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which(binary)


def _run(binary: str, args: List[str], poll_timeout: Optional[str],
         mpirun: Optional[str], num_ranks: int) -> float:
    """Run the perf binary once and return child CPU seconds consumed."""
    env = os.environ.copy()
    if poll_timeout is None:
        env.pop("NCCL_SOCKET_POLL_TIMEOUT_MSEC", None)
    else:
        env["NCCL_SOCKET_POLL_TIMEOUT_MSEC"] = poll_timeout

    cmd: List[str]
    if mpirun:
        cmd = [mpirun, "-np", str(num_ranks), binary, *args]
    else:
        cmd = [binary, *args]

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)

    if proc.returncode != 0:
        sys.stderr.write(
            f"[socket_poll_cpu_check] child failed (rc={proc.returncode}) for "
            f"poll_timeout={poll_timeout!r}\n")
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise SystemExit(2)

    user = after.ru_utime - before.ru_utime
    sysd = after.ru_stime - before.ru_stime
    return user + sysd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", default="sendrecv_perf",
                   help="Perf binary to drive (default: sendrecv_perf)")
    p.add_argument("--mpirun", default=os.environ.get("MPI_PATH", "") and
                   os.path.join(os.environ["MPI_PATH"], "bin", "mpirun") or
                   shutil.which("mpirun"),
                   help="Path to mpirun (default: $MPI_PATH/bin/mpirun or PATH)")
    p.add_argument("--num-ranks", type=int, default=2,
                   help="MPI ranks (default: 2)")
    p.add_argument("--poll-timeout", default="10",
                   help="Value to use for NCCL_SOCKET_POLL_TIMEOUT_MSEC in the "
                        "second run (default: 10)")
    p.add_argument("--ratio-threshold", type=float, default=0.7,
                   help="Assert cpu_with_poll / cpu_without_poll < this value "
                        "(default: 0.7)")
    p.add_argument("--perf-args", default="-b 8 -e 1K -f 2 -g 1 -n 5 -w 1",
                   help="Args passed to the perf binary; default keeps the run "
                        "small and bootstrap-dominated")
    args = p.parse_args()

    binary = _which(args.binary)
    if not binary:
        sys.stderr.write(f"[socket_poll_cpu_check] cannot find {args.binary}\n")
        return 2
    if args.num_ranks > 1 and not args.mpirun:
        sys.stderr.write("[socket_poll_cpu_check] mpirun not found\n")
        return 2

    perf_args = args.perf_args.split()

    print(f"[socket_poll_cpu_check] binary    = {binary}")
    print(f"[socket_poll_cpu_check] mpirun    = {args.mpirun}")
    print(f"[socket_poll_cpu_check] num_ranks = {args.num_ranks}")
    print(f"[socket_poll_cpu_check] perf_args = {perf_args}")

    print("[socket_poll_cpu_check] run A: NCCL_SOCKET_POLL_TIMEOUT_MSEC unset")
    cpu_spin = _run(binary, perf_args, None, args.mpirun, args.num_ranks)
    print(f"[socket_poll_cpu_check]   child CPU = {cpu_spin:.3f} s")

    print(f"[socket_poll_cpu_check] run B: NCCL_SOCKET_POLL_TIMEOUT_MSEC={args.poll_timeout}")
    cpu_poll = _run(binary, perf_args, args.poll_timeout, args.mpirun,
                    args.num_ranks)
    print(f"[socket_poll_cpu_check]   child CPU = {cpu_poll:.3f} s")

    if cpu_spin <= 0.0:
        sys.stderr.write("[socket_poll_cpu_check] baseline CPU time is zero; "
                         "cannot compute ratio\n")
        return 2

    ratio = cpu_poll / cpu_spin
    print(f"[socket_poll_cpu_check] ratio (poll/spin) = {ratio:.3f} "
          f"(threshold < {args.ratio_threshold})")

    if ratio >= args.ratio_threshold:
        sys.stderr.write(
            "[socket_poll_cpu_check] FAIL: NCCL_SOCKET_POLL_TIMEOUT_MSEC did "
            "not reduce CPU usage enough\n")
        return 1

    print("[socket_poll_cpu_check] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
