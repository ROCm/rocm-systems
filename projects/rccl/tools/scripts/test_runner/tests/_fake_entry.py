#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Fake init-pipeline test entry for host tests (stands in for a real gtest binary).

Mimics the C++ lifecycle under RCCL_TEST_RENDEZVOUS_DIR:
  warm (sleep) -> atomically publish READY -> wait for GO -> execute -> exit.

During "execute" it appends its [start, end] interval to a shared timeline file
so the test can assert no two entries execute concurrently. It also asserts GO is
never observed before it published READY.

Usage: _fake_entry.py <warm_s> <exec_s> <exit_code> <label> <timeline> [--skip-ready]
Env:   RCCL_TEST_RENDEZVOUS_DIR (dir holding ready/go tokens)
"""

import os
import sys
import time


def main():
    warm_s = float(sys.argv[1])
    exec_s = float(sys.argv[2])
    exit_code = int(sys.argv[3])
    label = sys.argv[4]
    timeline = sys.argv[5]
    skip_ready = "--skip-ready" in sys.argv[6:]

    # A rerun can flip a failure to a pass via rerun_env_variables={"FORCE_PASS":"1"}.
    if os.environ.get("FORCE_PASS"):
        exit_code = 0

    rdv = os.environ.get("RCCL_TEST_RENDEZVOUS_DIR", "")
    print(f"[fake {label}] pid {os.getpid()} start rdv={rdv}", flush=True)

    # Warm (device-code load stand-in).
    time.sleep(warm_s)

    if skip_ready:
        # Simulate a crash/exit before READY (init failure).
        print(f"[fake {label}] exiting before READY", flush=True)
        sys.exit(exit_code)

    # A serial entry has no rendezvous: just warm + execute (no READY/GO).
    if rdv:
        # Publish READY atomically (temp + rename), exactly like the C++ helper.
        ready_tmp = os.path.join(rdv, "ready.tmp")
        ready = os.path.join(rdv, "ready")
        with open(ready_tmp, "w") as f:
            f.write("ready\n")
        os.replace(ready_tmp, ready)
        t_ready = time.monotonic()
        print(f"[fake {label}] READY", flush=True)

        # Block until the runner writes GO.
        go = os.path.join(rdv, "go")
        while not os.path.exists(go):
            time.sleep(0.01)
        t_go = time.monotonic()
        # GO must never be observed before READY was published.
        assert t_go >= t_ready, f"{label}: GO observed before READY"
        print(f"[fake {label}] GO", flush=True)

    # Execute: record the interval so the test can prove serial (non-overlapping)
    # execution across entries.
    start = time.monotonic()
    time.sleep(exec_s)
    end = time.monotonic()
    with open(timeline, "a") as f:
        f.write(f"{label} {start:.6f} {end:.6f}\n")
    print(f"[fake {label}] exec done, exit {exit_code}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
