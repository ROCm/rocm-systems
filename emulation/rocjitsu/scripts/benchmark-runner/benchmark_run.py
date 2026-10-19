#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run an unprivileged benchmark with CPU allocation checks, affinity, and ASLR disabled."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

from cpu_policy import (
    BENCHMARK_CPUS,
    BENCHMARK_ALLOWED_CPUS,
    POLICY_CPUS,
    CPU_ROOT,
    check_policy,
    cpu_list,
)

CGROUP_ROOT = Path('/sys/fs/cgroup')
PROC_CGROUP = Path('/proc/self/cgroup')
SCHED_ROOT = Path('/sys/kernel/sched_ext')


def check_cpus():
    # A cgroup-v2 entry looks like: 0::/benchmark.slice/actions.runner.example.service
    # The "0::" prefix identifies v2; the rest is relative to the cgroup mount.
    membership = [
        line.removeprefix('0::/')
        for line in PROC_CGROUP.read_text().splitlines()
        if line.startswith('0::/benchmark.slice/')
    ]
    if len(membership) != 1:
        raise RuntimeError('benchmark must run under benchmark.slice')
    if (
        cpu_list(CGROUP_ROOT / 'benchmark.slice/cpuset.cpus.effective')
        != BENCHMARK_ALLOWED_CPUS
    ):
        raise RuntimeError('benchmark.slice must allow exactly CPUs 0-63')
    current = CGROUP_ROOT / membership[0]
    allowed = cpu_list(current / 'cpuset.cpus.effective')
    if not BENCHMARK_CPUS <= allowed <= BENCHMARK_ALLOWED_CPUS:
        raise RuntimeError('benchmark cgroup must allow 16-63 within CPUs 0-63')
    for unit in CGROUP_ROOT.iterdir():
        if not unit.is_dir() or unit.name == 'benchmark.slice':
            continue
        if cpu_list(unit / 'cpuset.cpus.effective') & POLICY_CPUS:
            if 'populated 0' in (unit / 'cgroup.events').read_text().splitlines():
                continue
            raise RuntimeError(
                f'{unit.name}: background work can use benchmark cores or siblings'
            )


def check_scheduler():
    sequence = int((SCHED_ROOT / 'enable_seq').read_text())
    state = (SCHED_ROOT / 'state').read_text().strip()
    name = (SCHED_ROOT / 'root/ops').read_text().strip()
    if state != 'enabled' or not name.startswith('bpfland_'):
        raise RuntimeError('Bpfland must be enabled')
    return name, sequence


def run(command):
    if os.geteuid() == 0:
        raise RuntimeError('benchmark commands must run as an unprivileged user')
    check_cpus()
    check_policy(CPU_ROOT)
    scheduler = check_scheduler()
    previous_affinity = os.sched_getaffinity(0)
    handlers = {}
    child = None
    pending = []

    def forward(signum, frame):
        if child is None:
            pending.append(signum)
        else:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[signum] = signal.signal(signum, forward)
        os.sched_setaffinity(0, BENCHMARK_CPUS)
        if os.sched_getaffinity(0) != BENCHMARK_CPUS:
            raise RuntimeError("benchmark affinity must be exactly 16-63")
        child = subprocess.Popen(
            ["/usr/bin/setarch", os.uname().machine, "-R", *command],
            start_new_session=True,
        )
        for signum in pending:
            forward(signum, None)
        # Blocking wait: no timed-wait polling in the harness.
        status = child.wait()
    finally:
        os.sched_setaffinity(0, previous_affinity)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    check_cpus()
    check_policy(CPU_ROOT)
    if check_scheduler() != scheduler:
        raise RuntimeError('scheduler changed during benchmark; discard results')
    return status if status >= 0 else 128 - status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('provide -- COMMAND [ARGS...]')
    try:
        return run(command)
    except (RuntimeError, OSError, ValueError) as error:
        print(f'benchmark_run: {error}', file=sys.stderr)
        return 125


if __name__ == '__main__':
    sys.exit(main())
