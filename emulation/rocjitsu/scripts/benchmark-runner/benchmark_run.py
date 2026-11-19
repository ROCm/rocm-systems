#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a benchmark with CPU allocation checks, affinity, and ASLR disabled."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from cpu_policy import (
    BENCHMARK_CPUS,
    BENCHMARK_ALLOWED_CPUS,
    POLICY_CPUS,
    CPU_ROOT,
    check_bpfland,
    check_policy,
    cpu_list,
    format_cpus,
)

CGROUP_ROOT = Path('/sys/fs/cgroup')
PROC_CGROUP = Path('/proc/self/cgroup')
SCHED_ROOT = Path('/sys/kernel/sched_ext')


def check_cpus() -> None:
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
        raise RuntimeError(
            f'benchmark.slice must allow exactly CPUs {format_cpus(BENCHMARK_ALLOWED_CPUS)}'
        )
    current = CGROUP_ROOT / membership[0]
    allowed = cpu_list(current / 'cpuset.cpus.effective')
    if not BENCHMARK_CPUS <= allowed <= BENCHMARK_ALLOWED_CPUS:
        raise RuntimeError(
            f'benchmark cgroup must allow {format_cpus(BENCHMARK_CPUS)} within CPUs {format_cpus(BENCHMARK_ALLOWED_CPUS)}'
        )
    for unit in CGROUP_ROOT.iterdir():
        if not unit.is_dir() or unit.name == 'benchmark.slice':
            continue
        if cpu_list(unit / 'cpuset.cpus.effective') & POLICY_CPUS:
            if 'populated 0' in (unit / 'cgroup.events').read_text().splitlines():
                continue
            raise RuntimeError(
                f'{unit.name}: background work can use benchmark cores or siblings'
            )


def check_scheduler() -> tuple[str, int]:
    sequence = int((SCHED_ROOT / 'enable_seq').read_text())
    name = check_bpfland(SCHED_ROOT)
    return name, sequence


def run(command: list[str]) -> int:
    check_cpus()
    check_policy(CPU_ROOT)
    scheduler = check_scheduler()
    previous_affinity = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, BENCHMARK_CPUS)
        if os.sched_getaffinity(0) != BENCHMARK_CPUS:
            raise RuntimeError(
                f'benchmark affinity must be exactly {format_cpus(BENCHMARK_CPUS)}'
            )
        # Keep descendants in the harness's process group for timeout cleanup.
        child = subprocess.Popen(
            ["/usr/bin/setarch", os.uname().machine, "-R", *command],
        )
        # Blocking wait: no timed-wait polling in the harness.
        status = child.wait()
    finally:
        os.sched_setaffinity(0, previous_affinity)
    check_cpus()
    check_policy(CPU_ROOT)
    if check_scheduler() != scheduler:
        raise RuntimeError('scheduler changed during benchmark; discard results')
    return status if status >= 0 else 128 - status


def main() -> int:
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
