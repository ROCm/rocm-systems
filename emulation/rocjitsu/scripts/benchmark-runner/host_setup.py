#!/usr/bin/python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply the fixed benchmark CPU policy or wait for scheduler startup."""

import argparse
import os
from pathlib import Path
import time

from cpu_policy import CPU_ROOT, FREQUENCY_KHZ, check_policy, policy_settings, read

SCHED_ROOT = Path('/sys/kernel/sched_ext')


def write(path: Path, value: str | int) -> None:
    path.write_text(str(value) + '\n')


def apply(cpu_root: Path) -> None:
    settings = policy_settings(cpu_root)
    for path, value in settings.items():
        # Lower min first when necessary to respect the kernel's min <= max rule.
        if path.name == 'scaling_max_freq':
            minimum = path.with_name('scaling_min_freq')
            if int(read(minimum)) > FREQUENCY_KHZ:
                write(minimum, FREQUENCY_KHZ)
        write(path, value)
    check_policy(cpu_root)


def wait_ready(sched_root: Path) -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            if read(sched_root / 'state') == 'enabled':
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise ValueError('Bpfland did not become ready within 30 seconds')
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--wait-ready', action='store_true', help='wait for Bpfland readiness'
    )
    args = parser.parse_args()
    try:
        if args.wait_ready:
            wait_ready(SCHED_ROOT)
        else:
            if os.geteuid() != 0:
                raise ValueError('CPU setup requires administrator privileges')
            apply(CPU_ROOT)
    except (OSError, ValueError) as error:
        parser.exit(1, f'host_setup: {error}\n')


if __name__ == '__main__':
    main()
