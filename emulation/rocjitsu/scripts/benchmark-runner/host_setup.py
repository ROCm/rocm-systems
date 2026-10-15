#!/usr/bin/python3
"""Apply the fixed benchmark CPU policy or wait for scheduler startup."""

import argparse
import os
from pathlib import Path
import time

CPU_ROOT = Path('/sys/devices/system/cpu')
SCHED_ROOT = Path('/sys/kernel/sched_ext')
RESERVED_CPUS = set(range(16, 64)) | set(range(80, 128))
FREQUENCY_KHZ = 3_200_000


def read(path):
    return path.read_text().strip()


def write(path, value):
    path.write_text(str(value) + '\n')


def apply(cpu_root):
    policies = [
        policy for policy in sorted((cpu_root / 'cpufreq').glob('policy*'))
        if set(map(int, read(policy / 'related_cpus').split())) & RESERVED_CPUS
    ]
    boost = cpu_root / 'cpufreq/boost'
    # Preserve housekeeping boost when per-policy controls are available.
    if any(not (policy / 'boost').exists() for policy in policies):
        write(boost, 0)
    for policy in policies:
        if (policy / 'boost').exists():
            write(policy / 'boost', 0)
        write(policy / 'scaling_governor', 'performance')
        write(policy / 'energy_performance_preference', 'performance')
        # Respect the kernel's min <= max constraint in either direction.
        if int(read(policy / 'scaling_min_freq')) > FREQUENCY_KHZ:
            write(policy / 'scaling_min_freq', FREQUENCY_KHZ)
        write(policy / 'scaling_max_freq', FREQUENCY_KHZ)
        write(policy / 'scaling_min_freq', FREQUENCY_KHZ)
    for cpu in sorted(RESERVED_CPUS):
        for state in sorted((cpu_root / f'cpu{cpu}/cpuidle').glob('state*')):
            if int(read(state / 'latency')) >= 10:
                write(state / 'disable', 1)


def wait_ready(sched_root):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-ready', action='store_true', help='wait for Bpfland readiness')
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
