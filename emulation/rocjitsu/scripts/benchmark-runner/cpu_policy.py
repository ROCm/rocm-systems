# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Describe and check the fixed benchmark runner CPU policy."""

from pathlib import Path

CPU_ROOT = Path('/sys/devices/system/cpu')
BENCHMARK_CPUS = frozenset(range(16, 64))
BENCHMARK_SIBLING_CPUS = frozenset(range(80, 128))
POLICY_CPUS = BENCHMARK_CPUS | BENCHMARK_SIBLING_CPUS
BENCHMARK_ALLOWED_CPUS = frozenset(range(64))
FREQUENCY_KHZ = 3_200_000


def read(path: Path) -> str:
    return path.read_text().strip()


def cpu_list(path: Path) -> set[int]:
    cpus = set()
    for item in read(path).split(','):
        ends = item.split('-')
        first, last = (int(ends[0]), int(ends[-1]))
        if len(ends) > 2 or first < 0 or last < first or last > 4095:
            raise ValueError(f'{path}: invalid CPU list')
        cpus.update(range(first, last + 1))
    return cpus


def policy_settings(cpu_root: Path) -> dict[Path, str]:
    """Validate the fixed layout and controls before returning settings to apply."""
    if cpu_list(cpu_root / 'online') != set(range(128)):
        raise ValueError('benchmark runner requires online CPUs 0-127')
    for cpu in range(128):
        path = cpu_root / f'cpu{cpu}/topology/thread_siblings_list'
        if cpu_list(path) != {cpu % 64, cpu % 64 + 64}:
            raise ValueError(f'{path}: expected SMT siblings N,N+64')

    settings = {}
    covered = set()
    for policy in sorted((cpu_root / 'cpufreq').glob('policy*')):
        cpus = set(map(int, read(policy / 'related_cpus').split()))
        if not cpus & POLICY_CPUS:
            continue
        if not cpus <= POLICY_CPUS:
            raise ValueError(f'{policy}: policy also controls housekeeping CPUs')
        covered.update(cpus)
        # Preserve housekeeping boost when per-policy controls are available.
        boost = policy / 'boost'
        if not boost.exists():
            boost = cpu_root / 'cpufreq/boost'
        settings[boost] = '0'
        settings[policy / 'scaling_governor'] = 'performance'
        settings[policy / 'energy_performance_preference'] = 'performance'
        settings[policy / 'scaling_max_freq'] = str(FREQUENCY_KHZ)
        settings[policy / 'scaling_min_freq'] = str(FREQUENCY_KHZ)
    if covered != POLICY_CPUS:
        raise ValueError(
            f'cpufreq policies missing CPUs {sorted(POLICY_CPUS - covered)}'
        )

    for cpu in sorted(POLICY_CPUS):
        idle = cpu_root / f'cpu{cpu}/cpuidle'
        states = sorted(idle.glob('state*'))
        if not states:
            raise ValueError(f'{idle}: idle-state controls unavailable')
        for state in states:
            if int(read(state / 'latency')) >= 10:
                settings[state / 'disable'] = '1'
    # Read every required control before setup makes its first write.
    for path, expected in settings.items():
        value = read(path)
        if expected.isdecimal():
            int(value)
    return settings


def check_policy(cpu_root: Path) -> None:
    for path, expected in policy_settings(cpu_root).items():
        actual = read(path)
        if actual != expected:
            raise ValueError(f'{path}: expected {expected}, got {actual}')
