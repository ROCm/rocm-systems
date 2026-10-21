# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Describe and check the fixed benchmark runner CPU policy."""

from collections.abc import Iterable
from pathlib import Path

CPU_ROOT = Path('/sys/devices/system/cpu')
PHYSICAL_CORE_COUNT = 64
ONLINE_CPUS = frozenset(range(2 * PHYSICAL_CORE_COUNT))
BENCHMARK_CPUS = frozenset(range(16, PHYSICAL_CORE_COUNT))
BENCHMARK_SIBLING_CPUS = frozenset(cpu + PHYSICAL_CORE_COUNT for cpu in BENCHMARK_CPUS)
POLICY_CPUS = BENCHMARK_CPUS | BENCHMARK_SIBLING_CPUS
BENCHMARK_ALLOWED_CPUS = frozenset(range(PHYSICAL_CORE_COUNT))
IDLE_DISABLE_LATENCY_US = 10


def read(path: Path) -> str:
    return path.read_text().strip()


def check_bpfland(sched_root: Path) -> str:
    state = read(sched_root / 'state')
    name = read(sched_root / 'root/ops')
    if state != 'enabled' or not name.startswith('bpfland_'):
        raise RuntimeError('Bpfland must be enabled')
    return name


def cpu_list(path: Path) -> set[int]:
    cpus = set()
    for item in read(path).split(','):
        ends = item.split('-')
        first, last = (int(ends[0]), int(ends[-1]))
        if len(ends) > 2 or first < 0 or last < first or last > 4095:
            raise ValueError(f'{path}: invalid CPU list')
        cpus.update(range(first, last + 1))
    return cpus


def format_cpus(cpus: Iterable[int]) -> str:
    ranges: list[list[int]] = []
    for cpu in sorted(set(cpus)):
        if ranges and cpu == ranges[-1][-1] + 1:
            ranges[-1].append(cpu)
        else:
            ranges.append([cpu])
    return ','.join(
        str(group[0]) if len(group) == 1 else f'{group[0]}-{group[-1]}'
        for group in ranges
    )


def nominal_frequency_khz(cpu_root: Path, cpu: int) -> int:
    path = cpu_root / f'cpu{cpu}/acpi_cppc/nominal_freq'
    frequency = int(read(path)) * 1000  # CPPC reports MHz; cpufreq uses kHz.
    if frequency <= 0:
        raise ValueError(f'{path}: nominal frequency must be positive')
    return frequency


def policy_settings(cpu_root: Path) -> dict[Path, str]:
    """Validate the fixed layout and controls before returning settings to apply."""
    if cpu_list(cpu_root / 'online') != ONLINE_CPUS:
        raise ValueError(
            f'benchmark runner requires online CPUs {format_cpus(ONLINE_CPUS)}'
        )
    for cpu in sorted(ONLINE_CPUS):
        path = cpu_root / f'cpu{cpu}/topology/thread_siblings_list'
        physical = cpu % PHYSICAL_CORE_COUNT
        if cpu_list(path) != {physical, physical + PHYSICAL_CORE_COUNT}:
            raise ValueError(f'{path}: expected SMT siblings N,N+{PHYSICAL_CORE_COUNT}')

    settings = {}
    covered = set()
    for policy in sorted((cpu_root / 'cpufreq').glob('policy*')):
        cpus = set(map(int, read(policy / 'related_cpus').split()))
        if not cpus & POLICY_CPUS:
            continue
        if not cpus <= POLICY_CPUS:
            raise ValueError(f'{policy}: policy also controls housekeeping CPUs')
        covered.update(cpus)
        frequency = nominal_frequency_khz(cpu_root, min(cpus))
        minimum = int(read(policy / 'cpuinfo_min_freq'))
        maximum = int(read(policy / 'cpuinfo_max_freq'))
        if not minimum <= frequency <= maximum:
            raise ValueError(
                f'{policy}: nominal frequency {frequency} kHz outside {minimum}-{maximum}'
            )
        # Preserve housekeeping boost when per-policy controls are available.
        boost = policy / 'boost'
        if not boost.exists():
            boost = cpu_root / 'cpufreq/boost'
        settings[boost] = '0'
        settings[policy / 'scaling_governor'] = 'performance'
        settings[policy / 'energy_performance_preference'] = 'performance'
        settings[policy / 'scaling_max_freq'] = str(frequency)
        settings[policy / 'scaling_min_freq'] = str(frequency)
    if covered != POLICY_CPUS:
        raise ValueError(
            f'cpufreq policies missing CPUs {format_cpus(POLICY_CPUS - covered)}'
        )

    for cpu in sorted(POLICY_CPUS):
        idle = cpu_root / f'cpu{cpu}/cpuidle'
        states = sorted(idle.glob('state*'))
        if not states:
            raise ValueError(f'{idle}: idle-state controls unavailable')
        for state in states:
            if int(read(state / 'latency')) >= IDLE_DISABLE_LATENCY_US:
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
