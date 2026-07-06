#
# Tests helpers for hardware counter collection.
#

import os

import pytest


def has_kfd_profiler_interface():
    """Return whether a native KFD device node is available to the test process."""
    return os.path.exists("/dev/kfd")


def skip_if_hw_counter_values_unavailable(values):
    """Skip value assertions when the platform did not report counter data.

    WSL/DXG does not expose /dev/kfd, but it can still collect counters through
    libhsakmt's DXG vendor-packet path when supported. Prefer the observed
    counter output over a raw /dev/kfd presence check: if every reported value is
    non-zero, keep validating it. Only skip when there is no KFD interface and
    at least one value was not reported or read back as zero. This preserves
    single-counter validation on WSL/DXG while avoiding false failures for
    multi-counter sets that currently exceed libhsakmt's PM4 frame-size limit.
    """
    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if numeric_values and all(value > 0 for value in numeric_values):
        return

    if not has_kfd_profiler_interface():
        pytest.skip(
            "hardware counter values were not fully reported on this platform; "
            "/dev/kfd is unavailable and dxg vendor-packet counter collection "
            "either is not available or did not arm this counter set"
        )


def skip_if_scratch_memory_unavailable(scratch_records):
    """Skip scratch-memory assertions when no scratch was reported on WSL/no-KFD.

    Scratch-memory allocation events are surfaced by the KFD scheduler. On
    WSL2/DXG there is no /dev/kfd; the GPU is scheduled through the dxg path,
    which does not report scratch allocations, so the trace contains zero
    scratch-memory records even though the workload ran correctly. Skip the
    scratch validation in that case rather than failing. Real-KFD platforms
    (where /dev/kfd is present) keep validating normally.
    """
    if scratch_records:
        return

    if not has_kfd_profiler_interface():
        pytest.skip(
            "no scratch memory was reported on this platform; /dev/kfd is "
            "unavailable and scratch allocations are not surfaced under dxg "
            "scheduling on WSL/DXG"
        )
