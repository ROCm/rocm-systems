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

    A KFD interface means the platform collects counters natively, so a value
    read back as zero there is a real result and is always asserted. Only where
    there is no /dev/kfd at all can this skip, and then only when at least one
    value was not reported or came back as zero.

    WSL/DXG is that case: it does not expose /dev/kfd, but it can still collect
    counters through libhsakmt's DXG vendor-packet path when supported. Prefer
    the observed counter output over a raw /dev/kfd presence check there: if
    every reported value is non-zero, keep validating it.

    This preserves single-counter validation on WSL/DXG while avoiding false
    failures against a stock libhsakmt that lacks the DXG PMC frame-size
    headroom in WDDMDevice::InitCmdbufInfo(): there a multi-counter pass
    overflows the PM4 frame and arms no counters. A libhsakmt that carries that
    headroom does report multi-counter values, and they are validated normally.
    """
    if has_kfd_profiler_interface():
        return

    numeric_values = []
    for value in values:
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue

    if numeric_values and all(value > 0 for value in numeric_values):
        return

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
