#
# Tests helpers for hardware counter collection.
#

import os

import pytest


def has_kfd_profiler_interface():
    """Return whether a native KFD device node is available to the test process."""
    return os.path.exists("/dev/kfd")


def skip_if_hw_counter_values_unavailable(values):
    """Fail when expected hardware counter output is missing or non-positive.

    Runtime capability checks must skip before result validation. Observed
    output must never be used to infer that counter collection is unsupported.
    """
    found_value = False
    for value in values:
        found_value = True
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            pytest.fail(f"hardware counter value is not numeric: {value!r}")

        if not numeric_value > 0:
            pytest.fail(f"hardware counter value is not positive: {value!r}")

    if not found_value:
        pytest.fail("no hardware counter values were reported")


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
