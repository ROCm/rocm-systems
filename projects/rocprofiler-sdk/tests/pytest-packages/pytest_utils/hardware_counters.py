#
# Tests helpers for hardware counter collection.
#

import os

import pytest

from . import gpu_uses_auto_perf_level


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


def skip_if_zero_counters_in_auto_mode(values, gpu_agent):
    """Skip when a counter reads zero on a gfx11/gfx12 GPU at the AUTO performance level.

    At AUTO these GPUs gate the perfmon clock of some hardware blocks, so counters from
    those blocks can read zero for some or all dispatches. See 'Setting GPU performance
    level for PMC profiling' in the rocprofv3 documentation.
    """
    try:
        numeric_values = [float(value) for value in values]
    except (TypeError, ValueError):
        return

    # negative and NaN values are failures, not the AUTO zero-counter condition
    if any(not value >= 0 for value in numeric_values):
        return

    has_zero = any(value == 0 for value in numeric_values)
    if has_zero and gpu_agent is not None and gpu_uses_auto_perf_level(gpu_agent):
        pytest.skip(
            "Zero counters with AMDSMI_DEV_PERF_LEVEL_AUTO; nonzero values not required"
        )


def first_gpu_agent(agent_info_rows):
    """Return the first GPU in an agent_info CSV as an agent for gpu_uses_auto_perf_level."""
    return next(
        ({"name": row["Name"]} for row in agent_info_rows if row["Agent_Type"] == "GPU"),
        None,
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
