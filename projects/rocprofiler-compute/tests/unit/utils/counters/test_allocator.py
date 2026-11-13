# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/counters/allocator.py."""

from utils.counters.allocator import allocate_single_pass_metrics, metric_counter_sets
from utils.counters.counter_file import flat_counters_in_perfmon_file

THREE_SQ_SLOTS = {"SQ": 3}


def layout(files):
    """Each pass's counters, sorted, in a stable order."""
    return sorted(sorted(flat_counters_in_perfmon_file(f)) for f in files)


def test_metric_counter_sets_match_channels_and_skip_unprofiled_metrics():
    """A TCC channel stands for all its profiled channels; a metric with no
    profiled counter is left out."""
    profile_counters = {"SQ_WAVES", "TCC_HIT[0]", "TCC_HIT[1]", "TCC_MISS[0]"}

    counter_sets = metric_counter_sets(
        [["AVG(TCC_HIT[0] / SQ_WAVES)"], ["SUM(TA_BUSY_avr)"]],
        "MI300",
        profile_counters,
    )

    assert counter_sets == [frozenset({"SQ_WAVES", "TCC_HIT[0]", "TCC_HIT[1]"})]


def test_allocate_copies_a_counter_so_each_metric_has_one_pass():
    """SQ_B goes into a second pass so that {SQ_B, SQ_D} has a pass of its own."""
    counters = {"SQ_A", "SQ_B", "SQ_C", "SQ_D", "SQ_E", "SQ_F"}
    metric_sets = [
        frozenset({"SQ_A", "SQ_B"}),
        frozenset({"SQ_C", "SQ_D", "SQ_E"}),
        frozenset({"SQ_A", "SQ_F"}),
        frozenset({"SQ_B", "SQ_D"}),
    ]

    files = allocate_single_pass_metrics(counters, metric_sets, THREE_SQ_SLOTS)

    assert layout(files) == [
        ["SQ_A", "SQ_B", "SQ_F"],
        ["SQ_B", "SQ_D"],
        ["SQ_C", "SQ_D", "SQ_E"],
    ]
    assert [counter_file.name for counter_file in files] == ["0", "1", "2"]


def test_allocate_puts_other_counters_in_the_first_pass_with_room():
    """A metric too large for one pass does not get a pass of its own."""
    counters = {"SQ_A", "SQ_B", "SQ_C", "SQ_D"}

    files = allocate_single_pass_metrics(
        counters, [frozenset(counters)], THREE_SQ_SLOTS
    )

    assert layout(files) == [["SQ_A", "SQ_B", "SQ_C"], ["SQ_D"]]


def test_allocate_extends_the_pass_holding_most_of_a_metric():
    """{TA_X, TA_Y} fits both passes and joins the one that holds TA_X."""
    counters = {"SQ_A", "SQ_B", "SQ_C", "TA_X", "TA_Y"}
    metric_sets = [
        frozenset({"SQ_A", "SQ_B"}),
        frozenset({"SQ_C", "TA_X"}),
        frozenset({"TA_X", "TA_Y"}),
    ]

    files = allocate_single_pass_metrics(counters, metric_sets, {"SQ": 2, "TA": 2})

    assert layout(files) == [["SQ_A", "SQ_B"], ["SQ_C", "TA_X", "TA_Y"]]
