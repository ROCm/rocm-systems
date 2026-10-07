# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the short-term TCC channel rules."""

from rocprof_compute_soc.counter_file import flat_counters_in_perfmon_file
from rocprof_compute_soc.counter_grouping_buckets import rebuild_counter_file
from rocprof_compute_soc.counter_grouping_tcc import (
    adjust_candidate_groups,
    keep_copies_groups_still_need,
)


def channel_bases(bucket):
    return {counter.split("[")[0] for counter in flat_counters_in_perfmon_file(bucket)}


def group_is_in_some_bucket(files, group):
    return any(
        set(group) <= set(flat_counters_in_perfmon_file(bucket)) for bucket in files
    )


def test_adjust_candidate_groups_keeps_1805_as_one_group():
    group = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ[1]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_WRREQ[1]",
        "TCC_EA0_ATOMIC[0]",
        "TCC_EA0_ATOMIC[1]",
    })
    assert adjust_candidate_groups(group, set(group)) == [group]


def test_adjust_candidate_groups_keeps_level_request_series():
    group = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ_LEVEL[0]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_WRREQ_LEVEL[0]",
    })
    assert adjust_candidate_groups(group, set(group)) == [group]


def test_adjust_candidate_groups_adds_matching_request_channels():
    group = frozenset({"TCC_EA0_RDREQ_LEVEL[0]", "TCC_EA0_RDREQ_LEVEL[1]"})
    profile = {
        "TCC_EA0_RDREQ_LEVEL[0]",
        "TCC_EA0_RDREQ_LEVEL[1]",
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ[1]",
        "TCC_EA0_WRREQ[0]",
    }
    adjusted = adjust_candidate_groups(group, profile)
    assert len(adjusted) == 1
    expanded = adjusted[0]
    assert "TCC_EA0_RDREQ[0]" in expanded
    assert "TCC_EA0_RDREQ[1]" in expanded
    assert "TCC_EA0_WRREQ[0]" not in expanded


def test_keep_copies_keeps_level_home_and_needed_request_copies():
    cfg = {"TCC": 4, "SQ": 8}
    atomic = rebuild_counter_file(
        "1",
        cfg,
        {
            "TCC_EA0_ATOMIC[0]",
            "TCC_EA0_ATOMIC_LEVEL[0]",
            "TCC_EA0_RDREQ[0]",
            "TCC_EA0_WRREQ[0]",
        },
    )
    latency = rebuild_counter_file(
        "2",
        cfg,
        {
            "TCC_EA0_RDREQ[0]",
            "TCC_EA0_RDREQ_LEVEL[0]",
            "TCC_EA0_WRREQ[0]",
            "TCC_EA0_WRREQ_LEVEL[0]",
        },
    )
    assert atomic is not None and latency is not None
    level_home = {
        "TCC_EA0_RDREQ",
        "TCC_EA0_RDREQ_LEVEL",
        "TCC_EA0_WRREQ",
        "TCC_EA0_WRREQ_LEVEL",
    }
    # The request series stays in the pass that also holds its LEVEL.
    without_request_row = keep_copies_groups_still_need([atomic, latency], cfg)
    assert channel_bases(without_request_row[1]) == level_home

    # Panel 1805 stays complete in the pass that already holds the three
    # request columns, including the extra read and write copies.
    request_row = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_ATOMIC[0]",
    })
    files = keep_copies_groups_still_need(
        [atomic, latency],
        cfg,
        [request_row],
    )
    assert channel_bases(files[0]) == {
        "TCC_EA0_ATOMIC",
        "TCC_EA0_ATOMIC_LEVEL",
        "TCC_EA0_RDREQ",
        "TCC_EA0_WRREQ",
    }
    assert channel_bases(files[1]) == level_home
    assert group_is_in_some_bucket(files, request_row)


def test_keep_copies_leaves_request_series_without_a_level_home():
    cfg = {"TCC": 4, "SQ": 8}
    only_req = rebuild_counter_file(
        "0",
        cfg,
        {"TCC_EA0_RDREQ[0]", "TCC_EA0_WRREQ[0]"},
    )
    assert only_req is not None
    files = keep_copies_groups_still_need([only_req], cfg)
    assert channel_bases(files[0]) == {
        "TCC_EA0_RDREQ",
        "TCC_EA0_WRREQ",
    }


def test_keep_copies_when_a_group_still_needs_them():
    cfg = {"TCC": 4, "SQ": 8}
    home = rebuild_counter_file(
        "0",
        cfg,
        {"TCC_EA0_RDREQ[0]", "TCC_EA0_RDREQ_LEVEL[0]"},
    )
    required = frozenset({"TCC_EA0_RDREQ[0]", "SQ_A"})
    extra = rebuild_counter_file("1", cfg, set(required))
    assert home is not None and extra is not None

    files = keep_copies_groups_still_need(
        [home, extra],
        cfg,
        [required],
    )

    assert group_is_in_some_bucket(files, required)
    assert "TCC_EA0_RDREQ" in channel_bases(files[1])
