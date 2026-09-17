# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from __future__ import annotations

import pytest

pytest.importorskip("ortools")

from utils.perfmon_cp_sat import (  # noqa: E402
    counter_ip_block_for_perfmon,
    cp_sat_partition_counters,
)


def test_counter_ip_block_sq_merge() -> None:
    assert counter_ip_block_for_perfmon("SQC_FOO_sum") == "SQ"
    assert counter_ip_block_for_perfmon("SP_BAR") == "SQ"
    assert counter_ip_block_for_perfmon("TCP_REQ_sum") == "TCP"


def test_cp_sat_trivial_single_bin() -> None:
    cfg = {"SQ": 4, "TCP": 2}
    items = ["SQ_A_sum", "SQ_B_sum", "TCP_REQ_sum"]
    part = cp_sat_partition_counters(items, cfg, [], time_limit_s=5.0)
    assert part is not None
    assert len(part) == 1
    assert set(part[0]) == set(items)


def test_cp_sat_two_bins_when_cap_forces_split() -> None:
    cfg = {"SQ": 1}
    items = ["SQ_A_sum", "SQ_B_sum"]
    part = cp_sat_partition_counters(items, cfg, [], time_limit_s=5.0)
    assert part is not None
    assert len(part) == 2


def test_cp_sat_same_bin_group() -> None:
    cfg = {"SQ": 2, "TCP": 2}
    items = ["SQ_A_sum", "SQ_B_sum", "TCP_REQ_sum"]
    groups = [frozenset({"SQ_A_sum", "TCP_REQ_sum"})]
    part = cp_sat_partition_counters(items, cfg, groups, time_limit_s=5.0)
    assert part is not None
    bucket_with_pair = [b for b in part if "SQ_A_sum" in b and "TCP_REQ_sum" in b]
    assert len(bucket_with_pair) == 1


def test_cp_sat_infeasible_group_over_cap() -> None:
    cfg = {"SQ": 1}
    items = ["SQ_A_sum", "SQ_B_sum"]
    groups = [frozenset(items)]
    part = cp_sat_partition_counters(items, cfg, groups, time_limit_s=5.0)
    assert part is None


def test_cp_sat_metric_spread_prefers_colocated_pairs() -> None:
    """Spread penalty should pair (SQ_i, TCP_i) in the same bin when possible."""
    cfg = {"SQ": 1, "TCP": 1}
    items = ["SQ_A_sum", "TCP_REQ_sum", "SQ_B_sum", "TCP_GL1_sum"]
    spread_groups = [[0, 1], [2, 3]]
    part = cp_sat_partition_counters(
        items,
        cfg,
        [],
        time_limit_s=5.0,
        metric_spread_index_groups=spread_groups,
        metric_spread_penalty=50,
        bin_used_weight=1,
    )
    assert part is not None
    with_a = next(b for b in part if "SQ_A_sum" in b)
    assert "TCP_REQ_sum" in with_a
    with_b = next(b for b in part if "SQ_B_sum" in b)
    assert "TCP_GL1_sum" in with_b
