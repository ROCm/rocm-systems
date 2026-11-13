# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/counters/counter_file.py."""

import pytest

from utils.counters.counter_file import (
    CounterFile,
    LimitedSet,
    flat_counters_in_perfmon_file,
    slot_cost,
)

PERFMON_CONFIG = {
    "SQ": 8,
    "TA": 2,
    "TD": 2,
    "TCP": 4,
    "TCC": 4,
    "CPC": 2,
    "CPF": 2,
    "SPI": 6,
    "GRBM": 2,
    "GDS": 4,
}


@pytest.fixture
def perfmon_config():
    return dict(PERFMON_CONFIG)


# =============================================================================
# LimitedSet
# =============================================================================


def test_limited_set_basic():
    ls = LimitedSet(2)
    assert ls.add("SQ_WAVES") is True
    assert ls.add("SQ_BUSY") is True
    assert ls.add("SQ_INSTS") is False  # capacity exhausted
    assert ls.add("SQ_WAVES") is True  # duplicate ok
    assert ls.avail == 0
    assert ls.elements == ["SQ_WAVES", "SQ_BUSY"]


def test_limited_set_tcc_channel_coalescing():
    ls = LimitedSet(1)
    assert ls.add("TCC_HIT[0]") is True
    assert ls.avail == 0
    # Same TCC base: bypasses capacity
    assert ls.add("TCC_HIT[1]") is True
    assert ls.add("TCC_HIT[2]") is True
    # Different TCC base: rejected (no capacity left)
    assert ls.add("TCC_MISS[0]") is False
    assert len(ls.elements) == 3


def test_limited_set_reserve_succeeds_within_capacity():
    """`reserve(n)` debits avail by n and returns True when capacity remains."""
    ls = LimitedSet(4)
    assert ls.add("SQ_WAVES") is True
    assert ls.reserve(2) is True


def test_limited_set_reserve_refuses_when_insufficient():
    """`reserve(n)` returns False and leaves avail untouched when n > avail."""
    ls = LimitedSet(2)
    assert ls.reserve(3) is False
    assert ls.avail == 2
    assert ls.elements == []


def test_limited_set_reserve_does_not_add_elements():
    """
    Reservation is opaque: it does not record a counter name and leaves
    subsequent add() free to use whatever capacity remains.
    """
    ls = LimitedSet(3)
    assert ls.reserve(2) is True
    assert ls.elements == []
    assert ls.avail == 1
    assert ls.add("SQ_WAVES") is True
    assert ls.elements == ["SQ_WAVES"]
    assert ls.avail == 0


# =============================================================================
# CounterFile
# =============================================================================


def test_counter_file_exposes_name_attribute(perfmon_config):
    """The per-block LimitedSet map is exposed via `blocks`."""
    cf = CounterFile("SQ_LEVEL_WAVES_ACCUM", perfmon_config)
    assert cf.name == "SQ_LEVEL_WAVES_ACCUM"
    assert set(cf.blocks.keys()) == set(perfmon_config.keys())
    for block, limited_set in cf.blocks.items():
        assert isinstance(limited_set, LimitedSet)
        assert limited_set.avail == perfmon_config[block]
        assert limited_set.elements == []


def test_counter_file_add_and_block_mapping(perfmon_config):
    cf = CounterFile("0", perfmon_config)

    # SQ, SQC, SP all map to the SQ block (capacity 8)
    assert cf.add("SQ_WAVES") is True
    assert cf.add("SQC_CACHE_HIT") is True
    assert cf.add("SP_SOMETHING") is True
    assert cf.blocks["SQ"].avail == 5  # 8 - 3

    # TA maps to its own block (capacity 2)
    assert cf.add("TA_ADDR") is True
    assert cf.add("TA_DATA") is True
    assert cf.add("TA_EXTRA") is False  # TA full

    # TCP maps to its own block (capacity 4)
    assert cf.add("TCP_READ") is True
    assert cf.blocks["TCP"].avail == 3


def test_counter_file_reserve_delegates_to_block(perfmon_config):
    """
    `reserve(counter, n)` debits the LimitedSet for the block selected by
    counter_to_block(counter) and returns the underlying boolean.
    """
    cf = CounterFile("0", perfmon_config)
    assert cf.add("SQ_WAVES") is True  # SQ avail: 8 -> 7

    assert cf.reserve("SQ_INSTS", 2) is True
    assert cf.blocks["SQ"].avail == 5  # 7 - 2

    # TA capacity is 2 in the fixture, so reserving 3 must fail.
    assert cf.reserve("TA_EXTRA", 3) is False
    assert cf.blocks["TA"].avail == 2  # unchanged after failed reserve

    # Reserve must never record a counter name in the block's elements.
    assert cf.blocks["SQ"].elements == ["SQ_WAVES"]
    assert cf.blocks["TA"].elements == []


# =============================================================================
# flat_counters_in_perfmon_file
# =============================================================================


def test_flat_counters_in_perfmon_file(perfmon_config):
    # Empty file returns empty list
    cf = CounterFile("0", perfmon_config)
    assert flat_counters_in_perfmon_file(cf) == []

    # Add counters across blocks and verify flattened order
    cf.add("SQ_WAVES")
    cf.add("TA_ADDR")
    cf.add("TCP_READ")
    result = flat_counters_in_perfmon_file(cf)
    assert "SQ_WAVES" in result
    assert "TA_ADDR" in result
    assert "TCP_READ" in result
    assert len(result) == 3


@pytest.mark.parametrize(
    ("counter", "cost"),
    [
        ("SQ_WAVES", 1),
        ("SQ_INST_LEVEL_SMEM", 1),
        ("SQ_INST_LEVEL_SMEM_ACCUM", 2),
        ("TCC_HIT[3]", 1),
    ],
)
def test_slot_cost(counter, cost):
    assert slot_cost(counter) == cost


def test_counter_file_accum_costs_two_slots_once(perfmon_config):
    cf = CounterFile("0", perfmon_config)
    assert cf.add("SQ_INST_LEVEL_SMEM_ACCUM") is True
    assert cf.add("SQ_INST_LEVEL_SMEM_ACCUM") is True
    assert cf.blocks["SQ"].avail == perfmon_config["SQ"] - 2


def test_counter_file_accum_with_its_base_costs_three_slots(perfmon_config):
    cf = CounterFile("0", perfmon_config)
    assert cf.add("SQ_INST_LEVEL_SMEM_ACCUM") is True
    assert cf.add("SQ_INST_LEVEL_SMEM") is True
    assert cf.blocks["SQ"].avail == perfmon_config["SQ"] - 3
