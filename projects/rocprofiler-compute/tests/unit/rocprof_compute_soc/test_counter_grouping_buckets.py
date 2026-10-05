# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pytest

from rocprof_compute_soc.counter_grouping_buckets import (
    counters_fit_one_bucket,
    rebuild_counter_file,
)
from rocprof_compute_soc.soc_base import flat_counters_in_perfmon_file
from tests.unit.rocprof_compute_soc.test_soc_base import PERFMON_CONFIG


@pytest.mark.misc
def test_counters_fit_one_bucket_respects_slot_limit():
    small = frozenset({"GRBM_GUI_ACTIVE", "GRBM_SPI_BUSY"})
    assert counters_fit_one_bucket(small, PERFMON_CONFIG)


@pytest.mark.misc
def test_counters_fit_one_bucket_accum_costs_two():
    seven_plus_accum = frozenset({
        "SQ_ACTIVE_INST_ANY",
        "SQ_INSTS",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_VALU",
        "SQ_VALU_MFMA_BUSY_CYCLES",
        "SQ_WAVES",
        "SQ_INST_LEVEL_SMEM_ACCUM",
    })
    assert not counters_fit_one_bucket(seven_plus_accum, PERFMON_CONFIG)
    assert counters_fit_one_bucket(
        frozenset({"SQ_INST_LEVEL_SMEM_ACCUM"}), PERFMON_CONFIG
    )
    # BASE already present: ACCUM only needs +1 → 7 plain + base + ACCUM = 8.
    six_plus_base_and_accum = frozenset({
        "SQ_ACTIVE_INST_ANY",
        "SQ_INSTS",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_VALU",
        "SQ_VALU_MFMA_BUSY_CYCLES",
        "SQ_WAVES",
        "SQ_INST_LEVEL_SMEM",
        "SQ_INST_LEVEL_SMEM_ACCUM",
    })
    assert counters_fit_one_bucket(six_plus_base_and_accum, PERFMON_CONFIG)


@pytest.mark.misc
def test_rebuild_counter_file_round_trip():
    counters = {"GRBM_GUI_ACTIVE", "GRBM_SPI_BUSY"}
    rebuilt = rebuild_counter_file("0", PERFMON_CONFIG, counters)
    assert rebuilt is not None
    assert set(flat_counters_in_perfmon_file(rebuilt)) == counters
