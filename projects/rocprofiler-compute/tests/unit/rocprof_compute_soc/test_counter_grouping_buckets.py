# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pytest

from rocprof_compute_soc.counter_grouping_buckets import (
    count_multi_bucket_metrics,
    count_single_pass_metrics_split_across_buckets,
    counters_fit_one_bucket,
    iter_metric_groups,
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
    # ACCUM costs 2 even beside its BASE, so the pair costs 3.
    five_plus_base_and_accum = frozenset({
        "SQ_ACTIVE_INST_ANY",
        "SQ_INSTS",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_VALU",
        "SQ_WAVES",
        "SQ_INST_LEVEL_SMEM",
        "SQ_INST_LEVEL_SMEM_ACCUM",
    })
    assert counters_fit_one_bucket(five_plus_base_and_accum, PERFMON_CONFIG)
    six_plus_base_and_accum = five_plus_base_and_accum | {"SQ_VALU_MFMA_BUSY_CYCLES"}
    assert not counters_fit_one_bucket(six_plus_base_and_accum, PERFMON_CONFIG)


@pytest.mark.misc
def test_sp_valu_and_sq_vmem_do_not_fit_one_pass():
    """gfx1250: both families charge SQ, and SP FLOP counters return zero together."""
    sq_budget = {"SQ": 8}
    sp_only = frozenset({"SP_VALU_FLOPS_FP32", "SQ_INSTS_VALU"})
    vmem_only = frozenset({"SQ_VMEM_REQ", "SQ_INSTS_VMEM"})
    mixed = frozenset({"SP_VALU_FLOPS_FP32", "SQ_VMEM_REQ"})
    assert counters_fit_one_bucket(sp_only, sq_budget)
    assert counters_fit_one_bucket(vmem_only, sq_budget)
    assert not counters_fit_one_bucket(mixed, sq_budget)
    assert rebuild_counter_file("0", sq_budget, set(mixed)) is None


@pytest.mark.misc
def test_rebuild_counter_file_round_trip():
    counters = {"GRBM_GUI_ACTIVE", "GRBM_SPI_BUSY"}
    rebuilt = rebuild_counter_file("0", PERFMON_CONFIG, counters)
    assert rebuilt is not None
    assert set(flat_counters_in_perfmon_file(rebuilt)) == counters


class FormulaSoc:
    """Serves a fixed metric list without reading analysis YAML."""

    def __init__(self, rows):
        self.rows = rows

    def metric_groups(self, profile_counters):
        return iter_metric_groups(self.rows, "MI300", profile_counters)


@pytest.mark.misc
def test_copied_counter_is_not_counted_as_split():
    """A group held by one bucket stays together when a member is copied."""
    files = [
        rebuild_counter_file("0", PERFMON_CONFIG, {"SQ_WAVES", "GRBM_GUI_ACTIVE"}),
        rebuild_counter_file("1", PERFMON_CONFIG, {"SQ_WAVES", "TA_BUSY"}),
    ]
    assert all(counter_file is not None for counter_file in files)
    soc = FormulaSoc([
        ("1", 1, 0, "waves and grbm", ["SQ_WAVES / GRBM_GUI_ACTIVE"]),
        ("1", 1, 1, "waves and ta", ["SQ_WAVES / TA_BUSY"]),
        ("1", 1, 2, "grbm and ta", ["GRBM_GUI_ACTIVE / TA_BUSY"]),
    ])
    profile = {"SQ_WAVES", "GRBM_GUI_ACTIVE", "TA_BUSY"}

    assert (
        count_single_pass_metrics_split_across_buckets(
            files, soc, profile, PERFMON_CONFIG
        )
        == 1
    )
    assert count_multi_bucket_metrics(files, soc, profile) == 1


@pytest.mark.misc
def test_iter_metric_groups_reads_only_the_given_inputs():
    """YAML rows, GPU series and expanded counters are enough."""
    rows = [
        ("1700", 1701, 1, "waves", ["SQ_WAVES / GRBM_GUI_ACTIVE"]),
        ("2", 2, 1, "tcc hits", ["TCC_HIT[ + SQ_WAVES"]),
        ("3", 3, 2, "not profiled", ["TA_BUSY"]),
    ]
    profile = {"SQ_WAVES", "GRBM_GUI_ACTIVE", "TCC_HIT[0]", "TCC_HIT[1]"}

    groups = iter_metric_groups(iter(rows), "MI300", profile)

    assert [(counters, label) for _key, counters, label in groups] == [
        (frozenset({"SQ_WAVES", "TCC_HIT[0]", "TCC_HIT[1]"}), "2.2.1 (tcc hits)"),
        (frozenset({"SQ_WAVES", "GRBM_GUI_ACTIVE"}), "1700.1701.1 (waves)"),
    ]
