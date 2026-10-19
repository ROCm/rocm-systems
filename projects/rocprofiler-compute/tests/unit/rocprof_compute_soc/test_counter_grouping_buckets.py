# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from types import SimpleNamespace

import pytest

from rocprof_compute_soc.counter_grouping_buckets import (
    count_multi_bucket_metrics,
    count_single_pass_metrics_split_across_buckets,
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


class _FormulaSoc:
    """Serves a fixed metric list without reading analysis YAML."""

    def __init__(self, rows):
        self._mspec = SimpleNamespace(gpu_series="MI300")
        self._rows = rows

    def _same_bucket_priority_metric_ids(self):
        return ()

    def _expand_tcc_template_counters(self, counters):
        return set(counters)

    def _iter_arch_analysis_yaml_metrics(self):
        return iter(self._rows)


@pytest.mark.misc
def test_copied_counter_is_not_counted_as_split():
    """A group held by one bucket stays together when a member is copied."""
    files = [
        rebuild_counter_file("0", PERFMON_CONFIG, {"SQ_WAVES", "GRBM_GUI_ACTIVE"}),
        rebuild_counter_file("1", PERFMON_CONFIG, {"SQ_WAVES", "TA_BUSY"}),
    ]
    assert all(counter_file is not None for counter_file in files)
    soc = _FormulaSoc([
        ("1", 1, 0, "waves and grbm", "SQ_WAVES / GRBM_GUI_ACTIVE"),
        ("1", 1, 1, "waves and ta", "SQ_WAVES / TA_BUSY"),
        ("1", 1, 2, "grbm and ta", "GRBM_GUI_ACTIVE / TA_BUSY"),
    ])
    profile = {"SQ_WAVES", "GRBM_GUI_ACTIVE", "TA_BUSY"}

    assert (
        count_single_pass_metrics_split_across_buckets(
            files, soc, profile, PERFMON_CONFIG
        )
        == 1
    )
    assert count_multi_bucket_metrics(files, soc, profile) == 1
