# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from unittest.mock import patch

import pytest

from rocprof_compute_soc.counter_grouping_refill import (
    apply_metric_coalesce_refill_pass,
    count_multi_bucket_metrics,
    counters_fit_one_bucket,
    rebuild_counter_file,
)
from rocprof_compute_soc.soc_base import flat_counters_in_perfmon_file
from tests.unit.rocprof_compute_soc.test_soc_base import PERFMON_CONFIG, _make_soc


@pytest.mark.misc
def test_counters_fit_one_bucket_respects_slot_limit():
    small = frozenset({"GRBM_GUI_ACTIVE", "GRBM_SPI_BUSY"})
    assert counters_fit_one_bucket(small, PERFMON_CONFIG)


@pytest.mark.misc
def test_rebuild_counter_file_round_trip():
    counters = {"GRBM_GUI_ACTIVE", "GRBM_SPI_BUSY"}
    rebuilt = rebuild_counter_file("0", PERFMON_CONFIG, counters)
    assert rebuilt is not None
    assert set(flat_counters_in_perfmon_file(rebuilt)) == counters


@pytest.mark.misc
def test_refill_does_not_add_buckets():
    soc = _make_soc(PERFMON_CONFIG)
    counters = {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum", "SQ_WAVES"}
    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=()):
        before_files, file_count, _acc = soc._allocate_perfmon_counter_files(
            counters,
            apply_refill=False,
        )
        after_files, _, stats = apply_metric_coalesce_refill_pass(
            soc,
            before_files,
            file_count,
            counters,
            PERFMON_CONFIG,
        )
    assert len(after_files) == len(before_files)
    assert stats.bucket_count == len(before_files)
    after_multi = count_multi_bucket_metrics(after_files, soc, counters)
    before_multi = count_multi_bucket_metrics(before_files, soc, counters)
    assert after_multi <= before_multi
