# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for experimental single-pass-packable packing helpers."""

from __future__ import annotations

from rocprof_compute_soc.counter_grouping_refill import rebuild_counter_file
from rocprof_compute_soc.counter_grouping_single_pass import (
    _any_bucket_has_full_group,
    _ensure_packable_union,
    _first_fit_unplaced,
    _reduce_passes,
    single_pass_packable_enabled_from_env,
    try_allocate_single_pass_packable,
)
from rocprof_compute_soc.soc_base import flat_counters_in_perfmon_file


class MinimalSoC:
    """Minimum interface required by the experimental allocator."""

    def _same_bucket_priority_metric_ids(self):
        return ()

    def _iter_arch_analysis_yaml_metrics(self):
        return []


def test_env_gate_default_off(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)
    assert single_pass_packable_enabled_from_env() is False
    monkeypatch.setenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", "1")
    assert single_pass_packable_enabled_from_env() is True


def test_allocator_disabled_preserves_work_set(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)
    work_set = {"SQ_A"}

    assert try_allocate_single_pass_packable(MinimalSoC(), work_set, {"SQ": 1}) is None
    assert work_set == {"SQ_A"}


def test_allocator_enabled_places_leftovers_and_clears_work_set(monkeypatch):
    monkeypatch.setenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", "1")
    work_set = {"SQ_A", "SQ_B"}

    result = try_allocate_single_pass_packable(
        MinimalSoC(),
        work_set,
        {"SQ": 2},
        file_count_start=4,
    )

    assert result is not None
    files, file_count, stats = result
    assert len(files) == 1
    assert set(flat_counters_in_perfmon_file(files[0])) == {"SQ_A", "SQ_B"}
    assert file_count == 5
    assert stats.packable_multi_after == 0
    assert work_set == set()


def test_overlapping_unions_share_bucket_when_cap_allows():
    cfg = {"SQ": 4}
    g1 = frozenset({"SQ_A", "SQ_B", "SQ_C"})
    g2 = frozenset({"SQ_C", "SQ_D"})
    files, fc = _ensure_packable_union([], g1, cfg, 0)
    files, fc = _ensure_packable_union(files, g2, cfg, fc)
    assert len(files) == 1
    assert _any_bucket_has_full_group(files, g1)
    assert _any_bucket_has_full_group(files, g2)


def test_conflicting_unions_open_second_bucket_with_duplicate():
    """When G1∪G2 does not fit, G2 gets its own bucket (may duplicate SQ_C)."""
    cfg = {"SQ": 3}
    g1 = frozenset({"SQ_A", "SQ_B", "SQ_C"})
    g2 = frozenset({"SQ_C", "SQ_D", "SQ_E"})
    files, fc = _ensure_packable_union([], g1, cfg, 0)
    files, fc = _ensure_packable_union(files, g2, cfg, fc)
    assert len(files) == 2
    assert _any_bucket_has_full_group(files, g1)
    assert _any_bucket_has_full_group(files, g2)
    all_ctr = [c for f in files for c in flat_counters_in_perfmon_file(f)]
    assert all_ctr.count("SQ_C") == 2


def test_reduce_passes_merges_compatible_buckets():
    cfg = {"SQ": 4}
    b0 = rebuild_counter_file("0", cfg, {"SQ_A", "SQ_B"})
    b1 = rebuild_counter_file("1", cfg, {"SQ_C", "SQ_D"})
    assert b0 is not None and b1 is not None
    files = [b0, b1]
    g1 = frozenset({"SQ_A", "SQ_B"})
    g2 = frozenset({"SQ_C", "SQ_D"})
    files, merges = _reduce_passes(files, [g1, g2], cfg)
    assert merges == 1
    assert len(files) == 1


def test_first_fit_unplaced_adds_missing_counters():
    cfg = {"SQ": 3}
    g1 = frozenset({"SQ_A", "SQ_B"})
    files, fc = _ensure_packable_union([], g1, cfg, 0)
    files, fc = _first_fit_unplaced(files, {"SQ_A", "SQ_B", "SQ_C"}, cfg, fc)
    assert len(files) == 1
    assert set(flat_counters_in_perfmon_file(files[0])) == {
        "SQ_A",
        "SQ_B",
        "SQ_C",
    }


def test_slot_limit_fill_uses_existing_then_opens_new():
    """SLOT_LIMIT set (4 PMCs, cap 2) fills leftover room then opens buckets."""
    from rocprof_compute_soc.counter_grouping_single_pass import (
        fill_slot_limit_into_existing_passes,
    )

    cfg = {"SQ": 2}
    b0 = rebuild_counter_file("0", cfg, {"SQ_A", "SQ_B"})
    assert b0 is not None
    slot = frozenset({"SQ_C", "SQ_D", "SQ_E", "SQ_F"})
    files, _fc, stats = fill_slot_limit_into_existing_passes(
        [b0], [slot], cfg, slot_limit_metric_count=1
    )
    assert stats.passes_before == 1
    assert stats.additional_passes == 2
    assert stats.passes_after == 3
    assert stats.pmc_already_covered == 0
    assert stats.pmc_placed_into_existing == 0
    assert stats.pmc_placed_into_new == 4
    placed = {c for f in files for c in flat_counters_in_perfmon_file(f)}
    assert slot <= placed


def test_slot_limit_fill_zero_extra_when_already_covered():
    from rocprof_compute_soc.counter_grouping_single_pass import (
        fill_slot_limit_into_existing_passes,
    )

    cfg = {"SQ": 2}
    b0 = rebuild_counter_file("0", cfg, {"SQ_A", "SQ_B"})
    b1 = rebuild_counter_file("1", cfg, {"SQ_C", "SQ_D"})
    assert b0 is not None and b1 is not None
    slot = frozenset({"SQ_A", "SQ_B", "SQ_C", "SQ_D"})
    files, _fc, stats = fill_slot_limit_into_existing_passes(
        [b0, b1], [slot], cfg, slot_limit_metric_count=1
    )
    assert stats.additional_passes == 0
    assert stats.passes_after == 2
    assert stats.pmc_already_covered == 4
    assert len(files) == 2
