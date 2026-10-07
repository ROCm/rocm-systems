# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for single-pass-packable packing helpers."""

from rocprof_compute_soc.counter_grouping_buckets import rebuild_counter_file
from rocprof_compute_soc.counter_grouping_single_pass import (
    _any_bucket_has_full_group,
    _ensure_packable_union,
    _first_fit_unplaced,
    _reduce_passes,
    collect_unique_packable_unions,
    fill_slot_limit_into_existing_passes,
    legacy_heuristic_enabled_from_env,
    single_pass_packable_enabled_from_env,
    try_allocate_single_pass_packable,
)
from rocprof_compute_soc.soc_base import flat_counters_in_perfmon_file

ITER_METRIC_GROUPS = (
    "rocprof_compute_soc.counter_grouping_single_pass.iter_metric_groups"
)
TCC_BUDGET_CONFIG = {"TCC": 4, "SQ": 8}


class MinimalSoC:
    """Minimum interface required by the single-pass-packable allocator."""

    def _same_bucket_priority_metric_ids(self):
        return ()

    def _iter_arch_analysis_yaml_metrics(self):
        return []


def metric_rows(*groups):
    """iter_metric_groups rows in the given order, one per PMC set."""
    return [
        ((1, -len(group), "x", "", index), frozenset(group), f"metric {index}")
        for index, group in enumerate(groups)
    ]


def channels(base, count=2):
    return {f"{base}[{index}]" for index in range(count)}


def bucket_holding(files, group):
    homes = [
        bucket
        for bucket in files
        if set(group) <= set(flat_counters_in_perfmon_file(bucket))
    ]
    return homes[0] if homes else None


def test_env_gate_default_on(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", raising=False)
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)
    assert single_pass_packable_enabled_from_env() is True
    assert legacy_heuristic_enabled_from_env() is False


def test_legacy_heuristic_env_disables_spp(monkeypatch):
    monkeypatch.setenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", "1")
    assert legacy_heuristic_enabled_from_env() is True
    assert single_pass_packable_enabled_from_env() is False


def test_explicit_spp_off_disables(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", raising=False)
    monkeypatch.setenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", "0")
    assert single_pass_packable_enabled_from_env() is False


def test_allocator_default_places_leftovers_and_clears_work_set(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", raising=False)
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)
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
    assert stats.slot_additional_passes == 0
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
    """When the union of G1 and G2 does not fit, G2 gets its own bucket."""
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
    """SPU set (4 PMCs, cap 2) fills leftover room then opens buckets."""
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


def test_slot_limit_fill_never_reuses_a_surviving_bucket_name():
    cfg = {"SQ": 1}
    b0 = rebuild_counter_file("0", cfg, {"SQ_A"})
    b2 = rebuild_counter_file("2", cfg, {"SQ_B"})
    assert b0 is not None and b2 is not None

    files, file_count, _stats = fill_slot_limit_into_existing_passes(
        [b0, b2],
        [frozenset({"SQ_C"})],
        cfg,
        file_count_start=2,
    )

    assert [bucket.name for bucket in files] == ["0", "2", "3"]
    assert file_count == 4


def test_slot_limit_fill_keeps_tcc_series_channels_together():
    cfg = {"TCC": 1, "SQ": 1}
    full = rebuild_counter_file("0", cfg, {"TCC_OTHER[0]", "SQ_A"})
    assert full is not None
    series = frozenset({"TCC_EA0_RDREQ[0]", "TCC_EA0_RDREQ[1]"})

    files, _file_count, _stats = fill_slot_limit_into_existing_passes(
        [full],
        [series],
        cfg,
    )

    homes = [
        bucket
        for bucket in files
        if series & set(flat_counters_in_perfmon_file(bucket))
    ]
    assert len(homes) == 1
    assert series <= set(flat_counters_in_perfmon_file(homes[0]))


def test_slot_limit_fill_zero_extra_when_already_covered():
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


def test_allocator_integrates_slot_limit_fill(monkeypatch):
    """try_allocate runs SPU fill after packable layout (may add passes)."""
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", raising=False)
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)

    slot = frozenset({"SQ_C", "SQ_D", "SQ_E", "SQ_F"})
    monkeypatch.setattr(
        "rocprof_compute_soc.counter_grouping_single_pass.collect_unique_packable_unions",
        lambda soc, counters, cfg: ([], 0),
    )
    monkeypatch.setattr(
        "rocprof_compute_soc.counter_grouping_single_pass.collect_unique_slot_limit_unions",
        lambda soc, counters, cfg: ([slot], 1),
    )
    # Leave SPU PMCs unplaced so fill must open buckets (gfx942 often +0).
    monkeypatch.setattr(
        "rocprof_compute_soc.counter_grouping_single_pass._first_fit_unplaced",
        lambda files, work_set, cfg, fc: _first_fit_unplaced(
            files, {"SQ_A", "SQ_B"}, cfg, fc
        ),
    )

    work_set = {"SQ_A", "SQ_B", "SQ_C", "SQ_D", "SQ_E", "SQ_F"}
    result = try_allocate_single_pass_packable(
        MinimalSoC(),
        work_set,
        {"SQ": 2},
    )
    assert result is not None
    files, _fc, stats = result
    assert stats.packable_multi_after == 0
    assert stats.slot_limit_metrics == 1
    assert stats.slot_additional_passes >= 1
    placed = {c for f in files for c in flat_counters_in_perfmon_file(f)}
    assert slot <= placed
    assert work_set == set()


def test_ensure_extends_the_bucket_holding_most_of_the_set():
    cfg = {"SQ": 4}
    one_shared = rebuild_counter_file("0", cfg, {"SQ_A"})
    two_shared = rebuild_counter_file("1", cfg, {"SQ_B", "SQ_C"})
    assert one_shared is not None and two_shared is not None
    group = frozenset({"SQ_A", "SQ_B", "SQ_C"})

    files, file_count = _ensure_packable_union([one_shared, two_shared], group, cfg, 2)

    assert file_count == 2
    assert len(files) == 2
    assert set(flat_counters_in_perfmon_file(files[0])) == {"SQ_A"}
    assert set(flat_counters_in_perfmon_file(files[1])) == set(group)


def test_ensure_rejects_a_bucket_with_one_block_over_budget():
    cfg = {"SQ": 2, "GRBM": 2}
    sq_full = rebuild_counter_file("0", cfg, {"SQ_A", "SQ_B"})
    assert sq_full is not None
    group = frozenset({"GRBM_COUNT", "SQ_C"})

    files, file_count = _ensure_packable_union([sq_full], group, cfg, 1)

    assert file_count == 2
    assert len(files) == 2
    assert set(flat_counters_in_perfmon_file(files[0])) == {"SQ_A", "SQ_B"}
    assert set(flat_counters_in_perfmon_file(files[1])) == set(group)


def test_ensure_keeps_bucket_count_when_an_existing_bucket_fits():
    cfg = {"SQ": 4, "GRBM": 2}
    room = rebuild_counter_file("0", cfg, {"SQ_A"})
    assert room is not None
    group = frozenset({"GRBM_COUNT", "SQ_B"})

    files, file_count = _ensure_packable_union([room], group, cfg, 1)

    assert file_count == 1
    assert len(files) == 1
    assert set(group) <= set(flat_counters_in_perfmon_file(files[0]))


def test_collect_orders_larger_sets_first(monkeypatch):
    small = {"SQ_A"}
    large = {"SQ_B", "SQ_C", "SQ_D"}
    middle = {"SQ_E", "SQ_F"}
    monkeypatch.setattr(
        ITER_METRIC_GROUPS,
        lambda soc, counters: metric_rows(small, large, middle),
    )

    groups, packable_count = collect_unique_packable_unions(
        MinimalSoC(), small | large | middle, {"SQ": 4}
    )

    assert packable_count == 3
    assert [len(group) for group in groups] == [3, 2, 1]


def test_collect_leaves_sets_without_tcc_series_unchanged(monkeypatch):
    sets = [{"SQ_A", "GRBM_COUNT"}, {"SQ_B"}]
    monkeypatch.setattr(ITER_METRIC_GROUPS, lambda soc, counters: metric_rows(*sets))
    profile = {"SQ_A", "SQ_B", "GRBM_COUNT", "TCC_EA0_RDREQ[0]"}

    groups, _count = collect_unique_packable_unions(
        MinimalSoC(), profile, {"SQ": 4, "GRBM": 2, "TCC": 4}
    )

    assert set(groups) == {frozenset(group) for group in sets}


def test_collect_adds_request_channels_to_a_level_set(monkeypatch):
    level = channels("TCC_EA0_RDREQ_LEVEL")
    request = channels("TCC_EA0_RDREQ")
    monkeypatch.setattr(ITER_METRIC_GROUPS, lambda soc, counters: metric_rows(level))

    groups, _count = collect_unique_packable_unions(
        MinimalSoC(), level | request, TCC_BUDGET_CONFIG
    )

    assert groups == [frozenset(level | request)]


def test_allocator_keeps_tcc_request_row_and_level_pairs(monkeypatch):
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", raising=False)
    monkeypatch.delenv("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", raising=False)
    read = channels("TCC_EA0_RDREQ")
    write = channels("TCC_EA0_WRREQ")
    atomic = channels("TCC_EA0_ATOMIC")
    read_level = channels("TCC_EA0_RDREQ_LEVEL")
    write_level = channels("TCC_EA0_WRREQ_LEVEL")
    atomic_level = channels("TCC_EA0_ATOMIC_LEVEL")
    request_row = read | write | atomic
    latency_sets = [read_level | read, write_level | write, atomic_level | atomic]
    monkeypatch.setattr(
        ITER_METRIC_GROUPS,
        lambda soc, counters: metric_rows(request_row, *latency_sets),
    )
    work_set = request_row | read_level | write_level | atomic_level

    result = try_allocate_single_pass_packable(
        MinimalSoC(), work_set, TCC_BUDGET_CONFIG
    )

    assert result is not None
    files, _file_count, stats = result
    assert stats.packable_multi_after == 0
    assert len(files) == 2
    row_home = bucket_holding(files, request_row)
    assert row_home is not None
    for latency in latency_sets:
        assert bucket_holding(files, latency) is not None
    read_home = bucket_holding(files, read_level | read)
    assert read_home is not row_home
