# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for single-pass-packable packing helpers."""

from rocprof_compute_soc.counter_grouping_buckets import rebuild_counter_file
from rocprof_compute_soc.counter_grouping_single_pass import (
    _any_bucket_has_full_group,
    _bucket_tcc_channel_bases,
    _ensure_packable_union,
    _expand_tcc_ea_affinity_partners,
    _first_fit_unplaced,
    _reduce_passes,
    _split_independent_tcc_ea_req_union,
    _strip_orphan_tcc_ea_req_duplicates,
    fill_slot_limit_into_existing_passes,
    legacy_heuristic_enabled_from_env,
    single_pass_packable_enabled_from_env,
    try_allocate_single_pass_packable,
)
from rocprof_compute_soc.soc_base import flat_counters_in_perfmon_file


class MinimalSoC:
    """Minimum interface required by the single-pass-packable allocator."""

    def _same_bucket_priority_metric_ids(self):
        return ()

    def _iter_arch_analysis_yaml_metrics(self):
        return []


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


def test_split_independent_tcc_ea_req_union_keeps_1805_style_one_group():
    group = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ[1]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_WRREQ[1]",
        "TCC_EA0_ATOMIC[0]",
        "TCC_EA0_ATOMIC[1]",
    })
    assert _split_independent_tcc_ea_req_union(group) == [group]


def test_split_keeps_level_req_affinity_pair_intact():
    group = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ_LEVEL[0]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_WRREQ_LEVEL[0]",
    })
    assert _split_independent_tcc_ea_req_union(group) == [group]


def test_expand_tcc_ea_affinity_partners_adds_matching_req():
    group = frozenset({"TCC_EA0_RDREQ_LEVEL[0]", "TCC_EA0_RDREQ_LEVEL[1]"})
    profile = {
        "TCC_EA0_RDREQ_LEVEL[0]",
        "TCC_EA0_RDREQ_LEVEL[1]",
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_RDREQ[1]",
        "TCC_EA0_WRREQ[0]",
    }
    expanded = _expand_tcc_ea_affinity_partners(group, profile)
    assert "TCC_EA0_RDREQ[0]" in expanded
    assert "TCC_EA0_RDREQ[1]" in expanded
    assert "TCC_EA0_WRREQ[0]" not in expanded


def test_strip_orphan_tcc_ea_req_duplicates_keeps_level_home():
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
    without_request_row = _strip_orphan_tcc_ea_req_duplicates([atomic, latency], cfg)
    assert _bucket_tcc_channel_bases(without_request_row[1]) == level_home

    # Panel 1805 stays complete in the pass that already holds the three
    # request columns, including the extra RDREQ and WRREQ copies.
    request_row = frozenset({
        "TCC_EA0_RDREQ[0]",
        "TCC_EA0_WRREQ[0]",
        "TCC_EA0_ATOMIC[0]",
    })
    files = _strip_orphan_tcc_ea_req_duplicates(
        [atomic, latency],
        cfg,
        [request_row],
    )
    assert _bucket_tcc_channel_bases(files[0]) == {
        "TCC_EA0_ATOMIC",
        "TCC_EA0_ATOMIC_LEVEL",
        "TCC_EA0_RDREQ",
        "TCC_EA0_WRREQ",
    }
    assert _bucket_tcc_channel_bases(files[1]) == level_home
    assert _any_bucket_has_full_group(files, request_row)


def test_strip_orphan_noop_without_level_home():
    cfg = {"TCC": 4, "SQ": 8}
    only_req = rebuild_counter_file(
        "0",
        cfg,
        {"TCC_EA0_RDREQ[0]", "TCC_EA0_WRREQ[0]"},
    )
    assert only_req is not None
    files = _strip_orphan_tcc_ea_req_duplicates([only_req], cfg)
    assert _bucket_tcc_channel_bases(files[0]) == {
        "TCC_EA0_RDREQ",
        "TCC_EA0_WRREQ",
    }


def test_strip_orphan_preserves_required_union_coverage():
    cfg = {"TCC": 4, "SQ": 8}
    home = rebuild_counter_file(
        "0",
        cfg,
        {"TCC_EA0_RDREQ[0]", "TCC_EA0_RDREQ_LEVEL[0]"},
    )
    required = frozenset({"TCC_EA0_RDREQ[0]", "SQ_A"})
    orphan = rebuild_counter_file("1", cfg, set(required))
    assert home is not None and orphan is not None

    files = _strip_orphan_tcc_ea_req_duplicates(
        [home, orphan],
        cfg,
        [required],
    )

    assert _any_bucket_has_full_group(files, required)
    assert "TCC_EA0_RDREQ" in _bucket_tcc_channel_bases(files[1])
