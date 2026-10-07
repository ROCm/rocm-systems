# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the short-term TCC channel rules."""

import importlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rocprof_compute_soc.counter_file import flat_counters_in_perfmon_file
from rocprof_compute_soc.counter_grouping_buckets import (
    counters_fit_one_bucket,
    rebuild_counter_file,
)
from rocprof_compute_soc.counter_grouping_tcc import (
    adjust_candidate_groups,
    collectable_tcc_channel_count,
    keep_copies_groups_still_need,
    tcc_channel_definition,
)
from tools.counter_grouping_inspector import (
    get_default_config_dir,
    run_soc_detect_and_coalesce,
)
from utils.mi_gpu_spec import mi_gpu_specs


def load_arch_soc(arch, num_xcd):
    """Construct the arch SoC so L2 banks come from that class, not a guess."""
    module = importlib.import_module(f"rocprof_compute_soc.soc_{arch}")
    soc_class = getattr(module, f"{arch}_soc")
    machine_spec = SimpleNamespace(
        rocminfo_lines=None,
        num_xcd=num_xcd,
        l2_banks=1,
        gpu_arch=arch,
    )
    with patch("rocprof_compute_soc.soc_base.console_debug"):
        return soc_class(SimpleNamespace(), machine_spec)


def expanded_channel_count(soc, template):
    expanded = soc._expand_tcc_template_counters({template})
    return len(expanded)


def gfx942_perfmon_config():
    """gfx942's perfmon block sizes. TCC event bases are 4 on this arch."""
    config = dict(mi_gpu_specs.get_perfmon_config("gfx942"))
    assert config["TCC"] == 4
    return config


def channel_bases(bucket):
    return {counter.split("[")[0] for counter in flat_counters_in_perfmon_file(bucket)}


def tcc_channel_bases(bucket):
    return {
        counter.split("[")[0]
        for counter in flat_counters_in_perfmon_file(bucket)
        if counter.startswith("TCC") and counter.endswith("]")
    }


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
    cfg = gfx942_perfmon_config()
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
    cfg = gfx942_perfmon_config()
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
    cfg = gfx942_perfmon_config()
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


def test_gfx942_request_row_stays_one_group_under_its_event_budget():
    soc = load_arch_soc("gfx942", num_xcd=1)
    banks = int(soc._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_EA0_RDREQ[") == banks
    request_row = soc._expand_tcc_template_counters({
        "TCC_EA0_RDREQ[",
        "TCC_EA0_WRREQ[",
        "TCC_EA0_ATOMIC[",
    })
    group = frozenset(request_row)
    assert adjust_candidate_groups(group, set(group)) == [group]
    config = gfx942_perfmon_config()
    assert counters_fit_one_bucket(group, config)
    five_bases = set(group)
    five_bases.update(soc._expand_tcc_template_counters({"TCC_EA0_RDREQ_LEVEL["}))
    five_bases.update(soc._expand_tcc_template_counters({"TCC_EA0_WRREQ_LEVEL["}))
    assert not counters_fit_one_bucket(frozenset(five_bases), config)


def test_gfx908_single_die_uses_its_own_channel_count():
    # A stray multi-die count must not scale MI100 channels.
    soc = load_arch_soc("gfx908", num_xcd=8)
    banks = int(soc._mspec.l2_banks)
    gfx942 = load_arch_soc("gfx942", num_xcd=1)
    assert banks != int(gfx942._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_EA0_RDREQ[") == banks
    assert collectable_tcc_channel_count("gfx908", banks, 8) == banks

    request_row = soc._expand_tcc_template_counters({
        "TCC_EA0_RDREQ[",
        "TCC_EA0_WRREQ[",
        "TCC_EA0_ATOMIC[",
    })
    group = frozenset(request_row)
    assert adjust_candidate_groups(group, set(group)) == [group]
    config = dict(mi_gpu_specs.get_perfmon_config("gfx908"))
    assert counters_fit_one_bucket(group, config)

    level = soc._expand_tcc_template_counters({"TCC_EA0_RDREQ_LEVEL["})
    profile = set(level) | set(soc._expand_tcc_template_counters({"TCC_EA0_RDREQ["}))
    expanded = adjust_candidate_groups(frozenset(level), profile)[0]
    request_channels = {
        counter for counter in expanded if counter.split("[")[0] == "TCC_EA0_RDREQ"
    }
    assert len(request_channels) == banks

    _description, expression = tcc_channel_definition(
        "gfx908", "TCC_EA0_RDREQ", 10, banks
    )
    assert "DIMENSION_XCC" not in expression
    assert "DIMENSION_INSTANCE=[10]" in expression


def test_gfx1151_single_die_without_ea_series_is_a_noop():
    soc = load_arch_soc("gfx1151", num_xcd=8)
    banks = int(soc._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_HIT[") == banks
    assert collectable_tcc_channel_count("gfx1151", banks, 8) == banks

    group = frozenset({"SQ_WAVES", "GRBM_GUI_ACTIVE"})
    assert adjust_candidate_groups(group, set(group)) == [group]
    config = dict(mi_gpu_specs.get_perfmon_config("gfx1151"))
    assert "TCC" not in config
    bucket = rebuild_counter_file("0", config, set(group))
    assert bucket is not None
    files = keep_copies_groups_still_need([bucket], config, [group])
    assert len(files) == 1
    assert set(flat_counters_in_perfmon_file(files[0])) == set(group)

    _description, expression = tcc_channel_definition("gfx1151", "TCC_HIT", 3, banks)
    assert "DIMENSION_XCC" not in expression


def test_gfx90a_channels_follow_its_spec_die_count():
    # gfx90a is multi-die hardware, but the spec reports one die for it.
    # Do not substitute another arch's die count.
    die_count = mi_gpu_specs.get_num_xcds("gfx90a")
    assert die_count == 1
    soc = load_arch_soc("gfx90a", num_xcd=die_count)
    banks = int(soc._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_EA_RDREQ[") == banks * die_count

    level = frozenset({"TCC_EA_RDREQ_LEVEL[0]", "TCC_EA_RDREQ_LEVEL[1]"})
    profile = {
        "TCC_EA_RDREQ_LEVEL[0]",
        "TCC_EA_RDREQ_LEVEL[1]",
        "TCC_EA_RDREQ[0]",
        "TCC_EA_RDREQ[1]",
        "TCC_EA0_RDREQ[0]",
    }
    expanded = adjust_candidate_groups(level, profile)[0]
    assert "TCC_EA_RDREQ[0]" in expanded
    assert "TCC_EA_RDREQ[1]" in expanded
    assert "TCC_EA0_RDREQ[0]" not in expanded

    _description, expression = tcc_channel_definition(
        "gfx90a", "TCC_EA_RDREQ", banks, banks
    )
    assert "DIMENSION_XCC=[1]" in expression
    assert "DIMENSION_INSTANCE=[0]" in expression


def test_gfx942_multi_die_select_uses_its_banks_per_die():
    soc = load_arch_soc("gfx942", num_xcd=2)
    banks = int(soc._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_EA0_RDREQ[") == banks * 2
    _description, expression = tcc_channel_definition(
        "gfx942", "TCC_EA0_RDREQ", banks + 4, banks
    )
    assert "DIMENSION_XCC=[1]" in expression
    assert "DIMENSION_INSTANCE=[4]" in expression


def test_gfx950_channels_follow_its_reported_die_count():
    die_count = mi_gpu_specs.get_num_xcds("gfx950", "mi350", "CPX")
    soc = load_arch_soc("gfx950", num_xcd=die_count)
    banks = int(soc._mspec.l2_banks)
    assert expanded_channel_count(soc, "TCC_EA0_RDREQ[") == banks * die_count
    config = dict(mi_gpu_specs.get_perfmon_config("gfx950"))
    request_row = frozenset(
        soc._expand_tcc_template_counters({
            "TCC_EA0_RDREQ[",
            "TCC_EA0_WRREQ[",
            "TCC_EA0_ATOMIC[",
        })
    )
    assert counters_fit_one_bucket(request_row, config)


def test_gfx942_offline_layout_stays_fourteen_buckets():
    config = gfx942_perfmon_config()
    with tempfile.TemporaryDirectory() as tmp:
        _counters, files = run_soc_detect_and_coalesce(
            "gfx942",
            get_default_config_dir(),
            None,
            config,
            Path(tmp),
        )
    assert len(files) == 14
    by_name = {bucket.name: tcc_channel_bases(bucket) for bucket in files}
    assert by_name["1"] == {
        "TCC_EA0_ATOMIC",
        "TCC_EA0_ATOMIC_LEVEL",
        "TCC_EA0_RDREQ",
        "TCC_EA0_WRREQ",
    }
    assert by_name["2"] == {
        "TCC_EA0_RDREQ",
        "TCC_EA0_RDREQ_LEVEL",
        "TCC_EA0_WRREQ",
        "TCC_EA0_WRREQ_LEVEL",
    }
