# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for counter allocation pipeline in soc_base.py.

Tests LimitedSet, CounterFile, and the bin-packing helpers used by
perfmon_coalesce — no GPU hardware required.
"""

from unittest.mock import MagicMock, patch

import pytest

from rocprof_compute_soc.soc_base import (
    CounterFile,
    LimitedSet,
    OmniSoC_Base,
    _flat_counters_in_perfmon_file,
    _rebuild_tcc_channel_file_map,
    _trial_counter_file_with_extra,
)

# =============================================================================
# Fixtures
# =============================================================================

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


@pytest.fixture
def empty_counter_file(perfmon_config):
    return CounterFile("0", perfmon_config)


def _make_soc(perfmon_config, arch="gfx908", num_xcd=1, l2_banks=4):
    """Build a minimal OmniSoC_Base without rocminfo or GPU access."""
    mspec = MagicMock()
    mspec.rocminfo_lines = None  # skip populate_mspec
    mspec.num_xcd = num_xcd
    mspec.l2_banks = l2_banks

    args = MagicMock()
    args.config_dir = "/dev/null"

    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, mspec)

    soc.set_arch(arch)
    soc.set_perfmon_config(perfmon_config)
    return soc


# =============================================================================
# A. LimitedSet
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
    # Same TCC base — bypasses capacity
    assert ls.add("TCC_HIT[1]") is True
    assert ls.add("TCC_HIT[2]") is True
    # Different TCC base — rejected (no capacity left)
    assert ls.add("TCC_MISS[0]") is False
    assert len(ls.elements) == 3


# =============================================================================
# B. CounterFile
# =============================================================================


def test_counter_file_naming(perfmon_config):
    cf = CounterFile("my_bucket", perfmon_config)
    assert cf.file_name_txt == "pmc_perf_my_bucket.txt"
    assert cf.pmc_filename == "pmc_perf_my_bucket.yaml"
    assert cf.counter_def_filename == "counter_def_my_bucket.yaml"


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


# =============================================================================
# C. _flat_counters_in_perfmon_file
# =============================================================================


def test_flat_counters_in_perfmon_file(perfmon_config):
    # Empty file returns empty list
    cf = CounterFile("0", perfmon_config)
    assert _flat_counters_in_perfmon_file(cf) == []

    # Add counters across blocks and verify flattened order
    cf.add("SQ_WAVES")
    cf.add("TA_ADDR")
    cf.add("TCP_READ")
    result = _flat_counters_in_perfmon_file(cf)
    assert "SQ_WAVES" in result
    assert "TA_ADDR" in result
    assert "TCP_READ" in result
    assert len(result) == 3


# =============================================================================
# D. _trial_counter_file_with_extra
# =============================================================================


def test_trial_counter_file_with_extra_fits(perfmon_config):
    basis = CounterFile("0", perfmon_config)
    basis.add("SQ_WAVES")
    basis.add("TA_ADDR")

    extras = ["TCP_READ", "TCC_HIT[0]"]
    trial = _trial_counter_file_with_extra(basis, perfmon_config, extras)
    assert trial is not None
    flat = _flat_counters_in_perfmon_file(trial)
    assert set(flat) == {"SQ_WAVES", "TA_ADDR", "TCP_READ", "TCC_HIT[0]"}

    # Original basis is unchanged
    assert set(_flat_counters_in_perfmon_file(basis)) == {"SQ_WAVES", "TA_ADDR"}


def test_trial_counter_file_with_extra_overflow(perfmon_config):
    basis = CounterFile("0", perfmon_config)
    # Fill TA to capacity (2)
    basis.add("TA_ADDR")
    basis.add("TA_DATA")

    # Try adding a third TA counter — should fail
    result = _trial_counter_file_with_extra(basis, perfmon_config, ["TA_EXTRA"])
    assert result is None

    # Basis still has only 2 TA counters
    assert len(basis.blocks["TA"].elements) == 2


# =============================================================================
# E. _rebuild_tcc_channel_file_map
# =============================================================================


def test_rebuild_tcc_channel_file_map(perfmon_config):
    bucket_a = CounterFile("a", perfmon_config)
    bucket_a.add("TCC_HIT[0]")
    bucket_a.add("TCC_HIT[1]")
    bucket_a.add("SQ_WAVES")  # non-TCC, should be ignored

    bucket_b = CounterFile("b", perfmon_config)
    bucket_b.add("TCC_MISS[0]")

    result = _rebuild_tcc_channel_file_map([bucket_a, bucket_b])
    assert result["TCC_HIT"] is bucket_a
    assert result["TCC_MISS"] is bucket_b
    assert "SQ" not in result


# =============================================================================
# F. _allocate_perfmon_counter_files
# =============================================================================


def test_allocate_level_counters_get_dedicated_files(perfmon_config):
    soc = _make_soc(perfmon_config)
    counters = {"SQ_LEVEL_WAVES", "TCP_LEVEL_READ", "TA_ADDR"}

    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=()):
        files, file_count, accu_count = soc._allocate_perfmon_counter_files(counters)

    # 2 LEVEL counters → 2 dedicated files with _ACCUM pairs
    level_files = [f for f in files if "LEVEL" in f.file_name_txt]
    assert len(level_files) == 2
    assert accu_count == 2

    for lf in level_files:
        flat = set(_flat_counters_in_perfmon_file(lf))
        # Each LEVEL file has the counter + its _ACCUM pair
        assert any(n.endswith("_ACCUM") for n in flat)

    # TA_ADDR placed somewhere (first-fit into a LEVEL file or its own)
    all_ctrs = set()
    for f in files:
        all_ctrs.update(_flat_counters_in_perfmon_file(f))
    assert "TA_ADDR" in all_ctrs


def test_allocate_first_fit_packing(perfmon_config):
    soc = _make_soc(perfmon_config)
    # 3 SQ counters — all fit in one bucket (SQ capacity 8)
    counters = {"SQ_WAVES", "SQ_BUSY", "SQ_INSTS"}

    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=()):
        files, file_count, accu_count = soc._allocate_perfmon_counter_files(counters)

    assert accu_count == 0
    assert len(files) == 1
    assert file_count == 1
    flat = set(_flat_counters_in_perfmon_file(files[0]))
    assert flat == counters


def test_allocate_tcc_channel_coalescing(perfmon_config):
    soc = _make_soc(perfmon_config)
    # TCC channels with same base should land in the same bucket
    counters = {"TCC_HIT[0]", "TCC_HIT[1]", "TCC_HIT[2]", "SQ_WAVES"}

    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=()):
        files, file_count, accu_count = soc._allocate_perfmon_counter_files(counters)

    # All TCC_HIT channels should be in the same file
    tcc_file = None
    for f in files:
        flat = _flat_counters_in_perfmon_file(f)
        if any("TCC_HIT" in c for c in flat):
            tcc_file = f
            break
    assert tcc_file is not None
    tcc_ctrs = [c for c in _flat_counters_in_perfmon_file(tcc_file) if "TCC_HIT" in c]
    assert set(tcc_ctrs) == {"TCC_HIT[0]", "TCC_HIT[1]", "TCC_HIT[2]"}


# =============================================================================
# G. _expand_tcc_template_counters
# =============================================================================


def test_expand_tcc_templates(perfmon_config):
    soc = _make_soc(perfmon_config, num_xcd=2, l2_banks=3)
    result = soc._expand_tcc_template_counters({"TCC_HIT[", "SQ_WAVES"})

    # Template replaced with 2*3=6 indexed counters
    assert "TCC_HIT[" not in result
    expected_tcc = {f"TCC_HIT[{i}]" for i in range(6)}
    assert expected_tcc.issubset(result)
    assert "SQ_WAVES" in result
    assert len(result) == 7  # 6 TCC + 1 SQ


def test_expand_tcc_no_templates(perfmon_config):
    soc = _make_soc(perfmon_config, num_xcd=1, l2_banks=4)
    inp = {"SQ_WAVES", "TA_ADDR", "TCC_HIT[0]"}
    result = soc._expand_tcc_template_counters(inp)

    # No templates — input unchanged
    assert result == inp
