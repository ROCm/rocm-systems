# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for counter allocation pipeline in soc_base.py.

Tests LimitedSet, CounterFile, and the bin-packing helpers used by
perfmon_coalesce — no GPU hardware required.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rocprof_compute_soc.soc_base import (
    CounterFile,
    LimitedSet,
    OmniSoC_Base,
    _rebuild_tcc_channel_file_map,
    _trial_counter_file_with_extra,
    flat_counters_in_perfmon_file,
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

# One unique synthetic counter per metric table, so the counter set returned by
# detect_counters() reveals exactly which tables were selected.
BASELINE_COUNTER = "SQ_BASELINE_COUNTER"  # table 201, outside block 30
TABLE_3012_COUNTER = "TCC_BOTTLENECK_COUNTER"  # block 30, table 3012
TABLE_3013_COUNTER = "TCC_EA_COUNTER"  # block 30, table 3013
FIXTURE_COUNTERS = {BASELINE_COUNTER, TABLE_3012_COUNTER, TABLE_3013_COUNTER}


@pytest.fixture
def perfmon_config():
    return dict(PERFMON_CONFIG)


@pytest.fixture
def empty_counter_file(perfmon_config):
    return CounterFile("0", perfmon_config)


@pytest.fixture
def membw_analysis_soc(tmp_path: Path) -> OmniSoC_Base:
    baseline_analysis_config = f"""\
Panel Config:
  id: 200
  data source:
  - metric_table:
      id: 201
      metric:
        Baseline:
          value: SUM({BASELINE_COUNTER})
"""
    membw_analysis_config = f"""\
Panel Config:
  id: 3000
  data source:
  - metric_table:
      id: 3012
      metric:
        L2 Bottleneck Detection Indicators:
          value: SUM({TABLE_3012_COUNTER})
  - metric_table:
      id: 3013
      metric:
        EA Interface:
          value: SUM({TABLE_3013_COUNTER})
"""
    config_root = tmp_path / "gfx950"
    config_root.mkdir()
    (config_root / "0200_baseline.yaml").write_text(
        baseline_analysis_config,
        encoding="utf-8",
    )
    (config_root / "3000_mem_bw.yaml").write_text(
        membw_analysis_config,
        encoding="utf-8",
    )

    args = SimpleNamespace(
        config_dir=tmp_path,
        filter_blocks=[],
        membw_analysis=False,
        roof_only=False,
        set_selected=None,
    )
    machine_specs = SimpleNamespace(
        gpu_arch="gfx950",
        gpu_series="MI350",
        l2_banks=1,
        num_xcd=1,
        rocminfo_lines=None,
    )
    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch("gfx950")
    return soc


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
# B. CounterFile
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
# C. flat_counters_in_perfmon_file
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


# =============================================================================
# D. _trial_counter_file_with_extra
# =============================================================================


def test_trial_counter_file_with_extra_fits(perfmon_config):
    basis = CounterFile("0", perfmon_config)
    basis.add("SQ_WAVES")
    basis.add("TA_ADDR")
    # Paired level-event slot, as held by an accumulator bucket.
    basis.reserve("SQ_WAVES", 1)

    extras = ["TCP_READ", "TCC_HIT[0]"]
    trial = _trial_counter_file_with_extra(basis, perfmon_config, extras)
    assert trial is not None
    flat = flat_counters_in_perfmon_file(trial)
    assert set(flat) == {"SQ_WAVES", "TA_ADDR", "TCP_READ", "TCC_HIT[0]"}

    # Reservations survive the clone, so the trial cannot spend a held slot.
    assert trial.blocks["SQ"].avail == basis.blocks["SQ"].avail

    # Original basis is unchanged
    assert set(flat_counters_in_perfmon_file(basis)) == {"SQ_WAVES", "TA_ADDR"}
    assert basis.blocks["SQ"].avail == perfmon_config["SQ"] - 2


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
    counters = {
        "SQ_LEVEL_WAVES_ACCUM",
        "SQC_DCACHE_INFLIGHT_LEVEL_ACCUM",
        "TA_ADDR",
    }

    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=()):
        files, file_count, accu_count = soc._allocate_perfmon_counter_files(counters)

    accum_files = [f for f in files if f.name.endswith("_ACCUM")]
    assert accu_count == 2
    assert len(accum_files) == 2
    assert {f.name for f in accum_files} == {
        "SQ_LEVEL_WAVES_ACCUM",
        "SQC_DCACHE_INFLIGHT_LEVEL_ACCUM",
    }

    for af in accum_files:
        assert af.name in set(flat_counters_in_perfmon_file(af))

    # Both accumulators land in the SQ block (SQC routes via BLOCK_REMAP). Each
    # file loses 2 SQ slots: 1 for add() + 1 for reserve(counter, 1) (the
    # paired level event the hardware programs alongside the accumulator).
    sq_waves_accum = next(f for f in accum_files if f.name == "SQ_LEVEL_WAVES_ACCUM")
    assert sq_waves_accum.blocks["SQ"].avail == perfmon_config["SQ"] - 2
    sqc_dcache_accum = next(
        f for f in accum_files if f.name == "SQC_DCACHE_INFLIGHT_LEVEL_ACCUM"
    )
    assert sqc_dcache_accum.blocks["SQ"].avail == perfmon_config["SQ"] - 2

    # TA_ADDR placed somewhere (first-fit into a LEVEL file or its own)
    all_ctrs = set()
    for f in files:
        all_ctrs.update(flat_counters_in_perfmon_file(f))
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
    flat = set(flat_counters_in_perfmon_file(files[0]))
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
        flat = flat_counters_in_perfmon_file(f)
        if any("TCC_HIT" in c for c in flat):
            tcc_file = f
            break
    assert tcc_file is not None
    tcc_ctrs = [c for c in flat_counters_in_perfmon_file(tcc_file) if "TCC_HIT" in c]
    assert set(tcc_ctrs) == {"TCC_HIT[0]", "TCC_HIT[1]", "TCC_HIT[2]"}


def test_metric_aware_coalesce_packs_regular_counters_into_accum_bucket(perfmon_config):
    """Accumulator buckets should accept other counters during metric-aware pass."""
    soc = _make_soc(perfmon_config)
    soc._profiling_metric_keys = {("9999", 9999, 0)}
    counters = {"SQ_LEVEL_WAVES_ACCUM", "GRBM_SPI_BUSY", "GRBM_GUI_ACTIVE"}

    yaml_text = """
avg: MAX(GRBM_SPI_BUSY / GRBM_GUI_ACTIVE)
"""
    with patch.object(soc, "_same_bucket_priority_metric_ids", return_value=("9999.9999.0",)):
        with patch.object(
            soc,
            "_iter_arch_analysis_yaml_metrics",
            return_value=iter([
                ("9999", 9999, 0, "WGM Utilization", yaml_text),
            ]),
        ):
            files, _, accu_count = soc._allocate_perfmon_counter_files(counters)

    assert accu_count == 1
    accum_files = [f for f in files if f.name.endswith("_ACCUM")]
    assert len(accum_files) == 1
    packed = set(flat_counters_in_perfmon_file(accum_files[0]))
    assert {"SQ_LEVEL_WAVES_ACCUM", "GRBM_SPI_BUSY", "GRBM_GUI_ACTIVE"}.issubset(packed)


def test_gfx1250_cp_utilization_metric_id_keeps_ratio_partners_together(tmp_path: Path):
    """Regression for PR #9324 review: 17.1.1 partners must share one bucket."""
    from utils.mi_gpu_spec import mi_gpu_specs

    arch = "gfx1250"
    config_root = tmp_path / arch
    config_root.mkdir()
    metric_yaml = """
avg: 100 * SUM(GRBM_CP_BUSY_sum) / SUM(GRBM_GUI_ACTIVE_sum)
min: 100 * MIN(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)
max: 100 * MAX(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)
"""
    metric_body = "\n".join(
        "          " + line if line.strip() else line
        for line in metric_yaml.strip().splitlines()
    )
    (config_root / "1700_grbm.yaml").write_text(
        f"""\
Panel Config:
  id: 1700
  data source:
  - metric_table:
      id: 1701
      metric:
        GPU Active Cycles:
          avg: AVG(GRBM_GUI_ACTIVE_sum)
        CP Utilization:
{metric_body}
""",
        encoding="utf-8",
    )

    args = SimpleNamespace(
        config_dir=tmp_path,
        filter_blocks=["17.1.1"],
        membw_analysis=False,
        roof_only=False,
        set_selected=None,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=arch,
        gpu_series=mi_gpu_specs.get_gpu_series(arch),
        l2_banks=4,
        num_xcd=1,
        rocminfo_lines=None,
    )
    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(arch)
    soc.set_perfmon_config(mi_gpu_specs.get_perfmon_config(arch))

    counters, _ = soc.detect_counters()
    assert {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}.issubset(counters)

    with patch.object(
        soc,
        "_same_bucket_priority_metric_ids",
        return_value=("17.1.1",),
    ):
        files, _, _ = soc._allocate_perfmon_counter_files(counters)

    counter_to_bucket: dict[str, str] = {}
    for counter_file in files:
        label = counter_file.name.replace(".txt", "")
        for ctr in flat_counters_in_perfmon_file(counter_file):
            counter_to_bucket[ctr] = label

    assert (
        counter_to_bucket["GRBM_CP_BUSY_sum"]
        == counter_to_bucket["GRBM_GUI_ACTIVE_sum"]
    )


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


# =============================================================================
# H. _append_analysis_yaml_for_filter_token alias handling
# =============================================================================


def _fake_soc_for_filter_token(arch: str = "gfx942") -> Any:
    """Minimal stand-in exposing only the _mspec.gpu_arch the method reads."""
    return SimpleNamespace(_mspec=SimpleNamespace(gpu_arch=arch))


def test_filter_token_unknown_alias_exits_instead_of_keyerror(monkeypatch):
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.get_arch_alias_to_panel_id",
        lambda arch: {"lds": "10"},
    )
    with pytest.raises(SystemExit):
        OmniSoC_Base._append_analysis_yaml_for_filter_token(
            _fake_soc_for_filter_token(), "SQ", {}, "/cfg", []
        )


def test_filter_token_known_alias_resolves_without_crash(monkeypatch):
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.get_arch_alias_to_panel_id",
        lambda arch: {"lds": "10"},
    )
    texts: list[str] = []
    # Alias resolves to block id 10 -> file id "1000", absent from the
    # empty config dict, so the token is skipped with a warning, not a crash.
    OmniSoC_Base._append_analysis_yaml_for_filter_token(
        _fake_soc_for_filter_token(), "lds", {}, "/cfg", texts
    )
    assert texts == []


# =============================================================================
# I. Memory Bandwidth Analysis counter selection
# =============================================================================


@pytest.mark.parametrize(
    ("membw_analysis", "filter_blocks", "expected_counters"),
    [
        pytest.param(
            False,
            [],
            {BASELINE_COUNTER},
            id="flag_off_drops_the_whole_block_30_file",
        ),
        pytest.param(
            True,
            [],
            {BASELINE_COUNTER, TABLE_3012_COUNTER, TABLE_3013_COUNTER},
            id="flag_on_no_filter_keeps_every_table",
        ),
        pytest.param(
            True,
            ["30"],
            {TABLE_3012_COUNTER, TABLE_3013_COUNTER},
            id="block_30_keeps_both_block_30_tables_and_drops_baseline",
        ),
        pytest.param(
            True,
            ["30.12"],
            {TABLE_3012_COUNTER},
            id="block_30_12_keeps_only_table_3012",
        ),
        pytest.param(
            True,
            ["2", "30.13"],
            {BASELINE_COUNTER, TABLE_3013_COUNTER},
            id="ordinary_block_and_membw_table_are_combined",
        ),
    ],
)
def test_membw_analysis_counter_selection(
    membw_analysis_soc: OmniSoC_Base,
    membw_analysis: bool,
    filter_blocks: list[str],
    expected_counters: set[str],
) -> None:
    """--membw-analysis admits block 30; --block then narrows within it.

    Each config table in the fixture owns one unique synthetic counter, so the
    selected fixture counters identify exactly which tables survived. The 30.12
    and mixed 30.13 cases prove selection is by table id; the mixed case also
    proves a memory-bandwidth table composes with an ordinary report block.
    """
    args = membw_analysis_soc.get_args()
    args.membw_analysis = membw_analysis
    args.filter_blocks = filter_blocks

    counters, effective_filter_blocks = membw_analysis_soc.detect_counters()

    assert counters & FIXTURE_COUNTERS == expected_counters
    assert effective_filter_blocks == filter_blocks
