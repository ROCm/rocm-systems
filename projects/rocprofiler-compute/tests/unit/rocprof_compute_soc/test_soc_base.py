# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for counter allocation pipeline in soc_base.py.

Tests LimitedSet, CounterFile, and the bin-packing helpers used by
perfmon_coalesce. No GPU hardware required.
"""

import os
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, NoReturn, Optional, Tuple
from unittest.mock import MagicMock, patch

import common
import pytest
import yaml

import config
from rocprof_compute_soc import soc_base as soc_base_module
from rocprof_compute_soc.soc_base import (
    AnalysisConfigMetric,
    CounterFile,
    LimitedSet,
    OmniSoC_Base,
    _rebuild_tcc_channel_file_map,
    _trial_counter_file_with_extra,
    flat_counters_in_perfmon_file,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.specs import MachineSpecs
from utils.utils_common import (
    MEMBW_ANALYSIS_PANEL_ID,
    canonical_config_arch,
    convert_metric_id_to_panel_info,
)
from utils.utils_counter_defs import extract_counters_and_variables

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

GFX1250_PERFMON_CONFIG = {
    "GRBM": 2,
    "SQ": 8,
    "SPI": 6,
}

# CP Utilization (metric id 17.1.1): a GRBM ratio whose numerator and
# denominator must land in the same perfmon pass.
CP_UTIL_METRIC_EXPRESSIONS = [
    "100 * SUM(GRBM_CP_BUSY_sum) / SUM(GRBM_GUI_ACTIVE_sum)",
    "100 * MIN(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)",
    "100 * MAX(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)",
]

# One unique synthetic counter per metric table, so the counter set returned by
# detect_counters() reveals exactly which tables were selected.
BASELINE_COUNTER = "SQ_BASELINE_COUNTER"  # table 201, outside block 30
TABLE_3012_COUNTER = "TCC_BOTTLENECK_COUNTER"  # block 30, table 3012
TABLE_3013_COUNTER = "TCC_EA_COUNTER"  # block 30, table 3013
FIXTURE_COUNTERS = {BASELINE_COUNTER, TABLE_3012_COUNTER, TABLE_3013_COUNTER}

HBM_TRAFFIC_METRIC_NAMES = {"HBM Read Traffic", "HBM Write and Atomic Traffic"}

# Every counter name outside the expressions is unique, so any leak from a
# comment, title, description or metric name shows up in the result.
PROSE_ANALYSIS_CONFIG = """\
# SQ_COMMENT_COUNTER in a comment
Panel Config:
  id: 200
  title: Speed of Light
  data source:
  - metric_table:
      id: 201
      title: SQ_TITLE_COUNTER table
      header:
        metric: Metric
        avg: Avg
        unit: Unit
      metric:
        SQ_METRIC_NAME_COUNTER:
          avg: SUM(SQ_LEVEL_WAVES_ACCUM) / SUM(SQ_BUSY_CYCLES)  # SQ_TRAILING_COUNTER
          unit: Wavefronts
  metrics_description:
    SQ_METRIC_NAME_COUNTER: Uses SQ_DESCRIPTION_COUNTER.
"""

# One arch per analysis config directory (gfx115x parts share one).
CONFIG_ARCHS = sorted(
    {
        canonical_config_arch(arch): arch for arch in mi_gpu_specs.get_gpu_series_dict()
    }.values()
)


def offline_soc(
    gpu_arch: str,
    filter_blocks: List[str],
    membw_analysis: bool,
    set_selected: Optional[str] = None,
) -> OmniSoC_Base:
    """OmniSoC_Base that detects counters from the analysis configs without a GPU.
    With one XCD and one L2 bank, each TCC channel template expands to channel 0
    only."""
    args = SimpleNamespace(
        filter_blocks=filter_blocks,
        membw_analysis=membw_analysis,
        set_selected=set_selected,
    )
    machine_specs = SimpleNamespace(
        gpu_arch=gpu_arch,
        gpu_series=mi_gpu_specs.get_gpu_series(gpu_arch),
        l2_banks=1,
        num_xcd=1,
        rocminfo_lines=None,
    )
    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch(gpu_arch)
    return soc


@pytest.fixture
def perfmon_config():
    return dict(PERFMON_CONFIG)


@pytest.fixture
def gfx1250_perfmon_config():
    return dict(GFX1250_PERFMON_CONFIG)


@pytest.fixture
def membw_analysis_soc(tmp_path: Path, monkeypatch) -> OmniSoC_Base:
    baseline_analysis_config = f"""\
Panel Config:
  id: 200
  data source:
  - metric_table:
      id: 201
      header:
        metric: Metric
        value: Value
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
      header:
        metric: Metric
        value: Value
      metric:
        L2 Bottleneck Detection Indicators:
          value: SUM({TABLE_3012_COUNTER})
  - metric_table:
      id: 3013
      header:
        metric: Metric
        value: Value
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

    monkeypatch.setattr(config, "analysis_configs_dir", tmp_path)
    return offline_soc("gfx950", [], membw_analysis=False)


def _make_soc(perfmon_config, arch="gfx908", num_xcd=1, l2_banks=4):
    """Build a minimal OmniSoC_Base without rocminfo or GPU access."""
    mspec = MagicMock()
    mspec.rocminfo_lines = None  # skip populate_mspec
    mspec.num_xcd = num_xcd
    mspec.l2_banks = l2_banks

    args = MagicMock()

    with patch("rocprof_compute_soc.soc_base.console_debug"):
        soc = OmniSoC_Base(args, mspec)

    soc.set_arch(arch)
    soc.set_perfmon_config(perfmon_config)
    return soc


@pytest.fixture
def gfx1250_soc(gfx1250_perfmon_config) -> OmniSoC_Base:
    """Minimal gfx1250 OmniSoC for metric-aware coalesce tests."""
    soc = _make_soc(gfx1250_perfmon_config, arch="gfx1250")
    soc._mspec.gpu_series = "GFX1250_SERIES"
    return soc


def apply_cp_util_priority_patches(soc: OmniSoC_Base, patch_stack: ExitStack) -> None:
    """Make CP Utilization the only same-bucket priority metric on soc."""
    patch_stack.enter_context(
        patch.object(soc, "_same_bucket_priority_metric_ids", return_value=("17.1.1",))
    )
    patch_stack.enter_context(
        patch.object(
            soc,
            "_iter_arch_analysis_yaml_metrics",
            return_value=iter([
                AnalysisConfigMetric(
                    "1700",
                    1701,
                    1,
                    "CP Utilization",
                    CP_UTIL_METRIC_EXPRESSIONS,
                    table={},
                    metric={},
                )
            ]),
        )
    )


def _resolve_metric_name(config_dir: Path, arch: str, metric_id: str) -> str | None:
    """Look up the metric name for *metric_id* in the analysis YAML tree."""
    file_id, panel_id, metric_idx = convert_metric_id_to_panel_info(metric_id)
    arch_dir = config_dir / (canonical_config_arch(arch) or arch)
    for ypath in sorted(arch_dir.glob("*.yaml")):
        if not ypath.name.startswith(file_id):
            continue
        doc = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            continue
        sources = doc.get("Panel Config", {}).get("data source", [])
        for src in sources:
            mt = src.get("metric_table", {})
            if mt.get("id") != panel_id:
                continue
            metrics = mt.get("metric", {})
            for idx, name in enumerate(metrics):
                if idx == metric_idx:
                    return name
    return None


# Supported-counter discovery


def fail_unexpected_device_query() -> NoReturn:
    raise AssertionError("SoC queried a device after selected metadata was supplied")


def fail_counter_enumeration() -> dict:
    raise RuntimeError("SDK enumeration failed")


@pytest.fixture
def counter_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """Counter discovery with SDK boundaries replaced and real temporary files."""
    source_root = tmp_path / "src"
    sdk_config = source_root / "rocprof_compute_soc/profile_configs/sdk_config.yaml"
    sdk_config.parent.mkdir(parents=True)
    sdk_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config, "rocprof_compute_home", source_root)

    metrics_path = tmp_path / "counter-definitions"
    metrics_path.mkdir()
    monkeypatch.setattr(
        soc_base_module,
        "create_temp_rocprofiler_metrics_path",
        lambda _config: str(metrics_path),
    )
    availability = SimpleNamespace(counters={}, agents={})
    avail_module = SimpleNamespace(
        get_counters=lambda: availability.counters,
        get_agent_info_map=lambda: availability.agents,
    )
    monkeypatch.setattr(
        soc_base_module.rocprofv3_avail_interface,
        "_load_avail_module",
        lambda _tool_path: avail_module,
    )
    warnings = []
    monkeypatch.setattr(soc_base_module, "console_warning", warnings.append)
    args = SimpleNamespace(
        rocprofiler_sdk_tool_path="sdk-tool.so",
        device=7,
        config_dir=str(tmp_path / "analysis-configs"),
        membw_analysis=False,
    )
    machine_specs = SimpleNamespace(rocminfo_lines=None, gpu_chip_id="5510")
    soc = OmniSoC_Base(args, machine_specs)
    soc.set_arch("gfx1151")
    monkeypatch.setenv("ROCPROFILER_METRICS_PATH", "original-metrics")
    return SimpleNamespace(
        soc=soc,
        availability=availability,
        avail_module=avail_module,
        metrics_path=metrics_path,
        warnings=warnings,
    )


def test_populate_mspec_preserves_selected_device_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine_specs = MachineSpecs(
        gpu_arch="gfx1151",
        gpu_chip_id="5510",
        rocminfo_lines=["  Max Clock Freq. (MHz): 2520"],
        max_mclk="2500",
        gpu_model_names=("Radeon 8060S", "Strix Halo", ""),
    )
    monkeypatch.setattr(
        soc_base_module.amdsmi_interface, "amdsmi_ctx", fail_unexpected_device_query
    )

    OmniSoC_Base(SimpleNamespace(), machine_specs)

    assert machine_specs.max_mclk == "2500"
    assert machine_specs.cur_mclk == "2500"
    assert machine_specs.max_sclk == "2520"
    assert machine_specs.gpu_model == "RDNA35_HALO"


@pytest.mark.parametrize(
    "gpu_arch, model_names, expected_model",
    [
        ("gfx908", ("AMD Instinct MI100", "", ""), "MI100"),
        ("gfx90a", ("AMD Instinct MI210", "", ""), "MI210"),
        ("gfx1151", ("", "", ""), None),
    ],
)
def test_populate_mspec_uses_selected_device_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
    gpu_arch: str,
    model_names: Tuple[str, str, str],
    expected_model: Optional[str],
) -> None:
    machine_specs = MachineSpecs(
        gpu_arch=gpu_arch,
        gpu_chip_id="unknown",
        rocminfo_lines=[],
        max_mclk="1000",
        gpu_model_names=model_names,
    )
    monkeypatch.setattr(mi_gpu_specs, "get_gpu_model", lambda *_: None)
    monkeypatch.setattr(
        soc_base_module.amdsmi_interface, "amdsmi_ctx", fail_unexpected_device_query
    )

    OmniSoC_Base(SimpleNamespace(), machine_specs)

    assert machine_specs.gpu_model == expected_model
    assert machine_specs.cur_mclk == "1000"


def test_populate_mspec_queries_missing_legacy_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine_specs = MachineSpecs(
        gpu_arch="gfx908", gpu_chip_id="unknown", rocminfo_lines=[]
    )
    monkeypatch.setattr(mi_gpu_specs, "get_gpu_model", lambda *_: None)
    monkeypatch.setattr(soc_base_module.amdsmi_interface, "amdsmi_ctx", nullcontext)
    monkeypatch.setattr(
        soc_base_module.amdsmi_interface, "get_mem_max_clock", lambda: 1000
    )
    monkeypatch.setattr(
        soc_base_module.amdsmi_interface,
        "get_gpu_model",
        lambda: ("AMD Instinct MI100", "", ""),
    )

    OmniSoC_Base(SimpleNamespace(), machine_specs)

    assert machine_specs.gpu_model == "MI100"
    assert machine_specs.max_mclk == "1000"
    assert machine_specs.cur_mclk == "1000"


@pytest.mark.parametrize(
    "counter_names, agent_info, chip_id, expected, unknown_support",
    [
        pytest.param(
            {31134: [], 31133: ["SQ_WAVES"]},
            {
                31134: {"name": "gfx1036", "device_id": 5056, "gpu_index": 1},
                31133: {"name": "gfx1151", "device_id": 5510, "gpu_index": 0},
            },
            "5510",
            {"SQ_WAVES"},
            False,
            id="unsupported-gpu-first",
        ),
        pytest.param(
            {42: ["OTHER_ARCH_COUNTER"], 91: ["SQ_WAVES"]},
            {
                42: {"name": "gfx950", "device_id": 30112},
                91: {"name": "gfx1151", "device_id": 5510},
            },
            "5510",
            {"SQ_WAVES"},
            False,
            id="heterogeneous-gpus",
        ),
        pytest.param(
            {42: ["OTHER_CHIP_COUNTER"], 91: ["SQ_WAVES"]},
            {
                42: {"name": "gfx1151", "device_id": 5511},
                91: {"name": "gfx1151", "device_id": 5510},
            },
            "5510",
            {"SQ_WAVES"},
            False,
            id="match-chip-within-architecture",
        ),
        pytest.param(
            {42: ["SQ_WAVES", "ONE_GPU_ONLY"], 91: ["SQ_WAVES"]},
            {
                42: {"name": "gfx1151", "device_id": 5510},
                91: {"name": "gfx1151", "device_id": 5510},
            },
            "5510",
            {"SQ_WAVES"},
            False,
            id="duplicate-matches-intersect",
        ),
        pytest.param(
            {42: ["SQ_WAVES"], 91: []},
            {
                42: {"name": "gfx1151", "device_id": 5510},
                91: {"name": "gfx1151", "device_id": 5510},
            },
            "5510",
            set(),
            False,
            id="empty-duplicate-is-not-ignored",
        ),
        pytest.param(
            {42: ["SQ_WAVES", "ONE_CHIP_ONLY"], 91: ["SQ_WAVES"]},
            {
                42: {"name": "gfx1151", "device_id": 5511},
                91: {"name": "gfx1151", "device_id": 5510},
            },
            None,
            {"SQ_WAVES"},
            False,
            id="unknown-chip-intersects-architecture",
        ),
        pytest.param({}, {}, "5510", None, True, id="no-agent-listings"),
        pytest.param(
            {91: ["SQ_WAVES"]},
            {},
            "5510",
            None,
            True,
            id="missing-metadata",
        ),
        pytest.param(
            {91: ["SQ_WAVES"]},
            {91: {"name": "gfx1151"}},
            "5510",
            None,
            True,
            id="missing-chip-metadata",
        ),
    ],
)
def test_supported_counters_match_gpu_identity(
    counter_discovery, counter_names, agent_info, chip_id, expected, unknown_support
) -> None:
    counter_discovery.soc._mspec.gpu_chip_id = chip_id
    counter_discovery.availability.counters = {
        handle: [SimpleNamespace(name=name, block="SQ") for name in names]
        for handle, names in counter_names.items()
    }
    counter_discovery.availability.agents = agent_info

    assert counter_discovery.soc.get_rocprof_supported_counters() == expected
    assert bool(counter_discovery.warnings) is unknown_support
    if unknown_support:
        assert "counter support is unknown" in counter_discovery.warnings[0]
    assert os.environ["ROCPROFILER_METRICS_PATH"] == "original-metrics"
    assert not counter_discovery.metrics_path.exists()


def test_supported_counters_include_raw_and_derived_but_exclude_constants(
    counter_discovery,
) -> None:
    counter_discovery.availability.counters = {
        91: [
            SimpleNamespace(name="SQ_WAVES", block="SQ"),
            SimpleNamespace(name="CUSTOM_SUM", expression="reduce(SQ_WAVES,sum)"),
            SimpleNamespace(name="CU_COUNT"),
        ]
    }
    counter_discovery.availability.agents = {91: {"name": "gfx1151", "device_id": 5510}}

    assert counter_discovery.soc.get_rocprof_supported_counters() == {
        "SQ_WAVES",
        "CUSTOM_SUM",
    }


def test_supported_counter_discovery_uses_selected_machine_node(counter_discovery):
    counter_discovery.soc._mspec.gpu_node_id = 2
    counter_discovery.availability.counters = {
        91: [SimpleNamespace(name="OTHER_GPU_ONLY", block="SQ")],
        42: [SimpleNamespace(name="SQ_WAVES", block="SQ")],
    }
    counter_discovery.availability.agents = {
        91: {"name": "gfx1151", "device_id": 5510, "node_id": 1},
        42: {"name": "gfx1151", "device_id": 5510, "node_id": 2},
    }
    assert counter_discovery.soc.get_rocprof_supported_counters() == {"SQ_WAVES"}
    assert not counter_discovery.warnings
    assert os.environ["ROCPROFILER_METRICS_PATH"] == "original-metrics"
    assert not counter_discovery.metrics_path.exists()


@pytest.mark.parametrize(
    "supported, should_warn",
    [
        pytest.param(None, False, id="support-unknown"),
        pytest.param(set(), True, id="known-empty-support"),
        pytest.param({"SQ_WAVES"}, False, id="requested-counter-supported"),
    ],
)
def test_perfmon_coalesce_reports_missing_counters_only_when_support_is_known(
    counter_discovery, monkeypatch, tmp_path, supported, should_warn
):
    soc = counter_discovery.soc
    soc.get_args().output_directory = str(tmp_path)
    soc.set_perfmon_config(PERFMON_CONFIG)
    monkeypatch.setattr(soc, "get_rocprof_supported_counters", lambda: supported)
    soc.perfmon_coalesce({"SQ_WAVES"})
    assert bool(counter_discovery.warnings) is should_warn
    if should_warn:
        assert "SQ_WAVES" in counter_discovery.warnings[0]


@pytest.mark.parametrize("query", ["get_counters", "get_agent_info_map"])
@pytest.mark.parametrize("original_path", [None, "original-metrics", ""])
def test_supported_counter_query_failure_restores_metrics_environment(
    counter_discovery, monkeypatch, query, original_path
) -> None:
    if original_path is None:
        monkeypatch.delenv("ROCPROFILER_METRICS_PATH")
    else:
        monkeypatch.setenv("ROCPROFILER_METRICS_PATH", original_path)
    monkeypatch.setattr(counter_discovery.avail_module, query, fail_counter_enumeration)

    with pytest.raises(RuntimeError, match="SDK enumeration failed"):
        counter_discovery.soc.get_rocprof_supported_counters()

    assert os.environ.get("ROCPROFILER_METRICS_PATH") == original_path
    assert not counter_discovery.metrics_path.exists()


# =============================================================================
# LimitedSet
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
    # Same TCC base: bypasses capacity
    assert ls.add("TCC_HIT[1]") is True
    assert ls.add("TCC_HIT[2]") is True
    # Different TCC base: rejected (no capacity left)
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
# CounterFile
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
# flat_counters_in_perfmon_file
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
# _trial_counter_file_with_extra
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

    # Try adding a third TA counter, should fail
    result = _trial_counter_file_with_extra(basis, perfmon_config, ["TA_EXTRA"])
    assert result is None

    # Basis still has only 2 TA counters
    assert len(basis.blocks["TA"].elements) == 2


# =============================================================================
# _rebuild_tcc_channel_file_map
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
# _allocate_perfmon_counter_files
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
    # 3 SQ counters all fit in one bucket (SQ capacity 8)
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


# =============================================================================
# metric-aware coalesce: accum buckets and formula-only grouping
# =============================================================================


def test_metric_aware_coalesce_packs_regular_counters_into_accum_bucket(
    gfx1250_soc, gfx1250_perfmon_config
):
    """Regular PMCs may fill spare capacity in a dedicated accumulator bucket."""
    accum = CounterFile("SQ_INST_LEVEL_LDS_ACCUM", gfx1250_perfmon_config)
    accum.add("SQ_INST_LEVEL_LDS_ACCUM")
    accum.reserve("SQ_INST_LEVEL_LDS_ACCUM", 1)

    work = {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}

    with ExitStack() as patch_stack:
        apply_cp_util_priority_patches(gfx1250_soc, patch_stack)
        remaining, files, _ = gfx1250_soc._metric_aware_coalesce_pass(work, [accum], 0)

    assert remaining == set()
    flat = set(flat_counters_in_perfmon_file(files[0]))
    assert {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}.issubset(flat)


def test_metric_aware_coalesce_groups_on_formula_counters_only(
    gfx1250_soc, gfx1250_perfmon_config
):
    """SQ_WAVES reaches this metric through SUPPORTED_DENOM, not the CP
    Utilization formula, so an exhausted SQ block must not push the GRBM pair
    out of a bucket that still has room for it."""
    bucket = CounterFile("0", gfx1250_perfmon_config)
    for filler_idx in range(gfx1250_perfmon_config["SQ"]):
        bucket.add(f"SQ_FILLER_{filler_idx}")

    work = {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum", "SQ_WAVES"}

    with ExitStack() as patch_stack:
        apply_cp_util_priority_patches(gfx1250_soc, patch_stack)
        remaining, files, _ = gfx1250_soc._metric_aware_coalesce_pass(work, [bucket], 0)

    assert len(files) == 1
    flat = set(flat_counters_in_perfmon_file(files[0]))
    assert {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}.issubset(flat)
    assert remaining == {"SQ_WAVES"}


def test_allocate_priority_ratio_partners_share_bucket_despite_global_denom(
    gfx1250_soc,
):
    """Ratio partners co-locate; SUPPORTED_DENOM spill may use other buckets."""
    counters = {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum", "SQ_WAVES"}

    with ExitStack() as patch_stack:
        apply_cp_util_priority_patches(gfx1250_soc, patch_stack)
        files, _, _ = gfx1250_soc._allocate_perfmon_counter_files(counters)

    def bucket_for(counter: str) -> str:
        for f in files:
            if counter in flat_counters_in_perfmon_file(f):
                return f.name
        msg = f"{counter!r} not allocated"
        raise AssertionError(msg)

    assert bucket_for("GRBM_CP_BUSY_sum") == bucket_for("GRBM_GUI_ACTIVE_sum")


def test_allocate_keeps_one_accum_counter_per_bucket(gfx1250_soc):
    """Every *_ACCUM counter aliases the single SQ_ACCUM_PREV_HIRES register,
    so no perfmon pass may hold two of them."""
    counters = {
        "SQ_INST_LEVEL_LDS_ACCUM",
        "SQ_IFETCH_LEVEL_ACCUM",
        "GRBM_CP_BUSY_sum",
        "GRBM_GUI_ACTIVE_sum",
        "SQ_WAVES",
    }

    with ExitStack() as patch_stack:
        apply_cp_util_priority_patches(gfx1250_soc, patch_stack)
        files, _, accu_file_count = gfx1250_soc._allocate_perfmon_counter_files(
            counters
        )

    assert accu_file_count == 2
    for counter_file in files:
        accum_counters = [
            ctr
            for ctr in flat_counters_in_perfmon_file(counter_file)
            if ctr.endswith("_ACCUM")
        ]
        assert len(accum_counters) <= 1


@pytest.mark.parametrize("gpu_arch", ["gfx1151", "gfx1152", "gfx1153"])
def test_same_bucket_priority_resolves_gfx115x_policy(gpu_arch):
    """gfx115x parts share one profiling_counter_grouping_policy.yaml block."""
    soc = _make_soc(PERFMON_CONFIG, arch=gpu_arch)
    ids = soc._same_bucket_priority_metric_ids()
    assert "2.1.3" in ids
    assert "8.3.0" in ids
    assert "11.3.0" in ids
    assert "17.1.0" in ids


@pytest.mark.parametrize("gpu_arch", ["gfx908", "gfx90a", "gfx942"])
def test_same_bucket_priority_hbm_traffic_ids_match_yaml(gpu_arch):
    """Policy metric IDs must resolve to 'HBM Read Traffic' and
    'HBM Write and Atomic Traffic' in the analysis YAMLs.  Guards against
    metric index drift after YAML re-org."""
    config_dir = config.analysis_configs_dir
    soc = _make_soc(PERFMON_CONFIG, arch=gpu_arch)
    ids = soc._same_bucket_priority_metric_ids()
    resolved = [_resolve_metric_name(config_dir, gpu_arch, mid) for mid in ids]
    assert None not in resolved, (
        f"{gpu_arch}: some policy IDs did not resolve: "
        f"{[mid for mid, name in zip(ids, resolved) if name is None]}"
    )
    unexpected = set(resolved) - HBM_TRAFFIC_METRIC_NAMES
    assert not unexpected, (
        f"{gpu_arch}: policy IDs resolve to unexpected metrics: {unexpected}"
    )


def test_same_bucket_priority_empty_for_gfx950():
    soc = _make_soc(PERFMON_CONFIG, arch="gfx950")
    assert soc._same_bucket_priority_metric_ids() == ()


# =============================================================================
# _expand_tcc_template_counters
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

    # No templates: input unchanged
    assert result == inp


# =============================================================================
# _append_expressions_for_filter_token alias handling
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
        OmniSoC_Base._append_expressions_for_filter_token(
            _fake_soc_for_filter_token(), "SQ", {}, "/cfg", []
        )


def test_filter_token_known_alias_resolves_without_crash(monkeypatch):
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.get_arch_alias_to_panel_id",
        lambda arch: {"lds": "10"},
    )
    expressions: list[str] = []
    # Alias resolves to block id 10 -> file id "1000", absent from the
    # empty config dict, so the token is skipped with a warning, not a crash.
    OmniSoC_Base._append_expressions_for_filter_token(
        _fake_soc_for_filter_token(), "lds", {}, "/cfg", expressions
    )
    assert expressions == []


# =============================================================================
# Memory Bandwidth Analysis counter selection
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


# =============================================================================
# Counters come only from metric expressions
# =============================================================================


@pytest.mark.parametrize(
    "filter_blocks",
    [
        pytest.param([], id="no_filter"),
        pytest.param(["2"], id="file"),
        pytest.param(["2.1"], id="table"),
        pytest.param(["2.1.0"], id="metric"),
    ],
)
def test_detect_counters_reads_only_metric_expressions(
    membw_analysis_soc: OmniSoC_Base, filter_blocks: list[str]
) -> None:
    """Comments, titles, descriptions and metric names add no counters,
    whichever way the metric is selected."""
    config_root = config.analysis_configs_dir / "gfx950"
    (config_root / "0200_baseline.yaml").write_text(
        PROSE_ANALYSIS_CONFIG, encoding="utf-8"
    )
    membw_analysis_soc.get_args().filter_blocks = filter_blocks

    counters, _ = membw_analysis_soc.detect_counters()

    formula_counters, _ = extract_counters_and_variables(
        "", membw_analysis_soc._mspec.gpu_series
    )
    formula_counters |= {"SQ_LEVEL_WAVES_ACCUM", "SQ_BUSY_CYCLES"}
    assert counters == formula_counters


@pytest.mark.parametrize("gpu_arch", CONFIG_ARCHS)
def test_panel_counters_add_up_to_full_profile(gpu_arch: str) -> None:
    """On the shipped configs, the counters of each panel (-b <panel>) add up to
    the default profile, and with panel 30 to --membw-analysis."""
    config_dir = config.analysis_configs_dir
    counters_by_file: dict[str, set[str]] = {}
    for path in (config_dir / canonical_config_arch(gpu_arch)).glob("*.yaml"):
        file_id = path.name.split("_")[0]
        panel = str(int(file_id) // 100)
        soc = offline_soc(gpu_arch, [panel], membw_analysis=True)
        counters_by_file[file_id], _ = soc.detect_counters()
    membw_file = str(MEMBW_ANALYSIS_PANEL_ID)

    default, _ = offline_soc(gpu_arch, [], False).detect_counters()
    membw, _ = offline_soc(gpu_arch, [], True).detect_counters()

    assert default == set().union(
        *(counters for file, counters in counters_by_file.items() if file != membw_file)
    )
    assert membw == set().union(*counters_by_file.values())


@pytest.mark.parametrize("gpu_arch", CONFIG_ARCHS)
def test_live_attach_set_fits_one_pass_on_each_architecture(gpu_arch: str) -> None:
    """Live attachment cannot replay the workload for additional counter passes."""
    soc = offline_soc(
        gpu_arch,
        [],
        membw_analysis=False,
        set_selected=common.SINGLE_PASS_SET,
    )
    soc.set_perfmon_config(mi_gpu_specs.get_perfmon_config(gpu_arch))
    requested_counters, selected_blocks = soc.detect_counters()
    assert selected_blocks
    assert requested_counters

    counter_files, _, accumulator_count = soc._allocate_perfmon_counter_files(
        requested_counters
    )
    assert len(counter_files) == 1, f"{gpu_arch}: live attachment needs one pass"
    assert set(flat_counters_in_perfmon_file(counter_files[0])) == requested_counters
    assert accumulator_count == 0, f"{gpu_arch}: live attachment set uses accumulators"


# =============================================================================
# post_profiling
# =============================================================================


@pytest.mark.parametrize(
    ("filter_blocks", "expect_benchmark"),
    [
        pytest.param([], True, id="no_filter_runs_benchmark"),
        pytest.param(["4"], True, id="block_4_runs_benchmark"),
        pytest.param(["roof"], True, id="roof_alias_runs_benchmark"),
        pytest.param(["11.2.3", "11.2.4"], False, id="set_metrics_skip_benchmark"),
        pytest.param(["2"], False, id="unrelated_block_skips_benchmark"),
    ],
)
def test_post_profiling_roofline_gating(
    perfmon_config,
    tmp_path: Path,
    monkeypatch,
    filter_blocks: list[str],
    expect_benchmark: bool,
) -> None:
    """Roofline runs only when the effective filter blocks cover block 4."""
    soc = _make_soc(perfmon_config, arch="gfx942")
    args = soc.get_args()
    args.device = 0
    args.filter_blocks = filter_blocks
    args.no_roof = False
    args.output_directory = str(tmp_path)
    mock_benchmark = MagicMock()
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.run_roofline_benchmark", mock_benchmark
    )
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.validate_roofline_csv",
        lambda _workload_dir: (True, ""),
    )

    soc.post_profiling()

    assert mock_benchmark.called is expect_benchmark
