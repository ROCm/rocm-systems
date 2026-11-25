# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for counter allocation pipeline in soc_base.py.

Tests the bin-packing helpers used by perfmon_coalesce. No GPU hardware
required.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import common
import pytest

from rocprof_compute_soc.metric_library import loader
from rocprof_compute_soc.metric_library.loader import Collectables
from rocprof_compute_soc.soc_base import (
    OmniSoC_Base,
    ProfileMetric,
)
from utils.counters.allocator import reads_one_pass
from utils.counters.counter_file import flat_counters_in_perfmon_file
from utils.mi_gpu_spec import mi_gpu_specs
from utils.utils_common import (
    MEMBW_ANALYSIS_PANEL_ID,
    canonical_config_arch,
    iter_view_rows,
    load_views,
    row_metric_ids,
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
TABLE_3001_COUNTER = "TCC_BOTTLENECK_COUNTER"  # block 30, table 3001
TABLE_3002_COUNTER = "TCC_EA_COUNTER"  # block 30, table 3002
FIXTURE_COUNTERS = {BASELINE_COUNTER, TABLE_3001_COUNTER, TABLE_3002_COUNTER}
VALUE_HEADER = {"metric": "Metric", "value": "Value"}
# Views of a baseline panel and of the memory bandwidth analysis panel, and the
# metrics they show, one per table.
MEMBW_VIEWS = {
    200: {
        "view": "Baseline",
        "tables": [
            {
                "title": "Baseline",
                "header": VALUE_HEADER,
                "metrics": [{"metric": "toy.baseline"}],
            }
        ],
    },
    MEMBW_ANALYSIS_PANEL_ID: {
        "view": "Memory Bandwidth Analysis",
        "tables": [
            {
                "title": "L2 Bottleneck Detection Indicators",
                "header": VALUE_HEADER,
                "metrics": [{"metric": "toy.l2_bottleneck"}],
            },
            {
                "title": "EA Interface",
                "header": VALUE_HEADER,
                "metrics": [{"metric": "toy.ea"}],
            },
        ],
    },
}
MEMBW_METRICS = {
    metric_id: {
        "name": metric_id,
        "formula": f"SUM({counter})",
        "unit": "Count",
        "description": "",
    }
    for metric_id, counter in (
        ("toy.baseline", BASELINE_COUNTER),
        ("toy.l2_bottleneck", TABLE_3001_COUNTER),
        ("toy.ea", TABLE_3002_COUNTER),
    )
}

# Every counter name outside the formula is unique, so any leak from a title,
# description or metric name shows up in the result.
PROSE_VIEW = {
    "view": "SQ_VIEW_TITLE_COUNTER",
    "tables": [
        {
            "title": "SQ_TITLE_COUNTER table",
            "header": {"metric": "Metric", "avg": "Avg", "unit": "Unit"},
            "metrics": [{"metric": "toy.waves"}],
        }
    ],
}
PROSE_METRICS = {
    "toy.waves": {
        "name": "SQ_METRIC_NAME_COUNTER",
        "formula": "SUM(SQ_LEVEL_WAVES_ACCUM) / SUM(SQ_BUSY_CYCLES)",
        "unit": "Wavefronts",
        "description": "Uses SQ_DESCRIPTION_COUNTER.",
    }
}

# One arch per metric library arch (the gfx115x parts share one).
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
    """OmniSoC_Base that detects counters from the analysis views without a GPU.
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
def membw_analysis_soc(monkeypatch) -> OmniSoC_Base:
    """A gfx950 soc whose views are MEMBW_VIEWS, showing MEMBW_METRICS."""
    soc = offline_soc("gfx950", [], membw_analysis=False)
    library = soc._profile_library()
    library.metrics = dict(MEMBW_METRICS)
    views = dict(MEMBW_VIEWS)
    monkeypatch.setattr(soc, "_profile_views", lambda: views)
    return soc


def _make_soc(perfmon_config, arch="gfx908", num_xcd=1, l2_banks=4):
    """Build a minimal OmniSoC_Base without rocminfo or GPU access."""
    mspec = MagicMock()
    mspec.gpu_arch = arch
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
    """Minimal gfx1250 OmniSoC for allocation tests."""
    soc = _make_soc(gfx1250_perfmon_config, arch="gfx1250")
    soc._mspec.gpu_series = "GFX1250_SERIES"
    return soc


def patch_cp_util_metric(soc: OmniSoC_Base) -> Any:
    """Make CP Utilization the only metric that soc profiles."""
    return patch.object(
        soc,
        "_iter_profile_metrics",
        return_value=iter([ProfileMetric("cpc.cp_util", CP_UTIL_METRIC_EXPRESSIONS)]),
    )


# =============================================================================
# _allocate_perfmon_counter_files
# =============================================================================


def test_allocate_accum_counters_cost_two_sq_slots(perfmon_config):
    """Each *_ACCUM counter costs two SQ slots and shares a bucket with others."""
    soc = _make_soc(perfmon_config)
    counters = {
        "SQ_LEVEL_WAVES_ACCUM",
        "SQC_DCACHE_INFLIGHT_LEVEL_ACCUM",
        "TA_ADDR",
    }

    files = soc._allocate_perfmon_counter_files(counters)

    assert len(files) == 1
    assert set(flat_counters_in_perfmon_file(files[0])) == counters
    assert files[0].blocks["SQ"].avail == perfmon_config["SQ"] - 4


def test_allocate_seven_sq_plus_one_accum_needs_second_bucket(perfmon_config):
    """Seven SQ counters and one *_ACCUM need nine slots against SQ capacity 8."""
    soc = _make_soc(perfmon_config)
    counters = {
        "SQ_ACTIVE_INST_ANY",
        "SQ_INSTS",
        "SQ_INSTS_MFMA",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_VALU",
        "SQ_VALU_MFMA_BUSY_CYCLES",
        "SQ_WAVES",
        "SQ_INST_LEVEL_SMEM_ACCUM",
    }

    files = soc._allocate_perfmon_counter_files(counters)

    assert len(files) == 2
    assert counters == {
        counter for f in files for counter in flat_counters_in_perfmon_file(f)
    }


def test_allocate_first_fit_packing(perfmon_config):
    soc = _make_soc(perfmon_config)
    # 3 SQ counters all fit in one bucket (SQ capacity 8)
    counters = {"SQ_WAVES", "SQ_BUSY", "SQ_INSTS"}

    files = soc._allocate_perfmon_counter_files(counters)

    assert len(files) == 1
    flat = set(flat_counters_in_perfmon_file(files[0]))
    assert flat == counters


def test_allocate_tcc_channel_coalescing(perfmon_config):
    soc = _make_soc(perfmon_config)
    # TCC channels with same base should land in the same bucket
    counters = {"TCC_HIT[0]", "TCC_HIT[1]", "TCC_HIT[2]", "SQ_WAVES"}

    files = soc._allocate_perfmon_counter_files(counters)

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


def test_allocate_ratio_partners_share_pass(gfx1250_soc):
    """A metric's numerator and denominator land in one pass."""
    counters = {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum", "SQ_WAVES"}

    with patch_cp_util_metric(gfx1250_soc):
        files = gfx1250_soc._allocate_perfmon_counter_files(counters)

    def bucket_for(counter: str) -> str:
        for f in files:
            if counter in flat_counters_in_perfmon_file(f):
                return f.name
        msg = f"{counter!r} not allocated"
        raise AssertionError(msg)

    assert bucket_for("GRBM_CP_BUSY_sum") == bucket_for("GRBM_GUI_ACTIVE_sum")


def test_allocate_allows_several_accum_counters_per_bucket(gfx1250_soc):
    """Each *_ACCUM counter has its own accumulating register, so several may
    share a pass when the SQ block has room."""
    counters = {
        "SQ_INST_LEVEL_LDS_ACCUM",
        "SQ_IFETCH_LEVEL_ACCUM",
        "GRBM_CP_BUSY_sum",
        "GRBM_GUI_ACTIVE_sum",
        "SQ_WAVES",
    }

    with patch_cp_util_metric(gfx1250_soc):
        files = gfx1250_soc._allocate_perfmon_counter_files(counters)

    assert len(files) == 1
    assert set(flat_counters_in_perfmon_file(files[0])) == counters


def test_allocate_keeps_sp_valu_and_sq_vmem_apart_on_gfx1250(gfx1250_soc):
    """On gfx1250 an SP_VALU_* and an SQ_VMEM_* counter get separate passes,
    even with SQ room for both."""
    files = gfx1250_soc._allocate_perfmon_counter_files({"SP_VALU_A", "SQ_VMEM_B"})

    assert sorted(flat_counters_in_perfmon_file(f) for f in files) == [
        ["SP_VALU_A"],
        ["SQ_VMEM_B"],
    ]


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
# _filter_token_metric_ids alias handling
# =============================================================================


def _fake_soc_for_filter_token(arch: str = "gfx942") -> Any:
    """Minimal stand-in with the arch and the (empty) views the method reads."""
    return SimpleNamespace(
        _mspec=SimpleNamespace(gpu_arch=arch), _profile_views=lambda: {}
    )


def test_filter_token_unknown_alias_exits_instead_of_keyerror(monkeypatch):
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.get_arch_alias_to_panel_id",
        lambda arch: {"lds": "10"},
    )
    with pytest.raises(SystemExit):
        OmniSoC_Base._filter_token_metric_ids(_fake_soc_for_filter_token(), "SQ")


def test_filter_token_known_alias_resolves_without_crash(monkeypatch):
    monkeypatch.setattr(
        "rocprof_compute_soc.soc_base.get_arch_alias_to_panel_id",
        lambda arch: {"lds": "10"},
    )
    # Alias resolves to block id 10 -> panel 1000, absent from the empty
    # views, so the token is skipped with a warning, not a crash.
    assert (
        OmniSoC_Base._filter_token_metric_ids(_fake_soc_for_filter_token(), "lds") == []
    )


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
            {BASELINE_COUNTER, TABLE_3001_COUNTER, TABLE_3002_COUNTER},
            id="flag_on_no_filter_keeps_every_table",
        ),
        pytest.param(
            True,
            ["30"],
            {TABLE_3001_COUNTER, TABLE_3002_COUNTER},
            id="block_30_keeps_both_block_30_tables_and_drops_baseline",
        ),
        pytest.param(
            True,
            ["30.1"],
            {TABLE_3001_COUNTER},
            id="block_30_1_keeps_only_table_3001",
        ),
        pytest.param(
            True,
            ["2", "30.2"],
            {BASELINE_COUNTER, TABLE_3002_COUNTER},
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

    Each table in the fixture owns one unique synthetic counter, so the
    selected fixture counters identify exactly which tables survived. The 30.1
    and mixed 30.2 cases prove selection is by table id; the mixed case also
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
    ],
)
def test_detect_counters_reads_only_metric_expressions(
    membw_analysis_soc: OmniSoC_Base, filter_blocks: list[str]
) -> None:
    """Titles, descriptions and metric names add no counters, whether a block or a
    table is selected. A row id selects its metric through the views and the metric
    library."""
    membw_analysis_soc._profile_views()[200] = PROSE_VIEW
    membw_analysis_soc._profile_library().metrics.update(PROSE_METRICS)
    membw_analysis_soc.get_args().filter_blocks = filter_blocks

    counters, _ = membw_analysis_soc.detect_counters()

    formula_counters, _ = extract_counters_and_variables(
        "", Collectables.of_arch(membw_analysis_soc._mspec.gpu_arch)
    )
    formula_counters |= {"SQ_LEVEL_WAVES_ACCUM", "SQ_BUSY_CYCLES"}
    assert counters == formula_counters


@pytest.mark.parametrize("gpu_arch", CONFIG_ARCHS)
def test_panel_counters_add_up_to_full_profile(gpu_arch: str) -> None:
    """On the shipped views, the counters of each panel (-b <panel>) add up to
    the default profile, and with panel 30 to --membw-analysis."""
    counters_by_file: dict[str, set[str]] = {}
    for panel_id in load_views(gpu_arch):
        soc = offline_soc(gpu_arch, [str(panel_id // 100)], membw_analysis=True)
        counters_by_file[f"{panel_id:04d}"], _ = soc.detect_counters()
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

    counter_files = soc._allocate_perfmon_counter_files(requested_counters)
    assert len(counter_files) == 1, f"{gpu_arch}: live attachment needs one pass"
    assert set(flat_counters_in_perfmon_file(counter_files[0])) == requested_counters
    assert not any(counter.endswith("_ACCUM") for counter in requested_counters), (
        f"{gpu_arch}: live attachment set uses accumulators"
    )


@pytest.mark.parametrize("gpu_arch", CONFIG_ARCHS)
def test_every_metric_reads_one_pass(gpu_arch: str) -> None:
    """Every metric of the default profile reads its counters from one pass."""
    soc = offline_soc(gpu_arch, [], membw_analysis=False)
    soc.set_perfmon_config(mi_gpu_specs.get_perfmon_config(gpu_arch))
    counters, _ = soc.detect_counters()
    passes = [
        set(flat_counters_in_perfmon_file(counter_file))
        for counter_file in soc._allocate_perfmon_counter_files(counters)
    ]
    collectables = Collectables.of_arch(gpu_arch)
    spanning = [
        metric.metric_id
        for metric in soc._iter_profile_metrics()
        if not reads_one_pass(metric.expressions, collectables, passes)
    ]
    assert spanning == [], (
        f"{gpu_arch}: split these metrics into collectables "
        "(see the metric library README and tools/counter_grouping_inspector.py)"
    )


def _metrics_to_select(gpu_arch: str) -> List[str]:
    """The metric shown in the most rows of gpu_arch's views, and the first two
    metrics shown in one row."""
    rows_by_metric: Dict[str, int] = {}
    for _, row in iter_view_rows(load_views(gpu_arch)):
        for metric_id in row_metric_ids(row):
            rows_by_metric[metric_id] = rows_by_metric.get(metric_id, 0) + 1
    single = [metric for metric, count in rows_by_metric.items() if count == 1]
    return [max(rows_by_metric, key=rows_by_metric.__getitem__), *single[:2]]


@pytest.mark.parametrize(
    "gpu_arch", [arch for arch in CONFIG_ARCHS if loader.has_metric_library(arch)]
)
def test_metric_id_collects_the_counters_of_its_rows(gpu_arch: str) -> None:
    """-b <metric id> takes its counters from the metric library and saves the id.
    Typing the ids analyze shows for its rows collects the same counters and saves
    the metric id instead of the row ids."""
    for metric_id in _metrics_to_select(gpu_arch):
        by_id, typed = offline_soc(gpu_arch, [metric_id], False).detect_counters()
        rows = [
            position
            for position, row in iter_view_rows(load_views(gpu_arch))
            if row_metric_ids(row) == [metric_id]
        ]
        by_rows, saved = offline_soc(gpu_arch, rows, False).detect_counters()
        assert (by_id, typed) == (by_rows, [metric_id]), metric_id
        assert saved == [metric_id], metric_id


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
