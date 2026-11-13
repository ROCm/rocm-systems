# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for COLLECT_RATIO / COLLECT_SUM Phase 2 production wiring."""

import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from utils import schema
from utils.counters.counter_file import CounterFile
from utils.metrics.aggregation import merge_dispatch_collect_ratio
from utils.metrics.collectable import (
    COLLECT_RATIO_SPECS_ATTR,
    COLLECT_SUM_SPECS_ATTR,
    COLLECTABLE_EXPR_CACHE_ATTR,
    apply_composite_metrics,
)
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import (
    build_metric_value_string,
    is_composite_avg_formula,
    parse_collect_ratio_parts,
    parse_collect_sum_submetrics,
)
from utils.metrics.metric_evaluator import MetricEvaluator
from utils.mi_gpu_spec import mi_gpu_specs
from utils.parser import build_dfs
from utils.utils_common import load_collectable_formulas, load_panel_file
from vendored import yaml

_GFX942 = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "gfx942"
)
_GFX950 = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "gfx950"
)


def fits_one_pass(counters: frozenset, perfmon_config: dict) -> bool:
    """Return whether every counter fits one empty perfmon pass."""
    counter_file = CounterFile("0", perfmon_config)
    return all(counter_file.add(counter) for counter in sorted(counters))


def _sys_info() -> pd.Series:
    return pd.Series({
        "ip_blocks": "standard",
        "gpu_arch": "gfx942",
        "se_per_gpu": 8,
        "sa_per_se": 2,
        "pipes_per_gpu": 4,
        "cu_per_gpu": 304,
        "simd_per_cu": 4,
        "sqc_per_gpu": 76,
        "lds_banks_per_cu": 32,
        "cur_sclk": 1800.0,
        "cur_mclk": 1200.0,
        "max_sclk": 2100.0,
        "max_mclk": 1600.0,
        "max_waves_per_cu": 40,
        "num_memory_channels": 32,
        "total_l2_chan": 128,
        "num_xcd": 8,
        "wave_size": 64,
    })


@pytest.mark.misc
def test_parse_collect_ratio_parts():
    formula = "COLLECT_RATIO(_collect.a + _collect.b, _collect.c + _collect.d)"
    parts = parse_collect_ratio_parts(formula)
    assert parts == (["_collect.a", "_collect.b"], ["_collect.c", "_collect.d"])


@pytest.mark.misc
def test_parse_collect_ratio_splits_leading_minus():
    parts = parse_collect_ratio_parts("COLLECT_RATIO(a - b, c)")
    assert parts == (["a", "-b"], ["c"])


@pytest.mark.misc
def test_merge_dispatch_collect_ratio():
    nums = [pd.Series({1: 100.0}), pd.Series({1: 50.0})]
    dens = [pd.Series({1: 10.0}), pd.Series({1: 5.0})]
    assert merge_dispatch_collect_ratio(nums, dens) == pytest.approx(10.0)


@pytest.mark.misc
def test_merge_dispatch_collect_ratio_pools_unequal_dispatches():
    nums = [pd.Series({1: 10.0, 2: 100.0})]
    dens = [pd.Series({1: 2.0, 2: 10.0})]
    mean_of_ratios = ((10.0 / 2.0) + (100.0 / 10.0)) / 2.0
    pooled = (10.0 + 100.0) / (2.0 + 10.0)
    result = merge_dispatch_collect_ratio(nums, dens)
    assert result == pytest.approx(pooled)
    assert result != pytest.approx(mean_of_ratios)


@pytest.mark.misc
def test_merge_dispatch_collect_ratio_keeps_numerator_when_denominator_is_zero(
    caplog,
):
    nums = [pd.Series({1: 1000.0, 2: 1000.0})]
    dens = [pd.Series({1: 100.0, 2: 0.0})]
    with caplog.at_level(logging.WARNING):
        assert merge_dispatch_collect_ratio(nums, dens) == pytest.approx(20.0)
    assert "missing" not in caplog.text


@pytest.mark.misc
def test_merge_dispatch_collect_ratio_is_unset_when_pooled_denominator_is_zero():
    nums = [pd.Series({1: 1000.0, 2: 500.0})]
    dens = [pd.Series({1: 0.0, 2: 0.0})]
    assert pd.isna(merge_dispatch_collect_ratio(nums, dens))


@pytest.mark.misc
def test_unset_collectables_are_one_summary(caplog):
    table = _two_piece_table(
        COLLECT_RATIO_SPECS_ATTR,
        _RATIO_SPEC,
        {},
    )
    table.loc["1", "Avg"] = ""
    table.loc["2", "Avg"] = ""
    raw = pd.DataFrame({"Dispatch_ID": [1], "A": [10.0], "B": [0.0]})
    with caplog.at_level(logging.WARNING):
        apply_composite_metrics(
            {1: table}, {1: "metric_table"}, [MetricEvaluator(raw, {}, {})], {}
        )
    assert caplog.text.count("were left unset") == 1
    assert "not found" not in caplog.text


def test_gfx942_hbm_bandwidth_is_collect_sum():
    doc = yaml.safe_load((_GFX942 / "0400_roofline.yaml").read_text())
    hbm = doc["Panel Config"]["data source"][0]["metric_table"]["metric"][
        "HBM Bandwidth"
    ]
    assert is_composite_avg_formula(hbm["value"])
    assert parse_collect_sum_submetrics(hbm["value"]) == [
        "_collect.hbm_rd_bw",
        "_collect.hbm_wr_bw",
    ]


@pytest.mark.misc
def test_gfx942_ai_hbm_is_collect_ratio():
    doc = yaml.safe_load((_GFX942 / "0400_roofline.yaml").read_text())
    ai = doc["Panel Config"]["data source"][1]["metric_table"]["metric"]["AI HBM"]
    assert is_composite_avg_formula(ai["value"])
    nums, dens = parse_collect_ratio_parts(ai["value"])
    assert "_collect.ai_flops_f16" in nums
    assert "_collect.hbm_rd_bytes" in dens


@pytest.mark.misc
def test_eval_valu_flops_collect_sum_on_gfx942_sol():
    panel = load_panel_file(
        _GFX942 / "0200_system_speed_of_light.yaml",
        load_collectable_formulas(_GFX942),
    )["Panel Config"]
    ac = schema.ArchConfig()
    ac.panel_configs = OrderedDict([(200, panel)])
    sys_info = _sys_info()
    build_dfs(ac, filter_metrics=None, sys_info=sys_info, profiling_config={})

    raw = pd.DataFrame({
        "Dispatch_ID": [1],
        "SQ_INSTS_VALU_ADD_F16": [10],
        "SQ_INSTS_VALU_MUL_F16": [0],
        "SQ_INSTS_VALU_TRANS_F16": [0],
        "SQ_INSTS_VALU_FMA_F16": [5],
        "SQ_INSTS_VALU_ADD_F32": [20],
        "SQ_INSTS_VALU_MUL_F32": [0],
        "SQ_INSTS_VALU_TRANS_F32": [0],
        "SQ_INSTS_VALU_FMA_F32": [0],
        "SQ_INSTS_VALU_ADD_F64": [0],
        "SQ_INSTS_VALU_MUL_F64": [0],
        "SQ_INSTS_VALU_TRANS_F64": [0],
        "SQ_INSTS_VALU_FMA_F64": [0],
        "Start_Timestamp": [0],
        "End_Timestamp": [1000],
        "GRBM_GUI_ACTIVE": [1],
    })
    build_metric_value_string(ac.dfs, ac.dfs_type, normal_unit="")
    eval_metric(
        ac.dfs,
        ac.dfs_type,
        ac.dfs_expressions,
        sys_info,
        pd.DataFrame(),
        raw,
        debug=False,
    )
    df = ac.dfs[201]
    parent = df[df["Metric"] == "VALU FLOPs"].iloc[0]["Avg"]
    # 64 * (10 + 2*5 + 20) / 1000 = 2.56
    assert parent == pytest.approx(2.56, rel=1e-6)


@pytest.mark.misc
def test_gfx950_hbm_bandwidth_is_collect_sum():
    doc = load_panel_file(
        _GFX950 / "0400_roofline.yaml", load_collectable_formulas(_GFX950)
    )
    metrics = doc["Panel Config"]["data source"][0]["metric_table"]["metric"]
    hbm = metrics["HBM Bandwidth"]
    assert is_composite_avg_formula(hbm["value"])
    assert parse_collect_sum_submetrics(hbm["value"]) == [
        "_collect.hbm_rd_bw",
        "_collect.hbm_wr_bw",
    ]
    read_bandwidth = metrics["_collect.hbm_rd_bw"]["value"]
    assert "TCC_EA0_RDREQ_128B_sum" in read_bandwidth
    assert "TCC_BUBBLE_sum" not in read_bandwidth


@pytest.mark.misc
def test_gfx950_ipc_fits_one_bucket_and_is_not_rewritten():
    doc = yaml.safe_load(
        (_GFX950 / "1100_compute_units_compute_pipeline.yaml").read_text(
            encoding="utf-8"
        )
    )
    ipc = doc["Panel Config"]["data source"][1]["metric_table"]["metric"]["IPC"]
    assert ipc["avg"] == "SUM(SQ_INSTS) / SUM(SQ_BUSY_CU_CYCLES)"
    assert not is_composite_avg_formula(ipc["avg"])
    perfmon_config = mi_gpu_specs.get_perfmon_config("gfx950")
    assert fits_one_pass(
        frozenset(["SQ_INSTS", "SQ_BUSY_CU_CYCLES"]),
        perfmon_config,
    )


def _two_piece_table(
    composite_attr: str,
    composite_spec: object,
    piece_exprs: dict,
) -> pd.DataFrame:
    """Two collectable rows and one composite parent."""
    table = pd.DataFrame(
        {"Metric": ["first", "second", "parent"], "Avg": ["", "", ""]},
        index=["1", "2", "3"],
    )
    table.attrs[composite_attr] = {"3": composite_spec}
    table.attrs[COLLECTABLE_EXPR_CACHE_ATTR] = piece_exprs
    return table


def _evaluate_parent(table: pd.DataFrame, pass_counters: tuple) -> object:
    """Evaluate the parent with one pass table per entry of pass_counters.

    A = 10 and B = 5 on the passes. The merged table holds 1 for both, so a
    piece that read it would change the result.
    """
    values = {"A": 10.0, "B": 5.0}
    pass_tables = [
        pd.DataFrame({"Dispatch_ID": [1], **{name: [values[name]] for name in names}})
        for names in pass_counters
    ]
    merged = pd.DataFrame({"Dispatch_ID": [1], "A": [1.0], "B": [1.0]})
    evaluators = [MetricEvaluator(table, {}, {}) for table in [*pass_tables, merged]]
    apply_composite_metrics({1: table}, {1: "metric_table"}, evaluators, {})
    return table.at["3", "Avg"]


_RATIO_SPEC = {"numerator": ["first"], "denominator": ["second"]}
_ABSOLUTE_PIECES = {
    "first": "to_sum(raw_pmc_df['A'])",
    "second": "to_sum(raw_pmc_df['B'])",
}


@pytest.mark.misc
def test_collect_ratio_divides_pieces_from_different_passes():
    table = _two_piece_table(
        COLLECT_RATIO_SPECS_ATTR,
        _RATIO_SPEC,
        _ABSOLUTE_PIECES,
    )
    assert _evaluate_parent(table, (("A",), ("B",))) == pytest.approx(2.0)


@pytest.mark.misc
def test_collect_ratio_on_one_pass_evaluates():
    table = _two_piece_table(
        COLLECT_RATIO_SPECS_ATTR,
        _RATIO_SPEC,
        _ABSOLUTE_PIECES,
    )
    assert _evaluate_parent(table, (("A", "B"),)) == pytest.approx(2.0)


@pytest.mark.misc
def test_collect_sum_of_absolute_pieces_across_passes_evaluates():
    table = _two_piece_table(
        COLLECT_SUM_SPECS_ATTR,
        ["first", "second"],
        _ABSOLUTE_PIECES,
    )
    assert _evaluate_parent(table, (("A",), ("B",))) == pytest.approx(15.0)


@pytest.mark.misc
def test_collect_sum_adds_tcp_hit_pieces_that_cannot_share_a_bucket():
    """Five TCP counters do not share gfx942's four TCP slots.

    Each piece still fits one pass. The parent adds those finished values.
    """
    perfmon_config = dict(mi_gpu_specs.get_perfmon_config("gfx942"))
    assert perfmon_config["TCP"] == 4
    non_rw = frozenset({
        "TCP_TOTAL_CACHE_ACCESSES_sum",
        "TCP_TCC_READ_REQ_sum",
        "TCP_TCC_WRITE_REQ_sum",
    })
    atom = frozenset({
        "TCP_TOTAL_CACHE_ACCESSES_sum",
        "TCP_TCC_ATOMIC_WITH_RET_REQ_sum",
        "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum",
    })
    assert fits_one_pass(non_rw, perfmon_config)
    assert fits_one_pass(atom, perfmon_config)
    assert not fits_one_pass(non_rw | atom, perfmon_config)
    table = _two_piece_table(
        COLLECT_SUM_SPECS_ATTR,
        ["first", "second"],
        {
            "first": "to_sum(raw_pmc_df['A'])",
            "second": "-100 * to_sum(raw_pmc_df['B'])",
        },
    )
    assert _evaluate_parent(table, (("A",), ("B",))) == pytest.approx(-490.0)


@pytest.mark.misc
def test_collect_sum_adds_a_negative_piece_from_another_pass():
    table = _two_piece_table(
        COLLECT_SUM_SPECS_ATTR,
        ["first", "second"],
        {"first": "to_sum(raw_pmc_df['A'])", "second": "to_sum(-(raw_pmc_df['B']))"},
    )
    assert _evaluate_parent(table, (("A",), ("B",))) == pytest.approx(5.0)


@pytest.mark.misc
def test_collect_sum_subtracting_on_one_pass_evaluates():
    table = _two_piece_table(
        COLLECT_SUM_SPECS_ATTR,
        ["first", "second"],
        {"first": "to_sum(raw_pmc_df['A'])", "second": "to_sum(-(raw_pmc_df['B']))"},
    )
    assert _evaluate_parent(table, (("A", "B"),)) == pytest.approx(5.0)
