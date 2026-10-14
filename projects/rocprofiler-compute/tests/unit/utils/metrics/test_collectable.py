# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for collectable evaluation graph and COLLECT_SUM."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from utils import schema
from utils.metrics.collectable import (
    COLLECT_RATIO_SPECS_ATTR,
    COLLECT_SUM_SPECS_ATTR,
    COLLECTABLE_IDS_ATTR,
    WEIGHTED_AVG_ATTR,
    WEIGHTED_AVG_SUBS_ATTR,
    CompositeKind,
    apply_composite_metrics,
    build_metric_eval_graph,
)
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import (
    build_metric_value_string,
    parse_collect_sum_submetrics,
)
from utils.parser import build_dfs
from vendored import yaml


def _collect_sum_arch_config() -> schema.ArchConfig:
    fixture = (
        Path(__file__).resolve().parents[3]
        / "fixtures"
        / "weighted_avg"
        / "collect_sum_metric_table.yaml"
    )
    with open(fixture, encoding="utf-8") as stream:
        doc = yaml.safe_load(stream)
    metric_table = doc["Panel Config"]["data source"][0]["metric_table"]
    panel = {
        "id": 1700,
        "title": "Collect sum pilot",
        "data source": [{"metric_table": metric_table}],
    }
    ac = schema.ArchConfig()
    ac.panel_configs = OrderedDict([(1700, panel)])
    return ac


@pytest.mark.misc
def test_parse_collect_sum_submetrics():
    assert parse_collect_sum_submetrics("COLLECT_SUM(a, b)") == ["a", "b"]


@pytest.mark.misc
def test_build_metric_eval_graph_weighted_and_sum():
    df = pd.DataFrame(
        [["1", "sub_a"], ["2", "sub_b"], ["3", "parent_w"], ["4", "parent_s"]],
        columns=["Metric_ID", "Metric"],
    ).set_index("Metric_ID")
    df.attrs[WEIGHTED_AVG_ATTR] = {
        "3": {"sub_a": {"weight_counter": "W0"}},
    }
    df.attrs[WEIGHTED_AVG_SUBS_ATTR] = {"3": ["sub_a", "sub_b"]}
    df.attrs[COLLECT_SUM_SPECS_ATTR] = {"4": ["sub_a", "sub_b"]}

    graph = build_metric_eval_graph(df)
    assert graph.collectable_row_names == {"sub_a", "sub_b"}
    kinds = {c.metric_id: c.kind for c in graph.composites}
    assert kinds["3"] is CompositeKind.WEIGHTED_AVG
    assert kinds["4"] is CompositeKind.COLLECT_SUM


@pytest.mark.misc
def test_collect_sum_pipeline_end_to_end():
    ac = _collect_sum_arch_config()
    sys_info = pd.Series({
        "ip_blocks": "standard",
        "gpu_arch": "gfx942",
        "se_per_gpu": 4,
        "sa_per_se": 2,
        "pipes_per_gpu": 4,
        "cu_per_gpu": 64,
        "simd_per_cu": 4,
        "sqc_per_gpu": 16,
        "lds_banks_per_cu": 32,
        "cur_sclk": 1800.0,
        "cur_mclk": 1200.0,
        "max_sclk": 2100.0,
        "max_mclk": 1600.0,
        "max_waves_per_cu": 40,
        "num_memory_channels": 4,
        "total_l2_chan": 32,
        "num_xcd": 1,
        "wave_size": 64,
    })
    build_dfs(ac, filter_metrics=None, sys_info=sys_info, profiling_config={})
    raw_pmc_df = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "TCC_EA0_RDREQ_DRAM_sum": [100, 40],
        "TCC_EA0_RDREQ_sum": [100, 50],
        "TCC_EA0_WRREQ_DRAM_sum": [50, 20],
        "TCC_EA0_WRREQ_sum": [50, 40],
        "GRBM_GUI_ACTIVE": [1, 1],
    })
    build_metric_value_string(ac.dfs, ac.dfs_type, normal_unit="")
    eval_metric(
        ac.dfs,
        ac.dfs_type,
        ac.dfs_expressions,
        sys_info,
        pd.DataFrame(),
        raw_pmc_df,
        debug=False,
    )
    table_id = next(tid for tid, dt in ac.dfs_type.items() if dt == "metric_table")
    parent = ac.dfs[table_id]
    row = parent[parent["Metric"] == "hbm_total_traffic"]
    # Pooled ratios: 100 * (140/150) + 100 * (70/90).
    assert row.iloc[0]["Avg"] == pytest.approx(100 * (140 / 150) + 100 * (70 / 90))


@pytest.mark.misc
def test_composite_order_is_topological_and_id_is_the_lookup_key():
    df = pd.DataFrame(
        [
            ["1", "child"],
            ["2", "parent"],
            ["3", "_collect.part"],
        ],
        columns=["Metric_ID", "Metric"],
    ).set_index("Metric_ID")
    df.attrs[COLLECTABLE_IDS_ATTR] = {"3": "collect.part"}
    df.attrs[COLLECT_SUM_SPECS_ATTR] = {
        "1": ["_collect.part"],
        "2": ["child"],
    }
    graph = build_metric_eval_graph(df)
    assert graph.collectable_ids["collect.part"] == "_collect.part"
    assert [item.metric_name for item in graph.composite_order()] == [
        "child",
        "parent",
    ]


@pytest.mark.misc
def test_composite_cycle_is_left_unset():
    df = pd.DataFrame(
        [["1", "left"], ["2", "right"]],
        columns=["Metric_ID", "Metric"],
    ).set_index("Metric_ID")
    df.attrs[COLLECT_SUM_SPECS_ATTR] = {
        "1": ["right"],
        "2": ["left"],
    }
    graph = build_metric_eval_graph(df)
    assert graph.composite_order() == []


@pytest.mark.misc
def test_collect_ratio_minus_negates_the_fragment():
    raw = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "SQ_A": [10.0, 30.0],
        "SQ_B": [4.0, 6.0],
        "SQ_C": [2.0, 4.0],
    })
    df = pd.DataFrame(
        [
            ["1", "num_a", "to_sum(raw_pmc_df['SQ_A'])"],
            ["2", "num_b", "to_sum(raw_pmc_df['SQ_B'])"],
            ["3", "den", "to_sum(raw_pmc_df['SQ_C'])"],
            ["4", "parent", ""],
        ],
        columns=["Metric_ID", "Metric", "Avg"],
    ).set_index("Metric_ID")
    df.attrs[COLLECT_RATIO_SPECS_ATTR] = {
        "4": {"numerator": ["num_a", "-num_b"], "denominator": ["den"]},
    }
    df.attrs["collectable_expr_cache"] = {
        "num_a": "to_sum(raw_pmc_df['SQ_A'])",
        "num_b": "to_sum(raw_pmc_df['SQ_B'])",
        "den": "to_sum(raw_pmc_df['SQ_C'])",
    }
    apply_composite_metrics({1: df}, {1: "metric_table"}, raw, {}, {})
    # (10-4 + 30-6) / (2+4) = 30/6 = 5. Mean of ratios would be (3 + 6) / 2.
    assert df.at["4", "Avg"] == pytest.approx(5.0)


def collect_sum_parent(raw, pieces):
    rows = []
    cache = {}
    for index, (name, expr) in enumerate(pieces, start=1):
        rows.append([str(index), name, expr])
        cache[name] = expr
    parent_id = str(len(pieces) + 1)
    rows.append([parent_id, "parent", ""])
    df = pd.DataFrame(rows, columns=["Metric_ID", "Metric", "Avg"]).set_index(
        "Metric_ID"
    )
    df.attrs[COLLECT_SUM_SPECS_ATTR] = {parent_id: [name for name, _expr in pieces]}
    df.attrs["collectable_expr_cache"] = cache
    apply_composite_metrics({1: df}, {1: "metric_table"}, raw, {}, {})
    return df.at[parent_id, "Avg"]


@pytest.mark.misc
def test_collect_sum_adds_pooled_hbm_rates():
    raw = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "RD_BYTES": [10.0, 90.0],
        "RD_DURATION": [10.0, 30.0],
        "WR_BYTES": [5.0, 15.0],
        "WR_DURATION": [10.0, 30.0],
    })
    result = collect_sum_parent(
        raw,
        [
            (
                "read",
                "to_sum(raw_pmc_df['RD_BYTES']) / to_sum(raw_pmc_df['RD_DURATION'])",
            ),
            (
                "write",
                "to_sum(raw_pmc_df['WR_BYTES']) / to_sum(raw_pmc_df['WR_DURATION'])",
            ),
        ],
    )
    # (10+90)/(10+30) + (5+15)/(10+30)
    assert result == pytest.approx(3.0)


@pytest.mark.misc
def test_collect_sum_adds_pooled_ipc_ratios():
    raw = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "INST_A": [2.0, 6.0],
        "INST_B": [1.0, 1.0],
        "CYCLES": [4.0, 4.0],
    })
    result = collect_sum_parent(
        raw,
        [
            ("slice_a", "to_sum(raw_pmc_df['INST_A']) / to_sum(raw_pmc_df['CYCLES'])"),
            ("slice_b", "to_sum(raw_pmc_df['INST_B']) / to_sum(raw_pmc_df['CYCLES'])"),
        ],
    )
    # (2+6)/(4+4) + (1+1)/(4+4) = (8+2)/8
    assert result == pytest.approx(1.25)


@pytest.mark.misc
def test_collect_sum_zero_denominator_is_unset():
    raw = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "RD_BYTES": [10.0, 90.0],
        "RD_DURATION": [0.0, 0.0],
        "WR_BYTES": [5.0, 15.0],
        "WR_DURATION": [10.0, 30.0],
    })
    result = collect_sum_parent(
        raw,
        [
            (
                "read",
                "to_sum(raw_pmc_df['RD_BYTES']) / to_sum(raw_pmc_df['RD_DURATION'])",
            ),
            (
                "write",
                "to_sum(raw_pmc_df['WR_BYTES']) / to_sum(raw_pmc_df['WR_DURATION'])",
            ),
        ],
    )
    assert result == "N/A"


@pytest.mark.misc
def test_collect_sum_missing_denominator_is_unset():
    raw = pd.DataFrame({
        "Dispatch_ID": [1, 2],
        "RD_BYTES": [10.0, 90.0],
        "WR_BYTES": [5.0, 15.0],
        "WR_DURATION": [10.0, 30.0],
    })
    result = collect_sum_parent(
        raw,
        [
            (
                "read",
                "to_sum(raw_pmc_df['RD_BYTES']) / to_sum(raw_pmc_df['RD_DURATION'])",
            ),
            (
                "write",
                "to_sum(raw_pmc_df['WR_BYTES']) / to_sum(raw_pmc_df['WR_DURATION'])",
            ),
        ],
    )
    assert result == "N/A"
