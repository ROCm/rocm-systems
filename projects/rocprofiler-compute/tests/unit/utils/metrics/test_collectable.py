# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for collectable evaluation graph and COLLECT_SUM."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from utils import schema
from utils.metrics.aggregation import merge_dispatch_collect_sum
from utils.metrics.collectable import (
    COLLECT_SUM_SPECS_ATTR,
    WEIGHTED_AVG_ATTR,
    WEIGHTED_AVG_SUBS_ATTR,
    CompositeKind,
    build_metric_eval_graph,
)
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import (
    build_metric_value_string,
    parse_collect_sum_submetrics,
)
from utils.parser import build_dfs
from vendored import yaml


@pytest.mark.misc
def test_parse_collect_sum_submetrics():
    assert parse_collect_sum_submetrics("COLLECT_SUM(a, b)") == ["a", "b"]


@pytest.mark.misc
def test_merge_dispatch_collect_sum_two_parts():
    ratios = [
        pd.Series({1: 60.0, 2: 80.0}),
        pd.Series({1: 40.0, 2: 20.0}),
    ]
    assert merge_dispatch_collect_sum(ratios) == pytest.approx(100.0)


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
    assert row.iloc[0]["Avg"] == pytest.approx(165.0)
