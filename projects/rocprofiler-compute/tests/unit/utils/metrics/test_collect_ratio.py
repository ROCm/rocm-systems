# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for COLLECT_RATIO / COLLECT_SUM Phase 2 production wiring."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from utils import schema
from utils.metrics.aggregation import (
    merge_dispatch_collect_ratio,
    merge_dispatch_collect_sum,
)
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import (
    build_metric_value_string,
    is_composite_avg_formula,
    parse_collect_ratio_parts,
    parse_collect_sum_submetrics,
)
from utils.parser import build_dfs
from vendored import yaml

_GFX942 = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "rocprof_compute_soc"
    / "analysis_configs"
    / "gfx942"
)


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
def test_merge_dispatch_collect_sum_identity():
    series = [pd.Series({1: 1.5}), pd.Series({1: 2.5})]
    assert merge_dispatch_collect_sum(series) == pytest.approx(4.0)


@pytest.mark.misc
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
    panel = yaml.safe_load((_GFX942 / "0200_system_speed_of_light.yaml").read_text())[
        "Panel Config"
    ]
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
