# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration-style tests: build_dfs + eval_metric for WEIGHTED_AVG pilot YAML."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pytest

from utils import schema
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import build_metric_value_string
from utils.parser import build_dfs
from vendored import yaml

_FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "weighted_avg"
_PILOT_YAML = _FIXTURE_DIR / "pilot_metric_table.yaml"


def _pilot_sys_info() -> pd.Series:
    return pd.Series({
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


def _pilot_arch_config() -> schema.ArchConfig:
    with open(_PILOT_YAML, encoding="utf-8") as stream:
        doc = yaml.safe_load(stream)
    metric_table = doc["Panel Config"]["data source"][0]["metric_table"]
    panel = {
        "id": 1700,
        "title": "Pilot WEIGHTED_AVG",
        "data source": [{"metric_table": metric_table}],
    }
    ac = schema.ArchConfig()
    ac.panel_configs = OrderedDict([(1700, panel)])
    return ac


@pytest.mark.misc
def test_build_dfs_and_eval_metric_weighted_avg_pilot():
    ac = _pilot_arch_config()
    sys_info = _pilot_sys_info()
    build_dfs(ac, filter_metrics=None, sys_info=sys_info, profiling_config={})

    table_ids = [tid for tid, dtype in ac.dfs_type.items() if dtype == "metric_table"]
    assert len(table_ids) == 1
    table_id = table_ids[0]

    raw_pmc_df = pd.DataFrame({
        "Dispatch_ID": [1, 1, 2, 2],
        "TCC_EA0_RDREQ_DRAM_sum": [100, 0, 40, 0],
        "TCC_EA0_RDREQ_sum": [100, 0, 50, 0],
        "TCC_EA0_WRREQ_DRAM_sum": [50, 0, 20, 0],
        "TCC_EA0_WRREQ_sum": [50, 0, 40, 0],
        "GRBM_GUI_ACTIVE": [1, 1, 1, 1],
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

    df = ac.dfs[table_id]
    parent_row = df[df["Metric"] == "hbm_combined_traffic"]
    assert not parent_row.empty
    assert parent_row.iloc[0]["Avg"] == pytest.approx(83.3333333333, rel=1e-6)
