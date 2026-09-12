# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for WEIGHTED_AVG (Phase 2)."""

import pandas as pd
import pytest

from utils.metrics.aggregation import merge_dispatch_weighted_avg
from utils.metrics.expression import build_eval_string, parse_weighted_avg_submetrics
from utils.metrics.weighted_avg import (
    evaluate_weighted_avg_parent,
)


@pytest.mark.misc
def test_parse_weighted_avg_submetrics_parses_comma_separated_names():
    formula = "WEIGHTED_AVG(hbm_read_sub, hbm_write_sub)"
    assert parse_weighted_avg_submetrics(formula) == [
        "hbm_read_sub",
        "hbm_write_sub",
    ]


@pytest.mark.misc
def test_build_eval_string_returns_empty_for_weighted_avg():
    assert build_eval_string("WEIGHTED_AVG(a, b)") == ""


@pytest.mark.misc
def test_merge_dispatch_weighted_avg_two_submetrics_three_dispatches():
    ratios = [
        pd.Series({1: 100.0, 2: 80.0, 3: 60.0}),
        pd.Series({1: 90.0, 2: 70.0, 3: 50.0}),
    ]
    weights = [
        pd.Series({1: 10.0, 2: 20.0, 3: 30.0}),
        pd.Series({1: 5.0, 2: 10.0, 3: 15.0}),
    ]
    # dispatch 1: (100*10 + 90*5) / 15 = 96.666...
    result = merge_dispatch_weighted_avg(ratios, weights)
    expected_dispatch = [
        (100 * 10 + 90 * 5) / 15,
        (80 * 20 + 70 * 10) / 30,
        (60 * 30 + 50 * 15) / 45,
    ]
    assert result == pytest.approx(sum(expected_dispatch) / len(expected_dispatch))


@pytest.mark.misc
def test_evaluate_weighted_avg_parent_end_to_end():
    raw_pmc_df = pd.DataFrame({
        "Dispatch_ID": [1, 1, 2, 2],
        "TCC_EA0_RDREQ_DRAM_sum": [100, 0, 40, 0],
        "TCC_EA0_RDREQ_sum": [100, 0, 50, 0],
        "TCC_EA0_WRREQ_DRAM_sum": [50, 0, 20, 0],
        "TCC_EA0_WRREQ_sum": [50, 0, 40, 0],
        "GRBM_GUI_ACTIVE": [1, 1, 1, 1],
    })
    read_avg = "100 * SUM(TCC_EA0_RDREQ_DRAM_sum) / SUM(TCC_EA0_RDREQ_sum)"
    write_avg = "100 * SUM(TCC_EA0_WRREQ_DRAM_sum) / SUM(TCC_EA0_WRREQ_sum)"
    df = pd.DataFrame(
        [
            ["17.2.0", "hbm_read_sub", read_avg],
            ["17.2.1", "hbm_write_sub", write_avg],
            [
                "17.2.2",
                "hbm_combined",
                "",
            ],
        ],
        columns=["Metric_ID", "Metric", "Avg"],
    ).set_index("Metric_ID")
    df.attrs["weighted_avg_specs"] = {
        "17.2.2": {
            "hbm_read_sub": {"weight_counter": "TCC_EA0_RDREQ_sum"},
            "hbm_write_sub": {"weight_counter": "TCC_EA0_WRREQ_sum"},
        },
    }
    df.attrs["weighted_avg_subs"] = {"17.2.2": ["hbm_read_sub", "hbm_write_sub"]}

    # Pre-build eval strings for submetrics (parent avg left empty).
    for row_id in ("17.2.0", "17.2.1"):
        df.at[row_id, "Avg"] = build_eval_string(df.at[row_id, "Avg"])

    result = evaluate_weighted_avg_parent(
        ["hbm_read_sub", "hbm_write_sub"],
        df.attrs["weighted_avg_specs"]["17.2.2"],
        df,
        raw_pmc_df,
        {},
        {},
    )
    assert result == pytest.approx(83.3333333333, rel=1e-6)
