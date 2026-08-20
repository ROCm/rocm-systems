# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import math

import pandas as pd

from membw.metric_extract import (
    check_metric_availability,
    extract_metric_values,
)


def make_table_df(metrics_and_values):
    """Build a DataFrame matching the panel 3000 structure."""
    return pd.DataFrame({
        "Metric": [m for m, _ in metrics_and_values],
        "Avg": [v for _, v in metrics_and_values],
        "Unit": ["Percent"] * len(metrics_and_values),
    })


class TestExtractMetricValues:
    def test_extracts_values_across_tables(self):
        dfs = {
            3001: make_table_df([("m_l1", 18.7)]),
            3012: make_table_df([("m_l2", 5.0)]),
        }
        keys = frozenset({"m_l1", "m_l2"})
        result = extract_metric_values(dfs, keys)
        assert result["m_l1"] == 18.7
        assert result["m_l2"] == 5.0

    def test_missing_and_nan_return_none(self):
        dfs = {3001: make_table_df([("m1", float("nan"))])}
        keys = frozenset({"m1", "m_missing"})
        result = extract_metric_values(dfs, keys)
        assert result["m1"] is None
        assert result["m_missing"] is None

    def test_inf_returns_none(self):
        dfs = {3001: make_table_df([("m1", math.inf)])}
        result = extract_metric_values(dfs, frozenset({"m1"}))
        assert result["m1"] is None

    def test_empty_dfs(self):
        result = extract_metric_values({}, frozenset({"m1"}))
        assert result["m1"] is None


class TestCheckMetricAvailability:
    def test_full_partial_unavailable(self):
        df = make_table_df([("m1", 10.0)])
        dfs = {3001: df}

        avail, _ = check_metric_availability(dfs, frozenset({"m1"}))
        assert avail == "full"

        avail, reason = check_metric_availability(dfs, frozenset({"m1", "m_missing"}))
        assert avail == "partial"
        assert "m_missing" in reason

        avail, _ = check_metric_availability({}, frozenset({"m1"}))
        assert avail == "unavailable"
