# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import math

import pandas as pd

from membw.metric_extract import extract_membw_metrics


def make_table_df(metrics_and_values):
    """Build a DataFrame matching the panel 3000 structure."""
    return pd.DataFrame({
        "Metric": [m for m, _ in metrics_and_values],
        "Avg": [v for _, v in metrics_and_values],
        "Unit": ["Percent"] * len(metrics_and_values),
    })


class TestExtractMembwMetrics:
    def test_extracts_values_across_tables(self):
        dfs = {
            3001: make_table_df([("m_l1", 18.7)]),
            3012: make_table_df([("m_l2", 5.0)]),
        }
        keys = frozenset({"m_l1", "m_l2"})
        result = extract_membw_metrics(dfs, keys)
        assert result.values["m_l1"] == 18.7
        assert result.values["m_l2"] == 5.0

    def test_missing_and_nan_return_none(self):
        dfs = {3001: make_table_df([("m1", float("nan"))])}
        keys = frozenset({"m1", "m_missing"})
        result = extract_membw_metrics(dfs, keys)
        assert result.values["m1"] is None
        assert result.values["m_missing"] is None

    def test_inf_returns_none(self):
        dfs = {3001: make_table_df([("m1", math.inf)])}
        result = extract_membw_metrics(dfs, frozenset({"m1"}))
        assert result.values["m1"] is None

    def test_empty_dfs_returns_unavailable(self):
        result = extract_membw_metrics({}, frozenset({"m1"}))
        assert result.values["m1"] is None
        assert result.availability == "unavailable"

    def test_extracts_units(self):
        dfs = {3001: make_table_df([("m1", 10.0)])}
        result = extract_membw_metrics(dfs, frozenset({"m1"}))
        assert result.units["m1"] == "Percent"

    def test_full_partial_unavailable(self):
        df = make_table_df([("m1", 10.0)])
        dfs = {3001: df}

        result = extract_membw_metrics(dfs, frozenset({"m1"}))
        assert result.availability == "full"

        result = extract_membw_metrics(dfs, frozenset({"m1", "m_missing"}))
        assert result.availability == "partial"
        assert "m_missing" in result.availability_reason

        result = extract_membw_metrics({}, frozenset({"m1"}))
        assert result.availability == "unavailable"

    def test_nan_value_not_counted_as_present(self):
        df = make_table_df([("m1", float("nan")), ("m2", 10.0)])
        dfs = {3001: df}
        result = extract_membw_metrics(dfs, frozenset({"m1", "m2"}))
        assert result.availability == "partial"
        assert "m1" in result.availability_reason
