# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration test: real YAML spec -> extract -> evaluate -> guidance.

Tests the full pipeline with the actual gfx950 tree spec to catch
regressions when the spec, extraction, or evaluation logic changes.
"""

import pandas as pd
import pytest

from membw.engine import evaluate_membw_tree
from membw.metric_extract import extract_membw_metrics
from membw.tree_spec import collect_metric_keys, load_tree_spec

# Representative metric values for a UTCL1-stall + HBM-BW-bound workload
UTCL1_HBM_WORKLOAD = {
    "L1 Cache - TA stalled by TCP (aggregated)": 22.4,
    "L1 Cache - TCP stalled by UTCL1": 18.7,
    "L1 Cache - TCP stalled by UTCL2": 1.2,
    "L1 Cache - TCP stalled by TD": 0.5,
    "L1 Cache - TCP stalled by L2": 2.1,
    "L1 Cache - VMEM stalled by L1 Cache": 3.0,
    "L2 Back Pressure Indicator": 5.0,
    "L2 Memory BW Bound - Combined Credit Pressure": 15.0,
    "L2 Memory BW Bound - Read Credit Pressure": 12.0,
    "L2 Memory BW Bound - Write Credit Pressure": 3.0,
    "L2 Internal Resource Pressure - Latency FIFO": 2.0,
    "L2 Internal Resource Pressure - Source FIFO": 1.0,
    "L2 Cache Efficiency": 80.0,
    "L2 Remote Access Pressure (GMI)": 0.5,
    "EA HBM BW Bound - Combined": 14.0,
    "EA HBM BW Bound - Read Credit Pressure": 11.0,
    "EA HBM BW Bound - Write Credit Pressure": 3.0,
    "EA GMI BW Bound - Combined": 0.1,
    "EA IO BW Bound - Combined": 0.0,
    "EA Write Backpressure": 1.0,
    "EA HBM Atomic Pressure": 0.5,
}


def build_mock_dfs(metric_values):
    """Build panel 3000 DataFrames from a flat metric dict."""
    table_3001_keys = [k for k in metric_values if k.startswith("L1")]
    table_3012_keys = [k for k in metric_values if k.startswith("L2")]
    table_3018_keys = [k for k in metric_values if k.startswith("EA")]

    dfs = {}
    for table_id, keys in [
        (3001, table_3001_keys),
        (3012, table_3012_keys),
        (3018, table_3018_keys),
    ]:
        if keys:
            dfs[table_id] = pd.DataFrame({
                "Metric": keys,
                "Avg": [metric_values[k] for k in keys],
                "Unit": ["Percent"] * len(keys),
            })
    return dfs


def collect_node_states(nodes):
    """Flatten all node states into a dict."""
    result = {}
    for node in nodes:
        result[node.id] = node.state
        result.update(collect_node_states(node.children))
    return result


class TestFullPipeline:
    """End-to-end tests using the real gfx950 tree spec."""

    @pytest.fixture()
    def gfx950_spec(self):
        return load_tree_spec("gfx950")

    def test_utcl1_hbm_workload(self, gfx950_spec):
        """UTCL1 stall + HBM BW bound active, rest correctly inactive."""
        metric_keys = collect_metric_keys(gfx950_spec)
        dfs = build_mock_dfs(UTCL1_HBM_WORKLOAD)
        extraction = extract_membw_metrics(dfs, metric_keys)

        assert extraction.availability == "full"

        result = evaluate_membw_tree(
            gfx950_spec,
            extraction.values,
            "gfx950",
            extraction.availability,
            extraction.availability_reason,
            metric_units=extraction.units,
        )
        states = collect_node_states(result.nodes)

        assert states["gl1_tcp_stall"] == "active"
        assert states["gl1_tcp_utcl1_stall"] == "active"
        assert states["gl1_tcp_other_stall"] == "inactive"
        assert states["gl1_vmem_stall"] == "inactive"
        assert states["gl2_mem_bw_bound"] == "active"
        assert states["gl2_mem_bw_read"] == "active"
        assert states["gl2_mem_bw_other"] == "inactive"
        assert states["ea_hbm_bw_bound"] == "active"
        assert states["ea_hbm_read"] == "active"
        assert states["ea_hbm_other"] == "inactive"

        assert len(result.guidance_blocks) > 0
        guidance_text = "\n".join(result.guidance_blocks)
        assert "UTCL1" in guidance_text
        assert "HBM" in guidance_text

    def test_balanced_pressure_triggers_catch_all(self, gfx950_spec):
        """Balanced read/write pressure fires catch-all children."""
        balanced = dict(UTCL1_HBM_WORKLOAD)
        balanced["L2 Memory BW Bound - Combined Credit Pressure"] = 12.0
        balanced["L2 Memory BW Bound - Read Credit Pressure"] = 6.0
        balanced["L2 Memory BW Bound - Write Credit Pressure"] = 6.0
        balanced["EA HBM BW Bound - Combined"] = 14.0
        balanced["EA HBM BW Bound - Read Credit Pressure"] = 7.0
        balanced["EA HBM BW Bound - Write Credit Pressure"] = 7.0

        metric_keys = collect_metric_keys(gfx950_spec)
        dfs = build_mock_dfs(balanced)
        extraction = extract_membw_metrics(dfs, metric_keys)

        result = evaluate_membw_tree(
            gfx950_spec,
            extraction.values,
            "gfx950",
            extraction.availability,
            extraction.availability_reason,
            metric_units=extraction.units,
        )
        states = collect_node_states(result.nodes)

        assert states["gl2_mem_bw_bound"] == "active"
        assert states["gl2_mem_bw_read"] == "inactive"
        assert states["gl2_mem_bw_write"] == "inactive"
        assert states["gl2_mem_bw_other"] == "active"

        assert states["ea_hbm_bw_bound"] == "active"
        assert states["ea_hbm_read"] == "inactive"
        assert states["ea_hbm_write"] == "inactive"
        assert states["ea_hbm_other"] == "active"

        guidance_text = "\n".join(result.guidance_blocks)
        assert "balanced" in guidance_text.lower()

    def test_all_below_threshold(self, gfx950_spec):
        """No bottlenecks when all metrics are well below thresholds."""
        low_values = {k: 1.0 for k in UTCL1_HBM_WORKLOAD}
        low_values["L2 Cache Efficiency"] = 90.0

        metric_keys = collect_metric_keys(gfx950_spec)
        dfs = build_mock_dfs(low_values)
        extraction = extract_membw_metrics(dfs, metric_keys)

        result = evaluate_membw_tree(
            gfx950_spec, extraction.values, "gfx950", "full", None
        )
        states = collect_node_states(result.nodes)
        assert all(s != "active" for s in states.values())
        assert len(result.guidance_blocks) == 0

    def test_all_metrics_none(self, gfx950_spec):
        """All root nodes indeterminate when all metrics are None."""
        none_values = {k: None for k in UTCL1_HBM_WORKLOAD}

        result = evaluate_membw_tree(gfx950_spec, none_values, "gfx950", "full", None)
        for root in result.nodes:
            assert root.state == "indeterminate"
        assert len(result.guidance_blocks) == 0

    def test_partial_availability(self, gfx950_spec):
        """Pipeline handles partial data gracefully."""
        metric_keys = collect_metric_keys(gfx950_spec)
        dfs = {3001: build_mock_dfs(UTCL1_HBM_WORKLOAD)[3001]}
        extraction = extract_membw_metrics(dfs, metric_keys)

        assert extraction.availability == "partial"

        result = evaluate_membw_tree(
            gfx950_spec,
            extraction.values,
            "gfx950",
            extraction.availability,
            extraction.availability_reason,
        )
        states = collect_node_states(result.nodes)
        assert states["gl1_tcp_stall"] == "active"
        assert states["gl2_back_pressure"] == "indeterminate"
        assert states["ea_hbm_bw_bound"] == "indeterminate"
