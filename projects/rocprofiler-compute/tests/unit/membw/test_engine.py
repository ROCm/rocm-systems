# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the bottleneck tree evaluation engine.

Focuses on functional correctness of evaluation rules:
operators, parent gating, sibling exclusion, indeterminate propagation,
guidance collection, and debug output.
"""

import logging

import pytest

from membw.engine import evaluate_membw_tree
from membw.models import NodeSpec, TreeSpec


def make_tree_spec(nodes, thresholds=None, templates=None):
    """Build a minimal TreeSpec for testing."""
    return TreeSpec(
        thresholds=thresholds or {"stall_pct_high": 10.0},
        roots=tuple(nodes),
        guidance_templates=templates or {},
        schema_hash="test",
    )


def make_node(
    node_id,
    level="GL1",
    metric=None,
    op=None,
    threshold_key=None,
    label=None,
    guidance_id=None,
    requires_parent=False,
    requires_siblings_false=(),
    children=(),
):
    """Build a NodeSpec for testing."""
    return NodeSpec(
        id=node_id,
        level=level,
        metric=metric,
        op=op,
        threshold_key=threshold_key,
        label=label or node_id,
        guidance_id=guidance_id,
        requires_parent=requires_parent,
        requires_siblings_false=tuple(requires_siblings_false),
        children=tuple(children),
    )


class TestOperatorEvaluation:
    @pytest.mark.parametrize(
        "op, value, threshold, expected",
        [
            ("gte", 15.0, 10.0, "active"),
            ("gte", 5.0, 10.0, "inactive"),
            ("gte", 10.0, 10.0, "active"),
            ("gt", 15.0, 10.0, "active"),
            ("gt", 10.0, 10.0, "inactive"),
            ("lt", 5.0, 10.0, "active"),
            ("lt", 15.0, 10.0, "inactive"),
            ("lte", 10.0, 10.0, "active"),
            ("lte", 15.0, 10.0, "inactive"),
        ],
    )
    def test_comparison_operators(self, op, value, threshold, expected):
        node = make_node("n", metric="m", op=op, threshold_key="t")
        spec = make_tree_spec([node], thresholds={"t": threshold})
        result = evaluate_membw_tree(spec, {"m": value}, "gfx950", "full", None)
        assert result.nodes[0].state == expected


class TestIndeterminatePropagation:
    def test_none_metric_is_indeterminate(self):
        node = make_node("n", metric="m", op="gte", threshold_key="stall_pct_high")
        spec = make_tree_spec([node])
        result = evaluate_membw_tree(spec, {"m": None}, "gfx950", "full", None)
        assert result.nodes[0].state == "indeterminate"

    def test_missing_metric_is_indeterminate(self):
        node = make_node("n", metric="m", op="gte", threshold_key="stall_pct_high")
        spec = make_tree_spec([node])
        result = evaluate_membw_tree(spec, {}, "gfx950", "full", None)
        assert result.nodes[0].state == "indeterminate"


class TestParentGating:
    def test_active_parent_evaluates_children(self):
        child = make_node("c", metric="m2", op="gte", threshold_key="stall_pct_high")
        parent = make_node(
            "p",
            metric="m1",
            op="gte",
            threshold_key="stall_pct_high",
            children=[child],
        )
        spec = make_tree_spec([parent])
        result = evaluate_membw_tree(
            spec, {"m1": 15.0, "m2": 15.0}, "gfx950", "full", None
        )
        assert result.nodes[0].children[0].state == "active"

    def test_inactive_parent_forces_children_inactive(self):
        child = make_node("c", metric="m2", op="gte", threshold_key="stall_pct_high")
        parent = make_node(
            "p",
            metric="m1",
            op="gte",
            threshold_key="stall_pct_high",
            children=[child],
        )
        spec = make_tree_spec([parent])
        result = evaluate_membw_tree(
            spec, {"m1": 5.0, "m2": 15.0}, "gfx950", "full", None
        )
        assert result.nodes[0].children[0].state == "inactive"


class TestSiblingExclusion:
    def test_catch_all_active_when_all_siblings_inactive(self):
        children = [
            make_node("a", metric="ma", op="gte", threshold_key="stall_pct_high"),
            make_node("b", metric="mb", op="gte", threshold_key="stall_pct_high"),
            make_node(
                "other",
                requires_parent=True,
                requires_siblings_false=["a", "b"],
            ),
        ]
        parent = make_node(
            "p",
            metric="mp",
            op="gte",
            threshold_key="stall_pct_high",
            children=children,
        )
        spec = make_tree_spec([parent])
        result = evaluate_membw_tree(
            spec, {"mp": 15.0, "ma": 5.0, "mb": 5.0}, "gfx950", "full", None
        )
        states = {c.id: c.state for c in result.nodes[0].children}
        assert states["other"] == "active"

    def test_catch_all_inactive_when_sibling_active(self):
        children = [
            make_node("a", metric="ma", op="gte", threshold_key="stall_pct_high"),
            make_node("other", requires_parent=True, requires_siblings_false=["a"]),
        ]
        parent = make_node(
            "p",
            metric="mp",
            op="gte",
            threshold_key="stall_pct_high",
            children=children,
        )
        spec = make_tree_spec([parent])
        result = evaluate_membw_tree(
            spec, {"mp": 15.0, "ma": 15.0}, "gfx950", "full", None
        )
        states = {c.id: c.state for c in result.nodes[0].children}
        assert states["other"] == "inactive"

    def test_catch_all_indeterminate_when_sibling_indeterminate(self):
        children = [
            make_node("a", metric="ma", op="gte", threshold_key="stall_pct_high"),
            make_node("b", metric="mb", op="gte", threshold_key="stall_pct_high"),
            make_node(
                "other",
                requires_parent=True,
                requires_siblings_false=["a", "b"],
            ),
        ]
        parent = make_node(
            "p",
            metric="mp",
            op="gte",
            threshold_key="stall_pct_high",
            children=children,
        )
        spec = make_tree_spec([parent])
        result = evaluate_membw_tree(
            spec, {"mp": 15.0, "ma": None, "mb": None}, "gfx950", "full", None
        )
        states = {c.id: c.state for c in result.nodes[0].children}
        assert states["a"] == "indeterminate"
        assert states["b"] == "indeterminate"
        assert states["other"] == "indeterminate"


class TestGuidanceCollection:
    def test_active_leaf_produces_guidance(self):
        node = make_node(
            "n",
            metric="m",
            op="gte",
            threshold_key="stall_pct_high",
            guidance_id="g1",
        )
        spec = make_tree_spec([node], templates={"g1": "Guidance {metric:m}%"})
        result = evaluate_membw_tree(spec, {"m": 15.0}, "gfx950", "full", None)
        assert len(result.guidance_blocks) == 1
        assert "15.0%" in result.guidance_blocks[0]

    def test_active_parent_with_active_child_skips_parent_guidance(self):
        """Only leaf nodes produce guidance -- parent defers to child."""
        child = make_node(
            "c",
            metric="m2",
            op="gte",
            threshold_key="stall_pct_high",
            guidance_id="gc",
        )
        parent = make_node(
            "p",
            metric="m1",
            op="gte",
            threshold_key="stall_pct_high",
            guidance_id="gp",
            children=[child],
        )
        spec = make_tree_spec(
            [parent],
            templates={"gp": "Parent", "gc": "Child"},
        )
        result = evaluate_membw_tree(
            spec, {"m1": 15.0, "m2": 15.0}, "gfx950", "full", None
        )
        assert len(result.guidance_blocks) == 1
        assert "Child" in result.guidance_blocks[0]

    def test_inactive_node_no_guidance(self):
        node = make_node(
            "n",
            metric="m",
            op="gte",
            threshold_key="stall_pct_high",
            guidance_id="g1",
        )
        spec = make_tree_spec([node], templates={"g1": "Should not appear"})
        result = evaluate_membw_tree(spec, {"m": 5.0}, "gfx950", "full", None)
        assert len(result.guidance_blocks) == 0


class TestSupportingMetrics:
    def test_metric_value_and_unit_attached(self):
        node = make_node("n", metric="m", op="gte", threshold_key="stall_pct_high")
        spec = make_tree_spec([node])
        result = evaluate_membw_tree(
            spec,
            {"m": 15.0},
            "gfx950",
            "full",
            None,
            metric_units={"m": "Percent"},
        )
        sm = result.nodes[0].supporting[0]
        assert sm.key == "m"
        assert sm.value == 15.0
        assert sm.display == "15.0%"
        assert sm.unit == "Percent"


class TestDebugOutput:
    def test_evaluation_produces_trace(self, caplog):
        node = make_node("n", metric="m", op="gte", threshold_key="stall_pct_high")
        spec = make_tree_spec([node])
        with caplog.at_level(logging.DEBUG):
            evaluate_membw_tree(spec, {"m": 15.0}, "gfx950", "full", None)
        assert "[membw]" in caplog.text
        assert "active" in caplog.text


class TestGL1Scenario:
    """Full GL1 scenario matching the LLD UTCL1-stall example."""

    def test_utcl1_stall_with_catch_all(self):
        children = [
            make_node(
                "gl1_tcp_utcl1_stall",
                metric="L1 Cache - TCP stalled by UTCL1",
                op="gte",
                threshold_key="stall_pct_high",
                guidance_id="gl1_tcp_utcl1_stall",
            ),
            make_node(
                "gl1_tcp_utcl2_stall",
                metric="L1 Cache - TCP stalled by UTCL2",
                op="gte",
                threshold_key="stall_pct_high",
            ),
            make_node(
                "gl1_tcp_td_stall",
                metric="L1 Cache - TCP stalled by TD",
                op="gte",
                threshold_key="stall_pct_high",
            ),
            make_node(
                "gl1_tcp_l2_stall",
                metric="L1 Cache - TCP stalled by L2",
                op="gte",
                threshold_key="stall_pct_high",
            ),
            make_node(
                "gl1_tcp_other_stall",
                requires_parent=True,
                requires_siblings_false=[
                    "gl1_tcp_utcl1_stall",
                    "gl1_tcp_utcl2_stall",
                    "gl1_tcp_td_stall",
                    "gl1_tcp_l2_stall",
                ],
                guidance_id="gl1_tcp_other_stall",
            ),
        ]
        parent = make_node(
            "gl1_tcp_stall",
            metric="L1 Cache - TA stalled by TCP (aggregated)",
            op="gte",
            threshold_key="stall_pct_high",
            children=children,
        )
        spec = make_tree_spec(
            [parent],
            templates={
                "gl1_tcp_utcl1_stall": "UTCL1 guidance",
                "gl1_tcp_other_stall": "Other guidance",
            },
        )
        metrics = {
            "L1 Cache - TA stalled by TCP (aggregated)": 22.4,
            "L1 Cache - TCP stalled by UTCL1": 18.7,
            "L1 Cache - TCP stalled by UTCL2": 1.2,
            "L1 Cache - TCP stalled by TD": 0.5,
            "L1 Cache - TCP stalled by L2": 2.1,
        }
        result = evaluate_membw_tree(spec, metrics, "gfx950", "full", None)

        states = {c.id: c.state for c in result.nodes[0].children}
        assert states["gl1_tcp_utcl1_stall"] == "active"
        assert states["gl1_tcp_utcl2_stall"] == "inactive"
        assert states["gl1_tcp_other_stall"] == "inactive"

        assert len(result.guidance_blocks) == 1
        assert "UTCL1 guidance" in result.guidance_blocks[0]
