# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for memory bandwidth summary text and tree activity."""

import pytest

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult
from membw_analysis.summary import (
    active_stall_leaves,
    has_active_nodes,
    status_text,
)


def make_node(state, children=()):
    return BottleneckNode(
        id="node",
        label="Node",
        level="GL1",
        state=state,
        supporting=(),
        children=children,
    )


def make_result(availability="full", reason=None, nodes=()):
    return MemBwAnalysisResult(
        arch="gfx950",
        availability=availability,
        availability_reason=reason,
        nodes=nodes,
        guidance_blocks=(),
    )


@pytest.mark.parametrize(
    "availability,reason,expected",
    [
        (
            "unavailable",
            None,
            "Memory Bandwidth Analysis: Unavailable (no data).",
        ),
        (
            "unavailable",
            "missing counters",
            "Memory Bandwidth Analysis: Unavailable (missing counters).",
        ),
        (
            "partial",
            "missing: key1",
            "Memory Bandwidth Analysis: Partial data (missing: key1).",
        ),
        (
            "full",
            None,
            "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA).",
        ),
    ],
)
def test_status_text(availability, reason, expected) -> None:
    assert status_text(make_result(availability, reason)) == expected


def test_indeterminate_roots_are_inconclusive() -> None:
    result = make_result(nodes=(make_node("indeterminate"),))
    assert status_text(result) == (
        "Memory Bandwidth Analysis: Inconclusive (insufficient counter data)."
    )


@pytest.mark.parametrize(
    "nodes,expected",
    [
        pytest.param(
            (make_node("inactive", (make_node("active"),)),),
            True,
            id="active-descendant",
        ),
        pytest.param(
            (make_node("inactive"),),
            False,
            id="inactive-leaf",
        ),
        pytest.param((), False, id="empty-input"),
    ],
)
def test_has_active_nodes(nodes, expected) -> None:
    assert has_active_nodes(nodes) is expected


def test_active_stall_leaves_select_terminal_nodes_by_block() -> None:
    """Select active terminal bottlenecks for known blocks only."""
    child = make_node("active")
    parent = make_node("active", (child, make_node("inactive")))
    l2 = BottleneckNode(
        id="l2",
        label="L2",
        level="GL2",
        state="active",
        supporting=(),
        children=(),
    )
    result = make_result(nodes=(parent, l2, make_node("inactive")))
    assert active_stall_leaves(result, "GL1") == (child,)
    assert active_stall_leaves(result, "GL2") == (l2,)
    assert active_stall_leaves(result, "unknown") == ()
