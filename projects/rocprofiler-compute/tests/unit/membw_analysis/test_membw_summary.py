# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for memory bandwidth summary text and tree activity."""

import pytest

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult
from membw_analysis.summary import (
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
