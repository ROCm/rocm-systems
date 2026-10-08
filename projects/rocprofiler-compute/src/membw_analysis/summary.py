# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Terminal-independent summaries of memory bandwidth analysis results."""

from typing import List, Optional, Tuple

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult

ACTIVE_FALLBACK_TEXT = (
    "Memory Bandwidth Analysis: Bottlenecks detected (see chart annotations)."
)


def has_active_nodes(nodes: tuple[BottleneckNode, ...]) -> bool:
    """Return whether any node in the tree is active."""
    for node in nodes:
        if node.state == "active" or has_active_nodes(node.children):
            return True
    return False


def active_stall_leaves(
    result: Optional[MemBwAnalysisResult], level: Optional[str]
) -> Tuple[BottleneckNode, ...]:
    """Return active leaf bottlenecks at a resolved layout stall level."""
    if result is None or level is None:
        return ()
    leaves: List[BottleneckNode] = []
    for node in result.nodes:
        _append_active_leaves(node, level, leaves)
    return tuple(leaves)


def _append_active_leaves(
    node: BottleneckNode, level: str, leaves: List[BottleneckNode]
) -> None:
    """Collect active terminal nodes at the requested memory level."""
    if node.state != "active":
        return
    if node.level == level and not any(
        child.state == "active" for child in node.children
    ):
        leaves.append(node)
    for child in node.children:
        _append_active_leaves(child, level, leaves)


def status_text(membw_result: MemBwAnalysisResult) -> str:
    """Describe an analysis with no active bottlenecks, without a newline."""
    if membw_result.availability == "unavailable":
        return (
            "Memory Bandwidth Analysis: Unavailable "
            f"({membw_result.availability_reason or 'no data'})."
        )
    if membw_result.availability == "partial":
        return (
            "Memory Bandwidth Analysis: Partial data "
            f"({membw_result.availability_reason})."
        )
    if bool(membw_result.nodes) and all(
        node.state == "indeterminate" for node in membw_result.nodes
    ):
        return "Memory Bandwidth Analysis: Inconclusive (insufficient counter data)."
    return "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA)."
