# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Terminal-independent summaries of memory bandwidth analysis results."""

from typing import List, Tuple

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult

STALL_BLOCK_LEVELS = {"vl1d": "GL1", "l2": "GL2", "data_fabric": "EA"}

ACTIVE_FALLBACK_TEXT = (
    "Memory Bandwidth Analysis: Bottlenecks detected (see chart annotations)."
)


def has_active_nodes(nodes: Tuple[BottleneckNode, ...]) -> bool:
    """Return whether any node in the tree is active."""
    for node in nodes:
        if node.state == "active" or has_active_nodes(node.children):
            return True
    return False


def active_stall_leaves(
    result: MemBwAnalysisResult, block_id: str
) -> Tuple[BottleneckNode, ...]:
    """Return the active terminal bottlenecks annotated on a chart block."""
    level = STALL_BLOCK_LEVELS.get(block_id)
    if level is None:
        return ()
    leaves: List[BottleneckNode] = []
    for node in result.nodes:
        _append_active_leaves(node, level, leaves)
    return tuple(leaves)


def _append_active_leaves(
    node: BottleneckNode, level: str, leaves: List[BottleneckNode]
) -> None:
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
    if _all_nodes_indeterminate(membw_result.nodes):
        return "Memory Bandwidth Analysis: Inconclusive (insufficient counter data)."
    return "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA)."


def _all_nodes_indeterminate(nodes: Tuple[BottleneckNode, ...]) -> bool:
    """Return whether every root node is indeterminate."""
    return bool(nodes) and all(node.state == "indeterminate" for node in nodes)
