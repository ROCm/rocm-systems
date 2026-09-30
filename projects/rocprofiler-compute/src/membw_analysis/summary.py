# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Terminal-independent summaries of memory bandwidth analysis results."""

from typing import Tuple

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult

ACTIVE_FALLBACK_TEXT = (
    "Memory Bandwidth Analysis: Bottlenecks detected (see chart annotations)."
)


def has_active_nodes(nodes: Tuple[BottleneckNode, ...]) -> bool:
    """Return whether any node in the tree is active."""
    for node in nodes:
        if node.state == "active" or has_active_nodes(node.children):
            return True
    return False


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
