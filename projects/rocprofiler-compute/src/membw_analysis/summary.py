# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Terminal-independent summaries of memory bandwidth analysis results."""

from typing import Iterator, List, Optional, Tuple

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


def active_leaves(nodes: Tuple[BottleneckNode, ...]) -> Iterator[BottleneckNode]:
    """Active nodes with no active child, in tree order: the bottlenecks to report."""
    for node in nodes:
        if node.state != "active":
            continue
        if not any(child.state == "active" for child in node.children):
            yield node
        yield from active_leaves(node.children)


def stalls_at(
    membw_result: Optional[MemBwAnalysisResult], level: Optional[str]
) -> List[Tuple[BottleneckNode, Optional[float]]]:
    """Each active leaf bottleneck at the given level, with its measured value."""
    if membw_result is None or level is None:
        return []
    return [
        (node, node.supporting[0].value if node.supporting else None)
        for node in active_leaves(membw_result.nodes)
        if node.level == level
    ]


def status_line(membw_result: MemBwAnalysisResult) -> Optional[str]:
    """The line to show instead of guidance, or None when there is guidance to show."""
    if not has_active_nodes(membw_result.nodes):
        return status_text(membw_result)
    if not any(membw_result.guidance_blocks):
        return ACTIVE_FALLBACK_TEXT
    return None


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
