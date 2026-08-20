# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Debug output for memory bandwidth bottleneck tree evaluation."""

from typing import Optional

from membw.models import BottleneckNode, MemBwAnalysisResult
from utils.logger import console_debug


def log_evaluation_trace(
    node_id: str,
    metric_key: Optional[str],
    metric_value: Optional[float],
    threshold_name: Optional[str],
    threshold_value: Optional[float],
    op: Optional[str],
    state: str,
    reason: str,
) -> None:
    """Print a structured trace line for a single node evaluation."""
    console_debug("membw", f"{node_id}: {state}")

    if metric_key is not None:
        display_value = f"{metric_value}" if metric_value is not None else "N/A"
        console_debug("membw", f'\tmetric: "{metric_key}" = {display_value}')

    if threshold_name is not None and threshold_value is not None:
        console_debug("membw", f"\tthreshold: {threshold_name} = {threshold_value}")

    if op is not None and metric_value is not None and threshold_value is not None:
        op_symbol = _op_symbol(op)
        console_debug(
            "membw",
            f"\tcomparison: {metric_value} {op_symbol} {threshold_value} -> {state}",
        )

    if reason:
        console_debug("membw", f"\treason: {reason}")


def log_evaluation_summary(result: MemBwAnalysisResult) -> None:
    """Print a compact summary of the evaluation result."""
    console_debug("membw", "=== Evaluation Summary ===")
    console_debug("membw", f"arch={result.arch} availability={result.availability}")
    if result.availability_reason is not None:
        console_debug("membw", f"reason: {result.availability_reason}")

    active_nodes = _collect_active_nodes(result.nodes)
    if active_nodes:
        console_debug("membw", "Active bottlenecks:")
        for node_id, label, level in active_nodes:
            console_debug("membw", f"\t[{level}] {label} ({node_id})")
    else:
        console_debug("membw", "No active bottlenecks")

    if result.guidance_blocks:
        console_debug("membw", f"Guidance blocks: {len(result.guidance_blocks)}")


# --- Private helpers ---


def _op_symbol(op: str) -> str:
    """Convert op name to display symbol."""
    symbols = {"gte": ">=", "gt": ">", "lt": "<", "lte": "<="}
    return symbols.get(op, op)


def _collect_active_nodes(
    nodes: tuple[BottleneckNode, ...],
) -> list[tuple[str, str, str]]:
    """Collect (id, label, level) tuples for all active nodes."""
    result: list[tuple[str, str, str]] = []
    for node in nodes:
        if node.state == "active":
            result.append((node.id, node.label, node.level))
        result.extend(_collect_active_nodes(node.children))
    return result
