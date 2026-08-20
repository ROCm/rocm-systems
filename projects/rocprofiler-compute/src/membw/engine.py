# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Evaluate the memory bandwidth bottleneck tree against profiled metrics."""

import operator
from typing import Callable, Optional

import pandas as pd

from membw.debug import log_evaluation_summary, log_evaluation_trace
from membw.guidance import render_guidance_blocks
from membw.metric_extract import extract_membw_metrics
from membw.models import (
    MEMBW_TABLE_IDS,
    BottleneckNode,
    MemBwAnalysisResult,
    NodeSpec,
    ResolvedNode,
    SupportingMetric,
    TreeSpec,
)
from membw.tree_spec import collect_metric_keys, load_tree_spec, tree_spec_path
from utils.logger import console_warning

_OPS: dict[str, Callable[[float, float], bool]] = {
    "gte": operator.ge,
    "gt": operator.gt,
    "lt": operator.lt,
    "lte": operator.le,
}

# Readability limit for terminal output -- keeps guidance scannable.
_MAX_GUIDANCE_BLOCKS = 5


def evaluate_membw_tree(
    tree_spec: TreeSpec,
    metric_values: dict[str, Optional[float]],
    arch: str,
    availability: str,
    availability_reason: Optional[str],
    metric_units: Optional[dict[str, str]] = None,
) -> MemBwAnalysisResult:
    """Evaluate the bottleneck tree and produce an analysis result."""
    units = metric_units or {}

    resolved_roots = tuple(
        _resolve_node(root, tree_spec.thresholds, metric_values, units)
        for root in tree_spec.roots
    )

    evaluated_roots = tuple(
        _evaluate_node(root, sibling_states={}) for root in resolved_roots
    )

    guidance_ids = _collect_active_leaf_guidance_ids(evaluated_roots)
    guidance_blocks = render_guidance_blocks(
        guidance_ids,
        tree_spec.guidance_templates,
        tree_spec.thresholds,
        metric_values,
        max_blocks=_MAX_GUIDANCE_BLOCKS,
    )

    return MemBwAnalysisResult(
        arch=arch,
        availability=availability,
        availability_reason=availability_reason,
        nodes=evaluated_roots,
        guidance_blocks=guidance_blocks,
    )


def run_membw_analysis(
    dfs: dict[int, pd.DataFrame],
    gpu_arch: str,
) -> Optional[MemBwAnalysisResult]:
    """Run the full membw pipeline: extract metrics, evaluate tree, return result."""
    membw_dfs = {tid: dfs[tid] for tid in MEMBW_TABLE_IDS if tid in dfs}
    if not membw_dfs:
        return None

    if not tree_spec_path(gpu_arch).exists():
        console_warning("membw", f"No tree spec for {gpu_arch}, skipping")
        return None

    spec = load_tree_spec(gpu_arch)

    metric_keys = collect_metric_keys(spec)
    extraction = extract_membw_metrics(membw_dfs, metric_keys)

    result = evaluate_membw_tree(
        spec,
        extraction.values,
        gpu_arch,
        extraction.availability,
        extraction.availability_reason,
        metric_units=extraction.units,
    )
    log_evaluation_summary(result)
    return result


# --- Resolve: replace references with concrete values ---


def _resolve_node(
    node_spec: NodeSpec,
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
    metric_units: dict[str, str],
) -> ResolvedNode:
    """Resolve a single node's references and recurse into children."""
    children = tuple(
        _resolve_node(child, thresholds, metric_values, metric_units)
        for child in node_spec.children
    )
    if node_spec.is_catch_all or node_spec.metric is None:
        return ResolvedNode(
            spec=node_spec,
            value=None,
            threshold=None,
            op=None,
            unit="",
            children=children,
        )
    return ResolvedNode(
        spec=node_spec,
        value=metric_values.get(node_spec.metric),
        threshold=(
            thresholds.get(node_spec.threshold_key)
            if node_spec.threshold_key is not None
            else None
        ),
        op=_OPS.get(node_spec.op) if node_spec.op else None,
        unit=metric_units.get(node_spec.metric, "Percent"),
        children=children,
    )


# --- Evaluate: pure logic, no lookups ---


def _evaluate_node(
    node: ResolvedNode,
    sibling_states: dict[str, str],
    parent_supporting: tuple[SupportingMetric, ...] = (),
) -> BottleneckNode:
    """Evaluate a single resolved node and recursively evaluate children."""
    if node.spec.is_catch_all:
        state, reason = _evaluate_catch_all(node.spec, sibling_states)
        supporting: tuple[SupportingMetric, ...] = (
            parent_supporting if state == "active" else ()
        )
        log_evaluation_trace(
            node_id=node.spec.id,
            metric_key=None,
            metric_value=None,
            threshold_name=None,
            threshold_value=None,
            op=None,
            state=state,
            reason=reason,
        )
    else:
        state, reason, supporting = _evaluate_metric_node(node)

    if state == "active" and node.children:
        children = _evaluate_siblings(node.children, parent_supporting=supporting)
    elif node.children:
        children = tuple(_make_inactive_subtree(child) for child in node.children)
    else:
        children = ()

    return BottleneckNode(
        id=node.spec.id,
        label=node.spec.label,
        level=node.spec.level,
        state=state,
        supporting=supporting,
        children=children,
        guidance_id=node.spec.guidance_id,
    )


def _evaluate_metric_node(
    node: ResolvedNode,
) -> tuple[str, str, tuple[SupportingMetric, ...]]:
    """Evaluate a metric-based node using its pre-resolved values."""
    metric_key = node.spec.metric
    if metric_key is None:
        state = "indeterminate"
        reason = "no metric defined"
        log_evaluation_trace(node.spec.id, None, None, None, None, None, state, reason)
        return (state, reason, ())

    value = node.value
    threshold_value = node.threshold
    op_fn = node.op
    op_name = node.spec.op

    if value is None:
        state = "indeterminate"
        reason = "metric value unavailable"
    elif threshold_value is None or op_fn is None:
        state = "indeterminate"
        reason = "threshold or op missing"
    elif op_fn(value, threshold_value):
        state = "active"
        reason = ""
    else:
        state = "inactive"
        reason = ""

    log_evaluation_trace(
        node_id=node.spec.id,
        metric_key=metric_key,
        metric_value=value,
        threshold_name=node.spec.threshold_key,
        threshold_value=threshold_value,
        op=op_name,
        state=state,
        reason=reason,
    )

    supporting = (_build_supporting_metric(metric_key, value, node.unit),)
    return (state, reason, supporting)


def _evaluate_catch_all(
    node_spec: NodeSpec,
    sibling_states: dict[str, str],
) -> tuple[str, str]:
    """Evaluate a catch-all node. Active only if all listed siblings are inactive."""
    has_indeterminate = False
    for sibling_id in node_spec.requires_siblings_false:
        sibling_state = sibling_states.get(sibling_id)
        if sibling_state == "active":
            return ("inactive", f"sibling {sibling_id} is active")
        if sibling_state != "inactive":
            has_indeterminate = True
    if has_indeterminate:
        return ("indeterminate", "one or more siblings indeterminate")
    return ("active", "all listed siblings inactive")


def _evaluate_siblings(
    siblings: tuple[ResolvedNode, ...],
    parent_supporting: tuple[SupportingMetric, ...] = (),
) -> tuple[BottleneckNode, ...]:
    """Two-pass sibling evaluation: metric-based first, then catch-all nodes."""
    metric_siblings = []
    catch_all_siblings = []

    for sibling in siblings:
        if sibling.spec.is_catch_all:
            catch_all_siblings.append(sibling)
        else:
            metric_siblings.append(sibling)

    pass1_states: dict[str, str] = {}
    pass1_nodes: list[BottleneckNode] = []
    for sibling in metric_siblings:
        node = _evaluate_node(sibling, sibling_states={})
        pass1_states[sibling.spec.id] = node.state
        pass1_nodes.append(node)

    pass2_nodes: list[BottleneckNode] = []
    for sibling in catch_all_siblings:
        node = _evaluate_node(
            sibling,
            sibling_states=pass1_states,
            parent_supporting=parent_supporting,
        )
        pass2_nodes.append(node)

    result_map: dict[str, BottleneckNode] = {}
    for node in pass1_nodes:
        result_map[node.id] = node
    for node in pass2_nodes:
        result_map[node.id] = node

    return tuple(
        result_map[sibling.spec.id]
        for sibling in siblings
        if sibling.spec.id in result_map
    )


def _make_inactive_subtree(node: ResolvedNode) -> BottleneckNode:
    """Create an inactive node with all children also inactive."""
    children = tuple(_make_inactive_subtree(child) for child in node.children)
    return BottleneckNode(
        id=node.spec.id,
        label=node.spec.label,
        level=node.spec.level,
        state="inactive",
        supporting=(),
        children=children,
        guidance_id=node.spec.guidance_id,
    )


def _build_supporting_metric(
    metric_key: str,
    value: Optional[float],
    unit: str,
) -> SupportingMetric:
    """Create a SupportingMetric with formatted display."""
    if value is None:
        display = "N/A"
    elif unit == "Percent":
        display = f"{value:.1f}%"
    else:
        display = f"{value:.1f}"

    return SupportingMetric(key=metric_key, value=value, unit=unit, display=display)


def _collect_active_leaf_guidance_ids(
    nodes: tuple[BottleneckNode, ...],
) -> list[str]:
    """Collect guidance_ids from active leaf nodes."""
    result: list[str] = []
    _walk_for_guidance(nodes, result)
    return result


def _walk_for_guidance(
    nodes: tuple[BottleneckNode, ...],
    result: list[str],
) -> None:
    """Recursively walk evaluated nodes for guidance IDs."""
    for node in nodes:
        if node.state != "active":
            continue
        has_active_child = any(child.state == "active" for child in node.children)
        if not has_active_child and node.guidance_id is not None:
            result.append(node.guidance_id)
        _walk_for_guidance(node.children, result)
