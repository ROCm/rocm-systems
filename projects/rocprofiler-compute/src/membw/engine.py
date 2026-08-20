# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Evaluate the memory bandwidth bottleneck tree against profiled metrics."""

import operator
from typing import Callable, Optional

from membw.debug import print_evaluation_summary, print_evaluation_trace
from membw.guidance import render_guidance_blocks
from membw.models import (
    BottleneckNode,
    MemBwAnalysisResult,
    NodeSpec,
    SupportingMetric,
    TreeSpec,
)

# YAML specs use SQL-style names (gte, lte); operator module uses ge, le.
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

    evaluated_roots: list[BottleneckNode] = []
    for root_spec in tree_spec.roots:
        node = _evaluate_node(
            root_spec,
            tree_spec.thresholds,
            metric_values,
            units,
            sibling_states={},
        )
        evaluated_roots.append(node)

    guidance_ids = _collect_active_leaf_guidance_ids(tuple(evaluated_roots), tree_spec)
    guidance_blocks = render_guidance_blocks(
        guidance_ids,
        tree_spec.guidance_templates,
        tree_spec.thresholds,
        metric_values,
        max_blocks=_MAX_GUIDANCE_BLOCKS,
    )

    result = MemBwAnalysisResult(
        arch=arch,
        availability=availability,
        availability_reason=availability_reason,
        nodes=tuple(evaluated_roots),
        guidance_blocks=guidance_blocks,
    )

    print_evaluation_summary(result)

    return result


# --- Private helpers ---


def _evaluate_node(
    node_spec: NodeSpec,
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
    metric_units: dict[str, str],
    sibling_states: dict[str, str],
) -> BottleneckNode:
    """Evaluate a single node and recursively evaluate children."""
    is_catch_all = (
        node_spec.requires_parent and len(node_spec.requires_siblings_false) > 0
    )

    if is_catch_all:
        state, reason = _evaluate_catch_all(node_spec, sibling_states)
        supporting: tuple[SupportingMetric, ...] = ()
        print_evaluation_trace(
            node_id=node_spec.id,
            metric_key=None,
            metric_value=None,
            threshold_name=None,
            threshold_value=None,
            op=None,
            state=state,
            reason=reason,
        )
    else:
        state, reason, supporting = _evaluate_metric_node(
            node_spec, thresholds, metric_values, metric_units
        )

    if state == "active" and node_spec.children:
        children = _evaluate_siblings(
            node_spec.children, thresholds, metric_values, metric_units
        )
    elif node_spec.children:
        # Inactive parent → inactive children. Indeterminate parents also
        # gate children inactive: if the parent metric is unavailable we
        # cannot confirm the subtree is relevant, so we suppress it.
        children = tuple(_make_inactive_subtree(child) for child in node_spec.children)
    else:
        children = ()

    return BottleneckNode(
        id=node_spec.id,
        label=node_spec.label,
        level=node_spec.level,
        state=state,
        supporting=supporting,
        children=children,
    )


def _evaluate_metric_node(
    node_spec: NodeSpec,
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
    metric_units: dict[str, str],
) -> tuple[str, str, tuple[SupportingMetric, ...]]:
    """Evaluate a metric-based node. Returns (state, reason, supporting)."""
    metric_key = node_spec.metric
    if metric_key is None:
        state = "indeterminate"
        reason = "no metric defined"
        print_evaluation_trace(
            node_spec.id, None, None, None, None, None, state, reason
        )
        return (state, reason, ())

    value = metric_values.get(metric_key)
    threshold_key = node_spec.threshold_key
    threshold_value = (
        thresholds.get(threshold_key) if threshold_key is not None else None
    )
    op_name = node_spec.op

    if value is None:
        state = "indeterminate"
        reason = "metric value unavailable"
    elif threshold_value is None or op_name is None:
        state = "indeterminate"
        reason = "threshold or op missing"
    else:
        op_fn = _OPS.get(op_name)
        if op_fn is None:
            state = "indeterminate"
            reason = f"unknown op {op_name!r}"
        elif op_fn(value, threshold_value):
            state = "active"
            reason = ""
        else:
            state = "inactive"
            reason = ""

    print_evaluation_trace(
        node_id=node_spec.id,
        metric_key=metric_key,
        metric_value=value,
        threshold_name=threshold_key,
        threshold_value=threshold_value,
        op=op_name,
        state=state,
        reason=reason,
    )

    unit = metric_units.get(metric_key, "Percent")
    supporting = (_build_supporting_metric(metric_key, value, unit),)
    return (state, reason, supporting)


def _evaluate_catch_all(
    node_spec: NodeSpec,
    sibling_states: dict[str, str],
) -> tuple[str, str]:
    """Evaluate a catch-all node. Active only if all listed siblings are inactive."""
    for sibling_id in node_spec.requires_siblings_false:
        sibling_state = sibling_states.get(sibling_id)
        if sibling_state != "inactive":
            return ("inactive", f"sibling {sibling_id} is {sibling_state}")
    return ("active", "all listed siblings inactive")


def _evaluate_siblings(
    siblings: tuple[NodeSpec, ...],
    thresholds: dict[str, float],
    metric_values: dict[str, Optional[float]],
    metric_units: dict[str, str],
) -> tuple[BottleneckNode, ...]:
    """Two-pass sibling evaluation: metric-based first, then catch-all nodes."""
    metric_siblings = []
    catch_all_siblings = []

    for sibling in siblings:
        is_catch_all = (
            sibling.requires_parent and len(sibling.requires_siblings_false) > 0
        )
        if is_catch_all:
            catch_all_siblings.append(sibling)
        else:
            metric_siblings.append(sibling)

    pass1_states: dict[str, str] = {}
    pass1_nodes: list[BottleneckNode] = []
    for sibling_spec in metric_siblings:
        node = _evaluate_node(
            sibling_spec,
            thresholds,
            metric_values,
            metric_units,
            sibling_states={},
        )
        pass1_states[sibling_spec.id] = node.state
        pass1_nodes.append(node)

    pass2_nodes: list[BottleneckNode] = []
    for sibling_spec in catch_all_siblings:
        node = _evaluate_node(
            sibling_spec,
            thresholds,
            metric_values,
            metric_units,
            sibling_states=pass1_states,
        )
        pass2_nodes.append(node)

    result_map: dict[str, BottleneckNode] = {}
    for node in pass1_nodes:
        result_map[node.id] = node
    for node in pass2_nodes:
        result_map[node.id] = node

    return tuple(
        result_map[sibling.id] for sibling in siblings if sibling.id in result_map
    )


def _make_inactive_subtree(node_spec: NodeSpec) -> BottleneckNode:
    """Create an inactive node with all children also inactive."""
    children = tuple(_make_inactive_subtree(child) for child in node_spec.children)
    return BottleneckNode(
        id=node_spec.id,
        label=node_spec.label,
        level=node_spec.level,
        state="inactive",
        supporting=(),
        children=children,
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
    tree_spec: TreeSpec,
) -> list[str]:
    """Collect guidance_ids from active leaf nodes."""
    spec_map = _build_spec_map(tree_spec.roots)
    result: list[str] = []
    _walk_for_guidance(nodes, spec_map, result)
    return result


def _walk_for_guidance(
    nodes: tuple[BottleneckNode, ...],
    spec_map: dict[str, NodeSpec],
    result: list[str],
) -> None:
    """Recursively walk evaluated nodes for guidance IDs."""
    for node in nodes:
        if node.state != "active":
            continue
        has_active_child = any(child.state == "active" for child in node.children)
        spec = spec_map.get(node.id)
        guidance_id = spec.guidance_id if spec is not None else None
        if not has_active_child and guidance_id is not None:
            result.append(guidance_id)
        _walk_for_guidance(node.children, spec_map, result)


def _build_spec_map(roots: tuple[NodeSpec, ...]) -> dict[str, NodeSpec]:
    """Build a flat id->NodeSpec mapping for lookup."""
    result: dict[str, NodeSpec] = {}
    for root in roots:
        _add_to_spec_map(root, result)
    return result


def _add_to_spec_map(node: NodeSpec, result: dict[str, NodeSpec]) -> None:
    """Recursively add nodes to the spec map."""
    result[node.id] = node
    for child in node.children:
        _add_to_spec_map(child, result)
