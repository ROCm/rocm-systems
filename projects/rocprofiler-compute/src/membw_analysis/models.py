# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Data structures for memory bandwidth bottleneck analysis.

Pipeline overview (models in CAPS, stages in brackets)::

          TREE_SPEC                 METRIC_EXTRACTION_RESULT
           .roots: NODE_SPEC        .values  .units
           .thresholds
           .guidance_templates
              |                        |
              +----------+  +----------+
                         v  v
                   [_resolve_node]  merge spec + runtime data
                         |
                         v
                   RESOLVED_NODE   concrete values per node:
                    .spec            value, threshold, op, unit
                    .value
                    .threshold
                    .op, .unit
                         |
                         v
                   [_evaluate_node]  pure logic, no lookups
                         |
                         v
                   BOTTLENECK_NODE   active / inactive / indet.
                    .supporting:
                      SUPPORTING_METRIC  (evidence for display)
                    .children:
                      BOTTLENECK_NODE
                         |
                         v
                   MEMBW_ANALYSIS_RESULT    final output
                    .nodes
                    .guidance_blocks
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

MEMBW_TABLE_IDS: tuple[int, ...] = (3001, 3012, 3018)


@dataclass(frozen=True)
class SupportingMetric:
    """Evidence metric attached to a bottleneck node."""

    key: str
    value: Optional[float]
    unit: str
    display: str


@dataclass(frozen=True)
class BottleneckNode:
    """A node in the evaluated bottleneck tree."""

    id: str
    label: str
    level: Literal["GL1", "GL2", "EA", ""]
    state: Literal["active", "inactive", "indeterminate"]
    supporting: tuple[SupportingMetric, ...]
    children: tuple["BottleneckNode", ...]
    guidance_id: Optional[str] = None


@dataclass(frozen=True)
class MemBwAnalysisResult:
    """Top-level bottleneck tree evaluation result."""

    arch: str
    availability: Literal["full", "partial", "unavailable"]
    availability_reason: Optional[str]
    nodes: tuple[BottleneckNode, ...]
    guidance_blocks: tuple[str, ...]


@dataclass(frozen=True)
class NodeSpec:
    """Parsed node from the tree spec YAML."""

    id: str
    level: str
    metric: Optional[str]
    op: Optional[str]
    threshold_key: Optional[str]
    label: str
    guidance_id: Optional[str]
    requires_parent: bool
    requires_siblings_false: tuple[str, ...]
    children: tuple["NodeSpec", ...]

    @property
    def is_catch_all(self) -> bool:
        """True when this node activates only if all listed siblings are inactive."""
        return self.requires_parent and len(self.requires_siblings_false) > 0


@dataclass(frozen=True)
class ResolvedNode:
    """A node with all metric/threshold references resolved to concrete values."""

    spec: NodeSpec
    value: Optional[float]
    threshold: Optional[float]
    op: Optional[Callable[[float, float], bool]]
    unit: str
    children: tuple["ResolvedNode", ...]


@dataclass(frozen=True)
class TreeSpec:
    """Validated tree specification."""

    thresholds: dict[str, float]
    roots: tuple[NodeSpec, ...]
    guidance_templates: dict[str, str]
    schema_hash: str
