# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Data structures for memory bandwidth bottleneck analysis."""

from dataclasses import dataclass
from typing import Literal, Optional

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


@dataclass(frozen=True)
class MemBwAnalysisResult:
    """Top-level bottleneck tree evaluation result."""

    arch: str
    availability: Literal["full", "partial", "unavailable"]
    availability_reason: Optional[str]
    nodes: tuple[BottleneckNode, ...]
    guidance_blocks: tuple[str, ...]


@dataclass(frozen=True)
class MembwMetricResult:
    """Result of loading and evaluating membw metrics from a workload."""

    metric_values: dict[str, Optional[float]]
    metric_units: dict[str, str]
    availability: Literal["full", "partial", "unavailable"]
    availability_reason: Optional[str]


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


@dataclass(frozen=True)
class TreeSpec:
    """Validated tree specification."""

    thresholds: dict[str, float]
    roots: tuple[NodeSpec, ...]
    guidance_templates: dict[str, str]
    schema_hash: str
