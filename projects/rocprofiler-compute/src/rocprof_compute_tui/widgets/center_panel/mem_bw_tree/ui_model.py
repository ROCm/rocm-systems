from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MetricMeta:
    metric_name: str
    panel: str
    table: str
    unit: Optional[str] = None


@dataclass
class MetricSnapshot:
    values: dict[str, float]  # Metric_ID -> value
    meta: dict[str, MetricMeta]  # Metric_ID -> meta
    missing: list[str]  # list[Metric_ID]


@dataclass(frozen=True)
class PredicateResult:
    passed: bool
    expression: str  # short human explanation
    details: str  # verbose detail
    inputs: dict[str, float]  # metric_id -> value used


@dataclass
class NodeEvaluation:
    node_id: str
    passed: bool
    predicate_results: list[PredicateResult]


@dataclass
class DecisionResult:
    active_path: list[str]  # list[node_id]
    node_eval: dict[str, NodeEvaluation]  # node_id -> evaluation
